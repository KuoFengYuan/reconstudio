"""Multi-camera rig support: group images into frames, then hand COLMAP a rig.

COLMAP's `rig_configurator` decides which images belong to the same *frame* (one
exposure event of the whole rig) purely by filename: it strips each camera's
`image_prefix` and groups the images whose **remainder is byte-identical**
(`scene/rig.cc`, `frame_name_to_images[StringGetAfter(name, prefix)]`). Any rig
whose bodies stamp their own serial into the filename therefore groups nothing
and dies in `UpdateRigsAndFramesFromDatabase`.

So this module does the grouping itself — with a strategy the user picks — and
then materialises a symlink tree whose names *do* satisfy COLMAP's rule:

    <staging>/<camera>/<frame_key><ext>   ->   original file

which makes `image_prefix = "<camera>/"` group correctly with no database
surgery. The strategies:

  auto    camera = first path component; the exposure key is *discovered* from
          the filenames by scoring every digit field (and pair of fields) on how
          many exposures it leaves covered by all cameras. Handles the usual
          `<cam>-<strip>_<index>-<serial>.jpg` shape with no configuration, and
          is why the panel does not ask for a regex up front.
  folder  camera = first path component, frame = the rest of the relative path.
          For rigs that already write matching filenames per camera.
  regex   camera / frame come from named groups in a user-supplied pattern,
          e.g. ``^(?P<cam>[NFBLR])-\\d+_(?P<frame>.+)$``. The general escape hatch.
  gps     camera as in `folder`, but frames are clustered by EXIF GPS: the heads
          of a rig fire together so they share a position exactly, while their
          body clocks (DateTimeOriginal) are usually not synced. Use when the
          filenames carry no shared key at all.

Nothing here writes to the COLMAP database; `rig_configurator` does that from
the generated rig_config.json.
"""
from __future__ import annotations

import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

RIG_MODES = ("auto", "folder", "regex", "gps")

# Panel-facing defaults (mirrored into COLMAP_DEFAULTS).
RIG_DEFAULTS = {
    "rig_enable": False,
    "rig_mode": "auto",
    "rig_regex": "",
    "rig_ref_camera": "",     # "" = first camera in sorted order
    "rig_gps_tol": "0.5",     # metres; gps mode only
}


@dataclass
class RigGrouping:
    """cameras -> {frame_key: relative image name}, plus what didn't fit."""
    frames: dict[str, dict[str, str]] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)

    @property
    def cameras(self) -> list[str]:
        return sorted(self.frames)

    @property
    def frame_keys(self) -> list[str]:
        keys: set[str] = set()
        for per_cam in self.frames.values():
            keys |= per_cam.keys()
        return sorted(keys)

    def complete_frames(self) -> list[str]:
        """Frame keys covered by every camera — the ones a rig actually constrains."""
        cams = self.cameras
        return [k for k in self.frame_keys
                if all(k in self.frames[c] for c in cams)]


def _split_folder(name: str) -> tuple[str, str] | None:
    """`N/0-61214.jpg` -> ("N", "0-61214.jpg"). Flat names have no camera."""
    parts = name.replace("\\", "/").split("/")
    if len(parts) < 2:
        return None
    return parts[0], "/".join(parts[1:])


def _split_regex(name: str, pattern: re.Pattern) -> tuple[str, str] | None:
    """Match against the basename first, then the full relative name, so a
    pattern written for the filename still works in a nested layout."""
    for cand in (Path(name).name, name):
        m = pattern.search(cand)
        if m:
            gd = m.groupdict()
            cam, frame = gd.get("cam"), gd.get("frame")
            if cam and frame:
                return cam, frame
    return None


def compile_rig_regex(pattern: str) -> re.Pattern:
    rx = re.compile(pattern)
    missing = {"cam", "frame"} - set(rx.groupindex)
    if missing:
        raise ValueError(
            "rig regex must define named groups (?P<cam>…) and (?P<frame>…); "
            f"missing: {', '.join(sorted(missing))}")
    return rx


def _gps_metres(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Equirectangular metre distance — exact enough at rig-baseline scale."""
    lat1, lon1 = a
    lat2, lon2 = b
    mlat = math.radians((lat1 + lat2) / 2.0)
    dx = math.radians(lon2 - lon1) * math.cos(mlat) * 6371000.0
    dy = math.radians(lat2 - lat1) * 6371000.0
    return math.hypot(dx, dy)


def _digit_tokens(name: str) -> list[str]:
    """Every maximal digit run in the filename stem, in order."""
    return re.findall(r"\d+", Path(name).stem)


def auto_frame_key(per_cam: dict[str, list[str]]) -> tuple[tuple[int, ...], int]:
    """Find which digit-token positions form the shared exposure key.

    Rig filenames are almost always `<camera><separator><strip>_<index><serial>`
    in some order: some digit fields identify the exposure (identical across the
    heads that fired together) and others identify the body or the file (unique
    per camera, so never shared). We do not have to know which is which — we can
    just score every candidate: the right key is the one that maximises the
    number of exposures covered by *every* camera.

    Returns (token positions, how many frames all cameras share). An empty tuple
    means nothing worked.
    """
    if len(per_cam) < 2:
        return (), 0
    token_counts = [min(len(_digit_tokens(n)) for n in names)
                    for names in per_cam.values() if names]
    if not token_counts:
        return (), 0

    best: tuple[tuple[int, ...], int] = ((), 0)
    # Single fields first, then pairs — a pair is only preferred when it strictly
    # beats every single field, so we keep the simplest key that works.
    positions = range(min(token_counts))
    for combo in [(i,) for i in positions] + list(itertools.combinations(positions, 2)):
        keyed = {cam: ["_".join(_digit_tokens(n)[i] for i in combo) for n in names]
                 for cam, names in per_cam.items()}
        shared = set.intersection(*(set(v) for v in keyed.values()))
        if len(shared) > best[1]:
            best = (combo, len(shared))
    return best


def _camera_from_filename(name: str) -> str:
    """The leading non-digit run of the filename: `N-1_0-61214.jpg` -> "N",
    `CAM_A_00042.jpg` -> "CAM_A". Nothing about the value is assumed — it is
    only used to tell one body from another."""
    stem = Path(name).name
    m = re.match(r"^([^\d]+?)[-_]?\d", stem)
    return m.group(1).strip("-_") if m else ""


def group_auto(names: list[str]) -> tuple[RigGrouping, list[str]]:
    """`auto` mode: split cameras, then discover the exposure key from the names.

    Cameras come from the folder when there is one folder per camera, and
    otherwise from the leading non-digit part of the filename, so a flat dataset
    whose bodies are only distinguishable by a filename prefix still works. No
    camera name is assumed anywhere: whatever strings the data uses become the
    camera ids.

    Also returns diagnostic lines, because both the camera split and the key are
    inferred and the user should be able to sanity-check them in the log.
    """
    notes: list[str] = []
    out = RigGrouping()
    per_cam: dict[str, list[str]] = {}
    for name in sorted(names):
        hit = _split_folder(name)
        if not hit:
            out.unmatched.append(name)
            continue
        per_cam.setdefault(hit[0], []).append(name)

    if len(per_cam) < 2:
        # One folder (or none): fall back to a filename prefix as the camera id.
        flat = sorted(names)
        by_prefix: dict[str, list[str]] = {}
        for name in flat:
            cam = _camera_from_filename(name)
            if cam:
                by_prefix.setdefault(cam, []).append(name)
        if len(by_prefix) >= 2 and sum(map(len, by_prefix.values())) == len(flat):
            notes.append(f"rig: auto split {len(by_prefix)} cameras by filename prefix "
                         f"({', '.join(sorted(by_prefix))}) — no per-camera folders found")
            per_cam, out.unmatched = by_prefix, []
        else:
            out.unmatched.extend(n for names_ in per_cam.values() for n in names_)
            notes.append("rig: auto found only one camera — expected either one folder "
                         "per camera, or filenames that start with a per-camera prefix")
            return out, notes

    combo, shared = auto_frame_key(per_cam)
    if not combo:
        out.unmatched.extend(n for names_ in per_cam.values() for n in names_)
        notes.append("rig: auto could not find a shared exposure key in the filenames "
                     "— try mode=gps, or mode=regex with an explicit pattern")
        return out, notes

    notes.append(f"rig: auto picked digit field(s) {list(combo)} as the exposure key "
                 f"({shared} exposures shared by every camera)")

    # A key that repeats inside one camera cannot identify an exposure; pairing on
    # it would bind images from *different* shots together, which is worse than
    # dropping them. Same key is dropped for every camera so frames stay aligned.
    keyed = {cam: [("_".join(_digit_tokens(n)[i] for i in combo), n) for n in names_]
             for cam, names_ in per_cam.items()}
    ambiguous: set[str] = set()
    for pairs in keyed.values():
        counts = Counter(k for k, _ in pairs)
        ambiguous |= {k for k, c in counts.items() if c > 1}
    if ambiguous:
        notes.append(f"rig: dropped {len(ambiguous)} ambiguous key(s) that repeat within a "
                     f"camera (e.g. {', '.join(sorted(ambiguous)[:3])}) — those exposures "
                     "cannot be matched safely")

    for cam, pairs in keyed.items():
        for key, name in pairs:
            if key in ambiguous:
                out.unmatched.append(name)
            else:
                out.frames.setdefault(cam, {})[key] = name
    return out, notes


def group_images(names: list[str], mode: str, regex: str = "",
                 gps: dict[str, tuple[float, float]] | None = None,
                 gps_tol: float = 0.5) -> RigGrouping:
    """Group relative image names into per-camera frame tables.

    `gps` maps image name -> (lat, lon) and is only used by mode="gps".
    """
    if mode not in RIG_MODES:
        raise ValueError(f"rig mode must be one of {RIG_MODES}, got {mode!r}")

    if mode == "auto":
        return group_auto(names)[0]

    out = RigGrouping()
    if mode == "regex":
        rx = compile_rig_regex(regex)

    if mode in ("folder", "regex"):
        for name in sorted(names):
            hit = (_split_folder(name) if mode == "folder"
                   else _split_regex(name, rx))
            if not hit:
                out.unmatched.append(name)
                continue
            cam, key = hit
            out.frames.setdefault(cam, {})[key] = name
        return out

    # --- gps: camera from the folder, frame from position clustering --------
    gps = gps or {}
    per_cam: dict[str, list[str]] = {}
    for name in sorted(names):
        hit = _split_folder(name)
        if not hit or name not in gps:
            out.unmatched.append(name)
            continue
        per_cam.setdefault(hit[0], []).append(name)

    # Cluster stations using the reference camera's images as anchors: every
    # other head joins the nearest anchor within tolerance. Anchors come from
    # the camera with the most images, which is the one least likely to have
    # dropped exposures.
    if not per_cam:
        return out
    anchor_cam = max(per_cam, key=lambda c: len(per_cam[c]))
    anchors = [(name, gps[name]) for name in per_cam[anchor_cam]]

    for cam, cam_names in per_cam.items():
        for name in cam_names:
            pos = gps[name]
            best, best_d = None, gps_tol
            for a_name, a_pos in anchors:
                d = _gps_metres(pos, a_pos)
                if d <= best_d:
                    best, best_d = a_name, d
            if best is None:
                out.unmatched.append(name)
                continue
            # frame key = the anchor image's stem: stable and human-readable
            out.frames.setdefault(cam, {})[Path(best).stem] = name
    return out


def build_staging(grouping: RigGrouping, img_root: Path,
                  staging: Path) -> dict[str, str]:
    """Materialise <staging>/<camera>/<frame_key><ext> symlinks.

    The extension is taken from the source file so COLMAP's reader still sees a
    normal image; the *stem* is what has to match across cameras.

    Returns {staged relative name: original relative name}. Downstream steps that
    key off the vendor's filenames — the EO CSV match above all — need that map,
    because restaging deliberately throws the original stem away.
    """
    mapping: dict[str, str] = {}
    for cam, table in grouping.frames.items():
        cam_dir = staging / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        for key, name in table.items():
            src = (img_root / name).resolve()
            link = cam_dir / (Path(key).stem + Path(name).suffix)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)
            mapping[f"{cam}/{link.name}"] = name
    return mapping


def write_rig_config(grouping: RigGrouping, path: Path,
                     ref_camera: str = "") -> str:
    """Emit rig_config.json for `colmap rig_configurator`. Returns the ref camera.

    Extrinsics are deliberately omitted: they are derived by passing an existing
    (rig-less) reconstruction as --input_path, which is both easier and better
    conditioned than hand-measured values.
    """
    cams = grouping.cameras
    if not cams:
        raise ValueError("no cameras were grouped — check the rig mode/regex")
    ref = ref_camera or cams[0]
    if ref not in cams:
        raise ValueError(f"ref camera {ref!r} not among grouped cameras: {cams}")

    # The ref sensor must be the FIRST entry, not merely flagged: ApplyRigConfig
    # walks config.cameras in array order and Rig::AddSensor aborts with
    # "The reference sensor needs to be added first" (rig.cc:42) if a non-ref
    # sensor is reached before AddRefSensor has run.
    entries: list[dict[str, object]] = [{"image_prefix": f"{ref}/", "ref_sensor": True}]
    entries += [{"image_prefix": f"{cam}/"} for cam in cams if cam != ref]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"cameras": entries}], indent=2), encoding="utf-8")
    return ref


def summarize(grouping: RigGrouping) -> list[str]:
    """Human-readable grouping report — read this before trusting a rig run."""
    cams = grouping.cameras
    keys = grouping.frame_keys
    complete = grouping.complete_frames()
    lines = [
        f"rig: {len(cams)} cameras {cams}",
        f"rig: {len(keys)} frames, {len(complete)} complete "
        f"({len(keys) - len(complete)} partial)",
    ]
    for cam in cams:
        lines.append(f"rig:   {cam}: {len(grouping.frames[cam])} images")
    if grouping.unmatched:
        head = ", ".join(grouping.unmatched[:5])
        lines.append(f"rig: {len(grouping.unmatched)} images did not match "
                     f"(e.g. {head})")
    if not complete:
        lines.append("rig: WARNING no frame is covered by every camera — the "
                     "grouping is wrong; a rig built from this constrains nothing")
    elif len(complete) < len(keys) * 0.5:
        lines.append("rig: WARNING more than half the frames are partial — check "
                     "the mode/regex before relying on the result")
    return lines
