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

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

RIG_MODES = ("folder", "regex", "gps")

# Panel-facing defaults (mirrored into COLMAP_DEFAULTS).
RIG_DEFAULTS = {
    "rig_enable": False,
    "rig_mode": "folder",
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


def group_images(names: list[str], mode: str, regex: str = "",
                 gps: dict[str, tuple[float, float]] | None = None,
                 gps_tol: float = 0.5) -> RigGrouping:
    """Group relative image names into per-camera frame tables.

    `gps` maps image name -> (lat, lon) and is only used by mode="gps".
    """
    if mode not in RIG_MODES:
        raise ValueError(f"rig mode must be one of {RIG_MODES}, got {mode!r}")

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


def build_staging(grouping: RigGrouping, img_root: Path, staging: Path) -> int:
    """Materialise <staging>/<camera>/<frame_key><ext> symlinks. Returns count.

    The extension is taken from the source file so COLMAP's reader still sees a
    normal image; the *stem* is what has to match across cameras.
    """
    n = 0
    for cam, table in grouping.frames.items():
        cam_dir = staging / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        for key, name in table.items():
            src = (img_root / name).resolve()
            link = cam_dir / (Path(key).stem + Path(name).suffix)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)
            n += 1
    return n


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
