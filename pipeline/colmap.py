"""Python port of colmap_pipeline.sh.

Stages (skippable via `stages`, re-runnable via force):
  stage -> extract -> match -> calibrate(global only) -> mapper -> undistort
Idempotency uses the same sentinels / output checks as the shell script, and the
banners it emits match `log()` so the panel's stage parser is unchanged.
"""
from __future__ import annotations

import concurrent.futures as futures
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from .config import settings
from .runner import Cancelled, Runner

COLMAP_STAGES = ["stage", "extract", "match", "calibrate", "mapper", "simplify",
                 "align", "undistort", "reorient"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
# colmap / ffmpeg binaries (PATH by default; override via COLMAP_BIN / FFMPEG_BIN).
COLMAP_BIN = settings.colmap_bin
FFMPEG_BIN = settings.ffmpeg_bin

COLMAP_DEFAULTS = {
    "vocab_tree": str(Path.home() / ".cache/colmap/vocab_tree_faiss_flickr100K_words256K.bin"),
    "vocab_tree_url": "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin",
    "camera_model": "OPENCV", "max_features": "4096", "camera_mode": "per_folder",
    "matcher": "both", "seq_overlap": "10", "num_matches": "50",
    "guided_matching": "1", "mapper": "global", "dataset_name": "training_dataset",
    "force": False, "nested_layout": False, "resize": "fullhd",
    # spatial_matcher (large GPS scenes): match only GPS-near images.
    "spatial_max_neighbors": "50", "spatial_max_distance": "100", "spatial_ignore_z": "1",
    # GPS metric alignment via model_aligner (optional, off by default): rewrite the
    # sparse model into a local ENU frame in real-world metres.
    "gps_align": False, "gps_align_type": "enu", "gps_align_max_error": "3.0",
    # pose_prior_mapper: GPS position uncertainty (metres). Consumer GPS ~3-5 m; RTK ~0.02.
    "prior_std_x": "3.0", "prior_std_y": "3.0", "prior_std_z": "5.0", "prior_robust_loss": "1",
    # GPU bundle adjustment for the incremental / pose_prior mappers (big speedup; on by default).
    "ba_gpu": True,
    # --- hierarchical-3d-gaussians large-scene method (MATCHER=custom, MAPPER=hierarchical) ---
    # custom matcher (make_colmap_custom_matcher): per-view match counts + loop anchors.
    "cm_n_seq": "0", "cm_n_quad": "10", "cm_n_loop": "5", "cm_n_gps": "25",
    "cm_loop_matches": "",
    # feature_extractor focal seed for uncalibrated large scenes (h3dgs uses 0.5); "" = COLMAP default.
    "focal_factor": "",
    # image_undistorter longest-side cap (h3dgs uses 2048); "" = no cap.
    "max_image_size": "",
    # optional foreground masks: undistort them through the same cameras as the images.
    "masks_dir": "",
    # simplify_images (drop pose-outlier cameras + trim observations) and auto_reorient
    # (PCA gravity-align + scale). Both off unless ticked; h3dgs runs both.
    "simplify": False, "simplify_mult_min_dist": "10",
    "reorient": False, "reorient_target_med_dist": "20", "reorient_upscale": "0",
    # hierarchical_mapper partition knobs (COLMAP 4.0.4); "" = COLMAP default
    # (leaf_max_num_images 500, image_overlap 50, num_workers -1 = auto).
    "hm_leaf_max_num_images": "", "hm_image_overlap": "", "hm_num_workers": "",
}
FULLHD_MAX = "1920"   # longest side cap for the "fullhd" resize option


def _download(url: str, dest: Path, r: Runner) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last: Exception | None = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(url) as resp, tmp.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            tmp.unlink(missing_ok=True)
            time.sleep(2)
    raise RuntimeError(f"vocab tree download failed: {last}")


def _resize_workers() -> int:
    return settings.resolved_resize_workers()


def _resize_enc_args(rel: str) -> list[str]:
    """Highest-quality encode for the output format. JPEG: q=1 + full-chroma 4:4:4
    (no subsampling). PNG/TIFF: lossless, so just rescale."""
    ext = Path(rel).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return ["-q:v", "1", "-pix_fmt", "yuvj444p"]
    if ext == ".png":
        return ["-compression_level", "1"]          # lossless; light zlib = faster write
    return []


def _resize_to_fullhd(img_root: str, lines: list[str], ws: Path,
                      force: bool, r: Runner, preserve_exif: bool = False) -> str:
    """Physically downscale every listed image so its longest side is <= FULLHD_MAX,
    writing real FullHD copies under ws/images_fullhd/<relpath> (aspect kept, never
    upscaled). The whole COLMAP run then operates on these — no in-COLMAP size cap.

    Speed: many ffmpeg workers in parallel (NVDEC/GPU can't accelerate JPEG stills,
    so we saturate CPU cores instead). Quality: Lanczos downscaling + max-quality
    encode. Idempotent via a sentinel; honors FORCE; resumes by skipping done files.

    preserve_exif (GPS flow): ffmpeg's JPEG encoder drops EXIF, so each resized JPEG
    gets the original's Exif APP1 grafted back in — keeping the FullHD downscale AND
    the GPS priors. Uses a distinct sentinel so toggling the mode rebuilds cleanly.
    Returns the new image root."""
    if not shutil.which(FFMPEG_BIN):
        raise RuntimeError(f"ffmpeg not found: '{FFMPEG_BIN}' (set FFMPEG_BIN); "
                           "needed for the FullHD resize")
    out_root = ws / "images_fullhd"
    sentinel = ws / (".resize_fullhd_exif.done" if preserve_exif else ".resize_fullhd.done")
    other = ws / (".resize_fullhd.done" if preserve_exif else ".resize_fullhd_exif.done")
    if sentinel.exists() and not force:
        r.log(f"skip resize (sentinel exists; set FORCE=1 to redo) -> {out_root}")
        return str(out_root)
    # force, or a complete run in the *other* EXIF mode -> rebuild from scratch (the
    # existing copies have the wrong EXIF state); otherwise keep partial files to resume.
    if force or (other.exists() and not sentinel.exists()):
        shutil.rmtree(out_root, ignore_errors=True)
        other.unlink(missing_ok=True)
    workers = _resize_workers()
    r.banner(f"resize input -> FullHD (longest side <= {FULLHD_MAX}px, Lanczos, "
             f"max quality, {workers} parallel"
             f"{', EXIF/GPS preserved' if preserve_exif else ''}) -> {out_root}")
    # cap the longest side, keep aspect, never upscale (min with original); -2 keeps
    # the other side an even number; Lanczos = best-quality downscaling filter.
    vf = (f"scale='if(gte(iw,ih),min({FULLHD_MAX},iw),-2)':"
          f"'if(gte(iw,ih),-2,min({FULLHD_MAX},ih))':flags=lanczos")

    def _one(rel: str) -> None:
        dst = out_root / rel
        if dst.exists() and not force:
            return                                   # resume a partial run cheaply
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = Path(img_root) / rel
        argv = [FFMPEG_BIN, "-y", "-nostdin", "-loglevel", "error",
                "-i", str(src), "-vf", vf, *_resize_enc_args(rel), str(dst)]
        res = subprocess.run(argv, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg resize failed for {rel}: "
                               f"{res.stderr.strip()[:200]}")
        if preserve_exif and Path(rel).suffix.lower() in (".jpg", ".jpeg"):
            app1 = _read_app1_exif(src)              # carry GPS (+focal) past the re-encode
            if app1:
                _graft_app1(dst, app1)

    n, done = len(lines), 0
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        pending = [ex.submit(_one, rel) for rel in lines]
        for fut in futures.as_completed(pending):
            if r.cancelled:
                ex.shutdown(wait=False, cancel_futures=True)
                raise Cancelled()
            fut.result()                             # propagate ffmpeg failures
            done += 1
            if done % 250 == 0 or done == n:
                r.log(f"  resized {done}/{n}")
    sentinel.touch()
    return str(out_root)


def _has_images(folder: Path) -> bool:
    return folder.is_dir() and any(
        c.is_file() and c.suffix.lower() in IMAGE_EXTS for c in folder.iterdir())


def _list_images(folder: Path, prefix: str) -> list[str]:
    """Image files directly in `folder` (symlinks followed), relative as prefix/name
    (or just name when prefix is '' — a single flat folder = image_root itself)."""
    out: list[str] = []
    if folder.is_dir():
        for entry in folder.iterdir():
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
                out.append(f"{prefix}/{entry.name}" if prefix else entry.name)
    return sorted(out)


def _list_image_names(folder: Path) -> list[str]:
    """Bare image filenames directly in `folder`, sorted. The h3dgs custom matcher
    pairs images by per-folder frame order, so it needs the names without the group
    prefix (the prefix is re-attached when emitting "prefix/name" match pairs)."""
    out: list[str] = []
    if folder.is_dir():
        for entry in folder.iterdir():
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
                out.append(entry.name)
    return sorted(out)


def _resolve_layout(img_root: str, folders: list[str], layout: str,
                    force_nested: bool, workspace: str | None = None) -> tuple[list[str], bool, str]:
    """Return (folders, nested, layout_name) for the three fixed input formats:
      single  : XXX/*.jpg              -> folders=[''] (image_root itself), 1 camera
      multi   : ROOT/<group>/*.jpg     -> folders=[groups], flat, camera per group
      nested  : ROOT/<group>/<vid>/*.jpg -> staged into groups, camera per group
    """
    root = Path(img_root)
    # Ignore the workspace dir if the user nested it inside image_root, so it isn't
    # mistaken for a camera group.
    skip = set()
    if workspace:
        try:
            wp = Path(workspace)
            if wp.resolve().parent == root.resolve():
                skip.add(wp.name)
        except OSError:
            pass
    subdirs = sorted([x.name for x in root.iterdir()
                      if x.is_dir() and not x.name.startswith(".") and x.name not in skip])

    # Explicit single, or auto when the root itself holds images (subdirs are then
    # likely junk such as the workspace) -> single flat folder.
    if layout == "single" or (layout == "auto" and _has_images(root)):
        return (folders if folders else [""]), False, "single"

    chosen = folders or subdirs
    if not chosen:
        if _has_images(root):
            return [""], False, "single"
        raise FileNotFoundError(f"no images or subfolders found under: {img_root}")

    if layout == "nested" or force_nested:
        return chosen, True, "nested"
    if layout == "multi":
        return chosen, False, "multi"

    # auto: nested if the first group has no direct images but does have subdirs
    nested = False
    for f in chosen:
        g = root / f
        if g.is_dir() and not _has_images(g) and any(c.is_dir() for c in g.iterdir()):
            nested = True
        break
    return chosen, nested, ("nested" if nested else "multi")


def _jpeg_gps_present(path: Path) -> bool:
    """True iff the JPEG at `path` carries a non-empty EXIF GPS IFD (lat + lon).

    Pure stdlib (the project ships no Pillow/piexif): walk JPEG segments to the Exif
    APP1, parse the TIFF/IFD0 for the GPS IFD pointer (tag 0x8825), then confirm that
    GPS IFD holds GPSLatitude (0x0002) and GPSLongitude (0x0004). Best-effort — any
    parse hiccup or non-JPEG returns False."""
    import struct
    try:
        with path.open("rb") as fh:
            if fh.read(2) != b"\xff\xd8":                 # not a JPEG (no SOI)
                return False
            app1 = b""
            while True:                                   # find the Exif APP1 segment
                hdr = fh.read(2)
                if len(hdr) < 2 or hdr[0] != 0xFF:
                    return False
                marker = hdr[1]
                if marker == 0xD8 or 0xD0 <= marker <= 0xD7:
                    continue                              # standalone markers, no length
                if marker == 0xDA or marker == 0xD9:      # SOS / EOI: pixel data, give up
                    return False
                seg_len = int.from_bytes(fh.read(2), "big")
                if seg_len < 2:
                    return False
                body = fh.read(seg_len - 2)
                if marker == 0xE1 and body[:6] == b"Exif\x00\x00":
                    app1 = body[6:]                       # the TIFF block
                    break
        if len(app1) < 8 or app1[:2] not in (b"II", b"MM"):
            return False
        bo = "<" if app1[:2] == b"II" else ">"
        ifd0 = struct.unpack(bo + "I", app1[4:8])[0]

        def tags(off: int) -> dict[int, int]:
            n = struct.unpack(bo + "H", app1[off:off + 2])[0]
            out: dict[int, int] = {}
            for k in range(n):
                e = off + 2 + k * 12
                tag = struct.unpack(bo + "H", app1[e:e + 2])[0]
                out[tag] = struct.unpack(bo + "I", app1[e + 8:e + 12])[0]
            return out

        gps_off = tags(ifd0).get(0x8825)
        if not gps_off:
            return False
        g = tags(gps_off)
        return 0x0002 in g and 0x0004 in g                # GPSLatitude + GPSLongitude
    except Exception:                                     # noqa: BLE001 — detection is best-effort
        return False


def _read_app1_exif(path: Path) -> bytes | None:
    """Return the raw Exif APP1 segment (FF E1 + length + body) from a JPEG, or None.
    Walks the JPEG segments rather than scanning, so it grabs the real Exif block."""
    try:
        with path.open("rb") as fh:
            if fh.read(2) != b"\xff\xd8":
                return None
            while True:
                hdr = fh.read(2)
                if len(hdr) < 2 or hdr[0] != 0xFF:
                    return None
                marker = hdr[1]
                if marker == 0xD8 or 0xD0 <= marker <= 0xD7:
                    continue
                if marker == 0xDA or marker == 0xD9:      # SOS / EOI: header is over
                    return None
                ln = fh.read(2)
                body = fh.read(int.from_bytes(ln, "big") - 2)
                if marker == 0xE1 and body[:6] == b"Exif\x00\x00":
                    return b"\xff\xe1" + ln + body
    except Exception:                                     # noqa: BLE001
        return None


def _graft_app1(dst: Path, app1: bytes) -> bool:
    """Splice an Exif APP1 segment in right after the SOI of the JPEG at `dst`.
    ffmpeg's mjpeg output has no EXIF, so re-inserting the original's restores GPS
    (and focal). The TIFF block is copied verbatim — its internal offsets stay valid.
    Returns True on success."""
    try:
        data = dst.read_bytes()
        if data[:2] != b"\xff\xd8" or not app1:
            return False
        dst.write_bytes(data[:2] + app1 + data[2:])
        return True
    except OSError:
        return False


def _gps_coverage(img_root: str, lines: list[str], r: Runner) -> tuple[int, int]:
    """Return (n_with_gps, n_total) for the inputs. The GPS pipeline needs a prior on
    EVERY image (a frame without one can't be spatially matched or anchored), so this
    can't sample — it checks each image (in parallel, the set can be large). Only JPEGs
    can carry EXIF GPS, so non-JPEG inputs count toward n_total but never toward
    n_with_gps — any PNG/TIFF therefore makes coverage incomplete."""
    n_total = len(lines)
    cand = [ln for ln in lines if Path(ln).suffix.lower() in (".jpg", ".jpeg")]
    if not cand:
        return 0, n_total
    n_gps = 0
    with futures.ThreadPoolExecutor(max_workers=_resize_workers()) as ex:
        pending = [ex.submit(_jpeg_gps_present, Path(img_root) / rel) for rel in cand]
        for fut in futures.as_completed(pending):
            if r.cancelled:
                ex.shutdown(wait=False, cancel_futures=True)
                raise Cancelled()
            if fut.result():
                n_gps += 1
    return n_gps, n_total


def run_colmap(p: dict, r: Runner) -> None:
    d = {**COLMAP_DEFAULTS, **{k: v for k, v in p.items() if v is not None}}
    img_root = p["image_root"]
    ws = Path(p["workspace"])
    folders = list(p.get("folders") or [])
    stages = list(p.get("stages") or COLMAP_STAGES)
    mapper, camera_mode, matcher = d["mapper"], d["camera_mode"], d["matcher"]
    force = bool(d["force"])
    vocab_tree = Path(d["vocab_tree"])
    # FullHD resize: cap longest side at 1920 for feature extraction + undistorted output.
    fullhd = str(d.get("resize")) == "fullhd"
    gps_align = bool(d["gps_align"])
    gps_align_type = str(d["gps_align_type"])
    gps_align_max_error = str(d["gps_align_max_error"])
    ba_gpu = bool(d["ba_gpu"])   # GPU bundle adjustment (incremental / pose_prior only)
    # h3dgs large-scene method options
    focal_factor = str(d["focal_factor"]).strip()
    max_image_size = str(d["max_image_size"]).strip()
    masks_dir = str(d["masks_dir"]).strip()
    simplify = bool(d["simplify"])
    reorient = bool(d["reorient"])
    cm_n_seq, cm_n_quad, cm_n_loop = str(d["cm_n_seq"]), str(d["cm_n_quad"]), str(d["cm_n_loop"])
    cm_n_gps = int(str(d["cm_n_gps"]) or 0)
    cm_loop_ints = [int(x) for x in str(d["cm_loop_matches"]).replace(",", " ").split()]
    hm_leaf = str(d["hm_leaf_max_num_images"]).strip()
    hm_overlap = str(d["hm_image_overlap"]).strip()
    hm_workers = str(d["hm_num_workers"]).strip()

    if mapper not in ("global", "incremental", "pose_prior", "hierarchical"):
        raise ValueError(f"MAPPER must be 'global', 'incremental', 'pose_prior', or "
                         f"'hierarchical' (got: {mapper})")
    if camera_mode not in ("per_folder", "single"):
        raise ValueError(f"CAMERA_MODE must be 'per_folder' or 'single' (got: {camera_mode})")
    if matcher not in ("sequential", "vocab", "both", "spatial", "custom"):
        raise ValueError(f"MATCHER must be 'sequential', 'vocab', 'both', 'spatial', or "
                         f"'custom' (got: {matcher})")

    if not Path(img_root).is_dir():
        raise FileNotFoundError(f"image_root not found: {img_root}")
    folders, nested, layout_name = _resolve_layout(
        img_root, folders, str(p.get("layout") or "auto"), bool(d["nested_layout"]),
        workspace=p.get("workspace"))
    shown = "<root>" if folders == [""] else " ".join(folders)
    r.log(f"layout={layout_name}  folders={shown}  nested={int(nested)}")
    for f in folders:
        if f and not (Path(img_root) / f).is_dir():
            raise FileNotFoundError(f"subfolder not found: {Path(img_root) / f}")

    def stage_on(s: str) -> bool:
        return s in stages

    def need(path: Path | str) -> bool:
        return force or not Path(path).exists()

    if not vocab_tree.exists():
        r.log(f"vocab tree not found, downloading: {d['vocab_tree_url']} -> {vocab_tree}")
        _download(d["vocab_tree_url"], vocab_tree, r)
    if not shutil.which(COLMAP_BIN):
        raise RuntimeError(f"colmap not found: '{COLMAP_BIN}' (set COLMAP_BIN or add to PATH)")

    dense_dir = ws / f"{d['dataset_name']}_{mapper}_mapper"
    (ws / "sparse").mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)
    db = ws / "database.db"
    lst = ws / "image_list.txt"
    r.log(f"=== run started {time.strftime('%F %T')} | folders: {' '.join(folders)} ===")

    # 0. stage (NESTED_LAYOUT): flatten <group>/<video>/*.jpg -> staging/<group>/<video>_<file> symlinks
    if stage_on("stage") and not nested:
        r.log("skip stage (not a nested layout)")
    if stage_on("stage") and nested:
        stage_root = ws / "staging"
        sentinel = ws / ".stage.done"
        abs_root = Path(os.path.realpath(img_root))
        if need(sentinel):
            r.banner(f"stage nested layout -> {stage_root} (NESTED_LAYOUT=1)")
            if force:
                shutil.rmtree(stage_root, ignore_errors=True)
            for grp in folders:
                (stage_root / grp).mkdir(parents=True, exist_ok=True)
                grp_total = 0
                for vdir in sorted([x for x in (abs_root / grp).iterdir() if x.is_dir()]):
                    vcount = 0
                    for img in sorted([f for f in vdir.iterdir()
                                       if f.is_file() and f.suffix.lower() in IMAGE_EXTS]):
                        link = stage_root / grp / f"{vdir.name}_{img.name}"
                        if link.is_symlink() or link.exists():
                            link.unlink()
                        link.symlink_to(img)
                        vcount += 1
                    if vcount > 0:
                        r.log(f"  {grp}/{vdir.name}: {vcount} images")
                        grp_total += vcount
                r.log(f"  {grp} total: {grp_total}")
                if grp_total == 0:
                    raise FileNotFoundError(f"no images found under group: {grp}")
            sentinel.touch()
        else:
            r.log("skip stage (sentinel exists; set FORCE=1 to redo)")
        img_root = str(stage_root)
        r.log(f"rebased IMG_ROOT={img_root}")

    # image_list.txt (paths relative to img_root)
    lines: list[str] = []
    for f in folders:
        lines += _list_images(Path(img_root) / f, f)
    lst.write_text("\n".join(lines) + ("\n" if lines else ""))
    r.log(f"image_list: {len(lines)} images across {len(folders)} folder(s): {' '.join(folders)}")
    if not lines:
        raise FileNotFoundError("no images found")

    # EXIF GPS coverage. COLMAP reads EXIF GPS into the DB pose priors that
    # spatial_matcher / pose_prior_mapper / model_aligner consume. The GPS pipeline
    # requires GPS on EVERY image — a frame without a prior can't be spatially matched
    # or anchored — so any GPS option needs FULL coverage, checked here before any work
    # runs. The FullHD resize still runs (so we keep the downscale); it just preserves
    # the EXIF GPS through the re-encode (see preserve_exif below).
    n_gps, n_total = _gps_coverage(img_root, lines, r)
    gps_present = (n_total > 0 and n_gps == n_total)   # GPS flow valid only at 100%
    gps_opts = [name for name, on in (("MATCHER=spatial", matcher == "spatial"),
                                      ("MAPPER=pose_prior", mapper == "pose_prior"),
                                      ("GPS_ALIGN", gps_align)) if on]
    if gps_opts:
        if not gps_present:
            raise RuntimeError(
                f"GPS option(s) selected ({', '.join(gps_opts)}) but only {n_gps}/{n_total} "
                "inputs carry EXIF GPS — the GPS pipeline needs GPS on EVERY image, so it "
                "aborts before any work runs. Provide GPS-tagged photos for all inputs, or "
                "switch to a non-GPS setup (MATCHER=vocab/both, MAPPER=global/incremental, "
                "GPS 對齊 off). Note: video frames carry no per-frame GPS — it lives in the "
                "container, not the frames.")
        r.log(f"EXIF GPS on all {n_total} inputs -> GPS flow enabled ({', '.join(gps_opts)})"
              + ("; FullHD resize will preserve the GPS EXIF" if fullhd else ""))
    elif n_gps:
        r.log(f"note: {n_gps}/{n_total} inputs have EXIF GPS but no GPS option selected; "
              "running normally" + (" (the FullHD resize will drop the GPS)" if fullhd else ""))

    # FullHD: physically downscale the inputs to FullHD copies and run the entire
    # pipeline on those (image_list paths stay relative, so they remain valid). When the
    # GPS flow is on, the resize grafts each original's EXIF back so GPS survives.
    if fullhd:
        # keep EXIF GPS through the re-encode when any GPS flow needs it — including
        # the custom matcher's optional GPS-neighbour pairs (not a hard-fail option,
        # so it's not in gps_opts, but it still reads GPS off the resized images).
        preserve = bool(gps_opts) or (matcher == "custom" and cm_n_gps > 0)
        img_root = _resize_to_fullhd(img_root, lines, ws, force, r, preserve_exif=preserve)

    # 1. feature_extractor
    if stage_on("extract"):
        if need(db):
            r.banner(f"feature_extractor (CAMERA_MODE={camera_mode}, layout={layout_name}) -> {db}")
            if layout_name == "single":
                cam = ["--ImageReader.single_camera", "1"]            # one flat folder = 1 camera
            elif camera_mode == "per_folder":
                cam = ["--ImageReader.single_camera_per_folder", "1"]
            else:
                cam = ["--ImageReader.single_camera", "1"]
            # h3dgs seeds the focal length for uncalibrated large scenes (factor 0.5);
            # only passed when set, so normal runs keep COLMAP's default behavior.
            if focal_factor:
                cam += ["--ImageReader.default_focal_length_factor", focal_factor]
            r.run([COLMAP_BIN, "feature_extractor", "--database_path", str(db),
                   "--image_path", img_root, "--image_list_path", str(lst), *cam,
                   "--ImageReader.camera_model", str(d["camera_model"]),
                   "--SiftExtraction.max_num_features", str(d["max_features"])])
        else:
            r.log("skip extract (database.db exists; set FORCE=1 to redo)")

    # 2. matcher
    if stage_on("match"):
        sentinel = ws / ".match.done"
        if need(sentinel):
            if matcher in ("sequential", "both"):
                loop = 1 if matcher == "sequential" else 0
                r.banner(f"sequential_matcher (overlap={d['seq_overlap']}, loop_detection={loop})")
                seq = ["--SequentialMatching.overlap", str(d["seq_overlap"]),
                       "--SequentialMatching.quadratic_overlap", "1"]
                if matcher == "sequential":
                    seq += ["--SequentialMatching.loop_detection", "1",
                            "--SequentialMatching.loop_detection_period", str(d["seq_overlap"]),
                            "--SequentialMatching.loop_detection_num_images", str(d["num_matches"]),
                            "--SequentialMatching.vocab_tree_path", str(vocab_tree)]
                r.run([COLMAP_BIN, "sequential_matcher", "--database_path", str(db),
                       "--FeatureMatching.guided_matching", str(d["guided_matching"]), *seq])
            if matcher in ("vocab", "both"):
                r.banner(f"vocab_tree_matcher (num_images={d['num_matches']})")
                r.run([COLMAP_BIN, "vocab_tree_matcher", "--database_path", str(db),
                       "--FeatureMatching.guided_matching", str(d["guided_matching"]),
                       "--VocabTreeMatching.vocab_tree_path", str(vocab_tree),
                       "--VocabTreeMatching.num_images", str(d["num_matches"])])
            if matcher == "spatial":
                r.banner(f"spatial_matcher (GPS priors; max_neighbors="
                         f"{d['spatial_max_neighbors']}, max_distance="
                         f"{d['spatial_max_distance']}m, ignore_z={d['spatial_ignore_z']})")
                r.run([COLMAP_BIN, "spatial_matcher", "--database_path", str(db),
                       "--FeatureMatching.guided_matching", str(d["guided_matching"]),
                       "--SpatialMatching.max_num_neighbors", str(d["spatial_max_neighbors"]),
                       "--SpatialMatching.max_distance", str(d["spatial_max_distance"]),
                       "--SpatialMatching.ignore_z", str(d["spatial_ignore_z"])])
            if matcher == "custom":
                # h3dgs custom matcher: build an explicit match list (sequential +
                # quadratic frame-steps + loop closure + GPS neighbours) then import it
                # via matches_importer — far fewer pairs than exhaustive for large,
                # ordered multi-camera captures. Names come from the same folder→sorted
                # grouping used for extraction, so they match the DB exactly.
                from . import large_scene
                groups = [(f, _list_image_names(Path(img_root) / f)) for f in folders]
                matching_txt = ws / "matching.txt"
                r.banner(f"matches_importer (custom matcher: seq={cm_n_seq} quad={cm_n_quad} "
                         f"loop={cm_n_loop} gps={cm_n_gps}) -> {matching_txt}")
                npairs = large_scene.make_custom_matcher(
                    groups, img_root, str(matching_txt),
                    n_seq=int(cm_n_seq), n_quad=int(cm_n_quad), n_loop=int(cm_n_loop),
                    loop_matches=cm_loop_ints, n_gps=cm_n_gps)
                r.log(f"  custom match list: {npairs} pairs")
                r.run([COLMAP_BIN, "matches_importer", "--database_path", str(db),
                       "--match_list_path", str(matching_txt)])
            sentinel.touch()
        else:
            r.log("skip match (sentinel exists; set FORCE=1 to redo)")

    # 3. view_graph_calibrator (global only)
    if stage_on("calibrate"):
        sentinel = ws / ".calibrate.done"
        if mapper != "global":
            r.log(f"skip calibrate (MAPPER={mapper}; only required for global)")
        elif need(sentinel):
            r.banner(f"view_graph_calibrator -> {db}")
            r.run([COLMAP_BIN, "view_graph_calibrator", "--database_path", str(db)])
            sentinel.touch()
        else:
            r.log("skip calibrate (sentinel exists; set FORCE=1 to redo)")

    # 4. mapper -> sparse/0
    if stage_on("mapper"):
        cameras = ws / "sparse" / "0" / "cameras.bin"
        if need(cameras):
            extra: list[str] = []
            if mapper == "global":
                sub, label = "global_mapper", "colmap global_mapper"
            elif mapper == "pose_prior":
                # GPS priors folded into BA -> the output is already georeferenced
                # and metric (model_aligner is then redundant). overwrite_priors_
                # covariance=1 means the std_* below set the GPS uncertainty (metres).
                sub, label = "pose_prior_mapper", "colmap pose_prior_mapper (GPS)"
                extra = ["--use_robust_loss_on_prior_position", str(d["prior_robust_loss"]),
                         "--overwrite_priors_covariance", "1",
                         "--prior_position_std_x", str(d["prior_std_x"]),
                         "--prior_position_std_y", str(d["prior_std_y"]),
                         "--prior_position_std_z", str(d["prior_std_z"])]
            elif mapper == "hierarchical":
                # h3dgs large-scene mapper: partition into overlapping sub-models,
                # reconstruct in parallel, then merge — scales where global/incremental
                # choke. Tight global-BA tolerance matches the h3dgs recipe. Partition
                # knobs (leaf_max_num_images / image_overlap / num_workers) are passed
                # only when set, else COLMAP's defaults (500 / 50 / auto) apply.
                sub, label = "hierarchical_mapper", "colmap hierarchical_mapper"
                extra = ["--Mapper.ba_global_function_tolerance", "0.000001"]
                if hm_leaf:
                    extra += ["--leaf_max_num_images", hm_leaf]
                if hm_overlap:
                    extra += ["--image_overlap", hm_overlap]
                if hm_workers:
                    extra += ["--num_workers", hm_workers]
            else:
                sub, label = "mapper", "colmap mapper (incremental)"
            # GPU bundle adjustment: BA dominates incremental/pose_prior runtime, so
            # offloading it to CUDA is a big speedup (global_mapper / hierarchical_mapper
            # don't take this flag — they'd reject it — so only the two incremental
            # mappers get it).
            if ba_gpu and mapper in ("incremental", "pose_prior"):
                extra += ["--Mapper.ba_use_gpu", "1"]
                label += " [GPU BA]"
            r.banner(f"{label} -> {ws / 'sparse'}")
            r.run([COLMAP_BIN, sub, "--database_path", str(db),
                   "--image_path", img_root, "--output_path", str(ws / "sparse"), *extra])
        else:
            r.log("skip mapper (sparse/0/cameras.bin exists; set FORCE=1 to redo)")

    # 4a. simplify_images (h3dgs): drop pose-outlier cameras + trim the unmatched 2D
    # observations from sparse/0/images.bin — makes the model far cheaper to read and
    # removes stray mis-localized cameras. The original is kept as images_heavy.bin.
    # Only runs when SIMPLIFY is on.
    if stage_on("simplify"):
        sentinel = ws / ".simplify.done"
        model0 = ws / "sparse" / "0"
        if not simplify:
            r.log("skip simplify (SIMPLIFY off)")
        elif need(sentinel):
            if not (model0 / "cameras.bin").is_file():
                raise RuntimeError("sparse model missing, cannot simplify")
            from . import large_scene
            heavy = model0 / "images_heavy.bin"
            if force and heavy.is_file():          # restore the full model before a redo
                (model0 / "images.bin").unlink(missing_ok=True)
                heavy.rename(model0 / "images.bin")
            r.banner(f"simplify_images (mult_min_dist={d['simplify_mult_min_dist']}) -> {model0}")
            before, after = large_scene.simplify_images(
                str(model0), mult_min_dist=float(d["simplify_mult_min_dist"]))
            r.log(f"  images: {before} -> {after} (full model backed up as images_heavy.bin)")
            sentinel.touch()
        else:
            r.log("skip simplify (sentinel exists; set FORCE=1 to redo)")

    # 4b. model_aligner (optional GPS metric alignment): rewrite sparse/0 in place
    # into a local ENU frame in real-world metres, using the DB's GPS pose priors.
    # Only runs when explicitly enabled AND GPS was actually found (otherwise it has
    # nothing to align to). Independent of MAPPER; complements the mesh ChArUco
    # mm-scaling — don't enable both on one dataset.
    if stage_on("align"):
        sentinel = ws / ".align.done"
        if not gps_align:
            r.log("skip align (GPS_ALIGN off)")
        elif not gps_present:
            r.log("skip align (GPS_ALIGN on but no EXIF GPS detected in inputs)")
        elif need(sentinel):
            if not (ws / "sparse" / "0" / "cameras.bin").is_file():
                raise RuntimeError("sparse model missing, cannot GPS-align")
            r.banner(f"model_aligner (GPS -> {gps_align_type} metres, "
                     f"max_error={gps_align_max_error}m) -> {ws / 'sparse' / '0'}")
            r.run([COLMAP_BIN, "model_aligner",
                   "--input_path", str(ws / "sparse" / "0"),
                   "--output_path", str(ws / "sparse" / "0"),
                   "--database_path", str(db), "--ref_is_gps", "1",
                   "--alignment_type", gps_align_type,
                   "--alignment_max_error", gps_align_max_error])
            sentinel.touch()
        else:
            r.log("skip align (sentinel exists; set FORCE=1 to redo)")

    # 5. image_undistorter -> dense_dir
    if stage_on("undistort"):
        uextra = ["--max_image_size", max_image_size] if max_image_size else []
        if need(dense_dir / "sparse" / "cameras.bin"):
            if not (ws / "sparse" / "0" / "cameras.bin").is_file():
                raise RuntimeError("sparse model missing, cannot undistort")
            r.banner(f"image_undistorter -> {dense_dir}"
                     + (f" (max_image_size={max_image_size})" if max_image_size else ""))
            r.run([COLMAP_BIN, "image_undistorter", "--image_path", img_root,
                   "--input_path", str(ws / "sparse" / "0"),
                   "--output_path", str(dense_dir), "--output_type", "COLMAP", *uextra])
        else:
            r.log(f"skip undistort ({dense_dir}/sparse/cameras.bin exists; set FORCE=1 to redo)")

        # 5b. masks (h3dgs, optional): undistort the foreground masks through the SAME
        # cameras as the images (a model copy with .png image names), then clean them
        # to uint8 0/255 under dense_dir/masks. Only when MASKS_DIR is set.
        if masks_dir:
            if not Path(masks_dir).is_dir():
                raise FileNotFoundError(f"masks_dir not found: {masks_dir}")
            mask_done = ws / ".masks.done"
            if need(mask_done):
                from . import large_scene
                src0 = ws / "sparse" / "0"
                mask_model = src0 / "masks"
                mask_model.mkdir(parents=True, exist_ok=True)
                shutil.copy(src0 / "cameras.bin", mask_model / "cameras.bin")
                shutil.copy(src0 / "points3D.bin", mask_model / "points3D.bin")
                large_scene.replace_images_by_masks(src0 / "images.bin",
                                                    mask_model / "images.bin")
                tmp_masks = ws / "tmp_masks"
                shutil.rmtree(tmp_masks, ignore_errors=True)
                r.banner(f"image_undistorter (masks) -> {dense_dir / 'masks'}")
                r.run([COLMAP_BIN, "image_undistorter", "--image_path", masks_dir,
                       "--input_path", str(mask_model), "--output_path", str(tmp_masks),
                       "--output_type", "COLMAP", *uextra])
                n = large_scene.make_mask_uint8(str(tmp_masks / "images"),
                                                str(dense_dir / "masks"))
                r.log(f"  wrote {n} uint8 masks -> {dense_dir / 'masks'}")
                shutil.rmtree(tmp_masks, ignore_errors=True)
                mask_done.touch()
            else:
                r.log("skip masks (sentinel exists; set FORCE=1 to redo)")

    # 6. auto_reorient (h3dgs): PCA gravity-align + uniformly scale the undistorted
    # model in place (poses + points rotate/scale; the images on disk are untouched,
    # so they stay valid). Force-safe: the pre-reorient model is snapshotted to
    # sparse_unaligned/ and always used as the input, so a redo never double-rotates.
    # Only runs when REORIENT is on.
    if stage_on("reorient"):
        sentinel = ws / ".reorient.done"
        dense_sparse = dense_dir / "sparse"
        backup = dense_dir / "sparse_unaligned"
        if not reorient:
            r.log("skip reorient (REORIENT off)")
        elif need(sentinel):
            if not (dense_sparse / "cameras.bin").is_file():
                raise RuntimeError("undistorted model missing, cannot reorient")
            from . import large_scene
            if not backup.exists():            # snapshot the un-reoriented model once
                shutil.copytree(dense_sparse, backup)
            r.banner(f"auto_reorient (gravity align + scale, target_med_dist="
                     f"{d['reorient_target_med_dist']}, upscale={d['reorient_upscale']}) "
                     f"-> {dense_sparse}")
            scale, ni, npts = large_scene.auto_reorient(
                str(backup), str(dense_sparse),
                upscale=float(d["reorient_upscale"]),
                target_med_dist=float(d["reorient_target_med_dist"]))
            r.log(f"  reoriented {ni} cams / {npts} points, scale={scale:.5g}")
            sentinel.touch()
        else:
            r.log("skip reorient (sentinel exists; set FORCE=1 to redo)")

    r.banner(f"done. workspace={ws}")
