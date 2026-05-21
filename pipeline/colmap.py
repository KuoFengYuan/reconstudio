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

from .runner import Cancelled, Runner

COLMAP_STAGES = ["stage", "extract", "match", "calibrate", "mapper", "align", "undistort"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
# colmap binary: PATH by default, override with COLMAP_BIN for non-standard installs.
COLMAP_BIN = os.environ.get("COLMAP_BIN", "colmap")
# ffmpeg: used for the FullHD resize step (PATH by default; override via FFMPEG_BIN).
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

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
}
FULLHD_MAX = "1920"   # longest side cap for the "fullhd" resize option


def _download(url: str, dest: Path, r: Runner) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last: Exception | None = None
    for attempt in range(3):
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
    env = os.environ.get("COLMAP_PANEL_RESIZE_WORKERS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min((os.cpu_count() or 8), 32))   # CPU-bound encode; cap to be polite


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
        return ([""] if not folders else folders), False, "single"

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
    cand = [l for l in lines if Path(l).suffix.lower() in (".jpg", ".jpeg")]
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

    if mapper not in ("global", "incremental", "pose_prior"):
        raise ValueError(f"MAPPER must be 'global', 'incremental', or 'pose_prior' (got: {mapper})")
    if camera_mode not in ("per_folder", "single"):
        raise ValueError(f"CAMERA_MODE must be 'per_folder' or 'single' (got: {camera_mode})")
    if matcher not in ("sequential", "vocab", "both", "spatial"):
        raise ValueError(f"MATCHER must be 'sequential', 'vocab', 'both', or 'spatial' (got: {matcher})")

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

    def need(path: "Path | str") -> bool:
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
        img_root = _resize_to_fullhd(img_root, lines, ws, force, r,
                                     preserve_exif=bool(gps_opts))

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
            else:
                sub, label = "mapper", "colmap mapper (incremental)"
            # GPU bundle adjustment: BA dominates incremental/pose_prior runtime, so
            # offloading it to CUDA is a big speedup (global_mapper has no such flag —
            # it'd reject the option — so only the two incremental-based mappers get it).
            if ba_gpu and mapper != "global":
                extra += ["--Mapper.ba_use_gpu", "1"]
                label += " [GPU BA]"
            r.banner(f"{label} -> {ws / 'sparse'}")
            r.run([COLMAP_BIN, sub, "--database_path", str(db),
                   "--image_path", img_root, "--output_path", str(ws / "sparse"), *extra])
        else:
            r.log("skip mapper (sparse/0/cameras.bin exists; set FORCE=1 to redo)")

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
        if need(dense_dir / "sparse" / "cameras.bin"):
            if not (ws / "sparse" / "0" / "cameras.bin").is_file():
                raise RuntimeError("sparse model missing, cannot undistort")
            r.banner(f"image_undistorter -> {dense_dir}")
            r.run([COLMAP_BIN, "image_undistorter", "--image_path", img_root,
                   "--input_path", str(ws / "sparse" / "0"),
                   "--output_path", str(dense_dir), "--output_type", "COLMAP"])
        else:
            r.log(f"skip undistort ({dense_dir}/sparse/cameras.bin exists; set FORCE=1 to redo)")

    r.banner(f"done. workspace={ws}")
