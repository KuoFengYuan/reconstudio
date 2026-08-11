"""Parallel ffmpeg FullHD resize that optionally preserves EXIF/GPS.

ffmpeg's mjpeg encoder drops EXIF, so each resized JPEG can be patched back with
the original's APP1 segment (see _gps.read_app1_exif / graft_app1).
"""
from __future__ import annotations

import concurrent.futures as futures
import os
import shutil
import struct
import subprocess
import threading
import time
from pathlib import Path

from ..config import settings
from ..runner import Cancelled, Runner
from ._gps import graft_app1, read_app1_exif

FFMPEG_BIN = settings.ffmpeg_bin
FULLHD_MAX = "1920"   # longest side cap for the "fullhd" resize option

# ffmpeg refuses to even OPEN a picture whose (w+128)*(h+128) reaches INT_MAX/8
# (libavutil av_image_check_size, compiled in — no CLI option raises it). That caps it
# at ~268 MP, so large-format aerial frames (e.g. a DMC's 14592x25728 = 375 MP) fail with
# "Picture size WxH is invalid" before a single pixel is decoded. Those go through
# _resize_oversized() below instead.
FFMPEG_SIZE_TERM_MAX = (2 ** 31 - 1) // 8
# Concurrent oversized decodes are capped separately: unlike ffmpeg (which streams), the
# fallback holds a whole decoded frame in RAM (375 MP RGB ~= 1.1 GB at full scale), so
# running one per resize worker would OOM the box.
OVERSIZED_WORKERS = 4


def resize_workers() -> int:
    return settings.resolved_resize_workers()


def _resize_enc_args(rel: str) -> list[str]:
    """Highest-quality encode for the output format. JPEG: q=1 + full-chroma 4:4:4
    (no subsampling). PNG/TIFF: lossless, so just rescale."""
    ext = Path(rel).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return ["-q:v", "1", "-pix_fmt", "yuvj444p"]
    if ext == ".png":
        return ["-compression_level", "1"]          # lossless; light zlib = faster write
    if ext in (".tif", ".tiff"):
        # lossless deflate (smallest of deflate/lzw/packbits; ffmpeg's tiff default is
        # uncompressed packbits). GPS does NOT survive this re-encode (ffmpeg's tiff
        # encoder drops EXIF, and the JPEG APP1 graft trick can't splice a TIFF's
        # offset-based GPS IFD) — for TIFF, GPS is read from the ORIGINALS and written
        # straight into the COLMAP DB pose_priors (see _gps.inject_pose_priors), so the
        # resized copies never need to carry it.
        return ["-compression_algo", "deflate"]
    return []


def _jpeg_dimensions(path: Path) -> tuple[int, int] | None:
    """(width, height) from a JPEG's SOF marker, reading only the header (stdlib, like
    _gps): walk the segments until a Start-Of-Frame, whose payload is
    precision(1) height(2) width(2). None if it isn't a parseable JPEG."""
    try:
        with path.open("rb") as fh:
            if fh.read(2) != b"\xff\xd8":                 # not a JPEG (no SOI)
                return None
            while True:
                hdr = fh.read(2)
                if len(hdr) < 2 or hdr[0] != 0xFF:
                    return None
                marker = hdr[1]
                if marker == 0xD8 or 0xD0 <= marker <= 0xD7:
                    continue                              # standalone markers, no length
                if marker in (0xDA, 0xD9):                # SOS / EOI: pixels, no SOF found
                    return None
                seg_len = int.from_bytes(fh.read(2), "big")
                if seg_len < 2:
                    return None
                body = fh.read(seg_len - 2)
                # SOF0..SOF15 carry the frame size; 0xC4/0xC8/0xCC are DHT/JPG/DAC, not SOF.
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    h, w = struct.unpack(">HH", body[1:5])
                    return (w, h)
    except Exception:                                     # noqa: BLE001 — probe is best-effort
        return None


def _ffmpeg_can_decode(path: Path) -> bool:
    """False only when we can prove the picture is past ffmpeg's built-in size ceiling."""
    dims = _jpeg_dimensions(path)
    if not dims:
        return True                                       # unknown -> let ffmpeg try
    w, h = dims
    return (w + 128) * (h + 128) < FFMPEG_SIZE_TERM_MAX


def _resize_oversized(src: Path, dst: Path, rel: str, max_size: str) -> None:
    """Downscale a picture ffmpeg cannot open, via OpenCV (already a requirement for the
    mask path). Decoding is done at a reduced DCT scale — libjpeg emits the 1/2, 1/4 or
    1/8 image directly, so a 375 MP frame costs ~1.5 s and ~11 MB instead of a full
    1.1 GB decode — then Lanczos brings it to the exact same target ffmpeg's filter would
    have produced. Output is 8-bit (matching ffmpeg's yuvj444p JPEG), and EXIF is dropped
    here exactly as it is on the ffmpeg path, so the caller's APP1 graft still restores GPS."""
    # must precede the cv2 import: OpenCV reads its own pixel ceiling at load time
    os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 ** 40))
    try:
        import cv2
    except ImportError as e:                              # pragma: no cover - env-dependent
        raise RuntimeError(
            f"{rel} is too large for ffmpeg to decode and OpenCV is missing "
            "(pip install opencv-python) — needed to resize oversized images") from e

    dims = _jpeg_dimensions(src)
    if dims:
        # cheapest DCT scale whose output is still >= the target (Lanczos then only downscales)
        w, h = dims
        cap, long_side = int(max_size), max(w, h)
        flag = cv2.IMREAD_COLOR
        for f, reduced in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4),
                           (2, cv2.IMREAD_REDUCED_COLOR_2)):
            if long_side // f >= min(cap, long_side):
                flag = reduced
                break
    else:
        flag = cv2.IMREAD_COLOR                        # not a JPEG: no header probe, no DCT scale
    img = cv2.imread(str(src), flag)
    if img is None:
        raise RuntimeError(f"{rel}: OpenCV could not decode the image either")
    if not dims:
        h, w = img.shape[:2]

    cap = int(max_size)
    long_side, short_side = max(w, h), min(w, h)
    # never upscale; keep aspect; even sides, matching the ffmpeg filter's -2
    new_long = min(cap, long_side)
    new_short = max(2, round(short_side * new_long / long_side / 2) * 2)
    target = (new_long, new_short) if w >= h else (new_short, new_long)
    if target != (img.shape[1], img.shape[0]):
        img = cv2.resize(img, target, interpolation=cv2.INTER_LANCZOS4)

    ext = dst.suffix.lower()
    if ext in (".jpg", ".jpeg"):                          # q~ffmpeg -q:v 1, no chroma subsampling
        params = [cv2.IMWRITE_JPEG_QUALITY, 98,
                  cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    elif ext in (".tif", ".tiff"):
        params = [cv2.IMWRITE_TIFF_COMPRESSION, 8]        # ADOBE_DEFLATE, lossless
    else:
        params = []
    if not cv2.imwrite(str(dst), img, params):
        raise RuntimeError(f"{rel}: OpenCV failed to write {dst}")


def resize_to_fullhd(img_root: str, lines: list[str], ws: Path, force: bool, r: Runner,
                     preserve_exif: bool = False, max_size: str = FULLHD_MAX) -> str:
    """Physically downscale every listed image so its longest side is <= `max_size`,
    writing real copies under ws/images_<max_size>/<relpath> (aspect kept, never upscaled).
    The whole COLMAP run then operates on these — no in-COLMAP size cap.

    `max_size` (default 1920) is configurable: raise it for higher-resolution training
    images. Crucially this step ALSO re-encodes via ffmpeg, which produces clean TIFFs —
    the original aerial TIFFs carry metadata that segfaults COLMAP's OpenImageIO TIFF
    writer during undistort, so running on these re-encoded copies fixes that even at near-
    full resolution (a 11664px source at max_size=4096 → clean 4096px copy).

    Speed: many ffmpeg workers in parallel (NVDEC/GPU can't accelerate JPEG stills,
    so we saturate CPU cores instead). Quality: Lanczos downscaling + max-quality
    encode. Idempotent via a sentinel; honors FORCE; resumes by skipping done files.

    preserve_exif (GPS flow): ffmpeg's JPEG encoder drops EXIF, so each resized JPEG
    gets the original's Exif APP1 grafted back in — keeping the downscale AND the GPS
    priors. Each size lands in its own images_<max_size> folder, so switching the cap
    builds a new folder and leaves the old one intact (toggling back is instant, no
    re-encode); the two EXIF modes share a folder, so the sentinel also tags the mode.
    Returns the new image root."""
    if not shutil.which(FFMPEG_BIN):
        raise RuntimeError(f"ffmpeg not found: '{FFMPEG_BIN}' (set FFMPEG_BIN); "
                           "needed for the resize")
    out_root = ws / f"images_{max_size}"
    tag = f"{max_size}{'_exif' if preserve_exif else ''}"
    sentinel = ws / f".resize_{tag}.done"
    if sentinel.exists() and not force:
        r.log(f"skip resize (sentinel exists; set FORCE=1 to redo) -> {out_root}")
        return str(out_root)
    # force, or a completed run at THIS size in the other EXIF mode (it shares this
    # images_<size> folder, so its copies carry the wrong EXIF state) -> rebuild. Other
    # SIZES keep their own images_<size> folder untouched, so switching the cap back is
    # instant; else keep partial files to resume this same size+mode.
    other = ws / f".resize_{max_size}{'' if preserve_exif else '_exif'}.done"
    if force or (other.exists() and not sentinel.exists()):
        shutil.rmtree(out_root, ignore_errors=True)
        other.unlink(missing_ok=True)
    workers = resize_workers()
    r.banner(f"resize input -> longest side <= {max_size}px (Lanczos, max quality, "
             f"{workers} parallel{', EXIF/GPS preserved' if preserve_exif else ''}; "
             f"also re-encodes to clean TIFF/JPEG) -> {out_root}")
    # cap the longest side, keep aspect, never upscale (min with original); -2 keeps
    # the other side an even number; Lanczos = best-quality downscaling filter.
    vf = (f"scale='if(gte(iw,ih),min({max_size},iw),-2)':"
          f"'if(gte(iw,ih),-2,min({max_size},ih))':flags=lanczos")

    big_sem = threading.Semaphore(max(1, min(OVERSIZED_WORKERS, workers)))
    announced = threading.Event()

    def _oversized(src: Path, dst: Path, rel: str) -> None:
        if not announced.is_set():
            announced.set()
            r.log(f"  {rel} exceeds ffmpeg's decode limit -> OpenCV reduced-scale "
                  f"fallback (max {max(1, min(OVERSIZED_WORKERS, workers))} at a time)")
        with big_sem:                                # bound peak RAM: whole frames in memory
            _resize_oversized(src, dst, rel, max_size)

    def _one(rel: str) -> None:
        dst = out_root / rel
        if dst.exists() and not force:
            return                                   # resume a partial run cheaply
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = Path(img_root) / rel
        if not _ffmpeg_can_decode(src):              # too big for ffmpeg; don't even spawn it
            _oversized(src, dst, rel)
        else:
            argv = [FFMPEG_BIN, "-y", "-nostdin", "-loglevel", "error",
                    "-i", str(src), "-vf", vf, *_resize_enc_args(rel), str(dst)]
            res = subprocess.run(argv, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                # the probe only proves the JPEG case; other formats (and any size check we
                # mis-predicted) land here, so retry oversized frames before giving up
                if "Picture size" in res.stderr and "is invalid" in res.stderr:
                    _oversized(src, dst, rel)
                else:
                    raise RuntimeError(f"ffmpeg resize failed for {rel}: "
                                       f"{res.stderr.strip()[:200]}")
        # both paths drop EXIF, so the graft below applies to either
        if preserve_exif and Path(rel).suffix.lower() in (".jpg", ".jpeg"):
            app1 = read_app1_exif(src)               # carry GPS (+focal) past the re-encode
            if app1:
                graft_app1(dst, app1)

    # Progress with a live ETA: emit on a time cadence (~every 4 s) AND a count cadence
    # (~20 updates), whichever comes first, so even a small batch of huge TIFFs (where the
    # old "every 250" never fired before completion) shows steady progress and a remaining
    # estimate instead of looking hung.
    n, done = len(lines), 0
    t0 = time.monotonic()
    next_log = t0 + 4.0
    log_every = max(1, n // 20)
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        pending = [ex.submit(_one, rel) for rel in lines]
        for fut in futures.as_completed(pending):
            if r.cancelled:
                ex.shutdown(wait=False, cancel_futures=True)
                raise Cancelled()
            fut.result()                             # propagate ffmpeg failures
            done += 1
            now = time.monotonic()
            if done == n or done % log_every == 0 or now >= next_log:
                next_log = now + 4.0
                el = now - t0
                rate = done / el if el > 0 else 0.0
                if done < n:
                    eta = (n - done) / rate if rate > 0 else 0.0
                    r.log(f"  resized {done}/{n}  ({rate:.1f} img/s, {el:.0f}s elapsed, "
                          f"~{eta:.0f}s left)")
                else:
                    r.log(f"  resized {done}/{n}  (done in {el:.0f}s, {rate:.1f} img/s avg)")
    sentinel.touch()
    return str(out_root)
