"""Parallel ffmpeg FullHD resize that optionally preserves EXIF/GPS.

ffmpeg's mjpeg encoder drops EXIF, so each resized JPEG can be patched back with
the original's APP1 segment (see _gps.read_app1_exif / graft_app1).
"""
from __future__ import annotations

import concurrent.futures as futures
import shutil
import subprocess
import time
from pathlib import Path

from ..config import settings
from ..runner import Cancelled, Runner
from ._gps import graft_app1, read_app1_exif

FFMPEG_BIN = settings.ffmpeg_bin
FULLHD_MAX = "1920"   # longest side cap for the "fullhd" resize option


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
