"""Parallel ffmpeg FullHD resize that optionally preserves EXIF/GPS.

ffmpeg's mjpeg encoder drops EXIF, so each resized JPEG can be patched back with
the original's APP1 segment (see _gps.read_app1_exif / graft_app1).
"""
from __future__ import annotations

import concurrent.futures as futures
import shutil
import subprocess
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
    return []


def resize_to_fullhd(img_root: str, lines: list[str], ws: Path,
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
    workers = resize_workers()
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
            app1 = read_app1_exif(src)               # carry GPS (+focal) past the re-encode
            if app1:
                graft_app1(dst, app1)

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
