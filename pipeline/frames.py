"""Python port of extract_frames.sh.

video -> (ffmpeg fps=N,blurdetect) -> JPGs + per-frame blur scores -> keep the
sharp ones. With flatten=True the sharp frames land directly in the video's
output dir (COLMAP-ready <group>/frames_<video>/*.jpg layout).

Decoding uses GPU (NVDEC, `-hwaccel cuda`) automatically when the ffmpeg binary
supports it, falling back to CPU per-video on failure. Set FFMPEG_HWACCEL=none
to force CPU. The sampled frames are identical to CPU decoding.
"""
from __future__ import annotations

import concurrent.futures
import functools
import os
import re
import shutil
import subprocess
from pathlib import Path

from .config import settings
from .runner import Cancelled, PipelineError, Runner

VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".mkv", ".avi"}
_BLUR_RE = re.compile(r"blur: ([0-9]+\.[0-9]+)")

FRAMES_DEFAULTS = {"fps": "1", "keep_pct": "90", "threshold": "", "flatten": True}


def _ffmpeg_bin(explicit: str | None) -> str:
    # PATH ffmpeg by default; FFMPEG_BIN (e.g. an NVDEC build) overrides. A bare name
    # like "ffmpeg" isn't a file -> fall through to PATH resolution.
    cand = explicit or settings.ffmpeg_bin
    if os.path.sep in cand and not (os.path.isfile(cand) and os.access(cand, os.X_OK)):
        return "ffmpeg"
    return cand


@functools.lru_cache(maxsize=8)
def _supports_cuda(ffmpeg: str) -> bool:
    if settings.ffmpeg_hwaccel.lower() in ("none", "0", "cpu"):
        return False
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-hwaccels"],
                             capture_output=True, text=True, timeout=10).stdout
        return "cuda" in out
    except Exception:
        return False


def _compute_out(video: str, base: str | None, outdir: str | None) -> str:
    """Mirror compute_out() in the shell script."""
    name = Path(video).stem
    if not outdir:
        return str(Path(video).parent / f"frames_{name}")
    if not base:
        return str(Path(outdir) / f"frames_{name}")
    rel = os.path.relpath(os.path.realpath(video), base)
    reldir = os.path.dirname(rel)
    if reldir in ("", "."):
        return str(Path(outdir) / f"frames_{name}")
    return str(Path(outdir) / reldir / f"frames_{name}")


def _expand_inputs(inputs: list[str], outdir: str | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            base = os.path.realpath(inp)
            vids = [str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
            for v in sorted(vids):
                pairs.append((v, _compute_out(v, base, outdir)))
        elif p.is_file():
            pairs.append((inp, _compute_out(inp, None, outdir)))
        else:
            raise FileNotFoundError(f"Not a file or directory: {inp}")
    return pairs


def _percentile_cut(scores: list[float], keep_pct: int) -> float:
    n = len(scores)
    k = int(n * keep_pct / 100)
    k = max(1, min(k, n))
    return sorted(scores)[k - 1]


def _process_video(video: str, out: str, *, fps: str, threshold: str | None,
                   keep_pct: str | None, flatten: bool, hwaccel: bool,
                   ffmpeg: str, r: Runner) -> bool:
    raw, sharp, blur = Path(out) / "raw", Path(out) / "sharp", Path(out) / "blurry"
    csv, fflog = Path(out) / "blur_scores.csv", Path(out) / "ffmpeg.log"
    for d in (raw, sharp, blur):
        d.mkdir(parents=True, exist_ok=True)

    r.log(f"==> Input : {video}")
    r.log(f"==> Output: {out}")
    r.log(f"==> fps={fps}  decode={'GPU(cuda)' if hwaccel else 'CPU'}  "
          f"threshold={threshold or '<unset>'}  keep_pct={keep_pct or '<unset>'}")

    def extract(use_hw: bool) -> None:
        # -hwaccel decodes on the GPU; frames auto-download for the CPU
        # fps/blurdetect filters, so the sampled output matches CPU decoding.
        pre = ["-hwaccel", "cuda"] if use_hw else []
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "verbose", "-y", *pre,
               "-i", video, "-vf", f"fps={fps},blurdetect", "-q:v", "2", "-an",
               str(raw / "frame_%06d.jpg")]
        r.run(cmd, stderr_to=fflog, check=True)

    r.log("==> Extracting frames and computing blur scores ...")
    try:
        extract(hwaccel)
    except PipelineError:
        if hwaccel:
            r.log("    GPU decode failed -> retrying on CPU")
            for f in raw.glob("frame_*.jpg"):
                f.unlink()
            extract(False)
        else:
            raise

    frames = sorted(raw.glob("frame_*.jpg"))
    nraw = len(frames)
    r.log(f"    extracted {nraw} frames")
    if nraw == 0:
        r.log(f"    ERROR: no frames extracted -- check {fflog}")
        return False

    r.log("==> Parsing blur scores ...")
    scores_txt = _BLUR_RE.findall(fflog.read_text(errors="replace"))
    if len(scores_txt) != nraw:
        r.log(f"    WARNING: {len(scores_txt)} scores vs {nraw} frames -- alignment may be off")

    rows: list[tuple[str, float]] = []
    with csv.open("w") as fh:
        fh.write("file,blur\n")
        for i, f in enumerate(frames):
            s = float(scores_txt[i]) if i < len(scores_txt) else 1.0
            rows.append((f.name, s))
            fh.write(f"{f.name},{s}\n")

    if keep_pct:
        cut = _percentile_cut([s for _, s in rows], int(keep_pct))
        r.log(f"==> Percentile cutoff: keep sharpest {keep_pct}% of {len(rows)} -> blur <= {cut}")
        local_thresh = cut
    else:
        local_thresh = float(threshold)
        r.log(f"==> Absolute cutoff: blur <= {local_thresh}")

    kept = dropped = 0
    for fname, score in rows:
        if score <= local_thresh:
            shutil.move(str(raw / fname), str(sharp / fname))
            kept += 1
        else:
            shutil.move(str(raw / fname), str(blur / fname))
            dropped += 1
    r.log(f"    -> {kept} kept / {dropped} dropped (of {nraw})")

    if flatten:
        for f in sharp.glob("*.jpg"):
            shutil.move(str(f), str(Path(out) / f.name))
        for d in (raw, sharp, blur):
            shutil.rmtree(d, ignore_errors=True)
        r.log(f"    flattened -> {out} (sharp frames only)")
    else:
        r.log(f"    sharp  : {sharp}")
    return True


def _default_workers(nvideos: int, hwaccel: bool) -> int:
    cpu = os.cpu_count() or 4
    # GPU decode frees the CPU, so more videos can overlap; CPU decode uses
    # ~6-7 threads each, so only a couple in parallel.
    per = 4 if hwaccel else max(1, cpu // 6)
    return max(1, min(per, nvideos, 8))


def run_frames(params: dict, r: Runner) -> None:
    """params: inputs(list), out_dir, fps, threshold, keep_pct, flatten,
    workers(int), ffmpeg_bin."""
    ffmpeg = _ffmpeg_bin(params.get("ffmpeg_bin"))
    fps = str(params.get("fps") or "1")
    threshold = (params.get("threshold") or "").strip() or None
    keep_pct = None if threshold else str(params.get("keep_pct") or "90")
    flatten = bool(params.get("flatten", True))
    outdir = (params.get("out_dir") or "").strip() or None
    hwaccel = _supports_cuda(ffmpeg)

    pairs = _expand_inputs(list(params["inputs"]), outdir)
    if not pairs:
        raise FileNotFoundError("No videos found.")

    workers = int(params.get("workers") or 0) or _default_workers(len(pairs), hwaccel)
    workers = max(1, min(workers, len(pairs)))
    r.log(f"==> Found {len(pairs)} video(s); fps={fps}, "
          f"decode={'GPU(NVDEC)' if hwaccel else 'CPU'}, parallel workers={workers}")
    for v, o in pairs:
        r.log(f"    - {v}  ->  {o}")
    r.log("")

    def work(j: int, video: str, out: str) -> bool:
        r.check_cancel()
        r.log(f"######## [{j}/{len(pairs)}] {video} ########")
        return _process_video(video, out, fps=fps, threshold=threshold,
                              keep_pct=keep_pct, flatten=flatten,
                              hwaccel=hwaccel, ffmpeg=ffmpeg, r=r)

    fail = 0
    cancelled = False
    if workers == 1:
        for j, (video, out) in enumerate(pairs, 1):
            try:
                if not work(j, video, out):
                    fail += 1
                    r.log("    !! failed, continuing")
            except Cancelled:
                cancelled = True
                break
            except Exception as exc:  # noqa: BLE001
                fail += 1
                r.log(f"    !! failed: {exc}")
            r.log("")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, j, v, o): v for j, (v, o) in enumerate(pairs, 1)}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    if not fut.result():
                        fail += 1
                except Cancelled:
                    cancelled = True
                except Exception as exc:  # noqa: BLE001
                    fail += 1
                    r.log(f"    !! failed ({futs[fut]}): {exc}")

    if cancelled or r.cancelled:
        raise Cancelled()

    ok = len(pairs) - fail
    r.log(f"==> All done: {ok} ok, {fail} failed (of {len(pairs)})")
    if flatten:
        r.log("    Sharp frames are in each video's dir; point COLMAP at the output root.")
    else:
        r.log("    Point COLMAP at each video's sharp/ directory.")
    if fail and ok == 0:
        raise RuntimeError(f"all {fail} video(s) failed")
