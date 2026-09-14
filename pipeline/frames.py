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
import csv
import functools
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import settings
from .runner import Cancelled, PipelineError, Runner

VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".mkv", ".avi"}
_BLUR_RE = re.compile(r"\bblur:\s*(\S+)", re.IGNORECASE)
_FRAME_RE = re.compile(r"frame_\d+\.jpg")

FRAMES_DEFAULTS = {"fps": "1", "mode": "hybrid", "keep_pct": "70", "threshold": "8", "flatten": True}


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


def validate_frame_options(params: dict) -> dict:
    """One validation path for the form and pipeline; old saved jobs keep their mode."""
    mode = params.get("mode") or ("threshold" if params.get("threshold") else
                                  "percentile" if params.get("keep_pct") else "hybrid")
    if mode not in ("hybrid", "percentile", "threshold"):
        raise ValueError("不支援的模糊篩選方式")
    def field(name):
        value = params.get(name)
        return str(FRAMES_DEFAULTS[name] if value is None or value == "" else value).strip()

    fps = field("fps")
    try:
        value = float(fps)
        if not math.isfinite(value) or value <= 0:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("每秒抽幀數必須是大於 0 的有限數字") from None
    options = {"mode": mode, "fps": fps, "threshold": None, "keep_pct": None}
    if mode in ("hybrid", "threshold"):
        threshold = field("threshold")
        try:
            value = float(threshold)
            if not math.isfinite(value) or value <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("模糊分數上限必須是大於 0 的有限數字") from None
        options["threshold"] = threshold
    if mode in ("hybrid", "percentile"):
        keep = field("keep_pct")
        if not keep.isdecimal() or not 1 <= int(keep) <= 100:
            raise ValueError("保留比例必須是 1 到 100 的整數")
        options["keep_pct"] = str(int(keep))
    return options


def _read_scores(log: Path, count: int) -> list[float]:
    scores = []
    with log.open(errors="replace") as stream:
        for line in stream:
            # Ignore metadata/filenames mentioning blur and the final blur mean.
            if "blurdetect" not in line:
                continue
            match = _BLUR_RE.search(line)
            if match:
                try:
                    scores.append(float(match[1]))
                except ValueError:
                    scores.append(math.nan)
    if len(scores) != count:
        raise PipelineError(f"模糊分數與影格數不一致（{len(scores)} / {count}），停止輸出以免錯配；請查看 {log}")
    return scores


def _select_frames(rows: list[tuple[str, float]], threshold: str | None,
                   keep_pct: str | None) -> tuple[dict[str, str], float | None]:
    valid = sorted((score, name) for name, score in rows if math.isfinite(score) and score >= 0)
    quota = max(1, int(len(valid) * int(keep_pct) / 100)) if keep_pct else len(valid)
    ranked = {name for _, name in valid[:quota]}
    cut = valid[quota - 1][0] if valid and keep_pct else None
    reasons = {}
    for name, score in rows:
        if not math.isfinite(score) or score < 0:
            reasons[name] = "invalid_score"
        elif threshold is not None and score > float(threshold):
            reasons[name] = "above_threshold"
        elif name not in ranked:
            reasons[name] = "percentile"
        else:
            reasons[name] = "kept"
    return reasons, cut


def _publish_frames(stage: Path, out: Path, moves: list[tuple[Path, Path]], r: Runner):
    """Replace only generated outputs, rolling them back on cancellation or I/O error."""
    backup = stage / "previous"
    backup.mkdir()
    saved, published = [], []
    old = []
    for folder in (out, out / "raw", out / "sharp", out / "blurry"):
        if folder.is_dir():
            old.extend(p for p in folder.glob("frame_*.jpg") if _FRAME_RE.fullmatch(p.name) and (p.is_file() or p.is_symlink()))
    old.extend(p for p in (out / "blur_scores.csv", out / "frame_quality.json") if p.is_file() or p.is_symlink())
    try:
        for source in old:
            r.check_cancel()
            dest = backup / source.relative_to(out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            source.replace(dest)
            saved.append((source, dest))
        for source, dest in moves:
            r.check_cancel()
            dest.parent.mkdir(parents=True, exist_ok=True)
            source.replace(dest)
            published.append(dest)
        r.check_cancel()
    except BaseException:
        for dest in reversed(published):
            dest.unlink(missing_ok=True)
        for original, stored in reversed(saved):
            stored.replace(original)
        shutil.rmtree(backup)
        raise
    shutil.rmtree(backup)


def _process_video(video: str, out: str, *, fps: str, threshold: str | None,
                   keep_pct: str | None, flatten: bool, hwaccel: bool,
                   ffmpeg: str, r: Runner) -> bool:
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".frames-work-", dir=output))
    raw = stage / "raw"
    raw.mkdir()
    fflog = output / "ffmpeg.log"
    r.log(f"==> Input : {video}")
    r.log(f"==> Output: {out}")
    r.log(f"==> fps={fps}  decode={'GPU(cuda)' if hwaccel else 'CPU'}  "
          f"threshold={threshold or '<unset>'}  keep_pct={keep_pct or '<unset>'}")

    def extract(use_hw: bool) -> None:
        pre = ["-hwaccel", "cuda"] if use_hw else []
        # Equal consecutive blur scores must remain separate log records.
        # Without repeat, FFmpeg collapses them and shifts every later frame.
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "repeat+verbose", "-y", *pre,
               "-i", video, "-map", "0:v:0", "-vf", f"fps={fps},blurdetect",
               "-fps_mode", "passthrough", "-q:v", "2", "-an", str(raw / "frame_%06d.jpg")]
        r.run(cmd, stderr_to=fflog, check=True)

    try:
        r.log("==> Extracting frames and computing blur scores ...")
        try:
            extract(hwaccel)
        except PipelineError:
            if not hwaccel:
                raise
            r.check_cancel()
            r.log("    GPU decode failed -> retrying on CPU")
            for f in raw.glob("frame_*.jpg"):
                f.unlink()
            extract(False)
        r.check_cancel()
        frames = sorted(raw.glob("frame_*.jpg"))
        nraw = len(frames)
        r.log(f"    extracted {nraw} frames")
        if not nraw:
            raise PipelineError(f"沒有抽取到影格，既有輸出保留；請查看 {fflog}")
        scores = _read_scores(fflog, nraw)
        rows = [(frame.name, score) for frame, score in zip(frames, scores, strict=True)]
        reasons, cut = _select_frames(rows, threshold, keep_pct)
        kept = sum(reason == "kept" for reason in reasons.values())
        counts = {reason: sum(value == reason for value in reasons.values())
                  for reason in ("kept", "above_threshold", "percentile", "invalid_score")}
        r.log(f"==> Quality cutoff: blur <= {threshold or '<unset>'}; percentile cutoff={cut}")
        r.log(f"    品質篩選：保留 {kept}、超過模糊門檻 {counts['above_threshold']}、"
              f"排序淘汰 {counts['percentile']}、無有效分數 {counts['invalid_score']}")
        report = stage / "blur_scores.csv"
        with report.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file", "blur", "decision"])
            writer.writerows((name, score, reasons[name]) for name, score in rows)
        summary = stage / "frame_quality.json"
        summary.write_text(json.dumps({"input": video, "fps": float(fps), "extracted": nraw,
                                      "kept": kept, "dropped": nraw - kept, "reasons": counts,
                                      "threshold": float(threshold) if threshold else None,
                                      "keep_pct": int(keep_pct) if keep_pct else None,
                                      "percentile_cutoff": cut}, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")
        if not kept:
            shutil.copy2(report, output / "rejected_frames.csv")
            shutil.copy2(summary, output / "rejected_quality.json")
            r.log(f"    -> 0 kept / {nraw} dropped (of {nraw})")
            raise PipelineError("沒有符合清晰度條件的影格；既有輸出保留，請查看 rejected_frames.csv，調整門檻或更換影片")
        moves = []
        for frame in frames:
            if reasons[frame.name] == "kept":
                moves.append((frame, (output if flatten else output / "sharp") / frame.name))
            elif not flatten:
                moves.append((frame, output / "blurry" / frame.name))
        moves.extend([(report, output / report.name), (summary, output / summary.name)])
        r.check_cancel()
        _publish_frames(stage, output, moves, r)
        r.log(f"    -> {kept} kept / {nraw - kept} dropped (of {nraw})")
        r.log(f"    清晰影格：{output if flatten else output / 'sharp'}")
        r.log(f"    篩選報告：{output / 'blur_scores.csv'}")
        return True
    finally:
        # If rollback itself failed, retain the backup for recovery.
        if (stage / "previous").exists():
            r.log(f"    輸出替換未完成，備份保留在 {stage / 'previous'}")
        else:
            shutil.rmtree(stage, ignore_errors=True)


def _default_workers(nvideos: int, hwaccel: bool) -> int:
    cpu = os.cpu_count() or 4
    # GPU decode frees the CPU, so more videos can overlap; CPU decode uses
    # ~6-7 threads each, so only a couple in parallel.
    per = 4 if hwaccel else max(1, cpu // 6)
    return max(1, min(per, nvideos, 8))


def run_frames(params: dict, r: Runner) -> None:
    """params: inputs(list), out_dir, fps, mode, threshold, keep_pct, flatten,
    workers(int), ffmpeg_bin."""
    ffmpeg = _ffmpeg_bin(params.get("ffmpeg_bin"))
    options = validate_frame_options(params)
    fps, threshold, keep_pct = options["fps"], options["threshold"], options["keep_pct"]
    flatten = bool(params.get("flatten", True))
    outdir = (params.get("out_dir") or "").strip() or None
    hwaccel = _supports_cuda(ffmpeg)

    pairs = _expand_inputs(list(params["inputs"]), outdir)
    if not pairs:
        raise FileNotFoundError("No videos found.")

    unique, destinations = [], {}
    for video, out in pairs:
        target, source = str(Path(out).resolve()), str(Path(video).resolve())
        if target in destinations:
            if destinations[target] != source:
                raise ValueError(f"兩支影片會寫入同一輸出資料夾：{out}；請先將同名影片重新命名")
            continue
        destinations[target] = source
        unique.append((video, out))
    pairs = unique

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
    if fail:
        raise RuntimeError(f"{fail} video(s) failed; {ok} succeeded. 請確認失敗影片後再進行重建。")
    if flatten:
        r.log("    Sharp frames are in each video's dir; point COLMAP at the output root.")
    else:
        r.log("    Point COLMAP at each video's sharp/ directory.")
