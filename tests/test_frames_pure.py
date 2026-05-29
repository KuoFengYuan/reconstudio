"""Pure helpers behind the frame-extraction pipeline.

These functions decide WHICH frames survive and WHERE they land before any
ffmpeg ever runs, so getting them wrong silently corrupts every reconstruction:
- _percentile_cut picks the blur cutoff that defines "sharp enough" (lower blur
  score == sharper, since _process_video keeps frames with score <= cutoff).
- _BLUR_RE scrapes the per-frame blur score out of ffmpeg's verbose log.
- _compute_out / _expand_inputs mirror the shell script's output-dir layout so
  COLMAP finds frames in a predictable <group>/frames_<video> tree.
- _default_workers throttles parallelism so CPU decoding doesn't thrash.

We test the real, observable behavior (no ffmpeg, no GPU, no network).
"""
import os
from pathlib import Path

import pytest

from pipeline.frames import (
    _BLUR_RE,
    _compute_out,
    _default_workers,
    _expand_inputs,
    _percentile_cut,
)


# --------------------------------------------------------------------------
# _percentile_cut: cutoff = k-th smallest blur score, k = floor(n*keep_pct/100)
# clamped to [1, n]. Sharp == low blur, so "keep sharpest keep_pct%" means keep
# the keep_pct% lowest scores; the cutoff returned is the largest kept score.
# --------------------------------------------------------------------------
def test_percentile_cut_keep_100_returns_max_score():
    # keep everything -> cutoff is the highest (blurriest) score so nothing drops.
    scores = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile_cut(scores, 100) == 50.0


def test_percentile_cut_mid_keeps_lowest_scoring_fraction():
    # keep_pct=60 of 5 frames: k = int(3.0) = 3 -> 3rd smallest score == 30.0.
    # Frames with blur <= 30.0 are exactly the 3 sharpest, i.e. 60%.
    scores = [50.0, 10.0, 40.0, 20.0, 30.0]
    cut = _percentile_cut(scores, 60)
    assert cut == 30.0
    assert sum(1 for s in scores if s <= cut) == 3


def test_percentile_cut_half_uses_floor_of_count():
    # keep_pct=50 of 5 -> k = int(2.5) = 2 -> 2nd smallest == 20.0.
    scores = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile_cut(scores, 50) == 20.0


def test_percentile_cut_floor_to_at_least_one():
    # keep_pct=0 would give k=0, but it is clamped up to 1 -> smallest score kept.
    scores = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile_cut(scores, 0) == 10.0


def test_percentile_cut_single_score():
    assert _percentile_cut([7.5], 90) == 7.5


# --------------------------------------------------------------------------
# _BLUR_RE
# --------------------------------------------------------------------------
def test_blur_re_extracts_float_from_log_line():
    m = _BLUR_RE.search("[Parsed_blurdetect ...] blur: 123.45 something")
    assert m is not None
    assert m.group(1) == "123.45"
    assert float(m.group(1)) == 123.45


def test_blur_re_findall_collects_every_score():
    log = "blur: 1.0\nnoise\nblur: 22.75\nblur: 0.50\n"
    assert _BLUR_RE.findall(log) == ["1.0", "22.75", "0.50"]


def test_blur_re_requires_decimal_point():
    # An integer-only "blur: 100" has no decimal point and must NOT match,
    # mirroring the regex [0-9]+\.[0-9]+.
    assert _BLUR_RE.search("blur: 100") is None


def test_blur_re_does_not_match_unrelated_line():
    assert _BLUR_RE.search("frame= 12 fps=24 q=2.0 size=1kB") is None


# --------------------------------------------------------------------------
# _compute_out
# --------------------------------------------------------------------------
def test_compute_out_no_outdir_uses_video_parent():
    # No outdir: frames_<stem> sits next to the source video.
    assert _compute_out("/x/y/clip.MOV", None, None) == str(
        Path("/x/y") / "frames_clip"
    )


def test_compute_out_outdir_without_base_flattens_to_outdir():
    assert _compute_out("/x/y/clip.mp4", None, "/out") == str(
        Path("/out") / "frames_clip"
    )


def test_compute_out_base_top_level_video_has_no_subdir(tmp_path):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"")
    base = os.path.realpath(str(tmp_path))
    assert _compute_out(str(v), base, "/out") == str(Path("/out") / "frames_clip")


def test_compute_out_base_nested_video_preserves_relative_dir(tmp_path):
    sub = tmp_path / "groupA"
    sub.mkdir()
    v = sub / "c2.mp4"
    v.write_bytes(b"")
    base = os.path.realpath(str(tmp_path))
    assert _compute_out(str(v), base, "/out") == str(
        Path("/out") / "groupA" / "frames_c2"
    )


# --------------------------------------------------------------------------
# _expand_inputs
# --------------------------------------------------------------------------
def test_expand_inputs_single_file_pairs_with_computed_out(tmp_path):
    v = tmp_path / "solo.MOV"
    v.write_bytes(b"")
    pairs = _expand_inputs([str(v)], "/out")
    # A file input is computed with base=None -> flattened under outdir.
    assert pairs == [(str(v), str(Path("/out") / "frames_solo"))]


def test_expand_inputs_directory_recurses_sorted_and_filters_extensions(tmp_path):
    # Build a tree with video and non-video files plus a nested dir.
    (tmp_path / "b.mp4").write_bytes(b"")
    (tmp_path / "a.MOV").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    nested = tmp_path / "nest"
    nested.mkdir()
    (nested / "c.mkv").write_bytes(b"")
    (nested / "ignore.jpg").write_bytes(b"")

    pairs = _expand_inputs([str(tmp_path)], "/out")

    videos = [v for v, _ in pairs]
    # Only the recognized video extensions, sorted by full path string.
    expected_videos = sorted(
        [
            str(tmp_path / "a.MOV"),
            str(tmp_path / "b.mp4"),
            str(nested / "c.mkv"),
        ]
    )
    assert videos == expected_videos

    # Each out dir mirrors the directory layout relative to base.
    out_map = dict(pairs)
    assert out_map[str(tmp_path / "a.MOV")] == str(Path("/out") / "frames_a")
    assert out_map[str(tmp_path / "b.mp4")] == str(Path("/out") / "frames_b")
    assert out_map[str(nested / "c.mkv")] == str(
        Path("/out") / "nest" / "frames_c"
    )


def test_expand_inputs_missing_path_raises(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(FileNotFoundError):
        _expand_inputs([str(missing)], "/out")


def test_expand_inputs_empty_directory_yields_no_pairs(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _expand_inputs([str(empty)], "/out") == []


# --------------------------------------------------------------------------
# _default_workers: per = 4 (GPU) or max(1, cpu//6) (CPU); result clamped to
# [1, min(per, nvideos, 8)]. Derive cpu from the running machine to stay portable.
# --------------------------------------------------------------------------
def test_default_workers_gpu_caps_at_four_then_at_video_count():
    # GPU decode frees the CPU -> up to 4 parallel, but never more than nvideos.
    assert _default_workers(10, True) == 4
    assert _default_workers(2, True) == 2
    assert _default_workers(1, True) == 1


def test_default_workers_gpu_hard_caps_at_eight():
    # per=4 < 8, so the global cap of 8 never lets it exceed 4 on the GPU path.
    assert _default_workers(100, True) == 4


def test_default_workers_cpu_uses_cpu_over_six_formula():
    cpu = os.cpu_count() or 4
    per = max(1, cpu // 6)
    # Plenty of videos: limited by the per-CPU formula (and the 8 cap).
    assert _default_workers(50, False) == min(per, 8)


def test_default_workers_never_below_one_and_not_above_videos():
    # Even with 0 videos the result is clamped to a floor of 1.
    assert _default_workers(0, False) == 1
    assert _default_workers(0, True) == 1
