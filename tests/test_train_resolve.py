"""Trainer source resolution accepts both binary and text COLMAP models.

LichtFeld and the GS-2M COLMAP loader read text and binary natively, so the panel
must not reject a text-only (cameras.txt) model — it used to require cameras.bin.
"""
from __future__ import annotations

import pytest

from pipeline.train import (
    _assert_pinhole,
    _build_scene,
    _model_ext,
    _resolve_dense,
)

_CAMERAS_TXT = "# comment line\n1 PINHOLE 1920 1440 1366.9 1366.9 960.0 720.0\n"
_CAMERAS_TXT_DISTORTED = "1 OPENCV 1920 1440 1 1 1 1 0 0 0 0\n"


class _R:
    def log(self, _msg):  # Runner stub: _assert_pinhole / _build_scene only log
        pass


def _make_model(root, ext: str, *, sub0: bool = True, cameras=_CAMERAS_TXT):
    """Create <root>/sparse[/0]/cameras<ext> (+ siblings) and <root>/images/."""
    sd = root / "sparse" / "0" if sub0 else root / "sparse"
    sd.mkdir(parents=True)
    if ext == ".txt":
        (sd / "cameras.txt").write_text(cameras)
        (sd / "images.txt").write_text("")
        (sd / "points3D.txt").write_text("")
    else:  # _resolve_dense / _model_ext only stat cameras.bin; content is irrelevant
        for stem in ("cameras", "images", "points3D"):
            (sd / f"{stem}.bin").write_bytes(b"")
    (root / "images").mkdir()
    return sd


def test_model_ext_prefers_bin(tmp_path):
    sd = _make_model(tmp_path, ".txt")
    (sd / "cameras.bin").write_bytes(b"")
    assert _model_ext(sd) == ".bin"


def test_model_ext_txt_and_none(tmp_path):
    assert _model_ext(_make_model(tmp_path / "a", ".txt")) == ".txt"
    assert _model_ext(tmp_path / "missing") is None


def test_resolve_dense_accepts_txt_sparse0(tmp_path):
    sd = _make_model(tmp_path, ".txt", sub0=True)
    sparse, images = _resolve_dense(tmp_path)
    assert sparse == sd and images == tmp_path / "images"


def test_resolve_dense_accepts_bin_flat(tmp_path):
    sd = _make_model(tmp_path, ".bin", sub0=False)
    sparse, images = _resolve_dense(tmp_path)
    assert sparse == sd and images == tmp_path / "images"


def test_resolve_dense_missing_raises(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError):
        _resolve_dense(tmp_path)


def test_assert_pinhole_reads_txt(tmp_path):
    sd = _make_model(tmp_path, ".txt")
    _assert_pinhole(sd, _R())  # PINHOLE → no raise


def test_assert_pinhole_rejects_distorted_txt(tmp_path):
    sd = _make_model(tmp_path, ".txt", cameras=_CAMERAS_TXT_DISTORTED)
    with pytest.raises(ValueError):
        _assert_pinhole(sd, _R())


def test_build_scene_links_txt(tmp_path):
    sd = _make_model(tmp_path, ".txt")
    scene = tmp_path / "scene"
    _build_scene(scene, sd, tmp_path / "images", False, _R())
    linked = sorted(p.name for p in (scene / "sparse" / "0").iterdir())
    assert linked == ["cameras.txt", "images.txt", "points3D.txt"]
    assert (scene / "images").is_symlink()


# --- --export / --output-name (LichtFeld) ---------------------------------

def test_export_stem_prefers_the_project_name_over_a_generic_leaf(tmp_path):
    """The panel's own convention is `<project>/model`, so `model` names nothing.
    A folder of `splat_30000.sog` is unusable once the file leaves the panel."""
    from pipeline.train import _export_stem
    assert _export_stem(tmp_path / "20260902_131140_2240" / "model") == "20260902_131140_2240"
    assert _export_stem(tmp_path / "20260902_131140_2240" / "Output") == "20260902_131140_2240"
    # a meaningful leaf is kept as-is
    assert _export_stem(tmp_path / "proj" / "run_a") == "run_a"


def test_has_flag_does_not_match_a_longer_flag():
    """`--export` must not be seen inside `--exclude-export`, or a job that only
    excludes frozen splats would silently get an --output-name it never asked for."""
    from pipeline.train import _has_flag
    assert _has_flag(["--exclude-export"], "--export") is False
    assert _has_flag(["--export", "sog"], "--export") is True
    assert _has_flag(["--export=sog"], "--export") is True


def test_trained_output_accepts_an_exported_sog(tmp_path):
    """LichtFeld sometimes segfaults during CUDA teardown AFTER writing everything.
    With the panel default (sog only, no PLY) a PLY-only check would call that a
    failed job and throw away a finished model."""
    from pipeline.train import _trained_output
    assert _trained_output(tmp_path) is None
    (tmp_path / "project.licht").write_bytes(b"x")      # not a model file on its own
    assert _trained_output(tmp_path) is None
    (tmp_path / "proj.sog").write_bytes(b"")            # 0 bytes = a failed write
    assert _trained_output(tmp_path) is None
    (tmp_path / "proj.sog").write_bytes(b"splat")
    assert _trained_output(tmp_path) == tmp_path / "proj.sog"


# --- texture bake: staying inside the machine's memory ---------------------- #
# The bake is the one stage that can take the whole box down (OpenMP over every
# core, each worker holding a decoded photo), so the plan it computes is pinned
# here rather than trusted to a comment.

def test_texture_plan_uses_every_core_when_memory_is_plentiful():
    from pipeline.train import _texture_plan
    threads, side = _texture_plan(px=3_000_000, cores=72, budget=400 * 10**9)
    assert threads == 72
    assert side is None          # native resolution — the point of the pipeline


def test_texture_plan_caps_threads_before_touching_resolution():
    from pipeline.train import _texture_plan
    # 24 MP photos, 8 GB budget: ~28 workers fit, so shrink the pool, not the photos.
    threads, side = _texture_plan(px=24_000_000, cores=72, budget=8 * 10**9)
    assert 1 < threads < 72
    assert side is None


def test_texture_plan_downscales_only_when_even_a_few_workers_dont_fit():
    from pipeline.train import _TEX_MIN_SIDE, _TEX_MIN_THREADS, _texture_plan
    threads, side = _texture_plan(px=100_000_000, cores=72, budget=2 * 10**9)
    assert threads == _TEX_MIN_THREADS
    assert side is not None and side >= _TEX_MIN_SIDE


def test_texture_plan_never_shrinks_below_the_floor():
    from pipeline.train import _TEX_MIN_SIDE, _texture_plan
    _, side = _texture_plan(px=100_000_000, cores=8, budget=10 * 10**6)
    assert side == _TEX_MIN_SIDE


def test_texture_plan_survives_an_unreadable_meminfo():
    from pipeline.train import _texture_plan
    # budget 0 = we could not measure; fall back to a bounded pool, no downscale.
    threads, side = _texture_plan(px=0, cores=72, budget=0)
    assert (threads, side) == (16, None)


def _atlases(tmp_path, n, side):
    from PIL import Image
    files = []
    for i in range(n):
        f = tmp_path / f"a{i}_{side}.png"
        if not f.exists():
            Image.new("RGB", (side, side)).save(f)
        files.append(f)
    return files


# --- _atlas_plan: the merged atlas size is decided, not asked ---------------- #

def test_atlas_plan_picks_the_smallest_square_that_fits(tmp_path):
    from pipeline.train import _atlas_plan
    # 3 × 1024² = 3.1 Mpx (+ packing slack): a 2048² canvas holds it, and a 16384
    # one would be 99% black.
    assert _atlas_plan(_atlases(tmp_path, 3, 1024), budget=512 * 10**9) == (2048, 1)
    # 8 × 2048² = 34 Mpx -> needs 8192² (4096² is 17 Mpx, too small).
    assert _atlas_plan(_atlases(tmp_path, 8, 2048), budget=512 * 10**9) == (8192, 1)


def test_atlas_plan_never_goes_past_the_hardware_limit(tmp_path):
    from pipeline.train import _atlas_plan
    # 24 × 4096² = 402 Mpx. 16384² is the ceiling, so the content shrinks instead —
    # the alternative (a canvas that grows past 16384) can't be bound as a texture.
    px, down = _atlas_plan(_atlases(tmp_path, 24, 4096), budget=512 * 10**9)
    assert px == 16384
    assert down == 2


def test_atlas_plan_respects_a_memory_budget(tmp_path):
    from pipeline.train import _atlas_plan
    # A 16384² RGB canvas needs ~4.8 GB with the sources and the encoder; under a
    # 200 MB budget only the small canvases are affordable.
    px, down = _atlas_plan(_atlases(tmp_path, 24, 4096), budget=200 * 10**6)
    assert px <= 4096
    assert down > 1


def test_atlas_plan_honours_an_explicit_size(tmp_path):
    from pipeline.train import _atlas_plan
    # A caller that pins the size gets it, with the shrink recomputed to fit.
    assert _atlas_plan(_atlases(tmp_path, 3, 1024), budget=512 * 10**9, user_px=8192) == (8192, 1)
    px, down = _atlas_plan(_atlases(tmp_path, 24, 4096), budget=512 * 10**9, user_px=4096)
    assert (px, down) == (4096, 8)


def test_atlas_plan_survives_unreadable_atlases(tmp_path):
    from pipeline.train import _atlas_plan
    missing = [tmp_path / "gone.png"]
    px, down = _atlas_plan(missing, budget=512 * 10**9)
    assert down == 1 and px in (2048, 4096, 8192, 16384)
