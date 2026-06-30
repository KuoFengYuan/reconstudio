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
