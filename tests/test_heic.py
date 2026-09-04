"""pipeline.heic: HEIC/HEIF -> JPEG conversion.

pillow-heif is optional (kept out of [project.dependencies], see the comment
next to it in pyproject.toml), so this whole module skips when it isn't
installed -- same pattern as numpy in test_matte_encode.
"""
from __future__ import annotations

import pytest

pillow_heif = pytest.importorskip("pillow_heif")

from PIL import Image  # noqa: E402

from pipeline.heic import convert_tree  # noqa: E402


def _write_heic(path, size=(8, 6), color=(200, 50, 50)):
    pillow_heif.register_heif_opener()
    Image.new("RGB", size, color).save(path, "HEIF")


def test_convert_tree_writes_sibling_jpeg(tmp_path):
    src = tmp_path / "IMG_0001.heic"
    _write_heic(src)

    n = convert_tree(tmp_path)

    assert n == 1
    out = tmp_path / "IMG_0001.jpg"
    assert out.is_file()
    with Image.open(out) as im:
        assert im.size == (8, 6)


def test_convert_tree_is_idempotent(tmp_path):
    src = tmp_path / "a.heic"
    _write_heic(src)
    convert_tree(tmp_path)
    stamp = (tmp_path / "a.jpg").stat().st_mtime_ns

    n = convert_tree(tmp_path)

    assert n == 0
    assert (tmp_path / "a.jpg").stat().st_mtime_ns == stamp


def test_convert_tree_skips_output_dirs(tmp_path):
    out_dir = tmp_path / "no_bg" / "cutout"
    out_dir.mkdir(parents=True)
    _write_heic(out_dir / "leftover.heic")

    n = convert_tree(tmp_path)

    assert n == 0
    assert not (out_dir / "leftover.jpg").exists()


def test_convert_tree_no_heic_files(tmp_path):
    (tmp_path / "photo.jpg").write_bytes(b"not a real jpeg but irrelevant here")

    assert convert_tree(tmp_path) == 0
