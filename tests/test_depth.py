"""Depth stage: the torch-free parts we can exercise without the model env.

The actual model call (load_model/infer_one/to_png8) needs torch+transformers+
PIL, which live only in the DEPTH_CONDA_ENV (not the panel's env), so they are
out of scope here. We lock down what the panel itself owns: the image
enumeration the infer script uses, and run_depth's pre-flight guards (which are
what surface a misconfigured machine as a clean FAILED job, not a stall).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pipeline.depth import _resolve_images, run_depth

# Load tools/depth_anything_infer.py by path (tools/ isn't an importable package).
_INFER = Path(__file__).resolve().parent.parent / "tools" / "depth_anything_infer.py"
_spec = importlib.util.spec_from_file_location("depth_infer", _INFER)
infer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(infer)


class _FakeRunner:
    """Minimal Runner stand-in: records log/banner calls, never spawns anything."""

    def __init__(self):
        self.lines: list[str] = []

    def banner(self, msg):
        self.lines.append(msg)

    def log(self, msg=""):
        self.lines.append(msg)

    def run(self, *a, **k):                       # should never be reached in these tests
        raise AssertionError("run() must not be called when a guard should fire first")


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


# --- list_images (the infer script's enumeration) -------------------------- #
def test_list_images_filters_and_sorts(tmp_path):
    for name in ["b.jpg", "a.png", "c.txt", "d.JPEG", "notes.md"]:
        _touch(tmp_path / name)
    got = [p.name for p in infer.list_images(tmp_path, recurse=False)]
    assert got == ["a.png", "b.jpg", "d.JPEG"]      # image exts only, sorted


def test_list_images_non_recursive_skips_subdirs(tmp_path):
    _touch(tmp_path / "top.jpg")
    _touch(tmp_path / "sub" / "deep.jpg")
    flat = [p.name for p in infer.list_images(tmp_path, recurse=False)]
    assert flat == ["top.jpg"]
    deep = sorted(p.name for p in infer.list_images(tmp_path, recurse=True))
    assert deep == ["deep.jpg", "top.jpg"]


# --- _resolve_images (workspace vs images folder) -------------------------- #
def test_resolve_images_prefers_nested_images_dir(tmp_path):
    (tmp_path / "images").mkdir()
    assert _resolve_images(tmp_path) == tmp_path / "images"


def test_resolve_images_falls_back_to_self(tmp_path):
    assert _resolve_images(tmp_path) == tmp_path


# --- run_depth pre-flight guards ------------------------------------------- #
def test_run_depth_missing_env_raises(tmp_path, monkeypatch):
    # No depth env resolvable -> a clear RuntimeError (not a silent stall).
    monkeypatch.setattr("pipeline.depth.depth_env_python", lambda: None)
    with pytest.raises(RuntimeError, match="conda env"):
        run_depth({"images": str(tmp_path), "model": "x"}, _FakeRunner())


def test_run_depth_bad_images_raises(tmp_path, monkeypatch):
    # Env present (faked) but images path is not a dir -> FileNotFoundError.
    fake_py = tmp_path / "python"
    fake_py.write_text("")
    monkeypatch.setattr("pipeline.depth.depth_env_python", lambda: fake_py)
    with pytest.raises(FileNotFoundError, match="images"):
        run_depth({"images": str(tmp_path / "nope"), "model": "x"}, _FakeRunner())
