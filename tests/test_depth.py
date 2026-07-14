"""Depth/normal stage: the torch-free parts we can exercise without the actual
LichtFeld-Studio binary. Locks down what the panel itself owns: dataset-root
resolution and run_depth's pre-flight guards (which are what surface a
misconfigured machine as a clean FAILED job, not a stall)."""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.depth import resolve_dataset, run_depth
from pipeline.runner import PipelineError


class _FakeRunner:
    """Minimal Runner stand-in: records log/banner calls, never spawns anything
    (unless run_returncode is set, simulating the child's exit code)."""

    def __init__(self, run_returncode=None):
        self.lines: list[str] = []
        self._run_returncode = run_returncode

    def banner(self, msg):
        self.lines.append(msg)

    def log(self, msg=""):
        self.lines.append(msg)

    def run(self, *a, **k):
        if self._run_returncode is None:
            raise AssertionError("run() must not be called when a guard should fire first")
        return self._run_returncode


# --- resolve_dataset (workspace vs images folder) --------------------------- #
def test_resolve_dataset_prefers_nested_images_dir(tmp_path):
    (tmp_path / "images").mkdir()
    assert resolve_dataset(tmp_path) == (tmp_path, "images")


def test_resolve_dataset_falls_back_to_parent(tmp_path):
    images_dir = tmp_path / "photos"
    images_dir.mkdir()
    assert resolve_dataset(images_dir) == (tmp_path, "photos")


# --- run_depth pre-flight guards ------------------------------------------- #
def test_run_depth_missing_binary_raises(tmp_path, monkeypatch):
    # No LichtFeld-Studio binary resolvable -> a clear RuntimeError (not a silent stall).
    monkeypatch.setattr("pipeline.depth.depth_binary_exec", lambda: None)
    with pytest.raises(RuntimeError, match="LichtFeld-Studio binary"):
        run_depth({"images": str(tmp_path), "mode": "both"}, _FakeRunner())


def test_run_depth_bad_images_raises(tmp_path, monkeypatch):
    # Binary present (faked) but images path is not a dir -> FileNotFoundError.
    fake_exe = tmp_path / "LichtFeld-Studio"
    fake_exe.write_text("")
    monkeypatch.setattr("pipeline.depth.depth_binary_exec", lambda: fake_exe)
    with pytest.raises(FileNotFoundError, match="images"):
        run_depth({"images": str(tmp_path / "nope"), "mode": "both"}, _FakeRunner())


def test_run_depth_bad_mode_raises(tmp_path, monkeypatch):
    fake_exe = tmp_path / "LichtFeld-Studio"
    fake_exe.write_text("")
    monkeypatch.setattr("pipeline.depth.depth_binary_exec", lambda: fake_exe)
    with pytest.raises(ValueError, match="mode"):
        run_depth({"images": str(tmp_path), "mode": "bogus"}, _FakeRunner())


# --- signal-crash-after-completion handling --------------------------------- #
# Known LichtFeld-Studio issue: `preprocess` can crash (killed by signal, e.g.
# SIGSEGV) in its post-processing depth-anchor cache step, AFTER every depth/
# normal PNG is already written. run_depth must swallow that (job succeeds)
# only when the outputs are verifiably complete on disk, and never swallow a
# genuine failure.
def _make_dataset_with_outputs(tmp_path, mode="both"):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"")
    (images / "b.jpg").write_bytes(b"")
    if mode in ("depth", "both"):
        (tmp_path / "depth").mkdir()
        (tmp_path / "depth" / "a.png").write_bytes(b"")
        (tmp_path / "depth" / "b.png").write_bytes(b"")
    if mode in ("normal", "both"):
        (tmp_path / "normals").mkdir()
        (tmp_path / "normals" / "a.png").write_bytes(b"")
        (tmp_path / "normals" / "b.png").write_bytes(b"")
    return tmp_path


def test_run_depth_swallows_signal_crash_when_outputs_complete(tmp_path, monkeypatch):
    fake_exe = tmp_path / "LichtFeld-Studio"
    fake_exe.write_text("")
    monkeypatch.setattr("pipeline.depth.depth_binary_exec", lambda: fake_exe)
    ds = _make_dataset_with_outputs(tmp_path, mode="both")

    runner = _FakeRunner(run_returncode=-11)
    run_depth({"images": str(ds / "images"), "mode": "both"}, runner)  # must not raise
    assert any("視為此工作成功" in line for line in runner.lines)


def test_run_depth_raises_when_signal_crash_and_outputs_incomplete(tmp_path, monkeypatch):
    fake_exe = tmp_path / "LichtFeld-Studio"
    fake_exe.write_text("")
    monkeypatch.setattr("pipeline.depth.depth_binary_exec", lambda: fake_exe)
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"")           # no matching depth/normals output written

    runner = _FakeRunner(run_returncode=-11)
    with pytest.raises(PipelineError):
        run_depth({"images": str(images), "mode": "both"}, runner)


def test_run_depth_raises_on_positive_returncode_even_if_outputs_complete(tmp_path, monkeypatch):
    # A clean `return 1` (real error path in run_preprocess) is never swallowed,
    # regardless of what's on disk — only signal kills go through the disk check.
    fake_exe = tmp_path / "LichtFeld-Studio"
    fake_exe.write_text("")
    monkeypatch.setattr("pipeline.depth.depth_binary_exec", lambda: fake_exe)
    ds = _make_dataset_with_outputs(tmp_path, mode="both")

    runner = _FakeRunner(run_returncode=1)
    with pytest.raises(PipelineError):
        run_depth({"images": str(ds / "images"), "mode": "both"}, runner)
