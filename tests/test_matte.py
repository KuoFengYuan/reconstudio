"""The torch-free half of the 去背 stage: run_matte's guards and its argv.

Mirrors tests/test_depth.py. What matters here is the argv: a knob that never
reaches tools/sam_matte.py fails silently — the run succeeds with the tool's
default and only the pixels are wrong — so every mapped flag is asserted both
present when set and ABSENT when blank (an absence-only test would pass against
a feature that was never wired up at all).
"""
from __future__ import annotations

import json

import pytest

from pipeline.matte import matte_ready, resolve_matte_dataset, run_matte
from pipeline.runner import PipelineError

BOXES = json.dumps({"ref": "a.jpg", "norm": True, "boxes": [[0.1, 0.1, 0.5, 0.6]]})


class _FakeRunner:
    """Records the argv instead of spawning anything."""

    def __init__(self, returncode: int = 0):
        self.lines: list[str] = []
        self.argv: list[str] = []
        self.env: dict = {}
        self._rc = returncode

    def banner(self, msg):
        self.lines.append(msg)

    def log(self, msg=""):
        self.lines.append(msg)

    def run(self, argv, **kw):
        self.argv = list(argv)
        self.env = kw.get("env") or {}
        return self._rc


@pytest.fixture()
def dataset(tmp_path):
    (tmp_path / "images").mkdir()
    return tmp_path


@pytest.fixture()
def fake_env(tmp_path, monkeypatch):
    py = tmp_path / "python"
    py.write_text("")
    monkeypatch.setattr("pipeline.matte.sam_python", lambda: py)
    # Default: a complete env. The probe would otherwise try to execute the empty
    # file above, and every argv assertion below would depend on that failing the
    # right way rather than on run_matte's own logic.
    monkeypatch.setattr("pipeline.matte._preflight", lambda *a, **k: None)
    return py


def _run(dataset, params=None, rc=0):
    r = _FakeRunner(rc)
    base = {"images": str(dataset), "boxes": "json", "boxes_json": BOXES}
    run_matte({**base, **(params or {})}, r)
    return r


# --- guards ----------------------------------------------------------------- #
def test_missing_env_raises_with_setup_instructions(dataset, monkeypatch):
    monkeypatch.setattr("pipeline.matte.sam_python", lambda: None)
    with pytest.raises(RuntimeError, match="sam conda env"):
        run_matte({"images": str(dataset)}, _FakeRunner())


def test_bad_images_dir_raises(tmp_path, fake_env):
    with pytest.raises(FileNotFoundError, match="images"):
        run_matte({"images": str(tmp_path / "nope")}, _FakeRunner())


def test_unknown_engine_raises(dataset, fake_env):
    with pytest.raises(ValueError, match="engine"):
        _run(dataset, {"matte_engine": "sam9"})


def test_unknown_box_source_raises(dataset, fake_env):
    with pytest.raises(ValueError, match="boxes"):
        _run(dataset, {"boxes": "telepathy"})


def test_box_modes_require_drawn_boxes(dataset, fake_env):
    with pytest.raises(ValueError, match="匡選"):
        _run(dataset, {"boxes": "track", "boxes_json": ""})


def test_text_mode_requires_a_phrase(dataset, fake_env):
    with pytest.raises(ValueError, match="文字提示"):
        _run(dataset, {"boxes": "text", "boxes_json": "", "text": ""})


def test_nonzero_exit_is_a_pipeline_error(dataset, fake_env):
    with pytest.raises(PipelineError, match="code 3"):
        _run(dataset, rc=3)


def test_matte_ready_is_false_without_the_env(monkeypatch):
    monkeypatch.setattr("pipeline.matte.sam_python", lambda: None)
    assert matte_ready() is False


# --- argv ------------------------------------------------------------------- #
def test_boxes_are_written_next_to_the_dataset_and_passed_by_path(dataset, fake_env):
    r = _run(dataset)
    written = dataset / "matte_boxes.json"
    assert written.is_file()
    assert json.loads(written.read_text())["boxes"] == [[0.1, 0.1, 0.5, 0.6]]
    assert ["--boxes-json", str(written)] == r.argv[r.argv.index("--boxes-json"):][:2]


def test_defaults_reach_the_script(dataset, fake_env):
    r = _run(dataset)
    assert "--engine" in r.argv and r.argv[r.argv.index("--engine") + 1] == "sam2"
    assert r.argv[r.argv.index("--images") + 1] == "images"
    assert r.argv[r.argv.index("--outputs") + 1] == "cutout,masks"


def test_blank_optional_knobs_are_omitted_entirely(dataset, fake_env):
    r = _run(dataset, {"erode": "", "feather": "", "min_area": "", "box_chunk": "",
                       "matte_model": "", "on_empty": ""})
    for flag in ("--erode", "--feather", "--min-area", "--box-chunk", "--model",
                 "--on-empty"):
        assert flag not in r.argv


def test_set_optional_knobs_are_passed(dataset, fake_env):
    r = _run(dataset, {"erode": "3", "feather": "5", "min_area": "0.01",
                       "box_chunk": "8", "matte_model": "facebook/sam2.1-hiera-small",
                       "on_empty": "skip"})
    for flag, value in (("--erode", "3"), ("--feather", "5"), ("--min-area", "0.01"),
                        ("--box-chunk", "8"),
                        ("--model", "facebook/sam2.1-hiera-small"),
                        ("--on-empty", "skip")):
        assert r.argv[r.argv.index(flag) + 1] == value


def test_unticked_switches_are_absent(dataset, fake_env):
    r = _run(dataset)
    for flag in ("--row-filter", "--largest-only", "--soft-masks", "--no-bleed",
                 "--overwrite"):
        assert flag not in r.argv


def test_ticked_switches_are_present(dataset, fake_env):
    r = _run(dataset, {"row_filter": True, "largest_only": True, "soft_masks": True,
                       "no_bleed": True, "overwrite": True})
    for flag in ("--row-filter", "--largest-only", "--soft-masks", "--no-bleed",
                 "--overwrite"):
        assert flag in r.argv


def test_text_mode_passes_the_phrase_and_no_boxes_file(dataset, fake_env):
    r = _run(dataset, {"boxes": "text", "boxes_json": "", "text": "parrot"})
    assert r.argv[r.argv.index("--text") + 1] == "parrot"
    assert "--boxes-json" not in r.argv


def test_gpu_becomes_cuda_visible_devices(dataset, fake_env):
    assert _run(dataset, {"gpu": "1"}).env == {"CUDA_VISIBLE_DEVICES": "1"}


def test_blank_gpu_leaves_the_env_alone(dataset, fake_env):
    assert _run(dataset, {"gpu": ""}).env == {}


# --- resize ------------------------------------------------------------------ #
def test_resize_keep_runs_at_the_original_dataset_root(dataset, fake_env, monkeypatch):
    called = []
    monkeypatch.setattr("pipeline.matte.resize_to_fullhd",
                        lambda *a, **k: called.append((a, k)) or "should not be used")
    r = _run(dataset, {"resize": "keep"})
    assert called == []
    assert str(dataset) in r.argv


def test_resize_switches_the_run_to_the_resized_copy(dataset, fake_env, monkeypatch):
    """A resize choice must move the WHOLE run (dataset_root, --images, and the
    boxes-json path) to images_<cap>/, not just be logged — that folder is what
    tools/sam_matte.py's output_path_for() bases every no_bg/ write on, and it's
    also what web/routers/create.py's resize_target() promises the job card."""
    calls = []

    def fake_resize(img_root, lines, ws, force, r, **kw):
        calls.append({"img_root": img_root, "ws": ws, "max_size": kw.get("max_size")})
        out = ws / f"images_{kw.get('max_size')}"
        out.mkdir(parents=True, exist_ok=True)
        return str(out)

    monkeypatch.setattr("pipeline.matte.resize_to_fullhd", fake_resize)
    r = _run(dataset, {"resize": "1920"})

    assert calls == [{"img_root": str(dataset / "images"), "ws": dataset, "max_size": "1920"}]
    resized_root = str(dataset / "images_1920")
    assert resized_root in r.argv
    assert r.argv[r.argv.index("--images") + 1] == ""          # resized copy IS the root
    assert (dataset / "images_1920" / "matte_boxes.json").is_file()


def test_unknown_resize_raises(dataset, fake_env):
    with pytest.raises(ValueError, match="resize"):
        _run(dataset, {"resize": "1080"})


# --- dataset root ----------------------------------------------------------- #
def test_workspace_layout_keeps_outputs_beside_images(tmp_path):
    (tmp_path / "images").mkdir()
    assert resolve_matte_dataset(tmp_path) == (tmp_path, "images")


def test_loose_photo_folder_writes_inside_itself_not_the_parent(tmp_path):
    """The parent of a photo folder is usually a folder of OTHER datasets; two
    runs sharing one cutout/ there would overwrite each other by filename."""
    photos = tmp_path / "20260902_shoot"
    photos.mkdir()
    assert resolve_matte_dataset(photos) == (photos, "")


# --- env preflight ---------------------------------------------------------- #
def _probe(monkeypatch, stdout: str, returncode: int = 0):
    class _Out:
        pass
    out = _Out()
    out.stdout, out.stderr, out.returncode = stdout, "", returncode
    monkeypatch.setattr("pipeline.matte.subprocess.run", lambda *a, **k: out)


def test_preflight_names_every_missing_package(dataset, tmp_path, monkeypatch):
    py = tmp_path / "python"
    py.write_text("")
    monkeypatch.setattr("pipeline.matte.sam_python", lambda: py)
    _probe(monkeypatch, "numpy torch sam2")
    with pytest.raises(RuntimeError) as exc:
        _run(dataset)
    for name in ("numpy", "torch", "sam2"):
        assert name in str(exc.value)


def test_preflight_checks_the_package_the_chosen_engine_needs(dataset, tmp_path,
                                                              monkeypatch):
    py = tmp_path / "python"
    py.write_text("")
    monkeypatch.setattr("pipeline.matte.sam_python", lambda: py)
    seen = {}

    class _Out:
        stdout, stderr, returncode = "", "", 0

    def fake_run(argv, **kw):
        seen["code"] = argv[-1]
        return _Out()

    monkeypatch.setattr("pipeline.matte.subprocess.run", fake_run)
    _run(dataset, {"matte_engine": "sam3"})
    assert "transformers" in seen["code"] and "sam2" not in seen["code"]


def test_preflight_passes_a_complete_env(dataset, tmp_path, monkeypatch):
    py = tmp_path / "python"
    py.write_text("")
    monkeypatch.setattr("pipeline.matte.sam_python", lambda: py)
    _probe(monkeypatch, "")
    assert _run(dataset).argv[0] == str(py)     # no exception, the run proceeds


# --- job meta --------------------------------------------------------------- #
def test_meta_does_not_shadow_the_parsed_box_count():
    """`_parse_matte` accumulates the per-image box total into meta["boxes"]. The
    創建 handler must not also put the prompt-mode string there — it did once, and
    the job chip rendered "共 track 個框"."""
    from jobs import Job, _parse_matte
    job = Job(id="t", kind="matte", title="x", subtitle="y",
              meta={"images": "/x", "box_source": "track"})
    _parse_matte(job, "[1/2] a.jpg  boxes=3 cover=10.0%")
    assert job.meta["boxes"] == 3
    assert job.meta["box_source"] == "track"


def test_outputs_are_announced_under_no_bg(dataset, fake_env):
    """Both folders live under no_bg/ so a plain photo directory does not get two
    loose output dirs dropped in beside the originals."""
    r = _run(dataset)
    joined = " ".join(r.lines)
    assert str(dataset / "no_bg" / "cutout") in joined
    assert str(dataset / "no_bg" / "masks") in joined


# --- repair: point prompts, scoped to one frame ------------------------------ #
REPAIR = json.dumps({
    "norm": True, "only": ["a.jpg"],
    "per_image": {"a.jpg": {"boxes": [[0.1, 0.1, 0.9, 0.9]],
                            "points": [[0.5, 0.5, 1], [0.02, 0.02, 0]]}},
})


def test_repair_json_reaches_the_script_unchanged(dataset, fake_env):
    """The whole repair prompt travels in the boxes file — no new argv — so the
    only thing to assert here is that it lands on disk intact."""
    r = _run(dataset, {"boxes": "json", "boxes_json": REPAIR, "overwrite": True})
    written = json.loads((dataset / "matte_boxes.json").read_text())
    assert written["only"] == ["a.jpg"]
    assert written["per_image"]["a.jpg"]["points"] == [[0.5, 0.5, 1], [0.02, 0.02, 0]]
    assert "--overwrite" in r.argv
