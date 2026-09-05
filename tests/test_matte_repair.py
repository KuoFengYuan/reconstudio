"""Repairing a bad cut-out across more than one frame.

The failure that motivated this: a 666-frame turntable capture of a glazed vase
where SAM dropped the neck and foot on *every* frame — the differently-glazed
parts read as their own object. A point prompt is what fixes that, but points
were resolved by exact relative path only, so a correction could never describe
more than the one frame it was clicked on. 666 single-frame repairs is not a
workflow, so the scope lives here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
pytest.importorskip("numpy", reason="the prompt decoder returns numpy arrays")

import sam_matte as sm  # noqa: E402

from pipeline.matte import _write_boxes  # noqa: E402
from web.routers.create import _repair_targets  # noqa: E402


# --- points honour apply=all, the way boxes always have ---------------------- #
def test_a_folder_wide_point_prompt_reaches_every_frame():
    spec = {"norm": True, "apply": "all", "points": [[0.5, 0.25, 1]]}
    coords, labels = sm.points_for_image(spec, "any/frame.jpg", 100, 200)
    assert coords.tolist() == [[50.0, 50.0]]      # normalised -> this frame's pixels
    assert labels.tolist() == [1]


def test_a_scoped_prompt_does_not_leak_onto_other_frames():
    """`only` frames carry their own entry; everything else must stay untouched,
    or a one-frame repair would silently re-cut the whole folder."""
    spec = {"norm": True, "apply": "all", "only": ["a.jpg"],
            "per_image": {"a.jpg": {"points": [[0.5, 0.5, 1]]}}}
    assert sm.points_for_image(spec, "b.jpg", 100, 200) == (None, None)
    assert sm.points_for_image(spec, "a.jpg", 100, 200)[0] is not None


def test_a_frames_own_points_beat_the_folder_wide_ones():
    spec = {"norm": True, "apply": "all", "points": [[0.1, 0.1, 1]],
            "per_image": {"a.jpg": {"points": [[0.9, 0.9, 0]]}}}
    coords, labels = sm.points_for_image(spec, "a.jpg", 100, 200)
    assert coords.tolist() == [[90.0, 180.0]] and labels.tolist() == [0]


def test_no_points_anywhere_is_still_none():
    assert sm.points_for_image({"norm": True, "apply": "all"}, "a.jpg", 8, 8) == (None, None)


# --- clicks must keep the run off the batched path --------------------------- #
class _FakeRunner:
    """Stands in for Sam2Runner: has the across-images API, so only the prompt
    shape decides whether a run is batchable."""

    def masks_for_image_batch(self, rgbs, boxes_per_image):
        raise AssertionError("a click-prompted run must not reach the batch path")


class _Args:
    image_batch = 8
    boxes = "json"


def test_a_repairs_clicks_keep_it_off_the_batched_path():
    """The bug this pins made every point-only repair a silent no-op.

    `batchable` looked for `spec["points"]`, but a repair puts its clicks in
    `per_image[rel]["points"]` and leaves `boxes` empty. So the run was judged
    batchable, took the box-only path, found no boxes, skipped the image, and
    finished "done" with processed=0 — the clicks never reached SAM at all.
    """
    spec = {"norm": True, "apply": "all", "only": ["a.jpg"],
            "per_image": {"a.jpg": {"boxes": [], "points": [[0.5, 0.5, 1]]}}}
    assert sm.has_points(spec)
    assert not sm.batchable(_Args(), _FakeRunner(), spec)


def test_a_folder_wide_click_prompt_also_stays_off_it():
    spec = {"norm": True, "apply": "all", "points": [[0.5, 0.5, 1]]}
    assert sm.has_points(spec) and not sm.batchable(_Args(), _FakeRunner(), spec)


def test_a_box_only_run_is_still_batchable():
    """The batching is a real speed win, so it must not be lost to over-caution."""
    spec = {"norm": True, "apply": "all", "boxes": [[0.1, 0.1, 0.9, 0.9]],
            "per_image": {"a.jpg": {"boxes": [[0.1, 0.1, 0.9, 0.9]]}}}
    assert not sm.has_points(spec)
    assert sm.batchable(_Args(), _FakeRunner(), spec)


@pytest.mark.parametrize("spec", [
    None, {}, {"per_image": {}}, {"per_image": {"a.jpg": []}},
    {"per_image": {"a.jpg": {"boxes": []}}}, {"points": []},
])
def test_specs_without_clicks_report_none(spec):
    assert not sm.has_points(spec)


# --- scope resolution ------------------------------------------------------- #
def _seq(d: Path, names: list[str], sub: str = "") -> Path:
    here = d / sub if sub else d
    here.mkdir(parents=True, exist_ok=True)
    for n in names:
        (here / n).write_bytes(b"\xff\xd8\xff\xd9")
    return d


def test_one_is_just_that_frame(tmp_path):
    _seq(tmp_path, ["a.jpg", "b.jpg"])
    assert _repair_targets(tmp_path, "a.jpg", "one") == ["a.jpg"]


def test_all_means_the_whole_folder(tmp_path):
    assert _repair_targets(tmp_path, "a.jpg", "all") is None


def test_a_dragged_range_is_the_span_between_its_endpoints(tmp_path):
    _seq(tmp_path, [f"IMG_{i:04d}.jpg" for i in range(10)])
    got = _repair_targets(tmp_path, "IMG_0005.jpg", "range",
                          "IMG_0003.jpg", "IMG_0006.jpg")
    assert got == ["IMG_0003.jpg", "IMG_0004.jpg", "IMG_0005.jpg", "IMG_0006.jpg"]


def test_a_range_dragged_backwards_is_the_same_span(tmp_path):
    """Anchoring on either end has to work, so the endpoints arrive unordered."""
    _seq(tmp_path, [f"IMG_{i:04d}.jpg" for i in range(6)])
    assert _repair_targets(tmp_path, "IMG_0001.jpg", "range",
                           "IMG_0004.jpg", "IMG_0001.jpg") == [
        f"IMG_{i:04d}.jpg" for i in range(1, 5)]


def test_a_range_cannot_reach_outside_the_frames_own_subfolder(tmp_path):
    """A sibling folder is a different capture. An endpoint naming one falls back
    to this frame rather than silently re-cutting the other capture."""
    _seq(tmp_path, ["IMG_0001.jpg", "IMG_0002.jpg"], sub="up")
    _seq(tmp_path, ["IMG_0003.jpg"], sub="down")
    got = _repair_targets(tmp_path, "up/IMG_0001.jpg", "range",
                          "up/IMG_0001.jpg", "down/IMG_0003.jpg")
    assert got == ["up/IMG_0001.jpg"]


def test_a_range_with_no_endpoints_is_just_this_frame(tmp_path):
    _seq(tmp_path, ["a.jpg", "b.jpg"])
    assert _repair_targets(tmp_path, "a.jpg", "range") == ["a.jpg"]


def test_a_bad_scope_is_rejected(tmp_path):
    for bad in ("near:5", "sideways", ""):
        with pytest.raises(ValueError, match="套用範圍"):
            _repair_targets(tmp_path, "a.jpg", bad)


def test_non_image_siblings_are_not_targets(tmp_path):
    _seq(tmp_path, ["a.jpg", "b.jpg"])
    (tmp_path / "matte_boxes.json").write_text("{}")
    assert _repair_targets(tmp_path, "a.jpg", "range", "a.jpg", "b.jpg") == \
        ["a.jpg", "b.jpg"]


# --- a scoped prompt must not clobber the picker's ---------------------------- #
def test_a_scoped_repair_is_written_beside_the_pickers_boxes(tmp_path):
    """It used to land on matte_boxes.json, replacing the prompt the other N-1
    frames were cut with — so re-running the folder afterwards re-cut everything
    from whatever the last repair happened to click."""
    picker = json.dumps({"norm": True, "apply": "all", "refs": {"a.jpg": [[0, 0, 1, 1]]}})
    _write_boxes(tmp_path, picker)
    repair = json.dumps({"norm": True, "apply": "all", "only": ["a.jpg"],
                         "per_image": {"a.jpg": {"points": [[0.5, 0.5, 1]]}}})
    out = _write_boxes(tmp_path, repair)

    assert out.name == "matte_repair.json"
    assert json.loads((tmp_path / "matte_boxes.json").read_text()) == json.loads(picker)


def test_a_folder_wide_prompt_still_owns_matte_boxes_json(tmp_path):
    spec = json.dumps({"norm": True, "apply": "all", "points": [[0.5, 0.5, 1]]})
    assert _write_boxes(tmp_path, spec).name == "matte_boxes.json"


def test_unparseable_json_keeps_the_old_destination(tmp_path):
    assert _write_boxes(tmp_path, "not json").name == "matte_boxes.json"
