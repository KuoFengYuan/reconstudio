"""Job → filesystem resolution, traversal containment, and edited-model derivation.

These helpers translate an opaque "job" into the actual sparse/dense/trained paths
the viz + create routers serve. Getting them wrong means either a 404 where data
exists, or — worse — serving/writing OUTSIDE the workspace. So the validators
(does images.bin + points3D.bin really exist? is sub confined to <ws>/cleaned?),
the highest-iteration .ply selection across two trainer layouts, and the
non-destructive sibling derivation (never touch the source) all deserve locking down.

The functions are duck-typed on `job`: anything with .meta and .params dicts works,
so we fake it with types.SimpleNamespace.
"""
import os
import types

import pytest

from web.services import models as M


def _job(meta=None, params=None):
    return types.SimpleNamespace(meta=meta, params=params)


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


# ---------------------------------------------------------------------------
# model_dir
# ---------------------------------------------------------------------------

def test_model_dir_none_without_workspace():
    # meta with no "workspace" key -> nothing to resolve.
    assert M.model_dir(_job(meta={})) is None
    # meta is None entirely -> the `job.meta or {}` guard kicks in.
    assert M.model_dir(_job(meta=None)) is None


def test_model_dir_none_when_only_one_bin_present(tmp_path):
    ws = tmp_path / "ws"
    sparse0 = ws / "sparse" / "0"
    _touch(sparse0 / "images.bin")  # points3D.bin deliberately missing
    assert M.model_dir(_job(meta={"workspace": str(ws)})) is None


def test_model_dir_returns_sparse0_when_both_bins_present(tmp_path):
    ws = tmp_path / "ws"
    sparse0 = ws / "sparse" / "0"
    _touch(sparse0 / "images.bin")
    _touch(sparse0 / "points3D.bin")
    got = M.model_dir(_job(meta={"workspace": str(ws)}))
    assert got == sparse0


# ---------------------------------------------------------------------------
# dense_dir
# ---------------------------------------------------------------------------

def test_dense_dir_none_without_workspace():
    assert M.dense_dir(_job(meta={}, params={})) is None


def test_dense_dir_uses_dataset_and_mapper_names(tmp_path):
    ws = tmp_path / "ws"
    dataset, mapper = "myset", "incremental"
    d = ws / f"{dataset}_{mapper}_mapper"
    _touch(d / "sparse" / "cameras.bin")
    (d / "images").mkdir(parents=True, exist_ok=True)
    job = _job(meta={"workspace": str(ws)}, params={"dataset_name": dataset, "mapper": mapper})
    assert M.dense_dir(job) == d


def test_dense_dir_defaults_dataset_and_mapper(tmp_path):
    # Missing params fall back to training_dataset + global.
    ws = tmp_path / "ws"
    d = ws / "training_dataset_global_mapper"
    _touch(d / "sparse" / "cameras.bin")
    (d / "images").mkdir(parents=True, exist_ok=True)
    job = _job(meta={"workspace": str(ws)}, params={})
    assert M.dense_dir(job) == d


def test_dense_dir_none_when_images_dir_missing(tmp_path):
    ws = tmp_path / "ws"
    d = ws / "training_dataset_global_mapper"
    _touch(d / "sparse" / "cameras.bin")  # cameras.bin present but no images/ dir
    job = _job(meta={"workspace": str(ws)}, params={})
    assert M.dense_dir(job) is None


def test_dense_dir_none_when_cameras_is_dir_not_file(tmp_path):
    # is_file() must fail if cameras.bin is itself a directory.
    ws = tmp_path / "ws"
    d = ws / "training_dataset_global_mapper"
    (d / "sparse" / "cameras.bin").mkdir(parents=True, exist_ok=True)
    (d / "images").mkdir(parents=True, exist_ok=True)
    job = _job(meta={"workspace": str(ws)}, params={})
    assert M.dense_dir(job) is None


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------

def test_resolve_model_delegates_to_model_dir_when_sub_none(tmp_path):
    ws = tmp_path / "ws"
    sparse0 = ws / "sparse" / "0"
    _touch(sparse0 / "images.bin")
    _touch(sparse0 / "points3D.bin")
    job = _job(meta={"workspace": str(ws)})
    assert M.resolve_model(job, None) == sparse0
    # And empty-string sub is falsy too, so it also delegates.
    assert M.resolve_model(job, "") == sparse0


def test_resolve_model_valid_sub_under_cleaned(tmp_path):
    ws = tmp_path / "ws"
    sub = "cleaned/20260522_103000/sparse/0"
    cand = ws / sub
    _touch(cand / "images.bin")
    _touch(cand / "points3D.bin")
    got = M.resolve_model(_job(meta={"workspace": str(ws)}), sub)
    assert got == cand.resolve()


def test_resolve_model_traversal_escape_is_blocked(tmp_path):
    # The classic attack: ../../etc resolves outside <ws>/cleaned -> refused,
    # even if the target happens to contain the expected bins.
    ws = tmp_path / "ws"
    (ws / "cleaned").mkdir(parents=True, exist_ok=True)
    evil = tmp_path / "etc"
    _touch(evil / "images.bin")
    _touch(evil / "points3D.bin")
    job = _job(meta={"workspace": str(ws)})
    assert M.resolve_model(job, "../etc") is None
    assert M.resolve_model(job, "../../etc") is None
    assert M.resolve_model(job, "cleaned/../../etc") is None


def test_resolve_model_sub_outside_cleaned_but_in_ws_is_blocked(tmp_path):
    # sub that stays inside the workspace but NOT under cleaned/ is still refused:
    # confinement is to <ws>/cleaned specifically, not just the workspace.
    ws = tmp_path / "ws"
    (ws / "cleaned").mkdir(parents=True, exist_ok=True)
    other = ws / "sparse" / "0"
    _touch(other / "images.bin")
    _touch(other / "points3D.bin")
    assert M.resolve_model(_job(meta={"workspace": str(ws)}), "sparse/0") is None


def test_resolve_model_sub_under_cleaned_missing_bins_is_none(tmp_path):
    # Inside cleaned/ but the bins aren't there -> None (validator, not guard).
    ws = tmp_path / "ws"
    cand = ws / "cleaned" / "ts" / "sparse" / "0"
    cand.mkdir(parents=True, exist_ok=True)  # no bins
    assert M.resolve_model(_job(meta={"workspace": str(ws)}), "cleaned/ts/sparse/0") is None


def test_resolve_model_with_sub_but_no_workspace_is_none():
    assert M.resolve_model(_job(meta={}), "cleaned/ts/sparse/0") is None


# ---------------------------------------------------------------------------
# trained_ply
# ---------------------------------------------------------------------------

def test_trained_ply_none_when_no_candidates(tmp_path):
    assert M.trained_ply(tmp_path) is None


def test_trained_ply_picks_highest_iteration_gs2m_layout(tmp_path):
    for n in (7, 30000, 100):
        _touch(tmp_path / "point_cloud" / f"iteration_{n}" / "point_cloud.ply")
    got = M.trained_ply(tmp_path)
    assert got == tmp_path / "point_cloud" / "iteration_30000" / "point_cloud.ply"


def test_trained_ply_picks_highest_iteration_lichtfeld_layout(tmp_path):
    for n in (5, 99, 42):
        _touch(tmp_path / f"splat_{n}.ply")
    got = M.trained_ply(tmp_path)
    assert got == tmp_path / "splat_99.ply"


def test_trained_ply_max_across_both_layouts(tmp_path):
    # Both layouts present together; the globally-highest N wins regardless of layout.
    _touch(tmp_path / "point_cloud" / "iteration_7000" / "point_cloud.ply")
    _touch(tmp_path / "point_cloud" / "iteration_15000" / "point_cloud.ply")
    _touch(tmp_path / "splat_9000.ply")
    _touch(tmp_path / "splat_30000.ply")
    assert M.trained_ply(tmp_path) == tmp_path / "splat_30000.ply"

    # And when the GS-2M iteration is the highest, it wins instead.
    other = tmp_path / "other"
    _touch(other / "point_cloud" / "iteration_40000" / "point_cloud.ply")
    _touch(other / "splat_30000.ply")
    assert M.trained_ply(other) == other / "point_cloud" / "iteration_40000" / "point_cloud.ply"


# ---------------------------------------------------------------------------
# trained_model_dir
# ---------------------------------------------------------------------------

def test_trained_model_dir_none_without_model_path():
    assert M.trained_model_dir(_job(meta={})) is None
    assert M.trained_model_dir(_job(meta=None)) is None


def test_trained_model_dir_none_when_dir_missing(tmp_path):
    missing = tmp_path / "nope"
    assert M.trained_model_dir(_job(meta={"model_path": str(missing)})) is None


def test_trained_model_dir_none_when_dir_has_no_ply(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    assert M.trained_model_dir(_job(meta={"model_path": str(d)})) is None


def test_trained_model_dir_returns_dir_when_ply_present(tmp_path):
    d = tmp_path / "model"
    _touch(d / "point_cloud" / "iteration_30000" / "point_cloud.ply")
    assert M.trained_model_dir(_job(meta={"model_path": str(d)})) == d


# ---------------------------------------------------------------------------
# make_edited_model
# ---------------------------------------------------------------------------

def test_make_edited_model_raises_when_no_point_cloud(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(ValueError):
        M.make_edited_model(model)


def test_make_edited_model_builds_sibling_with_symlink(tmp_path):
    model = tmp_path / "scene_model"
    _touch(model / "point_cloud" / "iteration_7" / "point_cloud.ply")
    cfg = model / "cfg_args"
    cfg.write_text("source_path=/data/scene")

    edited, dest = M.make_edited_model(model)

    # Sibling of the source, named <name>_edited_<timestamp>.
    assert edited.parent == model.parent
    assert edited.name.startswith("scene_model_edited_")

    # dest is the edited cloud path; its parent (the iteration dir) is created,
    # and it preserves the source's iteration_<N> directory name.
    assert dest == edited / "point_cloud" / "iteration_7" / "point_cloud.ply"
    assert dest.parent.is_dir()
    # The caller writes the bytes, so the .ply itself does not exist yet.
    assert not dest.exists()

    # cfg_args symlink points back at the original (never copied / never touched).
    link = edited / "cfg_args"
    assert link.is_symlink()
    assert os.path.realpath(link) == str(cfg.resolve())
    assert link.read_text() == "source_path=/data/scene"


def test_make_edited_model_symlinks_lighting_when_present(tmp_path):
    # lighting.pth lives beside the source cloud only when trained with --material;
    # it should be mirrored as a symlink into the sibling iteration dir.
    model = tmp_path / "mat_model"
    it_dir = model / "point_cloud" / "iteration_12"
    _touch(it_dir / "point_cloud.ply")
    (it_dir / "lighting.pth").write_text("lights")
    (model / "cfg_args").write_text("cfg")

    edited, dest = M.make_edited_model(model)
    mirrored = dest.parent / "lighting.pth"
    assert mirrored.is_symlink()
    assert mirrored.read_text() == "lights"


def test_make_edited_model_does_not_mutate_source(tmp_path):
    model = tmp_path / "src"
    _touch(model / "point_cloud" / "iteration_3" / "point_cloud.ply")
    (model / "cfg_args").write_text("x")
    before = sorted(p.name for p in model.iterdir())

    M.make_edited_model(model)

    after = sorted(p.name for p in model.iterdir())
    assert before == after  # original dir contents untouched
