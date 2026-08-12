"""Recovering a job record from an existing COLMAP workspace on disk.

讀取既有結果 exists so a reconstruction produced outside the panel (or one whose
job record was deleted) is still viewable/trainable. Everything downstream —
model_dir, dense_dir, _source_image — rebuilds its paths from the job's
meta/params, so inspect_workspace has to recover exactly those fields, and get
them wrong in a way that 404s rather than silently pointing elsewhere.
"""
from pathlib import Path

from web.services import models as M


def _sparse(root: Path, sub: str = "sparse/0") -> Path:
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    for f in ("images.bin", "points3D.bin", "cameras.bin"):
        (d / f).write_bytes(b"")
    return d


def _dense(root: Path, name: str) -> Path:
    d = root / name
    (d / "sparse").mkdir(parents=True, exist_ok=True)
    (d / "sparse" / "cameras.bin").write_bytes(b"")
    (d / "images").mkdir(parents=True, exist_ok=True)
    return d


def test_none_when_no_sparse_model(tmp_path):
    assert M.inspect_workspace(tmp_path) is None


# ---------------------------------------------------------------------------
# sparse layout: sparse/0 (this panel), flat sparse/ (GLOMAP, GUI exports,
# undistorted copies), sparse/<N> (mapper that dropped its first submodel)
# ---------------------------------------------------------------------------

def test_workspace_model_prefers_sparse_0(tmp_path):
    _sparse(tmp_path, "sparse/0")
    _sparse(tmp_path, "sparse")          # flat binaries alongside — sparse/0 still wins
    assert M.workspace_model(tmp_path) == tmp_path / "sparse" / "0"


def test_workspace_model_accepts_flat_sparse(tmp_path):
    _sparse(tmp_path, "sparse")
    assert M.workspace_model(tmp_path) == tmp_path / "sparse"


def test_workspace_model_falls_back_to_numbered_submodel(tmp_path):
    _sparse(tmp_path, "sparse/2")
    _sparse(tmp_path, "sparse/1")
    (tmp_path / "sparse" / "notanumber").mkdir()
    assert M.workspace_model(tmp_path) == tmp_path / "sparse" / "1"   # lowest number


def test_workspace_model_ignores_incomplete_dirs(tmp_path):
    (tmp_path / "sparse" / "0").mkdir(parents=True)
    (tmp_path / "sparse" / "0" / "cameras.bin").write_bytes(b"")   # no images/points3D
    assert M.workspace_model(tmp_path) is None


def test_model_dir_resolves_flat_sparse_workspace(tmp_path):
    import types
    _sparse(tmp_path, "sparse")
    job = types.SimpleNamespace(meta={"workspace": str(tmp_path)}, params={})
    assert M.model_dir(job) == tmp_path / "sparse"


def test_inspect_workspace_accepts_flat_sparse(tmp_path):
    _sparse(tmp_path, "sparse")
    info = M.inspect_workspace(tmp_path)
    assert info and info["model"] == str(tmp_path / "sparse")


def test_none_when_model_dir_is_incomplete(tmp_path):
    # a sparse/0 that only has cameras.bin is not something the viewer can read
    (tmp_path / "sparse" / "0").mkdir(parents=True)
    (tmp_path / "sparse" / "0" / "cameras.bin").write_bytes(b"")
    assert M.inspect_workspace(tmp_path) is None


def test_recovers_dataset_name_and_mapper(tmp_path):
    _sparse(tmp_path)
    _dense(tmp_path, "training_dataset_global_mapper")
    info = M.inspect_workspace(tmp_path)
    assert info["dataset_name"] == "training_dataset"
    assert info["mapper"] == "global"
    assert info["dataset"] == str(tmp_path / "training_dataset_global_mapper")


def test_recovered_fields_feed_dense_dir(tmp_path):
    # the whole point: the recovered params must resolve back to the same dir
    import types
    _sparse(tmp_path)
    dense = _dense(tmp_path, "mydata_hierarchical_mapper")
    info = M.inspect_workspace(tmp_path)
    job = types.SimpleNamespace(
        meta={"workspace": str(tmp_path)},
        params={"dataset_name": info["dataset_name"], "mapper": info["mapper"]})
    assert M.dense_dir(job) == dense
    assert M.model_dir(job) == tmp_path / "sparse" / "0"


def test_ignores_incomplete_mapper_dir(tmp_path):
    # a *_mapper dir left half-written (no images/) must not be reported as the dataset
    _sparse(tmp_path)
    (tmp_path / "aborted_global_mapper" / "sparse").mkdir(parents=True)
    (tmp_path / "aborted_global_mapper" / "sparse" / "cameras.bin").write_bytes(b"")
    info = M.inspect_workspace(tmp_path)
    assert info["dataset"] == ""
    assert info["dataset_name"] == "training_dataset"   # falls back to the defaults


def test_picks_largest_resize_copy_as_image_root(tmp_path):
    _sparse(tmp_path)
    (tmp_path / "images_1920").mkdir()
    (tmp_path / "images_4096").mkdir()
    (tmp_path / "images_notanumber").mkdir()
    info = M.inspect_workspace(tmp_path)
    assert info["resize_max"] == "4096"
    assert info["image_root"] == str(tmp_path / "images_4096")


def test_falls_back_to_plain_images_dir(tmp_path):
    _sparse(tmp_path)
    (tmp_path / "images").mkdir()
    info = M.inspect_workspace(tmp_path)
    assert info["image_root"] == str(tmp_path / "images")
    assert info["resize_max"] == "1920"


def test_no_images_anywhere_leaves_image_root_blank(tmp_path):
    _sparse(tmp_path)
    info = M.inspect_workspace(tmp_path)
    assert info["image_root"] == ""
