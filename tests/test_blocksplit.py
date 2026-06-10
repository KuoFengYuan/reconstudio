"""blocksplit: pure helpers, the form validator, and an end-to-end split of a
tiny synthetic dense scene (two clusters → two blocks, tiled and non-tiled)."""
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.blocksplit import (
    grid_cells,
    parse_region,
    run_blocksplit,
    shift_principal_point,
    tile_bounds,
    tile_layout,
    tile_name,
)
from pipeline.vendor.read_write_model import (
    Camera,
    Image,
    Point3D,
    read_model,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)
from web.services.forms import build_blocksplit_params


class _Runner:
    """Just enough of pipeline.Runner for in-process use."""
    def __init__(self):
        self.lines = []

    def log(self, msg=""):
        self.lines.append(str(msg))

    def banner(self, msg):
        self.lines.append(str(msg))

    def check_cancel(self):
        pass


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_tile_layout_and_bounds_cover_exactly():
    w, h = 14204, 10652                      # Phase One iXM-RS150F
    cols, rows = tile_layout(w, h, 4096)
    assert (cols, rows) == (4, 3)
    seen = np.zeros((h, w), bool)
    for tr in range(rows):
        for tc in range(cols):
            x0, y0, x1, y1 = tile_bounds(w, h, cols, rows, tc, tr)
            assert max(x1 - x0, y1 - y0) <= 4096
            assert not seen[y0:y1, x0:x1].any()   # no overlap
            seen[y0:y1, x0:x1] = True
    assert seen.all()                             # no gap


def test_tile_layout_small_image_single_tile():
    assert tile_layout(640, 480, 4096) == (1, 1)


def test_shift_principal_point_pinhole():
    cam = Camera(id=1, model="PINHOLE", width=100, height=80,
                 params=np.array([60.0, 61.0, 50.0, 40.0]))
    t = shift_principal_point(cam, 7, 50, 40, 100, 80)
    assert t.id == 7 and (t.width, t.height) == (50, 40)
    assert t.params.tolist() == [60.0, 61.0, 0.0, 0.0]   # focal kept, cx/cy shifted


def test_shift_principal_point_simple_pinhole_and_reject():
    cam = Camera(id=1, model="SIMPLE_PINHOLE", width=100, height=80,
                 params=np.array([60.0, 50.0, 40.0]))
    t = shift_principal_point(cam, 2, 10, 5, 60, 45)
    assert t.params.tolist() == [60.0, 40.0, 35.0]
    bad = Camera(id=1, model="OPENCV", width=10, height=10,
                 params=np.zeros(8))
    with pytest.raises(ValueError):
        shift_principal_point(bad, 2, 0, 0, 5, 5)


def test_parse_region():
    assert parse_region("") is None
    assert parse_region("0, 0, 10, 20") == (0.0, 0.0, 10.0, 20.0)
    assert parse_region("0 0 10 20") == (0.0, 0.0, 10.0, 20.0)
    for bad in ("1,2,3", "a,b,c,d", "10,0,0,20"):
        with pytest.raises(ValueError):
            parse_region(bad)


def test_grid_cells_cover_region():
    cells = grid_cells(0, 0, 250, 90, 100)
    assert len(cells) == 3 * 1
    assert cells[0][2:] == (0, 0, 100, 100)
    assert cells[-1][2:] == (200, 0, 300, 100)


def test_tile_name_keeps_subfolder():
    assert tile_name("nadir/N-1_0.jpg", 2, 3) == "nadir/N-1_0_r2c3.jpg"
    assert tile_name("img.png", 0, 0) == "img_r0c0.jpg"


# --------------------------------------------------------------------------- #
# form validator
# --------------------------------------------------------------------------- #
def test_build_blocksplit_params_defaults_and_tile_bool():
    p = build_blocksplit_params("/src", "/out", {"tile": "1"})
    assert p["block_size"] == "500" and p["buffer"] == "120"
    assert p["tile"] is True
    p = build_blocksplit_params("/src", "/out", {})
    assert p["tile"] is False


@pytest.mark.parametrize("field,value", [
    ("block_size", "abc"), ("block_size", "0"), ("buffer", "-1"),
    ("max_tile_px", "100"), ("jpeg_quality", "101"), ("workers", "0"),
    ("region", "1,2,3"),
])
def test_build_blocksplit_params_rejects(field, value):
    with pytest.raises(ValueError):
        build_blocksplit_params("/src", "/out", {field: value})


# --------------------------------------------------------------------------- #
# end-to-end on a synthetic scene
# --------------------------------------------------------------------------- #
_F, _W, _H = 60.0, 64, 48          # focal px, image size
_QVEC = np.array([0.0, 1.0, 0.0, 0.0])   # 180° about x: looking straight down
_R = np.diag([1.0, -1.0, -1.0])


def _project(c, pts):
    """COLMAP convention: x_cam = R (X - C); u = f·x/z + cx."""
    xc = (pts - c) @ _R.T
    return np.stack([_F * xc[:, 0] / xc[:, 2] + _W / 2,
                     _F * xc[:, 1] / xc[:, 2] + _H / 2], axis=1)


def _scene(tmp_path):
    """Flat dense layout: two point clusters 150 apart, 3 nadir cams each."""
    import cv2

    rng = np.random.default_rng(7)
    ws = tmp_path / "ws"
    (ws / "sparse").mkdir(parents=True)
    (ws / "images").mkdir()

    cameras = {1: Camera(id=1, model="PINHOLE", width=_W, height=_H,
                         params=np.array([_F, _F, _W / 2, _H / 2]))}
    clusters = {"A": (5.0, 5.0), "B": (155.0, 5.0)}
    pts = {}
    images = {}
    pid, iid = 1, 1
    for label, (cx, cy) in clusters.items():
        xy = rng.uniform(-5, 5, size=(60, 2)) + (cx, cy)
        ids = np.arange(pid, pid + 60)
        pid += 60
        cluster_pts = np.column_stack([xy, np.zeros(60)])
        cam_centers = [(cx - 2, cy, 50.0), (cx + 2, cy, 50.0), (cx, cy - 2, 50.0)]
        cam_iids = []
        for cc in cam_centers:
            c = np.array(cc)
            name = f"cam{label}_{iid}.png"
            images[iid] = Image(id=iid, qvec=_QVEC, tvec=-_R @ c, camera_id=1,
                                name=name, xys=_project(c, cluster_pts),
                                point3D_ids=ids.copy())
            grad = np.linspace(0, 255, _W, dtype=np.uint8)[None, :, None]
            cv2.imwrite(str(ws / "images" / name),
                        np.broadcast_to(grad, (_H, _W, 3)).copy())
            cam_iids.append(iid)
            iid += 1
        for k, p3 in zip(ids, cluster_pts):
            pts[int(k)] = Point3D(id=int(k), xyz=p3, rgb=np.array([10, 20, 30]),
                                  error=1.0, image_ids=np.array(cam_iids),
                                  point2D_idxs=np.array([0, 0, 0]))
    write_cameras_binary(cameras, str(ws / "sparse" / "cameras.bin"))
    write_images_binary(images, str(ws / "sparse" / "images.bin"))
    write_points3D_binary(pts, str(ws / "sparse" / "points3D.bin"))
    return ws


def _params(ws, out, **over):
    p = {"source": str(ws), "out_dir": str(out), "block_size": "100",
         "buffer": "5", "region": "0,0,200,10", "tile": True,
         "max_tile_px": "32", "jpeg_quality": "90", "min_obs": "10",
         "min_tile_obs": "1", "min_images": "2", "workers": "2"}
    p.update(over)
    return p


def test_run_blocksplit_tiled(tmp_path):
    ws = _scene(tmp_path)
    out = tmp_path / "out"
    r = _Runner()
    run_blocksplit(_params(ws, out), r)

    manifest = json.loads((out / "manifest.json").read_text())
    names = {b["name"] for b in manifest["blocks"]}
    assert names == {"block_0_0", "block_1_0"}

    cams, imgs, pts = read_model(str(out / "block_0_0" / "sparse"))
    assert imgs and pts
    # cluster A only: every view derives from a camA source, B never leaks in
    assert all(Path(im.name).name.startswith("camA") for im in imgs.values())
    # block points stay inside bounds+buffer
    assert all(p.xyz[0] < 105 for p in pts.values())
    # tile cameras: principal point shifted by the tile origin, sizes halved
    for c in cams.values():
        assert c.model == "PINHOLE"
        assert (c.width, c.height) == (32, 24)
        assert c.params[2] in (32.0, 0.0) and c.params[3] in (24.0, 0.0)
    # views carry the block's observations in tile coordinates, and the rebuilt
    # tracks round-trip: point → (view, idx) → same point
    for im in imgs.values():
        assert len(im.xys) == len(im.point3D_ids) > 0
        assert all(int(p) in pts for p in im.point3D_ids)
        assert (im.xys[:, 0] >= 0).all() and (im.xys[:, 0] < 32).all()
        assert (im.xys[:, 1] >= 0).all() and (im.xys[:, 1] < 24).all()
    tracked = [p for p in pts.values() if len(p.image_ids)]
    assert tracked
    p0 = tracked[0]
    vid, k = int(p0.image_ids[0]), int(p0.point2D_idxs[0])
    assert int(imgs[vid].point3D_ids[k]) == p0.id
    # every view's symlink resolves to a real crop in the pool
    for im in imgs.values():
        link = out / "block_0_0" / "images" / im.name
        assert link.is_symlink() and link.resolve().is_file()
        assert link.resolve().is_relative_to(out / "_tiles")
    # crop pixels match the source region (gradient image → unique columns)
    import cv2
    some = next(iter(imgs.values()))
    crop = cv2.imread(str(out / "block_0_0" / "images" / some.name))
    assert crop.shape[:2] == (24, 32)


def test_run_blocksplit_no_tile_links_originals(tmp_path):
    ws = _scene(tmp_path)
    out = tmp_path / "out_plain"
    run_blocksplit(_params(ws, out, tile=False), _Runner())

    cams, imgs, _ = read_model(str(out / "block_1_0" / "sparse"))
    assert all((c.width, c.height) == (_W, _H) for c in cams.values())
    assert all(im.name.startswith("camB") for im in imgs.values())
    for im in imgs.values():
        link = out / "block_1_0" / "images" / im.name
        assert link.is_symlink()
        assert link.resolve() == (ws / "images" / im.name).resolve()
    assert not (out / "_tiles").exists()


def test_viewer_resolution_for_blocksplit_job(tmp_path):
    """model_dir/resolve_model must surface block models for blocksplit jobs."""
    import types

    from web.services.models import model_dir, resolve_model

    ws = _scene(tmp_path)
    out = tmp_path / "out_viz"
    run_blocksplit(_params(ws, out), _Runner())

    job = types.SimpleNamespace(kind="blocksplit",
                                meta={"out_dir": str(out)}, params={})
    assert model_dir(job) == out / "block_0_0" / "sparse"          # default = first
    assert resolve_model(job, "block_1_0") == out / "block_1_0" / "sparse"
    assert resolve_model(job, "no_such_block") is None
    assert resolve_model(job, "../outside") is None                # confined to out_dir


def test_block_list_manifest_and_fallback(tmp_path):
    """The viewer's block switcher: manifest first, glob fallback, [] otherwise."""
    import types

    from web.services.models import block_list

    ws = _scene(tmp_path)
    out = tmp_path / "out_list"
    run_blocksplit(_params(ws, out), _Runner())

    job = types.SimpleNamespace(kind="blocksplit",
                                meta={"out_dir": str(out)}, params={})
    names = [b["name"] for b in block_list(job)]
    assert names == ["block_0_0", "block_1_0"]
    assert all(isinstance(b["views"], int) for b in block_list(job))

    (out / "manifest.json").unlink()                       # pre-manifest fallback
    assert [b["name"] for b in block_list(job)] == ["block_0_0", "block_1_0"]

    other = types.SimpleNamespace(kind="colmap", meta={"out_dir": str(out)}, params={})
    assert block_list(other) == []


def test_jobstatus_parser_collects_blocks():
    import types

    from jobs import _parse_blocksplit

    job = types.SimpleNamespace(meta={})
    _parse_blocksplit(job, "[blocksplit] block_0_0: 33 src images → 33 views, 17522 pts (33 tiles)")
    _parse_blocksplit(job, "[blocksplit] block_3_0: 110 src images → 139 views, 62449 pts (139 tiles)")
    _parse_blocksplit(job, "[blocksplit] block_0_0: 33 src images → 33 views, 17522 pts (33 tiles)")
    _parse_blocksplit(job, "[blocksplit] tiles [40/193] left/L-6_0-50336.tif")   # noise
    assert job.meta["blocks"] == [{"name": "block_0_0", "views": 33},
                                  {"name": "block_3_0", "views": 139}]


def test_run_blocksplit_refuses_existing_output(tmp_path):
    ws = _scene(tmp_path)
    out = tmp_path / "out_dup"
    run_blocksplit(_params(ws, out), _Runner())
    with pytest.raises(ValueError, match="已有分塊結果"):
        run_blocksplit(_params(ws, out), _Runner())


def test_run_blocksplit_no_blocks_raises(tmp_path):
    ws = _scene(tmp_path)
    with pytest.raises(ValueError, match="min_images"):
        run_blocksplit(_params(ws, tmp_path / "out_none", min_images="99"),
                       _Runner())
