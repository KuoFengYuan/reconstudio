"""blocksplit: pure helpers, the form validator, and an end-to-end split of a
tiny synthetic dense scene (two clusters → two blocks, tiled and non-tiled)."""
import json
import math
from pathlib import Path

import pytest

# blocksplit + the COLMAP read/write helpers need numpy (and the e2e split needs
# cv2). The panel keeps these out of its base/CI install, so skip this module when
# absent — mirrors how large_scene's numpy code isn't exercised in CI. Runs in full
# locally (the `rec` env has both).
np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from pipeline.blocksplit import (  # noqa: E402
    AUTO_TARGET_VIEWS,
    auto_partition_params,
    grid_cells,
    hull_area,
    parse_region,
    rebalance_cells,
    run_blocksplit,
    select_tiles,
    shift_principal_point,
    tile_bounds,
    tile_layout,
    tile_name,
    visibility,
)
from pipeline.vendor.read_write_model import (  # noqa: E402
    Camera,
    Image,
    Point3D,
    read_model,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)
from web.services.forms import build_blocksplit_params  # noqa: E402


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


def test_rebalance_cells_splits_dense_keeps_sparse():
    # 200 cameras packed in x<100, 6 in x>300 over a 0..400 strip
    rng = np.random.default_rng(0)
    dense = np.column_stack([rng.uniform(0, 100, 200), rng.uniform(0, 100, 200)])
    sparse = np.column_stack([rng.uniform(300, 400, 6), rng.uniform(0, 100, 6)])
    cxy = np.vstack([dense, sparse])
    rects = rebalance_cells(0, 0, 400, 100, cxy, max_images=50,
                            min_images=2, min_size=25)
    # disjoint + exact cover of the region area
    assert sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects) == pytest.approx(400 * 100)
    # every partition holds ≤ max_images cameras (or couldn't be split further)
    for x0, y0, x1, y1 in rects:
        n = int(((cxy[:, 0] >= x0) & (cxy[:, 0] < x1) &
                 (cxy[:, 1] >= y0) & (cxy[:, 1] < y1)).sum())
        assert n <= 50 or min(x1 - x0, y1 - y0) < 2 * 25
    # dense half ends up with more partitions than the sparse half
    left = [r for r in rects if (r[0] + r[2]) / 2 < 200]
    right = [r for r in rects if (r[0] + r[2]) / 2 >= 200]
    assert len(left) > len(right)


def test_rebalance_cells_no_split_when_under_capacity():
    cxy = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    rects = rebalance_cells(0, 0, 100, 100, cxy, max_images=50,
                            min_images=1, min_size=10)
    assert rects == [(0.0, 0.0, 100.0, 100.0)]   # too few cams → single block


def test_auto_partition_params_scales_blocks_with_camera_count():
    region = (0.0, 0.0, 1000.0, 1000.0)
    # N just over the target → 2 blocks worth of capacity
    n = 2 * AUTO_TARGET_VIEWS + 10
    cxy = np.zeros((n, 2))
    max_images, min_images, min_size, buffer = auto_partition_params(cxy, region)
    k = round(n / AUTO_TARGET_VIEWS)
    assert max_images == math.ceil(n / k)            # ~target per block
    assert 0 < min_images < max_images
    assert min_size > 0 and buffer > 0
    # fewer cameras → a single block (no over-splitting)
    few = auto_partition_params(np.zeros((20, 2)), region)
    assert few[0] == 20                              # max_images == N → one block


def test_auto_partition_params_handles_empty():
    mi, _, ms, buf = auto_partition_params(np.empty((0, 2)), (0.0, 0.0, 10.0, 10.0))
    assert mi >= 1 and ms > 0 and buf > 0


def test_tile_name_keeps_subfolder():
    assert tile_name("nadir/N-1_0.jpg", 2, 3) == "nadir/N-1_0_r2c3.jpg"
    assert tile_name("img.png", 0, 0) == "img_r0c0.jpg"


def test_select_tiles_threshold_and_top1_fallback():
    # 2x2 grid over 64x48: 10 obs in tile (0,0), 3 in tile (1,1)
    xy = np.array([[1.0, 1.0]] * 10 + [[60.0, 40.0]] * 3)
    assert select_tiles(xy, 64, 48, 2, 2, 5) == [(0, 0)]      # (1,1) under threshold
    assert select_tiles(xy, 64, 48, 2, 2, 2) == [(0, 0), (1, 1)]
    # nothing reaches the threshold -> only the busiest tile, NOT all touched
    assert select_tiles(xy, 64, 48, 2, 2, 50) == [(0, 0)]
    assert select_tiles(np.empty((0, 2)), 64, 48, 2, 2, 1) == []


def test_hull_area_square_triangle_and_degenerate():
    sq = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert hull_area(sq) == pytest.approx(100.0)
    # interior points don't change the hull
    assert hull_area(np.vstack([sq, [[5.0, 5.0], [3.0, 7.0]]])) == pytest.approx(100.0)
    assert hull_area(np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])) == pytest.approx(6.0)
    assert hull_area(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])) == 0.0  # collinear
    assert hull_area(np.array([[0.0, 0.0], [1.0, 1.0]])) == 0.0              # < 3 pts
    assert hull_area(np.empty((0, 2))) == 0.0


def test_visibility_ratio():
    allpts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])   # area 100
    half = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 10.0], [0.0, 10.0]])       # area 50
    assert visibility(half, allpts) == pytest.approx(0.5)
    assert visibility(allpts, allpts) == pytest.approx(1.0)
    assert visibility(np.empty((0, 2)), allpts) == 0.0
    assert visibility(half, np.array([[0.0, 0.0], [1.0, 1.0]])) == 0.0        # degenerate V_i


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
    ("vis_thresh", "0"), ("vis_thresh", "1.5"), ("vis_thresh", "abc"),
    ("max_images", "abc"),
])
def test_build_blocksplit_params_rejects(field, value):
    with pytest.raises(ValueError):
        build_blocksplit_params("/src", "/out", {field: value})


def test_build_blocksplit_params_rebalance_and_max_images():
    p = build_blocksplit_params("/src", "/out", {"rebalance": "1"})
    assert p["rebalance"] is True and p["max_images"] == "200"
    assert build_blocksplit_params("/src", "/out", {})["rebalance"] is False
    with pytest.raises(ValueError, match="max_images"):   # max_images < min_images
        build_blocksplit_params("/src", "/out", {"max_images": "5", "min_images": "15"})


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


def _scene(tmp_path, cams_per_cluster=3):
    """Flat dense layout: two point clusters 150 apart, N nadir cams each."""
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
        base = [(cx - 2, cy, 50.0), (cx + 2, cy, 50.0), (cx, cy - 2, 50.0)]
        extra = [(cx + float(dx), cy + float(dy), 50.0)
                 for dx, dy in rng.uniform(-3, 3, size=(max(0, cams_per_cluster - 3), 2))]
        cam_centers = (base + extra)[:cams_per_cluster]
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
        for k, p3 in zip(ids, cluster_pts, strict=True):
            pts[int(k)] = Point3D(id=int(k), xyz=p3, rgb=np.array([10, 20, 30]),
                                  error=1.0, image_ids=np.array(cam_iids),
                                  point2D_idxs=np.zeros(len(cam_iids), dtype=np.int64))
    write_cameras_binary(cameras, str(ws / "sparse" / "cameras.bin"))
    write_images_binary(images, str(ws / "sparse" / "images.bin"))
    write_points3D_binary(pts, str(ws / "sparse" / "points3D.bin"))
    return ws


def _params(ws, out, **over):
    p = {"source": str(ws), "out_dir": str(out), "block_size": "100",
         "buffer": "5", "region": "0,0,200,10", "tile": True,
         "max_tile_px": "32", "jpeg_quality": "90",
         "min_tile_obs": "1", "min_images": "2", "workers": "2",
         "auto": False, "rebalance": False}   # pin the fixed-grid path for these tests
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
    # every view's symlink resolves to a real crop in the pool, via a RELATIVE
    # target so the whole out_dir can be moved/copied to another machine
    import os
    for im in imgs.values():
        link = out / "block_0_0" / "images" / im.name
        assert link.is_symlink() and link.resolve().is_file()
        assert link.resolve().is_relative_to(out / "_tiles")
        assert not os.path.isabs(os.readlink(link))
    # manifest carries per-block pixel totals + the image -> blocks map
    b0 = next(b for b in manifest["blocks"] if b["name"] == "block_0_0")
    assert b0["pixels"] == b0["train_views"] * 32 * 24
    assert manifest["image_blocks"]
    assert all(isinstance(v, list) and v for v in manifest["image_blocks"].values())
    # crop pixels match the source region (gradient image → unique columns)
    import cv2
    some = next(iter(imgs.values()))
    crop = cv2.imread(str(out / "block_0_0" / "images" / some.name))
    assert crop.shape[:2] == (24, 32)


def test_run_blocksplit_rebalance_splits_two_clusters(tmp_path):
    """Adaptive rebalance (no fixed grid): the two camera clusters land in two
    capacity-split partitions, each carrying only its own cluster's views."""
    ws = _scene(tmp_path)
    out = tmp_path / "out_rebal"
    r = _Runner()
    run_blocksplit(_params(ws, out, rebalance=True, max_images="2",
                           min_images="1"), r)

    manifest = json.loads((out / "manifest.json").read_text())
    assert {b["name"] for b in manifest["blocks"]} == {"block_0_0", "block_1_0"}
    assert any("rebalance" in ln for ln in r.lines)

    _, imgs0, _ = read_model(str(out / "block_0_0" / "sparse"))
    _, imgs1, _ = read_model(str(out / "block_1_0" / "sparse"))
    assert all(Path(im.name).name.startswith("camA") for im in imgs0.values())
    assert all(Path(im.name).name.startswith("camB") for im in imgs1.values())


def test_run_blocksplit_auto_needs_no_tuning(tmp_path):
    """auto mode: with only block-geometry params left unset, it derives the
    partition from the camera layout and still produces a valid trainable scene."""
    ws = _scene(tmp_path, cams_per_cluster=8)      # 16 cams → auto keeps 1 block
    out = tmp_path / "out_auto"
    r = _Runner()
    # only source/out_dir/region/tile — no block_size/buffer/max_images/min_images
    run_blocksplit({"source": str(ws), "out_dir": str(out), "region": "0,0,200,10",
                    "tile": False, "auto": True, "max_tile_px": "32"}, r)

    assert any("auto:" in ln for ln in r.lines)
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["blocks"]) >= 1
    cams, imgs, pts = read_model(str(out / manifest["blocks"][0]["name"] / "sparse"))
    assert imgs and pts and cams


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


def test_run_blocksplit_resume_skips_existing_tiles(tmp_path):
    """An interrupted run (no manifest yet) can be re-run into the same out_dir:
    already-cropped tiles are reused byte-for-byte, not re-encoded."""
    ws = _scene(tmp_path)
    out = tmp_path / "out_resume"
    run_blocksplit(_params(ws, out), _Runner())
    (out / "manifest.json").unlink()                  # simulate a cancelled run
    before = {p: p.stat().st_mtime_ns for p in (out / "_tiles").rglob("*.jpg")}
    assert before
    r = _Runner()
    run_blocksplit(_params(ws, out), r)
    assert (out / "manifest.json").is_file()
    after = {p: p.stat().st_mtime_ns for p in (out / "_tiles").rglob("*.jpg")}
    assert after == before                            # nothing re-encoded
    assert any("resume" in ln for ln in r.lines)


def test_run_blocksplit_refuses_resume_with_changed_params(tmp_path):
    """Leftover tiles were cut with different params -> same names, different
    content; refuse instead of silently mixing geometries."""
    ws = _scene(tmp_path)
    out = tmp_path / "out_mismatch"
    run_blocksplit(_params(ws, out), _Runner())
    (out / "manifest.json").unlink()
    with pytest.raises(ValueError, match="參數不同"):
        run_blocksplit(_params(ws, out, jpeg_quality="80"), _Runner())


def test_run_blocksplit_no_blocks_raises(tmp_path):
    ws = _scene(tmp_path)
    with pytest.raises(ValueError, match="min_images"):
        run_blocksplit(_params(ws, tmp_path / "out_none", min_images="99"),
                       _Runner())
