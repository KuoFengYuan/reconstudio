"""Split a finished COLMAP reconstruction into per-block 3DGS training scenes
(VastGaussian-style divide and conquer), optionally tiling the images so each
block trains at the source pixels' native GSD without a giant render target.

Why: one training run's Gaussian budget is VRAM-bound; spread over a whole
survey area the density is too thin for street-level detail, and error-driven
densification dumps the budget into vegetation. Per-block training gives every
block its full budget. Blocks stay in the source model's coordinate frame, so
the trained PLYs merge by concatenation after cropping each to its core bounds.

Input : whatever train.py accepts (workspace / flat dense dir / sparse-0 scene);
        the model must be undistorted (PINHOLE/SIMPLE_PINHOLE), same as training.
Output: <out_dir>/block_<ix>_<iy>/   — flat-dense layout, directly trainable:
            sparse/{cameras,images,points3D}.bin
            images/…                 (symlinks: tile pool, or original files)
        <out_dir>/_tiles/…           (shared crop pool, tiling mode only)
        <out_dir>/manifest.json
Tile links are RELATIVE (the pool lives inside out_dir), so the whole output
tree can be moved or copied to another machine. Re-running into a cancelled
run's out_dir resumes it — already-cropped tiles are skipped; a param stamp
(.blocksplit_run.json) refuses resume when tile-shaping params changed, and a
finished run (manifest.json present) still refuses outright.

Block X/Y are the model's X/Y axes. With GPS_ALIGN (enu) those are metres.
Partitioning has three modes. Default is `auto`: derive everything from the
camera layout (no tuning) — aim for ~AUTO_TARGET_VIEWS cameras per block, so
k = round(N/target) blocks, and size min-partition-side and buffer from the
resulting block footprint. Turn auto off for manual control: `rebalance` does
the same REUrbanGS adaptive split but with your max_images/min_images/block_size
(block_size = minimum partition side); turning rebalance off too falls back to a
fixed grid (block_size square per cell). Adaptive split recursively cuts the
longer axis at the camera median while never creating a block below min_images
cameras or the min side, so dense areas get more/smaller blocks and sparse areas
stay merged. Assignment follows REUrbanGS (arXiv:2507.23006) two-phase visibility
selection: an image joins a block when its camera centre falls inside the block,
OR when its visibility of the block ≥ vis_thresh — visibility = V_ij/V_i, the
convex-hull area of the image's features that hit the block over the hull area
of all the image's features (so a few stray points can't pull an image in; it
must *see* a real area of the block; an image observing < 3 of the block's
points can't form a hull, so its visibility is 0 and it's dropped). A tile of a
kept image is kept when ≥ min_tile_obs of those features land in it (when no tile reaches
that, only the busiest tile is kept — spraying every touched tile into the block
would add near-empty views).
A tile is the same camera at the same pose with the principal point shifted by
the crop origin — geometrically exact, which is why no pose/GPS work is needed.
Each block is a self-consistent COLMAP model: views carry the 2D observations
of the block's points (shifted into tile pixel coordinates) and the points'
tracks are rebuilt against the block's view ids, so the viewer's per-image
"registered pts" / track-length stats stay meaningful. Trainers only read
poses/intrinsics/xyz/rgb and ignore the rest.
"""
from __future__ import annotations

import concurrent.futures as futures
import json
import math
import os
import time
from pathlib import Path, PurePosixPath

import numpy as np

from .runner import Runner
from .train import _resolve_dense
from .vendor.read_write_model import (
    Camera,
    Image,
    Point3D,
    qvec2rotmat,
    read_images_binary,
    read_model,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)

BLOCKSPLIT_DEFAULTS = {
    "block_size": "500", "buffer": "120", "region": "",
    "tile": True, "max_tile_px": "8192", "jpeg_quality": "95",
    "min_tile_obs": "8", "min_images": "15", "workers": "4",
    "vis_thresh": "0.1667",   # REUrbanGS visibility V_ij/V_i ≥ 1/6 (arXiv:2507.23006)
    "rebalance": True,        # adaptive image-count-balanced partition vs fixed grid
    "max_images": "200",      # rebalance: split a partition above this many cameras
    "auto": True,             # derive block_size/buffer/max_images/min_images from the data
}

_PINHOLE = {"PINHOLE", "SIMPLE_PINHOLE"}


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def parse_region(s: str) -> tuple[float, float, float, float] | None:
    """'minx,miny,maxx,maxy' (comma/space separated) → tuple; ''/None → None."""
    s = (s or "").strip()
    if not s:
        return None
    parts = [t for t in s.replace(",", " ").split() if t]
    if len(parts) != 4:
        raise ValueError("region 需為 minx,miny,maxx,maxy 四個數字（留空 = 點雲全範圍）")
    try:
        v = [float(t) for t in parts]
    except ValueError:
        raise ValueError("region 的四個值需為數字") from None
    if v[0] >= v[2] or v[1] >= v[3]:
        raise ValueError("region 需 minx < maxx 且 miny < maxy")
    return v[0], v[1], v[2], v[3]


def tile_layout(width: int, height: int, max_px: int) -> tuple[int, int]:
    """(cols, rows) such that every tile's longest side ≤ max_px."""
    return max(1, math.ceil(width / max_px)), max(1, math.ceil(height / max_px))


def tile_bounds(width: int, height: int, cols: int, rows: int,
                tc: int, tr: int) -> tuple[int, int, int, int]:
    """Pixel bounds (x0, y0, x1, y1) of tile (col=tc, row=tr); exact cover, no overlap."""
    x0, x1 = (width * tc) // cols, (width * (tc + 1)) // cols
    y0, y1 = (height * tr) // rows, (height * (tr + 1)) // rows
    return x0, y0, x1, y1


def shift_principal_point(cam, new_id: int, x0: int, y0: int,
                          x1: int, y1: int) -> Camera:
    """The tile's camera: same focal/pose-frame, principal point moved by the
    crop origin. Only undistorted models qualify (checked before we get here)."""
    p = [float(v) for v in cam.params]
    if cam.model == "PINHOLE":
        p2 = [p[0], p[1], p[2] - x0, p[3] - y0]
    elif cam.model == "SIMPLE_PINHOLE":
        p2 = [p[0], p[1] - x0, p[2] - y0]
    else:
        raise ValueError(f"unsupported camera model: {cam.model}")
    return Camera(id=new_id, model=cam.model, width=int(x1 - x0),
                  height=int(y1 - y0), params=np.array(p2, dtype=np.float64))


def grid_cells(minx: float, miny: float, maxx: float, maxy: float,
               size: float) -> list[tuple[int, int, float, float, float, float]]:
    """All (ix, iy, bx0, by0, bx1, by1) cells of a `size`-step grid covering the region."""
    nx = max(1, math.ceil((maxx - minx) / size))
    ny = max(1, math.ceil((maxy - miny) / size))
    return [(ix, iy, minx + ix * size, miny + iy * size,
             minx + (ix + 1) * size, miny + (iy + 1) * size)
            for iy in range(ny) for ix in range(nx)]


def rebalance_cells(minx: float, miny: float, maxx: float, maxy: float,
                    centers_xy: np.ndarray, max_images: int, min_images: int,
                    min_size: float) -> list[tuple[float, float, float, float]]:
    """REUrbanGS-style adaptive partition (arXiv:2507.23006 §3.2.2): instead of a
    fixed grid, recursively split the region so each partition holds ≤ max_images
    cameras, cutting the longer axis at the camera-centre median (so the two halves
    get balanced counts). A split is rejected when it would leave either side with
    < min_images cameras or a side < min_size — so dense areas end up with more,
    smaller partitions ('subdivide too-many') while sparse areas stay merged
    ('merge too-few', i.e. never split into under-full blocks). Returns disjoint
    rectangles (x0,y0,x1,y1) covering the region, sorted for determinism.

    Balance is on camera centres (capture density), the cheap, standard proxy for
    'images assigned' — the same signal VastGaussian/REUrbanGS partition on."""
    cx, cy = centers_xy[:, 0], centers_xy[:, 1]
    out: list[tuple[float, float, float, float]] = []
    stack = [(minx, miny, maxx, maxy)]
    while stack:
        x0, y0, x1, y1 = stack.pop()
        inr = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        n = int(inr.sum())
        w, h = x1 - x0, y1 - y0
        if n > max_images and max(w, h) >= 2 * min_size:
            if w >= h:
                s = min(max(float(np.median(cx[inr])), x0 + min_size), x1 - min_size)
                a, b = (x0, y0, s, y1), (s, y0, x1, y1)
            else:
                s = min(max(float(np.median(cy[inr])), y0 + min_size), y1 - min_size)
                a, b = (x0, y0, x1, s), (x0, s, x1, y1)
            na = int(((cx >= a[0]) & (cx < a[2]) & (cy >= a[1]) & (cy < a[3])).sum())
            if na >= min_images and (n - na) >= min_images:
                stack.extend((a, b))
                continue
        out.append((x0, y0, x1, y1))
    return sorted(out)


# auto-mode heuristics — chosen so the user sets nothing; tuned for per-block 3DGS
AUTO_TARGET_VIEWS = 120     # sweet-spot cameras per block (enough multi-view, not bloated)
AUTO_MIN_IMAGES_FRAC = 0.3  # a block may shrink to ~30% of capacity before it's too thin
AUTO_MIN_SIZE_FRAC = 0.35   # min partition side as a fraction of the nominal block side
AUTO_BUFFER_FRAC = 0.10     # overlap buffer as a fraction of the nominal block side


def auto_partition_params(centers_xy: np.ndarray,
                          region: tuple[float, float, float, float]
                          ) -> tuple[int, int, float, float]:
    """Derive (max_images, min_images, min_size, buffer) purely from the camera
    layout so the user tunes nothing. Targets ~AUTO_TARGET_VIEWS cameras per block
    (k = round(N / target) blocks), then sizes the geometry from the resulting
    nominal block footprint side = sqrt(region_area / k)."""
    n = max(1, len(centers_xy))
    minx, miny, maxx, maxy = region
    k = max(1, round(n / AUTO_TARGET_VIEWS))
    max_images = max(1, math.ceil(n / k))
    min_images = max(8, round(max_images * AUTO_MIN_IMAGES_FRAC))
    side = math.sqrt(max((maxx - minx) * (maxy - miny), 1e-9) / k)
    return max_images, min_images, AUTO_MIN_SIZE_FRAC * side, AUTO_BUFFER_FRAC * side


def tile_name(image_name: str, tr: int, tc: int) -> str:
    """Pool-relative tile path: keep the source subfolder, swap ext for .jpg."""
    rel = PurePosixPath(image_name)
    return str(rel.parent / f"{rel.stem}_r{tr}c{tc}.jpg")


def select_tiles(xy: np.ndarray, width: int, height: int, cols: int, rows: int,
                 min_tile_obs: int) -> list[tuple[int, int]]:
    """Tiles (tr, tc) holding ≥ min_tile_obs of the observations `xy`. When none
    reaches the threshold, keep only the busiest tile — the image barely grazes
    the block, and keeping every touched tile would spray near-empty views."""
    if xy.size == 0:
        return []
    tc = np.clip((xy[:, 0].astype(np.int64) * cols) // width, 0, cols - 1)
    tr = np.clip((xy[:, 1].astype(np.int64) * rows) // height, 0, rows - 1)
    counts = np.bincount(tr * cols + tc, minlength=cols * rows)
    keep = np.nonzero(counts >= min_tile_obs)[0]
    if keep.size == 0:
        keep = np.array([counts.argmax()])
    return [(int(k) // cols, int(k) % cols) for k in keep]


def hull_area(xy: np.ndarray) -> float:
    """Convex-hull area of 2D points `xy` (N×2); 0 for <3 distinct points or a
    degenerate (collinear) hull. Pure numpy — Andrew's monotone chain + the
    shoelace formula — so it needs no cv2/scipy and stays unit-testable."""
    if xy is None or len(xy) < 3:
        return 0.0
    pts = np.unique(np.asarray(xy, dtype=np.float64), axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        # 2D scalar cross (o→a) × (o→b); np.cross on 2-vectors is deprecated in NumPy 2.0
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    def _half(points: np.ndarray) -> list:
        h: list = []
        for p in points:
            while len(h) >= 2 and _cross(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h

    hull = np.array(_half(pts)[:-1] + _half(pts[::-1])[:-1])
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def visibility(xy_in: np.ndarray, xy_all: np.ndarray) -> float:
    """REUrbanGS visibility V_ij/V_i: hull area of the image's features that fall
    in the block over the hull area of all the image's features. 0 when the
    image's own hull is degenerate (can't be 'seen')."""
    vi = hull_area(xy_all)
    if vi <= 0.0:
        return 0.0
    return hull_area(xy_in) / vi


def _up_axis(sparse_dir: Path) -> int | None:
    """World axis (0=X,1=Y,2=Z) with the smallest camera-centre spread — i.e.
    the vertical/up axis (cameras sit at ~constant height, so the up axis varies
    least). None when the model can't be read or has <3 views."""
    try:
        imgs = read_images_binary(str(sparse_dir / "images.bin"))
    except Exception:
        return None
    if len(imgs) < 3:
        return None
    C = np.array([-qvec2rotmat(im.qvec).T @ im.tvec for im in imgs.values()])
    return int(np.argmin(C.max(0) - C.min(0)))


def _resolve_zup(src: Path, r: Runner) -> tuple[Path, Path]:
    """blocksplit grids on the X–Y plane, so it must read a *Z-up* model. The
    undistort/align step often emits both a reoriented model (…/sparse, Y-up
    after REORIENT) and the pre-reorient …/sparse_unaligned (Z-up). _resolve_dense
    returns the former; here we auto-swap to a Z-up sibling when the resolved one
    isn't Z-up — so source can just be the workspace / *_mapper, no hand-picking.
    Image names match across siblings (same reconstruction), so images_dir stays."""
    sparse_dir, images_dir = _resolve_dense(src)
    if _up_axis(sparse_dir) == 2:
        return sparse_dir, images_dir                       # already Z-up
    for sib in sorted(sparse_dir.parent.glob("sparse*")):
        if sib != sparse_dir and (sib / "cameras.bin").is_file() and _up_axis(sib) == 2:
            r.log(f"[blocksplit] 自動改用 Z-up 模型 {sib.name}"
                  f"（{sparse_dir.name} 非 Z-up,X-Y 切格會塌成一格）")
            return sib, images_dir
    r.log(f"[blocksplit] ⚠️ 找不到 Z-up 模型,沿用 {sparse_dir.name};"
          " 若切格塌成 1 格請改餵未 reorient 的模型。")
    return sparse_dir, images_dir


# --------------------------------------------------------------------------- #
# pipeline entry
# --------------------------------------------------------------------------- #
def run_blocksplit(p: dict, r: Runner) -> None:
    src, out = Path(p["source"]), Path(p["out_dir"])
    block_size = float(p.get("block_size") or BLOCKSPLIT_DEFAULTS["block_size"])
    buffer = float(p.get("buffer") or BLOCKSPLIT_DEFAULTS["buffer"])
    tile = bool(p.get("tile", True))
    max_tile_px = int(p.get("max_tile_px") or BLOCKSPLIT_DEFAULTS["max_tile_px"])
    jpeg_q = int(p.get("jpeg_quality") or BLOCKSPLIT_DEFAULTS["jpeg_quality"])
    min_tile_obs = int(p.get("min_tile_obs") or BLOCKSPLIT_DEFAULTS["min_tile_obs"])
    min_images = int(p.get("min_images") or BLOCKSPLIT_DEFAULTS["min_images"])
    workers = int(p.get("workers") or BLOCKSPLIT_DEFAULTS["workers"])
    vis_thresh = float(p.get("vis_thresh") or BLOCKSPLIT_DEFAULTS["vis_thresh"])
    rebalance = bool(p.get("rebalance", BLOCKSPLIT_DEFAULTS["rebalance"]))
    max_images = int(p.get("max_images") or BLOCKSPLIT_DEFAULTS["max_images"])
    auto = bool(p.get("auto", BLOCKSPLIT_DEFAULTS["auto"]))
    region = parse_region(p.get("region") or "")

    if (out / "manifest.json").is_file():
        raise ValueError(f"輸出目錄已有分塊結果，請換一個 out_dir：{out}")
    # resume guard: a cancelled run may be re-run into the same out_dir (cropped
    # tiles are reused), but only when the params that shape tile content match.
    stamp = out / ".blocksplit_run.json"
    stamp_params = {"source": str(src), "tile": tile,
                    "max_tile_px": max_tile_px, "jpeg_quality": jpeg_q}
    if stamp.is_file():
        try:
            prev = json.loads(stamp.read_text())
        except ValueError:
            prev = None
        if prev != stamp_params:
            raise ValueError(
                f"輸出目錄殘留參數不同的半成品（{stamp.name} 不符），"
                f"請清空再跑：{out}")
        r.log("[blocksplit] resume: out_dir 已有半成品（參數相符），已裁好的切片會跳過")
    else:
        out.mkdir(parents=True, exist_ok=True)
        stamp.write_text(json.dumps(stamp_params, ensure_ascii=False))

    sparse_dir, images_dir = _resolve_zup(src, r)
    r.banner(f"blocksplit start | model={sparse_dir} images={images_dir}")
    t0 = time.time()
    cameras, images, points3D = read_model(str(sparse_dir))
    models = {c.model for c in cameras.values()}
    if not models <= _PINHOLE:
        raise ValueError(
            f"相機模型為 {sorted(models)}，分塊需要去畸變的 PINHOLE 模型 —"
            " 請把 source 指向 undistort 後的輸出（跟訓練的要求相同）。")
    r.log(f"[blocksplit] {len(cameras)} cams, {len(images)} images, "
          f"{len(points3D)} points  (read {time.time() - t0:.1f}s)")

    # -- flatten points into sorted arrays for vectorized lookups ------------ #
    pid = np.fromiter(points3D.keys(), dtype=np.int64, count=len(points3D))
    pid.sort()
    xyz = np.empty((pid.size, 3), np.float64)
    rgb = np.empty((pid.size, 3), np.uint8)
    err = np.empty(pid.size, np.float64)
    for i, k in enumerate(pid):
        pt = points3D[k]
        xyz[i], rgb[i], err[i] = pt.xyz, pt.rgb, pt.error
    X, Y = xyz[:, 0], xyz[:, 1]

    # per image: rows into the sorted point arrays + the matching 2D pixels
    obs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for iid, im in images.items():
        ids3 = np.asarray(im.point3D_ids, np.int64)
        m = ids3 >= 0
        ids3 = ids3[m]
        pos = np.searchsorted(pid, ids3)
        ok = (pos < pid.size) & (pid[np.minimum(pos, pid.size - 1)] == ids3)
        obs[iid] = (pos[ok], np.asarray(im.xys, np.float64)[m][ok])

    # camera centres C = -R^T t (world frame) for the visibility phase-1 test
    centers = {iid: (-qvec2rotmat(im.qvec).T @ im.tvec)
               for iid, im in images.items()}
    # V_i (hull area of ALL the image's features) is block-independent — compute
    # it once and reuse; the per-block work is then just the in-block hull.
    vi_area = {iid: hull_area(xys2) for iid, (_rows, xys2) in obs.items()}

    if region is None:
        minx, maxx = np.percentile(X, [1, 99])
        miny, maxy = np.percentile(Y, [1, 99])
        region = (float(minx), float(miny), float(maxx), float(maxy))
        r.log(f"[blocksplit] region auto (1–99% of points): "
              f"{region[0]:.1f},{region[1]:.1f} → {region[2]:.1f},{region[3]:.1f}")
    if auto or rebalance:
        # adaptive partition balanced on camera-centre density (REUrbanGS §3.2.2):
        # dense areas → more/smaller blocks, sparse → fewer/larger.
        cxy = np.array([centers[iid][:2] for iid in images], dtype=np.float64)
        if auto:
            # derive everything from the camera layout — user tunes nothing
            max_images, min_images, min_size, buffer = auto_partition_params(cxy, region)
            r.log(f"[blocksplit] auto: {len(cxy)} cams → target ~{AUTO_TARGET_VIEWS}/block "
                  f"⇒ max_images={max_images}, min_images={min_images}, "
                  f"min side={min_size:.1f}, buffer={buffer:.1f}")
        else:
            min_size = block_size            # manual rebalance: block_size = min side
        rects = rebalance_cells(*region, cxy, max_images, min_images, min_size)
        cells = [(k, 0, *rc) for k, rc in enumerate(rects)]
        r.log(f"[blocksplit] {'auto' if auto else 'rebalance'}: adaptive partition "
              f"(≤{max_images} cams/block, ≥{min_images}, min side {min_size:g}, "
              f"buffer={buffer:g}) → {len(cells)} partitions")
    else:
        cells = grid_cells(*region, block_size)
        r.log(f"[blocksplit] grid: block_size={block_size:g} buffer={buffer:g} "
              f"→ {len(cells)} candidate cells")
    r.log(f"[blocksplit] image selection: REUrbanGS visibility "
          f"(camera-in-block OR V_ij/V_i ≥ {vis_thresh:g})")

    # -- assignment ----------------------------------------------------------- #
    # blocks: per kept cell → list of (image, [(tr,tc), …] or None when not tiling)
    blocks: list[dict] = []
    skipped: list[dict] = []
    tile_cams: dict[tuple[int, int, int], Camera] = {}   # (cam_id, tr, tc) → Camera
    needed: dict[str, set[tuple[int, int]]] = {}         # image name → tiles to crop
    next_cam_id = max(cameras) + 1 if cameras else 1

    for ix, iy, bx0, by0, bx1, by1 in cells:
        r.check_cancel()
        inb = ((bx0 - buffer <= X) & (bx1 + buffer > X) &
               (by0 - buffer <= Y) & (by1 + buffer > Y))
        if not inb.any():
            continue
        members: list[tuple[Image, list[tuple[int, int]] | None]] = []
        for iid, (rows, xys2) in obs.items():
            hit = inb[rows]
            # REUrbanGS two-phase: camera centre inside the block (core bounds,
            # no buffer) → keep; else the visibility ratio V_ij/V_i ≥ vis_thresh
            # (V_i precomputed; an image seeing < 3 block points gets a 0 hull → 0).
            cx, cy = centers[iid][0], centers[iid][1]
            cam_in = (bx0 <= cx < bx1) and (by0 <= cy < by1)
            if not cam_in:
                vi = vi_area[iid]
                if vi <= 0.0 or hull_area(xys2[hit]) / vi < vis_thresh:
                    continue
            im = images[iid]
            if not tile:
                members.append((im, None))
                continue
            cam = cameras[im.camera_id]
            cols, rows_n = tile_layout(cam.width, cam.height, max_tile_px)
            tiles = select_tiles(xys2[hit], cam.width, cam.height,
                                 cols, rows_n, min_tile_obs)
            for trr, tcc in tiles:
                key = (im.camera_id, trr, tcc)
                if key not in tile_cams:
                    tile_cams[key] = shift_principal_point(
                        cam, next_cam_id,
                        *tile_bounds(cam.width, cam.height, cols, rows_n, tcc, trr))
                    next_cam_id += 1
                needed.setdefault(im.name, set()).add((trr, tcc))
            members.append((im, tiles))
        if len(members) < min_images:
            skipped.append({"name": f"block_{ix}_{iy}", "images": len(members)})
            continue
        blocks.append({"ix": ix, "iy": iy,
                       "bounds": [bx0, by0, bx1, by1],
                       "members": members, "pmask": inb})

    for s in skipped:
        r.log(f"[blocksplit] skip {s['name']}: {s['images']} images < min_images={min_images}")
    if not blocks:
        raise ValueError(
            "沒有任何分塊達到 min_images 門檻 — 請確認 region/block_size 與模型單位"
            "（未做 GPS 對齊時模型單位不是公尺），或調低 min_images。")
    r.log(f"[blocksplit] kept {len(blocks)} blocks")

    # -- crop the shared tile pool (tiling mode) ------------------------------ #
    pool = out / "_tiles"
    if tile and needed:
        import cv2  # heavy/optional dep, house style: import where used

        def _crop_one(name: str, cells_: set[tuple[int, int]]) -> str:
            # resume: skip tiles already cropped (the param stamp guarantees a
            # leftover tile has the same geometry/quality); fully-done images
            # never even decode the source.
            def _done(p: Path) -> bool:
                try:
                    return p.stat().st_size > 0
                except OSError:
                    return False
            todo = [t for t in sorted(cells_)
                    if not _done(pool / tile_name(name, *t))]
            if not todo:
                return name
            img = cv2.imread(str(images_dir / name), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"影像讀取失敗：{images_dir / name}")
            h, w = img.shape[:2]
            cam = cameras[images_by_name[name].camera_id]
            if (w, h) != (cam.width, cam.height):
                raise ValueError(
                    f"{name} 的實際尺寸 {w}x{h} 與模型登記的 {cam.width}x{cam.height} 不符 —"
                    " 影像與 sparse 模型不是同一套，請確認 source。")
            cols, rows_n = tile_layout(w, h, max_tile_px)
            for trr, tcc in todo:
                x0, y0, x1, y1 = tile_bounds(w, h, cols, rows_n, tcc, trr)
                dst = pool / tile_name(name, trr, tcc)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(dst), img[y0:y1, x0:x1],
                                   [cv2.IMWRITE_JPEG_QUALITY, jpeg_q]):
                    raise ValueError(f"切片寫入失敗：{dst}")
            return name

        images_by_name = {im.name: im for im in images.values()}
        r.banner(f"blocksplit | cropping {sum(len(v) for v in needed.values())} tiles "
                 f"from {len(needed)} images (workers={workers})")
        done = 0
        with futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_crop_one, n, c): n for n, c in needed.items()}
            for fut in futures.as_completed(futs):
                r.check_cancel()
                name = fut.result()          # re-raises worker errors with context
                done += 1
                if done % 10 == 0 or done == len(needed):
                    r.log(f"[blocksplit] tiles [{done}/{len(needed)}] {name}")

    # -- write the per-block scenes ------------------------------------------ #
    summary = []
    next_img_id = 1
    for b in blocks:
        r.check_cancel()
        bdir = out / f"block_{b['ix']}_{b['iy']}"
        (bdir / "sparse").mkdir(parents=True, exist_ok=True)
        (bdir / "images").mkdir(parents=True, exist_ok=True)

        b_cams: dict[int, Camera] = {}
        b_imgs: dict[int, Image] = {}
        trk_img: dict[int, list[int]] = {}   # point row → view ids observing it
        trk_idx: dict[int, list[int]] = {}   # …and the obs index inside that view
        n_tiles = 0

        # default args bind THIS block's dicts at def-time (the closure is called
        # only within this iteration; the binding just satisfies bugbear B023).
        def _add_view(vid, src, cam_id, name, vxy, vrows,
                      _imgs=b_imgs, _trk_img=trk_img, _trk_idx=trk_idx):
            """Emit one view with its observations and grow the points' tracks."""
            _imgs[vid] = Image(id=vid, qvec=src.qvec, tvec=src.tvec,
                               camera_id=cam_id, name=name,
                               xys=np.asarray(vxy, np.float64),
                               point3D_ids=pid[vrows].astype(np.int64))
            for k, j in enumerate(vrows):
                _trk_img.setdefault(int(j), []).append(vid)
                _trk_idx.setdefault(int(j), []).append(k)

        for im, tiles in b["members"]:
            rows_im, xys_im = obs[im.id]
            hit = b["pmask"][rows_im]               # this view's obs of block points
            sub_rows, sub_xy = rows_im[hit], xys_im[hit]
            if tiles is None:                       # no tiling: link the original
                b_cams[im.camera_id] = cameras[im.camera_id]
                _add_view(im.id, im, im.camera_id, im.name, sub_xy, sub_rows)
                link = bdir / "images" / im.name
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.is_symlink():
                    link.symlink_to((images_dir / im.name).resolve())
                continue
            cam = cameras[im.camera_id]
            cols, rows_n = tile_layout(cam.width, cam.height, max_tile_px)
            for trr, tcc in tiles:
                cam_t = tile_cams[(im.camera_id, trr, tcc)]
                b_cams[cam_t.id] = cam_t
                x0, y0, x1, y1 = tile_bounds(cam.width, cam.height,
                                             cols, rows_n, tcc, trr)
                in_t = ((sub_xy[:, 0] >= x0) & (sub_xy[:, 0] < x1) &
                        (sub_xy[:, 1] >= y0) & (sub_xy[:, 1] < y1))
                nm = tile_name(im.name, trr, tcc)
                _add_view(next_img_id, im, cam_t.id, nm,
                          sub_xy[in_t] - (x0, y0), sub_rows[in_t])
                next_img_id += 1
                n_tiles += 1
                link = bdir / "images" / nm
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.is_symlink():
                    # relative: the pool lives inside out_dir, so the whole
                    # output tree stays valid when moved/copied elsewhere
                    link.symlink_to(os.path.relpath((pool / nm).resolve(),
                                                    link.parent.resolve()))

        rows = np.nonzero(b["pmask"])[0]
        b_pts = {}
        for j in rows:
            jj = int(j)
            b_pts[int(pid[jj])] = Point3D(
                id=int(pid[jj]), xyz=xyz[jj], rgb=rgb[jj], error=err[jj],
                image_ids=np.asarray(trk_img.get(jj, ()), dtype=np.int64),
                point2D_idxs=np.asarray(trk_idx.get(jj, ()), dtype=np.int64))
        write_cameras_binary(b_cams, str(bdir / "sparse" / "cameras.bin"))
        write_images_binary(b_imgs, str(bdir / "sparse" / "images.bin"))
        write_points3D_binary(b_pts, str(bdir / "sparse" / "points3D.bin"))

        # total training pixels: the per-block VRAM/wall-clock driver — the
        # number users need when tuning block_size.
        px = sum(b_cams[v.camera_id].width * b_cams[v.camera_id].height
                 for v in b_imgs.values())
        info = {"name": bdir.name, "bounds": b["bounds"],
                "src_images": len(b["members"]), "train_views": len(b_imgs),
                "points": len(b_pts), "pixels": px}
        summary.append(info)
        r.log(f"[blocksplit] {bdir.name}: {info['src_images']} src images → "
              f"{info['train_views']} views, {info['points']} pts, "
              f"{px / 1e6:.1f} MP"
              + (f" ({n_tiles} tiles)" if tile else ""))

    # which blocks each source image landed in — debugging aid for coverage
    # questions ("why is this photo missing from block X?")
    img_blocks: dict[str, list[str]] = {}
    for b in blocks:
        for im, _tiles in b["members"]:
            img_blocks.setdefault(im.name, []).append(f"block_{b['ix']}_{b['iy']}")
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(src), "sparse": str(sparse_dir), "images": str(images_dir),
        "params": {k: p.get(k) for k in BLOCKSPLIT_DEFAULTS} | {"region": list(region)},
        "blocks": summary, "skipped": skipped, "image_blocks": img_blocks,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    r.banner(f"blocksplit done | {len(blocks)} blocks → {out}")
    for info in summary:
        r.log(f"[blocksplit] train source: {out / info['name']}")
    if tile:
        r.log("[blocksplit] 切片已是原生解析度;LichtFeld 訓練時用 --max-width 0 "
              "(extra 欄位) 以免再被縮到 3840。")
    r.log("[blocksplit] 訓練後每塊先裁掉 buffer 區的 splat 再合併（塊間同一座標系,直接 concat）。")
