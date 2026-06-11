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

Block X/Y are the model's X/Y axes. With GPS_ALIGN (enu) those are metres, so
block_size=500 is a 500 m square; without metric alignment pick sizes in model
units. Assignment is observation-driven (no footprint geometry): an image joins
a block when ≥ min_obs of its tracked 3D points fall inside the block(+buffer);
a tile of that image is kept when ≥ min_tile_obs of those features land in it
(when no tile reaches that, only the busiest tile is kept — spraying every
touched tile into the block would add near-empty views).
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
    read_model,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)

BLOCKSPLIT_DEFAULTS = {
    "block_size": "500", "buffer": "120", "region": "",
    "tile": True, "max_tile_px": "8192", "jpeg_quality": "95",
    "min_obs": "30", "min_tile_obs": "8", "min_images": "15", "workers": "4",
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
    min_obs = int(p.get("min_obs") or BLOCKSPLIT_DEFAULTS["min_obs"])
    min_tile_obs = int(p.get("min_tile_obs") or BLOCKSPLIT_DEFAULTS["min_tile_obs"])
    min_images = int(p.get("min_images") or BLOCKSPLIT_DEFAULTS["min_images"])
    workers = int(p.get("workers") or BLOCKSPLIT_DEFAULTS["workers"])
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

    sparse_dir, images_dir = _resolve_dense(src)
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

    if region is None:
        minx, maxx = np.percentile(X, [1, 99])
        miny, maxy = np.percentile(Y, [1, 99])
        region = (float(minx), float(miny), float(maxx), float(maxy))
        r.log(f"[blocksplit] region auto (1–99% of points): "
              f"{region[0]:.1f},{region[1]:.1f} → {region[2]:.1f},{region[3]:.1f}")
    cells = grid_cells(*region, block_size)
    r.log(f"[blocksplit] grid: block_size={block_size:g} buffer={buffer:g} "
          f"→ {len(cells)} candidate cells")

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
            if int(hit.sum()) < min_obs:
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
