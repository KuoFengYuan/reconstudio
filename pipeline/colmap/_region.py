"""Region → image subset, for re-running COLMAP over an area picked in the viewer.

`select_region_images` answers one question: when the user boxes a region and asks
to re-run the pipeline on it, which images belong in the run? It reuses
blocksplit's REUrbanGS two-phase rule — camera centre inside the rectangle, OR the
image sees enough of the region's points (hull-area ratio V_ij/V_i ≥ vis_thresh) —
so the run keeps the oblique heads that look *into* the region from outside it, not
just the ones that fly over it. Selecting on camera centres alone would drop
exactly the multi-view coverage the reconstruction depends on (on the 20251223
five-head rig that is 38 of 193 images).

Unlike blocksplit this crops nothing and writes no model: it returns image names,
which `_build_image_list` intersects with the run's image list. Every stage
downstream (extract / match / mapper / align / undistort) then operates on the
subset with no further changes.

The rectangle is measured on the reference model's OWN horizontal axes — the same
plane the viewer drew it on and the same pair `region_stats` reports — so the
"框內相機 N/M" the panel showed is the phase-1 count here. `buffer` (外擴) grows
only the point mask that feeds the visibility test, mirroring blocksplit, where
phase 1 always tests the core bounds.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..blocksplit import hull_area, parse_region
from ..model import horiz_axes, up_axis
from ..vendor.read_write_model import qvec2rotmat, read_model

__all__ = ["parse_region", "select_region_images"]

VIS_THRESH_DEFAULT = 1.0 / 6.0          # REUrbanGS visibility ratio


def select_region_images(
    model_dir: Path,
    region: tuple[float, float, float, float],
    buffer: float = 0.0,
    vis_thresh: float = VIS_THRESH_DEFAULT,
    log: Callable[[str], None] | None = None,
) -> tuple[list[str], dict]:
    """Image names a COLMAP re-run over `region` should include, read from `model_dir`.

    Returns `(names, stats)`; `stats` carries the counts worth logging — how many
    images came in on the camera-centre test vs the visibility test, and the point
    totals — so the caller can show the user why the subset is the size it is.
    """
    def say(msg: str) -> None:
        if log:
            log(msg)

    _cameras, images, points3D = read_model(str(model_dir))
    if not images:
        raise FileNotFoundError(f"參考模型沒有註冊影像: {model_dir}")
    if not points3D:
        raise FileNotFoundError(f"參考模型沒有 3D 點,無法做可見度選片: {model_dir}")

    up = up_axis(model_dir)
    a0, a1 = horiz_axes(up) if up is not None else (0, 1)
    minx, miny, maxx, maxy = region

    # Flatten points into sorted arrays so each image's observations become one
    # searchsorted lookup instead of a dict probe per point (blocksplit does the
    # same; on 228k points × 193 images the difference is seconds vs minutes).
    pid = np.fromiter(points3D.keys(), dtype=np.int64, count=len(points3D))
    pid.sort()
    xyz = np.empty((pid.size, 3), np.float64)
    for i, k in enumerate(pid):
        xyz[i] = points3D[k].xyz

    inb = ((minx - buffer <= xyz[:, a0]) & (xyz[:, a0] <= maxx + buffer) &
           (miny - buffer <= xyz[:, a1]) & (xyz[:, a1] <= maxy + buffer))
    if not inb.any():
        raise ValueError(
            f"框內沒有任何 3D 點（region={minx:g},{miny:g},{maxx:g},{maxy:g}"
            f"{f', 外擴={buffer:g}' if buffer else ''}）— "
            "請確認這個範圍是在 region_model 的座標系上框的。")

    by_center: list[str] = []
    by_vis: list[str] = []
    for _iid, im in images.items():
        ids3 = np.asarray(im.point3D_ids, np.int64)
        m = ids3 >= 0
        ids3 = ids3[m]
        pos = np.searchsorted(pid, ids3)
        ok = (pos < pid.size) & (pid[np.minimum(pos, pid.size - 1)] == ids3)
        rows, xys2 = pos[ok], np.asarray(im.xys, np.float64)[m][ok]

        # phase 1 — camera centre in the core rectangle. Closed interval, matching
        # model.region_stats, so this count equals the panel's 「框內相機」.
        centre = -qvec2rotmat(im.qvec).T @ im.tvec
        if minx <= centre[a0] <= maxx and miny <= centre[a1] <= maxy:
            by_center.append(im.name)
            continue
        # phase 2 — how much of what this image sees lands in the region. An image
        # seeing < 3 in-region points gives a degenerate hull (0.0) and is dropped.
        vi = hull_area(xys2)
        if vi > 0.0 and hull_area(xys2[inb[rows]]) / vi >= vis_thresh:
            by_vis.append(im.name)

    names = sorted(set(by_center) | set(by_vis))
    stats = {
        "kept": len(names), "total": len(images),
        "by_center": len(by_center), "by_visibility": len(by_vis),
        "points_in": int(inb.sum()), "points_total": int(pid.size),
        "axes": (a0, a1), "up_axis": up,
    }
    say(f"region 選片: {len(names)}/{len(images)} 張影像"
        f"（框內相機 {len(by_center)} + 可見度 V_ij/V_i ≥ {vis_thresh:g} 另收 {len(by_vis)}）,"
        f"框內點 {stats['points_in']:,}/{stats['points_total']:,},"
        f"量測軸 {'XYZ'[a0]}-{'XYZ'[a1]}")
    if not names:
        raise ValueError(
            "region 選片結果是空的 — 範圍內沒有相機,可見度也都低於門檻。"
            "請放大範圍、調高外擴,或降低 vis_thresh。")
    return names, stats
