#!/usr/bin/env python3
"""Remove images from a COLMAP sparse model, by camera_id or by name substring.

Why this exists: a stray sensor in the input set gets its own intrinsic group and
poisons downstream work. The trigger case was DJI M3M *stitched panoramas*
(14400x7200, 2:1) mixed in with the regular 4:3 frames -- COLMAP fits them a
pinhole+OPENCV model (f ~ half the real frames', radial -0.28) that is physically
meaningless, and they carry no RTK EXIF so they also have no usable GPS prior.

Removing an image is not just dropping its row: every 3D point that observed it
must lose that track element, and a point left with < 2 observations is no longer
triangulable and must go too (COLMAP's own invariant). Cameras that end up with
no images are dropped as well, so `cameras.bin` stops advertising the bogus
intrinsics.

The model is rewritten in place ONLY after a full backup copy is made next to it
(<model>.bak-<n>), so a mistake is always recoverable. Use --output to write
elsewhere and leave the input untouched.

Usage:
    python tools/drop_images.py --model .../sparse/0 --camera-id 3
    python tools/drop_images.py --model .../sparse/0 --name-contains _0147_ --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.vendor.read_write_model import (  # noqa: E402
    Point3D,
    read_model,
    write_model,
)

_MIN_TRACK_LEN = 2          # COLMAP drops points observed fewer than twice


def drop(model_dir: Path, camera_ids: set[int], name_parts: list[str],
         out_dir: Path | None, dry_run: bool) -> dict:
    cameras, images, points3D = read_model(str(model_dir))

    doomed = {
        iid for iid, im in images.items()
        if im.camera_id in camera_ids or any(p in im.name for p in name_parts)
    }
    kept_images = {iid: im for iid, im in images.items() if iid not in doomed}

    # Orphan intrinsics are dropped even when no image matched: a camera row that
    # no image uses is exactly the bogus-sensor metadata we are here to remove
    # (it still shows up in viewers' "cameras" count and in downstream exports).
    orphans = set(cameras) - {im.camera_id for im in kept_images.values()}
    if not doomed and not orphans:
        return {"removed_images": 0, "note": "nothing matched, no orphan cameras"}

    # Prune the doomed images out of every track, then drop points that fall
    # below the minimum track length (a 1-observation point is not triangulated).
    # image_ids / point2D_idxs are numpy arrays, so a boolean mask keeps types intact.
    new_points: dict[int, Point3D] = {}
    n_pts_dropped = 0
    n_obs_dropped = 0
    for pid, pt in points3D.items():
        mask = np.array([iid not in doomed for iid in pt.image_ids], dtype=bool)
        n_kept = int(mask.sum())
        n_obs_dropped += len(pt.image_ids) - n_kept
        if n_kept < _MIN_TRACK_LEN:
            n_pts_dropped += 1
            continue
        new_points[pid] = pt._replace(image_ids=pt.image_ids[mask],
                                      point2D_idxs=pt.point2D_idxs[mask])

    # A kept image's point3D_ids must not reference points we just deleted, or
    # the model is internally inconsistent (COLMAP would read a dangling id).
    for iid, im in list(kept_images.items()):
        ids = np.asarray(im.point3D_ids)
        stale = (ids != -1) & ~np.isin(ids, list(new_points))
        if stale.any():
            fixed = ids.copy()
            fixed[stale] = -1
            kept_images[iid] = im._replace(point3D_ids=fixed)

    kept_cams = {cid: c for cid, c in cameras.items() if cid not in orphans}

    stats = {
        "removed_images": len(doomed),
        "removed_image_names": sorted(images[i].name for i in doomed),
        "kept_images": len(kept_images),
        "removed_cameras": sorted(orphans),
        "kept_cameras": sorted(kept_cams),
        "points_before": len(points3D),
        "points_after": len(new_points),
        "points_dropped": n_pts_dropped,
        "observations_dropped": n_obs_dropped,
    }
    if dry_run:
        stats["note"] = "dry run — nothing written"
        return stats

    if out_dir is None:                     # in place: back up first, always
        n = 0
        while (bak := model_dir.with_name(model_dir.name + f".bak-{n}")).exists():
            n += 1
        shutil.copytree(model_dir, bak)
        stats["backup"] = str(bak)
        out_dir = model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_model(kept_cams, kept_images, new_points, str(out_dir), ext=".bin")
    stats["written"] = str(out_dir)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path,
                    help="COLMAP sparse model dir (holds cameras/images/points3D.bin)")
    ap.add_argument("--camera-id", type=int, action="append", default=[],
                    help="drop every image using this camera_id (repeatable)")
    ap.add_argument("--name-contains", action="append", default=[],
                    help="drop images whose name contains this substring (repeatable)")
    ap.add_argument("--output", type=Path,
                    help="write here instead of rewriting --model in place")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    a = ap.parse_args()
    if not a.camera_id and not a.name_contains:
        ap.error("give at least one of --camera-id / --name-contains")

    stats = drop(a.model, set(a.camera_id), a.name_contains, a.output, a.dry_run)
    for k, v in stats.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
