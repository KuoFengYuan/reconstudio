#!/usr/bin/env python3
"""Measure how far a reconstructed model's camera centres sit from the DB pose priors.

This is the feedback loop for choosing PRIOR_STD_X/Y/Z. Set the stds loose for a
first run, measure here, then set them to the RMS this reports — the priors then
carry the weight the data actually supports instead of a guessed one.

It fits a similarity (Umeyama, scale+R+t) from the model's camera centres to the
priors, then reports the per-axis residual in the prior's own metric frame. The
fit is needed because a plain global/incremental model is in an arbitrary gauge;
for a pose_prior_mapper model the fit comes out at scale≈1 with no rotation, and
the residual is then the honest prior-vs-photogrammetry disagreement.

    python tools/check_pose_prior_residuals.py --sparse WS/sparse/0 --database WS/database.db

Large residuals mean one of: a wrong POSE_PRIOR_CRS, an ω/φ/κ convention problem
(only if gravity was used), a genuinely worse-than-advertised EO, or a bad
reconstruction. The `--top` list of worst images tells you which.
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.vendor.read_write_model import qvec2rotmat, read_images_binary  # noqa: E402

# COLMAP's PosePrior::CoordinateSystem
WGS84, CARTESIAN = 0, 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sparse", required=True, type=Path,
                   help="Model dir holding images.bin (e.g. WS/sparse/0)")
    p.add_argument("--database", required=True, type=Path,
                   help="database.db carrying the pose_priors to compare against")
    p.add_argument("--top", type=int, default=10,
                   help="How many worst-residual images to list (default 10)")
    return p.parse_args()


def read_priors(db_path: Path) -> tuple[dict[int, np.ndarray], int]:
    """image_id -> position, plus the single coordinate_system in use."""
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT corr_data_id, position, coordinate_system FROM pose_priors").fetchall()
    finally:
        con.close()
    out: dict[int, np.ndarray] = {}
    systems = set()
    for data_id, pos, cs in rows:
        xyz = np.array(struct.unpack("<3d", pos))
        if not np.isfinite(xyz).all():
            continue
        out[data_id] = xyz
        systems.add(cs)
    if len(systems) > 1:
        raise SystemExit(f"DB mixes pose-prior coordinate systems {sorted(systems)}; "
                         "COLMAP itself refuses this — re-run the inject.")
    return out, (systems.pop() if systems else CARTESIAN)


def wgs84_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    """WGS84 ellipsoidal -> geocentric metres, so lat/lon priors can be differenced."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    la, lo = math.radians(lat), math.radians(lon)
    n = a / math.sqrt(1 - e2 * math.sin(la) ** 2)
    return np.array([(n + alt) * math.cos(la) * math.cos(lo),
                     (n + alt) * math.cos(la) * math.sin(lo),
                     (n * (1 - e2) + alt) * math.sin(la)])


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity src -> dst (scale, rotation, translation)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    u, sv, vt = np.linalg.svd(d0.T @ s0 / len(src))
    d = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2, 2] = -1
    rot = u @ d @ vt
    scale = float(sv @ np.diag(d).T / (s0 ** 2).sum() * len(src))
    return scale, rot, mu_d - scale * rot @ mu_s


def main() -> None:
    args = parse_args()
    images = read_images_binary(str(args.sparse / "images.bin"))
    priors, coord_sys = read_priors(args.database)

    names, centres, refs = [], [], []
    for image_id, im in images.items():
        prior = priors.get(image_id)
        if prior is None:
            continue
        # camera centre in world = -Rᵀ t
        centres.append(-qvec2rotmat(im.qvec).T @ im.tvec)
        refs.append(wgs84_to_ecef(*prior) if coord_sys == WGS84 else prior)
        names.append(im.name)

    if len(names) < 3:
        raise SystemExit(f"only {len(names)} registered image(s) have a pose prior — "
                         "nothing to fit against")

    src, dst = np.asarray(centres), np.asarray(refs)
    scale, rot, trans = umeyama(src, dst)
    resid = (scale * (rot @ src.T).T + trans) - dst
    dist = np.linalg.norm(resid, axis=1)

    print(f"registered images with a prior : {len(names)} / {len(images)} registered, "
          f"{len(priors)} priors in DB")
    print(f"coordinate system              : "
          f"{'WGS84 (compared in ECEF metres)' if coord_sys == WGS84 else 'CARTESIAN'}")
    print(f"fitted similarity scale        : {scale:.6f}"
          + ("   <- ~1.0 expected for a pose_prior model" if 0.9 < scale < 1.1 else
             "   <- far from 1: model is in an arbitrary gauge (global/incremental)"))
    print()
    print("residual (model -> priors), metres")
    print(f"  RMS per axis   X {np.sqrt((resid[:, 0]**2).mean()):7.3f}"
          f"   Y {np.sqrt((resid[:, 1]**2).mean()):7.3f}"
          f"   Z {np.sqrt((resid[:, 2]**2).mean()):7.3f}")
    print(f"  3D  RMS {np.sqrt((dist**2).mean()):7.3f}   median {np.median(dist):7.3f}"
          f"   max {dist.max():7.3f}")
    print()
    print("-> a sensible PRIOR_STD_X/Y/Z is roughly the per-axis RMS above "
          "(round up, don't go below it)")
    if args.top:
        print(f"\nworst {min(args.top, len(names))} images:")
        for i in np.argsort(-dist)[:args.top]:
            print(f"  {dist[i]:8.3f} m  {names[i]}")


if __name__ == "__main__":
    main()
