#!/usr/bin/env python3
"""
POC: Estimate scale factor from a ChArUco board observed in a COLMAP
reconstruction. Reads sparse/0 + undistorted images, detects the board,
triangulates inner corners in the reconstruction frame, and compares to the
known physical board geometry.

Prints the scale factor and consistency stats. Does not modify any files.

Typical usage (board is 9x6 inner squares, 28 mm squares, 21 mm markers):
        python tools/estimate_marker_scale.py  
        --sparse ../../tmp/0423_iron/training_dataset/sparse/0/ 
        --images ../../tmp/0423_iron/training_dataset/images/ 
        --squares-x 9 --squares-y 6  
        --square-mm 28.806 --marker-mm 21.12
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_read_write_model import (  # noqa: E402
    read_cameras_binary,
    read_images_binary,
    qvec2rotmat,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sparse", required=True, type=Path, help="Path to COLMAP sparse/0 dir")
    p.add_argument("--images", required=True, type=Path, help="Path to undistorted images dir")
    p.add_argument("--squares-x", type=int, required=True, help="Number of squares along X (columns)")
    p.add_argument("--squares-y", type=int, required=True, help="Number of squares along Y (rows)")
    p.add_argument("--square-mm", type=float, required=True, help="Measured square edge length (mm)")
    p.add_argument("--marker-mm", type=float, required=True, help="Measured marker edge length (mm)")
    p.add_argument("--dict", default="DICT_5X5_100", dest="dict_name", help="ArUco dictionary name")
    p.add_argument("--reproj-thresh-px", type=float, default=2.0)
    p.add_argument("--min-views", type=int, default=3, help="Min observations per corner for triangulation")
    p.add_argument("--max-neighbor-dist", type=int, default=2, help="Use corner pairs within this grid hop")
    p.add_argument("--debug-dir", type=Path, default=None, help="If set, write detection overlays here")
    p.add_argument("--out-json", type=Path, default=None,
                   help="If set, write the scale result (mm_per_unit, stats) as JSON here")
    return p.parse_args()


def load_dictionary(name: str):
    const = getattr(aruco, name, None)
    if const is None:
        raise SystemExit(f"Unknown ArUco dictionary: {name}")
    return aruco.getPredefinedDictionary(const)


def detect_charuco_all(
    images_dir: Path,
    image_names: list[str],
    board: aruco.CharucoBoard,
    debug_dir: Path | None,
) -> dict[str, dict[int, tuple[float, float]]]:
    """Run ChArUco detection on every image. Returns {image_name: {corner_id: (u, v)}}."""
    detector = aruco.CharucoDetector(board)
    detections: dict[str, dict[int, tuple[float, float]]] = {}
    missing = 0
    for name in image_names:
        path = images_dir / name
        if not path.exists():
            missing += 1
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            missing += 1
            continue
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(img)
        if charuco_ids is None or len(charuco_ids) < 4:
            continue
        entry: dict[int, tuple[float, float]] = {}
        for cid, pt in zip(charuco_ids.flatten(), charuco_corners.reshape(-1, 2)):
            entry[int(cid)] = (float(pt[0]), float(pt[1]))
        detections[name] = entry
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
            cv2.imwrite(str(debug_dir / name), vis)
    if missing:
        print(f"  warning: {missing} images listed in sparse but not found on disk")
    return detections


def build_projection_matrices(sparse_dir: Path) -> dict[str, np.ndarray]:
    """Return {image_name: P (3x4)} in world-to-camera convention. Requires PINHOLE cameras."""
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    images = read_images_binary(sparse_dir / "images.bin")

    proj: dict[str, np.ndarray] = {}
    for img in images.values():
        cam = cameras[img.camera_id]
        model = cam.model
        params = cam.params
        if model == "PINHOLE":
            fx, fy, cx, cy = params
        elif model == "SIMPLE_PINHOLE":
            f, cx, cy = params
            fx = fy = f
        else:
            raise SystemExit(
                f"Camera model {model!r} not supported. Use undistorted images "
                f"(COLMAP writes PINHOLE after image_undistorter)."
            )
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        R = qvec2rotmat(img.qvec)
        t = np.asarray(img.tvec, dtype=np.float64).reshape(3)
        Rt = np.hstack([R, t.reshape(3, 1)])
        proj[img.name] = K @ Rt
    return proj


def triangulate_dlt(observations: list[tuple[np.ndarray, tuple[float, float]]]) -> tuple[np.ndarray, list[float]]:
    """Multi-view linear triangulation. Returns (X_world, per-view reproj errors in px)."""
    rows = []
    for P, (u, v) in observations:
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.stack(rows)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    if abs(X_h[3]) < 1e-12:
        return np.array([np.nan] * 3), [float("inf")] * len(observations)
    X = X_h[:3] / X_h[3]
    errs = []
    for P, (u, v) in observations:
        xh = P @ np.append(X, 1.0)
        if abs(xh[2]) < 1e-12:
            errs.append(float("inf"))
            continue
        up, vp = xh[0] / xh[2], xh[1] / xh[2]
        errs.append(float(np.hypot(up - u, vp - v)))
    return X, errs


def corner_grid_pos(cid: int, squares_x: int) -> tuple[int, int]:
    """ChArUco inner-corner id -> (i, j) in the inner grid (row-major, i=col, j=row)."""
    inner_x = squares_x - 1
    return cid % inner_x, cid // inner_x


def main() -> None:
    args = parse_args()

    dictionary = load_dictionary(args.dict_name)
    board = aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_mm / 1000.0,
        args.marker_mm / 1000.0,
        dictionary,
    )

    print(f"Loading COLMAP sparse model from {args.sparse}")
    proj = build_projection_matrices(args.sparse)
    print(f"  {len(proj)} registered images")
    image_names = sorted(proj.keys())

    print(f"Detecting ChArUco in {args.images} ...")
    detections = detect_charuco_all(args.images, image_names, board, args.debug_dir)
    print(f"  detected in {len(detections)} / {len(image_names)} images")
    if len(detections) < 2:
        raise SystemExit("ERROR: need board detected in >= 2 registered images.")

    # Per-corner observations
    corner_obs: dict[int, list[tuple[np.ndarray, tuple[float, float], str]]] = defaultdict(list)
    for img_name, corners in detections.items():
        P = proj[img_name]
        for cid, uv in corners.items():
            corner_obs[cid].append((P, uv, img_name))
    print(f"  {len(corner_obs)} unique corner IDs across all images")

    # Triangulate and filter by reprojection error
    corners_3d: dict[int, np.ndarray] = {}
    dropped_few_views = 0
    dropped_reproj = 0
    for cid, obs in corner_obs.items():
        if len(obs) < args.min_views:
            dropped_few_views += 1
            continue
        obs_xy = [(P, uv) for P, uv, _ in obs]
        X, errs = triangulate_dlt(obs_xy)
        med_err = float(np.median(errs))
        if not np.isfinite(med_err) or med_err > args.reproj_thresh_px:
            dropped_reproj += 1
            continue
        corners_3d[cid] = X
    print(
        f"  triangulated {len(corners_3d)} corners "
        f"(dropped: {dropped_few_views} few-view, {dropped_reproj} reproj)"
    )
    if len(corners_3d) < 4:
        raise SystemExit("ERROR: fewer than 4 usable corners; cannot estimate scale.")

    # Gather adjacent-corner scale samples
    inner_x = args.squares_x - 1
    inner_y = args.squares_y - 1

    def pos(c: int) -> tuple[int, int]:
        return c % inner_x, c // inner_x

    scales = []
    ids = sorted(corners_3d.keys())
    for a_idx, cid_a in enumerate(ids):
        ia, ja = pos(cid_a)
        if ia >= inner_x or ja >= inner_y:
            continue
        Xa = corners_3d[cid_a]
        for cid_b in ids[a_idx + 1:]:
            ib, jb = pos(cid_b)
            di, dj = ib - ia, jb - ja
            if abs(di) > args.max_neighbor_dist or abs(dj) > args.max_neighbor_dist:
                continue
            if di == 0 and dj == 0:
                continue
            real_mm = args.square_mm * float(np.hypot(di, dj))
            recon_d = float(np.linalg.norm(Xa - corners_3d[cid_b]))
            if recon_d < 1e-9:
                continue
            scales.append(real_mm / recon_d)
    if not scales:
        raise SystemExit("ERROR: no valid corner pairs; cannot estimate scale.")

    scales_arr = np.asarray(scales)
    median_scale = float(np.median(scales_arr))
    mad = float(np.median(np.abs(scales_arr - median_scale)))
    rel_mad_pct = (mad / median_scale * 100.0) if median_scale > 0 else float("nan")

    print()
    print("=" * 60)
    print("SCALE ESTIMATION RESULT")
    print("=" * 60)
    print(f"  pairs used:      {len(scales_arr)}")
    print(f"  scale (median):  {median_scale:.6f}  mm per recon unit")
    print(f"  scale (mean):    {scales_arr.mean():.6f}")
    print(f"  MAD:             {mad:.6f}  ({rel_mad_pct:.3f} %)")
    print(f"  min / max:       {scales_arr.min():.6f}  /  {scales_arr.max():.6f}")
    print()
    if rel_mad_pct > 5.0:
        print("  WARNING: rel MAD > 5 %. Check board print accuracy, rigidity,")
        print("  camera undistortion, and that the board wasn't moved during capture.")
    else:
        print("  Consistency OK (rel MAD < 5 %).")
    print()
    print(f"  1.0 recon unit  =  {median_scale:.3f} mm  =  {median_scale / 10.0:.3f} cm")
    print(f"  To metric metres, multiply recon coords by {median_scale / 1000.0:.6f}")

    if args.out_json is not None:
        import json
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "mm_per_unit": median_scale,
            "mean": float(scales_arr.mean()),
            "mad": mad,
            "rel_mad_pct": rel_mad_pct,
            "pairs": int(len(scales_arr)),
        }, indent=2))
        print(f"  wrote scale JSON: {args.out_json}")


if __name__ == "__main__":
    main()
