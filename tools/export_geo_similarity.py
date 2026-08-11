#!/usr/bin/env python3
"""
Export a COLMAP model's GPS alignment as a similarity-transform JSON for the
geo-register-plugin's "Similarity File" source mode
(https://github.com/dozeri83/geo-register-plugin):

    world_ecef = scale * R @ p_scene + translation

Runs `colmap model_aligner --alignment_type ecef` against the EXACT sparse
model whose camera poses match what was loaded into the splat viewer (usually
the undistorted --- and, if REORIENT ran, reoriented --- `sparse/` dir, NOT the
pre-undistort `sparse/0`), then decodes COLMAP's Sim3 transform file
("scale qw qx qy qz tx ty tz", see colmap/geometry/sim3.cc Sim3d::ToFile) into
the plugin's {scale, rotation, translation} JSON.

Axis flip: the plugin evaluates this transform against its own "visualizer
world" points (LichtFeld's pick_at_screen / splat export), which it gets by
negating Y and Z of the raw COLMAP dataset-world coordinates (see
geo/camera_reader.py's "Convert raw dataset world (Y-down, Z-forward) ->
visualizer world (Y-up, Z-backward)" and the same negation in
geo/las_exporter.py before it applies scale*R+t). Our R is fit directly
against COLMAP's raw (unflipped) images.bin, so it must be re-expressed for
that flipped input: p_ecef = s*R*p_raw + t = s*R*diag(1,-1,-1)*p_vis + t,
i.e. negate columns 2 and 3 of R. Skipping this puts the picked/exported
points' Y (COLMAP's "up" axis, since ReconStudio's reorient makes local Y
vertical) in with the wrong sign, which shows up almost entirely as an
altitude error (~2x the point's height above/below the GPS-align ENU origin)
since R's second column is close to the local vertical direction.

Requires `database.db` to already carry a GPS pose_prior for every image in
the target model (COLMAP fills these from JPEG EXIF at feature_extractor time;
ReconStudio's own gps_inject stage backfills TIFF/PNG). If model_aligner
reports fewer reference images than expected, re-run the panel's pipeline with
the gps_inject stage (FORCE=1) first.

Usage:
    python tools/export_geo_similarity.py \\
        --sparse /Disk0/x/colmap_1920/training_dataset_incremental_mapper/sparse \\
        --database /Disk0/x/colmap_1920/database.db \\
        --out /Disk0/x/similarity_transform.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sparse", required=True, type=Path,
                   help="Path to the COLMAP sparse model dir (cameras.bin/images.bin) "
                        "matching the poses actually used for the splat/viewer scene")
    p.add_argument("--database", required=True, type=Path, help="Path to database.db (needs GPS pose_priors)")
    p.add_argument("--colmap-bin", default="colmap", help="colmap executable (default: colmap)")
    p.add_argument("--alignment-type", default="ecef", choices=["ecef", "enu"],
                   help="Target frame for the fit (default: ecef, what the plugin expects)")
    p.add_argument("--max-error", type=float, default=5.0,
                   help="model_aligner --alignment_max_error in metres (default: 5.0; "
                        "raise this for noisy consumer-GPS EXIF)")
    p.add_argument("--out", required=True, type=Path, help="Output similarity_transform.json path")
    return p.parse_args()


def quat_wxyz_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> list[list[float]]:
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def flip_yz_columns(rotation: list[list[float]]) -> list[list[float]]:
    """Re-express R for the plugin's Y/Z-negated 'visualizer world' input:
    R_vis = R @ diag(1,-1,-1), i.e. negate columns 2 and 3 (see module docstring)."""
    return [[r[0], -r[1], -r[2]] for r in rotation]


def main() -> None:
    args = parse_args()
    if not (args.sparse / "cameras.bin").is_file():
        raise SystemExit(f"no cameras.bin under {args.sparse} (point --sparse at the model "
                          "actually used for the splat/viewer scene)")
    if not args.database.is_file():
        raise SystemExit(f"database not found: {args.database}")
    if not shutil.which(args.colmap_bin):
        raise SystemExit(f"colmap not found: {args.colmap_bin!r} (set --colmap-bin)")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_out = Path(tmp) / "aligned"       # scratch: we only need transform.txt
        tmp_out.mkdir()
        transform_path = Path(tmp) / "transform.txt"
        argv = [args.colmap_bin, "model_aligner",
                "--input_path", str(args.sparse), "--output_path", str(tmp_out),
                "--database_path", str(args.database), "--ref_is_gps", "1",
                "--alignment_type", args.alignment_type,
                "--alignment_max_error", str(args.max_error),
                "--transform_path", str(transform_path)]
        print(f"$ {' '.join(argv)}")
        res = subprocess.run(argv, capture_output=True, text=True)
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        if res.returncode != 0 or not transform_path.is_file():
            raise SystemExit("model_aligner failed (see output above)")

        scale, qw, qx, qy, qz, tx, ty, tz = (float(v) for v in transform_path.read_text().split())

    out = {
        "scale": scale,
        "rotation": flip_yz_columns(quat_wxyz_to_rotmat(qw, qx, qy, qz)),
        "translation": [tx, ty, tz],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    print(f"  scale = {scale}")
    print(f"  translation (ECEF m) = [{tx:.3f}, {ty:.3f}, {tz:.3f}]")


if __name__ == "__main__":
    main()
