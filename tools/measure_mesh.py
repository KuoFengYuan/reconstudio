#!/usr/bin/env python3
"""Auto-measure a mesh PLY's height/width/depth and render front/side/top
preview PNGs (occlusion-correct, with dimension lines baked in).

The mesh's orientation in raw COLMAP/3DGS coordinates is usually arbitrary —
there's no guarantee any of x/y/z (or even the PCA principal axes) line up
with "up". Run once with the defaults, look at front.png/side.png/top.png,
and if the figure is sideways or upside down, re-run with --flip-h / --axis
etc. until it looks right (this same flow is exposed as buttons on the web
mesh-measure page for jobs run through Recon Studio).

Example:
    python tools/measure_mesh.py --in mesh.ply --out-dir output/measure
    python tools/measure_mesh.py --in mesh.ply --out-dir output/measure --flip-h --axis 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.mesh_measure import measure_mesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", required=True, type=Path, dest="in_path", help="Input mesh PLY")
    p.add_argument("--out-dir", required=True, type=Path, help="Directory for front/side/top.png + summary.json")
    p.add_argument("--mode", choices=["pca", "raw"], default="pca",
                   help="pca (default): auto-find the most elongated axis, for tilted/unaligned "
                        "reconstructions. raw: use the mesh's literal x/y/z axes.")
    p.add_argument("--axis", type=int, default=0, choices=[0, 1, 2],
                   help="Which basis axis is height: 0=most elongated (pca) or x (raw), 1=y, 2=z")
    p.add_argument("--swap-wd", action="store_true", help="Swap which remaining axis is width vs depth")
    p.add_argument("--flip-h", action="store_true", help="Flip up/down (fixes upside-down renders)")
    p.add_argument("--flip-w", action="store_true", help="Mirror the width axis")
    p.add_argument("--flip-d", action="store_true", help="Mirror the depth axis")
    p.add_argument("--mm-per-unit", type=float, help="If known, also report height/width/depth in mm")
    p.add_argument("--force", action="store_true", help="Recompute even if summary.json already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.in_path.exists():
        raise SystemExit(f"ERROR: input not found: {args.in_path}")

    result = measure_mesh(
        args.in_path, args.out_dir,
        mode=args.mode, height_axis=args.axis, swap_wd=args.swap_wd,
        flip_h=args.flip_h, flip_w=args.flip_w, flip_d=args.flip_d,
        mm_per_unit=args.mm_per_unit, force=args.force,
    )

    unit = "mm" if args.mm_per_unit else "units"
    h = result.get("height_mm", result["height_units"])
    w = result.get("width_mm", result["width_units"])
    d = result.get("depth_mm", result["depth_units"])
    print(f"vertices: {result['n_vertices']:,}")
    print(f"height: {h:.3f} {unit}")
    print(f"width:  {w:.3f} {unit}")
    print(f"depth:  {d:.3f} {unit}")
    print(f"wrote {args.out_dir}/front.png, side.png, top.png, summary.json")
    print("if the figure looks sideways/upside-down in those images, re-run with "
          "--flip-h / --axis 1|2 / --swap-wd / --flip-w / --flip-d until it's correct")


if __name__ == "__main__":
    main()
