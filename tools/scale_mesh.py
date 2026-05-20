#!/usr/bin/env python3
"""
Scale a mesh PLY file by a known factor.

Intended use: after MILo / 3DGS training completes at arbitrary COLMAP scale,
convert the exported mesh PLY to real-world units using a scale factor measured
by tools/estimate_marker_scale.py.

Units: --mm-per-unit is "mm per recon unit" (what estimate_marker_scale.py
prints). With --target-unit mm (the default) the output mesh is in millimetres
(matches the physically measured size); 'meters' divides by 1000, 'cm' by 10.

Behaviour:
  - vertex.x, vertex.y, vertex.z  multiplied by the scale multiplier
  - vertex normals (nx, ny, nz), colors, face indices: UNCHANGED
  - All other elements and properties are copied verbatim

Gaussian 3DGS PLY files (with scale_0 / rot_0 / opacity / f_dc_0) are
DETECTED and a warning is emitted -- this tool only scales positions, so
gaussian scales would be left wrong. Scaling a 3DGS PLY with this tool is
NOT RECOMMENDED (use a dedicated gaussian-aware tool instead).

Example:
    python tools/scale_mesh_ply.py \
        --in  output/milo/mesh.ply \
        --out output/milo/mesh_metric.ply \
        --mm-per-unit 124.825
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


UNIT_TO_MM = {"meters": 1000.0, "mm": 1.0, "cm": 10.0}

GAUSSIAN_SIGNATURE = {
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "opacity", "f_dc_0",
}

# Standard mesh PLY properties that Open3D preserves losslessly.
# If the input has only these, we can use the Open3D fast path.
STANDARD_MESH_PROPS = {
    "x", "y", "z",
    "nx", "ny", "nz",
    "red", "green", "blue", "alpha",
    "s", "t",
    "texture_u", "texture_v",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", required=True, type=Path, dest="in_path", help="Input mesh PLY")
    p.add_argument("--out", required=True, type=Path, dest="out_path", help="Output PLY")

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--mm-per-unit", type=float,
                     help="Scale (mm per recon unit) from estimate_marker_scale.py")
    grp.add_argument("--scale", type=float, help="Direct multiplier applied to xyz (advanced)")

    p.add_argument("--target-unit", default="mm", choices=list(UNIT_TO_MM.keys()),
                   help="Only used with --mm-per-unit. Default 'mm' keeps real-world "
                        "millimetres (1 recon unit -> mm_per_unit mm). Use 'meters' for "
                        "metric metres (= mm/1000) or 'cm' for centimetres.")
    p.add_argument("--force", action="store_true", help="Overwrite existing output file")
    p.add_argument("--allow-gaussian", action="store_true",
                   help="Do not abort if input looks like a 3DGS gaussian PLY (NOT recommended)")
    p.add_argument("--fast", action="store_true",
                   help="Force Open3D fast path (5-10x faster, but loses non-standard properties). "
                        "Auto-enabled when input only has standard mesh properties.")
    p.add_argument("--no-fast", action="store_true",
                   help="Disable fast path even if eligible (use plyfile to preserve all properties)")
    return p.parse_args()


def compute_multiplier(args: argparse.Namespace) -> float:
    if args.scale is not None:
        return float(args.scale)
    return float(args.mm_per_unit) / UNIT_TO_MM[args.target_unit]


def scale_via_open3d(in_path: Path, out_path: Path, multiplier: float) -> None:
    """Fast path: use Open3D (C++ backend, 5-10x faster). Loses custom vertex properties."""
    import open3d as o3d

    t0 = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(str(in_path))
    t_read = time.perf_counter() - t0

    if len(mesh.vertices) == 0:
        raise SystemExit("ERROR: Open3D read 0 vertices. Try --no-fast to use plyfile.")

    verts = np.asarray(mesh.vertices)
    mn_b, mx_b = verts.min(0), verts.max(0)
    print("BEFORE scaling:")
    print(f"  bbox min : {mn_b}")
    print(f"  bbox max : {mx_b}")
    print(f"  extents  : {mx_b - mn_b}  (diag {np.linalg.norm(mx_b - mn_b):.4f})")

    t0 = time.perf_counter()
    verts *= multiplier  # in-place
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    t_scale = time.perf_counter() - t0

    mn_a, mx_a = verts.min(0), verts.max(0)
    print()
    print(f"Applying multiplier = {multiplier:.8f}")
    print()
    print("AFTER scaling:")
    print(f"  bbox min : {mn_a}")
    print(f"  bbox max : {mx_a}")
    print(f"  extents  : {mx_a - mn_a}  (diag {np.linalg.norm(mx_a - mn_a):.4f})")

    t0 = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(
        str(out_path), mesh,
        write_vertex_normals=mesh.has_vertex_normals(),
        write_vertex_colors=mesh.has_vertex_colors(),
        write_triangle_uvs=mesh.has_triangle_uvs(),
    )
    t_write = time.perf_counter() - t0

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print()
    print(f"Wrote {out_path}  ({size_mb:.2f} MB)")
    print(f"  read: {t_read:.2f}s | scale: {t_scale:.2f}s | write: {t_write:.2f}s")


def scale_via_plyfile(in_path: Path, out_path: Path, multiplier: float, allow_gaussian: bool) -> None:
    """Safe path: preserves ALL vertex properties (including custom 3DGS fields)."""
    t0 = time.perf_counter()
    ply = PlyData.read(str(in_path))
    t_read = time.perf_counter() - t0

    # Locate the vertex element
    vertex_idx = None
    for i, elem in enumerate(ply.elements):
        if elem.name == "vertex":
            vertex_idx = i
            break
    if vertex_idx is None:
        raise SystemExit("ERROR: no 'vertex' element found in PLY.")
    vertex_elem = ply.elements[vertex_idx]

    prop_names = [p.name for p in vertex_elem.properties]
    for coord in ("x", "y", "z"):
        if coord not in prop_names:
            raise SystemExit(f"ERROR: vertex has no '{coord}' property. Is this really a mesh PLY?")

    # Detect 3DGS gaussian PLY
    gaussian_hits = sorted(GAUSSIAN_SIGNATURE.intersection(prop_names))
    if gaussian_hits:
        msg = (
            f"Input looks like a 3DGS gaussian PLY (has {', '.join(gaussian_hits)}).\n"
            f"Scaling only xyz would leave the gaussian scales in the old frame.\n"
            f"Pass --allow-gaussian to proceed anyway, or use a gaussian-aware tool."
        )
        if not allow_gaussian:
            raise SystemExit("ERROR: " + msg)
        print("WARNING: " + msg)

    elements_summary = ", ".join(f"{e.name}({len(e.data)})" for e in ply.elements)
    print(f"  elements: {elements_summary}")
    print(f"  vertex properties: {', '.join(prop_names)}")

    # Stats before
    data = vertex_elem.data
    mn_b = np.array([data["x"].min(), data["y"].min(), data["z"].min()])
    mx_b = np.array([data["x"].max(), data["y"].max(), data["z"].max()])
    print()
    print("BEFORE scaling:")
    print(f"  bbox min : {mn_b}")
    print(f"  bbox max : {mx_b}")
    print(f"  extents  : {mx_b - mn_b}  (diag {np.linalg.norm(mx_b - mn_b):.4f})")

    # Apply scale IN-PLACE (no full copy — saves ~50% memory + skip an allocation)
    t0 = time.perf_counter()
    data["x"] *= multiplier
    data["y"] *= multiplier
    data["z"] *= multiplier
    t_scale = time.perf_counter() - t0

    # Stats after
    mn_a = np.array([data["x"].min(), data["y"].min(), data["z"].min()])
    mx_a = np.array([data["x"].max(), data["y"].max(), data["z"].max()])

    print()
    print(f"Applying multiplier = {multiplier:.8f}")
    print()
    print("AFTER scaling:")
    print(f"  bbox min : {mn_a}")
    print(f"  bbox max : {mx_a}")
    print(f"  extents  : {mx_a - mn_a}  (diag {np.linalg.norm(mx_a - mn_a):.4f})")

    # Write (ply.elements already holds the mutated data in-place)
    t0 = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(out_path))
    t_write = time.perf_counter() - t0

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print()
    print(f"Wrote {out_path}  ({size_mb:.2f} MB)")
    print(f"  read: {t_read:.2f}s | scale: {t_scale:.2f}s | write: {t_write:.2f}s")


def peek_vertex_properties(in_path: Path) -> list[str]:
    """Quickly read only the PLY header to get vertex property names."""
    with open(in_path, "rb") as f:
        props = []
        in_vertex = False
        for raw in f:
            line = raw.decode("ascii", errors="ignore").strip()
            if line == "end_header":
                break
            if line.startswith("element "):
                parts = line.split()
                in_vertex = (len(parts) >= 2 and parts[1] == "vertex")
            elif in_vertex and line.startswith("property "):
                props.append(line.split()[-1])
    return props


def main() -> None:
    args = parse_args()

    if not args.in_path.exists():
        raise SystemExit(f"ERROR: input not found: {args.in_path}")
    if args.out_path.exists() and not args.force:
        raise SystemExit(f"ERROR: {args.out_path} exists. Pass --force to overwrite.")
    if args.in_path.resolve() == args.out_path.resolve():
        raise SystemExit("ERROR: --in and --out must differ.")

    multiplier = compute_multiplier(args)

    # Decide fast vs safe path
    prop_names = peek_vertex_properties(args.in_path)
    if not prop_names:
        raise SystemExit("ERROR: could not parse PLY header.")
    non_standard = [p for p in prop_names if p not in STANDARD_MESH_PROPS]
    is_gaussian = bool(GAUSSIAN_SIGNATURE.intersection(prop_names))
    eligible_fast = (not non_standard) and (not is_gaussian)

    use_fast = args.fast or (eligible_fast and not args.no_fast)

    if args.fast and not eligible_fast:
        print(f"WARNING: --fast requested, but input has non-standard properties: {non_standard}")
        print("         Those properties will be DROPPED. Proceeding anyway.")

    print(f"Loading {args.in_path}")
    print(f"  path: {'fast (Open3D)' if use_fast else 'safe (plyfile)'}")
    if not use_fast and eligible_fast:
        print("  (tip: add --fast for 5-10x speedup on this mesh)")

    t_total = time.perf_counter()
    if use_fast:
        scale_via_open3d(args.in_path, args.out_path, multiplier)
    else:
        scale_via_plyfile(args.in_path, args.out_path, multiplier, args.allow_gaussian)
    print(f"\nTotal: {time.perf_counter() - t_total:.2f}s")


if __name__ == "__main__":
    main()
