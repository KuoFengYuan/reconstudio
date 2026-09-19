#!/usr/bin/env python3
"""Pack a textured OBJ (+ .mtl + atlas image) into ONE self-contained .glb.

Why the panel needs this: a textured mesh is 3+ files that reference each other
by relative path, and the viewer serves exactly one file per URL — so an .obj
opened through it renders untextured, which looks like a failed bake. A .glb
carries the geometry, the UVs and the atlas bytes in a single file, so the same
single-file endpoint can show the real texture.

Runs in the *backend's* conda env (trimesh + pillow live there, not in the
torch-free panel), like the other tools/ scripts.

    python tools/obj_to_glb.py --in merged/mesh.obj --out mesh_textured.glb
    python tools/obj_to_glb.py --in merged/mesh.obj --out mesh_mm.glb --scale 124.8

--scale multiplies vertex positions only. UVs and the atlas are unaffected by a
similarity transform, which is why texturing happens BEFORE marker scaling and
the metric copy is produced by re-exporting rather than re-baking.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _mtl_textures(obj: Path) -> list[Path]:
    """Texture files the OBJ's .mtl points at (whether or not they loaded)."""
    mtl = obj.with_suffix(".mtl")
    if not mtl.is_file():
        return []
    out = []
    for line in mtl.read_text().splitlines():
        if line.strip().startswith("map_Kd "):
            f = (mtl.parent / line.split(None, 1)[1].strip()).resolve()
            if f.is_file():
                out.append(f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", required=True, type=Path, dest="in_path", help="textured .obj")
    ap.add_argument("--out", required=True, type=Path, dest="out_path", help="output .glb")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply vertex positions by this (default 1 = unchanged)")
    ap.add_argument("--max-texture", type=int, default=8192,
                    help="downscale the atlas to at most this many pixels per side "
                         "before embedding (0 = keep as-is). Browsers refuse textures "
                         "above their GL_MAX_TEXTURE_SIZE, typically 16384.")
    args = ap.parse_args()

    import trimesh  # noqa: I001 — heavy import, and only the backend env has it
    from PIL import Image

    # Pillow refuses images over ~179 Mpx as decompression bombs, and trimesh
    # SWALLOWS that refusal: the OBJ then loads with UVs but no texture and the
    # .glb comes out silently untextured — a white model in the viewer. These
    # atlases are ours, so lift the limit before anything opens one.
    Image.MAX_IMAGE_PIXELS = None

    if not args.in_path.is_file():
        print(f"[glb] input not found: {args.in_path}", file=sys.stderr)
        return 2
    mesh = trimesh.load(args.in_path, process=False, force="mesh")
    if mesh.vertices.shape[0] == 0:
        print(f"[glb] empty mesh: {args.in_path}", file=sys.stderr)
        return 2
    if args.scale != 1.0:
        mesh.apply_scale(args.scale)

    material = getattr(getattr(mesh, "visual", None), "material", None)
    img = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
    if img is None:
        # Distinguish "this OBJ has no texture" from "the texture failed to load",
        # because the second one used to look exactly like a successful run.
        referenced = _mtl_textures(args.in_path)
        if referenced:
            print(f"[glb] ERROR: the .mtl references {referenced[0].name} but trimesh "
                  "loaded no texture — the .glb would be untextured. Refusing to write it.",
                  file=sys.stderr)
            return 3
        print("[glb] warning: the OBJ has no texture at all — writing an untextured .glb.")
    elif args.max_texture and max(img.size) > args.max_texture:
        f = args.max_texture / max(img.size)
        small = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))),
                           Image.Resampling.LANCZOS)
        mesh.visual.material.image = small
        print(f"[glb] atlas {img.size} → {small.size} (--max-texture {args.max_texture})")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_bytes(trimesh.exchange.gltf.export_glb(
        trimesh.Scene(mesh), include_normals=True))
    mb = args.out_path.stat().st_size / 1e6
    print(f"[glb] wrote {args.out_path}  ({mb:.1f} MB, {len(mesh.faces):,} faces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
