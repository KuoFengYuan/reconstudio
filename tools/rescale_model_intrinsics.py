#!/usr/bin/env python3
"""Rescale a COLMAP model's intrinsics to a different image resolution.

Use case: run the whole COLMAP pipeline on downscaled copies (fast extraction and
matching), then undistort from the ORIGINAL full-resolution images. That is not a
drop-in swap — `image_undistorter` hard-CHECKs that the camera dimensions equal the
bitmap's (`undistortion.cc:271`: "Check failed: distorted_camera.width ==
distorted_bitmap.Width()") and aborts otherwise. This rewrites the cameras so they
describe the originals.

    python tools/rescale_model_intrinsics.py \
        --sparse WS/sparse/0 --image-path /path/to/originals --output WS/sparse/0_full
    colmap image_undistorter --image_path /path/to/originals \
        --input_path WS/sparse/0_full --output_path WS/dense

Only the focal lengths, the principal point and width/height scale. Distortion
coefficients (k*, p*, s*, omega, alpha/beta) act on NORMALISED camera coordinates in
every COLMAP model, so they are dimensionless and must be left alone.

Caveat worth knowing before relying on this: intrinsics solved at N px carry a
sub-pixel error at N px, which becomes an (upscale-factor x) pixel error at the
original resolution. Going 8192 -> 14204 (1.7x) is mild; 1920 -> 14204 (7.4x) turns a
0.5 px solve into ~4 px of geometric error. Prefer solving as close to the target
resolution as the extractor allows.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.vendor.read_write_model import read_model, write_model  # noqa: E402

# model name -> number of leading focal-length params. The principal point is always
# the two params right after them, and everything past that is distortion, which is
# expressed in normalised coordinates and therefore resolution-independent.
# Mirrors colmap/src/colmap/sensor/models.h (`InitializeParamsInfo`).
FOCAL_COUNT = {
    "SIMPLE_PINHOLE": 1, "PINHOLE": 2, "SIMPLE_RADIAL": 1, "RADIAL": 1,
    "OPENCV": 2, "OPENCV_FISHEYE": 2, "FULL_OPENCV": 2, "FOV": 2,
    "SIMPLE_RADIAL_FISHEYE": 1, "RADIAL_FISHEYE": 1, "THIN_PRISM_FISHEYE": 2,
    "SIMPLE_DIVISION": 1, "DIVISION": 2, "SIMPLE_FISHEYE": 1, "FISHEYE": 2,
    "EUCM": 2,
}
# EQUIRECTANGULAR's params are (w, h), not a focal/principal point — it has no
# intrinsics to rescale beyond the dimensions themselves.
NO_INTRINSICS = {"EQUIRECTANGULAR"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sparse", required=True, type=Path,
                   help="Input model dir (cameras/images/points3D .bin or .txt)")
    p.add_argument("--output", required=True, type=Path,
                   help="Output model dir (created if missing)")
    p.add_argument("--image-path", required=True, type=Path,
                   help="Root of the target-resolution images; each camera's new size "
                        "is read from an actual file that uses it")
    p.add_argument("--ext", default="", choices=["", ".bin", ".txt"],
                   help="Input model format (default: auto-detect)")
    p.add_argument("--out-ext", default=".bin", choices=[".bin", ".txt"])
    p.add_argument("--anisotropy-tol", type=float, default=1e-3,
                   help="Warn when the x and y scale factors differ by more than this "
                        "relative amount (default 1e-3)")
    return p.parse_args()


def _tiff_size(fh) -> tuple[int, int] | None:
    """(width, height) from a TIFF's IFD0 tags 0x0100/0x0101, header reads only."""
    head = fh.read(8)
    if head[:2] not in (b"II", b"MM"):
        return None
    bo = "<" if head[:2] == b"II" else ">"
    fh.seek(struct.unpack(bo + "I", head[4:8])[0])
    n = struct.unpack(bo + "H", fh.read(2))[0]
    dims: dict[int, int] = {}
    for entry in struct.iter_unpack(bo + "HHI4s", fh.read(n * 12)):
        tag, typ, _cnt, raw = entry
        if tag in (0x0100, 0x0101):
            dims[tag] = (struct.unpack(bo + "H", raw[:2])[0] if typ == 3
                         else struct.unpack(bo + "I", raw)[0])
    if 0x0100 in dims and 0x0101 in dims:
        return dims[0x0100], dims[0x0101]
    return None


def _jpeg_size(fh) -> tuple[int, int] | None:
    """(width, height) from a JPEG's SOFn marker, header reads only."""
    if fh.read(2) != b"\xff\xd8":
        return None
    while True:
        head = fh.read(2)
        if len(head) < 2 or head[0] != 0xFF:
            return None
        marker = head[1]
        if marker == 0xD8 or 0xD0 <= marker <= 0xD7:
            continue
        if marker in (0xDA, 0xD9):
            return None
        seg = int.from_bytes(fh.read(2), "big") - 2
        body = fh.read(seg)
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", body[1:5])
            return w, h


def image_size(path: Path) -> tuple[int, int] | None:
    """(width, height) WITHOUT decoding pixels — these files can be 450 MB.

    Header parsing only, and it returns None rather than an approximation when it
    cannot be sure: an off-by-a-few size here silently corrupts every intrinsic it
    scales, and `image_undistorter` would then abort on the dimension CHECK anyway.
    (Notably OpenCV's IMREAD_REDUCED_* rounds to the reduction factor, so it must
    NOT be used as a fallback.)"""
    ext = path.suffix.lower()
    try:
        with path.open("rb") as fh:
            if ext in (".tif", ".tiff"):
                return _tiff_size(fh)
            if ext in (".jpg", ".jpeg"):
                return _jpeg_size(fh)
            if ext == ".png":
                if fh.read(8) != b"\x89PNG\r\n\x1a\n":
                    return None
                fh.seek(16)
                return struct.unpack(">II", fh.read(8))
    except Exception:  # noqa: BLE001 — treated as "cannot determine"
        return None
    return None


def main() -> None:
    args = parse_args()
    cameras, images, points3D = read_model(str(args.sparse), ext=args.ext)

    # one representative image file per camera
    sample: dict[int, str] = {}
    for im in images.values():
        sample.setdefault(im.camera_id, im.name)

    n_changed = 0
    for cam_id, cam in list(cameras.items()):
        name = sample.get(cam_id)
        if name is None:
            print(f"camera {cam_id}: no image references it, left unchanged")
            continue
        target = args.image_path / name
        if not target.is_file():
            raise SystemExit(f"camera {cam_id}: {target} not found — is --image-path "
                             "the root the model's image names are relative to?")
        size = image_size(target)
        if size is None:
            raise SystemExit(f"camera {cam_id}: could not read the size of {target}")
        new_w, new_h = size
        if (new_w, new_h) == (cam.width, cam.height):
            print(f"camera {cam_id} ({cam.model}): already {new_w}x{new_h}, unchanged")
            continue

        sx, sy = new_w / cam.width, new_h / cam.height
        aniso = abs(sx / sy - 1.0)
        if aniso > args.anisotropy_tol:
            print(f"  WARNING camera {cam_id}: x and y scales differ by {aniso:.2e} "
                  f"({sx:.6f} vs {sy:.6f}). The downscale did not preserve the aspect "
                  "ratio exactly; a single-focal model cannot represent that, so up to "
                  f"~{aniso * max(new_w, new_h):.1f} px of residual error remains.")

        params = list(cam.params)
        if cam.model in NO_INTRINSICS:
            params = [new_w, new_h]
        else:
            nf = FOCAL_COUNT.get(cam.model)
            if nf is None:
                raise SystemExit(f"camera {cam_id}: unknown camera model {cam.model!r} — "
                                 "add it to FOCAL_COUNT with its focal-param count")
            if nf == 1:
                # single focal: it cannot absorb anisotropy, so use the mean scale
                params[0] *= (sx + sy) / 2
            else:
                params[0] *= sx
                params[1] *= sy
            params[nf] *= sx          # cx
            params[nf + 1] *= sy      # cy
            # params[nf+2:] are distortion in normalised coords -> untouched

        print(f"camera {cam_id} ({cam.model}): {cam.width}x{cam.height} -> {new_w}x{new_h} "
              f"(scale {sx:.6f})")
        cameras[cam_id] = cam._replace(width=new_w, height=new_h, params=params)
        n_changed += 1

    args.output.mkdir(parents=True, exist_ok=True)
    write_model(cameras, images, points3D, str(args.output), ext=args.out_ext)
    print(f"\nrescaled {n_changed}/{len(cameras)} camera(s) -> {args.output}")
    print("poses and 3D points are unchanged — only the intrinsics describe a new "
          "image size, so the model stays in exactly the same world frame.")


if __name__ == "__main__":
    main()
