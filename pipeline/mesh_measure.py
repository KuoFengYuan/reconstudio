"""Auto-measure a mesh's height/width/depth and render front/side/top preview
PNGs, without needing numpy-heavy deps beyond numpy itself (no PIL/open3d/
plyfile — matches the panel's torch-free, dependency-light convention).

COLMAP/3DGS reconstructions rarely come out gravity-aligned, so which raw x/y/z
axis (or PCA principal axis) is actually "up" is ambiguous from the data alone.
This module computes candidate axes and renders a z-buffered (occlusion-correct)
preview per choice; picking the *correct* one still needs a human glance at the
rendered image — see the axis/flip params below.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

_PLY_TYPE_TO_NP = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def load_ply_vertices(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (xyz float64 Nx3, rgb uint8 Nx3 or None) from a binary PLY."""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header found")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        data_start = f.tell()

        fmt = None
        elements = []
        cur = None
        for raw in header_lines:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                _, name, count = line.split()
                cur = {"name": name, "count": int(count), "props": []}
                elements.append(cur)
            elif line.startswith("property"):
                parts = line.split()
                if parts[1] == "list":
                    cur["props"].append(("__list__", None))
                else:
                    cur["props"].append((parts[2], parts[1]))

        vertex_elem = next((e for e in elements if e["name"] == "vertex"), None)
        if vertex_elem is None:
            raise ValueError(f"{path}: no vertex element")
        if any(p[0] == "__list__" for p in vertex_elem["props"]):
            raise ValueError(f"{path}: vertex element has an unexpected list property")
        if fmt == "ascii":
            raise ValueError(f"{path}: ASCII PLY not supported, only binary")

        endian = "<" if "little" in fmt else ">"
        names = [n for n, _ in vertex_elem["props"]]
        np_dtype = np.dtype([(n, endian + _PLY_TYPE_TO_NP[t]) for n, t in vertex_elem["props"]])
        n = vertex_elem["count"]

        f.seek(data_start)
        data = np.frombuffer(f.read(np_dtype.itemsize * n), dtype=np_dtype, count=n)

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    rgb = None
    if "red" in names and "green" in names and "blue" in names:
        rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal stdlib PNG writer (8-bit truecolor, no filtering) — avoids a PIL dep."""
    h, w, _ = rgb.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + row.tobytes() for row in rgb)
    idat = zlib.compress(raw, 6)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    path.write_bytes(png)


def compute_basis(xyz: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroid, basis) — basis columns are candidate axes, sorted by
    descending extent so axis 0 is the "most elongated" direction."""
    centroid = xyz.mean(axis=0)
    if mode == "raw":
        return centroid, np.eye(3)
    centered = xyz - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return centroid, eigvecs[:, order]


def _zbuffer_render_dim(h, v, depth, rgb, size, margin, scale, near, out_path) -> None:
    """Orthographic point-splat with painter's-algorithm occlusion (nearest
    point per pixel wins) plus baked-in dimension lines along the two edges."""
    hmin, hmax = h.min(), h.max()
    vmin, vmax = v.min(), v.max()
    shape_w = (hmax - hmin) * scale
    shape_h = (vmax - vmin) * scale
    avail = size - 2 * margin
    off_x = margin + (avail - shape_w) / 2.0
    off_y = margin + (avail - shape_h) / 2.0

    px = (off_x + (h - hmin) * scale).astype(int)
    py = (off_y + (vmax - v) * scale).astype(int)
    valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    px, py, depth_v, rgb_v = px[valid], py[valid], depth[valid], rgb[valid]
    order = np.argsort(-depth_v) if near == "min" else np.argsort(depth_v)

    img = np.full((size, size, 3), 255, dtype=np.uint8)
    img[py[order], px[order]] = rgb_v[order]

    col = np.array([180, 30, 30], dtype=np.uint8)
    tick = 7

    def hline(y, x0, x1):
        y = int(round(y))
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        img[y, max(0, x0):min(size, x1 + 1)] = col

    def vline(x, y0, y1):
        x = int(round(x))
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        img[max(0, y0):min(size, y1 + 1), x] = col

    dim_y = off_y + shape_h + margin * 0.4
    hline(dim_y, off_x, off_x + shape_w)
    vline(off_x, dim_y - tick, dim_y + tick)
    vline(off_x + shape_w, dim_y - tick, dim_y + tick)
    dim_x = off_x - margin * 0.4
    vline(dim_x, off_y, off_y + shape_h)
    hline(off_y, dim_x - tick, dim_x + tick)
    hline(off_y + shape_h, dim_x - tick, dim_x + tick)

    _write_png(out_path, img)


def measure_mesh(
    ply_path: Path,
    out_dir: Path,
    *,
    mode: str = "pca",
    height_axis: int = 0,
    swap_wd: bool = False,
    flip_h: bool = False,
    flip_w: bool = False,
    flip_d: bool = False,
    mm_per_unit: float | None = None,
    size: int = 460,
    margin: int = 78,
    force: bool = False,
) -> dict:
    """Render front/side/top PNGs into out_dir and return the measured extents.
    The expensive part (load + PCA + render) is cached in out_dir/summary.json,
    keyed only by the orientation params — mm_per_unit is applied fresh on top of
    the cached raw-unit extents every call, so re-typing a scale factor doesn't
    re-run the renderer."""
    out_dir = Path(out_dir)
    summary_path = out_dir / "summary.json"
    if not force and summary_path.is_file():
        base = json.loads(summary_path.read_text())
    else:
        xyz, rgb = load_ply_vertices(Path(ply_path))
        if rgb is None:
            rgb = np.full((len(xyz), 3), 160, dtype=np.uint8)

        centroid, basis = compute_basis(xyz, mode)
        proj = (xyz - centroid) @ basis

        remaining = [i for i in range(3) if i != height_axis]
        if swap_wd:
            remaining = remaining[::-1]
        width_idx, depth_idx = remaining

        H = proj[:, height_axis] * (-1 if flip_h else 1)
        W = proj[:, width_idx] * (-1 if flip_w else 1)
        D = proj[:, depth_idx] * (-1 if flip_d else 1)

        height_ext = float(H.max() - H.min())
        width_ext = float(W.max() - W.min())
        depth_ext = float(D.max() - D.min())
        scale = (size - 2 * margin) / max(height_ext, width_ext, depth_ext)

        out_dir.mkdir(parents=True, exist_ok=True)
        _zbuffer_render_dim(W, H, D, rgb, size, margin, scale, "min", out_dir / "front.png")
        _zbuffer_render_dim(D, H, W, rgb, size, margin, scale, "min", out_dir / "side.png")
        _zbuffer_render_dim(W, D, H, rgb, size, margin, scale, "max", out_dir / "top.png")

        base = {
            "n_vertices": int(len(xyz)),
            "mode": mode, "height_axis": height_axis, "swap_wd": swap_wd,
            "flip_h": flip_h, "flip_w": flip_w, "flip_d": flip_d,
            "height_units": height_ext, "width_units": width_ext, "depth_units": depth_ext,
        }
        summary_path.write_text(json.dumps(base, indent=2))

    result = dict(base, mm_per_unit=mm_per_unit)
    if mm_per_unit:
        result["height_mm"] = base["height_units"] * mm_per_unit
        result["width_mm"] = base["width_units"] * mm_per_unit
        result["depth_mm"] = base["depth_units"] * mm_per_unit
    return result
