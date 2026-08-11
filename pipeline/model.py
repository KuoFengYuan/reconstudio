"""Read a COLMAP sparse model for visualization: camera poses (centers + axes
for frustums) and stats, plus a PLY export of the 3D points via model_converter.

images.bin is parsed directly but the per-image 2D observations are skipped
(seeked past), so even a 500 MB file is read in well under a second.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
from functools import lru_cache
from pathlib import Path

from .config import settings

COLMAP_BIN = settings.colmap_bin   # PATH by default; override via COLMAP_BIN

# COLMAP camera model -> (name, num params)
_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


def _qvec_to_rt(q, t):
    """Return (center, Rt) where Rt is R^T (rows = camera axes in world)."""
    w, x, y, z = q
    R = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    # Rt[i][j] = R[j][i]; center = -R^T t
    rt = [[R[0][i], R[1][i], R[2][i]] for i in range(3)]
    center = [-(rt[i][0] * t[0] + rt[i][1] * t[1] + rt[i][2] * t[2]) for i in range(3)]
    return center, rt


# --------------------------------------------------------------------------- #
# mtime-keyed read cache.
#
# The sparse-model .bin files are immutable once a job finishes — a camera cull
# writes a *new* dir (see cull_cameras) and a FORCE re-run rewrites them with a
# fresh mtime — so memoising the parsers by (path, mtime, size) is safe and
# collapses the viewer's repeated reads into one parse per model: opening the 3D
# view used to parse cameras/images/points on every request, and every camera
# double-click + every thumbnail fetch re-parsed the *whole* images.bin from
# scratch (image_detail does its own targeted scan and is intentionally left
# uncached — it only reads up to the clicked index).
#
# Callers must treat the returned dict/list as read-only (it is shared across
# requests); every reader here only ever reads from it.
# --------------------------------------------------------------------------- #
def _stat_key(path: Path) -> tuple[str, int, int]:
    """Cache key that busts when the file is replaced/rewritten."""
    st = os.stat(path)
    return (str(path), st.st_mtime_ns, st.st_size)


def read_cameras(path: Path) -> dict:
    return _read_cameras(_stat_key(path))


@lru_cache(maxsize=32)
def _read_cameras(key: tuple[str, int, int]) -> dict:
    cams: dict[int, dict] = {}
    with open(key[0], "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, model, w, h = struct.unpack("<IiQQ", f.read(24))
            name, k = _CAMERA_MODELS.get(model, (str(model), 0))
            params = struct.unpack(f"<{k}d", f.read(8 * k)) if k else ()
            cams[cid] = {"model": name, "width": w, "height": h, "params": list(params)}
    return cams


def read_images(path: Path) -> list[dict]:
    return _read_images(_stat_key(path))


@lru_cache(maxsize=32)
def _read_images(key: tuple[str, int, int]) -> list[dict]:
    out: list[dict] = []
    with open(key[0], "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)                                    # image_id
            q = struct.unpack("<4d", f.read(32))
            t = struct.unpack("<3d", f.read(24))
            cam = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c in (b"\0", b""):
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.seek(npts * 24, 1)                         # skip 2D observations
            center, rt = _qvec_to_rt(q, t)
            out.append({"name": name.decode(errors="replace"),
                        "camera_id": cam, "center": center, "Rt": rt})
    return out


def count_points(path: Path) -> int:
    with open(path, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def points_stats(path: Path) -> tuple[int, float, float]:
    return _points_stats(_stat_key(path))


@lru_cache(maxsize=32)
def _points_stats(key: tuple[str, int, int]) -> tuple[int, float, float]:
    """(num_points, mean_reprojection_error_px, mean_track_length).

    points3D.bin record: id(u64) xyz(3*f64) rgb(3*u8) error(f64) track_len(u64)
    then track_len*(image_id u32, point2D_idx u32). Track bytes are skipped.
    """
    with open(key[0], "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        err_sum, trk_sum = 0.0, 0
        for _ in range(n):
            f.read(35)                                  # id + xyz + rgb
            err = struct.unpack("<d", f.read(8))[0]
            tl = struct.unpack("<Q", f.read(8))[0]
            f.seek(tl * 8, 1)
            err_sum += err
            trk_sum += tl
    return n, (err_sum / n if n else 0.0), (trk_sum / n if n else 0.0)


# --------------------------------------------------------------------------- #
# region helpers — back the viewer's ⬚ 選訓練範圍 tool, which produces a
# blocksplit `region` (minx,miny,maxx,maxy on the model's X–Y plane).
#
# Deliberately numpy-free (struct-parsed like the readers above) so the web panel
# keeps its numpy-free base install: blocksplit itself needs numpy, but validating
# a region must not.
# --------------------------------------------------------------------------- #
def up_axis(model_dir: Path) -> int | None:
    """World axis (0=X, 1=Y, 2=Z) with the smallest camera-centre spread — i.e.
    the vertical/up axis, since cameras sit at roughly constant height so the up
    axis varies least. None when the model can't be read or has < 3 views.

    Single definition, shared with blocksplit._up_axis (which drives _resolve_zup)
    so the viewer's idea of "is this model Z-up" cannot drift from the splitter's.
    """
    try:
        imgs = read_images(model_dir / "images.bin")
    except OSError:
        return None
    if len(imgs) < 3:
        return None
    lo = [min(im["center"][k] for im in imgs) for k in range(3)]
    hi = [max(im["center"][k] for im in imgs) for k in range(3)]
    spread = [hi[k] - lo[k] for k in range(3)]
    return spread.index(min(spread))


def frame_delta(model_a: Path, model_b: Path) -> tuple[float, float, int]:
    """(max centre disagreement, scene size, n shared images) between two models,
    matching cameras by image name.

    Used to prove that a region picked in the viewer's model is meaningful in the
    model blocksplit will actually read. Undistortion preserves poses, so
    sparse/0 and the undistorted sibling normally agree exactly; a reoriented or
    GPS-aligned model does not, and picking a region there would silently produce
    numbers for the wrong frame. Returns (0.0, size, n) for identical paths.
    """
    a = {im["name"]: im["center"] for im in read_images(model_a / "images.bin")}
    b = {im["name"]: im["center"] for im in read_images(model_b / "images.bin")}
    shared = sorted(set(a) & set(b))
    if not shared:
        return (float("inf"), 0.0, 0)
    lo = [min(a[n][k] for n in shared) for k in range(3)]
    hi = [max(a[n][k] for n in shared) for k in range(3)]
    size = max(hi[k] - lo[k] for k in range(3))
    worst = max(max(abs(a[n][k] - b[n][k]) for k in range(3)) for n in shared)
    return (worst, size, len(shared))


def region_stats(model_dir: Path, region: tuple[float, float, float, float],
                 track_min: int = 5) -> dict:
    """What a blocksplit `region` actually contains, read from `model_dir`.

    Counts cameras whose centre falls inside the X–Y rectangle, points inside it,
    and the in-region track-length distribution — track length is the multi-view
    coverage signal that decides whether the region is worth training (a region
    full of 2-view points reconstructs badly however large it is).
    """
    minx, miny, maxx, maxy = region
    imgs = read_images(model_dir / "images.bin")
    cam_in = sum(1 for im in imgs
                 if minx <= im["center"][0] <= maxx and miny <= im["center"][1] <= maxy)

    n_pts = pts_in = trk_sum = trk_ge = 0
    z_lo = z_hi = None
    with open(model_dir / "points3D.bin", "rb") as f:
        n_pts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_pts):
            head = f.read(43)                    # id + xyz + rgb + error
            x, y, z = struct.unpack("<3d", head[8:32])
            tl = struct.unpack("<Q", f.read(8))[0]
            f.seek(tl * 8, 1)                    # skip the track
            if minx <= x <= maxx and miny <= y <= maxy:
                pts_in += 1
                trk_sum += tl
                if tl >= track_min:
                    trk_ge += 1
                z_lo = z if z_lo is None or z < z_lo else z_lo
                z_hi = z if z_hi is None or z > z_hi else z_hi
    return {
        "cameras_in": cam_in, "cameras_total": len(imgs),
        "points_in": pts_in, "points_total": n_pts,
        "mean_track_in": (trk_sum / pts_in) if pts_in else 0.0,
        "track_min": track_min,
        "frac_track_ge_in": (trk_ge / pts_in) if pts_in else 0.0,
        "z_range": [z_lo, z_hi] if pts_in else None,
    }


def ensure_ply(model_dir: Path, out_ply: Path, force: bool = False) -> Path:
    if out_ply.exists() and not force and out_ply.stat().st_size > 0:
        return out_ply
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([COLMAP_BIN, "model_converter", "--input_path", str(model_dir),
                    "--output_path", str(out_ply), "--output_type", "PLY"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_ply


def cull_cameras(src_model: Path, out_model: Path, names: list[str],
                 filter_points: bool = True) -> None:
    """Write a copy of `src_model` to `out_model` with the named images removed
    (their 2D observations and now-empty 3D points are pruned too), via
    `colmap image_deleter`. The input model is never modified; `out_model` is
    created if needed. Removal is keyed by image name, which is stable across the
    distorted (sparse/0) and undistorted (dense) models.

    `image_deleter` alone only drops a 3D point when ALL its observations are gone;
    points still seen by a kept camera survive. When `filter_points` is set, follow
    with `colmap point_filtering` (track_len>=2, reproj<=4px, tri_angle>=1.5deg) so
    the points a removed (e.g. mis-located) camera leaves under-supported — the
    floaters / junk it contributed — are dropped too."""
    out_model.mkdir(parents=True, exist_ok=True)
    import tempfile
    fd, names_path = tempfile.mkstemp(suffix=".txt", text=True)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(names) + "\n")
        subprocess.run([COLMAP_BIN, "image_deleter",
                        "--input_path", str(src_model),
                        "--output_path", str(out_model),
                        "--image_names_path", names_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        os.unlink(names_path)
    if filter_points:
        subprocess.run([COLMAP_BIN, "point_filtering",        # in-place is safe (tested)
                        "--input_path", str(out_model), "--output_path", str(out_model),
                        "--min_track_len", "2", "--max_reproj_error", "4",
                        "--min_tri_angle", "1.5"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _f32(v: float) -> float:
    """Round a float64 to its float32 value — the exact value COLMAP's model_converter
    wrote into the viewer's .ply (so positions sent back from the viewer match)."""
    return struct.unpack("<f", struct.pack("<f", float(v)))[0]


def cull_points(src_model: Path, out_model: Path, del_keys: set) -> int:
    """Write a copy of `src_model` to `out_model` with selected 3D points removed,
    producing a valid COLMAP model (cameras/images copied unchanged, points3D rewritten
    without the deleted points). The input is never modified. `del_keys` is a set of
    (f32x, f32y, f32z) keys — points whose float32 xyz matches are dropped (the viewer
    sends the float32 positions it loaded from the .ply). Returns #points removed.

    points3D.bin record: id(u64) xyz(3·f64) rgb(3·u8) error(f64) track_len(u64) then
    track_len·(image_id u32, point2D_idx u32). cameras.bin / images.bin (poses) are
    copied verbatim; image↔point observations are left intact (trainers read points3D
    for the init cloud)."""
    src, out = Path(src_model), Path(out_model)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "cameras.bin", out / "cameras.bin")
    shutil.copy2(src / "images.bin", out / "images.bin")
    removed = 0
    kept: list[bytes] = []
    with open(src / "points3D.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            head = f.read(51)                       # id(8)+xyz(24)+rgb(3)+error(8)+track_len(8)
            x, y, z = struct.unpack("<3d", head[8:32])
            track = f.read(struct.unpack("<Q", head[43:51])[0] * 8)
            if (_f32(x), _f32(y), _f32(z)) in del_keys:
                removed += 1
            else:
                kept.append(head + track)
    with open(out / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(kept)))
        for rec in kept:
            f.write(rec)
    return removed


def scene(model_dir: Path) -> dict:
    md = Path(model_dir)
    cams = read_cameras(md / "cameras.bin")
    imgs = read_images(md / "images.bin")
    npoints, mean_err, mean_trk = points_stats(md / "points3D.bin")
    # global_mapper stores error in normalized image coords (÷ focal); incremental
    # stores pixels. Sub-0.1 values are normalized -> scale by mean focal to get px.
    focals = [c["params"][0] for c in cams.values() if c["params"]]
    mean_focal = sum(focals) / len(focals) if focals else 1.0
    reproj_px = mean_err * mean_focal if mean_err < 0.1 else mean_err
    return {
        "num_cameras": len(cams),
        "num_images": len(imgs),
        "num_points": npoints,
        "mean_reproj_error": round(reproj_px, 2),  # px; <1 great, <2 ok, higher = worse
        "mean_track_length": round(mean_trk, 2),   # observations per 3D point
        "cameras_info": {str(k): v for k, v in cams.items()},
        "images": [{"center": i["center"], "Rt": i["Rt"], "camera_id": i["camera_id"],
                    "name": i["name"]} for i in imgs],
    }


def image_detail(model_dir: Path, idx: int) -> dict:
    """Per-image info for click-inspection: name, camera, pixel dims, #features,
    #registered 3D points (2D observations linked to a point3D)."""
    path = Path(model_dir) / "images.bin"
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for i in range(n):
            f.read(4)                                    # image_id
            f.read(32 + 24 + 4)                          # qvec, tvec, camera_id... read below
            f.seek(-4, 1)
            cam = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c in (b"\0", b""):
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            block = f.read(npts * 24)
            if i == idx:
                registered = sum(1 for (_x, _y, pid) in struct.iter_unpack("<ddq", block)
                                 if pid != -1)
                try:
                    c = read_cameras(Path(model_dir) / "cameras.bin").get(cam, {})
                except OSError:
                    c = {}
                return {"index": idx, "name": name.decode(errors="replace"),
                        "camera_id": cam, "n_features": npts, "n_registered": registered,
                        "width": c.get("width"), "height": c.get("height")}
    raise IndexError(idx)
