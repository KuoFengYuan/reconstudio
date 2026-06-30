"""In-process ports of hierarchical-3d-gaussians' large-scene preprocessing steps.

These mirror Inria's `preprocess/` scripts (generate_colmap.py,
make_colmap_custom_matcher.py, simplify_images.py, auto_reorient.py,
make_mask_uint8.py) but as importable functions driven by reconstudio's own
pipeline rather than re-shelling `python preprocess/xxx.py`. Behavior is kept
faithful; the only deliberate changes (documented per-function) are:
  * auto_reorient uses numpy instead of torch (the panel process is torch-free);
  * the custom matcher is driven by reconstudio's ordered folder→sorted-names
    grouping so the emitted match names match what feature_extractor wrote to the
    DB exactly (not a separate os.walk that could disagree).

Heavy / optional deps (exif, cv2) are imported lazily inside the function that
needs them, so importing this module only requires numpy (scipy/sklearn are used
by simplify/reorient). The panel still starts if an optional dep is missing — the
error surfaces only when that specific option is actually used.
"""
from __future__ import annotations

import concurrent.futures as futures
import os
from pathlib import Path

import numpy as np

from .vendor.read_write_model import (
    Image,
    Point3D,
    qvec2rotmat,
    read_images_binary,
    read_model,
    rotmat2qvec,
    write_images_binary,
    write_model,
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# 1. custom matcher  (make_colmap_custom_matcher.py)
# --------------------------------------------------------------------------- #
def _decimal_coords(coords, ref) -> float:
    deg = coords[0] + coords[1] / 60 + coords[2] / 3600
    return -deg if ref in ("S", "W") else deg


def _image_gps(path: Path):
    """[lat, lon] from a JPEG's EXIF GPS via the `exif` lib, or None. Lazy import
    so the panel doesn't hard-depend on `exif` unless the GPS matcher is used."""
    try:
        from exif import Image as ExifImage
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "custom matcher GPS neighbours need the `exif` package "
            "(pip install exif), or set n_gps_neighbours=0 to disable"
        ) from exc
    try:
        with open(path, "rb") as src:
            img = ExifImage(src)
        if not img.has_exif:
            return None
        _ = img.gps_longitude  # access raises AttributeError when GPS is absent
        return [
            _decimal_coords(img.gps_latitude, img.gps_latitude_ref),
            _decimal_coords(img.gps_longitude, img.gps_longitude_ref),
        ]
    except (AttributeError, KeyError, ValueError):
        return None


def make_custom_matcher(groups, image_root, output_path, *, n_seq=0, n_quad=10,
                        n_loop=5, loop_matches=None, n_gps=25) -> int:
    """Write a COLMAP custom match list (one "name1 name2" pair per line) for
    `colmap matches_importer`. Port of make_colmap_custom_matcher.py.

    groups: ordered list of (prefix, [sorted bare image filenames]) — the SAME
    grouping reconstudio fed to feature_extractor, so the emitted names
    ("prefix/file", or just "file" when prefix=="") match the DB. Pairs combine:
      * sequential   : current+0 .. current+(n_seq-1)
      * quadratic     : current + n_seq + 2^k - 1  for k in [0, n_quad)
      * loop closure  : ±2^k windows around each (a,b) in loop_matches
      * GPS neighbours: n_gps EXIF-GPS nearest neighbours across all images
    Returns the number of (deduplicated) pairs written.
    """
    loop_matches = np.array(loop_matches or [], dtype=np.int64).reshape(-1, 2)
    # ±2^k window (and 0) used to spread each loop-closure anchor over nearby frames
    loop_rel = 2 ** np.arange(0, n_loop)
    loop_rel = np.concatenate([-loop_rel[::-1], np.array([0]), loop_rel])

    names = [g[1] for g in groups]
    prefixes = [g[0] for g in groups]

    def full(i: int, fname: str) -> str:
        return f"{prefixes[i]}/{fname}" if prefixes[i] else fname

    matches: list[str] = []

    def add_match(cur_i, off, cur_file, matched_frame_id):
        m_i = cur_i + off
        # guard >= 0 too (upstream only checks the upper bound, so a negative
        # loop-closure index would silently wrap to the tail — we skip instead).
        if 0 <= matched_frame_id < len(names[m_i]):
            matches.append(f"{full(cur_i, cur_file)} {full(m_i, names[m_i][matched_frame_id])}\n")

    for cur_i in range(len(groups)):
        for off in range(len(groups) - cur_i):
            for cur_id, cur_file in enumerate(names[cur_i]):
                for step in range(n_seq):
                    add_match(cur_i, off, cur_file, cur_id + step)
                for k in range(n_quad):
                    step = n_seq + int(2 ** k) - 1
                    add_match(cur_i, off, cur_file, cur_id + step)
            for lm in loop_matches:
                for cur_rel in loop_rel:
                    cur_id = int(lm[0] + cur_rel)
                    if 0 <= cur_id < len(names[cur_i]):
                        cur_file = names[cur_i][cur_id]
                        for m_rel in loop_rel:
                            add_match(cur_i, off, cur_file, int(lm[1] + m_rel))

    # GPS nearest-neighbour pairs (optional)
    if n_gps > 0:
        from sklearn.neighbors import NearestNeighbors  # lazy: only for GPS path

        all_names = [full(i, f) for i in range(len(groups)) for f in names[i]]
        coords = [_image_gps(Path(image_root) / n) for n in all_names]
        gps_names = [n for n, c in zip(all_names, coords, strict=True) if c is not None]
        gps_coords = np.array([c for c in coords if c is not None])
        if gps_coords.shape[0] > 1:
            k = min(n_gps, gps_coords.shape[0])
            nbrs = NearestNeighbors(n_neighbors=k).fit(gps_coords)
            for name, center in zip(gps_names, gps_coords, strict=True):
                _, idx = nbrs.kneighbors(center[None])
                for j in idx[0, 1:]:
                    matches.append(f"{name} {gps_names[j]}\n")

    # dedup, then drop any pair that is the reverse of one already kept
    intermediate = list(dict.fromkeys(matches))
    reciprocal = {f"{m.split(' ')[1][:-1]} {m.split(' ')[0]}\n" for m in intermediate}
    out = [m for m in intermediate if m not in reciprocal]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("".join(out))
    return len(out)


# --------------------------------------------------------------------------- #
# 2. simplify_images.py  — identify pose-outlier cameras to drop
# --------------------------------------------------------------------------- #
def outlier_image_ids(model_dir, mult_min_dist: float = 10.0, ext: str = "bin"):
    """Return (sorted list of image_ids to drop, n_total) for model_dir.

    An image is a pose outlier when its nearest-neighbour camera spacing exceeds
    mult_min_dist × the median spacing, or it sees no registered 3D point. This
    is the h3dgs simplify_images.py keep-rule, inverted to a drop-list.

    Pure analysis — the actual deletion is delegated to COLMAP's image_deleter so
    images / point tracks / rigs / frames stay mutually consistent. Hand-editing
    images.bin (as the original port did) strands frames.bin's data_ids under the
    COLMAP 4.x rig/frame model and breaks the point2D/point3D invariants, which
    aborts every downstream loader (e.g. image_undistorter). NearestNeighbors is
    imported lazily."""
    from sklearn.neighbors import NearestNeighbors

    images_metas = read_images_binary(str(Path(model_dir) / f"images.{ext}"))
    keys = list(images_metas)
    cam_centers = np.array([
        -qvec2rotmat(images_metas[k].qvec).T @ images_metas[k].tvec for k in keys
    ])
    nbrs = NearestNeighbors(n_neighbors=2).fit(cam_centers)
    second_min = np.array([nbrs.kneighbors(c[None])[0][0, -1] for c in cam_centers])
    med = np.median(second_min)

    drop: list[int] = []
    for key, smd in zip(keys, second_min, strict=True):
        meta = images_metas[key]
        sees_point = len(meta.point3D_ids) > 0 and int((meta.point3D_ids >= 0).sum()) > 0
        if not (sees_point and smd <= mult_min_dist * med):
            drop.append(int(key))
    return sorted(drop), len(keys)


# --------------------------------------------------------------------------- #
# 3. auto_reorient.py  (torch ops ported to numpy)
# --------------------------------------------------------------------------- #
def _fit_plane_least_squares(points):
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    B = points[:, 2]
    a, b, c = np.linalg.lstsq(A, B, rcond=None)[0]
    normal = np.array([a, b, -1.0])
    normal /= np.linalg.norm(normal)
    in_plane = np.cross(normal, np.array([0, 0, 1.0]))
    if np.linalg.norm(in_plane) == 0:
        in_plane = np.cross(normal, np.array([0, 1.0, 0]))
    in_plane /= np.linalg.norm(in_plane)
    return normal, in_plane, np.mean(points, axis=0)


def _rotate_camera(qvec, tvec, rot_matrix, upscale):
    R = qvec2rotmat(qvec)
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = np.array(tvec)
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = np.copy(C2W[:3, 3])
    cam_rot_orig = np.copy(C2W[:3, :3])
    cam_center = np.matmul(cam_center, rot_matrix)
    cam_rot = np.linalg.inv(rot_matrix) @ cam_rot_orig
    C2W[:3, 3] = upscale * cam_center
    C2W[:3, :3] = cam_rot
    Rt = np.linalg.inv(C2W)
    return Rt[:3, 3], rotmat2qvec(Rt[:3, :3])


def auto_reorient(input_path, output_path, upscale: float = 0.0,
                  target_med_dist: float = 20.0, ext: str = "bin"):
    """PCA gravity-align (camera-centre plane normal = up; longest camera span =
    right) and uniformly scale a COLMAP model so the median camera→point distance
    becomes `target_med_dist` (unless `upscale` is given). Output is Y-up (gravity
    up -> -Y), the same convention as zup_to_yup, so reconstudio's viewer renders it
    upright. Reads input_path, writes output_path. Ported from auto_reorient.py with
    numpy instead of torch; the basis is reordered from the original Z-up so the
    viewer's fixed flip shows +Y up. scipy.spatial is imported lazily."""
    from scipy import spatial

    cameras, images_in, points_in = read_model(str(input_path), ext=f".{ext}")

    if upscale != 0:
        scale = float(upscale)
    else:
        dists = []
        for meta in images_in.values():
            center = -qvec2rotmat(meta.qvec).astype(np.float32).T @ meta.tvec.astype(np.float32)
            dists.extend(
                np.linalg.norm(points_in[pid].xyz - center)
                for pid in meta.point3D_ids if pid != -1 and pid in points_in
            )
        med = np.median(np.array(dists))
        scale = float(target_med_dist) / med

    cam_centers = np.array([
        -qvec2rotmat(m.qvec).T @ m.tvec for m in images_in.values()
    ])
    up, _, _ = _fit_plane_least_squares(cam_centers)

    # the two cameras furthest apart (hull vertices) define the "right" axis
    candidates = cam_centers[spatial.ConvexHull(cam_centers).vertices]
    dmat = spatial.distance_matrix(candidates, candidates)
    i, j = np.unravel_index(dmat.argmax(), dmat.shape)
    right = candidates[i] - candidates[j]
    right /= np.linalg.norm(right)

    up = up.astype(np.float64)
    right = right.astype(np.float64)
    forward = np.cross(up, right)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    # Y-up convention, matching zup_to_yup so the viewer (fixed 180°-about-X flip,
    # which expects the model's up on -Y) renders it upright: gravity-up -> -Y,
    # forward -> +Z, right -> +X. (columns = images of the new basis axes; det=+1.)
    rotation_matrix = np.stack([right, -up, forward], axis=1)

    positions = np.array([points_in[k].xyz for k in points_in])
    rotated = scale * (positions @ rotation_matrix)
    points_out = {
        k: Point3D(id=p.id, xyz=rot, rgb=p.rgb, error=p.error,
                   image_ids=p.image_ids, point2D_idxs=p.point2D_idxs)
        for (k, p), rot in zip(points_in.items(), rotated, strict=True)
    }

    images_out = {}
    for k, m in images_in.items():
        new_pos, new_rot = _rotate_camera(m.qvec, m.tvec, rotation_matrix, scale)
        images_out[k] = Image(id=m.id, qvec=new_rot, tvec=new_pos,
                              camera_id=m.camera_id, name=m.name,
                              xys=m.xys, point3D_ids=m.point3D_ids)

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    write_model(cameras, images_out, points_out, str(out), f".{ext}")
    # COLMAP 4.x treats frames.bin/rigs.bin as the authoritative pose source and ignores
    # the (rotated) pose we just wrote into images.bin, so the reorient would silently no-
    # op. write_model doesn't emit those files; drop any left over from the undistort step
    # in the output dir so COLMAP falls back to images.bin's rotated/scaled poses (the
    # plain COLMAP-3.x layout h3dgs training consumes anyway).
    for f in ("frames", "rigs"):
        (out / f"{f}.{ext}").unlink(missing_ok=True)
    return scale, len(images_out), len(points_out)


def zup_to_yup(input_path, output_path, ext: str = "bin"):
    """Rigidly rotate a COLMAP model Z-up -> Y-up, with NO scaling and NO gravity
    guessing — a fixed +90° about X: (x,y,z) -> (x,-z,y).

    Use this instead of auto_reorient when the model is already metric and
    gravity-aligned by GPS (model_aligner's ENU frame, up=+Z): it only fixes the
    up-axis convention for reconstudio's viewer (which flips COLMAP Y-down -> three
    Y-up), so ENU up (+Z) lands on -Y and the viewer shows it upright, while GPS's
    metric scale is preserved exactly. Reads input_path, writes output_path; drops
    frames.bin/rigs.bin so the rotated images.bin poses are authoritative — both
    COLMAP's own loaders and LichtFeld (which reads pose straight from images.bin)
    then consume the rotated poses. Returns (n_cams, n_points)."""
    cameras, images_in, points_in = read_model(str(input_path), ext=f".{ext}")

    # row-vector convention (p' = p @ R), matching auto_reorient / _rotate_camera:
    # column j = image of basis axis j, so (x,y,z) -> (x, -z, y); +Z (ENU up) -> -Y.
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, -1.0, 0.0]])

    positions = np.array([points_in[k].xyz for k in points_in])
    rotated = positions @ R
    points_out = {
        k: Point3D(id=p.id, xyz=rot, rgb=p.rgb, error=p.error,
                   image_ids=p.image_ids, point2D_idxs=p.point2D_idxs)
        for (k, p), rot in zip(points_in.items(), rotated, strict=True)
    }
    images_out = {}
    for k, m in images_in.items():
        new_pos, new_rot = _rotate_camera(m.qvec, m.tvec, R, 1.0)   # upscale=1 -> no rescale
        images_out[k] = Image(id=m.id, qvec=new_rot, tvec=new_pos,
                              camera_id=m.camera_id, name=m.name,
                              xys=m.xys, point3D_ids=m.point3D_ids)

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    write_model(cameras, images_out, points_out, str(out), f".{ext}")
    for f in ("frames", "rigs"):   # see auto_reorient: keep images.bin poses authoritative
        (out / f"{f}.{ext}").unlink(missing_ok=True)
    return len(images_out), len(points_out)


# --------------------------------------------------------------------------- #
# 4. mask helpers  (replace_images_by_masks from generate_colmap.py)
# --------------------------------------------------------------------------- #
def replace_images_by_masks(images_file, out_file) -> None:
    """Copy a model's images.bin, swapping each image NAME's extension to .png, so
    the masks can be undistorted through the exact same camera model as the images.
    (h3dgs uses name[:-3]+'png'; we use splitext so non-3-char extensions also work.)"""
    metas = read_images_binary(str(images_file))
    out = {
        k: Image(id=m.id, qvec=m.qvec, tvec=m.tvec, camera_id=m.camera_id,
                 name=os.path.splitext(m.name)[0] + ".png",
                 xys=m.xys, point3D_ids=m.point3D_ids)
        for k, m in metas.items()
    }
    write_images_binary(out, str(out_file))


def make_mask_uint8(in_dir, out_dir, workers: int | None = None) -> int:
    """Turn undistorted RGBA mask images under in_dir into clean uint8 0/255 masks
    under out_dir (alpha>250 → 255, then a 3×3 erosion), mirroring the relative
    tree. Port of make_mask_uint8.py (cv2 lazy import; stdlib threads, not joblib).
    Returns the number of masks written."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mask processing needs OpenCV (pip install opencv-python), "
            "or leave masks_dir empty to skip"
        ) from exc

    in_root = Path(in_dir)
    rels = [str(p.relative_to(in_root)) for p in in_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    def one(rel: str) -> bool:
        img = cv2.imread(str(in_root / rel), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim < 3 or img.shape[2] < 4:
            return False
        dst = Path(out_dir) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        mask = (img[..., -1] > 250).astype(np.uint8) * 255
        eroded = (cv2.erode(mask, np.ones([3, 3], np.uint8)) > 250).astype(np.uint8) * 255
        cv2.imwrite(str(dst), eroded)
        return True

    n = 0
    with futures.ThreadPoolExecutor(max_workers=workers or (os.cpu_count() or 8)) as ex:
        for ok in ex.map(one, rels):
            n += int(ok)
    return n
