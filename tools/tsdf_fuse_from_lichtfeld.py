#!/usr/bin/env python3
"""TSDF-fuse a mesh from LichtFeld-Studio's `render-depth-normal` output.

Consumes the rgb/depth/normal/alpha renders produced by LichtFeld-Studio's
`render-depth-normal` CLI tool, re-reads the SAME COLMAP sparse model for
camera poses/intrinsics (matching files by COLMAP image name), and fuses
them into a mesh with Open3D's ScalableTSDFVolume -- the same approach
GS-2M uses for its own Gaussian-splat-to-mesh pipeline.

Run in the `gs2m` conda env (has open3d + opencv):
    conda run -n gs2m python3 tsdf_fuse_from_lichtfeld.py \
        --sparse /path/to/sparse --renders /path/to/renders --output /path/to/mesh.ply
"""

import argparse
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colmap_read_write_model as colmap_model

# First 4 params, in order, needed to recover fx,fy,cx,cy -- radial/tangential
# terms (if any) follow and are ignored, since the renders are pure pinhole.
_FXFYCXCY_BY_MODEL = {
    "SIMPLE_PINHOLE": lambda p: (p[0], p[0], p[1], p[2]),
    "PINHOLE": lambda p: (p[0], p[1], p[2], p[3]),
    "SIMPLE_RADIAL": lambda p: (p[0], p[0], p[1], p[2]),
    "RADIAL": lambda p: (p[0], p[0], p[1], p[2]),
    "OPENCV": lambda p: (p[0], p[1], p[2], p[3]),
    "FULL_OPENCV": lambda p: (p[0], p[1], p[2], p[3]),
}


def camera_intrinsics(cam):
    fn = _FXFYCXCY_BY_MODEL.get(cam.model)
    if fn is None:
        raise ValueError(
            f"Unsupported camera model '{cam.model}' (fisheye/equirectangular "
            "cameras aren't produced by render-depth-normal's pinhole rasterizer)"
        )
    return fn(cam.params)


def relative_stem(image_name):
    p = Path(image_name)
    return p.parent / p.stem


def load_exr_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32)


def load_exr_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


def sparse_point_spacing(sparse):
    """Median nearest-neighbor distance between COLMAP sparse points -- a floor on
    the detail actually resolved by triangulation (voxels much finer than this
    are mostly fusing depth/TSDF noise, not new geometry)."""
    bin_path, txt_path = sparse / "points3D.bin", sparse / "points3D.txt"
    if bin_path.is_file():
        points3d = colmap_model.read_points3D_binary(str(bin_path))
    elif txt_path.is_file():
        points3d = colmap_model.read_points3D_text(str(txt_path))
    else:
        return None
    if len(points3d) < 10:
        return None
    xyz = np.array([p.xyz for p in points3d.values()])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    return float(np.median(dists))


def median_ground_sample_distance(image_items, cameras, renders, sample_size=30):
    """Median metric size of one pixel on the visible surface (depth / focal_length_px),
    sampled from actually-rendered depth maps -- ties voxel resolution to what the
    imagery can actually resolve, at wherever the subject really sits in depth."""
    step = max(1, len(image_items) // sample_size)
    gsds = []
    for _, img in image_items[::step]:
        stem = relative_stem(img.name)
        depth_path = renders / "depth" / stem.with_suffix(".exr")
        if not depth_path.is_file():
            continue
        depth = load_exr_gray(depth_path)
        valid = depth[depth > 0]
        if valid.size == 0:
            continue
        cam = cameras[img.camera_id]
        fx, fy, _, _ = camera_intrinsics(cam)
        f_px = (fx + fy) / 2.0 * (depth.shape[1] / cam.width)
        gsds.append(float(np.median(valid)) / f_px)
    if not gsds:
        return None
    return float(np.median(gsds))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sparse", required=True, help="COLMAP sparse dir (same one passed to render-depth-normal)")
    ap.add_argument("--renders", required=True, help="Output dir from render-depth-normal (has rgb/depth/normal/alpha subfolders)")
    ap.add_argument("--output", required=True, help="Output mesh path (.ply)")
    ap.add_argument("--voxel-length", type=float, default=None,
                     help="TSDF voxel size, in reconstruction units (default: auto, from sparse point "
                          "spacing and rendered-depth ground-sample-distance -- see sparse_point_spacing() "
                          "and median_ground_sample_distance())")
    ap.add_argument("--sdf-trunc", type=float, default=None, help="TSDF truncation distance (default: voxel-length * 4)")
    ap.add_argument("--depth-trunc", type=float, default=None, help="Max depth to integrate (default: scene-extent * 2)")
    ap.add_argument("--normal-angle-thresh", type=float, default=75.0,
                     help="Drop pixels where the rendered normal grazes the view ray past this angle, in degrees (default: 75; 0 disables)")
    ap.add_argument("--min-alpha", type=float, default=0.0,
                     help="Re-threshold against the raw (unmasked) alpha.exr on top of render-depth-normal's own --alpha-threshold (default: 0, no extra filtering)")
    ap.add_argument("--keep-largest-clusters", type=int, default=1,
                     help="After extraction, keep only the N largest connected triangle clusters (by triangle count) and drop the rest as background/floaters. 0 disables (default: 1)")
    ap.add_argument("--max-images", type=int, default=None, help="Only fuse the first N cameras (debug/speed)")
    args = ap.parse_args()

    sparse = Path(args.sparse)
    renders = Path(args.renders)

    if (sparse / "cameras.bin").is_file():
        cameras = colmap_model.read_cameras_binary(str(sparse / "cameras.bin"))
        images = colmap_model.read_images_binary(str(sparse / "images.bin"))
    else:
        cameras = colmap_model.read_cameras_text(str(sparse / "cameras.txt"))
        images = colmap_model.read_images_text(str(sparse / "images.txt"))
    print(f"Read {len(cameras)} camera(s), {len(images)} image(s) from {sparse}")

    image_items = list(images.items())
    if args.max_images:
        image_items = image_items[: args.max_images]

    # Camera centers (world-space) purely to auto-scale TSDF parameters to
    # this reconstruction's (arbitrary, uncalibrated) unit scale.
    centers = []
    for _, img in image_items:
        r = colmap_model.qvec2rotmat(img.qvec)
        centers.append(-r.T @ img.tvec)
    centers = np.array(centers)
    scene_extent = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    if scene_extent <= 0:
        scene_extent = 1.0

    depth_trunc = args.depth_trunc or (scene_extent * 2.0)

    if args.voxel_length:
        voxel_length = args.voxel_length
        print(f"voxel_length={voxel_length:.4g} (explicit)")
    else:
        point_spacing = sparse_point_spacing(sparse)
        gsd = median_ground_sample_distance(image_items, cameras, renders)
        print(f"auto voxel_length inputs: sparse_point_spacing={point_spacing}  median_gsd={gsd}")
        # gsd (median depth / focal_length_px) is the imagery's actual per-pixel
        # footprint and is recomputed fresh from whatever is in the CURRENT
        # renders -- it correctly tracks a checkpoint that's been manually
        # cropped/cleaned to a small subject. point_spacing (COLMAP's sparse SfM
        # points) does NOT: those points are frozen at the original full-scene
        # reconstruction and know nothing about a later Gaussian edit, so using
        # it as a coarsening floor badly under-resolved a cleaned-down subject
        # (0.03 units, vs. the ~0.0004-0.0008 the user actually wanted) on
        # 2026-07-14. It's reported for reference only, not used as a bound.
        # /4 leans on multi-view averaging across many overlapping views (301
        # here) resolving finer than any single frame's own pixel footprint.
        if gsd:
            voxel_length = gsd / 4.0
        elif point_spacing:
            voxel_length = point_spacing
        else:
            voxel_length = scene_extent / 512.0
    sdf_trunc = args.sdf_trunc or (voxel_length * 4.0)
    cos_thresh = np.cos(np.radians(args.normal_angle_thresh)) if args.normal_angle_thresh > 0 else -1.0
    print(f"scene_extent={scene_extent:.4g}  voxel_length={voxel_length:.4g}  "
          f"sdf_trunc={sdf_trunc:.4g}  depth_trunc={depth_trunc:.4g}  "
          f"normal_angle_thresh={args.normal_angle_thresh}deg")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    fused, missing = 0, 0
    for _image_id, img in image_items:
        stem = relative_stem(img.name)
        rgb_path = renders / "rgb" / stem.with_suffix(".png")
        depth_path = renders / "depth" / stem.with_suffix(".exr")
        normal_path = renders / "normal" / stem.with_suffix(".exr")
        alpha_path = renders / "alpha" / stem.with_suffix(".exr")

        if not (rgb_path.is_file() and depth_path.is_file()):
            missing += 1
            continue

        color_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        depth = load_exr_gray(depth_path)
        height, width = depth.shape

        if args.min_alpha > 0.0 and alpha_path.is_file():
            alpha = load_exr_gray(alpha_path)
            depth = np.where(alpha >= args.min_alpha, depth, 0.0).astype(np.float32)

        # Scale COLMAP's stored intrinsics to whatever resolution the renderer
        # actually wrote (render-depth-normal's --scale need not match here).
        cam = cameras[img.camera_id]
        fx, fy, cx, cy = camera_intrinsics(cam)
        fx *= width / cam.width
        fy *= height / cam.height
        cx *= width / cam.width
        cy *= height / cam.height

        if cos_thresh > -1.0 and normal_path.is_file():
            normal = load_exr_rgb(normal_path)  # camera-space unit normal, (0,0,0) where invalid
            ys, xs = np.mgrid[0:height, 0:width]
            view_ray = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs, dtype=np.float32)], axis=-1)
            view_ray /= np.linalg.norm(view_ray, axis=-1, keepdims=True)
            cos_angle = -np.sum(normal * view_ray, axis=-1)
            depth = np.where(cos_angle >= cos_thresh, depth, 0.0).astype(np.float32)

        color_o3d = o3d.geometry.Image(np.ascontiguousarray(color))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        r = colmap_model.qvec2rotmat(img.qvec)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = r
        extrinsic[:3, 3] = img.tvec

        volume.integrate(rgbd, intrinsic, extrinsic)
        fused += 1
        if fused % 50 == 0:
            print(f"  fused {fused}/{len(image_items)}")

    print(f"Fused {fused} view(s), {missing} missing render(s)")
    if fused == 0:
        print("Error: nothing was fused, no mesh to extract", file=sys.stderr)
        sys.exit(1)

    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()

    if args.keep_largest_clusters > 0:
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        keep = set(np.argsort(-cluster_n_triangles)[: args.keep_largest_clusters].tolist())
        print(f"Mesh has {len(cluster_n_triangles)} connected cluster(s); "
              f"keeping the {len(keep)} largest ({sum(cluster_n_triangles[i] for i in keep)} "
              f"of {len(mesh.triangles)} triangles), dropping the rest as background/floaters")
        drop_mask = np.array([c not in keep for c in triangle_clusters])
        mesh.remove_triangles_by_mask(drop_mask)
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), mesh)

    aabb = mesh.get_axis_aligned_bounding_box()
    print(f"Wrote {output_path}")
    print(f"  vertices: {len(mesh.vertices)}")
    print(f"  triangles: {len(mesh.triangles)}")
    print(f"  bbox extent: {aabb.get_extent()}")


if __name__ == "__main__":
    main()
