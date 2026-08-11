"""The COLMAP sparse-model readers feed the 3D viewer, so their byte-for-byte
parsing of cameras/images/points3D .bin files — and the new mtime-keyed lru_cache
that collapses the viewer's repeated reads into a single parse per model — are
worth pinning down.

These tests synthesize the COLMAP binary layout by hand (little-endian struct) so
they need no colmap binary, no GPU, no network. The cache tests are the headline:
identical reads must return the *same* object (a hit), and replacing the file (new
mtime) must bust the key and force a fresh parse (a miss + a different object).
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

import pipeline.model as m


# --------------------------------------------------------------------------- #
# Local fixture builders — write the exact COLMAP binary layout.
# --------------------------------------------------------------------------- #
def _write_cameras(path: Path, cams):
    """cams: list of (cam_id, model_id, width, height, [params])."""
    buf = struct.pack("<Q", len(cams))
    for cid, model, w, h, params in cams:
        buf += struct.pack("<IiQQ", cid, model, w, h)
        buf += struct.pack(f"<{len(params)}d", *params)
    path.write_bytes(buf)


def _write_images(path: Path, images):
    """images: list of (image_id, qvec(4), tvec(3), camera_id, name, obs).

    obs: list of (x, y, point3D_id) where point3D_id == -1 means unregistered.
    """
    buf = struct.pack("<Q", len(images))
    for image_id, qvec, tvec, cam, name, obs in images:
        buf += struct.pack("<I", image_id)
        buf += struct.pack("<4d", *qvec)
        buf += struct.pack("<3d", *tvec)
        buf += struct.pack("<I", cam)
        buf += name.encode() + b"\x00"
        buf += struct.pack("<Q", len(obs))
        for x, y, pid in obs:
            buf += struct.pack("<ddq", x, y, pid)
    path.write_bytes(buf)


def _write_points(path: Path, points):
    """points: list of (point_id, xyz(3), rgb(3), error, track).

    track: list of (image_id, point2D_idx); only its length matters to the reader.
    """
    buf = struct.pack("<Q", len(points))
    for pid, xyz, rgb, error, track in points:
        buf += struct.pack("<Q", pid)
        buf += struct.pack("<3d", *xyz)
        buf += struct.pack("<3B", *rgb)
        buf += struct.pack("<d", error)
        buf += struct.pack("<Q", len(track))
        for img_id, p2d in track:
            buf += struct.pack("<II", img_id, p2d)
    path.write_bytes(buf)


_IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)


def _bump_mtime(path: Path):
    """Force a strictly later mtime so _stat_key busts the cache deterministically."""
    st = os.stat(path)
    later = st.st_mtime_ns + 1_000_000_000  # +1s, in ns
    os.utime(path, ns=(later, later))


# --------------------------------------------------------------------------- #
# read_cameras: each model's parameter count
# --------------------------------------------------------------------------- #
def test_read_cameras_parses_model_param_counts(tmp_path):
    path = tmp_path / "cameras.bin"
    _write_cameras(path, [
        (1, 0, 640, 480, [500.0, 320.0, 240.0]),              # SIMPLE_PINHOLE, 3
        (2, 1, 800, 600, [700.0, 701.0, 400.0, 300.0]),       # PINHOLE, 4
        (3, 4, 1920, 1080, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),  # OPENCV, 8
    ])
    cams = m.read_cameras(path)

    assert set(cams) == {1, 2, 3}
    assert cams[1]["model"] == "SIMPLE_PINHOLE"
    assert cams[1]["width"] == 640 and cams[1]["height"] == 480
    assert cams[1]["params"] == [500.0, 320.0, 240.0]

    assert cams[2]["model"] == "PINHOLE"
    assert cams[2]["params"] == [700.0, 701.0, 400.0, 300.0]

    assert cams[3]["model"] == "OPENCV"
    assert len(cams[3]["params"]) == 8
    assert cams[3]["params"][-1] == 8.0


# --------------------------------------------------------------------------- #
# read_images: name decode, camera_id, identity pose -> identity Rt / origin
# --------------------------------------------------------------------------- #
def test_read_images_identity_pose_and_observations_skipped(tmp_path):
    path = tmp_path / "images.bin"
    obs = [(1.0, 2.0, 10), (3.0, 4.0, -1), (5.0, 6.0, 99)]  # any obs must be skipped
    _write_images(path, [
        (7, _IDENTITY_Q, (0.0, 0.0, 0.0), 42, "frame_0001.png", obs),
    ])
    imgs = m.read_images(path)

    assert len(imgs) == 1
    img = imgs[0]
    assert img["name"] == "frame_0001.png"
    assert img["camera_id"] == 42

    # identity quaternion + zero translation -> camera at world origin (allow -0.0)
    assert img["center"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert [abs(c) for c in img["center"]] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)

    # Rt is the identity rotation
    assert img["Rt"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_read_images_handles_two_images_with_differing_obs_counts(tmp_path):
    path = tmp_path / "images.bin"
    _write_images(path, [
        (1, _IDENTITY_Q, (0.0, 0.0, 0.0), 1, "a.png", []),
        (2, _IDENTITY_Q, (1.0, 2.0, 3.0), 2, "b.png", [(0.0, 0.0, 5)]),
    ])
    imgs = m.read_images(path)
    assert [i["name"] for i in imgs] == ["a.png", "b.png"]
    assert [i["camera_id"] for i in imgs] == [1, 2]
    # second image had a translation -> not at the origin
    assert imgs[1]["center"] != [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# count_points
# --------------------------------------------------------------------------- #
def test_count_points_returns_header_count(tmp_path):
    path = tmp_path / "points3D.bin"
    _write_points(path, [
        (1, (0.0, 0.0, 0.0), (1, 2, 3), 0.5, [(1, 0), (2, 0)]),
        (2, (1.0, 1.0, 1.0), (4, 5, 6), 1.5, [(1, 1)]),
    ])
    assert m.count_points(path) == 2


# --------------------------------------------------------------------------- #
# points_stats: mean reproj error + mean track length, and empty file
# --------------------------------------------------------------------------- #
def test_points_stats_means(tmp_path):
    path = tmp_path / "points3D.bin"
    _write_points(path, [
        (1, (0.0, 0.0, 0.0), (0, 0, 0), 1.0, [(1, 0), (2, 0)]),       # track len 2
        (2, (0.0, 0.0, 0.0), (0, 0, 0), 2.0, [(1, 1), (2, 1), (3, 1)]),  # track len 3
        (3, (0.0, 0.0, 0.0), (0, 0, 0), 3.0, [(1, 2)]),               # track len 1
    ])
    n, mean_err, mean_trk = m.points_stats(path)
    assert n == 3
    assert mean_err == pytest.approx((1.0 + 2.0 + 3.0) / 3)
    assert mean_trk == pytest.approx((2 + 3 + 1) / 3)


def test_points_stats_empty_file(tmp_path):
    path = tmp_path / "points3D.bin"
    _write_points(path, [])
    assert m.points_stats(path) == (0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# image_detail
# --------------------------------------------------------------------------- #
def test_image_detail_counts_features_and_registered(tmp_path):
    obs = [(1.0, 1.0, 5), (2.0, 2.0, -1), (3.0, 3.0, 7), (4.0, 4.0, -1)]
    _write_images(tmp_path / "images.bin", [
        (1, _IDENTITY_Q, (0.0, 0.0, 0.0), 11, "first.png", [(0.0, 0.0, 1)]),
        (2, _IDENTITY_Q, (0.0, 0.0, 0.0), 22, "second.png", obs),
    ])
    detail = m.image_detail(tmp_path, 1)
    assert detail["index"] == 1
    assert detail["name"] == "second.png"
    assert detail["camera_id"] == 22
    assert detail["n_features"] == 4               # num_points2D for that image
    assert detail["n_registered"] == 2             # only obs with point3D_id != -1


def test_image_detail_out_of_range_raises_index_error(tmp_path):
    _write_images(tmp_path / "images.bin", [
        (1, _IDENTITY_Q, (0.0, 0.0, 0.0), 1, "only.png", []),
    ])
    with pytest.raises(IndexError):
        m.image_detail(tmp_path, 5)


# --------------------------------------------------------------------------- #
# up_axis / frame_delta / region_stats — back the viewer's ⬚ 選訓練範圍 tool.
# A region is 4 numbers on a Z-up model's X–Y plane, so these three answer:
# "which axis is up", "is the model I framed the one blocksplit will read", and
# "what is actually inside the rectangle".
# --------------------------------------------------------------------------- #
def _model_with_centers(model_dir: Path, centers):
    """A model whose cameras sit at `centers` (identity rotation -> center == -t)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_cameras(model_dir / "cameras.bin", [(1, 1, 8, 8, [1.0, 1.0, 4.0, 4.0])])
    _write_images(model_dir / "images.bin", [
        (i + 1, _IDENTITY_Q, (-c[0], -c[1], -c[2]), 1, f"img{i}.png", [])
        for i, c in enumerate(centers)
    ])
    return model_dir


def test_up_axis_picks_the_least_varying_axis(tmp_path):
    # cameras spread over X and Y at near-constant Z -> Z is up
    zup = [(x * 10.0, y * 10.0, 1.0 + (x % 2) * 0.01) for x in range(4) for y in range(4)]
    assert m.up_axis(_model_with_centers(tmp_path / "zup", zup)) == 2
    # the same layout with Y and Z swapped -> Y is up
    yup = [(c[0], c[2], c[1]) for c in zup]
    assert m.up_axis(_model_with_centers(tmp_path / "yup", yup)) == 1


def test_up_axis_none_for_too_few_views_or_missing_model(tmp_path):
    assert m.up_axis(_model_with_centers(tmp_path / "two", [(0, 0, 0), (1, 1, 1)])) is None
    assert m.up_axis(tmp_path / "does_not_exist") is None


def test_frame_delta_zero_for_same_poses_and_large_for_a_rotation(tmp_path):
    centers = [(x * 10.0, y * 10.0, 1.0) for x in range(3) for y in range(3)]
    a = _model_with_centers(tmp_path / "a", centers)
    same = _model_with_centers(tmp_path / "same", centers)
    delta, size, shared = m.frame_delta(a, same)
    assert shared == len(centers)
    assert delta == pytest.approx(0.0, abs=1e-9)
    assert size == pytest.approx(20.0)          # widest axis span of the shared cameras

    # a reoriented sibling (Y/Z swapped) must NOT look like the same frame — this is
    # the case that would otherwise hand blocksplit a region for the wrong plane.
    rot = _model_with_centers(tmp_path / "rot", [(c[0], c[2], c[1]) for c in centers])
    delta_rot, _size, shared_rot = m.frame_delta(a, rot)
    assert shared_rot == len(centers)
    assert delta_rot > size * 1e-4


def test_frame_delta_infinite_when_no_shared_image_names(tmp_path):
    a = _model_with_centers(tmp_path / "x", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    b = tmp_path / "y"
    b.mkdir()
    _write_cameras(b / "cameras.bin", [(1, 1, 8, 8, [1.0, 1.0, 4.0, 4.0])])
    _write_images(b / "images.bin", [
        (1, _IDENTITY_Q, (0.0, 0.0, 0.0), 1, "other_a.png", []),
        (2, _IDENTITY_Q, (0.0, 0.0, 0.0), 1, "other_b.png", []),
    ])
    delta, _size, shared = m.frame_delta(a, b)
    assert shared == 0 and delta == float("inf")


def test_region_stats_counts_only_what_is_inside_the_rectangle(tmp_path):
    model = tmp_path / "region"
    model.mkdir()
    _write_cameras(model / "cameras.bin", [(1, 1, 8, 8, [1.0, 1.0, 4.0, 4.0])])
    _write_images(model / "images.bin", [
        (1, _IDENTITY_Q, (-1.0, -1.0, 0.0), 1, "in.png", []),      # centre (1,1) inside
        (2, _IDENTITY_Q, (-9.0, -9.0, 0.0), 1, "out.png", []),     # centre (9,9) outside
    ])
    _write_points(model / "points3D.bin", [
        # inside, long track (2 of them) — these drive frac_track_ge_in
        (1, (1.0, 1.0, 5.0), (0, 0, 0), 0.5, [(1, 0)] * 6),
        (2, (2.0, 2.0, 7.0), (0, 0, 0), 0.5, [(1, 0)] * 4),        # inside, short track
        (3, (9.0, 9.0, 9.0), (0, 0, 0), 0.5, [(1, 0)] * 9),        # outside entirely
        (4, (1.0, 9.0, 9.0), (0, 0, 0), 0.5, [(1, 0)] * 9),        # x in, y out -> excluded
    ])
    s = m.region_stats(model, (0.0, 0.0, 3.0, 3.0), track_min=5)
    assert (s["cameras_in"], s["cameras_total"]) == (1, 2)
    assert (s["points_in"], s["points_total"]) == (2, 4)
    assert s["mean_track_in"] == pytest.approx((6 + 4) / 2)
    assert s["frac_track_ge_in"] == pytest.approx(0.5)             # only point 1 has >= 5
    assert s["z_range"] == [5.0, 7.0]                              # up-axis span of in-region pts


def test_region_stats_empty_region_reports_zeros_not_a_crash(tmp_path):
    model = tmp_path / "empty_region"
    model.mkdir()
    _write_cameras(model / "cameras.bin", [(1, 1, 8, 8, [1.0, 1.0, 4.0, 4.0])])
    _write_images(model / "images.bin", [(1, _IDENTITY_Q, (0.0, 0.0, 0.0), 1, "a.png", [])])
    _write_points(model / "points3D.bin", [(1, (0.0, 0.0, 0.0), (0, 0, 0), 0.5, [(1, 0)])])
    s = m.region_stats(model, (100.0, 100.0, 200.0, 200.0))
    assert s["points_in"] == 0 and s["cameras_in"] == 0
    assert s["mean_track_in"] == 0.0 and s["frac_track_ge_in"] == 0.0
    assert s["z_range"] is None


# --------------------------------------------------------------------------- #
# _qvec_to_rt
# --------------------------------------------------------------------------- #
def test_qvec_to_rt_identity():
    center, rt = m._qvec_to_rt(_IDENTITY_Q, (0.0, 0.0, 0.0))
    assert rt == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert center == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_qvec_to_rt_center_is_minus_rt_t():
    # identity rotation -> R^T == I, so center == -t
    t = (1.0, -2.0, 3.0)
    center, rt = m._qvec_to_rt(_IDENTITY_Q, t)
    assert center == pytest.approx([-1.0, 2.0, -3.0])
    # rotate 180deg about Z: q = (0,0,0,1) -> R = diag(-1,-1,1)
    center2, rt2 = m._qvec_to_rt((0.0, 0.0, 0.0, 1.0), (2.0, 0.0, 0.0))
    assert rt2[0][0] == pytest.approx(-1.0)
    assert rt2[1][1] == pytest.approx(-1.0)
    assert rt2[2][2] == pytest.approx(1.0)
    # center == -R^T t == -(-2, 0, 0) == (2, 0, 0)
    assert center2 == pytest.approx([2.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# scene: aggregation + reprojection-error normalization branch
# --------------------------------------------------------------------------- #
def _write_full_model(model_dir: Path, *, point_error, focal=1000.0):
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_cameras(model_dir / "cameras.bin", [
        (1, 1, 800, 600, [focal, focal, 400.0, 300.0]),  # PINHOLE, params[0] == focal
    ])
    _write_images(model_dir / "images.bin", [
        (1, _IDENTITY_Q, (0.0, 0.0, 0.0), 1, "a.png", []),
        (2, _IDENTITY_Q, (0.0, 0.0, 0.0), 1, "b.png", [(0.0, 0.0, 1)]),
    ])
    _write_points(model_dir / "points3D.bin", [
        (1, (0.0, 0.0, 0.0), (0, 0, 0), point_error, [(1, 0), (2, 0)]),
    ])


def test_scene_aggregates_counts(tmp_path):
    model_dir = tmp_path / "model_counts"
    _write_full_model(model_dir, point_error=0.5)
    s = m.scene(model_dir)
    assert s["num_cameras"] == 1
    assert s["num_images"] == 2
    assert s["num_points"] == 1
    assert s["mean_track_length"] == 2.0
    assert set(s["cameras_info"]) == {"1"}
    assert [i["name"] for i in s["images"]] == ["a.png", "b.png"]


def test_scene_normalized_error_scaled_by_focal(tmp_path):
    # mean_err < 0.1 is treated as normalized image coords -> multiply by mean focal.
    model_dir = tmp_path / "model_norm"
    _write_full_model(model_dir, point_error=0.05, focal=1000.0)
    s = m.scene(model_dir)
    assert s["mean_reproj_error"] == pytest.approx(round(0.05 * 1000.0, 2))  # 50.0


def test_scene_pixel_error_left_as_is(tmp_path):
    # mean_err >= 0.1 is already in pixels -> left unscaled.
    model_dir = tmp_path / "model_px"
    _write_full_model(model_dir, point_error=1.5, focal=1000.0)
    s = m.scene(model_dir)
    assert s["mean_reproj_error"] == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# THE CACHE: same path -> same object (hit); rewrite+mtime bump -> fresh parse.
# --------------------------------------------------------------------------- #
def test_read_cameras_cache_hit_returns_same_object(tmp_path):
    m._read_cameras.cache_clear()
    path = tmp_path / "cameras.bin"
    _write_cameras(path, [(1, 1, 640, 480, [500.0, 500.0, 320.0, 240.0])])

    first = m.read_cameras(path)
    before = m._read_cameras.cache_info()
    second = m.read_cameras(path)
    after = m._read_cameras.cache_info()

    assert first is second                       # shared, read-only object
    assert after.hits == before.hits + 1


def test_read_cameras_cache_busts_on_rewrite(tmp_path):
    m._read_cameras.cache_clear()
    path = tmp_path / "cameras.bin"
    _write_cameras(path, [(1, 1, 640, 480, [500.0, 500.0, 320.0, 240.0])])
    first = m.read_cameras(path)

    # rewrite with different content and a strictly later mtime -> new key
    _write_cameras(path, [(1, 1, 800, 600, [700.0, 700.0, 400.0, 300.0])])
    _bump_mtime(path)
    misses_before = m._read_cameras.cache_info().misses
    second = m.read_cameras(path)

    assert second is not first
    assert m._read_cameras.cache_info().misses == misses_before + 1
    assert second[1]["width"] == 800            # fresh parse reflects new bytes


def test_read_images_cache_hit_then_bust(tmp_path):
    m._read_images.cache_clear()
    path = tmp_path / "images.bin"
    _write_images(path, [(1, _IDENTITY_Q, (0.0, 0.0, 0.0), 1, "a.png", [])])

    first = m.read_images(path)
    second = m.read_images(path)
    assert first is second
    assert m._read_images.cache_info().hits >= 1

    _write_images(path, [(1, _IDENTITY_Q, (0.0, 0.0, 0.0), 1, "renamed.png", [])])
    _bump_mtime(path)
    third = m.read_images(path)
    assert third is not first
    assert third[0]["name"] == "renamed.png"


def test_points_stats_cache_hit_then_bust(tmp_path):
    m._points_stats.cache_clear()
    path = tmp_path / "points3D.bin"
    _write_points(path, [(1, (0.0, 0.0, 0.0), (0, 0, 0), 1.0, [(1, 0)])])

    first = m.points_stats(path)
    second = m.points_stats(path)
    assert first is second                       # tuple result is memoised & shared
    assert m._points_stats.cache_info().hits >= 1

    _write_points(path, [
        (1, (0.0, 0.0, 0.0), (0, 0, 0), 4.0, [(1, 0), (2, 0)]),
        (2, (0.0, 0.0, 0.0), (0, 0, 0), 6.0, [(1, 1), (2, 1)]),
    ])
    _bump_mtime(path)
    third = m.points_stats(path)
    assert third is not first
    assert third[0] == 2
    assert third[1] == pytest.approx(5.0)        # mean of 4 and 6


def test_missing_file_raises_file_not_found(tmp_path):
    m._read_cameras.cache_clear()
    missing = tmp_path / "does_not_exist.bin"
    with pytest.raises(FileNotFoundError):
        m.read_cameras(missing)          # from os.stat in _stat_key
    with pytest.raises(FileNotFoundError):
        m.read_images(missing)
    with pytest.raises(FileNotFoundError):
        m.points_stats(missing)
