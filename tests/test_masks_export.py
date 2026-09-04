"""The mask export that feeds a trainer: `large_scene.make_mask_uint8` and
`apply_masks_to_images`.

Both were pinned after a fused run finished "successfully" with an empty
`masks/`: make_mask_uint8 was a port of Inria's script, which reads `img[...,-1]`
and assumes the undistorted files are RGBA. They never are on an OIIO-based
COLMAP — `Bitmap::Read` drops alpha unconditionally (4→3 channels, 2→1), so a
single-channel mask comes back single-channel and a cut-out comes back as plain
RGB. Requiring 4 channels rejected every file, silently. Hence the channel
layouts below are the actual contract, not an implementation detail.
"""
from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2", reason="the mask export is cv2 work")
np = pytest.importorskip("numpy")

from pipeline import large_scene  # noqa: E402  (after the skip guards)


def _write(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), arr)


def _disc(w=32, h=24, val=255):
    """A mask with both a bright and a dark region, so a threshold that reads the
    wrong plane shows up as a wrong area rather than a wrong file count."""
    m = np.zeros((h, w), np.uint8)
    m[:, : w // 2] = val
    return m


# --- make_mask_uint8: every layout image_undistorter can emit ---------------- #
def test_single_channel_masks_survive_the_round_trip(tmp_path):
    """What this COLMAP writes when handed `no_bg/masks/*.png`."""
    _write(tmp_path / "in" / "cap" / "a.png", _disc())
    n = large_scene.make_mask_uint8(tmp_path / "in", tmp_path / "out")
    assert n == 1
    out = cv2.imread(str(tmp_path / "out" / "cap" / "a.png"), cv2.IMREAD_UNCHANGED)
    assert out is not None and set(np.unique(out)) <= {0, 255}
    assert out[:, 0].max() == 255 and out[:, -1].max() == 0     # sides not swapped


def test_a_mask_replicated_to_rgb_survives(tmp_path):
    """And what it writes when the undistorter widened the mask to 3 channels."""
    _write(tmp_path / "in" / "a.png", cv2.cvtColor(_disc(), cv2.COLOR_GRAY2BGR))
    assert large_scene.make_mask_uint8(tmp_path / "in", tmp_path / "out") == 1


def test_rgba_still_uses_the_alpha_channel(tmp_path):
    """Kept for COLMAP builds that preserve alpha: a cut-out's *colour* would
    threshold a dark subject away, so alpha must win when it is there."""
    rgba = np.zeros((24, 32, 4), np.uint8)
    rgba[..., :3] = 12                       # a near-black subject
    rgba[:, :16, 3] = 255                    # ...opaque on the left half
    _write(tmp_path / "in" / "a.png", rgba)
    assert large_scene.make_mask_uint8(tmp_path / "in", tmp_path / "out") == 1
    out = cv2.imread(str(tmp_path / "out" / "a.png"), cv2.IMREAD_UNCHANGED)
    assert out[:, 0].max() == 255 and out[:, -1].max() == 0


def test_unreadable_files_are_skipped_not_fatal(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "junk.png").write_bytes(b"not a png")
    _write(tmp_path / "in" / "a.png", _disc())
    assert large_scene.make_mask_uint8(tmp_path / "in", tmp_path / "out") == 1


def test_the_relative_tree_is_mirrored(tmp_path):
    """`images/<pass>/x.jpg` must resolve to `masks/<pass>/x.png`, so the pass
    folder cannot be flattened away."""
    _write(tmp_path / "in" / "up" / "a.png", _disc())
    _write(tmp_path / "in" / "down" / "b.png", _disc())
    large_scene.make_mask_uint8(tmp_path / "in", tmp_path / "out")
    assert (tmp_path / "out" / "up" / "a.png").is_file()
    assert (tmp_path / "out" / "down" / "b.png").is_file()


# --- apply_masks_to_images -------------------------------------------------- #
def _scene(tmp_path, name="up/a", ext=".jpg"):
    img = np.full((24, 32, 3), 200, np.uint8)
    _write(tmp_path / "images" / f"{name}{ext}", img)
    _write(tmp_path / "masks" / f"{name}.png", _disc())
    return tmp_path / "images", tmp_path / "masks"


def test_the_background_is_blacked_out_and_the_subject_kept(tmp_path):
    images, masks = _scene(tmp_path)
    assert large_scene.apply_masks_to_images(images, masks) == (1, 0)
    out = cv2.imread(str(images / "up" / "a.jpg"))
    assert out[:, :16].mean() > 150          # subject kept
    assert out[:, 16:].max() == 0            # background gone


def test_it_pairs_a_jpg_with_its_png_mask(tmp_path):
    """image_undistorter keeps the image's own extension while
    replace_images_by_masks swapped the model's to .png, so the two sides differ
    by exactly that."""
    images, masks = _scene(tmp_path, ext=".JPG")
    assert large_scene.apply_masks_to_images(images, masks) == (1, 0)


def test_running_twice_changes_nothing_more(tmp_path):
    images, masks = _scene(tmp_path)
    large_scene.apply_masks_to_images(images, masks)
    first = (images / "up" / "a.jpg").read_bytes()
    large_scene.apply_masks_to_images(images, masks)
    # blacking out already-black pixels is a no-op, so a redo is safe
    assert (images / "up" / "a.jpg").read_bytes() == first


def test_an_image_with_no_mask_is_left_alone_and_reported(tmp_path):
    images, masks = _scene(tmp_path)
    orphan = np.full((24, 32, 3), 111, np.uint8)
    _write(images / "up" / "orphan.jpg", orphan)
    assert large_scene.apply_masks_to_images(images, masks) == (1, 1)
    assert cv2.imread(str(images / "up" / "orphan.jpg")).mean() > 100


def test_a_mismatched_mask_size_is_reported_not_applied(tmp_path):
    """The geometry check up front should make this unreachable; if it ever is
    reached, a wrongly-sized mask must not be stretched over the image."""
    images, masks = _scene(tmp_path)
    _write(masks / "up" / "a.png", _disc(w=8, h=6))
    assert large_scene.apply_masks_to_images(images, masks) == (0, 1)
    assert cv2.imread(str(images / "up" / "a.jpg")).min() > 150      # untouched
