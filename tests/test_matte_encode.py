"""Pins the 去背 maths in pipeline.matte_encode.

Like the depth encoding next door, most of this fails *silently* when wrong: a
reversed box, a mask union that dropped a subject, or a feather applied before
the erode all produce a perfectly valid PNG that is simply cut in the wrong
place. So each rule gets a test that would notice.

The one with real teeth is the colour bleed. Without it the semi-transparent
band keeps the background's RGB, and the cut-out shows the dark fringe everyone
blames on the segmentation — a defect that is invisible in the alpha channel and
only appears once something composites the image.
"""
from __future__ import annotations

import pytest

# numpy is a dev/test dependency only (the panel install stays pure-Python), so
# skip rather than fail where it is absent — same as test_moge3_encode.
np = pytest.importorskip("numpy")

from pipeline.matte_encode import (  # noqa: E402
    as_boxes,
    bleed_color,
    box_blur,
    clip_boxes,
    compose_rgba,
    encode_mask_l,
    filter_small,
    largest_component,
    merge_masks,
    refine_alpha,
    scale_boxes,
    select_row_boxes,
    suppress_overlaps,
)


def _square(shape=(40, 40), lo=10, hi=30) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[lo:hi, lo:hi] = True
    return m


# --------------------------------------------------------------------------- #
# boxes
# --------------------------------------------------------------------------- #
def test_as_boxes_normalises_a_reversed_drag():
    # Dragging up-and-left gives x0>x1; left as-is it silently yields an empty mask.
    out = as_boxes([[30, 40, 10, 20]])
    assert out.tolist() == [[10.0, 20.0, 30.0, 40.0]]


def test_as_boxes_accepts_a_flat_quadruple():
    assert as_boxes([1, 2, 3, 4]).shape == (1, 4)


def test_clip_boxes_clamps_and_drops_degenerate():
    boxes = as_boxes([[-5, -5, 50, 50], [10, 10, 10, 10]])
    out = clip_boxes(boxes, 20, 20)
    assert out.tolist() == [[0.0, 0.0, 19.0, 19.0]]      # the zero-area one is gone


def test_scale_boxes_denormalises():
    assert scale_boxes(as_boxes([[0.5, 0.25, 1.0, 0.5]]), 100, 200).tolist() == \
        [[50.0, 50.0, 100.0, 100.0]]


def test_filter_small_drops_below_the_fraction_and_zero_disables():
    boxes = as_boxes([[0, 0, 10, 10], [0, 0, 2, 2]])     # 1 % and 0.04 % of 100x100
    assert len(filter_small(boxes, 0.005, 100, 100)) == 1
    assert len(filter_small(boxes, 0.0, 100, 100)) == 2


def test_suppress_overlaps_removes_duplicates_keeps_distinct():
    out = suppress_overlaps(as_boxes([[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]]))
    assert len(out) == 2


def test_select_row_boxes_keeps_the_row_left_to_right():
    boxes = as_boxes([[30, 10, 40, 20], [10, 11, 20, 21], [5, 300, 15, 310]])
    out = select_row_boxes(boxes)
    assert out[:, 0].tolist() == [10.0, 30.0]           # sorted by x, off-row dropped


def test_select_row_boxes_falls_back_rather_than_over_filtering():
    # Two subjects at different heights: filtering would leave one, which is worse
    # than keeping both, so everything comes back (still sorted).
    boxes = as_boxes([[30, 200, 40, 210], [10, 10, 20, 20]])
    out = select_row_boxes(boxes, min_count=2)
    assert len(out) == 2
    assert out[0][0] == 10.0


# --------------------------------------------------------------------------- #
# mask merging
# --------------------------------------------------------------------------- #
def test_merge_masks_is_a_union_not_a_pick():
    a, b = np.zeros((4, 4), bool), np.zeros((4, 4), bool)
    a[0, 0] = True
    b[3, 3] = True
    out = merge_masks(np.stack([a, b]), (4, 4))
    assert out[0, 0] and out[3, 3] and out.sum() == 2


def test_merge_masks_accepts_the_channel_dim_sam_returns():
    m = np.zeros((2, 1, 4, 4), bool)
    m[0, 0, 1, 1] = True
    assert merge_masks(m, (4, 4)).sum() == 1


def test_merge_masks_on_nothing_is_empty_not_an_error():
    assert merge_masks(np.zeros((0, 4, 4)), (4, 4)).sum() == 0


def test_largest_component_keeps_only_the_biggest_blob():
    m = np.zeros((6, 6), bool)
    m[0:3, 0:3] = True          # 9 px
    m[5, 5] = True              # speck
    out = largest_component(m)
    assert out.sum() == 9 and not out[5, 5]


# --------------------------------------------------------------------------- #
# edge refinement
# --------------------------------------------------------------------------- #
def test_box_blur_of_a_constant_is_that_constant():
    assert np.allclose(box_blur(np.full((7, 7), 3.0, np.float32), 2), 3.0)


def test_erode_shrinks_and_dilate_grows():
    m = _square()
    assert refine_alpha(m, erode=2).sum() < m.sum()
    assert refine_alpha(m, dilate=2).sum() > m.sum()


def test_feather_creates_a_soft_band_and_no_feather_stays_binary():
    soft = refine_alpha(_square(), feather=3)
    assert ((soft > 0.01) & (soft < 0.99)).any()
    hard = refine_alpha(_square())
    assert set(np.unique(hard).tolist()) <= {0.0, 1.0}


def test_alpha_stays_in_range():
    a = refine_alpha(_square(), erode=1, dilate=2, feather=3)
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_erode_runs_before_feather():
    """Order matters: eroding first pulls the ramp inside the subject, so the
    outermost original-boundary pixel ends up less than fully opaque."""
    m = _square()
    a = refine_alpha(m, erode=2, feather=2)
    assert a[10, 20] < 0.9          # the original edge row
    assert a[20, 20] == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# composition — the black-fringe contract
# --------------------------------------------------------------------------- #
def test_compose_rgba_shape_and_alpha():
    rgb = np.zeros((8, 8, 3), np.uint8)
    out = compose_rgba(rgb, np.ones((8, 8), np.float32))
    assert out.shape == (8, 8, 4) and out.dtype == np.uint8
    assert out[..., 3].min() == 255


def test_bleed_fills_the_soft_band_with_foreground_colour():
    # Red subject on a black background: without the bleed the feathered rim keeps
    # the background's RGB (0,0,0) and composites as a dark halo.
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[10:30, 10:30] = (200, 40, 40)
    alpha = refine_alpha(_square(), feather=3)
    rim = (alpha > 0.05) & (alpha < 0.95)
    bled = compose_rgba(rgb, alpha, bleed=True)[..., :3]
    plain = compose_rgba(rgb, alpha, bleed=False)[..., :3]
    assert bled[rim][:, 0].mean() > 150          # rim now carries the subject's red
    assert plain[rim][:, 0].mean() < 120         # ...and would not have, without it


def test_bleed_leaves_the_solid_interior_untouched():
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[10:30, 10:30] = (200, 40, 40)
    out = compose_rgba(rgb, refine_alpha(_square(), feather=3))
    assert out[20, 20, :3].tolist() == [200, 40, 40]


def test_bleed_color_with_no_foreground_returns_the_input():
    rgb = np.full((5, 5, 3), 7, np.uint8)
    assert np.array_equal(bleed_color(rgb, np.zeros((5, 5), bool)), rgb)


def test_encode_mask_l_thresholds_or_keeps_the_ramp():
    alpha = np.array([[0.0, 0.4, 0.6, 1.0]], np.float32)
    assert encode_mask_l(alpha, 0.5).tolist() == [[0, 0, 255, 255]]
    assert encode_mask_l(alpha).tolist() == [[0, 102, 153, 255]]
