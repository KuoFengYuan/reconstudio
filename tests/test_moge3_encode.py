"""Pins the LichtFeld depth/normal PNG encoding that pipeline.moge3_encode ports.

These rules are a contract with a C++ program in another repo, and breaking one
does not raise anywhere: LichtFeld's depth/normal losses would just read wrong
numbers. The two sentinels are the easiest to get backwards, so they get their
own tests — depth marks "no data" with 0, normals with the NEUTRAL mid value,
because 0 is a legitimate encoded normal (-1).

The 16-bit floor asserted below, 1311, is the value LichtFeld's own MoGe-2
output was observed to bottom out at, so it pins the port against the real
thing rather than against a re-derivation of the same formula.
"""
from __future__ import annotations

import pytest

# The encoders are numpy by nature, and the panel deliberately keeps numpy out of
# its base/CI install (see jobs._run_blocksplit), so skip rather than fail there —
# same as test_blocksplit / test_scale_check. Note the consequence: these rules are
# NOT guarded by CI, only by a local run in an env that has numpy.
np = pytest.importorskip("numpy")

from pipeline.moge3_encode import MAX_VALUE, encode_depth, encode_normals  # noqa: E402

U16 = MAX_VALUE[16]
U8 = MAX_VALUE[8]
FLOOR_16 = 1311            # round(0.02 * 65535)
NEUTRAL_16 = (U16 + 1) // 2


def _all_valid(shape) -> np.ndarray:
    return np.ones(shape, dtype=bool)


# --------------------------------------------------------------------------- #
# depth
# --------------------------------------------------------------------------- #
def test_depth_spans_the_floor_to_full_scale():
    z = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = encode_depth(z, _all_valid(z.shape), U16)
    assert out.dtype == np.uint16
    assert out[0, 0] == FLOOR_16          # nearest depth sits on the 0.02 floor
    assert out[0, 2] == U16               # farthest saturates full scale
    assert FLOOR_16 < out[0, 1] < U16


def test_depth_reserves_zero_for_invalid_pixels():
    z = np.array([[1.0, 2.0]], dtype=np.float32)
    mask = np.array([[True, False]])
    out = encode_depth(z, mask, U16)
    assert out[0, 1] == 0                 # masked-out -> the no-data sentinel
    assert out[0, 0] >= 1                 # valid pixels never collide with it


def test_depth_rejects_non_finite_and_non_positive_z():
    # A mask-valid pixel whose z is nan/inf/<=0 must not participate: it would
    # otherwise poison min/max and rescale the whole image.
    z = np.array([[10.0, np.nan, np.inf, -5.0, 0.0, 20.0]], dtype=np.float32)
    out = encode_depth(z, _all_valid(z.shape), U16)
    assert list(out[0, 1:5]) == [0, 0, 0, 0]
    assert out[0, 0] == FLOOR_16 and out[0, 5] == U16   # anchored on 10 and 20 only


def test_depth_of_an_all_invalid_image_is_all_zero():
    z = np.array([[1.0, 2.0]], dtype=np.float32)
    out = encode_depth(z, np.zeros(z.shape, dtype=bool), U16)
    assert out.max() == 0                 # no anchor to normalise against


def test_depth_survives_a_constant_image():
    # max_z == min_z: the 1e-6 denominator guard is what keeps this finite.
    z = np.full((2, 2), 7.0, dtype=np.float32)
    out = encode_depth(z, _all_valid(z.shape), U16)
    assert np.isfinite(out).all() and (out == FLOOR_16).all()


def test_depth_honours_8_bit():
    z = np.array([[1.0, 2.0]], dtype=np.float32)
    out = encode_depth(z, _all_valid(z.shape), U8)
    assert out.dtype == np.uint8
    assert out[0, 0] == round(0.02 * U8) and out[0, 1] == U8


# --------------------------------------------------------------------------- #
# normals
# --------------------------------------------------------------------------- #
def test_normals_map_minus_one_to_zero_and_plus_one_to_full_scale():
    n = np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32)
    out = encode_normals(n, _all_valid(n.shape[:2]), U16)
    assert out.dtype == np.uint16
    assert list(out[0, 0]) == [0, NEUTRAL_16, U16]


def test_normals_use_the_neutral_sentinel_not_zero():
    # The opposite of depth: 0 means "pointing -1 on this axis", so invalid
    # pixels have to be the mid value instead.
    n = np.array([[[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]], dtype=np.float32)
    mask = np.array([[True, False]])
    out = encode_normals(n, mask, U16)
    assert list(out[0, 0]) == [0, 0, 0]                      # a real -1 normal
    assert list(out[0, 1]) == [NEUTRAL_16] * 3               # invalid -> neutral


def test_normals_treat_non_finite_components_as_neutral():
    n = np.array([[[np.nan, np.inf, 1.0]]], dtype=np.float32)
    out = encode_normals(n, _all_valid(n.shape[:2]), U16)
    assert list(out[0, 0][:2]) == [NEUTRAL_16, NEUTRAL_16]
    assert out[0, 0][2] == U16


def test_normals_clamp_out_of_range_vectors():
    n = np.array([[[-3.0, 3.0, 0.0]]], dtype=np.float32)
    out = encode_normals(n, _all_valid(n.shape[:2]), U16)
    assert list(out[0, 0]) == [0, U16, NEUTRAL_16]


def test_normals_keep_channel_order():
    # x,y,z must land in R,G,B; a transposed port would still "look" plausible.
    n = np.array([[[1.0, 0.0, -1.0]]], dtype=np.float32)
    out = encode_normals(n, _all_valid(n.shape[:2]), U16)
    assert list(out[0, 0]) == [U16, NEUTRAL_16, 0]
