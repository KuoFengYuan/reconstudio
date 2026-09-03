"""The depth/normal PNG encoding LichtFeld-Studio expects, as pure numpy.

Kept free of torch/cv2/pydantic on purpose: `tools/moge3_preprocess.py` imports
it from inside the separate `moge3` conda env (which has torch but not this
app's deps), while the test suite imports it with neither. That is also why it
does not live in `moge3.py` next door, which pulls in `.runner` / `.config`.

Every rule below is a line-by-line port of LichtFeld's `build_depth_png` /
`build_normals_png` (src/preprocessing/preprocess.cpp). Getting one wrong does
not raise — it silently feeds the depth/normal losses garbage — so the port is
deliberately literal, and test_moge3_encode.py pins each rule.
"""
from __future__ import annotations

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# This tool's own output folders. In a flat dataset layout they sit directly
# under the same root a recursive scan walks, so they must never be inputs.
OUTPUT_DIR_NAMES = {"depth", "normals"}

# LichtFeld writes 16-bit by default; 8-bit is the only other option it takes.
MAX_VALUE = {8: 255, 16: 65535}


def encode_depth(z: np.ndarray, mask: np.ndarray, max_value: int) -> np.ndarray:
    """Point-map Z -> single-channel depth PNG.

    Depth on disk is NOT metric: it is min-max normalised per image into
    [0.02, 1.0] of full scale and floored at 1, which is what keeps 0 free as
    the "no data" sentinel. A pixel counts as valid only when the model's mask
    says so AND z is finite and positive — matching the C++ predicate exactly,
    because a mask-valid pixel with a non-finite z would otherwise poison the
    min/max and rescale the whole image.
    """
    dtype = np.uint16 if max_value > 255 else np.uint8
    valid = mask & np.isfinite(z) & (z > 0.0)
    out = np.zeros(z.shape, dtype=dtype)
    if not valid.any():
        return out                      # no anchor to normalise against: all invalid
    zv = z[valid]
    min_z, max_z = float(zv.min()), float(zv.max())
    denom = max(max_z - min_z, 1e-6)    # guards the constant-depth image
    normalized = 0.02 + 0.98 * ((zv - min_z) / denom)
    out[valid] = np.clip(np.rint(normalized * max_value), 1, max_value).astype(dtype)
    return out


def encode_normals(n: np.ndarray, mask: np.ndarray, max_value: int) -> np.ndarray:
    """Normal vectors (H, W, 3) -> three-channel PNG, n*0.5+0.5 per channel.

    Note the sentinel is the opposite of depth's: invalid pixels keep the
    NEUTRAL mid value, not 0, since 0 is a legitimate encoded normal (-1).
    """
    dtype = np.uint16 if max_value > 255 else np.uint8
    neutral = (max_value + 1) // 2
    out = np.full((*n.shape[:2], 3), neutral, dtype=dtype)
    encoded = np.where(np.isfinite(n), n * 0.5 + 0.5, 0.5)
    quantized = np.clip(np.rint(encoded * max_value), 0, max_value).astype(dtype)
    out[mask] = quantized[mask]
    return out
