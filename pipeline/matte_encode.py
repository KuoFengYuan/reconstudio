"""The去背 (matting) maths: box bookkeeping, mask merging, edge refinement, RGBA.

Kept free of torch/cv2/pydantic for the same reason as `moge3_encode.py`:
`tools/sam_matte.py` imports it from inside the separate `sam` conda env (which
has torch but none of this app's deps), while the test suite imports it with
neither. Only numpy — no cv2 — so the morphology/blur below are hand-rolled
rather than delegated; they are small, separable, and exact.

Two conventions the rest of the pipeline depends on:

* The alpha this module produces is **float [0,1]**, and only `compose_rgba`
  quantises it. Keeping the soft band in float until the last step is what makes
  feathering meaningful at 8 bits.
* `compose_rgba` bleeds foreground colour outward before writing. Without it a
  feathered cut-out carries the *background's* RGB inside its semi-transparent
  band, which composites as the 黑邊/dark fringe that people then blame on the
  segmentation. The mask is right; the colour under it was wrong.
"""
from __future__ import annotations

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# This tool's own output folders plus the depth stage's. In a flat dataset layout
# they sit directly under the root a recursive scan walks, so they must never be
# read back as inputs (a second run would matte its own cut-outs).
OUTPUT_DIR_NAMES = {"depth", "normals", "masks", "cutout"}
# Everything a recursive photo scan must skip: the above, plus COLMAP's own
# subtrees — a workspace's `colmap/` holds undistorted copies of the same photos,
# so descending into it mattes every image two or three times over. Must stay
# equal to pipeline/matte.py's SKIP_DIR_NAMES (pinned by test_matte_encode.py);
# duplicated rather than imported because that module cannot be loaded from the
# `sam` env, and this one cannot join the panel's numpy-free import chain.
SKIP_DIR_NAMES = frozenset(OUTPUT_DIR_NAMES | {
    "colmap", "sparse", "dense", "stereo", "distorted",
})


# --------------------------------------------------------------------------- #
# 1. boxes
# --------------------------------------------------------------------------- #
def as_boxes(raw) -> np.ndarray:
    """Anything box-shaped -> a float32 (N, 4) xyxy array, normalised so x0<x1.

    Detectors and hand-drawn UI boxes both hand over corners in whatever order
    the drag happened, and a reversed box silently produces an empty mask rather
    than an error, so the swap happens once, here.
    """
    arr = np.asarray(raw, dtype=np.float32).reshape(-1, 4)
    x0 = np.minimum(arr[:, 0], arr[:, 2])
    x1 = np.maximum(arr[:, 0], arr[:, 2])
    y0 = np.minimum(arr[:, 1], arr[:, 3])
    y1 = np.maximum(arr[:, 1], arr[:, 3])
    return np.stack([x0, y0, x1, y1], axis=1)


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp to the image and drop degenerate (sub-pixel) boxes.

    SAM happily accepts an out-of-frame box and returns nonsense for it; a
    zero-area box from a stray click would otherwise become a stray hole.
    """
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    b = boxes.copy()
    b[:, 0] = np.clip(b[:, 0], 0, width - 1)
    b[:, 2] = np.clip(b[:, 2], 0, width - 1)
    b[:, 1] = np.clip(b[:, 1], 0, height - 1)
    b[:, 3] = np.clip(b[:, 3], 0, height - 1)
    keep = (b[:, 2] - b[:, 0] >= 1) & (b[:, 3] - b[:, 1] >= 1)
    return b[keep]


def scale_boxes(boxes: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Rescale xyxy boxes (the picker draws on a downscaled preview)."""
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    return boxes * np.array([sx, sy, sx, sy], dtype=np.float32)


def iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU, (N, N)."""
    if len(boxes) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    x0 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y0 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x1 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y1 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area[:, None] + area[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def suppress_overlaps(boxes: np.ndarray, scores=None, iou_thresh: float = 0.9) -> np.ndarray:
    """Greedy NMS. Duplicates cost a decoder slot each and, worse, make the
    per-image box count meaningless as a sanity signal in the log."""
    n = len(boxes)
    if n < 2:
        return boxes
    order = (np.argsort(-np.asarray(scores, dtype=np.float32)) if scores is not None
             else np.argsort(-((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))))
    ious = iou_matrix(boxes)
    keep: list[int] = []
    for i in order:
        if all(ious[i, j] <= iou_thresh for j in keep):
            keep.append(int(i))
    return boxes[sorted(keep)]


def filter_small(boxes: np.ndarray, min_area_frac: float, width: int, height: int) -> np.ndarray:
    """Drop boxes smaller than `min_area_frac` of the frame (0 disables).

    The usual noise source is a detector locking onto a background twig; a
    fraction rather than an absolute pixel count keeps one setting valid across
    a mixed-resolution folder.
    """
    if min_area_frac <= 0 or len(boxes) == 0:
        return boxes
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[area >= min_area_frac * float(width) * float(height)]


def select_row_boxes(boxes: np.ndarray, tol: float = 0.6,
                     min_count: int = 2) -> np.ndarray:
    """Keep only the boxes lying on the dominant **horizontal row**, left to right.

    The target case is a line of subjects across one frame: the row is defined by
    the median vertical centre, and a box joins it when its own centre sits
    within `tol` × the median box height of that line. Deliberately a *fallback*
    filter — if it would leave fewer than `min_count` boxes it returns everything
    sorted instead, because throwing away real subjects is worse than keeping an
    off-row one that the union will simply include.
    """
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    ref_cy, ref_h = float(np.median(cy)), float(np.median(heights))
    on_row = np.abs(cy - ref_cy) <= tol * ref_h
    picked = boxes[on_row] if int(on_row.sum()) >= min_count else boxes
    return picked[np.argsort(picked[:, 0])]


# --------------------------------------------------------------------------- #
# 2. mask merging
# --------------------------------------------------------------------------- #
def merge_masks(masks, shape: tuple[int, int]) -> np.ndarray:
    """Logical OR of every per-box mask -> one bool (H, W) foreground.

    The union is the whole point of prompting with N boxes at once: each box
    yields its own subject, and one alpha channel has to hold all of them. An
    empty stack returns an all-False mask rather than raising, so the caller's
    "nothing found" fallback is the single place that decides what that means.
    """
    arr = np.asarray(masks)
    if arr.size == 0:
        return np.zeros(shape, dtype=bool)
    arr = arr.reshape(-1, *shape)
    return np.logical_or.reduce(arr.astype(bool), axis=0)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest 4-connected blob (numpy flood fill, no scipy/cv2).

    Optional cleanup for the single-subject case: SAM's box prompt sometimes
    also grabs a same-coloured speck elsewhere in the box. Not applied to
    multi-box unions, where the extra blobs are the other subjects.
    """
    if not mask.any():
        return mask
    labels = np.zeros(mask.shape, dtype=np.int32)
    best_label, best_size, cur = 0, 0, 0
    h, w = mask.shape
    for start in zip(*np.nonzero(mask), strict=True):
        if labels[start]:
            continue
        cur += 1
        stack, size = [start], 0
        labels[start] = cur
        while stack:
            y, x = stack.pop()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = cur
                    stack.append((ny, nx))
        if size > best_size:
            best_label, best_size = cur, size
    return labels == best_label


# --------------------------------------------------------------------------- #
# 3. edge refinement  (morphology + feather, both separable, numpy only)
# --------------------------------------------------------------------------- #
def _extremum(a: np.ndarray, radius: int, take_max: bool) -> np.ndarray:
    """Square-kernel max (dilate) or min (erode) filter, separable per axis.

    Implemented as 2r+1 shifted in-place np.maximum/minimum passes rather than a
    sliding_window_view reduction: same result, but it touches each pixel O(r)
    times with contiguous memory instead of building a strided (H, W, 2r+1) view
    that thrashes cache on full-resolution photos.
    """
    if radius <= 0:
        return a
    f = np.maximum if take_max else np.minimum
    out = a
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        acc = None
        for off in range(2 * radius + 1):
            sl: list[slice] = [slice(None), slice(None)]
            sl[axis] = slice(off, off + n)
            chunk = padded[tuple(sl)]
            acc = chunk.copy() if acc is None else f(acc, chunk, out=acc)
        out = acc
    return out


def box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    """Exact separable box mean via a cumulative sum (O(1) per pixel, any radius)."""
    if radius <= 0:
        return a.astype(np.float32, copy=False)
    out = a.astype(np.float32, copy=False)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        cum = np.cumsum(padded, axis=axis)
        zeros = np.zeros_like(np.take(cum, [0], axis=axis))
        cum = np.concatenate([zeros, cum], axis=axis)   # cum[k] = sum(padded[:k])
        n = out.shape[axis]
        hi: list[slice] = [slice(None), slice(None)]
        lo: list[slice] = [slice(None), slice(None)]
        hi[axis] = slice(2 * radius + 1, 2 * radius + 1 + n)
        lo[axis] = slice(0, n)
        out = (cum[tuple(hi)] - cum[tuple(lo)]) / float(2 * radius + 1)
    return out


def refine_alpha(mask: np.ndarray, erode: int = 0, dilate: int = 0,
                 feather: int = 0) -> np.ndarray:
    """Binary mask -> float32 alpha in [0, 1].

    Order is erode → dilate → feather, and it matters: eroding *first* pulls the
    boundary inside the subject so the feather ramp is cut out of foreground
    pixels instead of straddling background ones. That single step is what
    removes the halo on a busy background; `dilate` exists for the opposite case
    (a model that cuts a thin subject too tight, e.g. hair or wire).

    Two box blurs rather than one: repeated box filters converge on a Gaussian,
    and a single box leaves a visibly linear ramp with a hard corner at each end.
    """
    alpha = mask.astype(np.float32)
    if erode > 0:
        alpha = _extremum(alpha, erode, take_max=False)
    if dilate > 0:
        alpha = _extremum(alpha, dilate, take_max=True)
    if feather > 0:
        alpha = box_blur(box_blur(alpha, feather), feather)
    return np.clip(alpha, 0.0, 1.0)


def bleed_color(rgb: np.ndarray, core: np.ndarray, need: np.ndarray | None = None,
                radius: int = 2, max_iterations: int = 64) -> np.ndarray:
    """Push foreground colour outward into the background, by diffusion.

    This is the 黑邊 fix. A feathered alpha makes edge pixels ~50 % opaque, but
    their RGB is still the *background* (a dark tree, a black backdrop), so
    compositing over white shows a dark rim that no amount of mask tuning
    removes. Each iteration grows the known region by `radius` using a weighted
    box blur — a cheap push-pull inpaint — so the soft band ends up holding an
    extrapolation of the subject's own colour.

    `need` is the region that must end up filled (in practice "every pixel with
    any opacity"), and the loop runs until it is covered rather than for a fixed
    count. A fixed count is the bug this signature exists to prevent: the ramp
    from a wide feather reaches ~2× the feather radius past the solid core, so
    any hard-coded number of passes silently leaves the *outermost*, most
    visible rim un-bled — exactly the pixels the fix is for.
    """
    out = rgb.astype(np.float32).copy()
    known = core.astype(bool).copy()
    if not known.any():
        return rgb.astype(np.uint8, copy=False)
    out[~known] = 0.0
    target = np.ones(known.shape, dtype=bool) if need is None else need.astype(bool)
    for _ in range(max(0, max_iterations)):
        if not (target & ~known).any():
            break
        weight = box_blur(known.astype(np.float32), radius)
        fill = (weight > 1e-6) & ~known
        if not fill.any():
            break
        for c in range(out.shape[2]):
            est = box_blur(out[:, :, c], radius) / np.maximum(weight, 1e-6)
            out[:, :, c][fill] = est[fill]
        known |= fill
    return np.clip(out, 0, 255).astype(np.uint8)


def compose_rgba(rgb: np.ndarray, alpha: np.ndarray, bleed: bool = True) -> np.ndarray:
    """(H, W, 3) uint8 RGB + float alpha -> (H, W, 4) uint8 RGBA.

    `bleed` is skipped when the alpha is already hard (no semi-transparent
    pixels): there is no band to fill, and the diffusion would only burn time.
    """
    a = np.clip(alpha, 0.0, 1.0)
    soft = np.any((a > 1e-3) & (a < 1.0 - 1e-3))
    color = (bleed_color(rgb, a >= 1.0 - 1e-3, need=a > 1e-3) if (bleed and soft)
             else rgb.astype(np.uint8))
    a8 = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    return np.dstack([color, a8])


def encode_mask_l(alpha: np.ndarray, binary_threshold: float | None = None) -> np.ndarray:
    """Alpha -> the single-channel 0..255 mask PNG that trainers' `--masks` want.

    With a threshold the output is hard 0/255 (what a mask *predicate* means);
    without one the soft ramp is preserved for consumers that weight by it.
    """
    a = np.clip(alpha, 0.0, 1.0)
    if binary_threshold is not None:
        return ((a >= binary_threshold).astype(np.uint8) * 255)
    return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
