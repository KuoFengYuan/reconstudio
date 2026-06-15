#!/usr/bin/env python3
"""Depth Anything 3 monocular-depth inference → LichtFeld-Studio `depth/` folder.

Runs INSIDE the depth model's own conda env (needs torch + the model package +
pillow). The Recon Studio panel itself stays torch-free and only spawns this as
a subprocess (see ``pipeline/depth.py``), exactly like the trainer backends.

Output format follows LichtFeld-Studio's depth convention so that its
``--use-depth-loss`` discovers the maps automatically (it scans
``<dataset>/depth`` | ``<dataset>/depths`` and matches by image stem):

  * one 8-bit grayscale **PNG** per input image,
  * the **same pixel size** as the source image,
  * the **same relative path + stem** under ``--out`` (the dataset's ``depth/``).

8-bit is deliberate: LichtFeld loads images as ``UInt8`` and divides depth by 255
to get ``[0,1]``; its depth loss (``pearson`` / ``adaptive-warped-l1``) is scale-
and orientation-invariant (it derives a sign from the correlation), so a per-image
min-max normalization is all that is required — storing depth vs. inverse-depth
does not change the result.

Two model backends are supported, auto-detected by ``load_predictor()``:

  * **Depth Anything 3** (preferred) — its own package
    ``from depth_anything_3.api import DepthAnything3``; monocular checkpoints are
    ``da3mono-large`` / ``da3metric-large`` (or any ``depth-anything/DA3*`` id).
  * **transformers fallback** — ``AutoModelForDepthEstimation`` (Depth Anything V2
    and any other HF depth-estimation checkpoint), used when the DA3 package is
    not importable in this env.

Progress is printed as newline-terminated ``[depth] i/N <relpath>`` lines (parsed
by jobs.py); run with ``python -u`` so they stream live.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DA3_PROCESS_RES = 504          # Depth Anything 3 default working resolution


def log(msg: str = "") -> None:
    print(msg, flush=True)


def list_images(root: Path, recurse: bool) -> list[Path]:
    it = root.rglob("*") if recurse else root.iterdir()
    imgs = [p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(imgs)


def _load_da3(model_id: str, device: str):
    """Depth Anything 3 native API. ``predict(pil)`` -> HxW float32 ndarray."""
    import numpy as np
    from depth_anything_3.api import DepthAnything3

    # slug id ("depth-anything/da3mono-large") vs bare model_name ("da3mono-large")
    if "/" in model_id:
        model = DepthAnything3.from_pretrained(model_id)
    else:
        model = DepthAnything3(model_name=model_id)
    model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    def predict(pil):
        pred = model.inference(image=[pil], process_res=DA3_PROCESS_RES,
                               process_res_method="upper_bound_resize")
        return np.asarray(pred.depth[0], dtype=np.float32)   # depth: (N,H,W)

    return predict


def _load_hf(model_id: str, device: str):
    """transformers depth-estimation fallback (Depth Anything V2, …)."""
    import numpy as np
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    def predict(pil):
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            pred = model(**inputs).predicted_depth          # (1,h,w)
        if pred.ndim == 4:
            pred = pred.squeeze(1)
        if pred.ndim == 3:
            pred = pred[0]
        return pred.detach().cpu().numpy().astype(np.float32)

    return predict


def load_predictor(model_id: str, device: str):
    """Return a ``predict(pil_rgb) -> HxW float32 ndarray`` (model's native
    resolution; the caller resizes to each source image's size). Prefers Depth
    Anything 3's own package, falling back to the transformers interface."""
    try:
        import depth_anything_3.api  # noqa: F401
    except Exception:
        log("[depth] depth_anything_3 not importable — using transformers fallback.")
        return _load_hf(model_id, device), "transformers"
    return _load_da3(model_id, device), "depth-anything-3"


def resize_depth(depth, size):
    """Resize a HxW float array to ``size`` = (W, H) with bicubic interpolation,
    dependency-light via PIL's 32-bit float mode (no torch/cv2 needed here)."""
    import numpy as np
    from PIL import Image

    w, h = size
    if depth.shape[1] == w and depth.shape[0] == h:
        return depth
    im = Image.fromarray(depth.astype(np.float32), mode="F").resize((w, h), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32)


def to_png8(depth, invert: bool):
    """Per-image min-max normalize a HxW float array to an 8-bit 'L' PIL image."""
    import numpy as np
    from PIL import Image

    d = depth.astype(np.float32)
    lo, hi = float(np.nanmin(d)), float(np.nanmax(d))
    norm = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
    if invert:                                          # store metric-like depth (near=dark)
        norm = 1.0 - norm
    arr = np.clip(norm * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def main() -> int:
    ap = argparse.ArgumentParser(description="Depth Anything 3 -> depth/ folder")
    ap.add_argument("--images", required=True, help="source image folder")
    ap.add_argument("--out", required=True, help="output depth folder (e.g. <dataset>/depth)")
    ap.add_argument("--model", required=True, help="DA3 model name/id or HF model id")
    ap.add_argument("--recurse", action="store_true", help="recurse into subfolders")
    ap.add_argument("--invert", action="store_true",
                    help="store 1-normalized (metric-like) instead of raw model output")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo maps that already exist (default: skip them — resumable)")
    args = ap.parse_args()

    from PIL import Image

    images_dir = Path(args.images).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    if not images_dir.is_dir():
        log(f"[depth] ERROR: images folder not found: {images_dir}")
        return 2
    if out_dir == images_dir:
        log("[depth] ERROR: --out must differ from --images (would overwrite originals)")
        return 2

    imgs = list_images(images_dir, args.recurse)
    if not imgs:
        if not args.recurse and list_images(images_dir, recurse=True):
            log(f"[depth] ERROR: no images at the top level of {images_dir}, but there "
                "are images in subfolders — enable 遞迴子資料夾 (--recurse) and re-run.")
        else:
            log(f"[depth] ERROR: no images under {images_dir}")
        return 2

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[depth] model={args.model} device={device} images={len(imgs)} out={out_dir}")
    if device == "cpu":
        log("[depth] WARNING: CUDA not visible — running on CPU (slow).")

    try:
        predict, backend = load_predictor(args.model, device)
        log(f"[depth] backend={backend}")
    except Exception as exc:                              # noqa: BLE001
        log(f"[depth] ERROR loading model '{args.model}': {exc}")
        log("[depth] hint: Depth Anything 3 has no PyPI package — in this env install its "
            "deps and the repo: `pip install xformers torchvision` then "
            "`pip install -e .` inside a clone of github.com/ByteDance-Seed/Depth-Anything-3. "
            "Alternatively use the transformers fallback: `pip install transformers torchvision` "
            "and set the model to a V2 id (depth-anything/Depth-Anything-V2-Large-hf).")
        return 3

    out_dir.mkdir(parents=True, exist_ok=True)
    total, ok, fail, skipped = len(imgs), 0, 0, 0
    for i, src in enumerate(imgs, 1):
        rel = src.relative_to(images_dir)
        dst = (out_dir / rel).with_suffix(".png")
        if dst.exists() and not args.overwrite:
            skipped += 1
            log(f"[depth] {i}/{total} {rel} (skip, exists)")
            continue
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                depth = resize_depth(predict(im), im.size)
            dst.parent.mkdir(parents=True, exist_ok=True)
            to_png8(depth, args.invert).save(dst)
            ok += 1
            log(f"[depth] {i}/{total} {rel}")
        except Exception as exc:                          # noqa: BLE001
            fail += 1
            log(f"[depth] {i}/{total} {rel} FAILED: {exc}")

    log(f"[depth] done: {out_dir} ({ok} written, {skipped} skipped, {fail} failed)")
    return 1 if ok == 0 and fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
