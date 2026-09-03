"""MoGe-3 depth/normal generation, drop-in compatible with LichtFeld `preprocess`.

LichtFeld-Studio can only run MoGe-2: it links no ONNX runtime, its `preprocess`
is a hand-written C++/CUDA implementation of the MoGe-2 ViT-B/14 graph, and its
`--model` flag swaps that graph's *weights*, not its architecture. MoGe-3 is a
different architecture (a sparse-3D-convolution refiner), so it cannot be loaded
there. This script bypasses `preprocess` and produces the same two folders with
the official PyTorch MoGe-3 instead.

Runs inside its own `moge3` conda env (torch + microsoft/MoGe); driven by
pipeline/moge3.py. The encoding itself lives in pipeline/moge3_encode.py so the
app's test suite can pin it without torch installed.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path


def _load_encoder():
    """Load pipeline/moge3_encode.py by path rather than as `pipeline.moge3_encode`.

    Importing it through the package would execute pipeline/__init__.py, which
    pulls in the whole app (pydantic-settings, the backends registry, …) — none
    of which exists in the `moge3` env this script runs in. The module itself
    has no intra-package imports, so loading the file directly is safe.
    """
    path = Path(__file__).resolve().parent.parent / "pipeline" / "moge3_encode.py"
    spec = importlib.util.spec_from_file_location("moge3_encode", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the shared encoder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_enc = _load_encoder()
IMAGE_EXTS = _enc.IMAGE_EXTS
MAX_VALUE = _enc.MAX_VALUE
OUTPUT_DIR_NAMES = _enc.OUTPUT_DIR_NAMES
encode_depth = _enc.encode_depth
encode_normals = _enc.encode_normals

DEFAULT_MODEL = "Ruicheng/moge-3-vitl"


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        and p.relative_to(images_dir).parts[0] not in OUTPUT_DIR_NAMES
    )


def output_path_for(dataset_root: Path, folder: str, image: Path, images_dir: Path) -> Path:
    """Port of LichtFeld's output_path_for: mirror the image's relative path
    under the dataset root, with the extension replaced by .png."""
    return dataset_root / folder / image.relative_to(images_dir).with_suffix(".png")


def main() -> int:
    ap = argparse.ArgumentParser(description="MoGe-3 depth/normal maps for LichtFeld")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--images", default="images",
                    help="image folder under dataset_root ('' = the root itself)")
    ap.add_argument("--mode", choices=("depth", "normal", "both"), default="both")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HuggingFace id or local .pt checkpoint")
    ap.add_argument("--bit-depth", type=int, choices=(8, 16), default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import cv2
    import torch

    images_dir = args.dataset_root / args.images if args.images else args.dataset_root
    if not images_dir.is_dir():
        print(f"error: images folder not found: {images_dir}", file=sys.stderr)
        return 2
    images = list_images(images_dir)
    if not images:
        print(f"error: no images under {images_dir}", file=sys.stderr)
        return 2

    want_depth = args.mode in ("depth", "both")
    want_normal = args.mode in ("normal", "both")
    max_value = MAX_VALUE[args.bit_depth]

    if not torch.cuda.is_available():
        print("error: CUDA is not available in this environment", file=sys.stderr)
        return 3
    device = torch.device("cuda")

    def save_png(path: Path, array) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # cv2 writes BGR; our normal channels are XYZ in RGB order, so reverse
        # them on the way out to land the same bytes LichtFeld's writer does.
        data = array[:, :, ::-1] if array.ndim == 3 else array
        if not cv2.imwrite(str(path), data):
            raise RuntimeError(f"failed to write {path}")

    from moge.model.v3 import MoGeModel
    print(f"Model: {args.model}", flush=True)
    model = MoGeModel.from_pretrained(args.model).to(device).eval()
    # The three lines the panel's depth progress parser keys on (_DE_IMAGES /
    # _DE_CUR / _DE_DONE in jobs.py) are LichtFeld preprocess's own wording. This
    # engine reproduces them verbatim so one parser drives both engines' UI.
    print(f"Images: {len(images)} under {images_dir}", flush=True)
    print(f"Mode: {args.mode} | bit depth: {args.bit_depth}", flush=True)

    written = skipped = 0
    for i, image_path in enumerate(images, 1):
        depth_path = output_path_for(args.dataset_root, "depth", image_path, images_dir)
        normal_path = output_path_for(args.dataset_root, "normals", image_path, images_dir)
        if (not args.overwrite
                and (not want_depth or depth_path.is_file())
                and (not want_normal or normal_path.is_file())):
            skipped += 1
            continue

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[{i}/{len(images)}] skip unreadable {image_path.name}", flush=True)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(rgb / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

        started = time.perf_counter()
        with torch.no_grad():
            out = model.infer(tensor)
        elapsed_ms = (time.perf_counter() - started) * 1000

        mask = out["mask"].cpu().numpy().astype(bool)
        if want_depth:
            # Depth comes from the point map's Z, the same channel LichtFeld
            # reads; MoGe's own "depth" key is the same quantity, but going
            # through points keeps this a faithful port rather than a lookalike.
            points = out["points"].float().cpu().numpy()
            save_png(depth_path, encode_depth(points[:, :, 2], mask, max_value))
        if want_normal:
            if out.get("normal") is None:
                print("error: this checkpoint returns no normals; use a model that has them",
                      file=sys.stderr)
                return 4
            normals = out["normal"].float().cpu().numpy()
            save_png(normal_path, encode_normals(normals, mask, max_value))
        written += 1
        print(f"[{i}/{len(images)}] {image_path.name}  inference {elapsed_ms:.0f} ms",
              flush=True)

    print(f"Done. processed={written} skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
