"""SAM 去背: batch bounding-box prompting -> RGBA cut-outs + foreground masks.

Runs inside its own `sam` conda env (torch + a SAM package + opencv); driven by
pipeline/matte.py, but fully usable on its own:

    python tools/sam_matte.py /data/scene --images images \
        --engine sam2 --boxes json --boxes-json boxes.json --feather 2

Design notes that are easy to get wrong:

* **One encoder pass, N boxes.** The image encoder is ~95 % of the cost and does
  not depend on the prompt, so every engine here calls `set_image()` once per
  photo and hands the *whole* box array to the decoder in one call. Looping
  `predict()` per box would re-run nothing but the decoder, yet the SAM 2 / SAM 1
  APIs make that mistake easy — hence `masks_for_boxes()` taking an (N, 4) array
  as its only shape.
* **The union is the output.** N boxes give N masks; the alpha channel gets their
  logical OR, so "keep all the subjects in this row" is one image, not N.
* **The maths lives elsewhere.** Merging, morphology, feathering and the colour
  bleed are in pipeline/matte_encode.py so the test suite can pin them without
  torch — see the module docstring there for why the bleed exists.
* **Non-destructive**, like the depth stage: only `cutout/` and `masks/` are
  created next to `images/`; source photos are never touched.

The progress lines ("Images: N under …", "[i/N] name", "Done. processed=…") are
LichtFeld preprocess's wording on purpose: jobs.py already parses that vocabulary
for the depth stage, so the panel's progress bar works here with no new parser
dialect (see _parse_matte).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path


def _load_shared():
    """Load pipeline/matte_encode.py by path, not as `pipeline.matte_encode`.

    Importing through the package would execute pipeline/__init__.py, which pulls
    in the whole app (pydantic-settings, the backends registry, …) — none of which
    exists in the `sam` env. matte_encode has no intra-package imports, so loading
    the file directly is safe. Same trick as tools/moge3_preprocess.py.
    """
    path = Path(__file__).resolve().parent.parent / "pipeline" / "matte_encode.py"
    spec = importlib.util.spec_from_file_location("matte_encode", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the shared matting maths from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_enc = _load_shared()
IMAGE_EXTS = _enc.IMAGE_EXTS
SKIP_DIR_NAMES = _enc.SKIP_DIR_NAMES
OUTPUT_ROOT = _enc.OUTPUT_ROOT

DEFAULT_MODELS = {
    "sam2": "facebook/sam2.1-hiera-large",
    "sam3": "facebook/sam3",
    "sam1": "vit_h",            # segment-anything needs a local .pth too (--checkpoint)
}
DEFAULT_TRACK_MODEL = "facebook/sam2.1-hiera-large"


# --------------------------------------------------------------------------- #
# image discovery / IO
# --------------------------------------------------------------------------- #
def list_images(images_dir: Path) -> list[Path]:
    """Every photo under `images_dir`, skipping our own outputs and COLMAP's tree.

    The skip test walks the WHOLE relative path, not just its first component: a
    COLMAP workspace nests the undistorted copies several levels down
    (`colmap/training_dataset_*/images/…`), and checking only the top level would
    let all of them in.
    """
    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        and not (SKIP_DIR_NAMES & set(p.relative_to(images_dir).parts[:-1]))
    )


def output_path_for(dataset_root: Path, folder: str, image: Path, images_dir: Path) -> Path:
    """Mirror the image's relative path under <root>/no_bg/<folder>, ext -> .png.

    The relative path and the .png are required by COLMAP's mask pass:
    `replace_images_by_masks` rewrites every image name in the model to `.png`,
    so a mask that isn't at exactly that relative path is simply not found. Only
    the `no_bg/` prefix is ours — it keeps both output folders out of the photo
    directory itself.
    """
    return (dataset_root / OUTPUT_ROOT / folder
            / image.relative_to(images_dir).with_suffix(".png"))


_EXIF_ORIENTATION = 274          # the tag id, so PIL's ExifTags table isn't needed


def _exif_orientation(path: Path) -> int:
    """The EXIF Orientation tag, or 1 when there isn't one."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int((im.getexif() or {}).get(_EXIF_ORIENTATION, 1) or 1)
    except Exception:
        return 1


def _image_size(path: Path) -> tuple[int, int] | None:
    """(h, w) **as cv2 will hand it to us** — header only, no full decode.

    The EXIF swap is the whole point. Phone and DSLR "portrait" shots are very
    often landscape files carrying an Orientation tag, and the two readers in
    this pipeline disagree about them: cv2.imread applies the rotation, PIL's
    .height/.width report the stored pixels. Grouping the sequence by PIL's
    numbers while compositing against cv2's is what turned a mixed-orientation
    folder into perfectly plausible, completely wrong masks.
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            h, w = im.height, im.width
    except Exception:
        return None
    if _exif_orientation(path) in (5, 6, 7, 8):     # 90°/270° rotations swap the axes
        h, w = w, h
    return (h, w)


def read_rgb(path: Path):
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_png(path: Path, array) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 3 and array.shape[2] == 4:
        data = array[:, :, [2, 1, 0, 3]]          # RGBA -> BGRA
    elif array.ndim == 3:
        data = array[:, :, ::-1]                  # RGB -> BGR
    else:
        data = array
    if not cv2.imwrite(str(path), data):
        raise RuntimeError(f"failed to write {path}")


# --------------------------------------------------------------------------- #
# box sources
# --------------------------------------------------------------------------- #
def load_boxes_file(path: Path) -> dict:
    """Normalise every accepted boxes-JSON shape into {ref, apply, boxes, per_image}.

    Three shapes are accepted because three things write this file: the panel's
    box picker (full dict), a hand-written per-image map, and a bare list from a
    script that just wants the same crop everywhere.
    """
    raw = json.loads(Path(path).read_text())
    out = {"ref": "", "apply": "all", "norm": False, "boxes": [], "per_image": {},
           "refs": {}, "only": []}
    if isinstance(raw, list):
        out["boxes"] = raw
        return out
    if not isinstance(raw, dict):
        raise ValueError(f"unsupported boxes JSON: {path}")
    if not ({"boxes", "per_image", "refs", "only"} & set(raw)):
        out["per_image"] = raw            # bare {rel: [[...]]} map
        return out
    keys = ("ref", "apply", "norm", "boxes", "per_image", "refs", "only")
    out.update({k: raw[k] for k in keys if k in raw})
    # `refs` is the multi-frame seed map: the SAME object boxed on several frames.
    # Single-frame files predate it, so synthesise it rather than special-casing
    # every reader below.
    if not out["refs"] and out["ref"] and out["boxes"]:
        out["refs"] = {out["ref"]: out["boxes"]}
    return out


def denorm(boxes, spec: dict, width: int, height: int):
    """Scale 0..1 boxes to this image's pixels when the file says they're normalised.

    The panel's box picker stores normalised corners on purpose: it draws on a
    downscaled JPEG preview, and a folder may mix resolutions (a rig with two
    camera models, or a FullHD-resized subset). Pixel coordinates from the
    preview would be silently wrong in both cases — off by the preview scale, or
    off for every image that isn't the reference.
    """
    if not spec.get("norm") or len(boxes) == 0:
        return boxes
    return _enc.scale_boxes(boxes, float(width), float(height))


def boxes_for_image(spec: dict, rel: str, width: int, height: int):
    """Per-image boxes from a loaded boxes file, honouring apply=all|ref."""
    per = spec.get("per_image") or {}
    entry = per.get(rel)
    if isinstance(entry, dict):
        entry = entry.get("boxes") or []
    if entry is not None and rel in per:
        return denorm(_enc.as_boxes(entry), spec, width, height)
    if spec.get("apply") == "ref" and spec.get("ref") and rel != spec["ref"]:
        return _enc.as_boxes([])
    return denorm(_enc.as_boxes(spec.get("boxes") or []), spec, width, height)


def points_for_image(spec: dict, rel: str, width: int, height: int):
    """(coords, labels) for the repair prompts on this image, or (None, None).

    Points are how SAM is *corrected*: a box says "the subject is roughly here",
    which is exactly the information that already failed if the cut-out came out
    wrong. A click labelled 1 says "this pixel is subject" and 0 says "this pixel
    is not", which is new information and usually fixes a bad mask in one or two
    clicks. Stored as [x, y, label] triples alongside the box.
    """
    import numpy as np
    entry = (spec.get("per_image") or {}).get(rel)
    if not isinstance(entry, dict) or not entry.get("points"):
        return None, None
    raw = entry["points"]
    coords = np.asarray([[float(p[0]), float(p[1])] for p in raw], dtype=np.float32)
    labels = np.asarray([int(p[2]) for p in raw], dtype=np.int32)
    if spec.get("norm") and len(coords):
        coords = coords * np.asarray([float(width), float(height)], dtype=np.float32)
    return coords, labels


def auto_row_boxes(rgb, min_area_frac: float):
    """Detector-free fallback: threshold against the background, then component boxes.

    Works for the studio/turntable case the classic "row of products on a plain
    sweep" shot is — no extra model, no download. It is a *heuristic*: on a busy
    natural background it will return one huge box, which the min-area and row
    filters cannot rescue, so it is never the default.
    """
    import cv2
    import numpy as np
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu both ways: the subjects may be darker OR lighter than the sweep, and
    # picking the polarity by which one covers less of the frame beats guessing.
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, light = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = dark if dark.mean() < light.mean() else light
    kernel = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    num, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    h, w = gray.shape
    boxes = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < max(1.0, min_area_frac * h * w):
            continue
        boxes.append([x, y, x + bw, y + bh])
    return _enc.as_boxes(boxes)


class GroundingDinoDetector:
    """Text -> boxes for the engines that cannot read a prompt themselves.

    SAM 1 / SAM 2 are promptable *segmenters*, not detectors: they turn a box
    into a mask but have no idea what a "parrot" is. Grounding DINO supplies the
    boxes, which is the standard "Grounded-SAM" pairing. SAM 3 needs none of
    this — it takes the phrase directly.
    """

    def __init__(self, model_id: str, device, box_threshold: float, text_threshold: float):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def __call__(self, rgb, text: str):
        import torch
        from PIL import Image
        # Grounding DINO wants lower-case phrases, each terminated by a period.
        phrase = ". ".join(t.strip().strip(".") for t in text.split(",") if t.strip()).lower()
        if phrase and not phrase.endswith("."):
            phrase += "."
        inputs = self.proc(images=Image.fromarray(rgb), text=phrase,
                           return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        sizes = [rgb.shape[:2]]
        try:
            res = self.proc.post_process_grounded_object_detection(
                out, inputs["input_ids"], box_threshold=self.box_threshold,
                text_threshold=self.text_threshold, target_sizes=sizes)[0]
        except TypeError:
            # transformers >= 4.51 renamed box_threshold -> threshold.
            res = self.proc.post_process_grounded_object_detection(
                out, inputs["input_ids"], threshold=self.box_threshold,
                text_threshold=self.text_threshold, target_sizes=sizes)[0]
        boxes = res["boxes"].detach().cpu().numpy()
        scores = res["scores"].detach().cpu().numpy() if "scores" in res else None
        return _enc.as_boxes(boxes), scores


# --------------------------------------------------------------------------- #
# model runners  (swap a checkpoint or a whole SAM variant by adding one class)
# --------------------------------------------------------------------------- #
class MaskRunner:
    """The only interface the pipeline below knows about.

    Everything model-specific — package name, checkpoint format, prompt encoding,
    output layout — is confined to a subclass, so a new SAM variant is a new class
    plus one line in `build_runner()`; nothing in the batching, merging or output
    code changes.
    """

    name = "base"
    takes_text = False          # can it turn a phrase into masks with no detector?

    def set_image(self, rgb) -> None:
        raise NotImplementedError

    def masks_for_boxes(self, boxes):
        """(N, 4) xyxy -> (N, H, W) bool, in ONE decoder call."""
        raise NotImplementedError

    def masks_for_text(self, rgb, text: str):
        """(masks, boxes, scores) for a phrase — text-native engines only."""
        raise NotImplementedError

    def masks_for_prompt(self, boxes, points, labels):
        """ONE object from a box plus +/- clicks — the repair path.

        Separate from `masks_for_boxes` because the shapes mean different things:
        there, N boxes are N independent objects decoded together; here a single
        object is being argued about, and the box and the clicks must reach the
        decoder as one prompt or the corrections are ignored.
        """
        raise NotImplementedError


def _as_nhw(masks, hw):
    """Normalise every SAM's mask output to a bool (N, H, W) numpy array.

    The shape differs per engine and per prompt count — (H, W), (C, H, W),
    (N, 1, H, W) — and guessing wrong yields a silently empty union rather than
    an exception, so the normalisation is explicit and central.
    """
    import numpy as np
    if hasattr(masks, "detach"):
        masks = masks.detach().float().cpu().numpy()
    arr = np.asarray(masks)
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 4:
        arr = arr[:, 0]
    if arr.shape[-2:] != tuple(hw):
        raise RuntimeError(f"mask shape {arr.shape} does not match image {tuple(hw)}")
    return arr > 0.0 if arr.dtype != bool else arr


class Sam2Runner(MaskRunner):
    """SAM 2.1 image predictor (`facebook/sam2.1-hiera-*`) — the default.

    `predict(box=(N,4))` is genuinely batched: `set_image` already ran the
    encoder, and the N boxes go through the prompt encoder + mask decoder
    together.
    """

    name = "sam2"

    def __init__(self, model_id: str, device, dtype):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.p = (SAM2ImagePredictor.from_pretrained(model_id) if "/" in model_id
                  else self._from_config(model_id))
        self.p.model.to(device)
        self.device, self.dtype = device, dtype
        self._hw = (0, 0)

    @staticmethod
    def _from_config(path: str):
        """Local checkpoint form: `<config.yaml>:<checkpoint.pt>`."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        cfg, _, ckpt = path.partition(":")
        if not ckpt:
            raise ValueError("local SAM 2 model must be '<config.yaml>:<checkpoint.pt>'")
        return SAM2ImagePredictor(build_sam2(cfg, ckpt))

    def set_image(self, rgb) -> None:
        import torch
        self._hw = rgb.shape[:2]
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.dtype):
            self.p.set_image(rgb)

    def masks_for_boxes(self, boxes):
        import torch
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.dtype):
            masks, _, _ = self.p.predict(point_coords=None, point_labels=None,
                                         box=boxes, multimask_output=False)
        return _as_nhw(masks, self._hw)

    def masks_for_prompt(self, boxes, points, labels):
        import torch
        box = boxes[0] if boxes is not None and len(boxes) else None
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.dtype):
            masks, _, _ = self.p.predict(point_coords=points, point_labels=labels,
                                         box=box, multimask_output=False)
        return _as_nhw(masks, self._hw)

    def masks_for_image_batch(self, rgbs, boxes_per_image):
        """One encoder pass for N *photos* — `masks_for_boxes` batches boxes
        within one photo, this batches the photos themselves.

        `set_image_batch` + `predict_batch` is the only across-images path SAM 2
        offers, and it is worth having because the encoder is ~95 % of the cost
        and a single 1024² frame does not fill this GPU. Mixed sizes are fine:
        the predictor keeps each entry's own original (h, w) and un-normalises
        each box list against it, which is why the boxes stay in each photo's
        own pixel frame here.

        MEASURED, because "plausible but silently misplaced" is the failure this
        file fears most (RTX PRO 6000, sam2.1-hiera-large, bf16 autocast, real
        3475×4633 frames + resized portrait/landscape copies):

        * Geometry is right. Batches of 2-3 — including ones deliberately mixing
          portrait, landscape and a second resolution — come back **bit-identical**
          to the one-at-a-time path, so nothing is being transformed against the
          wrong entry's shape. The one-at-a-time path is itself reproducible
          run-to-run (diff = 0), so that comparison means something.
        * From 4 photos up, masks differ **at the boundary only**: worst IoU
          ~0.97, and of the ~15-31 k differing pixels per 16 MP frame, only
          125-234 sit more than 3 px from the mask edge. After the pipeline's own
          `--erode 1 --feather 2`, mean |Δalpha| is ~0.001-0.002. That is batched
          matmul reduction order in bf16 flipping pixels whose logit sits on the
          0.0 threshold — inherent to batched GPU inference, an order of magnitude
          smaller than the feather it then passes through, and neither version is
          "the correct one". `--image-batch 1` restores bit-reproducibility.

        Returns one (N_boxes, h, w) bool array per input photo, in order.
        """
        import torch
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.dtype):
            self.p.set_image_batch(list(rgbs))
            masks, _, _ = self.p.predict_batch(
                point_coords_batch=None, point_labels_batch=None,
                box_batch=list(boxes_per_image), multimask_output=False)
        # strict: one mask set per photo, or the pairing below would silently
        # attach a photo's masks to a different photo's geometry.
        return [_as_nhw(m, rgb.shape[:2])
                for m, rgb in zip(masks, rgbs, strict=True)]


class Sam1Runner(MaskRunner):
    """Original segment-anything. Batched boxes go through `predict_torch`.

    `SamPredictor.predict()` takes ONE box; the batched path is
    `apply_boxes_torch` + `predict_torch`, which is why this class exists at all.
    """

    name = "sam1"

    def __init__(self, model_type: str, checkpoint: str, device, dtype):
        from segment_anything import SamPredictor, sam_model_registry
        if not checkpoint:
            raise ValueError("engine sam1 needs --checkpoint (e.g. sam_vit_h_4b8939.pth)")
        sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
        self.p = SamPredictor(sam)
        self.device, self.dtype = device, dtype
        self._hw = (0, 0)

    def set_image(self, rgb) -> None:
        self._hw = rgb.shape[:2]
        self.p.set_image(rgb)

    def masks_for_prompt(self, boxes, points, labels):
        box = boxes[0] if boxes is not None and len(boxes) else None
        masks, _, _ = self.p.predict(point_coords=points, point_labels=labels,
                                     box=box, multimask_output=False)
        return _as_nhw(masks, self._hw)

    def masks_for_boxes(self, boxes):
        import torch
        t = torch.as_tensor(boxes, dtype=torch.float, device=self.device)
        t = self.p.transform.apply_boxes_torch(t, self._hw)
        with torch.inference_mode():
            masks, _, _ = self.p.predict_torch(point_coords=None, point_labels=None,
                                               boxes=t, multimask_output=False)
        return _as_nhw(masks, self._hw)


class Sam3Runner(MaskRunner):
    """SAM 3 concept prompting: a phrase, or a drawn box as a visual *exemplar*.

    This is the mode in the product screenshot — box one parrot, get every
    parrot — and unlike SAM 1/2 it needs no separate detector, because SAM 3
    returns all instances matching the concept along with their boxes.

    ADAPTER NOTE: SAM 3's Python surface is newer than the rest of this file and
    has moved once already (standalone `sam3` package vs. the `transformers`
    integration). Everything version-specific is inside this one class: if a
    release renames something, `_load` and `_run` are the only things to edit.
    """

    name = "sam3"
    takes_text = True

    def __init__(self, model_id: str, device, dtype, score_threshold: float = 0.4):
        self.device, self.dtype = device, dtype
        self.score_threshold = score_threshold
        self._rgb = None
        self._load(model_id)

    def _load(self, model_id: str) -> None:
        try:
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:
            raise RuntimeError(
                "engine sam3 需要含 SAM 3 的 transformers:\n"
                "  conda run -n sam pip install -U transformers accelerate\n"
                "(或改用 --engine sam2;SAM 3 的 API 若有變動,只需改 tools/sam_matte.py "
                "的 Sam3Runner。)") from exc
        try:
            self.proc = Sam3Processor.from_pretrained(model_id)
            self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        except OSError as exc:
            # facebook/sam3 is a MANUALLY gated repo: the download 401s until the
            # account has been approved and a token is on this machine. That reads
            # as a network error, so name the actual fix instead.
            if "gated" not in str(exc).lower() and "401" not in str(exc):
                raise
            raise RuntimeError(
                f"{model_id} 是 HuggingFace 的 gated repo,這台機器還沒有存取權。\n"
                f"  1) 到 https://huggingface.co/{model_id} 按同意條款(人工審核,不是即時的)\n"
                "  2) 核准後在這台機器登入一次:\n"
                "       conda run -n sam hf auth login\n"
                "     (或設 HF_TOKEN 環境變數)\n"
                "還沒核准的話,先用 --engine sam2 —— SAM 2.1 沒有 gating,框提示的品質一樣好。"
            ) from exc

    def _run(self, rgb, text: str | None, exemplar_boxes):
        import torch
        from PIL import Image
        kwargs: dict = {"images": Image.fromarray(rgb), "return_tensors": "pt"}
        if text:
            kwargs["text"] = text
        if exemplar_boxes is not None and len(exemplar_boxes):
            # One positive exemplar per drawn box, in the image's own pixel frame.
            kwargs["input_boxes"] = [[[float(v) for v in b] for b in exemplar_boxes]]
            kwargs["input_boxes_labels"] = [[1] * len(exemplar_boxes)]
        inputs = self.proc(**kwargs).to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)
        res = self.proc.post_process_instance_segmentation(
            out, threshold=self.score_threshold, mask_threshold=0.5,
            target_sizes=[rgb.shape[:2]])[0]
        masks = res.get("masks")
        boxes = res.get("boxes")
        scores = res.get("scores")
        import numpy as np
        if masks is None or len(masks) == 0:
            return (np.zeros((0, *rgb.shape[:2]), dtype=bool), _enc.as_boxes([]), None)
        boxes_np = (boxes.detach().cpu().numpy() if hasattr(boxes, "detach")
                    else np.asarray(boxes if boxes is not None else []))
        scores_np = (scores.detach().cpu().numpy() if hasattr(scores, "detach") else scores)
        return _as_nhw(masks, rgb.shape[:2]), _enc.as_boxes(boxes_np), scores_np

    def set_image(self, rgb) -> None:
        # SAM 3 fuses the prompt into the forward pass, so there is no reusable
        # per-image encoding to cache here; keep the frame for the prompt calls.
        self._rgb = rgb

    def masks_for_boxes(self, boxes):
        """Boxes as exemplars — note this can return MORE masks than boxes,
        which is the point: one example, every match."""
        masks, _, _ = self._run(self._rgb, None, boxes)
        return masks

    def masks_for_prompt(self, boxes, points, labels):
        raise RuntimeError(
            "SAM 3 的提示是「概念」(文字或範例框),沒有 +/- 點提示;"
            "修圖請用 --engine sam2。")

    def masks_for_text(self, rgb, text: str):
        return self._run(rgb, text, None)


def build_runner(engine: str, model: str, checkpoint: str, device, dtype,
                 score_threshold: float) -> MaskRunner:
    if engine == "sam2":
        return Sam2Runner(model, device, dtype)
    if engine == "sam1":
        return Sam1Runner(model, checkpoint, device, dtype)
    if engine == "sam3":
        return Sam3Runner(model, device, dtype, score_threshold)
    raise ValueError(f"unknown engine: {engine!r}")


# --------------------------------------------------------------------------- #
# per-image inference, with the OOM ladder
# --------------------------------------------------------------------------- #
def refine_boxes(args, boxes, scores, w: int, h: int):
    """The filter chain every box source feeds through, in the one order that is
    correct: clip to the frame, drop specks, de-duplicate overlaps, then (opt-in)
    keep only the row. Shared by the one-at-a-time loop and the batched one so
    the two cannot drift into producing different boxes for the same photo."""
    boxes = _enc.clip_boxes(boxes, w, h)
    boxes = _enc.filter_small(boxes, args.min_area, w, h)
    if len(boxes) > 1:
        boxes = _enc.suppress_overlaps(
            boxes, scores if scores is not None and len(scores) == len(boxes) else None)
    if args.row_filter:
        boxes = _enc.select_row_boxes(boxes)
    return boxes


def masks_in_chunks(runner: MaskRunner, boxes, chunk: int, hw):
    """Decode the boxes in groups, halving the group on CUDA OOM.

    Each decoded mask is a full H×W tensor, so on a 4-6 K photo it is the *mask
    count*, not the encoder, that decides whether the step fits in VRAM — a
    crowded frame is what actually OOMs. Halving and retrying keeps a single busy
    photo from failing the whole folder; only when a chunk of one still cannot
    fit does the error propagate.
    """
    import numpy as np
    import torch
    out: list[np.ndarray] = []
    i, size = 0, max(1, int(chunk))
    while i < len(boxes):
        part = boxes[i:i + size]
        try:
            out.append(runner.masks_for_boxes(part))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if size == 1:
                raise
            size = max(1, size // 2)
            print(f"    [vram] OOM -> retrying with box chunk {size}", flush=True)
            continue
        i += size
    if not out:
        return np.zeros((0, *hw), dtype=bool)
    return np.concatenate(out, axis=0)


def finish_image(rgb, mask, args, dataset_root: Path, image: Path, images_dir: Path) -> None:
    """Mask -> the requested output files. The only place that writes pixels."""
    if mask.shape != rgb.shape[:2]:
        # Track mode's masks all come back at the FIRST frame's geometry, because
        # SAM 2's video predictor resizes the whole sequence to it. A folder that
        # mixes orientations — or carries one stray thumbnail — therefore hands
        # back a mask that does not fit its own photo. Nearest-neighbour, since
        # this is a hard mask and any interpolation would invent edge values.
        import cv2
        import numpy as np
        print(f"    [note] mask {mask.shape} != image {rgb.shape[:2]} "
              f"({image.name}) — resizing", flush=True)
        mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    alpha = _enc.refine_alpha(mask, erode=args.erode, dilate=args.dilate,
                              feather=args.feather)
    if "cutout" in args.outputs:
        write_png(output_path_for(dataset_root, "cutout", image, images_dir),
                  _enc.compose_rgba(rgb, alpha, bleed=not args.no_bleed))
    if "masks" in args.outputs:
        threshold = None if args.soft_masks else 0.5
        write_png(output_path_for(dataset_root, "masks", image, images_dir),
                  _enc.encode_mask_l(alpha, threshold))


def outputs_present(args, dataset_root: Path, image: Path, images_dir: Path) -> bool:
    return all(output_path_for(dataset_root, folder, image, images_dir).is_file()
               for folder in args.outputs)


def default_write_workers() -> int:
    """A polite CPU-count cap, like pipeline.config.resolved_resize_workers.

    Capped well below the core count on purpose: the numpy half of the write is
    single-threaded (so threads help), but cv2's decode/encode is ALREADY
    internally multi-threaded, and one pool thread per core would have the two
    fighting each other for the same cores plus the memory bandwidth. Eight
    covers most of the win; --write-workers raises it for very large photos.
    """
    import os
    return max(1, min(os.cpu_count() or 8, 8))


class WriteFanout:
    """`finish_image` across a thread pool — where a run actually spends its time.

    Measured on a 3475×4633 iPhone frame: SAM's own work is a few hundred ms,
    while writing that frame out is ~5.3 s — 4.3 s of it inside `compose_rgba`'s
    colour bleed, which is single-threaded numpy. Multiply by a few hundred
    frames and the GPU is idle for the better part of an hour. Every photo's
    write is independent (one frame in, its own two PNGs out) and both halves
    release the GIL (cv2 decode/encode, numpy elementwise), so threads scale it
    nearly linearly.

    Two things this owns deliberately:

    * **The image read.** Workers re-read the frame from disk rather than having
      the caller hand over the decoded array. A decode is ~0.1 s against ~5 s of
      compositing, and keeping pixels out of the queue is what bounds memory to
      `workers` frames instead of "however far the GPU has run ahead" — the
      producer is 10-20x faster than the writers, so an unbounded queue would
      grow until it filled RAM.
    * **Every `[i/N]` line.** The counter is *completion order*, not the photo's
      index in the folder: jobs.py sets the panel's progress from whatever
      `[i/N]` it last saw, so N threads reporting their own indices would make
      the bar jump around. The filename on the line is what identifies a photo.
    """

    def __init__(self, args, dataset_root: Path, images_dir: Path, total: int,
                 workers: int) -> None:
        from concurrent.futures import ThreadPoolExecutor
        self.args = args
        self.dataset_root = dataset_root
        self.images_dir = images_dir
        self.total = total
        self.workers = max(1, int(workers))
        self.written = 0
        self.skipped = 0
        self._done = 0
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=self.workers,
                                        thread_name_prefix="matte-write")
        self._futures: list = []

    # -- reporting ---------------------------------------------------------- #
    def _emit(self, name: str, detail: str) -> None:
        with self._lock:
            self._done += 1
            n = self._done
        print(f"[{n}/{self.total}] {name}  {detail}".rstrip(), flush=True)

    def note(self, image: Path, detail: str, *, skipped: bool = False) -> None:
        """Account for a photo no write was needed for (already done, unreadable)."""
        if skipped:
            with self._lock:
                self.skipped += 1
        self._emit(image.name, detail)

    # -- work --------------------------------------------------------------- #
    def submit(self, image: Path, mask, detail: str = "") -> None:
        self._futures.append(self._pool.submit(self._run_one, image, mask, detail))

    def _run_one(self, image: Path, mask, detail: str) -> None:
        try:
            rgb = read_rgb(image)
            if rgb is None:
                self.note(image, "skip unreadable", skipped=True)
                return
            note = ""
            if mask is None or not mask.any():
                mask, note = handle_empty(self.args, rgb, None)
                if mask is None:
                    self.note(image, note, skipped=True)
                    return
            finish_image(rgb, mask, self.args, self.dataset_root, image, self.images_dir)
            with self._lock:
                self.written += 1
            cover = f"cover={100.0 * float(mask.mean()):.1f}%"
            self._emit(image.name, " ".join(x for x in (detail, cover, note) if x))
        except BaseException as exc:                          # noqa: BLE001 — re-raised in close()
            with self._lock:
                if self._error is None:
                    self._error = exc
            raise

    def close(self) -> tuple[int, int]:
        """Drain the pool and hand back (written, skipped). Re-raises the first
        worker failure: a half-written folder that reported success is exactly
        the silent-partial-output failure this tool warns about elsewhere."""
        try:
            for f in self._futures:
                f.result()
        finally:
            self._pool.shutdown(wait=True)
        if self._error is not None:
            raise self._error
        return self.written, self.skipped


# --------------------------------------------------------------------------- #
# track mode: draw a box on ONE frame, propagate through the sequence
# --------------------------------------------------------------------------- #
def _stage_jpegs(images: list[Path], indices: list[int], staging: Path) -> None:
    """Materialise the frames as the numerically-named JPEG dir SAM 2 wants.

    Symlinks where the source is already JPEG (no copy of a few GB of photos),
    transcode otherwise. The names are the *position in this group*, which is why
    the caller keeps a local→global index map.
    """
    import cv2
    for local, gi in enumerate(indices):
        src = images[gi]
        dst = staging / f"{local:06d}.jpg"
        # Symlink only when SAM 2's reader (PIL, which ignores Orientation) will
        # see the same pixels cv2 gave us. An EXIF-rotated JPEG must be baked out
        # instead, or the tracker works on a sideways image and every mask it
        # returns is rotated with respect to the photo we composite it onto.
        if src.suffix.lower() in (".jpg", ".jpeg") and _exif_orientation(src) == 1:
            dst.symlink_to(src.resolve())
        else:
            bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"unreadable image: {src}")
            cv2.imwrite(str(dst), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def run_track(args, dataset_root: Path, images_dir: Path, images: list[Path],
              spec: dict, device) -> int:
    """SAM 2 video predictor over the image sequence, one session per geometry.

    This is what makes "匡選一次" usable on a folder shot from a moving camera:
    the same boxes reused on every frame (固定框) only works for a locked-off
    camera, whereas memory-attention propagation follows the subject. The frames
    must therefore be an ordered sequence — which they are when they came from
    the frames job.

    **Grouped by image size, because a video predictor takes a video.** SAM 2
    resizes the whole sequence to one geometry and hands every mask back at the
    first frame's shape; feed it a folder that mixes landscape and portrait and
    the odd frames are tracked against a stretched image, producing a mask that
    is not merely the wrong size but the wrong *shape* — silently, since it still
    writes a plausible-looking PNG. So each distinct (h, w) gets its own session,
    and a group with no box of its own is reported rather than guessed at: with
    several seed frames allowed, the fix is simply to box one frame per group.

    Both offload flags are on: without them the whole sequence's features sit in
    VRAM and a few hundred 4 K frames will not fit on a 16 GB card.
    """
    import numpy as np
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    # Always SAM 2.1's video predictor, even when --engine is sam3. SAM 3's own
    # video session (`Sam3TrackerVideoProcessor.init_video_session`) takes box
    # prompts too, but it materialises EVERY frame at 1024² up front — several GB
    # for a few hundred photos — whereas SAM 2's reads them from disk one at a
    # time. Its streaming path (`add_new_frame`) would fix that and is the natural
    # upgrade here.
    if args.engine != "sam2":
        print(f"note: --boxes track 一律使用 SAM 2.1 影片模式(--engine {args.engine} "
              "只影響其它提示方式)", flush=True)

    rels = [p.relative_to(images_dir).as_posix() for p in images]
    # Every annotated frame becomes a CONDITIONING frame. Boxing the same subject
    # on two or three frames spread through the sequence is the cheapest accuracy
    # win here: memory attention propagated from one seed drifts through occlusion
    # and large viewpoint changes, and each extra seed re-anchors it — frames
    # between two seeds are corrected from both sides. Box index is the object
    # identity, so box #1 on frame A and box #1 on frame C are one object.
    seeds = {}          # global frame index -> boxes (numpy is a local import here)
    for rel, raw_boxes in sorted((spec.get("refs") or {}).items()):
        if rel not in rels:
            print(f"warning: 匡選的參考影格不在這個資料夾裡,略過: {rel}", flush=True)
            continue
        gi = rels.index(rel)
        rgb = read_rgb(images[gi])
        if rgb is None:
            print(f"warning: 讀不到參考影格,略過: {rel}", flush=True)
            continue
        boxes = _enc.clip_boxes(
            denorm(_enc.as_boxes(raw_boxes), spec, rgb.shape[1], rgb.shape[0]),
            rgb.shape[1], rgb.shape[0])
        if len(boxes):
            seeds[gi] = boxes
    if not seeds:
        print("error: track 模式需要至少一個匡選框 (--boxes-json)", file=sys.stderr)
        return 2

    groups: dict[tuple, list[int]] = {}
    for gi, path in enumerate(images):
        groups.setdefault(_image_size(path) or ("?",), []).append(gi)
    if len(groups) > 1:
        summary = ", ".join(f"{k}×{len(v)}張" for k, v in groups.items())
        print(f"note: 這個資料夾有 {len(groups)} 種影像尺寸({summary}),"
              "每種各跑一次追蹤 —— 每種尺寸都要至少匡選一張", flush=True)

    model_id = args.model or DEFAULT_TRACK_MODEL
    predictor = SAM2VideoPredictor.from_pretrained(model_id)
    predictor.to(device)
    print(f"Model: {model_id} (track)", flush=True)
    print(f"Images: {len(images)} under {images_dir}", flush=True)

    per_frame = {}      # global frame index -> merged bool mask
    unseeded: list[int] = []
    n_boxes = max(len(b) for b in seeds.values())
    for size, indices in groups.items():
        group_seeds = {local: seeds[gi] for local, gi in enumerate(indices) if gi in seeds}
        if not group_seeds:
            unseeded.extend(indices)
            print(f"warning: 尺寸 {size} 的 {len(indices)} 張沒有任何匡選框,"
                  f"不做追蹤(依 --on-empty {args.on_empty} 處理);"
                  "請在其中一張上也匡一次", flush=True)
            continue
        staging = Path(tempfile.mkdtemp(prefix="sam_track_"))
        try:
            _stage_jpegs(images, indices, staging)
            start_local = min(group_seeds)
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
                state = predictor.init_state(video_path=str(staging),
                                             offload_video_to_cpu=True,
                                             offload_state_to_cpu=True)
                for local, boxes in sorted(group_seeds.items()):
                    for obj_id, box in enumerate(boxes, 1):
                        predictor.add_new_points_or_box(
                            state, frame_idx=local, obj_id=obj_id,
                            box=np.asarray(box, dtype=np.float32))
                    print(f"    seeded {len(boxes)} object(s) on {rels[indices[local]]}",
                          flush=True)
                # Two passes: forward from the earliest seed, then reverse, so a box
                # drawn on a middle frame still covers the start of the sequence.
                for reverse in (False, True):
                    for local, _ids, logits in predictor.propagate_in_video(
                            state, start_frame_idx=start_local, reverse=reverse):
                        m = (logits > 0.0).squeeze(1).cpu().numpy()
                        merged = _enc.merge_masks(m, m.shape[-2:])
                        gi = indices[local]
                        per_frame[gi] = (merged | per_frame[gi]) if gi in per_frame else merged
            predictor.reset_state(state)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    # The tracking above is strictly sequential (memory attention conditions each
    # frame on the last), but writing the frames out is not — and it is the
    # slower half by an order of magnitude, so it goes wide. See WriteFanout.
    fanout = WriteFanout(args, dataset_root, images_dir, len(images),
                         args.write_workers or default_write_workers())
    print(f"writing with {fanout.workers} worker(s)", flush=True)
    for i, image in enumerate(images):
        if not args.overwrite and outputs_present(args, dataset_root, image, images_dir):
            fanout.note(image, "skip (exists)", skipped=True)
            continue
        fanout.submit(image, per_frame.get(i), f"boxes={n_boxes}")
    written, skipped = fanout.close()
    if unseeded:
        print(f"warning: {len(unseeded)} 張因為所屬尺寸沒有匡選框而未追蹤", flush=True)
    print(f"Done. processed={written} skipped={skipped}", flush=True)
    return 0


BATCHABLE_BOX_SOURCES = ("json", "full", "auto")


def batchable(args, runner: MaskRunner, spec: dict) -> bool:
    """Whether this run can encode several photos per forward pass.

    Three things have to hold. The engine must actually have an across-images
    API (`Sam2Runner.masks_for_image_batch`; SAM 1 has none and SAM 3 fuses the
    prompt into the forward pass, so there is nothing to reuse). The boxes must
    be derivable without the GPU — true for json/full/auto, false for `text`
    (a detector or the model itself produces them) and `exemplar` (SAM 3 only).
    And there must be no click prompts: that is the single-frame repair path,
    where there is nothing to batch.
    """
    return (args.image_batch > 1
            and args.boxes in BATCHABLE_BOX_SOURCES
            and hasattr(runner, "masks_for_image_batch")
            and not (spec or {}).get("points"))


def run_batched(args, runner: MaskRunner, images: list, images_dir: Path,
                spec: dict, used_boxes: dict) -> tuple[int, int]:
    """The json/full/auto loop, `--image-batch` photos per encoder pass.

    Same semantics as the one-at-a-time loop below — same box filters, same
    empty handling, same outputs — only the encoder is fed in groups. Worth it
    because the image encoder is ~95 % of SAM's cost and one 1024² frame does
    not fill a modern card; it is NOT the dominant cost of a run overall, which
    is the write phase WriteFanout deals with.

    A photo with no boxes leaves the batch: `--on-empty image` re-prompts
    through the runner, which needs that one frame staged on its own.
    """
    import numpy as np
    import torch

    total = len(images)
    fanout = WriteFanout(args, args.dataset_root, images_dir, total,
                         args.write_workers or default_write_workers())
    print(f"batching {args.image_batch} image(s) per encoder pass, "
          f"writing with {fanout.workers} worker(s)", flush=True)

    def emit(image: Path, rgb, boxes, mask) -> None:
        note = ""
        if not mask.any():
            runner.set_image(rgb)
            mask, note = handle_empty(args, rgb, runner)
            if mask is None:
                fanout.note(image, note, skipped=True)
                return
        if args.largest_only:
            mask = _enc.largest_component(mask)
        used_boxes[image.relative_to(images_dir).as_posix()] = (
            np.asarray(boxes).round(1).tolist())
        fanout.submit(image, mask, " ".join(x for x in (f"boxes={len(boxes)}", note) if x))

    def flush(pending: list) -> None:
        """Encode+decode `pending` as one batch, halving on CUDA OOM."""
        size = len(pending)
        while pending:
            group, rest = pending[:size], pending[size:]
            try:
                masks_list = runner.masks_for_image_batch([rgb for _, rgb, _ in group],
                                                          [b for _, _, b in group])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                size = max(1, size // 2)
                print(f"    [vram] OOM -> retrying with image batch {size}", flush=True)
                continue
            for (image, rgb, boxes), masks in zip(group, masks_list, strict=True):
                emit(image, rgb, boxes, _enc.merge_masks(masks, rgb.shape[:2]))
            pending = rest

    pending: list = []
    for image in images:
        if not args.overwrite and outputs_present(args, args.dataset_root, image, images_dir):
            fanout.note(image, "skip (exists)", skipped=True)
            continue
        rgb = read_rgb(image)
        if rgb is None:
            fanout.note(image, "skip unreadable", skipped=True)
            continue
        h, w = rgb.shape[:2]
        rel = image.relative_to(images_dir).as_posix()
        if args.boxes == "json":
            boxes = boxes_for_image(spec, rel, w, h)
        elif args.boxes == "full":
            boxes = _enc.as_boxes([[0, 0, w - 1, h - 1]])
        else:                                       # auto
            boxes = auto_row_boxes(rgb, max(args.min_area, 1e-4))
        boxes = refine_boxes(args, boxes, None, w, h)
        if len(boxes) == 0:
            runner.set_image(rgb)
            mask, note = handle_empty(args, rgb, runner)
            if mask is None:
                fanout.note(image, note, skipped=True)
            else:
                fanout.submit(image, mask, f"boxes=0 {note}")
            continue
        pending.append((image, rgb, boxes))
        if len(pending) >= args.image_batch:
            flush(pending)
            pending = []
    if pending:
        flush(pending)
    return fanout.close()


def run_one_at_a_time(args, runner: MaskRunner, detector, images: list,
                      images_dir: Path, spec: dict, exemplar,
                      used_boxes: dict) -> tuple[int, int]:
    """The general loop: one encoder pass per photo.

    Handles everything `run_batched` cannot — `text` (the boxes come from a
    detector, or from the model's own fused forward pass), `exemplar`, click
    prompts (repair), and the engines with no across-images API. The writes
    still fan out to threads; only the inference is serial here.
    """
    import numpy as np

    total = len(images)
    fanout = WriteFanout(args, args.dataset_root, images_dir, total,
                         args.write_workers or default_write_workers())
    print(f"writing with {fanout.workers} worker(s)", flush=True)
    for image in images:
        rel = image.relative_to(images_dir).as_posix()
        if not args.overwrite and outputs_present(args, args.dataset_root, image, images_dir):
            fanout.note(image, "skip (exists)", skipped=True)
            continue
        rgb = read_rgb(image)
        if rgb is None:
            fanout.note(image, "skip unreadable", skipped=True)
            continue
        h, w = rgb.shape[:2]
        started = time.perf_counter()

        masks = None
        scores = None
        points, labels = points_for_image(spec, rel, w, h)
        if args.boxes == "json":
            boxes = boxes_for_image(spec, rel, w, h)
        elif args.boxes == "full":
            boxes = _enc.as_boxes([[0, 0, w - 1, h - 1]])
        elif args.boxes == "auto":
            boxes = auto_row_boxes(rgb, max(args.min_area, 1e-4))
        elif args.boxes == "exemplar":
            boxes = denorm(exemplar, spec, w, h)
        elif detector is not None:
            boxes, scores = detector(rgb, args.text)
        else:                                   # text, on a text-native engine
            masks, boxes, scores = runner.masks_for_text(rgb, args.text)

        boxes = refine_boxes(args, boxes, scores, w, h)

        # `staged` tracks whether the encoder has already seen this frame. It
        # matters twice: --on-empty image re-prompts through the runner and would
        # read a stale (or unset) image without it, and re-running set_image on a
        # frame that already has one would pay the encoder cost — the expensive
        # part — a second time.
        staged = False
        note = ""
        if points is not None and len(points):
            # Repair: one object, box + clicks in a single prompt.
            runner.set_image(rgb)
            staged = True
            masks = runner.masks_for_prompt(boxes, points, labels)
            mask = _enc.merge_masks(masks, (h, w))
        elif masks is None:
            runner.set_image(rgb)
            staged = True
            if len(boxes) == 0:
                mask, note = handle_empty(args, rgb, runner)
                if mask is None:
                    fanout.note(image, note, skipped=True)
                    continue
            else:
                masks = masks_in_chunks(runner, boxes, args.box_chunk, (h, w))
                mask = _enc.merge_masks(masks, (h, w))
        else:
            mask = _enc.merge_masks(masks, (h, w))
        if not mask.any():
            if not staged:
                runner.set_image(rgb)
            mask, note = handle_empty(args, rgb, runner)
            if mask is None:
                fanout.note(image, note, skipped=True)
                continue
        if args.largest_only:
            mask = _enc.largest_component(mask)

        used_boxes[rel] = np.asarray(boxes).round(1).tolist()
        pts = f"points={len(points)}" if points is not None and len(points) else ""
        detail = " ".join(x for x in (
            f"boxes={len(boxes)}", pts, note,
            f"inference {(time.perf_counter() - started) * 1000:.0f} ms") if x)
        fanout.submit(image, mask, detail)
    return fanout.close()


def handle_empty(args, rgb, runner):
    """Nothing was detected. Decide what that means, loudly.

    Returns `(mask, note)`: the mask to write (None = skip this photo), and the
    warning text the caller must put on that photo's progress line.

    It does NOT print. There is exactly one thing in this file allowed to emit a
    `[i/N] …` line — `WriteFanout` — because jobs.py drives the panel's progress
    bar off that prefix and a second, unsynchronised printer would make the bar
    jump backwards once the writes run on threads. `[warn] no boxes` still ends
    up in the line verbatim, which is what _parse_matte counts.

    Silence here is how a folder ends up half-masked: the run "succeeds" and the
    missing files only surface later as a COLMAP mask-pass error, or as a trainer
    quietly reading no mask at all. `opaque` is the default because a fully
    opaque alpha is the neutral element — downstream behaves exactly as if the
    去背 step had not run for that frame — and it keeps the output file count
    equal to the image count, which the COLMAP mask pass requires.
    """
    import numpy as np
    note = f"[warn] no boxes -> {args.on_empty}"
    if args.on_empty == "skip":
        return None, note
    if args.on_empty == "opaque":
        return np.ones(rgb.shape[:2], dtype=bool), note
    if args.on_empty == "image" and runner is not None:      # whole frame as one box
        h, w = rgb.shape[:2]
        boxes = _enc.as_boxes([[0, 0, w - 1, h - 1]])
        return _enc.merge_masks(runner.masks_for_boxes(boxes), (h, w)), note
    return np.ones(rgb.shape[:2], dtype=bool), note


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SAM batch background removal (RGBA + masks)")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--images", default="images",
                    help="image folder under dataset_root ('' = the root itself)")
    ap.add_argument("--engine", choices=("sam2", "sam3", "sam1"), default="sam2")
    ap.add_argument("--model", default="", help="HF id, or '<cfg.yaml>:<ckpt.pt>' / vit_h")
    ap.add_argument("--checkpoint", default="", help="sam1 only: path to the .pth")
    ap.add_argument("--boxes", default="json",
                    choices=("json", "track", "text", "exemplar", "auto", "full"),
                    help="where the prompt boxes come from")
    ap.add_argument("--boxes-json", type=Path, default=None)
    ap.add_argument("--text", default="", help="concept prompt, e.g. 'parrot'")
    ap.add_argument("--detector", default="IDEA-Research/grounding-dino-base",
                    help="text->box detector for sam1/sam2 (sam3 needs none)")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--score-threshold", type=float, default=0.40, help="sam3 instance score")
    ap.add_argument("--outputs", default="cutout,masks",
                    help="comma list: cutout (RGBA) and/or masks (0/255 L)")
    ap.add_argument("--erode", type=int, default=1, help="shrink the mask N px before feathering")
    ap.add_argument("--dilate", type=int, default=0)
    ap.add_argument("--feather", type=int, default=2, help="0 = hard edge")
    ap.add_argument("--no-bleed", action="store_true",
                    help="skip the foreground colour bleed (leaves a dark fringe)")
    ap.add_argument("--soft-masks", action="store_true",
                    help="write masks/ with the soft ramp instead of hard 0/255")
    ap.add_argument("--min-area", type=float, default=0.0,
                    help="drop boxes below this fraction of the frame")
    ap.add_argument("--row-filter", action="store_true",
                    help="keep only the dominant horizontal row of subjects")
    ap.add_argument("--largest-only", action="store_true",
                    help="keep only the biggest blob (single-subject shots)")
    ap.add_argument("--box-chunk", type=int, default=32, help="boxes per decoder call")
    ap.add_argument("--image-batch", type=int, default=8,
                    help="photos per ENCODER pass (json/full/auto on sam2 only; "
                         "1 disables batching). Halves itself on CUDA OOM.")
    ap.add_argument("--write-workers", type=int, default=0,
                    help="threads writing the PNGs out (0 = a CPU-count cap). This is "
                         "the slow half of a run at full resolution, not the GPU.")
    ap.add_argument("--on-empty", choices=("opaque", "skip", "image"), default="opaque")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dump-boxes", type=Path, default=None,
                    help="write the boxes actually used, for audit / re-runs")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    args.outputs = [s.strip() for s in args.outputs.split(",") if s.strip()]
    bad = [o for o in args.outputs if o not in ("cutout", "masks")]
    if bad:
        print(f"error: unknown --outputs entry: {bad}", file=sys.stderr)
        return 2

    import torch

    images_dir = args.dataset_root / args.images if args.images else args.dataset_root
    if not images_dir.is_dir():
        print(f"error: images folder not found: {images_dir}", file=sys.stderr)
        return 2
    images = list_images(images_dir)
    if not images:
        print(f"error: no images under {images_dir}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("error: CUDA is not available in this environment", file=sys.stderr)
        return 3
    device = torch.device("cuda")
    # bf16 over fp16: SAM's mask decoder produces logits around zero either way,
    # but bf16 needs no gradient/loss scaling and every card this panel targets
    # (Ampere and later) has it.
    dtype = torch.bfloat16

    spec = load_boxes_file(args.boxes_json) if args.boxes_json else {}
    # `only` scopes a run to named frames. That is what makes the repair loop
    # cheap: fixing one bad cut-out re-runs one image, not the folder, and it
    # goes through the same job/log/parse machinery as a full run.
    only = {str(r) for r in (spec.get("only") or [])}
    if only:
        images = [p for p in images if p.relative_to(images_dir).as_posix() in only]
        if not images:
            print(f"error: --boxes-json 的 only 清單沒有對到任何影像: {sorted(only)[:5]}",
                  file=sys.stderr)
            return 2
        print(f"note: 只處理 {len(images)} 張指定的影像(修圖模式)", flush=True)
    if args.boxes in ("json", "track") and not spec:
        print(f"error: --boxes {args.boxes} 需要 --boxes-json", file=sys.stderr)
        return 2
    if args.boxes == "text" and not args.text.strip():
        print("error: --boxes text 需要 --text", file=sys.stderr)
        return 2
    if args.boxes == "exemplar" and args.engine != "sam3":
        print("error: --boxes exemplar 只有 SAM 3 支援(其它引擎請用 track 或 json)",
              file=sys.stderr)
        return 2

    if args.boxes == "track":
        return run_track(args, args.dataset_root, images_dir, images, spec, device)

    model_id = args.model.strip() or DEFAULT_MODELS[args.engine]
    runner = build_runner(args.engine, model_id, args.checkpoint, device, dtype,
                          args.score_threshold)
    detector = None
    if args.boxes == "text" and not runner.takes_text:
        detector = GroundingDinoDetector(args.detector, device, args.box_threshold,
                                         args.text_threshold)

    print(f"Model: {model_id} ({args.engine})"
          + (f" + {args.detector}" if detector else ""), flush=True)
    print(f"Images: {len(images)} under {images_dir}", flush=True)
    print(f"Mode: {args.boxes} | outputs: {','.join(args.outputs)} | "
          f"erode={args.erode} dilate={args.dilate} feather={args.feather}", flush=True)

    exemplar = _enc.as_boxes(spec.get("boxes") or []) if spec else _enc.as_boxes([])
    used_boxes: dict[str, list] = {}

    if batchable(args, runner, spec):
        written, skipped = run_batched(args, runner, images, images_dir, spec, used_boxes)
    else:
        written, skipped = run_one_at_a_time(args, runner, detector, images, images_dir,
                                             spec, exemplar, used_boxes)

    if args.dump_boxes:
        args.dump_boxes.parent.mkdir(parents=True, exist_ok=True)
        args.dump_boxes.write_text(json.dumps({"per_image": used_boxes}, indent=1))
        print(f"boxes -> {args.dump_boxes}", flush=True)
    print(f"Done. processed={written} skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
