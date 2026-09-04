"""融合: stage several matted captures of ONE subject into a single COLMAP input.

An object photographed standing up cannot show its underside, so it gets shot
again flipped over. Both passes are of the same subject but the subject *moved*
between them, which is what makes this different from a two-folder scene: the
room is no longer rigid with respect to the thing being reconstructed, so the
background must not reach the solve at all. That is why this stage exists at the
去背 output rather than at the photos: a pass is only fusible once it has masks.

What it does is exactly the staging, nothing else. Given each pass's `no_bg/`
folder it builds

    <out>/images/<pass>  ->  the photos      (symlink to no_bg/..)
    <out>/masks/<pass>   ->  no_bg/masks     (symlink)

and hands the two roots back for a perfectly ordinary `colmap` job with
MASK_FEATURES on — the COLMAP form, its options, the log parser, the viewer and
the training hand-off all stay untouched. Symlinks, not copies: a pass is
thousands of full-resolution frames, and COLMAP's own image listing follows
links (`_layout.list_images`).

Why the mask folder mirrors the image folder name: COLMAP resolves a mask by the
image's path *relative to image_path*, so `images/<pass>/x.jpg` is looked up as
`masks/<pass>/x.png`. Getting that mapping wrong is silent — COLMAP reports
`Mask: No` per image and reconstructs the background anyway.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# Mirrors matte_encode.OUTPUT_ROOT / the matte stage's folder names. Duplicated
# rather than imported for the same reason pipeline/matte.py duplicates them:
# matte_encode pulls in numpy and this module must stay eagerly importable.
OUTPUT_ROOT = "no_bg"
MASKS_SUB = "masks"
CUTOUT_SUB = "cutout"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# Where the staged roots land under the fusion folder, and where the COLMAP
# workspace goes, so one folder holds the whole fusion.
IMAGES_ROOT = "images"
MASKS_ROOT = "masks"
WORKSPACE_SUB = "colmap"


@dataclass
class Pass:
    """One matted capture, resolved from the `no_bg/` folder the user picked."""
    name: str               # the staged folder name = the COLMAP camera group
    no_bg: Path
    images_dir: Path        # the photos the matte ran on = no_bg's parent
    masks_dir: Path         # no_bg/masks (single-channel 0/255)
    n_images: int
    n_masks: int
    sizes: tuple[tuple[int, int], ...]   # distinct (w, h) seen, most common first

    @property
    def unmatched(self) -> int:
        return max(0, self.n_images - self.n_masks)


def _slug(name: str) -> str:
    """A folder name safe to use as an image-name prefix. Non-ASCII is kept — the
    whole pipeline already runs on Chinese paths — only separators are dropped."""
    return re.sub(r"[\\/\s]+", "_", name).strip("._") or "pass"


def resolve_pass(no_bg: str | Path) -> tuple[Path, Path, Path]:
    """(no_bg, images_dir, masks_dir) from whatever the user pasted.

    Accepts the `no_bg/` folder itself (what the 去背 job reports), or its
    `cutout/`/`masks/` child, or the image folder that contains a `no_bg/` —
    all four are things someone reasonably copies out of the panel.
    """
    p = Path(no_bg).expanduser()
    if p.name in (MASKS_SUB, CUTOUT_SUB) and p.parent.name == OUTPUT_ROOT:
        p = p.parent
    elif p.name != OUTPUT_ROOT and (p / OUTPUT_ROOT).is_dir():
        p = p / OUTPUT_ROOT
    if p.name != OUTPUT_ROOT:
        raise ValueError(
            f"這不像去背輸出的資料夾: {no_bg}\n"
            f"要填的是去背產生的 {OUTPUT_ROOT}/ 資料夾(或它的上一層)。")
    if not p.is_dir():
        raise FileNotFoundError(f"找不到資料夾: {p}")
    masks = p / MASKS_SUB
    if not masks.is_dir():
        raise FileNotFoundError(
            f"{p} 底下沒有 {MASKS_SUB}/。融合需要單通道遮罩 —— 去背時「輸出」"
            f"要包含 masks/(只選 cutout/ 的話沒有這一份)。")
    return p, p.parent, masks


def _scan(images_dir: Path, masks_dir: Path) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    """Count the photos, count the ones with a mask, and collect the sizes.

    Sizes matter for two later decisions the caller reports on: CAMERA_MODE
    (one shared camera is only right when every pass shares a geometry) and the
    mask/image size agreement COLMAP hard-requires.
    """
    counts: dict[tuple[int, int], int] = {}
    n_images = n_masks = 0
    for f in sorted(images_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue
        n_images += 1
        if (masks_dir / f.with_suffix(".png").name).is_file():
            n_masks += 1
        size = _size(f)
        if size:
            counts[size] = counts.get(size, 0) + 1
    ordered = tuple(s for s, _ in sorted(counts.items(), key=lambda kv: -kv[1]))
    return n_images, n_masks, ordered


def _size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:                       # noqa: BLE001 — header probe is best-effort
        return None


def inspect(inputs: list[str]) -> list[Pass]:
    """Resolve and measure each input, without writing anything.

    Separate from `stage` so the form can show what it found — counts, sizes,
    missing masks — before anything is created on disk.
    """
    if len(inputs) < 2:
        raise ValueError("融合至少要兩組去背結果(例如正擺一組、倒擺一組)")
    out: list[Pass] = []
    used: set[str] = set()
    for raw in inputs:
        no_bg, images_dir, masks_dir = resolve_pass(raw)
        name = _slug(images_dir.parent.name if images_dir.name.startswith("images")
                     else images_dir.name)
        while name in used:                 # two passes under the same dataset name
            name += "_2"
        used.add(name)
        n_images, n_masks, sizes = _scan(images_dir, masks_dir)
        if not n_images:
            raise FileNotFoundError(f"{images_dir} 裡沒有照片")
        if not n_masks:
            raise FileNotFoundError(
                f"{masks_dir} 裡沒有對得上的遮罩(照片 {n_images} 張,遮罩 0 張)。"
                "遮罩檔名要跟照片同名、副檔名換成 .png。")
        out.append(Pass(name=name, no_bg=no_bg, images_dir=images_dir,
                        masks_dir=masks_dir, n_images=n_images, n_masks=n_masks,
                        sizes=sizes))
    return out


def stage(passes: list[Pass], out_dir: Path) -> tuple[Path, Path, Path]:
    """Build `<out>/images/<pass>` and `<out>/masks/<pass>`; return the three roots.

    Idempotent: an existing correct link is left alone, an existing WRONG link is
    repointed, and a real directory in the way is an error rather than something
    to delete — that path may be someone's data.
    """
    images_root = out_dir / IMAGES_ROOT
    masks_root = out_dir / MASKS_ROOT
    for root in (images_root, masks_root):
        root.mkdir(parents=True, exist_ok=True)
    for p in passes:
        _link(images_root / p.name, p.images_dir)
        _link(masks_root / p.name, p.masks_dir)
    return images_root, masks_root, out_dir / WORKSPACE_SUB


def _link(link: Path, target: Path) -> None:
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(
            f"{link} 已經存在而且不是 symlink。融合資料夾裡的 {IMAGES_ROOT}/ 與 "
            f"{MASKS_ROOT}/ 只放連結,請換一個融合輸出資料夾。")
    link.symlink_to(target, target_is_directory=True)


def default_out_dir(passes: list[Pass]) -> Path:
    """`<nearest common parent of the passes>/fusion`, which for two datasets
    sitting side by side is the folder that already holds them both."""
    parents = [p.images_dir.parent.resolve() for p in passes]
    common = parents[0]
    for q in parents[1:]:
        while common != common.parent and common not in (q, *q.parents):
            common = common.parent
    return common / "fusion"


def summarize(passes: list[Pass]) -> tuple[list[str], list[str]]:
    """(facts, warnings) for the form to show before the job is submitted."""
    facts = [f"{p.name}: {p.n_images} 張照片 · {p.n_masks} 張遮罩 · "
             + ("×".join(map(str, p.sizes[0])) if p.sizes else "尺寸未知")
             + (f" (+{len(p.sizes) - 1} 種其他尺寸)" if len(p.sizes) > 1 else "")
             for p in passes]
    warn: list[str] = []
    for p in passes:
        if p.unmatched:
            warn.append(f"{p.name} 有 {p.unmatched} 張照片沒有對應的遮罩 —— "
                        "COLMAP 會因為找不到遮罩而中止,請先把去背補跑完。")
    sizes = {p.sizes[0] for p in passes if p.sizes}
    if len(sizes) > 1:
        warn.append("各組的影像尺寸不一致 —— CAMERA_MODE 要用 per_folder "
                    "(每組各自一組內參),不要用 single。")
    return facts, warn
