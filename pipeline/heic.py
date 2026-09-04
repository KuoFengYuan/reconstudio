"""HEIC/HEIF -> JPEG conversion.

iPhones default to HEIC, which nothing downstream here reads: COLMAP's own
image loader, the gallery's ffmpeg thumbnailer, and PIL all need libheif
support this box doesn't have wired in. Rather than teach every `IMAGE_EXTS`
scanner in the pipeline a new extension (and hope the tool it hands the path
to can actually decode it), each source HEIC gets a same-stem sibling JPEG
written once -- every existing scanner then just sees a normal `.jpg`, same
as it always has.

`pillow-heif` is a genuinely optional, non-pure-Python dependency (see the
comment on `dependencies` in pyproject.toml on why the panel stays
pip-installable without it), so the import is lazy and the call sites that
don't touch a HEIC folder never pay for it.
"""
from __future__ import annotations

from pathlib import Path

HEIC_EXTS = {".heic", ".heif"}

# Never descend into a pipeline's own output folders -- mirrors the
# SKIP_DIR_NAMES convention in pipeline/matte.py.
_SKIP_DIR_NAMES = frozenset({
    "no_bg", "cutout", "masks", "depth", "normals",
    "colmap", "sparse", "dense", "stereo", "distorted",
})


def convert_tree(root: Path) -> int:
    """Convert every `*.heic`/`*.heif` under `root` to a sibling `.jpg`.

    Non-destructive (same convention as depth/matte): originals are left in
    place, and a source that already has a same-stem `.jpg` is left alone
    (so re-running a job never redoes this work). Returns how many files
    were converted.
    """
    root = Path(root)
    if not root.is_dir():
        return 0

    to_convert = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in HEIC_EXTS
        and _SKIP_DIR_NAMES.isdisjoint(part for part in p.relative_to(root).parts[:-1])
        and not p.with_suffix(".jpg").exists()
    ]
    if not to_convert:
        return 0

    try:
        import pillow_heif
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            f"{len(to_convert)} 個 HEIC/HEIF 檔需要轉檔,但面板 env 沒裝 pillow-heif:\n"
            "  pip install pillow-heif\n"
            "(iPhone 拍的照片預設是 HEIC,COLMAP/去背/深度都讀不懂,要先轉成 JPEG。)"
        ) from exc
    pillow_heif.register_heif_opener()

    n = 0
    for src in to_convert:
        img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        kwargs = {"quality": 95}
        if img.info.get("exif"):
            kwargs["exif"] = img.info["exif"]
        img.save(src.with_suffix(".jpg"), "JPEG", **kwargs)
        n += 1
    return n
