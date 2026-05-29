"""Image folder layout detection + image listing for the COLMAP run.

The pipeline supports three fixed input shapes:
  single  : XXX/*.jpg              -> folders=[''] (image_root itself), 1 camera
  multi   : ROOT/<group>/*.jpg     -> folders=[groups], flat, camera per group
  nested  : ROOT/<group>/<vid>/*.jpg -> staged into groups, camera per group
"""
from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _has_images(folder: Path) -> bool:
    return folder.is_dir() and any(
        c.is_file() and c.suffix.lower() in IMAGE_EXTS for c in folder.iterdir())


def list_images(folder: Path, prefix: str) -> list[str]:
    """Image files directly in `folder` (symlinks followed), relative as prefix/name
    (or just name when prefix is '' — a single flat folder = image_root itself)."""
    out: list[str] = []
    if folder.is_dir():
        for entry in folder.iterdir():
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
                out.append(f"{prefix}/{entry.name}" if prefix else entry.name)
    return sorted(out)


def list_image_names(folder: Path) -> list[str]:
    """Bare image filenames directly in `folder`, sorted. The h3dgs custom matcher
    pairs images by per-folder frame order, so it needs the names without the group
    prefix (the prefix is re-attached when emitting "prefix/name" match pairs)."""
    out: list[str] = []
    if folder.is_dir():
        for entry in folder.iterdir():
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
                out.append(entry.name)
    return sorted(out)


def resolve_layout(img_root: str, folders: list[str], layout: str,
                   force_nested: bool, workspace: str | None = None
                   ) -> tuple[list[str], bool, str]:
    """Return (folders, nested, layout_name)."""
    root = Path(img_root)
    # Ignore the workspace dir if the user nested it inside image_root, so it isn't
    # mistaken for a camera group.
    skip = set()
    if workspace:
        try:
            wp = Path(workspace)
            if wp.resolve().parent == root.resolve():
                skip.add(wp.name)
        except OSError:
            pass
    subdirs = sorted([x.name for x in root.iterdir()
                      if x.is_dir() and not x.name.startswith(".") and x.name not in skip])

    # Explicit single, or auto when the root itself holds images (subdirs are then
    # likely junk such as the workspace) -> single flat folder.
    if layout == "single" or (layout == "auto" and _has_images(root)):
        return (folders if folders else [""]), False, "single"

    chosen = folders or subdirs
    if not chosen:
        if _has_images(root):
            return [""], False, "single"
        raise FileNotFoundError(f"no images or subfolders found under: {img_root}")

    if layout == "nested" or force_nested:
        return chosen, True, "nested"
    if layout == "multi":
        return chosen, False, "multi"

    # auto: nested if the first group has no direct images but does have subdirs
    nested = False
    for f in chosen:
        g = root / f
        if g.is_dir() and not _has_images(g) and any(c.is_dir() for c in g.iterdir()):
            nested = True
        break
    return chosen, nested, ("nested" if nested else "multi")
