"""Standalone mesh viewer: open ANY local mesh file (not tied to a job).

The job-bound viewer (/viz/mesh/{job_id}) only shows a mesh job's own outputs;
this one takes a file path — picked via the same 瀏覽 modal or typed — so any
.ply/.obj/.stl/.glb on the server can be inspected with the same orbit/measure
tooling. Paths are confined to the browse roots, like the directory picker.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from pipeline.config import settings
from web.shared import (
    BROWSE_ROOT,
    DEST_ROOT,
    IMAGE_EXTS,
    IMAGE_EXTS_ALL,
    MESH_EXTS,
    SPLAT_EXTS,
    _page,
)

router = APIRouter()


def _safe_mesh_file(path: str, exts: frozenset | set = MESH_EXTS) -> Path:
    """Resolve `path` and require: inside a browse root (same confinement as the
    folder picker — this endpoint can read arbitrary server files otherwise), an
    existing regular file, and a viewer-supported extension (`exts`)."""
    p = Path(path or "").resolve()
    if not any(root == p or root in p.parents for root in (BROWSE_ROOT, DEST_ROOT)):
        raise HTTPException(400, f"path outside browse roots ({BROWSE_ROOT}, {DEST_ROOT})")
    if p.suffix.lower() not in exts:
        raise HTTPException(415, f"unsupported format: {p.suffix or '(no extension)'} "
                                 f"(supported: {', '.join(sorted(exts))})")
    if not p.is_file():
        raise HTTPException(404, f"file not found: {p}")
    return p


@router.get("/viewer", response_class=HTMLResponse)
async def mesh_viewer(request: Request, path: str = ""):
    """Standalone in-browser mesh viewer (orbit/zoom + ruler).

    With `path`: load that server-side file. Without: the page runs in local mode —
    the browser picks/drag-drops a file from the USER'S machine and parses it
    client-side (nothing is uploaded)."""
    if not path:
        return _page(request, "mesh_view.html", path="", fname="")
    p = _safe_mesh_file(path)
    # `fname` not `name`: _page's second positional arg is already called `name`
    return _page(request, "mesh_view.html", path=str(p), fname=p.name)


@router.get("/api/viewer/meshfile")
async def mesh_file(path: str):
    """Serve the raw file a standalone viewer loads: meshes for /viewer AND splat
    point clouds for the SuperSplat tool (it streams from this same endpoint)."""
    p = _safe_mesh_file(path, MESH_EXTS | SPLAT_EXTS)
    return FileResponse(p, media_type="application/octet-stream", filename=p.name)


# --- standalone image gallery (/gallery) ----------------------------------- #
# Like /viewer but for photos: point it at ANY folder under the browse roots and
# it shows every image inside as a clickable grid (lightbox + prev/next). Useful
# for eyeballing a 資料/抽幀/undistort folder without tying it to a job — the
# viz photo panel only shows a reconstructed job's registered images.

_GALLERY_MAX = 5000          # cap images listed per gallery (huge multi-rig frame dirs)
# transcoded-TIFF preview cache (the source folder may be read-only, so cache in
# the system temp dir, keyed by the absolute source path + mtime, not in-tree).
_PREVIEW_ROOT = Path(tempfile.gettempdir()) / "reconstudio_gallery_preview"


def _safe_image_dir(path: str) -> Path:
    """Resolve `path`, require it inside a browse root and be an existing dir."""
    p = Path(path or "").resolve()
    if not any(root == p or root in p.parents for root in (BROWSE_ROOT, DEST_ROOT)):
        raise HTTPException(400, f"path outside browse roots ({BROWSE_ROOT}, {DEST_ROOT})")
    if not p.is_dir():
        raise HTTPException(404, f"not a directory: {p}")
    return p


def _safe_image_file(path: str) -> Path:
    """Same confinement as _safe_mesh_file, but for the gallery's image formats."""
    p = Path(path or "").resolve()
    if not any(root == p or root in p.parents for root in (BROWSE_ROOT, DEST_ROOT)):
        raise HTTPException(400, f"path outside browse roots ({BROWSE_ROOT}, {DEST_ROOT})")
    if p.suffix.lower() not in IMAGE_EXTS_ALL:
        raise HTTPException(415, f"unsupported image format: {p.suffix or '(no extension)'}")
    if not p.is_file():
        raise HTTPException(404, f"file not found: {p}")
    return p


def _preview_jpeg(src: Path, max_w: int = 1920) -> Path | None:
    """Transcode an image to a cached JPEG whose width is capped at `max_w` (with
    `-2` so the height stays even, aspect preserved). Used both for TIFFs the browser
    can't render natively AND for downscaled grid thumbnails of huge originals — the
    grid otherwise downloads + decodes every full-res photo just to paint a 160px
    cell, which is what makes a high-resolution folder crawl. Cached in the temp dir
    keyed by the source path + width, reused while newer than the source. Returns
    None if ffmpeg is missing/fails. Caches outside any workspace (gallery folders
    aren't job workspaces and may be read-only)."""
    key = hashlib.sha1(f"{src}|{max_w}".encode()).hexdigest()[:20]
    dst = _PREVIEW_ROOT / f"{key}.jpg"
    try:
        if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
            return dst
    except OSError:
        pass
    ff = settings.ffmpeg_bin
    if not shutil.which(ff):
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".jpg.tmp")
    r = subprocess.run(
        [ff, "-y", "-nostdin", "-loglevel", "error", "-i", str(src),
         "-vf", f"scale='min({max_w},iw)':-2:flags=area", "-q:v", "3", "-f", "mjpeg", str(tmp)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(dst)
    return dst


def _walk_images(root: Path, recursive: bool, cap: int) -> tuple[list[str], bool]:
    """Collect image paths under `root`, each RELATIVE to root (posix). Recursive
    walk descends subfolders (skipping hidden ones) depth-first so every subdir's
    images stay contiguous and ordered; non-recursive lists only the top level.
    Returns (rel_paths, truncated) — capped at `cap` to bound the page size."""
    rels: list[str] = []
    if not recursive:
        names = sorted((f.name for f in root.iterdir()
                        if f.is_file() and not f.name.startswith(".")
                        and f.suffix.lower() in IMAGE_EXTS_ALL), key=str.lower)
        return names[:cap], len(names) > cap
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))  # prune hidden, order
        dp = Path(dirpath)
        for name in sorted(filenames, key=str.lower):
            if name.startswith(".") or Path(name).suffix.lower() not in IMAGE_EXTS_ALL:
                continue
            rels.append((dp / name).relative_to(root).as_posix())
            if len(rels) >= cap:
                return rels, True
    return rels, False


@router.get("/gallery", response_class=HTMLResponse)
async def gallery(request: Request, path: str = "", recursive: int = 1):
    """Server-side image gallery. With `path` (a folder under the browse roots),
    show every image inside as a grid — by default recursing into subfolders and
    grouping the grid by subfolder (so e.g. a backward/forward/left/nadir/right rig
    layout shows all photos at once, each under its folder heading). `recursive=0`
    lists only the top level. Without `path`, the page shows a how-to hint."""
    if not path:
        return _page(request, "gallery.html", path="", groups=[], items=[],
                     truncated=False, recursive=bool(recursive))
    d = _safe_image_dir(path)
    rels, truncated = await asyncio.to_thread(_walk_images, d, bool(recursive), _GALLERY_MAX)
    # flat ordered list (drives the lightbox prev/next) + per-subfolder groups (layout).
    items, groups = [], []
    for i, rel in enumerate(rels):
        pp = PurePosixPath(rel)
        sub = "" if pp.parent == PurePosixPath(".") else pp.parent.as_posix()
        it = {"i": i, "rel": rel, "sub": sub, "name": pp.name}
        items.append(it)
        if not groups or groups[-1]["sub"] != sub:     # walk keeps each sub contiguous
            groups.append({"sub": sub, "files": []})
        groups[-1]["files"].append(it)
    return _page(request, "gallery.html", path=str(d), groups=groups, items=items,
                 truncated=truncated, recursive=bool(recursive))


@router.get("/api/gallery/imagefile")
async def gallery_image(path: str, w: int = 0):
    """Serve one image the gallery shows. With `w` (the grid's thumbnail width) the
    image is downscaled to a cached JPEG so a high-res folder doesn't ship hundreds
    of full-size photos to the browser; without it (the lightbox), web-native formats
    stream at full resolution and TIFF is transcoded to a <=1920px preview (browsers
    can't render TIFF)."""
    p = _safe_image_file(path)
    web_native = p.suffix.lower() in IMAGE_EXTS
    if w > 0:                                   # grid thumbnail: always downscale + cache
        jpg = await asyncio.to_thread(_preview_jpeg, p, max(64, min(w, 4096)))
        if jpg:
            return FileResponse(jpg, media_type="image/jpeg")
        if web_native:                          # ffmpeg unavailable → original still works (slow)
            return FileResponse(p)
        raise HTTPException(415, f"cannot preview {p.suffix} image (transcode failed)")
    if web_native:                              # lightbox / full view
        return FileResponse(p)
    jpg = await asyncio.to_thread(_preview_jpeg, p)
    if jpg:
        return FileResponse(jpg, media_type="image/jpeg")
    raise HTTPException(415, f"cannot preview {p.suffix} image (transcode failed)")
