"""去背 box picker: the frame browser you drag the keep-boxes on.

Only the *frame* side lives here — listing a folder's images and rendering one
of them with a drawing overlay. The boxes themselves never round-trip through
this router: the fragment writes them into a hidden field of the 去背 form, and
`POST /ui/matte` (web/routers/create.py) hands that string to the pipeline.

Boxes are stored **normalised** (0..1 of width/height), because this page draws
on a downscaled JPEG preview and a folder may mix resolutions; see `denorm` in
tools/sam_matte.py for the other half of that contract.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from jobs import manager
from pipeline.config import settings
from pipeline.heic import convert_tree
from pipeline.matte import OUTPUT_ROOT, SKIP_DIR_NAMES, resolve_matte_dataset
from web.routers.viewer import _safe_image_dir, _safe_image_file
from web.shared import IMAGE_EXTS_ALL, _page

router = APIRouter()

# The picker only needs enough frames to choose a representative one; a 20k-frame
# aerial folder must not turn the fragment into a multi-megabyte <select>.
_PICK_MAX = 3000
# Shared with the pipeline so the frames you page through here are exactly the
# frames the run will process — offering a cut-out, or a COLMAP-undistorted copy,
# as the photo to draw on is never what anyone means.
_SKIP_DIRS = SKIP_DIR_NAMES


def _walk(root: Path) -> list[str]:
    rels: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d not in _SKIP_DIRS)
        dp = Path(dirpath)
        for name in sorted(filenames, key=str.lower):
            if name.startswith(".") or Path(name).suffix.lower() not in IMAGE_EXTS_ALL:
                continue
            rels.append((dp / name).relative_to(root).as_posix())
            if len(rels) >= _PICK_MAX:
                return rels
    return rels


def _cutout_dir(root: Path) -> Path | None:
    """Where this dataset's RGBA cut-outs live, or None if it has none.

    Outputs moved under `no_bg/`; datasets matted before that have a bare
    `cutout/` at the root, and silently telling their owner "還沒有去背結果"
    would be worse than the one extra stat.
    """
    for candidate in (root / OUTPUT_ROOT / "cutout", root / "cutout"):
        if candidate.is_dir():
            return candidate
    return None


@router.get("/ui/matte_pick", response_class=HTMLResponse)
async def matte_pick(request: Request, images: str = "", i: int = 0):
    """One frame from `images`, ready to be drawn on. `i` selects which."""
    images = (images or "").strip().rstrip("/")
    if not images:
        raise HTTPException(400, "images is required")
    # Accept either a photo folder or a COLMAP workspace, exactly like the job does,
    # so the picker shows the same frames the run will actually process.
    root, folder = resolve_matte_dataset(Path(images).expanduser())
    d = _safe_image_dir(str(root / folder if folder else root))
    try:
        await asyncio.to_thread(convert_tree, d)
    except RuntimeError as exc:
        return _page(request, "_error.html", message=str(exc))
    rels = await asyncio.to_thread(_walk, d)
    if not rels:
        return _page(request, "_error.html", message=f"這個資料夾裡沒有照片: {d}")
    idx = max(0, min(int(i), len(rels) - 1))
    rel = rels[idx]
    return _page(request, "_matte_pick.html", images=images, dir=str(d), rel=rel,
                 idx=idx, total=len(rels), names=rels[:400],
                 src=f"/api/gallery/imagefile?path={quote(str(d / rel))}&w=1400")


# --------------------------------------------------------------------------- #
# review: look at what came out
# --------------------------------------------------------------------------- #
_REVIEW_MAX = 3000
_PREVIEW_ROOT = Path(tempfile.gettempdir()) / "reconstudio-matte-previews"


def _preview_png(src: Path, max_w: int) -> Path | None:
    """Downscale to a cached **PNG**, keeping the alpha channel.

    The gallery's `_preview_jpeg` cannot be reused here: JPEG has no alpha, so
    every cut-out would come back composited on black — which is precisely the
    failure mode (a dark fringe) this page exists to let you spot. Cached in the
    temp dir keyed by source path + width, reused while newer than the source.
    """
    key = hashlib.sha1(f"{src}|{max_w}|png".encode()).hexdigest()[:20]
    dst = _PREVIEW_ROOT / f"{key}.png"
    try:
        if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
            return dst
    except OSError:
        pass
    ff = settings.ffmpeg_bin
    if not shutil.which(ff):
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".png.tmp")
    r = subprocess.run(
        [ff, "-y", "-nostdin", "-loglevel", "error", "-i", str(src),
         "-vf", f"scale='min({max_w},iw)':-2:flags=area", "-pix_fmt", "rgba",
         "-f", "image2", "-c:v", "png", str(tmp)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(dst)
    return dst


@router.get("/api/matte/imagefile")
async def matte_image(path: str, w: int = 0):
    """One image for the review page. `w` asks for a cached, alpha-preserving thumbnail."""
    p = _safe_image_file(path)
    if w > 0:
        png = await asyncio.to_thread(_preview_png, p, max(64, min(w, 4096)))
        if png:
            return FileResponse(png, media_type="image/png")
    return FileResponse(p)          # ffmpeg missing, or the full-size view


def _pair(images_dir: Path, cutout_dir: Path) -> tuple[list[dict], int]:
    """Match each cut-out back to the photo it came from.

    Outputs mirror the source tree with the extension swapped to .png, so the
    join key is the relative path *without* its extension. Reporting the
    unmatched count matters more than it looks: a folder that is quietly one
    third done is the normal outcome of a prompt that stopped matching, and the
    file grid alone will not tell you that.
    """
    originals = {}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS_ALL:
            rel = p.relative_to(images_dir)
            if rel.parts[0] in _SKIP_DIRS:
                continue
            originals.setdefault(rel.with_suffix("").as_posix(), rel.as_posix())
    items, orphans = [], 0
    for p in sorted(cutout_dir.rglob("*.png")):
        key = p.relative_to(cutout_dir).with_suffix("").as_posix()
        orig = originals.pop(key, None)
        if orig is None:
            orphans += 1
        items.append({"cut": p.relative_to(cutout_dir).as_posix(), "orig": orig or ""})
        if len(items) >= _REVIEW_MAX:
            break
    return items, orphans


@router.get("/matte_review", response_class=HTMLResponse)
async def matte_review(request: Request, images: str = ""):
    """Standalone page: every cut-out this dataset has, on a transparency grid.

    Deliberately not the shared /gallery: that grid transcodes to JPEG, which
    flattens alpha onto black and makes every result look like it has a halo.
    """
    images = (images or "").strip().rstrip("/")
    if not images:
        raise HTTPException(400, "images is required")
    root, folder = resolve_matte_dataset(Path(images).expanduser())
    images_dir = _safe_image_dir(str(root / folder if folder else root))
    cutout_dir = _cutout_dir(root)
    if cutout_dir is None:
        return _page(request, "_error.html",
                     message=f"還沒有去背結果: 找不到 {root / OUTPUT_ROOT / 'cutout'}")
    items, orphans = await asyncio.to_thread(_pair, images_dir, cutout_dir)
    total_src = sum(1 for p in images_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS_ALL
                    and p.relative_to(images_dir).parts[0] not in _SKIP_DIRS)
    return _page(request, "matte_review.html", root=str(root), images=images,
                 cut_root=str(cutout_dir), img_root=str(images_dir),
                 items=items, orphans=orphans, total_src=total_src,
                 missing=max(0, total_src - len(items)))


# --------------------------------------------------------------------------- #
# live: watch the cut-outs land while the job runs
# --------------------------------------------------------------------------- #
_LIVE_MAX = 12


def _newest_cutouts(cutout_dir: Path, limit: int) -> tuple[list[dict], int]:
    """The most recently written cut-outs, newest first, plus the total count.

    Sorted by mtime rather than by name: the point of this strip is to show what
    the running job just produced, and a resumed run fills gaps out of
    alphabetical order. The mtime also goes into the thumbnail URL, so a frame
    that gets re-cut by a repair job busts the browser's cache for that one file.
    """
    files = []
    for p in cutout_dir.rglob("*.png"):
        try:
            files.append((p.stat().st_mtime, p))
        except OSError:
            continue
    files.sort(key=lambda t: t[0], reverse=True)
    items = [{"rel": p.relative_to(cutout_dir).as_posix(), "t": int(m)}
             for m, p in files[:limit]]
    return items, len(files)


@router.get("/ui/matte_live", response_class=HTMLResponse)
async def matte_live(request: Request, images: str = "", n: int = 8, job_id: str = ""):
    """Self-polling htmx fragment: the newest cut-outs a running 去背 job has
    written so far.

    Deliberately a poll rather than a push. The job's own SSE stream carries log
    lines, and these images are produced by a *subprocess* writing to disk — there
    is no event to forward, so a periodic re-read of the directory is both
    simpler and exactly as fresh.

    Swaps its own outerHTML (like `#jobstatus`) rather than living *inside* it:
    `#jobstatus` above replaces its whole outerHTML every 2s, and an element
    nested in there with `hx-trigger="load, every 4s"` gets its "load" fired on
    every one of those replacements — collapsing the thumbnail refresh to 2s and
    tearing the `<img>`s down and rebuilding them that often, which is what made
    the strip visibly flicker/jitter during a long run. Being its own island
    means it only reloads on its own schedule, and decides whether to keep
    polling from the *job's* status (via `job_id`) rather than piggy-backing on
    the parent's re-render.
    """
    images = (images or "").strip().rstrip("/")
    if not images:
        raise HTTPException(400, "images is required")
    job = manager.get(job_id) if job_id else None
    running = bool(job and job.status in ("queued", "running"))
    root, _ = resolve_matte_dataset(Path(images).expanduser())
    cutout_dir = _cutout_dir(root)
    if cutout_dir is None:
        items, total = [], 0
    else:
        items, total = await asyncio.to_thread(
            _newest_cutouts, cutout_dir, max(1, min(n, _LIVE_MAX)))
    return _page(request, "_matte_live.html", images=images, items=items,
                 total=total, cut_root=str(cutout_dir) if cutout_dir else "",
                 job_id=job_id, running=running)


# --------------------------------------------------------------------------- #
# repair: fix one cut-out that came out wrong
# --------------------------------------------------------------------------- #
@router.get("/matte_repair", response_class=HTMLResponse)
async def matte_repair(request: Request, images: str = "", rel: str = ""):
    """Single-image refine view: the photo, its current mask, and +/- clicks.

    A box alone is what already produced the bad cut-out, so the tool here is
    point prompts — "this pixel is subject", "this one is not" — which is the
    information SAM was missing. Applying runs a normal `matte` job scoped to
    this one frame, so the fix goes through the same queue, log parser and
    overwrite rules as a full run.
    """
    images = (images or "").strip().rstrip("/")
    rel = (rel or "").strip().lstrip("/")
    if not images or not rel:
        raise HTTPException(400, "images and rel are required")
    root, folder = resolve_matte_dataset(Path(images).expanduser())
    images_dir = _safe_image_dir(str(root / folder if folder else root))
    cutout_dir = _cutout_dir(root)
    # `rel` names the cut-out (always .png); the photo it came from keeps its own
    # extension, so it is matched on the stem rather than assumed to be .png.
    stem = PurePosixPath(rel).with_suffix("")
    source = None
    for candidate in sorted(images_dir.glob(f"{stem}.*")):
        if candidate.suffix.lower() in IMAGE_EXTS_ALL:
            source = candidate
            break
    if source is None:
        return _page(request, "_error.html", message=f"找不到這張的原圖: {rel}")
    cut = cutout_dir / rel if cutout_dir else None
    return _page(request, "matte_repair.html", images=images, rel=rel,
                 src=f"/api/matte/imagefile?path={quote(str(source))}&w=1600",
                 cut=(f"/api/matte/imagefile?path={quote(str(cut))}&w=1600"
                      f"&t={int(cut.stat().st_mtime)}") if cut and cut.is_file() else "",
                 source=str(source))
