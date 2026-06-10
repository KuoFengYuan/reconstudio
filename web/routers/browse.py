"""Local directory picker + GCS bucket browser (htmx fragments)."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from pipeline import gcs_ls, gcs_parent
from web.shared import BROWSE_ROOT, DEST_ROOT, GSUTIL_BIN, MESH_EXTS, SPLAT_EXTS, VIDEO_EXTS, _page

router = APIRouter()


def _safe_dir(path: str, base: Path | None = None) -> Path:
    base = base or BROWSE_ROOT
    p = (base if not path else Path(path)).resolve()
    if p != base and base not in p.parents:
        raise HTTPException(400, f"path outside browse root ({base})")
    if not p.is_dir():
        raise HTTPException(400, f"not a directory: {p}")
    return p


_MAX_FILES = 1000   # cap files listed per dir in multi-select mode (huge image dirs)


@router.get("/ui/browse", response_class=HTMLResponse)
async def browse(request: Request, path: str = "", target: str = "image_root",
                 root: str = "input", pick: str = "dir", exts: str = "mesh"):
    """Local picker. pick='dir' (default) lists folders only and picks one folder.
    pick='multi' also lists files and renders checkboxes so the caller can select
    any mix of files + folders (used by the GCS upload source). pick='file' lists
    folders to navigate plus viewer-openable files to pick ONE of — `exts` chooses
    the filter: 'mesh' (.ply/.obj/.stl/.glb, the mesh viewer) or 'splat'
    (.ply/.splat/…, the SuperSplat point-cloud viewer)."""
    base = DEST_ROOT if root == "dest" else BROWSE_ROOT
    p = _safe_dir(path, base)
    dirs = sorted((d.name for d in p.iterdir()
                   if d.is_dir() and not d.name.startswith(".")), key=str.lower)
    files: list[str] = []
    truncated = False
    if pick in ("multi", "file"):
        pickable = SPLAT_EXTS if exts == "splat" else MESH_EXTS
        allf = sorted((f.name for f in p.iterdir()
                       if f.is_file() and not f.name.startswith(".")
                       and (pick == "multi" or f.suffix.lower() in pickable)),
                      key=str.lower)
        files = allf[:_MAX_FILES]
        truncated = len(allf) > _MAX_FILES
    nvideos = sum(1 for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
    return _page(request, "_browse.html", path=str(p),
                 parent=(str(p.parent) if p != base else None),
                 dirs=dirs, files=files, truncated=truncated, pick=pick,
                 target=target, nvideos=nvideos, root=root, exts=exts)


@router.get("/ui/gcs_browse", response_class=HTMLResponse)
async def gcs_browse(request: Request, prefix: str = ""):
    prefix = (prefix or "").strip()
    if prefix in ("gs://", "gs:/"):
        prefix = ""
    try:
        if prefix and not prefix.startswith("gs://"):
            raise ValueError("prefix 必須是 gs:// 開頭")
        data = await asyncio.to_thread(gcs_ls, prefix, gsutil_bin=GSUTIL_BIN)
    except Exception as exc:  # noqa: BLE001 — show the error inside the picker
        return _page(request, "_gcs_browse.html", error=str(exc), prefix=prefix,
                     dirs=[], nfiles=0, parent=gcs_parent(prefix))
    return _page(request, "_gcs_browse.html", error=None, prefix=data["prefix"],
                 dirs=data["dirs"], nfiles=data["nfiles"], parent=data["parent"])
