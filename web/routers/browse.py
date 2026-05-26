"""Local directory picker + GCS bucket browser (htmx fragments)."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from pipeline import gcs_ls, gcs_parent
from web.shared import BROWSE_ROOT, DEST_ROOT, GSUTIL_BIN, VIDEO_EXTS, _page

router = APIRouter()


def _safe_dir(path: str, base: Path | None = None) -> Path:
    base = base or BROWSE_ROOT
    p = (base if not path else Path(path)).resolve()
    if p != base and base not in p.parents:
        raise HTTPException(400, f"path outside browse root ({base})")
    if not p.is_dir():
        raise HTTPException(400, f"not a directory: {p}")
    return p


@router.get("/ui/browse", response_class=HTMLResponse)
async def browse(request: Request, path: str = "", target: str = "image_root",
                 root: str = "input"):
    base = DEST_ROOT if root == "dest" else BROWSE_ROOT
    p = _safe_dir(path, base)
    dirs = sorted((d.name for d in p.iterdir()
                   if d.is_dir() and not d.name.startswith(".")), key=str.lower)
    nvideos = sum(1 for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
    return _page(request, "_browse.html", path=str(p),
                 parent=(str(p.parent) if p != base else None),
                 dirs=dirs, target=target, nvideos=nvideos, root=root)


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
