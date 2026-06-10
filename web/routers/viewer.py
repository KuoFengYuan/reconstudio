"""Standalone mesh viewer: open ANY local mesh file (not tied to a job).

The job-bound viewer (/viz/mesh/{job_id}) only shows a mesh job's own outputs;
this one takes a file path — picked via the same 瀏覽 modal or typed — so any
.ply/.obj/.stl/.glb on the server can be inspected with the same orbit/measure
tooling. Paths are confined to the browse roots, like the directory picker.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from web.shared import BROWSE_ROOT, DEST_ROOT, MESH_EXTS, SPLAT_EXTS, _page

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
