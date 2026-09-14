"""Reconstruction acceptance check — page + htmx result fragment.

`model_aligner` reports an EO RMS over the reference sensor's *positions* only,
so a reconstruction can look aligned while an entire camera holds zero
observations or the sensors sit tens of metres apart. This page runs the checks
that see that (pipeline/verify.py -> tools/verify_recon.py).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from jobs import manager
from pipeline.verify import guess_inputs, run_verify, verify_python
from web.services.models import model_dir, workspace_model
from web.shared import _page

router = APIRouter()

_MODEL_MARKERS = ("cameras.bin", "cameras.txt")


def _resolve(path: str) -> Path | None:
    """Accept either a model dir or the workspace holding one.

    Users think in workspaces ('the folder the job wrote to'), not in
    `sparse/0`, so take both and probe the same order the viewer does."""
    if not path.strip():
        return None
    p = Path(path.strip()).expanduser()
    if not p.is_dir():
        return None
    if any((p / m).is_file() for m in _MODEL_MARKERS):
        return p
    return workspace_model(p)


@router.get("/verify", response_class=HTMLResponse)
async def verify_page(request: Request, job: str = "", path: str = ""):
    """Standalone page. `job` prefills from a finished job, `path` from a URL."""
    prefill = {"path": path, "db": "", "manifest": "", "undistorted": False}
    if job and (j := manager.get(job)):
        if md := model_dir(j):
            prefill["path"] = str(md)
    if resolved := _resolve(prefill["path"]):
        prefill.update(guess_inputs(resolved))
        prefill["path"] = str(resolved)
    recent = [j for j in manager.list() if j["kind"] == "colmap"][:15]
    return _page(request, "verify.html", prefill=prefill, recent=recent,
                 checker_python=str(verify_python() or ""))


@router.post("/ui/verify", response_class=HTMLResponse)
async def verify_run(request: Request,
                     path: str = Form(""),
                     db: str = Form(""),
                     manifest: str = Form(""),
                     undistorted: str = Form(""),
                     maxper: int = Form(12)):
    model = _resolve(path)
    if model is None:
        return _page(request, "_verify_result.html",
                     res={"ok": False, "rc": -1, "cmd": "", "python": "",
                          "text": f"找不到 COLMAP 模型。給模型目錄或含 sparse/ 的 workspace:\n{path}"},
                     model="")
    guessed = guess_inputs(model)
    res = await asyncio.to_thread(
        run_verify, model,
        db or guessed["db"] or None,
        manifest or guessed["manifest"] or None,
        bool(undistorted) or guessed["undistorted"],
        int(maxper))
    return _page(request, "_verify_result.html", res=res, model=str(model))
