"""融合 form: two matted captures of one subject -> one COLMAP input.

Deliberately NOT a job kind. All this does is build the staging links and then
hand the result to the existing COLMAP form (`fillFusionColmap` in index.html),
so the reconstruction itself is an ordinary `colmap` job with the user's own
COLMAP interface, options, log parser, viewer and training hand-off. A second
copy of the COLMAP form's knobs is exactly what this must not grow into.

Staging is instant (symlinks), so it is a plain request rather than something
queued: the answer the form needs — how many photos and masks each pass has,
whether the sizes agree — is only useful before submitting.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from pipeline import fusion
from web.shared import _page

router = APIRouter()


@router.post("/ui/fusion/prepare", response_class=HTMLResponse)
async def fusion_prepare(request: Request):
    form = dict(await request.form())
    inputs = [(form.get(k) or "").strip().rstrip("/")
              for k in ("input_a", "input_b") if (form.get(k) or "").strip()]
    out_raw = (form.get("out_dir") or "").strip().rstrip("/")
    try:
        passes = await asyncio.to_thread(fusion.inspect, inputs)
        out_dir = Path(out_raw).expanduser() if out_raw else fusion.default_out_dir(passes)
        images_root, masks_root, workspace = await asyncio.to_thread(
            fusion.stage, passes, out_dir)
    except (ValueError, OSError) as exc:
        return _page(request, "_error.html", message=str(exc))

    facts, warnings = fusion.summarize(passes)
    single_camera = len({p.sizes[0] for p in passes if p.sizes}) <= 1
    return _page(request, "_fusion_ready.html", passes=passes, facts=facts,
                 warnings=warnings, out_dir=str(out_dir),
                 images_root=str(images_root), masks_root=str(masks_root),
                 workspace=str(workspace), single_camera=single_camera,
                 total=sum(p.n_images for p in passes))
