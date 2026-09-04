"""Top-level HTML pages."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from jobs import COLMAP_STAGES
from pipeline import (
    DEPTH_DEFAULTS,
    MATTE_DEFAULTS,
    MOGE3_DEFAULTS,
    TRAIN_DEFAULTS,
    available_backends,
    depth_ready,
    list_gpus,
    matte_ready,
    moge3_ready,
)
from web.shared import (
    BROWSE_ROOT,
    COLMAP_DEFAULTS,
    ENUMS,
    FRAMES_DEFAULTS,
    GCS_ROOT,
    _page,
)

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _page(request, "index.html", colmap_defaults=COLMAP_DEFAULTS,
                 frames_defaults=FRAMES_DEFAULTS, enums=ENUMS,
                 stages=COLMAP_STAGES, browse_root=str(BROWSE_ROOT),
                 train_defaults=TRAIN_DEFAULTS, backends=available_backends(),
                 gpus=list_gpus(), gcs_root=GCS_ROOT,
                 depth_defaults=DEPTH_DEFAULTS, depth_ready=depth_ready(),
                 moge3_defaults=MOGE3_DEFAULTS, moge3_ready=moge3_ready(),
                 matte_defaults=MATTE_DEFAULTS, matte_ready=matte_ready(),
                 # the form is usable as long as EITHER engine can run
                 depth_any_ready=depth_ready() or moge3_ready())
