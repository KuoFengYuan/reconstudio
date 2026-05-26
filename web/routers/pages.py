"""Top-level HTML pages."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from jobs import COLMAP_STAGES
from pipeline import TRAIN_DEFAULTS, available_backends, list_gpus
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
                 gpus=list_gpus(), gcs_root=GCS_ROOT)
