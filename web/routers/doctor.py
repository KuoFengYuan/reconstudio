"""Preflight / environment check — the "deploy to a new machine" checklist."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from pipeline import doctor as run_doctor
from pipeline import list_gpus
from web.shared import _page

router = APIRouter()


@router.get("/doctor", response_class=HTMLResponse)
async def doctor_page(request: Request, deep: int = 1):
    report = await asyncio.to_thread(run_doctor, bool(deep))
    return _page(request, "doctor.html", report=report, deep=bool(deep))


@router.get("/api/doctor")
async def api_doctor(deep: int = 1):
    return JSONResponse(await asyncio.to_thread(run_doctor, bool(deep)))


@router.get("/api/gpus")
async def api_gpus():
    return JSONResponse(await asyncio.to_thread(list_gpus))
