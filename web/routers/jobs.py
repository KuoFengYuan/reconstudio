"""Job views / status / list (htmx fragments) + JSON API + SSE log stream."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from jobs import COLMAP_STAGES, MAX_JOBS, manager
from web.shared import _page

router = APIRouter()


@router.get("/ui/joblist", response_class=HTMLResponse)
async def joblist(request: Request):
    running = sum(1 for j in manager.jobs.values() if j.status == "running")
    return _page(request, "_joblist.html", jobs=manager.list(),
                 max_jobs=MAX_JOBS, running=running)


@router.get("/ui/jobs/{job_id}", response_class=HTMLResponse)
async def jobview(request: Request, job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return _page(request, "_jobview.html", job=job.to_dict(), stages=COLMAP_STAGES)


@router.get("/ui/jobstatus/{job_id}", response_class=HTMLResponse)
async def jobstatus(request: Request, job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return _page(request, "_jobstatus.html", job=job.to_dict(), stages=COLMAP_STAGES)


@router.get("/api/jobs")
async def api_list_jobs():
    return manager.list()


@router.get("/api/jobs/{job_id}")
async def api_get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return job.to_dict()


@router.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    if not await manager.cancel(job_id):
        raise HTTPException(409, "job not cancellable")
    return {"ok": True}


@router.post("/api/jobs/delete")
async def delete_jobs(request: Request):
    """Multi-select: cancel active jobs, remove finished records (+ their files)."""
    data = await request.json()
    ids = data.get("ids", []) if isinstance(data, dict) else []
    return {"results": {jid: await manager.delete(jid) for jid in ids}}


@router.get("/api/jobs/{job_id}/logs")
async def stream_logs(job_id: str):
    """SSE: replay the whole console.log, then tail new lines until the job ends."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")

    async def gen():
        path = job.log_path
        pos = 0
        while True:
            if path.exists():
                with path.open("r", errors="replace") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
                if chunk:
                    # one SSE event per chunk (multi-line data) so the browser does a
                    # single DOM write — the initial history replay appears instantly
                    # instead of trickling in line-by-line.
                    payload = "".join(f"data: {ln}\n" for ln in chunk.splitlines())
                    yield payload + "\n"
            cur = manager.get(job_id)
            ended = cur and cur.status in ("done", "failed", "cancelled")
            no_more = not (path.exists() and path.stat().st_size > pos)
            if ended and no_more:
                yield f"event: end\ndata: {cur.status}\n\n"
                return
            await asyncio.sleep(0.4)

    # Disable proxy / port-forward buffering so events flush immediately.
    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })
