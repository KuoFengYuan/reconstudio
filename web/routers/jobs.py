"""Job views / status / list (htmx fragments) + JSON API + SSE log/jobs streams."""
from __future__ import annotations

import asyncio
from collections import Counter

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from jobs import COLMAP_STAGES, MAX_JOBS, manager
from web.shared import _page

router = APIRouter()


@router.get("/ui/joblist", response_class=HTMLResponse)
async def joblist(request: Request,
                  q: str = "", kind: str = "all", status: str = "all",
                  limit: int = 50):
    """Filtered + paginated job table fragment.

    `q` matches title or id (substring, case-insensitive); kind/status='all' means
    no filter on that axis. `limit` is the visible-row cap ('Load more' grows it).
    Chip counts are computed on the FULL set so the user sees true totals."""
    all_jobs = manager.list()                                       # sorted newest-first
    kind_counts = Counter(j["kind"] for j in all_jobs)
    status_counts = Counter(j["status"] for j in all_jobs)

    qn = (q or "").strip().lower()

    def match(j) -> bool:
        if kind != "all" and j["kind"] != kind:
            return False
        if status != "all" and j["status"] != status:
            return False
        if qn:
            hay = ((j.get("title") or "") + "\0" + (j.get("id") or "")).lower()
            if qn not in hay:
                return False
        return True
    filtered = [j for j in all_jobs if match(j)]
    limit = max(10, min(int(limit or 50), 1000))
    visible = filtered[:limit]

    running = sum(1 for j in all_jobs if j["status"] == "running")
    return _page(request, "_joblist.html",
                 jobs=visible, total=len(filtered), all_total=len(all_jobs),
                 kind_counts=dict(kind_counts), status_counts=dict(status_counts),
                 q=q, sel_kind=kind, sel_status=status, limit=limit,
                 max_jobs=MAX_JOBS, running=running)


@router.get("/api/jobs/stream")
async def jobs_stream():
    """SSE: emit `refresh` whenever any job's status/stage changes.

    No pub/sub hooks in JobManager — we poll its in-memory state at 1 Hz here
    and only push when the signature differs. Idle = 0 events; the client's
    EventSource auto-reconnects on drop."""
    async def gen():
        last = None
        while True:
            cur = tuple(sorted(
                (j.id, j.status, j.current_stage) for j in manager.jobs.values()
            ))
            if cur != last:
                last = cur
                yield "event: refresh\ndata: ok\n\n"
            await asyncio.sleep(1.0)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })


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

    # Replay only the last TAIL_BYTES on connect, then tail live. A 4–5k-image
    # COLMAP log is many MB; dumping the whole thing froze the browser. The recent
    # tail is what matters live, and the full log stays on disk for forensics.
    TAIL_BYTES = 256 * 1024

    async def gen():
        path = job.log_path
        pos = None            # byte offset we've streamed up to; None = not started
        trim_partial = False  # drop the half line at the tail's start
        while True:
            if path.exists():
                size = path.stat().st_size
                if pos is None:
                    pos = max(0, size - TAIL_BYTES)
                    trim_partial = pos > 0
                if size > pos:
                    with path.open("rb") as fh:
                        fh.seek(pos)
                        data = fh.read()
                        pos = fh.tell()
                    text = data.decode("utf-8", "replace")
                    if trim_partial:                 # we started mid-line: drop it
                        nl = text.find("\n")
                        text = text[nl + 1:] if nl >= 0 else ""
                        trim_partial = False
                    if text:
                        # one SSE event per chunk (multi-line data) so the browser does
                        # a single DOM write instead of trickling in line-by-line.
                        payload = "".join(f"data: {ln}\n" for ln in text.splitlines())
                        yield payload + "\n"
            cur = manager.get(job_id)
            ended = cur and cur.status in ("done", "failed", "cancelled")
            no_more = not (path.exists() and path.stat().st_size > (pos or 0))
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
