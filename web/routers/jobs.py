"""Job views / status / list (htmx fragments) + JSON API + SSE log/jobs streams."""
from __future__ import annotations

import asyncio
from collections import Counter

from fastapi import APIRouter, HTTPException, Request, WebSocket
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


# Replay only the last _LOG_TAIL_BYTES on connect, then tail live. A 4–5k-image
# COLMAP log is many MB; dumping the whole thing froze the browser. _LOG_READ_CHUNK
# caps how much we read per tick so a 20k-image run dumping megabytes between ticks
# can't turn one read into a multi-MB blocking op — the backlog is drained over
# successive reads instead (the `more` flag below skips the inter-tick sleep until
# caught up). The read+decode itself runs in a worker thread (asyncio.to_thread)
# so it never blocks the event loop / starves other requests.
_LOG_TAIL_BYTES = 256 * 1024
_LOG_READ_CHUNK = 512 * 1024


def _tail_read(path, pos, tail_bytes, max_chunk):
    """Blocking (call via asyncio.to_thread): read up to `max_chunk` new bytes.

    Returns (text, new_pos, started_mid_file, more_pending). First call (pos is
    None) starts from the last `tail_bytes` of the file."""
    if not path.exists():
        return "", pos, False, False
    size = path.stat().st_size
    started = False
    if pos is None:
        pos = max(0, size - tail_bytes)
        started = pos > 0
    if size <= pos:
        return "", pos, started, False
    with path.open("rb") as fh:
        fh.seek(pos)
        data = fh.read(max_chunk)
        pos = fh.tell()
    return data.decode("utf-8", "replace"), pos, started, size > pos


@router.get("/api/jobs/{job_id}/logs")
async def stream_logs(job_id: str):
    """SSE: replay the recent tail of console.log, then tail new lines until the job ends."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")

    async def gen():
        path = job.log_path
        pos = None            # byte offset we've streamed up to; None = not started
        trim_partial = False  # drop the half line at the tail's start
        first = True
        while True:
            text, pos, started, more = await asyncio.to_thread(
                _tail_read, path, pos, _LOG_TAIL_BYTES, _LOG_READ_CHUNK)
            if first:
                trim_partial = started
                first = False
            if text and trim_partial:            # started mid-line: drop the half line
                nl = text.find("\n")
                text = text[nl + 1:] if nl >= 0 else ""
                trim_partial = False
            if text:
                # one SSE event per chunk (multi-line data) so the browser does
                # a single DOM write instead of trickling in line-by-line.
                payload = "".join(f"data: {ln}\n" for ln in text.splitlines())
                yield payload + "\n"
            if more:
                continue                          # backlog remains — read again now, no wait
            cur = manager.get(job_id)
            # End only once the job is finished AND this read drained the file (no
            # text left), so a final burst is never cut off.
            if cur and cur.status in ("done", "failed", "cancelled") and not text:
                yield f"event: end\ndata: {cur.status}\n\n"
                return
            await asyncio.sleep(0.4)

    # Disable proxy / port-forward buffering so events flush immediately.
    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })


@router.websocket("/ws")
async def ws_bus(websocket: WebSocket):
    """One multiplexed channel per browser tab — supersedes the per-tab SSE pair
    (/api/jobs/stream + /api/jobs/{id}/logs). Both the job-list 'refresh' signal
    and the watched job's log lines ride this single WebSocket, so a tab holds ONE
    connection no matter how many jobs/logs it shows. WebSockets aren't subject to
    the browser's ~6-connection-per-host HTTP cap that the old SSE design
    exhausted (which froze new tabs on a blank "Loading…").

    Protocol — client → server: {"action":"watch_log","job_id":"…"} to tail a job's
    log, {"action":"unwatch_log"} to stop. server → client: {"type":"jobs"} on any
    job-state change; {"type":"log","job":id,"lines":[…]} for tailed output;
    {"type":"log_end","job":id,"status":…} once the tailed job finishes."""
    await websocket.accept()
    # Per-connection log-tail cursor; `job` None = not tailing anything.
    log = {"job": None, "pos": None, "first": True, "trim": False}

    async def reader():
        """client → server: (un)subscribe the connection to a job's log."""
        while True:
            try:
                msg = await websocket.receive_json()
            except ValueError:                       # malformed frame — ignore, keep socket
                continue
            action = (msg or {}).get("action")
            if action == "watch_log":
                log.update(job=(msg.get("job_id") or "").strip() or None,
                           pos=None, first=True, trim=False)
            elif action == "unwatch_log":
                log["job"] = None

    async def pusher():
        """server → client: job-state refresh + the watched job's log tail."""
        last_sig = None
        while True:
            sig = tuple(sorted((j.id, j.status, j.current_stage)
                               for j in manager.jobs.values()))
            if sig != last_sig:
                last_sig = sig
                await websocket.send_json({"type": "jobs"})
            jid = log["job"]
            if jid:
                job = manager.get(jid)
                if not job:
                    log["job"] = None
                else:
                    text, log["pos"], started, more = await asyncio.to_thread(
                        _tail_read, job.log_path, log["pos"], _LOG_TAIL_BYTES, _LOG_READ_CHUNK)
                    if log["first"]:
                        log["trim"] = started
                        log["first"] = False
                    if text and log["trim"]:            # started mid-line: drop the half line
                        nl = text.find("\n")
                        text = text[nl + 1:] if nl >= 0 else ""
                        log["trim"] = False
                    if text:
                        await websocket.send_json(
                            {"type": "log", "job": jid, "lines": text.splitlines()})
                    if more:
                        continue                        # backlog remains — read again, no wait
                    cur = manager.get(jid)
                    if cur and cur.status in ("done", "failed", "cancelled") and not text:
                        await websocket.send_json(
                            {"type": "log_end", "job": jid, "status": cur.status})
                        log["job"] = None               # stop tailing the finished job
            await asyncio.sleep(0.4)

    tasks = [asyncio.create_task(reader()), asyncio.create_task(pusher())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
