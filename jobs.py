"""Job manager: serialized background worker that runs the Python pipeline
modules (pipeline.run_frames / pipeline.run_colmap) and streams their log.

Each kind has a line parser that turns the pipeline's log output into progress:
COLMAP uses the "=== [HH:MM:SS] <stage> ===" / "skip <stage>" banners; frames
uses the "######## [j/N] …" and "-> K kept / D dropped" lines.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from pipeline import (COLMAP_STAGES, Cancelled, Runner, run_colmap, run_frames)
from pipeline.colmap import COLMAP_DEFAULTS  # noqa: F401  (re-exported for app)
from pipeline.frames import FRAMES_DEFAULTS  # noqa: F401

DATA_DIR = Path(os.environ.get("COLMAP_PANEL_DATA", str(Path.home() / ".colmap_panel")))
JOBS_DIR = DATA_DIR / "jobs"

RUN_FUNCS: dict[str, Callable[[dict, Runner], None]] = {
    "frames": run_frames,
    "colmap": run_colmap,
}

# --- COLMAP stage parsing --------------------------------------------------- #
_RUN_MARKERS = [
    (re.compile(r"stage nested layout"), "stage"),
    (re.compile(r"feature_extractor"), "extract"),
    (re.compile(r"sequential_matcher|vocab_tree_matcher"), "match"),
    (re.compile(r"view_graph_calibrator"), "calibrate"),
    (re.compile(r"global_mapper|colmap mapper "), "mapper"),
    (re.compile(r"image_undistorter"), "undistort"),
]
_SKIP_RE = re.compile(r"^\s*skip (\w+)")
_BANNER_RE = re.compile(r"^=== \[\d\d:\d\d:\d\d\]")
_DONE_RE = re.compile(r"done\. workspace=")

# --- frames parsing --------------------------------------------------------- #
_FR_FOUND = re.compile(r"Found (\d+) video")
_FR_CUR = re.compile(r"######## \[(\d+)/(\d+)\]")
_FR_RESULT = re.compile(r"-> (\d+) kept / (\d+) dropped")


@dataclass
class Job:
    id: str
    kind: str                       # "frames" | "colmap"
    title: str
    subtitle: str
    params: dict = field(default_factory=dict)   # default lets old-schema job.json load
    meta: dict = field(default_factory=dict)
    mirror: Optional[str] = None    # extra log file (e.g. workspace/pipeline.log)
    status: str = "queued"          # queued | running | done | failed | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    current_stage: Optional[str] = None
    stage_status: dict[str, str] = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    @property
    def log_path(self) -> Path:
        return self.dir / "console.log"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "job.json").write_text(json.dumps(asdict(self), indent=2))


def _parse_colmap(job: Job, line: str) -> None:
    skip = _SKIP_RE.match(line)
    if skip and skip.group(1) in COLMAP_STAGES:
        job.stage_status[skip.group(1)] = "skipped"
        return
    if _DONE_RE.search(line) and _BANNER_RE.match(line):
        if job.current_stage:
            job.stage_status[job.current_stage] = "done"
        return
    if not _BANNER_RE.match(line):
        return
    for rx, stage in _RUN_MARKERS:
        if rx.search(line):
            if job.current_stage and job.current_stage != stage and \
                    job.stage_status.get(job.current_stage) == "running":
                job.stage_status[job.current_stage] = "done"
            job.current_stage = stage
            job.stage_status[stage] = "running"
            break


def _parse_frames(job: Job, line: str) -> None:
    m = _FR_FOUND.search(line)
    if m:
        job.meta["total"] = int(m.group(1))
    m = _FR_CUR.search(line)
    if m:
        job.meta["cur"], job.meta["total"] = int(m.group(1)), int(m.group(2))
        job.current_stage = f"video {m.group(1)}/{m.group(2)}"
    m = _FR_RESULT.search(line)
    if m:
        job.meta["kept"] = job.meta.get("kept", 0) + int(m.group(1))
        job.meta["dropped"] = job.meta.get("dropped", 0) + int(m.group(2))


PARSERS: dict[str, Callable[[Job, str], None]] = {
    "colmap": _parse_colmap,
    "frames": _parse_frames,
}


class JobManager:
    """Single-worker, serialized execution (both ffmpeg and COLMAP are heavy)."""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.runners: dict[str, Runner] = {}
        self._worker: Optional[asyncio.Task] = None
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        for jf in sorted(JOBS_DIR.glob("*/job.json")):
            try:
                data = json.loads(jf.read_text())
                if data.get("status") in ("queued", "running"):
                    data["status"] = "failed"   # stale from a previous server run
                job = Job(**{k: data[k] for k in data if k in Job.__dataclass_fields__})
                self.jobs[job.id] = job
            except Exception:
                continue

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_worker())

    def submit(self, job: Job) -> None:
        if job.kind == "colmap":
            for s in (job.params.get("stages") or COLMAP_STAGES):
                if s in COLMAP_STAGES:
                    job.stage_status[s] = "pending"
        self.jobs[job.id] = job
        job.save()
        self.queue.put_nowait(job.id)

    def list(self) -> list[dict]:
        return [j.to_dict() for j in sorted(self.jobs.values(),
                                            key=lambda j: j.created_at, reverse=True)]

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    async def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.status == "queued":
            job.status = "cancelled"
            job.save()
            return True
        runner = self.runners.get(job_id)
        if runner:
            runner.cancel()
            return True
        return False

    async def delete(self, job_id: str) -> str:
        """Cancel an active job, or remove a finished job's record + files.
        Returns 'cancelled' | 'deleted' | 'missing'."""
        job = self.jobs.get(job_id)
        if not job:
            return "missing"
        if job.status in ("queued", "running"):
            await self.cancel(job_id)
            return "cancelled"          # finished records can be removed on a 2nd pass
        self.jobs.pop(job_id, None)
        shutil.rmtree(job.dir, ignore_errors=True)
        return "deleted"

    async def _run_worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            job = self.jobs.get(job_id)
            if not job or job.status != "queued":
                continue
            await self._run_job(job)

    async def _run_job(self, job: Job) -> None:
        job.status = "running"
        job.started_at = time.time()
        job.save()

        parser = PARSERS.get(job.kind, lambda j, l: None)
        if job.mirror:
            Path(job.mirror).parent.mkdir(parents=True, exist_ok=True)
        job.dir.mkdir(parents=True, exist_ok=True)
        runner = Runner(job.log_path, on_line=lambda ln: parser(job, ln), mirror=job.mirror)
        self.runners[job.id] = runner
        runner.log(f"[panel] {job.kind} job {job.id}")
        runner.log(f"[panel] params: {json.dumps(job.params, ensure_ascii=False)}")
        runner.log("")

        fn = RUN_FUNCS[job.kind]
        try:
            await asyncio.to_thread(fn, job.params, runner)
            job.status = "done"
        except Cancelled:
            job.status = "cancelled"
            runner.log("\n[panel] cancelled")
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = str(exc)
            runner.log(f"\n[panel] FAILED: {exc}")
        finally:
            if job.current_stage and job.stage_status.get(job.current_stage) == "running":
                job.stage_status[job.current_stage] = "done" if job.status == "done" else "failed"
            runner.close()
            self.runners.pop(job.id, None)
            job.finished_at = time.time()
            job.save()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


manager = JobManager()
