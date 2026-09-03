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
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pipeline import (
    COLMAP_STAGES,
    Cancelled,
    Runner,
    run_colmap,
    run_depth,
    run_frames,
    run_gcs_sync,
    run_gcs_upload,
    run_mesh,
    run_train,
)
from pipeline.colmap import COLMAP_DEFAULTS  # noqa: F401  (re-exported for app)
from pipeline.config import settings
from pipeline.frames import FRAMES_DEFAULTS  # noqa: F401

DATA_DIR = settings.data_dir
JOBS_DIR = settings.jobs_dir
# How many jobs may run concurrently. Tune per machine via COLMAP_PANEL_MAX_JOBS.
MAX_JOBS = settings.max_jobs

def _run_gcs(params: dict, runner: Runner) -> None:
    """A 'gcs' job downloads by default, or uploads when direction == 'upload'."""
    if params.get("direction") == "upload":
        return run_gcs_upload(params, runner)
    return run_gcs_sync(params, runner)


def _run_blocksplit(params: dict, runner: Runner) -> None:
    """Lazy import: blocksplit pulls numpy, which the panel base / CI's pure-Python
    install doesn't carry. Kept out of pipeline/__init__'s eager chain (like
    large_scene) so importing pipeline/jobs stays numpy-free."""
    from pipeline.blocksplit import run_blocksplit
    return run_blocksplit(params, runner)


def _run_depth_engine(params: dict, runner: Runner) -> None:
    """Pick the depth/normal engine, keeping both under the one "depth" job kind.

    Same kind on purpose: the engines write the same folders and emit the same
    progress markers, so the job view and _parse_depth below cover both. The
    dispatch lives here rather than in depth.py because moge3.py already imports
    from depth.py, and the reverse import would be circular. Lazy import for the
    same reason as _run_blocksplit: moge3.py should not join the eager chain.
    """
    if (params.get("engine") or "lichtfeld").strip() == "moge3":
        from pipeline.moge3 import run_moge3
        return run_moge3(params, runner)
    return run_depth(params, runner)


RUN_FUNCS: dict[str, Callable[[dict, Runner], None]] = {
    "frames": run_frames,
    "colmap": run_colmap,
    "blocksplit": _run_blocksplit,
    "train": run_train,
    "mesh": run_mesh,
    "gcs": _run_gcs,
    "depth": _run_depth_engine,
}

# --- COLMAP stage parsing --------------------------------------------------- #
_RUN_MARKERS = [
    (re.compile(r"stage nested layout"), "stage"),
    (re.compile(r"feature_extractor"), "extract"),
    (re.compile(r"sequential_matcher|vocab_tree_matcher|spatial_matcher|matches_importer"), "match"),
    (re.compile(r"view_graph_calibrator"), "calibrate"),
    (re.compile(r"colmap global_mapper|colmap mapper |colmap pose_prior_mapper|"
                r"colmap hierarchical_mapper"), "mapper"),
    (re.compile(r"simplify_images"), "simplify"),
    (re.compile(r"model_aligner"), "align"),
    (re.compile(r"image_undistorter"), "undistort"),
    (re.compile(r"auto_reorient"), "reorient"),
]
_SKIP_RE = re.compile(r"^\s*skip (\w+)")
_BANNER_RE = re.compile(r"^=== \[\d\d:\d\d:\d\d\]")
_DONE_RE = re.compile(r"done\. workspace=")

# --- frames parsing --------------------------------------------------------- #
_FR_FOUND = re.compile(r"Found (\d+) video")
_FR_CUR = re.compile(r"######## \[(\d+)/(\d+)\]")
_FR_RESULT = re.compile(r"-> (\d+) kept / (\d+) dropped")

# --- train parsing ---------------------------------------------------------- #
# tqdm's live bar is \r-based (won't stream as discrete lines through a pipe), so
# we lean on the trainer's newline-terminated markers: tqdm.write("[ITER n] …")
# and "Training complete!". The "n/total [" form is parsed too, for when a bar
# line does flush.
_TR_BAR = re.compile(r"(\d+)/(\d+)\s*\[")
_TR_ITER = re.compile(r"\[ITER\s+(\d+)\]")
_TR_LOSS = re.compile(r"loss[=:]\s*([0-9.]+)", re.IGNORECASE)
_TR_DONE = re.compile(r"Training complete")     # also matches LichtFeld "Training completed in"
# LichtFeld Studio's headless progress bar postfix: "<iter>/<total> | Loss: .. | Splats: ..".
# Its "Loss:" is already caught by _TR_LOSS; this captures iter/total from the same line.
_LF_BAR = re.compile(r"(\d+)/(\d+)\s*\|\s*Loss:")
# Human-readable phase, so the status chip shows life during the long, output-quiet
# stages (camera loading etc.) instead of looking frozen.
_TR_PHASES = [
    (re.compile(r"Loading train cameras"), "載入訓練相機"),
    (re.compile(r"Loading test cameras"), "載入測試相機"),
    (re.compile(r"Loading COLMAP|Loading dataset"), "載入資料集"),   # LichtFeld
    (re.compile(r"Number of points at init"), "初始化點雲"),
    (re.compile(r"Populating"), "建立多視角資料"),
    (re.compile(r"Saving Gaussians|Saving final model"), "存檔中"),  # +LichtFeld
    (re.compile(r"Saving checkpoint"), "存 checkpoint"),
]

# --- gcs parsing ------------------------------------------------------------ #
# gsutil rsync emits "Building/Starting synchronization", one "Copying gs://…"
# per transferred file, and a final "Operation completed over N objects/SIZE.".
_GCS_SYNC = re.compile(r"Building synchronization state|Starting synchronization")
_GCS_COPY = re.compile(r"^\s*(?:Copying|Downloading)\b")
_GCS_DONE = re.compile(r"Operation completed over (\d+) objects(?:/([\d.]+\s*\w+))?")

# --- mesh parsing ----------------------------------------------------------- #
_ME_RESULT = re.compile(r"\[mesh\] result:\s*(\S+)")
_ME_SCALED = re.compile(r"\[mesh\] scaled result:\s*(\S+)")
_ME_SCALE = re.compile(r"mm_per_unit=([0-9.]+)")
_ME_VERT = re.compile(r"Num vertices post:\s*(\d+)")
_ME_EXTRACT = re.compile(r"Extracting mesh from TSDF")
_ME_PHASES = [
    (re.compile(r"Found \d+ cameras"), "載入相機"),
    (re.compile(r"Estimated voxel|Voxel_size|TSDF config"), "估計體素 / 融合"),
    (re.compile(r"Extracting mesh"), "抽取 mesh"),
    (re.compile(r"Post processing|Post-processed"), "後處理"),
]


@dataclass
class Job:
    id: str
    kind: str                       # "frames" | "colmap" | "train" | "mesh"
    title: str
    subtitle: str
    params: dict = field(default_factory=dict)   # default lets old-schema job.json load
    meta: dict = field(default_factory=dict)
    mirror: str | None = None    # extra log file (e.g. workspace/pipeline.log)
    status: str = "queued"          # queued | running | done | failed | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    current_stage: str | None = None
    stage_status: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The Runner's reader thread mutates meta/stage_status (via the log
        # parsers) while the event loop snapshots the job for save()/to_dict().
        # Kept off the dataclass fields so asdict()/Job(**data) stay unaffected.
        self._lock = threading.Lock()

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    @property
    def log_path(self) -> Path:
        return self.dir / "console.log"

    def parse_line(self, parser: Callable[[Job, str], None], line: str) -> None:
        """Apply a log parser under the job lock (called from the Runner thread)."""
        with self._lock:
            parser(self, line)

    def to_dict(self) -> dict:
        with self._lock:
            return asdict(self)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = json.dumps(asdict(self), indent=2)
        # tmp + rename: a crash mid-write must never leave a truncated job.json
        # (_load_existing would silently drop the record).
        tmp = self.dir / "job.json.tmp"
        tmp.write_text(payload)
        os.replace(tmp, self.dir / "job.json")


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


def _parse_train(job: Job, line: str) -> None:
    for rx, label in _TR_PHASES:
        if rx.search(line):
            job.meta["phase"] = label
            break
    # Only the *training* bar (desc "Training") counts as the iteration progress;
    # the "Loading train cameras: …/245 [" bar must not be mistaken for it.
    if "Training" in line:
        m = _TR_BAR.search(line)
        if m:
            job.meta["iter"], job.meta["total"] = int(m.group(1)), int(m.group(2))
            job.meta["phase"] = "訓練中"
            job.current_stage = "train"
    m = _LF_BAR.search(line)          # LichtFeld: "<iter>/<total> | Loss: …"
    if m:
        job.meta["iter"], job.meta["total"] = int(m.group(1)), int(m.group(2))
        job.meta["phase"] = "訓練中"
        job.current_stage = "train"
    m = _TR_ITER.search(line)
    if m:
        job.meta["iter"] = int(m.group(1))
        job.current_stage = "train"
    m = _TR_LOSS.search(line)
    if m:
        job.meta["loss"] = m.group(1)
    if _TR_DONE.search(line):
        job.meta["complete"] = True
        job.meta["phase"] = "完成"
        job.current_stage = "done"


def _parse_mesh(job: Job, line: str) -> None:
    for rx, label in _ME_PHASES:
        if rx.search(line):
            job.meta["phase"] = label
            break
    if _ME_EXTRACT.search(line):
        job.current_stage = "extract"
    m = _ME_VERT.search(line)
    if m:
        job.meta["vertices"] = int(m.group(1))
    m = _ME_RESULT.search(line)
    if m:
        job.meta["mesh_path"] = m.group(1)
        job.current_stage = "done"
    m = _ME_SCALE.search(line)
    if m:
        job.meta["mm_per_unit"] = float(m.group(1))
    m = _ME_SCALED.search(line)
    if m:
        job.meta["mesh_scaled_path"] = m.group(1)
        job.meta["phase"] = "已縮放 (mm)"


def _parse_gcs(job: Job, line: str) -> None:
    if _GCS_SYNC.search(line):
        job.meta.setdefault("phase", "比對差異中")
    if _GCS_COPY.search(line):
        job.meta["copied"] = job.meta.get("copied", 0) + 1
        job.meta["phase"] = "下載中"
    m = _GCS_DONE.search(line)
    if m:
        job.meta["objects"] = int(m.group(1))
        if m.group(2):
            job.meta["size"] = m.group(2).strip()
        job.meta["phase"] = "完成"
        job.current_stage = "done"


# --- depth/normal parsing ---------------------------------------------------- #
# LichtFeld-Studio's `preprocess` subcommand: since stdout isn't a tty when piped
# from a subprocess, it skips its interactive progress bar and prints plain
# "Images: N under <dir>", one "[i/N] <path>" per image, and "Done. processed=N
# skipped=K" at the end (see src/preprocessing/preprocess.cpp).
_DE_IMAGES = re.compile(r"^Images: (\d+) under (.+)$")
_DE_CUR = re.compile(r"^\[(\d+)/(\d+)\] (.+)$")
_DE_DONE = re.compile(r"^Done\. processed=(\d+) skipped=(\d+)$")


def _parse_depth(job: Job, line: str) -> None:
    m = _DE_IMAGES.match(line)
    if m:
        job.meta["total"] = int(m.group(1))
        job.meta["out_dir"] = m.group(2).strip()
        job.meta["phase"] = "推論中"
        job.current_stage = "depth"
    m = _DE_CUR.match(line)
    if m:
        job.meta["cur"], job.meta["total"] = int(m.group(1)), int(m.group(2))
    m = _DE_DONE.match(line)
    if m:
        job.meta["written"] = int(m.group(1))
        job.meta["skipped"] = int(m.group(2))
        job.meta["phase"] = "完成"
        job.current_stage = "done"


# blocksplit: collect the per-block summary lines so the job view can offer a
# block picker (檢視/帶入訓練) once the split is done.
_BLOCKSPLIT_LINE = re.compile(r"^\[blocksplit\] (block_\d+_\d+): .*?→ (\d+) views")


def _parse_blocksplit(job: Job, line: str) -> None:
    m = _BLOCKSPLIT_LINE.match(line)
    if not m:
        return
    blocks = job.meta.setdefault("blocks", [])
    if not any(b.get("name") == m.group(1) for b in blocks):
        blocks.append({"name": m.group(1), "views": int(m.group(2))})


PARSERS: dict[str, Callable[[Job, str], None]] = {
    "colmap": _parse_colmap,
    "frames": _parse_frames,
    "train": _parse_train,
    "mesh": _parse_mesh,
    "gcs": _parse_gcs,
    "blocksplit": _parse_blocksplit,
    "depth": _parse_depth,
}


class JobManager:
    """Runs up to MAX_JOBS jobs concurrently (one asyncio worker per slot, each
    job's heavy work offloaded to a thread). GPU is chosen manually per job (the
    train/mesh forms' GPU field -> CUDA_VISIBLE_DEVICES); no auto GPU selection."""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.runners: dict[str, Runner] = {}
        self._workers: list[asyncio.Task] = []
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
        if not self._workers:
            self._workers = [asyncio.create_task(self._run_worker(i))
                             for i in range(MAX_JOBS)]

    def submit(self, job: Job) -> None:
        if job.kind == "colmap":
            for s in (job.params.get("stages") or COLMAP_STAGES):
                if s in COLMAP_STAGES:
                    job.stage_status[s] = "pending"
        self.jobs[job.id] = job
        job.save()
        self.queue.put_nowait(job.id)

    def register(self, job: Job) -> None:
        """Record an already-finished job (a COLMAP workspace imported from disk)
        without queueing any work — submit() would re-run the pipeline over it."""
        self.jobs[job.id] = job
        job.save()

    def find_by_workspace(self, kind: str, workspace: str) -> Job | None:
        """Newest finished job of `kind` whose meta points at this workspace, so
        re-importing the same folder reuses its record instead of piling up
        duplicates in 歷史."""
        hits = [j for j in self.jobs.values()
                if j.kind == kind and j.status == "done"
                and (j.meta or {}).get("workspace") == workspace]
        return max(hits, key=lambda j: j.created_at, default=None)

    def list(self) -> list[dict]:
        return [j.to_dict() for j in sorted(self.jobs.values(),
                                            key=lambda j: j.created_at, reverse=True)]

    def get(self, job_id: str) -> Job | None:
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

    async def _run_worker(self, wid: int) -> None:
        while True:
            job_id = await self.queue.get()
            job = None
            try:
                job = self.jobs.get(job_id)
                if job and job.status == "queued":
                    await self._run_job(job)
            except Exception as exc:    # never let an error kill the slot — but don't hide it:
                if job and job.status not in ("done", "failed", "cancelled"):
                    job.status = "failed"
                    job.error = str(exc)
                    job.save()
            finally:
                self.queue.task_done()

    async def _run_job(self, job: Job) -> None:
        job.status = "running"
        job.started_at = time.time()
        job.save()

        parser = PARSERS.get(job.kind, lambda j, ln: None)
        # Setup (make the workspace/mirror dir, open the log files) runs before any runner
        # exists and can fail on its own — most often an unwritable workspace path. Guard
        # it so the failure surfaces as a FAILED job with a logged reason, instead of the
        # job silently stalling in "running" with no console output.
        try:
            if job.mirror:
                Path(job.mirror).parent.mkdir(parents=True, exist_ok=True)
            job.dir.mkdir(parents=True, exist_ok=True)
            runner = Runner(job.log_path, on_line=lambda ln: job.parse_line(parser, ln),
                            mirror=job.mirror)
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = f"setup failed: {exc}"
            try:                          # console.log dir (job.dir) was made by save() above
                with open(job.log_path, "a") as fh:
                    fh.write(f"[panel] FAILED before start: {exc}\n"
                             f"[panel] workspace not writable? check the path/permissions: "
                             f"{job.mirror}\n")
            except Exception:
                pass
            job.finished_at = time.time()
            job.save()
            return
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
