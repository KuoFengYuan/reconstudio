"""Recon Studio - web front-end over the pure-Python pipeline modules.

Single-machine local tool. Binds to 127.0.0.1 by default. Run:
    ./run.sh            # or: uvicorn app:app --host 127.0.0.1 --port 8077

Two job kinds, end-to-end:
    video dir ──[Frames]──► <out>/<group>/frames_<video>/*.jpg ──[COLMAP]──► sparse/dense

Frontend is server-rendered Jinja + htmx (partial fragments under /ui/*).
JSON API lives under /api/*; logs stream over SSE.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from jobs import COLMAP_STAGES, MAX_JOBS, Job, manager, new_id
from pipeline import (TRAIN_DEFAULTS, available_backends, build_cli,
                      doctor as run_doctor, get_backend, list_gpus)

BASE = Path(__file__).parent
# Restrict the directory browser to this root (the data disk by default).
BROWSE_ROOT = Path(os.environ.get("RECON_STUDIO_BROWSE_ROOT", "/")).resolve()
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")   # PATH by default; override via env

app = FastAPI(title="Recon Studio")
templates = Jinja2Templates(directory=str(BASE / "templates"))
(BASE / "static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

# Form pre-fill defaults (mirror the pipeline module defaults).
COLMAP_DEFAULTS = {
    "CAMERA_MODEL": "OPENCV", "MAX_FEATURES": "4096", "CAMERA_MODE": "per_folder",
    "MATCHER": "both", "SEQ_OVERLAP": "10", "NUM_MATCHES": "50",
    "GUIDED_MATCHING": "1", "MAPPER": "global", "DATASET_NAME": "training_dataset",
    "FORCE": "0", "NESTED_LAYOUT": "1", "VOCAB_TREE": "", "VOCAB_TREE_URL": "",
}
FRAMES_DEFAULTS = {"FPS": "1", "MODE": "percentile", "KEEP_PCT": "70", "THRESHOLD": ""}
ENUMS = {
    "CAMERA_MODE": ["per_folder", "single"],
    "MATCHER": ["sequential", "vocab", "both"],
    "MAPPER": ["global", "incremental"],
}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".mkv", ".avi"}


@app.on_event("startup")
async def _startup() -> None:
    manager.start()


def _page(request: Request, name: str, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name=name, context=ctx)


def _scene_label(path: str) -> str:
    """Short, dataset-distinguishing label = '<workspace>/<leaf>'.

    Train/mesh inputs share generic leaf names (training_dataset_global_mapper,
    or a model dir named after the backend), so the leaf alone can't tell two
    datasets apart. The parent (the COLMAP workspace) is what differs, e.g.
    /Disk0/.../0520_iron_13mm_colmap/gs2m -> '0520_iron_13mm_colmap/gs2m'.
    """
    p = Path(path)
    return f"{p.parent.name}/{p.name}" if p.parent.name else p.name


def _parse_marker(form: dict, spec: dict) -> dict | None:
    """Build the marker-scaling config when「提供 marker」is ticked, else None.

    The board geometry is taken from the backend's configured `marker_defaults`
    (so the user never types numbers — that's the whole point). A per-job text
    field may override any value, but blank means "use the configured default"."""
    if not form.get("marker_enable"):
        return None
    d = dict(spec.get("marker_defaults") or {})
    if not d:
        raise ValueError("此 backend 未設定標定板規格 (marker_defaults);請在 backends.json 補上後再用。")

    def pick(key):  # per-job override (text field) wins over the configured default
        v = (form.get("marker_" + key) or "").strip()
        return v if v else d.get(key)
    try:
        sx, sy = int(pick("squares_x")), int(pick("squares_y"))
        sq, mk = float(pick("square_mm")), float(pick("marker_mm"))
    except (TypeError, ValueError):
        raise ValueError("marker 覆寫值需為數字 (方格數為整數、邊長 mm 為數值),或留空用預設規格。")
    if min(sx, sy) < 2 or min(sq, mk) <= 0:
        raise ValueError("marker 參數不合理:方格數需 ≥ 2,邊長 (mm) 需 > 0。")
    if mk >= sq:
        raise ValueError("marker 邊長應小於方格邊長 (ChArUco marker 嵌在方格內)。")
    return {"enable": True, "squares_x": sx, "squares_y": sy,
            "square_mm": sq, "marker_mm": mk,
            "dict": str(pick("dict") or "DICT_5X5_100").strip()}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _page(request, "index.html", colmap_defaults=COLMAP_DEFAULTS,
                 frames_defaults=FRAMES_DEFAULTS, enums=ENUMS,
                 stages=COLMAP_STAGES, browse_root=str(BROWSE_ROOT),
                 train_defaults=TRAIN_DEFAULTS, backends=available_backends(),
                 gpus=list_gpus())


# --------------------------------------------------------------------------- #
# Directory picker (htmx fragment)
# --------------------------------------------------------------------------- #
def _safe_dir(path: str) -> Path:
    p = (BROWSE_ROOT if not path else Path(path)).resolve()
    if p != BROWSE_ROOT and BROWSE_ROOT not in p.parents:
        raise HTTPException(400, f"path outside browse root ({BROWSE_ROOT})")
    if not p.is_dir():
        raise HTTPException(400, f"not a directory: {p}")
    return p


@app.get("/ui/browse", response_class=HTMLResponse)
async def browse(request: Request, path: str = "", target: str = "image_root"):
    p = _safe_dir(path)
    dirs = sorted((d.name for d in p.iterdir()
                   if d.is_dir() and not d.name.startswith(".")), key=str.lower)
    nvideos = sum(1 for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
    return _page(request, "_browse.html", path=str(p),
                 parent=(str(p.parent) if p != BROWSE_ROOT else None),
                 dirs=dirs, target=target, nvideos=nvideos)


# --------------------------------------------------------------------------- #
# Frames job  (pipeline.run_frames, flatten)
# --------------------------------------------------------------------------- #
@app.post("/ui/frames", response_class=HTMLResponse)
async def create_frames(request: Request):
    form = dict(await request.form())
    inp = (form.get("input") or "").strip().rstrip("/")
    out = (form.get("out_dir") or "").strip().rstrip("/")
    fps = (form.get("fps") or "1").strip()
    mode = form.get("mode") or "percentile"
    keep = (form.get("keep_pct") or "70").strip()
    thr = (form.get("threshold") or "").strip()
    try:
        if not (Path(inp).is_dir() or Path(inp).is_file()):
            raise ValueError(f"input not found: {inp!r}")
        if not out:
            raise ValueError("output dir is required")
        try:
            float(fps)
        except ValueError:
            raise ValueError("fps must be a number")
        params = {"inputs": [inp], "out_dir": out, "fps": fps,
                  "flatten": True, "ffmpeg_bin": FFMPEG_BIN}
        if mode == "threshold":
            if not thr:
                raise ValueError("threshold mode needs a value")
            params["threshold"] = thr
        else:
            if not keep.isdigit():
                raise ValueError("keep_percent must be an integer")
            params["keep_pct"] = keep
    except ValueError as exc:
        return _page(request, "_error.html", message=str(exc))

    ws = out + "_ws"
    job = Job(id=new_id(), kind="frames",
              title=f"{Path(inp).name} → {Path(out).name}",
              subtitle=f"{inp}  →  {out}",
              params=params, meta={"input": inp, "out": out, "ws": ws})
    manager.submit(job)
    return _page(request, "_jobview.html", job=job.to_dict(), stages=COLMAP_STAGES)


# --------------------------------------------------------------------------- #
# COLMAP job  (pipeline.run_colmap)
# --------------------------------------------------------------------------- #
def _build_colmap_params(image_root: str, workspace: str, folders: list[str],
                         stages: list[str], form: dict) -> dict:
    def g(key: str, default: str) -> str:
        v = (form.get(key) or "").strip()
        return v if v else default

    params = {
        "image_root": image_root, "workspace": workspace, "folders": folders,
        "stages": stages,
        "camera_model": g("CAMERA_MODEL", "OPENCV"),
        "camera_mode": g("CAMERA_MODE", "per_folder"),
        "max_features": g("MAX_FEATURES", "4096"),
        "matcher": g("MATCHER", "both"),
        "seq_overlap": g("SEQ_OVERLAP", "10"),
        "num_matches": g("NUM_MATCHES", "50"),
        "guided_matching": "1" if form.get("GUIDED_MATCHING") else "0",
        "mapper": g("MAPPER", "global"),
        "dataset_name": g("DATASET_NAME", "training_dataset"),
        "vocab_tree": (form.get("VOCAB_TREE") or "").strip() or None,
        "vocab_tree_url": (form.get("VOCAB_TREE_URL") or "").strip() or None,
        "force": bool(form.get("FORCE")),
        "layout": form.get("layout") or "auto",
        "resize": form.get("resize") or "fullhd",
    }
    for key, allowed in (("camera_mode", ENUMS["CAMERA_MODE"]),
                         ("matcher", ENUMS["MATCHER"]), ("mapper", ENUMS["MAPPER"]),
                         ("layout", ["auto", "single", "multi", "nested"]),
                         ("resize", ["keep", "fullhd"])):
        if params[key] not in allowed:
            raise ValueError(f"{key} must be one of {allowed}")
    for key in ("max_features", "seq_overlap", "num_matches"):
        if not str(params[key]).isdigit():
            raise ValueError(f"{key} must be an integer")
    return params


@app.post("/ui/jobs", response_class=HTMLResponse)
async def create_colmap(request: Request):
    raw = await request.form()
    form = dict(raw)
    image_root = (form.get("image_root") or "").strip().rstrip("/")
    workspace = (form.get("workspace") or "").strip().rstrip("/")
    subfolders = form.get("subfolders") or ""
    stages = raw.getlist("stages") or list(COLMAP_STAGES)
    try:
        if not Path(image_root).is_dir():
            raise ValueError(f"image_root is not a directory: {image_root!r}")
        if not workspace:
            raise ValueError("workspace is required")
        sub = [s.strip() for s in subfolders.replace(",", "\n").splitlines() if s.strip()]
        for s in sub:
            if not (Path(image_root) / s).is_dir():
                raise ValueError(f"subfolder not found: {s}")
        params = _build_colmap_params(image_root, workspace, sub, list(stages), form)
    except ValueError as exc:
        return _page(request, "_error.html", message=str(exc))

    job = Job(id=new_id(), kind="colmap",
              title=f"{Path(image_root).name} → {Path(workspace).name}",
              subtitle=f"{image_root}  →  {workspace}",
              params=params, mirror=str(Path(workspace) / "pipeline.log"),
              meta={"image_root": image_root, "workspace": workspace, "subfolders": sub})
    manager.submit(job)
    return _page(request, "_jobview.html", job=job.to_dict(), stages=COLMAP_STAGES)


# --------------------------------------------------------------------------- #
# Train job  (pipeline.run_train; runs in the backend's own conda env)
# --------------------------------------------------------------------------- #
@app.post("/ui/train", response_class=HTMLResponse)
async def create_train(request: Request):
    form = dict(await request.form())
    source = (form.get("source") or "").strip().rstrip("/")
    out = (form.get("model_path") or "").strip().rstrip("/")
    backend = (form.get("backend") or "gs2m").strip()
    gpu = (form.get("gpu") or "").strip()
    extra = (form.get("extra") or "").strip()
    try:
        if not Path(source).is_dir():
            raise ValueError(f"source 不是資料夾: {source!r}")
        if not out:
            raise ValueError("輸出模型目錄 (model_path) 必填")
        spec = get_backend(backend)
        if not spec:
            raise ValueError(f"未知 backend: {backend}")
        cli = build_cli(spec.get("params", []), form)   # tunable params -> CLI
        params = {"backend": backend, "source": source, "model_path": out,
                  "gpu": gpu, "args": cli, "extra": extra,
                  "force": bool(form.get("force"))}
    except ValueError as exc:
        return _page(request, "_error.html", message=str(exc))

    # Total iterations for the progress bar: --iterations in the assembled CLI/extra.
    total = 30000
    m = re.search(r"--iterations\s+(\d+)", f"{cli} {extra}")
    if m:
        total = int(m.group(1))

    job = Job(id=new_id(), kind="train",
              title=f"{backend} · {_scene_label(out)}",
              subtitle=f"{source}  →  {out}  (gpu={gpu or 'default'})",
              params=params, mirror=str(Path(out) / "train.log"),
              meta={"source": source, "model_path": out, "backend": backend,
                    "total": total, "can_mesh": bool(spec.get("mesh_args"))})
    manager.submit(job)
    return _page(request, "_jobview.html", job=job.to_dict(), stages=COLMAP_STAGES)


# --------------------------------------------------------------------------- #
# Mesh job  (pipeline.run_mesh; backend-specific, e.g. GS-2M's render.py)
# --------------------------------------------------------------------------- #
@app.post("/ui/mesh", response_class=HTMLResponse)
async def create_mesh(request: Request):
    form = dict(await request.form())
    upload = form.pop("edited_ply", None)        # optional SuperSplat-edited (去背) cloud
    model = (form.get("model_path") or "").strip().rstrip("/")
    backend = (form.get("backend") or "").strip()
    gpu = (form.get("gpu") or "").strip()
    extra = (form.get("extra") or "").strip()
    edited_from = None
    try:
        if not Path(model).is_dir():
            raise ValueError(f"model_path 不是資料夾: {model!r}")
        if not (Path(model) / "cfg_args").is_file():
            raise ValueError("這不像訓練輸出（缺 cfg_args）;請指向 train.py 的 -m 輸出目錄。")
        spec = get_backend(backend)
        if not spec:
            raise ValueError(f"未知 backend: {backend}")
        if not spec.get("mesh_args"):
            raise ValueError(f"backend '{backend}' 不支援 mesh 抽取。")
        # A去背過的點雲上傳時:從原模型衍生一個非破壞性的 _edited_<時間> 兄弟目錄,
        # 對它抽 mesh;原模型完全不動(這就是 pipeline 層的「還原」)。
        if upload is not None and getattr(upload, "filename", ""):
            if not upload.filename.lower().endswith(".ply"):
                raise ValueError("編輯後的點雲需為 .ply 檔（SuperSplat 匯出的 PLY）。")
            edited_dir = await asyncio.to_thread(_prepare_edited_model, Path(model), upload)
            edited_from, model = model, str(edited_dir)
        cli = build_cli(spec.get("mesh_params", []), form)
        params = {"backend": backend, "model_path": model, "gpu": gpu,
                  "args": cli, "extra": extra}
        marker = _parse_marker(form, spec)   # optional ChArUco scaling -> mm mesh
        if marker:
            params["marker"] = marker
    except ValueError as exc:
        return _page(request, "_error.html", message=str(exc))

    meta = {"model_path": model, "backend": backend, "marker": bool(marker)}
    if edited_from:
        meta["edited_from"] = edited_from
    job = Job(id=new_id(), kind="mesh",
              title=f"mesh · {_scene_label(model)}" + (" · 去背" if edited_from else "")
                    + (" · scaled" if marker else ""),
              subtitle=f"{model}  (gpu={gpu or 'default'})",
              params=params, mirror=str(Path(model) / "mesh.log"),
              meta=meta)
    manager.submit(job)
    return _page(request, "_jobview.html", job=job.to_dict(), stages=COLMAP_STAGES)


# --------------------------------------------------------------------------- #
# Preflight / environment check — the "deploy to a new machine" checklist
# --------------------------------------------------------------------------- #
@app.get("/doctor", response_class=HTMLResponse)
async def doctor_page(request: Request, deep: int = 1):
    report = await asyncio.to_thread(run_doctor, bool(deep))
    return _page(request, "doctor.html", report=report, deep=bool(deep))


@app.get("/api/doctor")
async def api_doctor(deep: int = 1):
    return JSONResponse(await asyncio.to_thread(run_doctor, bool(deep)))


@app.get("/api/gpus")
async def api_gpus():
    return JSONResponse(await asyncio.to_thread(list_gpus))


# --------------------------------------------------------------------------- #
# Job views / status / list  (htmx fragments) + JSON + SSE logs
# --------------------------------------------------------------------------- #
@app.get("/ui/joblist", response_class=HTMLResponse)
async def joblist(request: Request):
    running = sum(1 for j in manager.jobs.values() if j.status == "running")
    return _page(request, "_joblist.html", jobs=manager.list(),
                 max_jobs=MAX_JOBS, running=running)


@app.get("/ui/jobs/{job_id}", response_class=HTMLResponse)
async def jobview(request: Request, job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return _page(request, "_jobview.html", job=job.to_dict(), stages=COLMAP_STAGES)


@app.get("/ui/jobstatus/{job_id}", response_class=HTMLResponse)
async def jobstatus(request: Request, job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return _page(request, "_jobstatus.html", job=job.to_dict(), stages=COLMAP_STAGES)


@app.get("/api/jobs")
async def api_list_jobs():
    return manager.list()


@app.get("/api/jobs/{job_id}")
async def api_get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    if not await manager.cancel(job_id):
        raise HTTPException(409, "job not cancellable")
    return {"ok": True}


@app.post("/api/jobs/delete")
async def delete_jobs(request: Request):
    """Multi-select: cancel active jobs, remove finished records (+ their files)."""
    data = await request.json()
    ids = data.get("ids", []) if isinstance(data, dict) else []
    return {"results": {jid: await manager.delete(jid) for jid in ids}}


# --------------------------------------------------------------------------- #
# Result visualization (sparse point cloud + camera poses)
# --------------------------------------------------------------------------- #
def _model_dir(job) -> Path | None:
    ws = (job.meta or {}).get("workspace")
    if not ws:
        return None
    md = Path(ws) / "sparse" / "0"
    return md if (md / "images.bin").exists() and (md / "points3D.bin").exists() else None


def _trained_model_dir(job) -> Path | None:
    """The trained-model output dir, from a train job's meta. Validated by the
    presence of a trained 3DGS .ply (backend-agnostic). Distinct from _model_dir,
    which is the COLMAP sparse reconstruction."""
    mp = (job.meta or {}).get("model_path")
    if not mp:
        return None
    d = Path(mp)
    return d if (d.is_dir() and _trained_ply(d)) else None


def _trained_ply(model_dir: Path) -> Path | None:
    """Highest-iteration trained 3DGS .ply, across trainer layouts:
       GS-2M     → point_cloud/iteration_<N>/point_cloud.ply
       LichtFeld → splat_<N>.ply (in the output dir)."""
    cands = list(model_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    cands += list(model_dir.glob("splat_*.ply"))
    if not cands:
        return None
    def it(p: Path) -> int:
        token = p.parent.name if p.name == "point_cloud.ply" else p.stem
        m = re.search(r"(\d+)", token)
        return int(m.group(1)) if m else -1
    return max(cands, key=it)


def _make_edited_model(model: Path) -> tuple[Path, Path]:
    """Create a non-destructive sibling model dir for a SuperSplat-edited (去背)
    cloud and return (edited_root, dest_ply_path). The caller writes the bytes.

    Mirrors the `<model>_scene` symlink trick (train.py): the original model dir
    is never touched. The sibling gets its own
    point_cloud/iteration_<N>/point_cloud.ply (the edited cloud goes there) and a
    symlink to the original cfg_args. GS-2M's render.py then loads the edited
    cloud, reads cameras from the original source_path (recorded in cfg_args),
    and writes the mesh under the sibling — so a bad去背 can't destroy the source."""
    src_ply = _trained_ply(model)
    if not src_ply:
        raise ValueError(f"原模型找不到 point_cloud.ply（{model}）。")
    it_dir = src_ply.parent                       # .../point_cloud/iteration_<N>
    edited = model.parent / f"{model.name}_edited_{time.strftime('%Y%m%d_%H%M%S')}"
    dst_it = edited / "point_cloud" / it_dir.name
    dst_it.mkdir(parents=True, exist_ok=True)
    (edited / "cfg_args").symlink_to((model / "cfg_args").resolve())
    light = it_dir / "lighting.pth"               # only present if trained with --material
    if light.is_file():
        (dst_it / "lighting.pth").symlink_to(light.resolve())
    return edited, dst_it / "point_cloud.ply"


def _prepare_edited_model(model: Path, upload) -> Path:
    """Build the sibling edited model dir from an uploaded file (Mesh form)."""
    edited, dest = _make_edited_model(model)
    with open(dest, "wb") as out:
        shutil.copyfileobj(upload.file, out)
    return edited


@app.get("/viz/{job_id}", response_class=HTMLResponse)
async def viz(request: Request, job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return _page(request, "viz.html", job=job.to_dict(), has_model=bool(_model_dir(job)))


@app.get("/viz/mesh/{job_id}", response_class=HTMLResponse)
async def mesh_viz(request: Request, job_id: str):
    """In-browser mesh viewer (orbit/zoom + mm ruler). Lets the user switch
    between the raw (recon-unit) mesh and the marker-scaled (mm) mesh."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job.kind != "mesh":
        raise HTTPException(404, "not a mesh job")
    meta = job.meta or {}
    return _page(request, "mesh_viz.html", job=job.to_dict(),
                 has_scaled=bool(meta.get("mesh_scaled_path")),
                 mm_per_unit=meta.get("mm_per_unit"))


@app.get("/api/jobs/{job_id}/scene.json")
async def scene_json(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    md = _model_dir(job)
    if not md:
        raise HTTPException(404, "no sparse model for this job")
    from pipeline.model import scene
    return JSONResponse(await asyncio.to_thread(scene, md))


@app.get("/api/jobs/{job_id}/image/{idx}")
async def image_detail_ep(job_id: str, idx: int):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    md = _model_dir(job)
    if not md:
        raise HTTPException(404, "no sparse model for this job")
    from pipeline.model import image_detail
    return JSONResponse(await asyncio.to_thread(image_detail, md, idx))


@app.get("/api/jobs/{job_id}/imagefile/{idx}")
async def image_file(job_id: str, idx: int):
    """Serve the source photo for image #idx (checked across the undistorted output,
    the original image_root, and the NESTED staging dir)."""
    job = manager.get(job_id)
    md = _model_dir(job) if job else None
    if not md:
        raise HTTPException(404, "no sparse model for this job")
    from pipeline.model import read_images
    imgs = await asyncio.to_thread(read_images, md / "images.bin")
    if idx < 0 or idx >= len(imgs):
        raise HTTPException(404, "bad index")
    name = imgs[idx]["name"]
    ws = Path(job.meta.get("workspace", ""))
    p = job.params or {}
    dense = ws / f"{p.get('dataset_name', 'training_dataset')}_{p.get('mapper', 'global')}_mapper"
    for cand in (dense / "images" / name, Path(job.meta.get("image_root", "")) / name, ws / "staging" / name):
        if cand.is_file():
            return FileResponse(cand)
    raise HTTPException(404, f"image file not found: {name}")


@app.get("/api/jobs/{job_id}/mesh.ply")
async def mesh_ply(job_id: str):
    """Serve the extracted mesh (tsdf_post.ply) of a mesh job for download."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job.kind != "mesh":
        raise HTTPException(404, "not a mesh job")
    # Prefer the path the parser captured; else glob the model output dir.
    p = (job.meta or {}).get("mesh_path")
    path = Path(p) if p else None
    if not (path and path.is_file()):
        model = Path((job.meta or {}).get("model_path", ""))
        cands = sorted(model.glob("*/*/mesh/tsdf_post.ply")) if str(model) else []
        path = cands[-1] if cands else None
    if not (path and path.is_file()):
        raise HTTPException(404, "mesh not found (尚未產出或已被移動)")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@app.get("/api/jobs/{job_id}/mesh_scaled.ply")
async def mesh_scaled_ply(job_id: str):
    """Serve the marker-scaled mesh (tsdf_post_scaled_mm.ply, in millimetres)."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job.kind != "mesh":
        raise HTTPException(404, "not a mesh job")
    p = (job.meta or {}).get("mesh_scaled_path")
    path = Path(p) if p else None
    if not (path and path.is_file()):
        model = Path((job.meta or {}).get("model_path", ""))
        cands = sorted(model.glob("*/*/mesh/tsdf_post_scaled_mm.ply")) if str(model) else []
        path = cands[-1] if cands else None
    if not (path and path.is_file()):
        raise HTTPException(404, "scaled mesh not found (未啟用 marker 或縮放失敗)")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@app.get("/api/jobs/{job_id}/model.ply")
async def model_ply(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    md = _model_dir(job)
    if not md:
        raise HTTPException(404, "no sparse model for this job")
    from pipeline.model import ensure_ply
    out = Path(job.meta["workspace"]) / "sparse" / "points.ply"
    await asyncio.to_thread(ensure_ply, md, out)
    return FileResponse(out, media_type="application/octet-stream", filename="points.ply")


@app.get("/api/jobs/{job_id}/gaussians.ply")
async def gaussians_ply(job_id: str):
    """Serve the trained 3DGS cloud (point_cloud/iteration_*/point_cloud.ply) so it
    can be opened in SuperSplat for background removal. Unlike /model.ply (the sparse
    COLMAP cloud), this is the full Gaussian model the mesh stage renders from."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    md = _trained_model_dir(job)
    if not md:
        raise HTTPException(404, "此 job 沒有訓練輸出（缺 model_path / cfg_args）。")
    ply = _trained_ply(md)
    if not ply:
        raise HTTPException(404, "找不到 point_cloud.ply（訓練可能尚未完成）。")
    return FileResponse(ply, media_type="application/octet-stream", filename="point_cloud.ply")


@app.post("/api/jobs/{job_id}/edited_ply")
async def edited_ply_upload(job_id: str, request: Request):
    """Receive a SuperSplat-edited (去背) .ply (raw body) from the embedded editor.

    Mesh backends (GS-2M): stage a non-destructive `_edited_<time>` sibling model
    dir and return its path → the UI prefills + re-meshes from the clean cloud.
    No-mesh backends (LichtFeld): save the cleaned cloud under `<model>/edited/`
    and return its path → the UI offers a download. The original is never touched."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    md = _trained_model_dir(job)
    if not md:
        raise HTTPException(404, "此 job 沒有訓練輸出（找不到 point_cloud / splat ply）。")
    spec = get_backend((job.meta or {}).get("backend", "") or "")
    can_mesh = bool(spec and spec.get("mesh_args"))

    async def _save(dest: Path) -> None:          # stream to disk (the cloud may be large)
        with open(dest, "wb") as out:
            async for chunk in request.stream():
                out.write(chunk)

    if can_mesh:
        edited, dest = await asyncio.to_thread(_make_edited_model, md)
        await _save(dest)
        return JSONResponse({"ok": True, "mode": "mesh", "model_path": str(edited)})
    # no-mesh backend (e.g. LichtFeld): keep the cleaned cloud for download, no re-mesh dir
    ed = md / "edited"
    ed.mkdir(parents=True, exist_ok=True)
    dest = ed / f"cleaned_{time.strftime('%Y%m%d_%H%M%S')}.ply"
    await _save(dest)
    return JSONResponse({"ok": True, "mode": "download", "saved": str(dest)})


@app.get("/api/jobs/{job_id}/logs")
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
