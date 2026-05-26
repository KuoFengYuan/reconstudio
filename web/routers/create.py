"""Job-creation forms (htmx): GCS download, frames, COLMAP, train, mesh.

Each validates the form, submits a Job to the manager, and returns the
_jobview fragment (or _error.html on a ValueError).
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from jobs import COLMAP_STAGES, Job, manager, new_id
from pipeline import build_cli, default_dest, get_backend
from web.services.forms import build_colmap_params, parse_marker, scene_label
from web.services.models import prepare_edited_model
from web.shared import BROWSE_ROOT, DEST_ROOT, FFMPEG_BIN, GSUTIL_BIN, _page

router = APIRouter()


@router.post("/ui/gcs_sync", response_class=HTMLResponse)
async def create_gcs_sync(request: Request):
    form = dict(await request.form())
    src = (form.get("src") or "").strip()
    dest = (form.get("dest") or "").strip().rstrip("/")
    try:
        if not src.startswith("gs://"):
            raise ValueError("來源必須是 gs:// 路徑")
        if not dest:
            dest = default_dest(src, str(BROWSE_ROOT))   # convenience default only
        destp = Path(dest).expanduser()
        if not destp.is_absolute():
            raise ValueError(f"下載目的需為絕對路徑:{dest}")
        destp = destp.resolve()
        if destp.exists() and not destp.is_dir():
            raise ValueError(f"下載目的已存在且不是資料夾:{destp}")
        # Free to land anywhere under DEST_ROOT (default '/', i.e. unrestricted).
        if destp != DEST_ROOT and DEST_ROOT not in destp.parents:
            raise ValueError(f"下載目的需在 {DEST_ROOT} 底下:{destp}")
        params = {"src": src, "dest": str(destp), "gsutil_bin": GSUTIL_BIN,
                  "delete": bool(form.get("delete"))}
    except ValueError as exc:
        return _page(request, "_error.html", message=str(exc))

    job = Job(id=new_id(), kind="gcs",
              title=f"{src.rstrip('/').rsplit('/', 1)[-1]} ⬇",
              subtitle=f"{src}  →  {destp}",
              params=params, meta={"src": src, "dest": str(destp)})
    manager.submit(job)
    return _page(request, "_jobview.html", job=job.to_dict(), stages=COLMAP_STAGES)


@router.post("/ui/frames", response_class=HTMLResponse)
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
            raise ValueError("fps must be a number") from None
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


@router.post("/ui/jobs", response_class=HTMLResponse)
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
        params = build_colmap_params(image_root, workspace, sub, list(stages), form)
    except ValueError as exc:
        return _page(request, "_error.html", message=str(exc))

    job = Job(id=new_id(), kind="colmap",
              title=f"{Path(image_root).name} → {Path(workspace).name}",
              subtitle=f"{image_root}  →  {workspace}",
              params=params, mirror=str(Path(workspace) / "pipeline.log"),
              meta={"image_root": image_root, "workspace": workspace, "subfolders": sub})
    manager.submit(job)
    return _page(request, "_jobview.html", job=job.to_dict(), stages=COLMAP_STAGES)


@router.post("/ui/train", response_class=HTMLResponse)
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
              title=f"{backend} · {scene_label(out)}",
              subtitle=f"{source}  →  {out}  (gpu={gpu or 'default'})",
              params=params, mirror=str(Path(out) / "train.log"),
              meta={"source": source, "model_path": out, "backend": backend,
                    "total": total, "can_mesh": bool(spec.get("mesh_args"))})
    manager.submit(job)
    return _page(request, "_jobview.html", job=job.to_dict(), stages=COLMAP_STAGES)


@router.post("/ui/mesh", response_class=HTMLResponse)
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
            edited_dir = await asyncio.to_thread(prepare_edited_model, Path(model), upload)
            edited_from, model = model, str(edited_dir)
        cli = build_cli(spec.get("mesh_params", []), form)
        params = {"backend": backend, "model_path": model, "gpu": gpu,
                  "args": cli, "extra": extra}
        marker = parse_marker(form, spec)   # optional ChArUco scaling -> mm mesh
        if marker:
            params["marker"] = marker
    except ValueError as exc:
        return _page(request, "_error.html", message=str(exc))

    meta = {"model_path": model, "backend": backend, "marker": bool(marker)}
    if edited_from:
        meta["edited_from"] = edited_from
    job = Job(id=new_id(), kind="mesh",
              title=f"mesh · {scene_label(model)}" + (" · 去背" if edited_from else "")
                    + (" · scaled" if marker else ""),
              subtitle=f"{model}  (gpu={gpu or 'default'})",
              params=params, mirror=str(Path(model) / "mesh.log"),
              meta=meta)
    manager.submit(job)
    return _page(request, "_jobview.html", job=job.to_dict(), stages=COLMAP_STAGES)
