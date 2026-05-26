"""Result visualization + downloads: sparse cloud / camera poses viewer, mesh
viewer, PLY downloads, non-destructive camera culling, and SuperSplat 去背 upload.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from jobs import manager
from pipeline import get_backend
from pipeline.model import (
    count_points,
    cull_cameras,
    ensure_ply,
    image_detail,
    read_images,
    scene,
)
from web.services.models import (
    dense_dir,
    make_edited_model,
    model_dir,
    resolve_model,
    trained_model_dir,
    trained_ply,
)
from web.shared import _page

router = APIRouter()


@router.get("/viz/{job_id}", response_class=HTMLResponse)
async def viz(request: Request, job_id: str, m: str | None = None):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    md = resolve_model(job, m)
    # when viewing a cleaned variant, surface its folder + trainable dataset path so the
    # user can see/copy where it lives (the post-cull message is transient).
    clean_dir = clean_dataset = ""
    if m and md and md != model_dir(job):
        root = md.parent.parent                         # .../cleaned/<ts>
        clean_dir = str(root)
        ds = root / "dataset"
        clean_dataset = str(ds) if (ds / "sparse" / "cameras.bin").is_file() else ""
    return _page(request, "viz.html", job=job.to_dict(), has_model=bool(md),
                 model_sub=(m or ""), clean_dir=clean_dir, clean_dataset=clean_dataset)


@router.get("/viz/mesh/{job_id}", response_class=HTMLResponse)
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


@router.get("/api/jobs/{job_id}/scene.json")
async def scene_json(job_id: str, m: str | None = None):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    md = resolve_model(job, m)
    if not md:
        raise HTTPException(404, "no sparse model for this job")
    return JSONResponse(await asyncio.to_thread(scene, md))


@router.get("/api/jobs/{job_id}/image/{idx}")
async def image_detail_ep(job_id: str, idx: int, m: str | None = None):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    md = resolve_model(job, m)
    if not md:
        raise HTTPException(404, "no sparse model for this job")
    return JSONResponse(await asyncio.to_thread(image_detail, md, idx))


@router.get("/api/jobs/{job_id}/imagefile/{idx}")
async def image_file(job_id: str, idx: int, m: str | None = None):
    """Serve the source photo for image #idx (checked across the undistorted output,
    the original image_root, and the NESTED staging dir)."""
    job = manager.get(job_id)
    md = resolve_model(job, m) if job else None
    if not md:
        raise HTTPException(404, "no sparse model for this job")
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


@router.get("/api/jobs/{job_id}/mesh.ply")
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


@router.get("/api/jobs/{job_id}/mesh_scaled.ply")
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


@router.get("/api/jobs/{job_id}/model.ply")
async def model_ply(job_id: str, m: str | None = None):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    md = resolve_model(job, m)
    if not md:
        raise HTTPException(404, "no sparse model for this job")
    out = md / "points.ply"          # next to the model (cleaned variants get their own)
    await asyncio.to_thread(ensure_ply, md, out)
    return FileResponse(out, media_type="application/octet-stream", filename="points.ply")


@router.post("/api/jobs/{job_id}/cull")
async def cull_ep(job_id: str, request: Request):
    """Remove the named (bad) cameras and write a NON-DESTRUCTIVE cleaned copy under
    <workspace>/cleaned/<timestamp>/:
        sparse/0/            pruned distorted model — reopen in the viewer to confirm
        dataset/sparse/      pruned undistorted model (what training reads)
        dataset/images  ->   symlink to the original images (never copied)
        removed.json         audit (which cameras, counts, source job)
    The originals (sparse/0, *_mapper) are only read. Works for one camera or many —
    the body is just the list of image names to drop. Returns the cleaned paths so
    the UI can reopen the result and prefill it as a Train `source`."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    md = model_dir(job)                        # distorted model the viewer shows
    if not md:
        raise HTTPException(404, "no sparse model for this job")
    body = await request.json()
    names = sorted({str(n).strip() for n in (body.get("names") or []) if str(n).strip()})
    if not names:
        raise HTTPException(400, "沒有指定要移除的相機（names 為空）。")
    # also drop the 3D points the removed cameras leave under-supported (default on)
    filter_points = bool(body.get("filter_points", True))

    # Iterative culling: cull FROM the model currently shown in the viewer (a cleaned
    # variant, passed as `from`), so repeated removals ACCUMULATE instead of each
    # starting over from the original. Empty/stale `from` -> cull the original.
    from_ = (body.get("from") or "").strip()
    src_model = (resolve_model(job, from_) if from_ else None) or md
    dense = dense_dir(job)
    src_dataset = dense / "sparse" if dense else None
    prev_removed: list[str] = []
    if src_model != md:                                # culling a previous cleaned variant
        variant_root = src_model.parent.parent         # .../cleaned/<ts>
        cand = variant_root / "dataset" / "sparse"
        if (cand / "cameras.bin").is_file():
            src_dataset = cand                          # keep pruning the variant's dataset
        rj = variant_root / "removed.json"
        if rj.is_file():
            try:
                prev_removed = json.loads(rj.read_text()).get("removed_total") \
                    or json.loads(rj.read_text()).get("removed", [])
            except (ValueError, OSError):
                prev_removed = []

    ws = Path(job.meta["workspace"])
    ts = time.strftime("%Y%m%d_%H%M%S")
    clean_root = ws / "cleaned" / ts
    img_target = (dense / "images") if dense else None  # symlink always -> ORIGINAL images

    def _build() -> dict:
        s0 = clean_root / "sparse" / "0"
        cull_cameras(src_model, s0, names, filter_points=filter_points)   # for the viewer
        kept = len(read_images(s0 / "images.bin"))
        pts = count_points(s0 / "points3D.bin")
        dataset = None
        if src_dataset is not None:                                      # for training/mesh
            ds = clean_root / "dataset"
            cull_cameras(src_dataset, ds / "sparse", names, filter_points=filter_points)
            link = ds / "images"
            if img_target and not os.path.lexists(link):
                link.symlink_to(os.path.relpath(img_target, ds))
            dataset = str(ds)
        return {"kept": kept, "points": pts, "dataset": dataset}

    pts_before = await asyncio.to_thread(count_points, src_model / "points3D.bin")
    info = await asyncio.to_thread(_build)
    removed_total = sorted(set(prev_removed) | set(names))
    audit = {"removed": names, "removed_count": len(names),
             "removed_total": removed_total, "removed_total_count": len(removed_total),
             "from": from_ or None, "kept_images": info["kept"],
             "filter_points": filter_points, "points_before": pts_before,
             "points_after": info["points"], "source_job": job_id,
             "source_model": str(src_model), "created": ts}
    (clean_root / "removed.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    rel = clean_root.relative_to(ws).as_posix()                    # e.g. cleaned/20260522_103000
    return JSONResponse({"ok": True, "cleaned_dir": str(clean_root),
                         "dataset_dir": info["dataset"], "viz_model": f"{rel}/sparse/0",
                         "removed": len(names), "removed_total": len(removed_total),
                         "kept": info["kept"], "points_before": pts_before,
                         "points_after": info["points"]})


@router.get("/api/jobs/{job_id}/gaussians.ply")
async def gaussians_ply(job_id: str):
    """Serve the trained 3DGS cloud (point_cloud/iteration_*/point_cloud.ply) so it
    can be opened in SuperSplat for background removal. Unlike /model.ply (the sparse
    COLMAP cloud), this is the full Gaussian model the mesh stage renders from."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    md = trained_model_dir(job)
    if not md:
        raise HTTPException(404, "此 job 沒有訓練輸出（缺 model_path / cfg_args）。")
    ply = trained_ply(md)
    if not ply:
        raise HTTPException(404, "找不到 point_cloud.ply（訓練可能尚未完成）。")
    return FileResponse(ply, media_type="application/octet-stream", filename="point_cloud.ply")


@router.post("/api/jobs/{job_id}/edited_ply")
async def edited_ply_upload(job_id: str, request: Request):
    """Receive a SuperSplat-edited (去背) .ply (raw body) from the embedded editor.

    Mesh backends (GS-2M): stage a non-destructive `_edited_<time>` sibling model
    dir and return its path → the UI prefills + re-meshes from the clean cloud.
    No-mesh backends (LichtFeld): save the cleaned cloud under `<model>/edited/`
    and return its path → the UI offers a download. The original is never touched."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    md = trained_model_dir(job)
    if not md:
        raise HTTPException(404, "此 job 沒有訓練輸出（找不到 point_cloud / splat ply）。")
    spec = get_backend((job.meta or {}).get("backend", "") or "")
    can_mesh = bool(spec and spec.get("mesh_args"))

    async def _save(dest: Path) -> None:          # stream to disk (the cloud may be large)
        with open(dest, "wb") as out:
            async for chunk in request.stream():
                out.write(chunk)

    if can_mesh:
        edited, dest = await asyncio.to_thread(make_edited_model, md)
        await _save(dest)
        return JSONResponse({"ok": True, "mode": "mesh", "model_path": str(edited)})
    # no-mesh backend (e.g. LichtFeld): keep the cleaned cloud for download, no re-mesh dir
    ed = md / "edited"
    ed.mkdir(parents=True, exist_ok=True)
    dest = ed / f"cleaned_{time.strftime('%Y%m%d_%H%M%S')}.ply"
    await _save(dest)
    return JSONResponse({"ok": True, "mode": "download", "saved": str(dest)})
