"""Auto-measure a mesh's height/width/depth and render front/side/top previews
in-browser, either bound to a mesh job's own output (/viz/mesh/{job_id}/measure)
or standalone against ANY server-side mesh file (/measure?path=..., mirroring
how /viewer opens any mesh outside the job system).

Orientation is ambiguous for an unaligned reconstruction, so the page exposes
axis/flip toggles (each a plain link that re-renders with one param flipped)
instead of assuming the auto-detected axis is correct.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote, urlencode

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from jobs import manager
from pipeline.mesh_measure import measure_mesh
from web.routers.viewer import _safe_mesh_file
from web.shared import _page

router = APIRouter()


def _resolve_job_mesh_path(job) -> Path | None:
    p = (job.meta or {}).get("mesh_path")
    path = Path(p) if p else None
    if not (path and path.is_file()):
        model = Path((job.meta or {}).get("model_path", ""))
        cands = sorted(model.glob("*/*/mesh/tsdf_post.ply")) if str(model) else []
        path = cands[-1] if cands else None
    return path if (path and path.is_file()) else None


def _key(mode: str, axis: int, swap_wd: bool, flip_h: bool, flip_w: bool, flip_d: bool) -> str:
    return f"{mode}-{axis}-{int(swap_wd)}-{int(flip_h)}-{int(flip_w)}-{int(flip_d)}"


def _parse_mm_per_unit(raw: str | float | None) -> tuple[float | None, str | None]:
    """Lenient parse for the mm-per-unit form field: returns (value, error_message).
    Accepts a bare number ("20.26") or "1"/"mm" to mean the mesh is already in mm."""
    if raw in (None, ""):
        return None, None
    if isinstance(raw, (int, float)):
        return float(raw), None
    text = str(raw).strip().lower()
    if text in ("mm", "1mm"):
        return 1.0, None
    try:
        return float(text), None
    except ValueError:
        return None, f"「{raw}」不是有效的數字。要嘛留空（顯示重建單位），要嘛填一個倍率（例如 20.26），" \
                      f"或者這個 mesh 已經是 mm 單位的話就填 1。"


def _render_measure_page(request: Request, mesh_path: Path, *, mode: str, axis: int,
                          swap_wd: bool, flip_h: bool, flip_w: bool, flip_d: bool,
                          mm_per_unit_raw: str | float | None, base_url: str, base_params: dict,
                          img_url_fn, back_href: str, back_label: str, title_label: str) -> HTMLResponse:
    mode = mode if mode in ("pca", "raw") else "pca"
    axis = axis % 3
    mm_per_unit, mm_error = _parse_mm_per_unit(mm_per_unit_raw)
    key = _key(mode, axis, swap_wd, flip_h, flip_w, flip_d)
    out_dir = mesh_path.parent / "measure" / key
    try:
        result = measure_mesh(
            mesh_path, out_dir, mode=mode, height_axis=axis, swap_wd=swap_wd,
            flip_h=flip_h, flip_w=flip_w, flip_d=flip_d, mm_per_unit=mm_per_unit,
        )
    except ValueError as e:
        raise HTTPException(400, f"無法量測這個 mesh：{e}")

    def toggled(**over):
        p = dict(base_params, mode=mode, axis=axis, swap_wd=swap_wd,
                  flip_h=flip_h, flip_w=flip_w, flip_d=flip_d)
        if mm_per_unit:
            p["mm_per_unit"] = mm_per_unit
        p.update(over)
        qs = urlencode({k: (str(v).lower() if isinstance(v, bool) else v) for k, v in p.items()})
        return f"{base_url}?{qs}"

    return _page(
        request, "mesh_measure.html", title_label=title_label, back_href=back_href,
        back_label=back_label, result=result, key=key, base_url=base_url, mm_error=mm_error,
        mm_per_unit_raw=("" if mm_per_unit_raw is None else mm_per_unit_raw),
        base_params={k: v for k, v in base_params.items()},
        mode=mode, axis=axis, swap_wd=swap_wd, flip_h=flip_h, flip_w=flip_w, flip_d=flip_d,
        img_front=img_url_fn(key, "front"), img_side=img_url_fn(key, "side"), img_top=img_url_fn(key, "top"),
        url_mode_pca=toggled(mode="pca"), url_mode_raw=toggled(mode="raw"),
        url_axis0=toggled(axis=0), url_axis1=toggled(axis=1), url_axis2=toggled(axis=2),
        url_swap_wd=toggled(swap_wd=not swap_wd), url_flip_h=toggled(flip_h=not flip_h),
        url_flip_w=toggled(flip_w=not flip_w), url_flip_d=toggled(flip_d=not flip_d),
        url_reset=base_url,
    )


# --- job-bound: measure a mesh job's own output ----------------------------- #

@router.get("/viz/mesh/{job_id}/measure", response_class=HTMLResponse)
async def measure_job_page(request: Request, job_id: str, mode: str = "pca", axis: int = 0,
                            swap_wd: bool = False, flip_h: bool = False, flip_w: bool = False,
                            flip_d: bool = False, mm_per_unit: str = ""):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job.kind != "mesh":
        raise HTTPException(404, "not a mesh job")
    mesh_path = _resolve_job_mesh_path(job)
    if not mesh_path:
        raise HTTPException(404, "mesh not found (尚未產出或已被移動)")

    return _render_measure_page(
        request, mesh_path, mode=mode, axis=axis, swap_wd=swap_wd, flip_h=flip_h,
        flip_w=flip_w, flip_d=flip_d,
        mm_per_unit_raw=mm_per_unit or (job.meta or {}).get("mm_per_unit"),
        base_url=f"/viz/mesh/{job_id}/measure", base_params={},
        img_url_fn=lambda key, view: f"/api/jobs/{job_id}/measure/{key}/{view}.png",
        back_href=f"/viz/mesh/{job_id}", back_label="← 回 3D 檢視", title_label=job.title,
    )


@router.get("/api/jobs/{job_id}/measure/{key}/{view}.png")
async def measure_job_image(job_id: str, key: str, view: str):
    if view not in ("front", "side", "top"):
        raise HTTPException(404, "unknown view")
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    mesh_path = _resolve_job_mesh_path(job)
    if not mesh_path:
        raise HTTPException(404, "mesh not found")
    img_path = mesh_path.parent / "measure" / key / f"{view}.png"
    if not img_path.is_file():
        raise HTTPException(404, "尚未產生量測圖，請先開啟量測頁面")
    return FileResponse(img_path, media_type="image/png")


# --- standalone: measure ANY server-side mesh file (not tied to a job) ------ #

@router.get("/measure", response_class=HTMLResponse)
async def measure_standalone_page(request: Request, path: str = "", mode: str = "pca", axis: int = 0,
                                   swap_wd: bool = False, flip_h: bool = False, flip_w: bool = False,
                                   flip_d: bool = False, mm_per_unit: str = ""):
    if not path:
        return _page(request, "mesh_measure.html", need_path=True)
    mesh_path = _safe_mesh_file(path, exts={".ply"})  # measure_mesh only parses binary PLY

    return _render_measure_page(
        request, mesh_path, mode=mode, axis=axis, swap_wd=swap_wd, flip_h=flip_h,
        flip_w=flip_w, flip_d=flip_d, mm_per_unit_raw=mm_per_unit,
        base_url="/measure", base_params={"path": path},
        img_url_fn=lambda key, view: f"/api/measure/image?path={quote(str(mesh_path))}&key={key}&view={view}",
        back_href="javascript:history.back()", back_label="← 返回", title_label=mesh_path.name,
    )


@router.get("/api/measure/image")
async def measure_standalone_image(path: str, key: str, view: str):
    if view not in ("front", "side", "top"):
        raise HTTPException(404, "unknown view")
    mesh_path = _safe_mesh_file(path, exts={".ply"})
    img_path = mesh_path.parent / "measure" / key / f"{view}.png"
    if not img_path.is_file():
        raise HTTPException(404, "尚未產生量測圖，請先開啟量測頁面")
    return FileResponse(img_path, media_type="image/png")
