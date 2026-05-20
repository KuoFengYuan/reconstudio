"""Training stage: run a Gaussian-splatting trainer (GS-2M, …) on a COLMAP
workspace, as a subprocess inside the trainer's own conda env.

The panel stays torch-free (env resolution lives in pipeline.backends). This
module only:
  1. adapts the COLMAP output into the layout the trainer expects, and
  2. builds + runs the command, streaming its log through the Runner.

Why the adaptation matters: colmap_panel's `undistort` stage writes a *flat*
`sparse/` with a PINHOLE model + undistorted `images/`, but GS-2M reads
`sparse/0/{cameras,images,points3D}.bin`. We expose that via symlinks in a
dedicated scene dir, so the original workspace is never touched. We also refuse
a *distorted* model up front (GS-2M only accepts SIMPLE_PINHOLE/PINHOLE), which
is the single most common way this integration goes wrong.
"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from .backends import env_python, get_backend, repo_path
from .model import read_cameras
from .runner import Cancelled, Runner

TRAIN_DEFAULTS = {"backend": "gs2m", "gpu": "0", "extra": "", "force": False}

# Marker-scaling tools are panel-owned (live in this project's tools/), but run
# in the backend's env (they need cv2/open3d/plyfile, which the trainer env has).
TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
MARKER_SCRIPT, SCALE_SCRIPT = "estimate_marker_scale.py", "scale_mesh.py"

_NEEDED = ("cameras.bin", "images.bin", "points3D.bin")
_PINHOLE = {"PINHOLE", "SIMPLE_PINHOLE"}


def _resolve_dense(src: Path) -> tuple[Path, Path]:
    """Locate a PINHOLE (undistorted) COLMAP model under `src`, returning
    (sparse_model_dir, images_dir). Accepts, in priority order:
      (a) a flat dense dir          : src/sparse/cameras.bin + src/images/
      (b) a workspace with a dense  : src/<name>_mapper/sparse/... + .../images/
      (c) an already sparse/0 scene : src/sparse/0/cameras.bin + src/images/
    """
    if (src / "sparse" / "cameras.bin").is_file() and (src / "images").is_dir():
        return src / "sparse", src / "images"
    for d in sorted(src.glob("*_mapper")):
        if (d / "sparse" / "cameras.bin").is_file() and (d / "images").is_dir():
            return d / "sparse", d / "images"
    if (src / "sparse" / "0" / "cameras.bin").is_file() and (src / "images").is_dir():
        return src / "sparse" / "0", src / "images"
    raise FileNotFoundError(
        f"找不到去畸變的 COLMAP 模型（需要 sparse/cameras.bin + images/）於 {src}。"
        " 請先在 COLMAP 階段跑完 undistort，並把 source 指向 workspace 或其去畸變輸出。")


def _assert_pinhole(sparse_dir: Path, r: Runner) -> None:
    cams = read_cameras(sparse_dir / "cameras.bin")
    models = {c["model"] for c in cams.values()}
    if not models <= _PINHOLE:
        raise ValueError(
            f"相機模型為 {sorted(models)}，但 GS-2M 只接受 PINHOLE/SIMPLE_PINHOLE。"
            " 你八成指到了 mapper 的原始 sparse（含畸變），請改用 undistort 後的輸出。")
    r.log(f"camera model OK: {sorted(models)} ({len(cams)} cam)")


def _build_scene(scene: Path, sparse_dir: Path, images_dir: Path,
                 force: bool, r: Runner) -> None:
    """Materialize a GS-2M scene dir of symlinks into the COLMAP output.
    Non-destructive: only this scene dir is written (the trainer's points3D.ply
    lands in the real sparse/0 dir here, not in the source workspace)."""
    s0 = scene / "sparse" / "0"
    s0.mkdir(parents=True, exist_ok=True)
    for f in _NEEDED:
        link = s0 / f
        if link.is_symlink() or link.exists():
            if not force:
                continue
            link.unlink()
        link.symlink_to((sparse_dir / f).resolve())
    img = scene / "images"
    if force and (img.is_symlink() or img.exists()):
        img.unlink()
    if not (img.is_symlink() or img.exists()):
        img.symlink_to(images_dir.resolve())
    r.log(f"scene ready: {scene}  (sparse/0 + images → {sparse_dir.parent})")


def run_train(p: dict, r: Runner) -> None:
    name = p.get("backend") or "gs2m"
    spec = get_backend(name)
    if not spec:
        raise ValueError(f"unknown backend: {name}")

    py = env_python(spec)
    if not py:
        raise RuntimeError(
            f"backend '{name}' 的 conda env '{spec.get('conda_env')}' 找不到 python。"
            " 請在 backends.json 設定，或開 /doctor 檢查環境。")
    repo = repo_path(spec)
    script = spec.get("train_script", "train.py")
    if not (repo / script).is_file():
        raise FileNotFoundError(f"trainer 不存在: {repo / script}")

    src = Path(p["source"])
    out = Path(p["model_path"])
    if not src.is_dir():
        raise FileNotFoundError(f"source not found: {src}")

    sparse_dir, images_dir = _resolve_dense(src)
    _assert_pinhole(sparse_dir, r)
    scene = out.parent / f"{out.name}_scene"
    _build_scene(scene, sparse_dir, images_dir, bool(p.get("force")), r)

    args = (p.get("args") or "").strip()        # tunable params (assembled in app)
    extra = (p.get("extra") or "").strip()       # free escape-hatch flags
    cmd = spec.get("train_args", "-s {scene} -m {out} {args} {extra}").format(
        scene=str(scene), out=str(out), args=args, extra=extra)
    argv = [str(py), "-u", script, *shlex.split(cmd)]

    env = {"PYTHONUNBUFFERED": "1"}
    gpu = str(p.get("gpu", "")).strip()
    if gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = gpu        # honored regardless of the queue's scheduling

    out.mkdir(parents=True, exist_ok=True)
    r.banner(f"train start | backend={name} env={spec.get('conda_env')} gpu={gpu or 'default'}")
    r.log(f"trainer: {py} -u {script} {cmd}")
    r.log(f"cwd:     {repo}")
    r.log(f"output:  {out}")
    r.run(argv, cwd=str(repo), env=env)
    r.banner(f"train done. model={out}")
    if spec.get("mesh_args"):
        r.log(f"[next] 可在「Mesh」分頁對 {out} 抽 mesh（此 backend 支援）")


def _scene_from_model(out: Path) -> Path:
    """Find the COLMAP scene a model was trained on. Prefer the source_path
    recorded in the model's cfg_args (ground truth), fall back to the panel's
    `<model>_scene` convention from _build_scene."""
    cfg = out / "cfg_args"
    if cfg.is_file():
        m = re.search(r"source_path\s*=\s*'([^']*)'", cfg.read_text())
        if m and m.group(1):
            return Path(m.group(1))
    return out.parent / f"{out.name}_scene"


def _run_marker_scale(p: dict, r: Runner, py: Path, mesh_ply: Path) -> None:
    """If a ChArUco marker board was captured, estimate the recon→mm scale from it
    and write a physically-scaled copy of the mesh (in millimetres) next to it.

    The tools are panel-owned (tools/) but run with the backend's env python `py`
    (they need cv2/open3d/plyfile, which the trainer env provides)."""
    mk = p.get("marker") or {}
    out = Path(p["model_path"])
    scene = _scene_from_model(out)
    sparse, images = scene / "sparse" / "0", scene / "images"
    if not (sparse / "cameras.bin").is_file() or not images.exists():
        raise FileNotFoundError(
            f"找不到場景的 sparse/0 與 images（{scene}）。此模型可能不是用本面板訓練的,"
            " 無法定位 COLMAP 場景做標定。")
    for s in (MARKER_SCRIPT, SCALE_SCRIPT, "colmap_read_write_model.py"):
        if not (TOOLS_DIR / s).is_file():
            raise FileNotFoundError(f"找不到標定工具: {TOOLS_DIR / s}")

    env = {"PYTHONUNBUFFERED": "1"}
    scale_json = mesh_ply.parent / "marker_scale.json"
    r.banner("marker scale | 偵測標定板、估算尺度")
    est = [str(py), "-u", str(TOOLS_DIR / MARKER_SCRIPT),
           "--sparse", str(sparse), "--images", str(images),
           "--squares-x", str(mk["squares_x"]), "--squares-y", str(mk["squares_y"]),
           "--square-mm", str(mk["square_mm"]), "--marker-mm", str(mk["marker_mm"]),
           "--dict", str(mk.get("dict") or "DICT_5X5_100"),
           "--out-json", str(scale_json)]
    r.log("estimate: " + " ".join(est))
    r.run(est, cwd=str(TOOLS_DIR), env=env)

    data = json.loads(scale_json.read_text())
    mm_per_unit = float(data["mm_per_unit"])
    r.log(f"[scale] mm_per_unit={mm_per_unit:.6f}  rel_mad={data.get('rel_mad_pct', float('nan')):.3f}%"
          f"  pairs={data.get('pairs')}")

    scaled = mesh_ply.with_name(mesh_ply.stem + "_scaled_mm.ply")
    r.banner("marker scale | 套用尺度,輸出 mm 實際尺寸 mesh")
    sc = [str(py), "-u", str(TOOLS_DIR / SCALE_SCRIPT),
          "--in", str(mesh_ply), "--out", str(scaled),
          "--mm-per-unit", f"{mm_per_unit:.8f}", "--target-unit", "mm", "--force"]
    r.log("scale: " + " ".join(sc))
    r.run(sc, cwd=str(TOOLS_DIR), env=env)
    if scaled.is_file():
        r.log(f"[mesh] scaled result: {scaled}")
        r.log(f"[mesh] scale: 1 recon unit = {mm_per_unit:.4f} mm（mesh 已換算為實際 mm）")
    else:
        r.log("[mesh] warning: 縮放後的 mesh 沒產出,請檢查上面的輸出。")


def run_mesh(p: dict, r: Runner) -> None:
    """Extract a triangle mesh from a trained model. Backend-specific: only runs
    for backends that declare `mesh_args` (e.g. GS-2M's render.py --extract_mesh)."""
    name = p.get("backend") or "gs2m"
    spec = get_backend(name)
    if not spec:
        raise ValueError(f"unknown backend: {name}")
    if not spec.get("mesh_args"):
        raise ValueError(f"backend '{name}' 不支援 mesh 抽取（沒有定義 mesh_args）。")

    py = env_python(spec)
    if not py:
        raise RuntimeError(
            f"backend '{name}' 的 conda env '{spec.get('conda_env')}' 找不到 python，開 /doctor 檢查。")
    repo = repo_path(spec)
    script = spec.get("mesh_script", "render.py")
    if not (repo / script).is_file():
        raise FileNotFoundError(f"mesh 腳本不存在: {repo / script}")

    out = Path(p["model_path"])
    if not (out / "cfg_args").is_file():
        raise FileNotFoundError(
            f"{out} 不像訓練輸出（缺 cfg_args）。請先完成訓練，再對該模型目錄抽 mesh。")

    args = (p.get("args") or "").strip()
    extra = (p.get("extra") or "").strip()
    cmd = spec["mesh_args"].format(out=str(out), args=args, extra=extra)
    argv = [str(py), "-u", script, *shlex.split(cmd)]

    env = {"PYTHONUNBUFFERED": "1"}
    gpu = str(p.get("gpu", "")).strip()
    if gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = gpu

    r.banner(f"mesh start | backend={name} env={spec.get('conda_env')} gpu={gpu or 'default'}")
    r.log(f"render:  {py} -u {script} {cmd}")
    r.log(f"cwd:     {repo}")
    r.run(argv, cwd=str(repo), env=env)
    # render.py writes to <out>/<split>/<label>_<iter>/mesh/tsdf_post.ply
    found = sorted(out.glob("*/*/mesh/tsdf_post.ply"))
    if found:
        mesh_ply = found[-1]
        r.log(f"[mesh] result: {mesh_ply}")
        if (p.get("marker") or {}).get("enable"):
            try:
                _run_marker_scale(p, r, py, mesh_ply)
            except Cancelled:
                raise
            except Exception as exc:   # keep the (valid) unscaled mesh; just flag scaling
                r.banner("marker scale 失敗 — 保留未縮放的 mesh")
                r.log(f"[mesh] marker scale 失敗: {exc}")
    else:
        r.log("[mesh] warning: 找不到 tsdf_post.ply,請檢查上面的 render 輸出。")
    r.banner(f"mesh done. model={out}")
