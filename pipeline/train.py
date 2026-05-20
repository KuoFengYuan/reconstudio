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

import shlex
from pathlib import Path

from .backends import env_python, get_backend, repo_path
from .model import read_cameras
from .runner import Runner

TRAIN_DEFAULTS = {"backend": "gs2m", "gpu": "0", "extra": "", "force": False}

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
        r.log(f"[mesh] result: {found[-1]}")
    else:
        r.log("[mesh] warning: 找不到 tsdf_post.ply,請檢查上面的 render 輸出。")
    r.banner(f"mesh done. model={out}")
