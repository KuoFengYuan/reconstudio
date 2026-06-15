"""Depth-map generation stage: run Depth Anything over an images folder and
write a LichtFeld-Studio-style ``depth/`` folder, as a subprocess inside the
depth model's own conda env.

The panel stays torch-free (env resolution is reused from ``pipeline.backends``).
This module only locates the env python, resolves the images/out folders, and
runs the panel-owned ``tools/depth_anything_infer.py`` — the actual model call
lives there (it needs torch), streaming its log through the Runner.

Non-destructive (see project convention): we only CREATE a new ``depth/`` folder;
the source images are never modified or moved. By default ``depth/`` lands next
to ``images/`` so LichtFeld's ``--use-depth-loss`` discovers it automatically (it
scans ``<dataset>/depth`` | ``<dataset>/depths`` and matches by image stem).
"""
from __future__ import annotations

from pathlib import Path

from .backends import conda_envs_dir
from .config import settings
from .runner import Runner

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
INFER_SCRIPT = "depth_anything_infer.py"

DEPTH_DEFAULTS = {
    "model": settings.depth_model,
    "conda_env": settings.depth_conda_env,
    "gpu": "0",
    "recurse": True,
    "invert": False,
    "overwrite": False,
}


def depth_env_python() -> Path | None:
    """Absolute python for the depth model's conda env, or None if not found.
    Mirrors backends.env_python but for the single depth env (DEPTH_CONDA_ENV)."""
    envs = conda_envs_dir()
    if not envs:
        return None
    py = envs / settings.depth_conda_env / "bin" / "python"
    return py if py.is_file() else None


def depth_ready() -> bool:
    """True when both the env python and the inference script are present (cheap
    path check only — torch/model availability is verified at run time)."""
    return bool(depth_env_python()) and (TOOLS_DIR / INFER_SCRIPT).is_file()


def _resolve_images(src: Path) -> Path:
    """Accept either an images folder directly, or a COLMAP workspace/dataset that
    contains an ``images/`` (so the user can point at a workspace and get depth
    next to it). Returns the folder that actually holds the photos."""
    if (src / "images").is_dir():
        return src / "images"
    return src


def run_depth(p: dict, r: Runner) -> None:
    py = depth_env_python()
    if not py:
        raise RuntimeError(
            f"找不到深度模型 conda env '{settings.depth_conda_env}' 的 python。"
            " 請建立該 env（裝 torch + transformers + pillow）並在 local.env 設 "
            "DEPTH_CONDA_ENV，或開 /doctor 檢查。")
    script = TOOLS_DIR / INFER_SCRIPT
    if not script.is_file():
        raise FileNotFoundError(f"推論腳本不存在: {script}")

    src = Path(p["images"]).expanduser()
    if not src.is_dir():
        raise FileNotFoundError(f"images 不是資料夾: {src}")
    images_dir = _resolve_images(src)

    out_dir = Path(p["out_dir"]).expanduser() if p.get("out_dir") else images_dir.parent / "depth"
    out_dir = out_dir.resolve()
    if out_dir == images_dir.resolve():
        raise ValueError("輸出資料夾不可與來源 images 相同（會覆蓋原圖）。")

    model = (p.get("model") or settings.depth_model).strip()
    if not model:
        raise ValueError("需要指定深度模型 id（model）。")

    argv = [str(py), "-u", str(script),
            "--images", str(images_dir), "--out", str(out_dir), "--model", model]
    if p.get("recurse"):
        argv.append("--recurse")
    if p.get("invert"):
        argv.append("--invert")
    if p.get("overwrite"):
        argv.append("--overwrite")

    env = {"PYTHONUNBUFFERED": "1"}
    gpu = str(p.get("gpu", "")).strip()
    if gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = gpu        # honored regardless of the queue's scheduling

    out_dir.mkdir(parents=True, exist_ok=True)
    r.banner(f"depth start | env={settings.depth_conda_env} model={model} gpu={gpu or 'default'}")
    r.log(f"infer:   {py} -u {script.name}")
    r.log(f"images:  {images_dir}")
    r.log(f"output:  {out_dir}  (LichtFeld 會掃描 <dataset>/depth 自動對應)")
    r.run(argv, cwd=str(TOOLS_DIR), env=env)
    r.banner(f"depth done. out={out_dir}")
    r.log("[next] 在 LichtFeld-Studio 訓練時加 --use-depth-loss 即可自動讀取此 depth/。")
