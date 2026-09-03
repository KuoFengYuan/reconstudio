"""MoGe-3 engine for the depth/normal stage — the alternative to `depth.py`.

`depth.py` shells out to LichtFeld-Studio's own `preprocess`, which is locked to
MoGe-2: it links no ONNX runtime, implements that one ViT-B/14 graph by hand in
C++/CUDA, and its `--model` flag swaps that graph's weights rather than its
architecture. MoGe-3 (sparse-3D-conv refiner) cannot be loaded there at all, so
this module runs the official PyTorch MoGe-3 in its own conda env instead and
writes the very same `depth/` + `normals/` folders.

Both engines are therefore interchangeable from the panel's point of view, and
from training's: `--use-depth-loss` / `--use-normal-loss` scan those folders and
neither knows nor cares which model produced them. The PNG encoding they must
agree on lives in `moge3_encode.py` (and is pinned by tests/test_moge3_encode.py).

Non-destructive, like depth.py: only CREATES `depth/`/`normals/` next to
`images/`; source images are never touched.
"""
from __future__ import annotations

from pathlib import Path

from .backends import BASE, env_python
from .depth import _outputs_complete, resolve_dataset
from .runner import PipelineError, Runner

# Its own env, not a training env: MoGe pins its own torch, and the panel's env
# has no torch at all. Override with {"python": "..."} semantics via env_python.
_MOGE3_SPEC = {"conda_env": "moge3"}
_SCRIPT = BASE / "tools" / "moge3_preprocess.py"

# vitl (370M) over vitg (1.25B): the large model is the practical default for
# whole-dataset runs, and both carry the normals head this stage needs.
DEFAULT_MODEL = "Ruicheng/moge-3-vitl"

MOGE3_DEFAULTS = {
    "moge3_model": DEFAULT_MODEL,
}


def moge3_python() -> Path | None:
    return env_python(_MOGE3_SPEC)


def moge3_ready() -> bool:
    return bool(moge3_python()) and _SCRIPT.is_file()


def run_moge3(p: dict, r: Runner) -> None:
    py = moge3_python()
    if not py:
        raise RuntimeError(
            "找不到 moge3 conda env。請先建立:\n"
            "  conda create -y -n moge3 python=3.11\n"
            "  conda run -n moge3 pip install torch "
            "'git+https://github.com/microsoft/MoGe.git'\n"
            "(MoGe-3 需要自己的 env:它會固定自己的 torch 版本,而面板本身的 env 沒有 torch。)")
    if not _SCRIPT.is_file():
        raise RuntimeError(f"找不到 MoGe-3 腳本: {_SCRIPT}")

    src = Path(p["images"]).expanduser()
    if not src.is_dir():
        raise FileNotFoundError(f"images 不是資料夾: {src}")
    dataset_root, images_folder = resolve_dataset(src)

    mode = (p.get("mode") or "both").strip()
    if mode not in ("depth", "normal", "both"):
        raise ValueError(f"mode 必須是 depth/normal/both,收到: {mode!r}")

    argv = [str(py), "-u", str(_SCRIPT), str(dataset_root),
            "--images", images_folder, "--mode", mode,
            "--model", (p.get("moge3_model") or DEFAULT_MODEL).strip()]
    bit_depth = str(p.get("bit_depth") or "").strip()
    if bit_depth:
        argv += ["--bit-depth", bit_depth]
    if p.get("overwrite"):
        argv.append("--overwrite")

    env = {}
    gpu = str(p.get("gpu", "")).strip()
    if gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = gpu

    outputs = []
    if mode in ("depth", "both"):
        outputs.append(str(dataset_root / "depth"))
    if mode in ("normal", "both"):
        outputs.append(str(dataset_root / "normals"))

    r.banner(f"MoGe-3 start | mode={mode} dataset={dataset_root} images={images_folder}")
    r.log(f"exec: {' '.join(argv)}")
    returncode = r.run(argv, cwd=str(dataset_root), env=env, check=False)
    if returncode != 0:
        # Unlike LichtFeld's preprocess (which has a known teardown crash that
        # depth.py forgives), this is plain Python: a non-zero code is a real
        # failure. Still say what did land, so a partial run is recoverable.
        done = _outputs_complete(dataset_root, images_folder, mode)
        raise PipelineError(
            f"MoGe-3 exited with code {returncode}"
            + ("(輸出看起來已完整,可重跑確認)" if done else ""))

    r.banner(f"MoGe-3 done. out={', '.join(outputs)}")
    hints = []
    if mode in ("depth", "both"):
        hints.append("--use-depth-loss")
    if mode in ("normal", "both"):
        hints.append("--use-normal-loss")
    r.log(f"[next] 訓練時加 {' '.join(hints)} 即可自動讀取 "
          f"{'/'.join(Path(o).name for o in outputs)}。")
