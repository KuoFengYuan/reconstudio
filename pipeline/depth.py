"""Depth/normal-map generation stage: run LichtFeld-Studio's own `preprocess`
subcommand (MoGe-2 ONNX, self-downloading, no conda env / torch needed) over a
dataset and write its `depth/` and/or `normals/` folders.

The panel spawns the same compiled binary the LichtFeld training backends use
(see `pipeline.backends.binary_exec`) — no separate depth conda env.

Non-destructive (see project convention): this only CREATES `depth/`/`normals/`
next to `images/`; the source images are never modified or moved. Those are
exactly the folder names LichtFeld's `--use-depth-loss` / `--use-normal-loss`
scan for automatically.
"""
from __future__ import annotations

from pathlib import Path

from .backends import BUILTIN_BACKENDS, binary_exec
from .runner import PipelineError, Runner

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# Never treat files under these as source images: they're this tool's own
# output folders, and in a flat dataset layout (no images/ subfolder) they sit
# directly under the same dataset_root a recursive scan would walk.
_OUTPUT_DIR_NAMES = {"depth", "normals"}

DEPTH_DEFAULTS = {
    "mode": "both",       # depth, normal, both
    "model": "",          # blank = auto-download the default MoGe-2 ONNX model
    "max_side": "",       # blank = LichtFeld default (518)
    "bit_depth": "",      # blank = LichtFeld default (16)
    "cpu": False,
    "overwrite": False,
}

# Reuse the exact same binary path the LichtFeld training backends resolve
# (../LichtFeld-Studio/build/LichtFeld-Studio), so there's one place that knows
# where the compiled app lives.
_LICHTFELD_SPEC = {"exec": BUILTIN_BACKENDS["lichtfeld-mrnf"]["exec"]}


def depth_binary_exec() -> Path | None:
    return binary_exec(_LICHTFELD_SPEC)


def depth_ready() -> bool:
    return bool(depth_binary_exec())


def resolve_dataset(src: Path) -> tuple[Path, str]:
    """Accept either a dataset workspace (containing an `images/` folder) or the
    images folder itself, and return (dataset_root, images_folder_name) for
    LichtFeld's `preprocess <dataset> --images <folder>`."""
    if (src / "images").is_dir():
        return src, "images"
    return src.parent, src.name


def _outputs_complete(dataset_root: Path, images_folder: str, mode: str) -> bool:
    """True if every source image already has its expected depth/normals PNG
    (same relative path + stem, per LichtFeld's own output_path_for). Used to
    tell a genuine failure apart from the known LichtFeld-Studio issue where
    `preprocess` crashes (SIGSEGV/SIGABRT) in its post-processing depth-anchor
    cache step — AFTER all depth/normal PNGs are already written correctly."""
    images_dir = dataset_root / images_folder if images_folder else dataset_root
    if not images_dir.is_dir():
        return False

    src_images = [
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        and p.relative_to(images_dir).parts[0] not in _OUTPUT_DIR_NAMES
    ]
    if not src_images:
        return False

    need_depth = mode in ("depth", "both")
    need_normal = mode in ("normal", "both")
    for img in src_images:
        rel = img.relative_to(images_dir).with_suffix(".png")
        if need_depth and not (dataset_root / "depth" / rel).is_file():
            return False
        if need_normal and not (dataset_root / "normals" / rel).is_file():
            return False
    return True


def run_depth(p: dict, r: Runner) -> None:
    exe = depth_binary_exec()
    if not exe:
        raise RuntimeError(
            "找不到 LichtFeld-Studio binary。請先在 ../LichtFeld-Studio 建置"
            "（見 lichtfeld-mrnf 訓練 backend 的同一份 build 說明）。")

    src = Path(p["images"]).expanduser()
    if not src.is_dir():
        raise FileNotFoundError(f"images 不是資料夾: {src}")
    dataset_root, images_folder = resolve_dataset(src)

    mode = (p.get("mode") or "both").strip()
    if mode not in ("depth", "normal", "both"):
        raise ValueError(f"mode 必須是 depth/normal/both,收到: {mode!r}")

    argv = [str(exe), "preprocess", str(dataset_root), "--images", images_folder,
            "--mode", mode]
    model = (p.get("model") or "").strip()
    if model:
        argv += ["--model", model]
    max_side = str(p.get("max_side") or "").strip()
    if max_side:
        argv += ["--max-side", max_side]
    bit_depth = str(p.get("bit_depth") or "").strip()
    if bit_depth:
        argv += ["--bit-depth", bit_depth]
    if p.get("cpu"):
        argv.append("--cpu")
    if p.get("overwrite"):
        argv.append("--overwrite")

    env = {}
    gpu = str(p.get("gpu", "")).strip()
    if gpu != "" and not p.get("cpu"):
        env["CUDA_VISIBLE_DEVICES"] = gpu

    outputs = []
    if mode in ("depth", "both"):
        outputs.append(str(dataset_root / "depth"))
    if mode in ("normal", "both"):
        outputs.append(str(dataset_root / "normals"))

    r.banner(f"preprocess start | mode={mode} dataset={dataset_root} images={images_folder}")
    r.log(f"exec: {' '.join(argv)}")
    returncode = r.run(argv, cwd=str(dataset_root), env=env, check=False)

    if returncode != 0:
        # Known LichtFeld-Studio issue (as of the 2026-07-11 "Better depth" PR):
        # `preprocess` can crash (killed by signal, e.g. SIGSEGV/-11) in its
        # post-processing depth-anchor cache step, AFTER every depth/normal PNG
        # has already been written correctly — the crash is upstream's, in
        # PinnedMemoryAllocator's static-destruction teardown, not in the actual
        # depth/normal generation. Verify on disk rather than trust the exit code:
        # only swallow this if it's a signal kill (negative returncode) AND every
        # expected output file genuinely exists; a real failure still raises.
        if returncode < 0 and _outputs_complete(dataset_root, images_folder, mode):
            r.log(f"[warn] LichtFeld preprocess 在收尾階段(深度錨點快取)當機(exit {returncode}),"
                  f"但深度/法向量圖已在當機前全部正確寫出(已逐檔核對)。這是上游最近合併的功能已知的"
                  f"收尾 bug,不影響輸出結果——訓練時只是會在啟動時即時擬合深度錨點,而不是讀取快取。"
                  f"視為此工作成功。")
        else:
            raise PipelineError(f"{argv[0]} preprocess exited with code {returncode}")

    r.banner(f"preprocess done. out={', '.join(outputs)}")
    hints = []
    if mode in ("depth", "both"):
        hints.append("--use-depth-loss")
    if mode in ("normal", "both"):
        hints.append("--use-normal-loss")
    r.log(f"[next] 訓練時加 {' '.join(hints)} 即可自動讀取 {'/'.join(Path(o).name for o in outputs)}。")
