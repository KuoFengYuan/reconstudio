"""去背 (matting) stage: SAM-prompted foreground extraction, torch-free side.

Same shape as `moge3.py`: the panel never imports torch or a SAM package — it
resolves the separate `sam` conda env, builds an argv, and streams
`tools/sam_matte.py`'s output through the Runner. All the model work, and all
the numpy maths, lives on the other side of that subprocess boundary.

Non-destructive: only `cutout/` (RGBA) and `masks/` (0/255) are created next to
`images/`; the source photos are never modified.

Why both folders: `cutout/` is the human-facing result *and* the input COLMAP's
mask pass wants — `_stage_undistort` runs `image_undistorter` over MASKS_DIR and
then `large_scene.make_mask_uint8`, which reads the **alpha channel** and so
requires 4-channel PNGs. `masks/` is the single-channel form the trainers'
`--masks` flag expects. Producing both costs one extra PNG write per image and
removes the "which one does this consumer want" question entirely.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .backends import BASE, env_python
from .runner import PipelineError, Runner

# Its own env: every SAM package pins its own torch, and the panel's env has no
# torch at all. Override with {"python": "..."} semantics via env_python.
_SAM_SPEC = {"conda_env": "sam"}
_SCRIPT = BASE / "tools" / "sam_matte.py"

ENGINES = ("sam2", "sam3", "sam1")
# Directory names a recursive photo scan must never descend into: this tool's own
# outputs, the depth stage's, and COLMAP's — a workspace's `colmap/` subtree holds
# the undistorted copies of the very same photos, so walking it would matte every
# image two or three times over and quietly triple the run. Mirrored in
# matte_encode.SKIP_DIR_NAMES for the subprocess side; the two are pinned equal by
# tests/test_matte_encode.py.
SKIP_DIR_NAMES = frozenset({
    "cutout", "masks", "depth", "normals",
    "colmap", "sparse", "dense", "stereo", "distorted",
})
# Where the prompt boxes come from. Ordered as the form presents them.
BOX_SOURCES = ("json", "track", "text", "exemplar", "auto", "full")

MATTE_DEFAULTS = {
    "matte_engine": "sam3",
    "matte_model": "",          # blank = the script's per-engine default
    "boxes": "track",
    "text": "",
    "outputs": "cutout,masks",
    "erode": "1",
    "dilate": "0",
    "feather": "2",
    "min_area": "",
    "box_chunk": "",
    "on_empty": "opaque",
}


def resolve_matte_dataset(src: Path) -> tuple[Path, str]:
    """(dataset_root, images_folder) — outputs always land INSIDE the folder you picked.

    The depth stage uses `depth.resolve_dataset`, which for a loose photo folder
    returns its PARENT as the dataset root; that is correct there, because
    LichtFeld scans for `depth/` as a sibling of `images/`. Nothing downstream
    requires it here — COLMAP's MASKS_DIR and the trainers' `--masks` are both
    explicit paths — and the parent of a photo folder is very often a folder of
    *other* datasets, where two runs would share one `cutout/` and silently
    overwrite each other by filename. So: `<src>/images` when that exists (the
    COLMAP workspace layout), otherwise `<src>` itself.
    """
    if (src / "images").is_dir():
        return src, "images"
    return src, ""


def sam_python() -> Path | None:
    return env_python(_SAM_SPEC)


def matte_ready() -> bool:
    return bool(sam_python()) and _SCRIPT.is_file()


# What each engine additionally needs on top of the always-required trio.
_ENGINE_IMPORTS = {
    "sam2": ["sam2"],
    "sam1": ["segment_anything"],
    "sam3": ["transformers"],
}
_PIP_HINT = {
    "numpy": "pip install numpy",
    "cv2": "pip install opencv-python-headless",
    "torch": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128",
    "sam2": 'pip install "git+https://github.com/facebookresearch/sam2.git"',
    "segment_anything": 'pip install "git+https://github.com/facebookresearch/segment-anything.git"',
    "transformers": "pip install -U transformers accelerate",
}


def _preflight(py: Path, engine: str, r: Runner) -> None:
    """Fail with a shopping list instead of a traceback.

    `matte_ready()` only proves the env *exists* — a half-installed `sam` env
    (created, packages still downloading) passes it, and the run then dies on
    `ModuleNotFoundError: numpy` from inside a subprocess, which reads like a bug
    in this tool rather than an unfinished install. One ~1 s import probe up
    front turns that into a line saying exactly what to pip install.
    """
    wanted = ["numpy", "cv2", "torch", *_ENGINE_IMPORTS.get(engine, [])]
    # `import importlib.util`, not `import importlib`: the submodule is not
    # imported by the parent package, so the short form raises AttributeError and
    # the probe exits non-zero — which this function reads as "cannot tell", i.e.
    # it would wave every broken env straight through.
    code = ("import importlib.util as u;"
            f"print(' '.join(m for m in {wanted!r} if u.find_spec(m) is None))")
    try:
        out = subprocess.run([str(py), "-c", code], capture_output=True, text=True,
                             timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        r.log(f"[warn] 無法檢查 sam env 內容 ({exc});直接嘗試執行。")
        return
    if out.returncode != 0:
        r.log(f"[warn] sam env 套件檢查失敗 (exit {out.returncode}): "
              f"{(out.stderr or '').strip()[:200]};直接嘗試執行。")
        return
    missing = out.stdout.split()
    if not missing:
        return
    lines = "\n".join(f"  conda run -n sam {_PIP_HINT.get(m, 'pip install ' + m)}"
                       for m in missing)
    raise RuntimeError(
        f"sam env 缺少套件: {', '.join(missing)}。請先裝完再跑:\n{lines}\n"
        "(env 還在安裝中的話,等它裝完即可 —— 這裡不是程式錯誤。)")


def _write_boxes(dataset_root: Path, boxes_json: str) -> Path:
    """Persist the picker's boxes next to the outputs, and hand back the path.

    Kept on disk rather than passed on the command line for two reasons: a
    hundred boxes would blow past the argv limit, and the file is the record of
    what a run was actually told to cut — re-running or auditing a job later
    needs it, and job.params only holds the raw string.
    """
    out = dataset_root / "matte_boxes.json"
    out.write_text(boxes_json)
    return out


def run_matte(p: dict, r: Runner) -> None:
    py = sam_python()
    if not py:
        raise RuntimeError(
            "找不到 sam conda env。請先建立:\n"
            "  conda create -y -n sam python=3.11\n"
            "  conda run -n sam pip install torch torchvision opencv-python\n"
            "  conda run -n sam pip install 'git+https://github.com/facebookresearch/sam2.git'\n"
            "  # 文字提示(SAM 3 / Grounding DINO)另外加: pip install -U transformers accelerate\n"
            "(SAM 需要自己的 env:它會固定自己的 torch 版本,而面板本身的 env 沒有 torch。)")
    if not _SCRIPT.is_file():
        raise RuntimeError(f"找不到去背腳本: {_SCRIPT}")

    src = Path(p["images"]).expanduser()
    if not src.is_dir():
        raise FileNotFoundError(f"images 不是資料夾: {src}")
    dataset_root, images_folder = resolve_matte_dataset(src)

    engine = (p.get("matte_engine") or "sam2").strip()
    if engine not in ENGINES:
        raise ValueError(f"engine 必須是 {'/'.join(ENGINES)},收到: {engine!r}")
    _preflight(py, engine, r)
    boxes = (p.get("boxes") or "track").strip()
    if boxes not in BOX_SOURCES:
        raise ValueError(f"boxes 必須是 {'/'.join(BOX_SOURCES)},收到: {boxes!r}")
    outputs = (p.get("outputs") or "cutout,masks").strip()
    if not outputs:
        raise ValueError("outputs 至少要選一種(cutout 或 masks)")

    argv = [str(py), "-u", str(_SCRIPT), str(dataset_root),
            "--images", images_folder, "--engine", engine,
            "--boxes", boxes, "--outputs", outputs]

    if boxes in ("json", "track", "exemplar"):
        raw = (p.get("boxes_json") or "").strip()
        if not raw:
            raise ValueError("這個模式需要先在照片上匡選要保留的物體(至少一個框)")
        argv += ["--boxes-json", str(_write_boxes(dataset_root, raw))]
    if boxes == "text":
        text = (p.get("text") or "").strip()
        if not text:
            raise ValueError("文字提示模式需要填入要保留的物體名稱(英文,例如 parrot)")
        argv += ["--text", text]

    # Only pass what deviates from the script's own argparse defaults: a blank
    # field means "use the tool's default", and an explicit flag would otherwise
    # freeze today's default into every saved job.
    for key, flag in (("matte_model", "--model"), ("checkpoint", "--checkpoint"),
                      ("detector", "--detector"), ("erode", "--erode"),
                      ("dilate", "--dilate"), ("feather", "--feather"),
                      ("min_area", "--min-area"), ("box_chunk", "--box-chunk"),
                      ("on_empty", "--on-empty")):
        value = str(p.get(key) or "").strip()
        if value:
            argv += [flag, value]
    for key, flag in (("row_filter", "--row-filter"), ("largest_only", "--largest-only"),
                      ("soft_masks", "--soft-masks"), ("no_bleed", "--no-bleed"),
                      ("overwrite", "--overwrite")):
        if p.get(key):
            argv.append(flag)

    env = {}
    gpu = str(p.get("gpu", "")).strip()
    if gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = gpu

    dests = [str(dataset_root / f) for f in outputs.split(",") if f.strip()]
    r.banner(f"去背 start | engine={engine} boxes={boxes} dataset={dataset_root} "
             f"images={images_folder}")
    r.log(f"exec: {' '.join(argv)}")
    returncode = r.run(argv, cwd=str(dataset_root), env=env, check=False)
    if returncode != 0:
        raise PipelineError(f"sam_matte exited with code {returncode}")

    r.banner(f"去背 done. out={', '.join(dests)}")
    if "cutout" in outputs:
        r.log(f"[next] COLMAP 的 MASKS_DIR 填 {dataset_root / 'cutout'} "
              f"(遮罩階段讀的是 alpha 通道,所以要 RGBA 的 cutout/,不是 masks/)。")
    if "masks" in outputs:
        r.log(f"[next] 訓練時 --masks {dataset_root / 'masks'} 可只擬合前景物件。")
