"""Training stage: run a Gaussian-splatting trainer (GS-2M, …) on a COLMAP
workspace, as a subprocess inside the trainer's own conda env.

The panel stays torch-free (env resolution lives in pipeline.backends). This
module only:
  1. adapts the COLMAP output into the layout the trainer expects, and
  2. builds + runs the command, streaming its log through the Runner.

Why the adaptation matters: Recon Studio's `undistort` stage writes a *flat*
`sparse/` with a PINHOLE model + undistorted `images/`, but GS-2M reads
`sparse/0/{cameras,images,points3D}.bin`. We expose that via symlinks in a
dedicated scene dir, so the original workspace is never touched. We also refuse
a *distorted* model up front (GS-2M only accepts SIMPLE_PINHOLE/PINHOLE), which
is the single most common way this integration goes wrong.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .backends import binary_exec, env_python, get_backend, repo_path, texrecon_bin
from .model import read_cameras, read_images
from .runner import Cancelled, PipelineError, Runner

TRAIN_DEFAULTS = {"backend": "lichtfeld-mrnf", "gpu": "0", "extra": "", "force": False}

# Panel repo root (configs/ live here; resolved for backends that ship a config).
PANEL_BASE = Path(__file__).resolve().parent.parent

# Marker-scaling tools are panel-owned (live in this project's tools/), but run
# in the backend's env (they need cv2/open3d/plyfile, which the trainer env has).
TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
MARKER_SCRIPT, SCALE_SCRIPT = "estimate_marker_scale.py", "scale_mesh.py"
GLB_SCRIPT = "obj_to_glb.py"

_MODEL_STEMS = ("cameras", "images", "points3D")
_PINHOLE = {"PINHOLE", "SIMPLE_PINHOLE"}


def _model_ext(d: Path) -> str | None:
    """The COLMAP model format present in dir `d`: '.bin' (preferred) or '.txt',
    or None if neither. Both trainers' COLMAP loaders (LichtFeld, GS-2M) read text
    and binary natively, so the panel accepts whichever the source ships."""
    if (d / "cameras.bin").is_file():
        return ".bin"
    if (d / "cameras.txt").is_file():
        return ".txt"
    return None


def _resolve_dense(src: Path) -> tuple[Path, Path]:
    """Locate a PINHOLE (undistorted) COLMAP model under `src`, returning
    (sparse_model_dir, images_dir). Accepts, in priority order:
      (a) a flat dense dir          : src/sparse/cameras.bin + src/images/
      (b) a workspace with a dense  : src/<name>_mapper/sparse/... + .../images/
      (c) an already sparse/0 scene : src/sparse/0/cameras.bin + src/images/
    """
    if _model_ext(src / "sparse") and (src / "images").is_dir():
        return src / "sparse", src / "images"
    for d in sorted(src.glob("*_mapper")):
        if _model_ext(d / "sparse") and (d / "images").is_dir():
            return d / "sparse", d / "images"
    if _model_ext(src / "sparse" / "0") and (src / "images").is_dir():
        return src / "sparse" / "0", src / "images"
    raise FileNotFoundError(
        f"找不到去畸變的 COLMAP 模型（需要 sparse/cameras.bin 或 cameras.txt + images/）於 {src}。"
        " 請先在 COLMAP 階段跑完 undistort，並把 source 指向 workspace 或其去畸變輸出。")


def _camera_models_text(path: Path) -> tuple[set[str], int]:
    """(set of camera-model names, camera count) from a COLMAP cameras.txt —
    one camera per non-comment line: `CAM_ID MODEL W H PARAMS...`. Hand-parsed so
    pipeline stays numpy-free (the vendored read_write_model imports numpy)."""
    models, n = set(), 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                models.add(parts[1])
                n += 1
    return models, n


def _assert_pinhole(sparse_dir: Path, r: Runner) -> None:
    if _model_ext(sparse_dir) == ".txt":
        models, ncam = _camera_models_text(sparse_dir / "cameras.txt")
    else:
        cams = read_cameras(sparse_dir / "cameras.bin")
        models, ncam = {c["model"] for c in cams.values()}, len(cams)
    if not models <= _PINHOLE:
        raise ValueError(
            f"相機模型為 {sorted(models)}，但 GS-2M 只接受 PINHOLE/SIMPLE_PINHOLE。"
            " 你八成指到了 mapper 的原始 sparse（含畸變），請改用 undistort 後的輸出。")
    r.log(f"camera model OK: {sorted(models)} ({ncam} cam)")


def _find_image(images_dir: Path, name: str) -> Path | None:
    """`images_dir/name`, or its case-insensitive match (the trainers do the same)."""
    f = images_dir / name
    if f.is_file():
        return f
    if f.parent.is_dir():
        for g in f.parent.iterdir():
            if g.name.lower() == f.name.lower():
                return g
    return None


def _assert_image_sizes(sparse_dir: Path, images_dir: Path, r: Runner) -> None:
    """Every image on disk must be the size its camera was calibrated at. One
    re-exported (cropped/resized) file among a thousand trains wrong silently or
    crashes GS-2M's multi-view loss minutes in (a 'shape ... is invalid' reshape),
    so check the headers up front. A camera whose images are *all* uniformly
    resized is fine — the trainers derive focal from FoV — and is only noted."""
    Image = _pillow()
    if Image is None or _model_ext(sparse_dir) != ".bin":
        return
    cams = read_cameras(sparse_dir / "cameras.bin")
    sizes: dict[int, dict[tuple[int, int], list[str]]] = {}
    missing: list[str] = []
    for im in read_images(sparse_dir / "images.bin"):
        f = _find_image(images_dir, im["name"])
        if f is None:
            missing.append(im["name"])
            continue
        try:
            with Image.open(f) as pic:
                sz = pic.size
        except Exception:                                        # noqa: BLE001
            missing.append(im["name"])
            continue
        sizes.setdefault(im["camera_id"], {}).setdefault(sz, []).append(im["name"])
    bad: list[str] = []
    for cid, by_size in sizes.items():
        want = (cams[cid]["width"], cams[cid]["height"])
        odd = {sz: names for sz, names in by_size.items() if sz != want}
        if odd and len(by_size) == 1:
            r.log(f"[note] 相機 {cid} 的影像全部是 {next(iter(odd))}(模型 {want}),視為整批縮放")
            continue
        bad += [f"{n} {sz}≠{want}" for sz, names in odd.items() for n in names]
    if missing or bad:
        ex = "; ".join((missing and [f"{n}: 找不到/讀不到" for n in missing[:3]]) + bad[:5])
        og = images_dir.parent / f"{images_dir.name}_og"
        raise ValueError(
            f"{len(bad)} 張影像尺寸與相機模型不符、{len(missing)} 張找不到（例: {ex}）。"
            " 這通常是影像在 undistort 後被重新匯出（調色/裁切）時改了尺寸。"
            + (f" {og} 有原始檔，可用它替換這幾張。" if og.is_dir() else ""))


def _build_scene(scene: Path, sparse_dir: Path, images_dir: Path,
                 force: bool, r: Runner) -> None:
    """Materialize a GS-2M scene dir of symlinks into the COLMAP output.
    Non-destructive: only this scene dir is written (the trainer's points3D.ply
    lands in the real sparse/0 dir here, not in the source workspace)."""
    s0 = scene / "sparse" / "0"
    s0.mkdir(parents=True, exist_ok=True)
    ext = _model_ext(sparse_dir) or ".bin"      # whichever the source ships (bin/txt)
    for stem in _MODEL_STEMS:
        f = stem + ext
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
    # COLMAP's undistort stage writes masks undistorted with the same cameras next to
    # images/; expose them so `--masks masks` (relative to the scene) just works
    dense_masks = images_dir.parent / "masks"
    msk = scene / "masks"
    if dense_masks.is_dir():
        if force and msk.is_symlink():
            msk.unlink()
        if not (msk.is_symlink() or msk.exists()):
            msk.symlink_to(dense_masks.resolve())
    r.log(f"scene ready: {scene}  (sparse/0 + images → {sparse_dir.parent})")


def _png_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from a PNG's IHDR chunk — the panel env has no PIL."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def _mask_for(mask_dir: Path, image_name: str) -> Path | None:
    """The mask GS-2M will load for `image_name` (mirrors its resolveAuxPath):
    the same subfolder as the image first, then a flat `<stem>.png`."""
    rel = Path(image_name)
    for cand in (mask_dir / rel.parent / f"{rel.stem}.png", mask_dir / f"{rel.stem}.png"):
        if cand.is_file():
            return cand
    return None


def _mask_problems(mask_dir: Path, sparse_dir: Path) -> tuple[int, int, list[str]]:
    """(n_missing, n_wrong_size, examples) for every registered image. GS-2M
    resizes a wrong-size mask to the image without complaint, so a mask made on the
    *distorted* photos trains silently misaligned — catch it here instead."""
    cams = read_cameras(sparse_dir / "cameras.bin")
    missing = wrong = 0
    examples: list[str] = []
    for im in read_images(sparse_dir / "images.bin"):
        m = _mask_for(mask_dir, im["name"])
        cam = cams.get(im["camera_id"], {})
        want = (cam.get("width"), cam.get("height"))
        if m is None:
            missing += 1
            if len(examples) < 3:
                examples.append(f"{im['name']}: 無遮罩")
        elif (got := _png_size(m)) != want:
            wrong += 1
            if len(examples) < 3:
                examples.append(f"{m.name}: {got} ≠ 影像 {want}")
    return missing, wrong, examples


def _check_masks(args: str, scene: Path, sparse_dir: Path, images_dir: Path, r: Runner) -> str:
    """Validate the `--masks` dir in `args` against the model before a long run.
    If it doesn't fit but the undistorted masks beside images/ do (the usual slip:
    pointing at the pre-COLMAP mask folder), switch to those; else fail fast."""
    toks = shlex.split(args)
    idx = val = None
    for i, t in enumerate(toks):
        if t == "--masks" and i + 1 < len(toks):
            idx, val = i + 1, toks[i + 1]
        elif t.startswith("--masks="):
            idx, val = i, t.split("=", 1)[1]
    if not val or _model_ext(sparse_dir) != ".bin":   # read_images is .bin-only
        return args
    mask_dir = Path(val) if Path(val).is_absolute() else scene / val
    missing, wrong, ex = _mask_problems(mask_dir, sparse_dir)
    if not (missing or wrong):
        r.log(f"masks OK: {mask_dir}")
        return args
    dense_masks = images_dir.parent / "masks"
    if dense_masks.is_dir() and dense_masks.resolve() != mask_dir.resolve() \
            and _mask_problems(dense_masks, sparse_dir)[:2] == (0, 0):
        r.log(f"[fix] --masks {mask_dir} 不對應去畸變影像（缺 {missing}、尺寸不符 {wrong}；"
              f"例: {'; '.join(ex)}）→ 改用 undistort 產生的 {dense_masks}")
        toks[idx] = str(dense_masks) if toks[idx] == val else f"--masks={dense_masks}"
        return shlex.join(toks)
    raise ValueError(
        f"--masks {mask_dir} 與模型影像對不上：缺 {missing}、尺寸不符 {wrong}（例: {'; '.join(ex)}）。"
        " GS-2M 要的是「去畸變後」的遮罩：在 COLMAP 階段填 MASKS_DIR 跑 undistort，"
        " 它會輸出到 <workspace>/*_mapper/masks，訓練時 --masks 填 masks 即可。")


def _trained_ply(out: Path) -> Path | None:
    """A trained 3DGS PLY in the output dir, if any, across trainer layouts —
    LichtFeld `splat_<N>.ply` or GS-2M `point_cloud/iteration_<N>/point_cloud.ply`.
    Mirrors web.services.models.trained_ply (existence is all we need here; pipeline
    must not import the web layer)."""
    cands = list(out.glob("splat_*.ply"))
    cands += list(out.glob("point_cloud/iteration_*/point_cloud.ply"))
    return cands[0] if cands else None


# Formats LichtFeld's `--export` can write (see final_export_extension in
# training/training_setup.cpp). Used only to recognise "a model was produced".
_EXPORT_EXTS = (".ply", ".sog", ".spz", ".usd", ".usda", ".usdc", ".html", ".rad")

# Output-dir leaf names that don't identify a run — the panel's own convention is
# `<project>/model`, so for these the parent is the real name (same reasoning as
# web.services.forms.scene_label).
_GENERIC_OUT_LEAVES = {"model", "models", "output", "outputs", "out", "result", "results"}


def _trained_output(out: Path) -> Path | None:
    """Any finished model file in the output dir — a trained PLY, or a file written
    by `--export` (sog/spz/...). Exporting *only* sog is the panel default, so the
    PLY-only check would report "no model" on a shutdown segfault that in fact
    produced everything."""
    ply = _trained_ply(out)
    if ply is not None:
        return ply
    for f in sorted(out.glob("*")):
        if f.is_file() and f.suffix.lower() in _EXPORT_EXTS and f.stat().st_size > 0:
            return f
    return None


def _has_flag(toks: list[str], flag: str) -> bool:
    """Is `flag` present in an argv list, in either `--f v` or `--f=v` form?
    A substring test on the joined command would match `--exclude-export` for
    `--export`."""
    return any(t == flag or t.startswith(flag + "=") for t in toks)


def _export_stem(out: Path) -> str:
    """Filename stem for LichtFeld `--export` outputs: the project name rather than
    the built-in `splat_<iter>`. A folder of `splat_30000.sog` files is unusable once
    they leave the panel, and the user has to rename every one by hand."""
    if out.name.lower() in _GENERIC_OUT_LEAVES and out.parent.name:
        return out.parent.name
    return out.name or "splat"


def _run_train_binary(p: dict, spec: dict, r: Runner) -> None:
    """Run a compiled (non-Python) trainer, e.g. LichtFeld Studio. Unlike the
    conda-python path we invoke the executable directly and point `-d` at the
    undistorted COLMAP dir (sparse/ + images/) — no symlink scene needed, since
    LichtFeld's COLMAP loader reads `<data>/sparse/` natively. Strategy defaults
    come from a shipped `--config` JSON; the panel's curated fields override via CLI."""
    name = p.get("backend")
    exe = binary_exec(spec)          # resolves a relative "exec" against BASE, same as /doctor
    if exe is None:
        raw = Path(spec.get("exec", "")).expanduser()
        raise FileNotFoundError(
            f"backend '{name}' 的執行檔不存在或不可執行：{str(raw)!r}。"
            " 請在 backends.json 把 \"exec\" 指向已編譯的 binary（開 /doctor 檢查）。")

    src = Path(p["source"])
    out = Path(p["model_path"])
    if not src.is_dir():
        raise FileNotFoundError(f"source not found: {src}")

    sparse_dir, images_dir = _resolve_dense(src)
    _assert_pinhole(sparse_dir, r)
    _assert_image_sizes(sparse_dir, images_dir, r)
    data_dir = images_dir.parent              # dir holding sparse/ + images/

    config = (spec.get("config") or "").strip()
    if config:
        cp = Path(config).expanduser()
        if not cp.is_absolute():
            cp = PANEL_BASE / cp
        if not cp.is_file():
            raise FileNotFoundError(f"backend '{name}' 的 config 不存在：{cp}")
        config = str(cp)
    config_arg = f"--config {shlex.quote(config)}" if config else ""

    args = (p.get("args") or "").strip()      # curated params (assembled in app)
    extra = (p.get("extra") or "").strip()     # free escape-hatch flags
    cmd = spec.get("train_args", "-d {scene} -o {out} {config} {args} {extra}").format(
        scene=str(data_dir), out=str(out), config=config_arg, args=args, extra=extra)
    toks = shlex.split(cmd)
    # `--export` writes <out>/<stem><ext> with stem = --output-name, defaulting to
    # `splat_<iter>`. Name it after the project so the file identifies itself.
    # Skipped when the flag was given by hand (panel field or `extra`).
    if _has_flag(toks, "--export") and not _has_flag(toks, "--output-name"):
        toks += ["--output-name", _export_stem(out)]
    cmd = shlex.join(toks)
    argv = [str(exe), *toks]

    env = {}
    gpu = str(p.get("gpu", "")).strip()
    if gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = gpu       # honored regardless of the queue's scheduling

    out.mkdir(parents=True, exist_ok=True)
    r.banner(f"train start | backend={name} (binary) exe={exe.name} gpu={gpu or 'default'}")
    r.log(f"trainer: {exe} {cmd}")
    r.log(f"data:    {data_dir}")
    r.log(f"output:  {out}")
    rc = r.run(argv, cwd=str(out), env=env, check=False)   # cwd=out so ./train.log + splat_*.ply co-locate
    if rc != 0:
        # Compiled trainers (notably LichtFeld Studio) sometimes segfault during
        # shutdown — CUDA-context teardown / static-destructor order — AFTER every
        # output has been written. Don't fail a job whose model is already on disk:
        # if the trained PLY exists, warn and treat as success; otherwise it really
        # did fail before producing a model, so surface the error.
        ply = _trained_output(out)
        if ply is None:
            raise PipelineError(f"{exe.name} exited with code {rc}")
        r.log(f"[warn] {exe.name} 退出碼 {rc}（多半是收尾時 segfault）,但模型已寫出："
              f"{ply.name} → 視為訓練成功。")
    r.banner(f"train done. model={out}")
    exported = [f for f in sorted(out.glob("*"))
                if f.is_file() and f.suffix.lower() in _EXPORT_EXTS]
    if exported:
        for f in exported:
            r.log(f"[export] {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    else:
        r.log("[note] 只寫出 project.licht(LichtFeld 內部格式)。要能直接使用的檔案,"
              "請在「匯出格式」填 sog(或 ply)。")
    r.log("[note] 此 backend 不支援 mesh;訓練雲可用「🧹 在 SuperSplat 去背景」清背景後下載。")


def run_train(p: dict, r: Runner) -> None:
    name = p.get("backend") or "gs2m"
    spec = get_backend(name)
    if not spec:
        raise ValueError(f"unknown backend: {name}")

    if spec.get("launch") == "binary":          # compiled trainer (e.g. LichtFeld)
        return _run_train_binary(p, spec, r)

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
    _assert_image_sizes(sparse_dir, images_dir, r)
    scene = out.parent / f"{out.name}_scene"
    _build_scene(scene, sparse_dir, images_dir, bool(p.get("force")), r)

    args = (p.get("args") or "").strip()        # tunable params (assembled in app)
    args = _check_masks(args, scene, sparse_dir, images_dir, r)
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


# --------------------------------------------------------------------------- #
# Texture baking (texrecon) — photo texture instead of per-vertex colour
# --------------------------------------------------------------------------- #
# TSDF gives one colour per VERTEX, so its resolution is the triangle count: a
# 1 M-face mesh carries ~1 M colour samples no matter how sharp the photos were.
# texrecon instead projects the original undistorted images onto the faces and
# writes real texture atlases, which decouples appearance from tessellation.
#
# It emits as many atlases as it needs (a 1.9 M-face mesh from 543 views came out
# as 21 images), and almost nothing downstream — viewers, slicers, Blender
# imports — wants 21 materials, so merge_atlases.py repacks them into ONE. That
# repack is lossless: each source atlas is pasted at an integer pixel offset and
# every `vt` is rescaled into its sub-rectangle, so no texel is resampled.

# --- keeping the bake inside the machine's memory -------------------------- #
# texrecon undistorts and scores every view with OpenMP across ALL cores, and each
# worker holds a decoded image plus a float gradient image of the same size. On a
# 72-core box with 24 MP photos that is ~20 GB of *transient* peak before a single
# texture patch exists — the classic way this stage dies is the OOM killer, which
# takes the panel with it rather than failing the job cleanly.
#
# So the panel budgets the run instead of hoping: cap the thread count to what
# fits, and only if even a few threads don't fit, downscale the images. Downscaling
# is safe without touching the intrinsics because mvs-texturing normalises the NVM
# focal by the *loaded image's* longest side (generate_texture_views.cpp:196) —
# a uniform resize cancels out exactly.
_TEX_BYTES_PER_PX = 12          # decoded RGB + float gradient + undistort copy
_TEX_MIN_THREADS = 4            # below this the bake gets painfully slow
_TEX_MIN_SIDE = 1600            # never downscale past this; it's photo texture
_TEX_BUDGET = 0.55              # of MemAvailable, leaving room for everything else


def _pillow():
    """The Pillow `Image` module, or None when it isn't installed.

    Pillow is deliberately NOT a runtime dependency of this panel (pyproject keeps
    the install pure-Python), so every use of it here needs an answer for "it
    isn't there". Measuring degrades to a safe default; the one path that really
    needs it — rewriting images before the bake — says so instead of raising an
    ImportError from three frames down."""
    try:
        from PIL import Image
        return Image
    except ImportError:
        return None


def _mem_available() -> int:
    """Bytes of memory the kernel thinks we can take without swapping. MemAvailable,
    not MemFree: page cache is reclaimable and this box runs with most of RAM in it."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _texture_plan(px: int, cores: int, budget: int) -> tuple[int, int | None]:
    """(threads, max_side) for a bake of `px`-pixel images under `budget` bytes.

    max_side is None when the images can be used at native resolution — which is
    the normal case and the one worth protecting: this pipeline exists to keep
    full-resolution detail, so shrinking is the last resort, not the default.
    """
    if px <= 0 or budget <= 0:
        return (min(cores, 16), None)
    per_thread = px * _TEX_BYTES_PER_PX
    threads = max(1, min(cores, int(budget / per_thread)))
    if threads >= _TEX_MIN_THREADS:
        return (threads, None)
    # Even a handful of workers doesn't fit: shrink the images until they do.
    fit_px = max(1, int(budget / (_TEX_MIN_THREADS * _TEX_BYTES_PER_PX)))
    side = max(_TEX_MIN_SIDE, int(fit_px ** 0.5))
    return (_TEX_MIN_THREADS, side)


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _image_files(images_dir: Path) -> list[Path]:
    """Every image under `images_dir`, sorted, recursing into the per-camera
    subfolders COLMAP keeps (images.bin names them `cam/IMG_1.JPG`). Symlinked
    group folders — the fusion layout — are followed."""
    out: list[Path] = []
    for root, dirs, files in os.walk(images_dir, followlinks=True):
        dirs.sort()
        out += [Path(root) / f for f in sorted(files) if Path(f).suffix.lower() in _IMAGE_EXTS]
    return out


def _image_pixels(images_dir: Path) -> tuple[int, int, tuple[int, int]]:
    """(count, pixels of the largest image, its (w, h)). One PIL open per file
    header — no decoding, so this is cheap even for a few thousand photos.

    Without Pillow the pixel count comes back 0, which _texture_plan reads as
    "couldn't measure" and answers with a conservative thread count."""
    Image = _pillow()
    n, best, size = 0, 0, (0, 0)
    files = _image_files(images_dir)
    if Image is None:
        return len(files), 0, (0, 0)
    for f in files:
        n += 1
        try:
            with Image.open(f) as im:
                w, h = im.size
        except Exception:                                        # noqa: BLE001
            continue
        if w * h > best:
            best, size = w * h, (w, h)
    return n, best, size


def _stage_one(src: str, dst: str, mask: str | None, max_side: int | None) -> None:
    Image = _pillow()
    if Image is None:
        raise RuntimeError(
            "貼圖前要處理影像（套遮罩或縮圖）需要 Pillow,但這個環境沒裝:"
            " `pip install pillow`,或不要填遮罩資料夾 / 影像上限（原尺寸不需要前處理）。")
    with Image.open(src) as im:
        im = im.convert("RGB")
        if mask:
            with Image.open(mask) as mk:
                mk = mk.convert("L").point(lambda v: 255 if v > 128 else 0)
                if mk.size != im.size:
                    raise ValueError(f"遮罩與影像尺寸不符: {mask} {mk.size} vs {im.size}")
                # Background to black so texrecon's outlier removal rejects the
                # views that see backdrop on a face, instead of smearing the fake
                # background onto the object's silhouette.
                im = Image.composite(im, Image.new("RGB", im.size, (0, 0, 0)), mk)
        if max_side and max(im.size) > max_side:
            f = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))),
                           Image.Resampling.LANCZOS)
        # Keep the original extension: the NVM refers to images by name, and mve
        # picks its decoder from that extension.
        if Path(dst).suffix.lower() in (".jpg", ".jpeg"):
            im.save(dst, quality=95)
        else:
            im.save(dst)


def _stage_texture_scene(scene: Path, dest: Path, max_side: int | None,
                         mask_dir: str, r: Runner) -> Path:
    """Materialize the COLMAP dir the bake reads from: `sparse` symlinked, `images`
    either symlinked (native resolution, no masks — the common case) or rewritten.

    Staging happens even when nothing is rewritten, because the bake writes its
    model.nvm *next to the images* — into the user's dataset otherwise, where two
    texture jobs on one scene would overwrite each other's file mid-run.
    """
    src_images = scene / "images"
    shutil.rmtree(dest, ignore_errors=True)
    (dest / "images").mkdir(parents=True)
    sparse = scene / "sparse"
    (dest / "sparse").symlink_to(sparse.resolve())

    # keep the subfolder layout: the bake resolves images by their images.bin name
    rels = [f.relative_to(src_images) for f in _image_files(src_images)]
    if not rels:
        raise FileNotFoundError(f"{src_images} 底下(含子資料夾)沒有影像。")
    for d in {rel.parent for rel in rels}:
        (dest / "images" / d).mkdir(parents=True, exist_ok=True)
    if not max_side and not mask_dir:
        for rel in rels:
            (dest / "images" / rel).symlink_to((src_images / rel).resolve())
        return dest

    jobs = []
    for rel in rels:
        mask = None
        if mask_dir:
            m = _mask_for(Path(mask_dir), rel.as_posix())
            if m is None:
                raise FileNotFoundError(
                    f"缺少 {rel} 的遮罩: {Path(mask_dir) / rel.with_suffix('.png')}")
            mask = str(m)
        jobs.append((str(src_images / rel), str(dest / "images" / rel), mask, max_side))
    what = "去背+縮圖" if mask_dir and max_side else ("去背" if mask_dir else "縮圖")
    r.log(f"[texture] 前處理 {len(jobs)} 張影像({what}"
          + (f",最長邊 {max_side}px" if max_side else "") + ")…")
    # Few workers on purpose: this staging is itself a decode-and-resize, so it has
    # the same per-worker footprint the whole plan is trying to bound.
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        for _ in ex.map(lambda a: _stage_one(*a), jobs):
            r.check_cancel()
    return dest


TEX_DIR = "textured"            # under the mesh dir: <...>/mesh/textured/
TEX_RAW = "raw"                 # texrecon's own multi-atlas output
TEX_MM = "mm"                   # the marker-scaled (millimetre) copy
TEX_PREFIX = "mesh"             # <prefix>.obj / .mtl / texture/<prefix>_atlas.*


def _atlas_files(obj: Path) -> list[Path]:
    """Texture images the OBJ's .mtl references, in file order."""
    mtl = obj.with_suffix(".mtl")
    if not mtl.is_file():
        return []
    out = []
    for line in mtl.read_text().splitlines():
        if line.strip().startswith("map_Kd "):
            rel = line.split(None, 1)[1].strip()
            f = (mtl.parent / rel).resolve()
            if f.is_file():
                out.append(f)
    return out


# Shelf packing leaves gaps between differently-sized atlases; assume the merged
# canvas needs a bit more area than the sum of its parts.
_PACK_SLACK = 1.15
_MAX_DOWNSCALE = 8
# Square canvases the merge may choose from. 16384 is the ceiling on purpose: it
# is the largest texture essentially every GPU (and every viewer) accepts, and an
# atlas nothing can bind is not an atlas.
_ATLAS_SIZES = (2048, 4096, 8192, 16384)


def _atlas_area(atlases: list[Path]) -> int:
    """Total pixels across the source atlases; 0 when they can't be measured."""
    Image = _pillow()
    if Image is None:
        return 0
    Image.MAX_IMAGE_PIXELS = None          # we are the ones writing these; not a bomb
    area = 0
    for a in atlases:
        try:
            with Image.open(a) as im:
                area += im.width * im.height
        except Exception:                                         # noqa: BLE001
            continue
    return area


def _atlas_plan(atlases: list[Path], budget: int, user_px: int = 0) -> tuple[int, int]:
    """(canvas size, shrink factor) for merging `atlases` into ONE texture.

    Chosen from the atlases themselves rather than asked of the user, because the
    right answer is a property of the bake — how much texture texrecon actually
    produced — and nobody can know it before the bake runs. The rule:

    * take the SMALLEST square that holds every source atlas at full resolution,
      so a small object doesn't get a 16k canvas that is mostly black;
    * never exceed 16384, the practical hardware limit for one texture;
    * if the content doesn't fit even there, shrink by the smallest power of two
      that makes it fit — losing some texel density is the price of "one texture",
      and it beats the alternative: an 8192×44800 strip that GPUs refuse to bind
      and Pillow refuses to open, which is how a good bake became a white model.

    `user_px` (blank in the form, an explicit override elsewhere) pins the canvas
    size; the shrink is still computed so the result stays inside it.
    """
    area = _atlas_area(atlases)
    if area <= 0:
        # Nothing measurable (no Pillow, or unreadable files): take the widest
        # canvas, which is the one that keeps the packed height smallest, and let
        # the merge — which runs in the backend env, where Pillow exists — pack it.
        return (user_px or _ATLAS_SIZES[-1], 1)
    need = area * _PACK_SLACK
    # Canvas RGB buffer + the source images + the encoder's own copy.
    sizes = [user_px] if user_px else [
        px for px in _ATLAS_SIZES if not budget or px * px * 6 <= budget] or [_ATLAS_SIZES[0]]
    for px in sizes:
        if need <= px * px:
            return (px, 1)
    px = sizes[-1]
    down = 1
    while down < _MAX_DOWNSCALE and need / (down * down) > px * px:
        down *= 2
    return (px, down)


def _atlas_dims(path: Path) -> str:
    """', 8192×8192' for a log line, or '' if the size can't be read."""
    Image = _pillow()
    if Image is None:
        return ""
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as im:
            return f", {im.width}×{im.height}"
    except Exception:                                             # noqa: BLE001
        return ""


def _run_glb(r: Runner, py: Path, obj: Path, glb: Path, scale: float = 1.0) -> Path | None:
    """Re-export a textured OBJ as a single self-contained .glb (the only textured
    format the panel's one-file viewer endpoint can actually show). Never fatal —
    the OBJ is the deliverable; the GLB is the preview."""
    script = TOOLS_DIR / GLB_SCRIPT
    if not script.is_file():
        r.log(f"[texture] 略過 GLB: 找不到 {script}")
        return None
    # trimesh parses the whole OBJ into arrays and then builds the GLB in memory —
    # roughly 8× the file on disk. The GLB is only the preview, so a mesh too big
    # to convert safely is a skipped step, not a killed machine.
    need = obj.stat().st_size * 8
    avail = _mem_available()
    if avail and need > avail * _TEX_BUDGET:
        r.log(f"[texture] 略過 GLB:轉檔約需 {need / 1e9:.1f} GB,目前可用 {avail / 1e9:.1f} GB。"
              " OBJ + 貼圖仍然完整,用外部軟體開即可。")
        return None
    cmd = [str(py), "-u", str(script), "--in", str(obj), "--out", str(glb)]
    if scale != 1.0:
        cmd += ["--scale", f"{scale:.8f}"]
    r.log("glb: " + " ".join(cmd))
    rc = r.run(cmd, cwd=str(TOOLS_DIR), env={"PYTHONUNBUFFERED": "1"}, check=False)
    if rc != 0 or not glb.is_file():
        r.log(f"[texture] warning: GLB 轉檔失敗 (rc={rc}),OBJ 仍可用。")
        return None
    return glb


def _run_texture(p: dict, r: Runner, py: Path, repo: Path, spec: dict, mesh_ply: Path) -> Path | None:
    """Bake photo texture onto `mesh_ply` and return the final textured OBJ.

    Deliberately runs BEFORE marker scaling: texrecon needs the mesh in the SAME
    coordinate frame as the COLMAP cameras, and a rescaled mesh silently textures
    to ~nothing (almost every face reported unseen, then hole-filled). Scaling the
    already-textured OBJ afterwards is exact — a similarity transform moves
    vertices and leaves UVs and atlases untouched.
    """
    cfg = p.get("texture") or {}
    script = spec.get("texture_script")
    if not script or not (repo / script).is_file():
        raise FileNotFoundError(f"找不到貼圖腳本: {repo / (script or '')}")
    texrecon = texrecon_bin(spec)
    if not texrecon:
        raise FileNotFoundError(
            "找不到 texrecon 執行檔(mvs-texturing)。請先建置,或在 backends.json 設 "
            '"texrecon" 絕對路徑;開 /doctor 可確認。')

    out = Path(p["model_path"])
    scene = _scene_from_model(out)
    if not (scene / "images").exists():
        raise FileNotFoundError(
            f"找不到訓練場景的 images/（{scene}）。貼圖需要原始的去畸變影像 —— "
            "此模型可能不是用本面板訓練的,或場景已被移動。")

    tex_dir = mesh_ply.parent / TEX_DIR
    raw_dir = tex_dir / TEX_RAW
    env = {"PYTHONUNBUFFERED": "1"}

    mask_dir = str(cfg.get("mask_dir") or "").strip()
    if mask_dir and not Path(mask_dir).is_dir():
        raise FileNotFoundError(f"遮罩資料夾不存在: {mask_dir}")

    # --- memory plan ------------------------------------------------------- #
    n_img, px, (iw, ih) = _image_pixels(scene / "images")
    cores = os.cpu_count() or 1
    avail = _mem_available()
    budget = int(avail * _TEX_BUDGET)
    threads, auto_side = _texture_plan(px, cores, budget)
    # An explicit cap from the form always applies; auto only ever tightens it.
    user_side = int(cfg.get("max_image_px") or 0)
    max_side = min([v for v in (user_side or None, auto_side) if v], default=None)
    user_threads = int(cfg.get("threads") or 0)
    if user_threads:
        threads = max(1, min(user_threads, cores))
    r.log(f"[texture] {n_img} 張影像 · 最大 {iw}×{ih} · 可用記憶體 {avail / 1e9:.0f} GB "
          f"→ 執行緒 {threads}/{cores}"
          + (f" · 影像縮到最長邊 {max_side}px" if max_side else " · 影像原尺寸"))
    if auto_side:
        r.log(f"[texture] 記憶體不足以用原尺寸貼圖,已自動縮圖到 {auto_side}px 以避免 OOM。"
              " 想保留原尺寸請加大記憶體或分批處理。")

    staged = _stage_texture_scene(scene, tex_dir / "scene", max_side, mask_dir, r)

    bake = [str(py), "-u", str(repo / script), "-s", str(staged),
            "--mesh", str(mesh_ply), "-o", str(raw_dir), "--prefix", TEX_PREFIX,
            "--texrecon", str(texrecon), "--num-threads", str(threads),
            f"--data-term={cfg.get('data_term', 'area')}",
            f"--outlier-removal={cfg.get('outlier_removal', 'gauss_clamping')}"]
    if cfg.get("keep_unseen"):
        bake.append("--keep-unseen-faces")
    if cfg.get("low_memory"):
        # The global seam leveling builds one sparse system over every patch
        # vertex; it is the single biggest allocation after the views, and the
        # local (Poisson) leveling still hides most seams without it.
        bake += ["--", "--skip_global_seam_leveling"]

    r.banner("texture | 把照片投影到 mesh 上")
    r.log("bake: " + " ".join(bake))
    try:
        r.run(bake, cwd=str(repo), env=env)
    finally:
        shutil.rmtree(staged, ignore_errors=True)

    raw_obj = raw_dir / f"{TEX_PREFIX}.obj"
    if not raw_obj.is_file():
        raise PipelineError(f"貼圖沒有產出 OBJ: {raw_obj}(原因請看上面的輸出)")
    n_atlas = len(_atlas_files(raw_obj))
    r.log(f"[texture] 產出 {n_atlas} 張貼圖")

    final = raw_obj
    merge_script = spec.get("atlas_script")
    if cfg.get("merge", True) and n_atlas > 1:
        if not merge_script or not (repo / merge_script).is_file():
            r.log(f"[texture] warning: 找不到合併腳本 {repo / (merge_script or '')},保留 {n_atlas} 張貼圖。")
        else:
            r.banner(f"texture | 把 {n_atlas} 張貼圖合併成 1 張")
            # Size and shrink are decided from the bake's own output, not asked
            # of the user — see _atlas_plan.
            canvas_px, down = _atlas_plan(_atlas_files(raw_obj), budget,
                                          int(cfg.get("atlas_px") or 0))
            merge = [str(py), "-u", str(repo / merge_script), "--obj", str(raw_obj),
                     "-o", str(tex_dir), "--prefix", TEX_PREFIX,
                     "--canvas-width", str(canvas_px)]
            q = int(cfg.get("jpg_quality", 95) or 0)
            if q > 0:
                merge += ["--jpg-quality", str(q)]
            if down > 1:
                merge += ["--downscale", str(down)]
                r.log(f"[texture] 貼圖總量超過單張 {canvas_px}×{canvas_px} 的上限,"
                      f"合併時縮 {down}×（{canvas_px} 已是單張貼圖的硬體上限;"
                      "要保留全解析度就取消「合併成單張貼圖」）。")
            else:
                r.log(f"[texture] 單張貼圖尺寸自動選用 {canvas_px}（原解析度,不縮圖）。")
            r.log("merge: " + " ".join(merge))
            r.run(merge, cwd=str(repo), env=env)
            merged = tex_dir / f"{TEX_PREFIX}.obj"
            if merged.is_file():
                final = merged
                # raw/ is a full second copy of the OBJ plus every source atlas —
                # on a 2 M-face mesh that is ~200 MB of exact duplicate. It is a
                # reproducible intermediate, so it goes unless asked for.
                if not cfg.get("keep_raw"):
                    mb = sum(f.stat().st_size for f in raw_dir.rglob("*") if f.is_file()) / 1e6
                    shutil.rmtree(raw_dir, ignore_errors=True)
                    r.log(f"[texture] 已刪除合併前的中間檔 raw/（省下 {mb:.0f} MB）")
            else:
                r.log("[texture] warning: 合併沒有產出 OBJ,改用未合併的版本。")
    elif n_atlas == 1:
        # Nothing to merge, but the output should still land where a merged one
        # would — every download path and the viewer look in textured/, not raw/.
        r.log("[texture] 本來就只有 1 張貼圖,跳過合併。")
        for item in (f"{TEX_PREFIX}.obj", f"{TEX_PREFIX}.mtl", "texture"):
            src = raw_dir / item
            if src.exists():
                dst = tex_dir / item
                if dst.is_dir():
                    shutil.rmtree(dst, ignore_errors=True)
                elif dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))
        final = tex_dir / f"{TEX_PREFIX}.obj"
        if not cfg.get("keep_raw"):
            shutil.rmtree(raw_dir, ignore_errors=True)

    r.log(f"[texture] obj: {final}")
    for a in _atlas_files(final):
        r.log(f"[texture] atlas: {a}  ({a.stat().st_size / 1e6:.1f} MB{_atlas_dims(a)})")
    if cfg.get("glb", True):
        glb = _run_glb(r, py, final, tex_dir / f"{TEX_PREFIX}.glb")
        if glb:
            r.log(f"[texture] glb: {glb}")
    r.banner(f"texture done. {final}")
    return final


def _scale_textured_obj(obj: Path, dest_dir: Path, mult: float) -> Path:
    """Copy a textured OBJ package into `dest_dir` with vertex positions multiplied
    by `mult`. Only `v` lines change — `vt`/`vn`/`f` and the atlas are identical,
    because scaling is a similarity transform and UVs don't live in world space."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / obj.name
    with obj.open() as fin, dest.open("w") as fout:
        for line in fin:
            if line.startswith("v "):
                parts = line.split()
                x, y, z = (float(v) * mult for v in parts[1:4])
                fout.write(f"v {x:.6f} {y:.6f} {z:.6f}{''.join(' ' + t for t in parts[4:])}\n")
            else:
                fout.write(line)
    mtl = obj.with_suffix(".mtl")
    if mtl.is_file():
        shutil.copy2(mtl, dest_dir / mtl.name)
    for a in _atlas_files(obj):
        rel = a.relative_to(obj.parent) if obj.parent in a.parents else Path(a.name)
        (dest_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a, dest_dir / rel)
    return dest


def _run_marker_scale(p: dict, r: Runner, py: Path, mesh_ply: Path,
                      textured: Path | None = None) -> None:
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

    # The textured OBJ gets the same factor. No re-bake and no texel resampling:
    # only `v` lines are rewritten, so the atlas and every UV stay byte-identical.
    if textured is not None and textured.is_file():
        mm_dir = textured.parent / TEX_MM if textured.parent.name != TEX_RAW \
            else textured.parent.parent / TEX_MM
        scaled_obj = _scale_textured_obj(textured, mm_dir, mm_per_unit)
        r.log(f"[texture] scaled obj: {scaled_obj}")
        glb = _run_glb(r, py, scaled_obj, mm_dir / f"{TEX_PREFIX}.glb")
        if glb:
            r.log(f"[texture] scaled glb: {glb}")


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
        # Texture first, marker scale second — see _run_texture's docstring: texrecon
        # needs the mesh in the cameras' own frame, and scaling a textured OBJ after
        # the fact is exact.
        textured = None
        if (p.get("texture") or {}).get("enable"):
            try:
                textured = _run_texture(p, r, py, repo, spec, mesh_ply)
            except Cancelled:
                raise
            except Exception as exc:   # the vertex-coloured mesh is still valid output
                r.banner("貼圖失敗 — 保留未貼圖的 mesh")
                r.log(f"[texture] 失敗: {exc}")
        if (p.get("marker") or {}).get("enable"):
            try:
                _run_marker_scale(p, r, py, mesh_ply, textured)
            except Cancelled:
                raise
            except Exception as exc:   # keep the (valid) unscaled mesh; just flag scaling
                r.banner("marker scale 失敗 — 保留未縮放的 mesh")
                r.log(f"[mesh] marker scale 失敗: {exc}")
    else:
        r.log("[mesh] warning: 找不到 tsdf_post.ply,請檢查上面的 render 輸出。")
    r.banner(f"mesh done. model={out}")
