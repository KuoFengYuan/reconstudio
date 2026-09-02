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
import re
import shlex
from pathlib import Path

from .backends import binary_exec, env_python, get_backend, repo_path
from .model import read_cameras
from .runner import Cancelled, PipelineError, Runner

TRAIN_DEFAULTS = {"backend": "lichtfeld-mrnf", "gpu": "0", "extra": "", "force": False}

# Panel repo root (configs/ live here; resolved for backends that ship a config).
PANEL_BASE = Path(__file__).resolve().parent.parent

# Marker-scaling tools are panel-owned (live in this project's tools/), but run
# in the backend's env (they need cv2/open3d/plyfile, which the trainer env has).
TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
MARKER_SCRIPT, SCALE_SCRIPT = "estimate_marker_scale.py", "scale_mesh.py"

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
    r.log(f"scene ready: {scene}  (sparse/0 + images → {sparse_dir.parent})")


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
