"""Reconstruction acceptance check — subprocess wrapper around tools/verify_recon.py.

Why a subprocess: the checker needs `pycolmap`, which the panel env deliberately
does not carry (it keeps the web layer torch-free and light). Same approach
`backends.py` uses for the trainer envs — locate a sibling conda env that has the
import and shell out to it.

What it checks (see tools/verify_recon.README.md): per-sensor observation counts,
distortion round-trip invertibility, rig extrinsics vs a calibration certificate,
camera centres vs vendor EO, same-frame sensor offsets, and cross-sensor epipolar
Sampson error. `model_aligner`'s EO RMS validates only the reference sensor's
*position*, so a reconstruction can report a healthy alignment while an entire
camera holds zero observations — these checks are what catch that.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pipeline.backends import conda_envs_dir

TOOL = Path(__file__).resolve().parent.parent / "tools" / "verify_recon.py"

# Envs likely to carry pycolmap on a recon workstation, tried in order before the
# broader scan. Cheap to extend; order only affects which one wins a tie.
ENV_CANDIDATES = ("gsplat", "gs2m", "clm_gs", "colmap_panel", "rec", "milo")

_cached_python: Path | None = None
_probed = False


def _has_pycolmap(py: Path) -> bool:
    """True if `py` can import pycolmap. Filters on site-packages first so the
    common 'env exists but lacks the import' case costs a stat, not a process."""
    if not py.is_file():
        return False
    site = list(py.parent.parent.glob("lib/python*/site-packages"))
    if site and not any((s / "pycolmap").exists() or list(s.glob("pycolmap*")) for s in site):
        return False
    try:
        return subprocess.run([str(py), "-c", "import pycolmap"],
                              capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def verify_python() -> Path | None:
    """Absolute python that can run the checker, or None if no env has pycolmap.

    Override with RECON_VERIFY_PYTHON when the auto-pick guesses wrong."""
    global _cached_python, _probed
    if _probed:
        return _cached_python
    _probed = True

    if override := os.environ.get("RECON_VERIFY_PYTHON"):
        p = Path(override)
        _cached_python = p if p.is_file() else None
        return _cached_python

    if _has_pycolmap(Path(sys.executable)):
        _cached_python = Path(sys.executable)
        return _cached_python

    envs = conda_envs_dir()
    if not envs:
        return None
    seen = set()
    for name in ENV_CANDIDATES:
        seen.add(name)
        if _has_pycolmap(py := envs / name / "bin" / "python"):
            _cached_python = py
            return py
    for d in sorted(envs.iterdir()):
        if d.name in seen or not d.is_dir():
            continue
        if _has_pycolmap(py := d / "bin" / "python"):
            _cached_python = py
            return py
    return None


def run_verify(model: str | Path,
               db: str | Path | None = None,
               manifest: str | Path | None = None,
               undistorted: bool = False,
               maxper: int = 12,
               timeout: int = 900) -> dict:
    """Run the checker. Returns {ok, rc, text, cmd, python}.

    rc 0 = every check passed, 1 = at least one failed, other = the tool itself
    errored (text carries the traceback). `ok` is rc == 0."""
    model = Path(model)
    if not (model / "cameras.bin").is_file() and not (model / "cameras.txt").is_file():
        return {"ok": False, "rc": -1, "text": f"不是 COLMAP 模型目錄(缺 cameras.bin): {model}",
                "cmd": "", "python": ""}
    py = verify_python()
    if py is None:
        return {"ok": False, "rc": -1, "python": "", "cmd": "",
                "text": ("找不到裝有 pycolmap 的 conda env。\n"
                         "在 local.env 設 RECON_VERIFY_PYTHON=/path/to/env/bin/python,"
                         "或 `conda install -n gsplat pycolmap`。")}

    cmd = [str(py), str(TOOL), str(model), "--maxper", str(maxper)]
    if db:
        cmd += ["--db", str(db)]
    if manifest:
        cmd += ["--manifest", str(manifest)]
    if undistorted:
        cmd += ["--undistorted"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        text = (r.stdout or "") + (("\n" + r.stderr) if r.stderr.strip() else "")
        rc = r.returncode
    except subprocess.TimeoutExpired:
        text, rc = f"驗收逾時 ({timeout}s)。大模型可以調小 --maxper。", -1
    return {"ok": rc == 0, "rc": rc, "text": text.strip(),
            "cmd": " ".join(cmd), "python": str(py)}


def guess_inputs(model: str | Path) -> dict:
    """Best-effort sibling paths for a model dir: the workspace database and any
    case.yaml, plus whether the model looks undistorted (an images/ next to it)."""
    model = Path(model)
    out = {"db": "", "manifest": "", "undistorted": False}
    for ws in (model.parent, model.parent.parent, model.parent.parent.parent):
        if not out["db"] and (c := ws / "database.db").is_file():
            out["db"] = str(c)
        if not out["manifest"]:
            for name in ("case.yaml", "case.yml", "case.json"):
                if (c := ws / name).is_file():
                    out["manifest"] = str(c)
                    break
    # image_undistorter writes <out>/sparse + <out>/images; that model's coordinates
    # no longer match the DB keypoints, so the epipolar check must be skipped.
    if model.name == "sparse" and (model.parent / "images").is_dir():
        out["undistorted"] = True
    return out
