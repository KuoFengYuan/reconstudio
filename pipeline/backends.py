"""Training-backend registry + environment resolution + preflight checks.

The panel itself stays torch-free: it never imports a trainer, it only spawns
one as a subprocess inside that trainer's own conda env. Each backend is *data*
(which env, which repo, how to build the command), so enabling a trainer on a
new machine is a config entry — not a code change. This is what makes the panel
portable: deploying it is `pip install -r requirements.txt`; the heavy CUDA env
is a per-machine prerequisite that backends.py merely *locates* and *checks*.

Interpreter resolution order for a backend (most portable first):
  1. an explicit absolute "python" in the backend spec
  2. $CONDA_ROOT/envs/<env>/bin/python
  3. <envs>/<env>/bin/python  derived from this process's own sys.prefix
  4. `conda info --base`/envs/<env>/bin/python, then common install locations

Per-machine overrides live in backends.json (gitignored; see
backends.example.json), shallow-merged over the BUILTIN_BACKENDS below.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent          # colmap_panel/
COLMAP_BIN = os.environ.get("COLMAP_BIN", "colmap")
BACKENDS_FILE = Path(os.environ.get("COLMAP_PANEL_BACKENDS", BASE / "backends.json"))

# Built-in specs. Zero-config when the conda env name + sibling repo match;
# override/extend via backends.json. Repos are relative to BASE unless absolute.
BUILTIN_BACKENDS: dict[str, dict] = {
    "gs2m": {
        "label": "GS-2M（材質感知 → mesh）",
        "conda_env": "gs2m",
        "repo": "../GS-2M",
        "train_script": "train.py",
        # {scene}/{out} filled by train.py; {args} = the tunable params below;
        # {extra} = a free escape-hatch text field for anything not exposed.
        "train_args": "-s {scene} -m {out} {args} {extra}",
        "default_iterations": 30000,
        # GS-2M reads sparse/0 + images/; we adapt a COLMAP workspace to that shape.
        "scene_layout": "colmap_sparse0",
        # imported in the trainer env by /doctor to catch arch/CUDA mismatch early.
        "probe_imports": ["torch", "diff_gaussian_rasterization"],
        # post-training mesh extraction the user can run next (shown on "done").
        "mesh_cmd": "python render.py -m {out} --extract_mesh --skip_test",
        # Tunable params rendered as form fields — NOT hardcoded in the panel, and
        # overridable per machine via backends.json. type: int|float|str|bool|select.
        "params": [
            {"key": "iterations", "flag": "--iterations", "type": "int",
             "default": "30000", "label": "迭代次數",
             "hint": "總訓練步數。預設 30000 通常足夠;資料量大或要更精細可加到 40000–50000，但時間正比拉長。"},
            {"key": "resolution", "flag": "-r", "type": "int", "default": "1",
             "label": "解析度降採樣 (-r)",
             "hint": "1 = 用原解析度訓練（預設、品質最佳）。當影像長/寬 > 1600px，建議設 2（半解析度）以加速並省顯存;3、4 再更小。"},
            {"key": "data_device", "flag": "--data_device", "type": "select",
             "options": ["cpu", "cuda"], "default": "cpu", "label": "影像快取裝置",
             "hint": "訓練影像快取在哪。cuda 最快但吃顯存，影像多（數百張以上）容易爆顯存;放 cpu 改用系統記憶體較穩，速度差距不大。"},
            {"key": "sh_degree", "flag": "--sh_degree", "type": "int", "default": "3",
             "label": "球諧階數 (SH degree)",
             "hint": "顏色隨視角變化的能力。3 = 最高品質（預設）;調低（1–2）可省記憶體、訓練略快，但反光/視角相依的外觀會變差。"},
            {"key": "reflection_threshold", "flag": "--reflection_threshold",
             "type": "float", "default": "1.0", "label": "反射閾值 (reflection_threshold)",
             "hint": "判定「平滑/反光面」的靈敏度，影響多視角光度變化的解讀。漫反射（霧面）場景設 ≥ 1.0;反光/光澤面設 < 1.0（越低越敏感）。"},
            {"key": "lambda_normal", "flag": "--lambda_normal", "type": "float",
             "default": "0.1", "label": "法線正則權重 λ_normal",
             "hint": "影響表面完整度。重建出的 mesh 不封閉、有破洞時調高（例如 0.2–0.5）來補洞;太高則細節會被抹平。"},
            {"key": "eval", "flag": "--eval", "type": "bool", "default": False,
             "label": "切 train/test 評估 (--eval)",
             "hint": "每 8 張抽 1 張當測試集以計算 PSNR 等指標;純做 mesh 重建時可不開，讓全部影像都參與訓練。"},
            # --- group: 材質分解（只在 --material 開啟時生效） ---
            {"key": "material", "flag": "--material", "type": "bool", "default": False,
             "group": "材質分解（進階）", "label": "材質分解 (--material)",
             "hint": "開啟 PBR 材質分解（albedo 反照率 / roughness 粗糙度 / 金屬度），自 iter 5000 起自動開始（不必加迭代數）。"
                     "建議開啟的時機：① 物體有反光或光澤面 — 把高光從幾何裡分離，避免反光被誤烤成凸起的假幾何;"
                     "② 你需要 relighting（重打光）或要匯出材質貼圖。純霧面物體、或只想要 mesh，就不用開，可省時間與顯存。"},
            {"key": "metallic", "flag": "--metallic", "type": "bool", "default": False,
             "group": "材質分解（進階）", "label": "金屬度通道 (--metallic)",
             "hint": "僅在 --material 開啟時有效。重建金屬材質（金屬器物、金屬漆）時才建議開，多估一個金屬度通道;非金屬物體請留關。"},
            {"key": "gamma", "flag": "--gamma", "type": "bool", "default": False,
             "group": "材質分解（進階）", "label": "Gamma 色調映射 (--gamma)",
             "hint": "僅在 --material 開啟時有效。輸入影像偏暗或動態範圍大、材質分解結果過曝/過暗時可試開。"},
            {"key": "lambda_smooth", "flag": "--lambda_smooth", "type": "float",
             "default": "0.0", "group": "材質分解（進階）", "label": "粗糙度平滑 λ_smooth",
             "hint": "僅在 --material 有效。當反射線索不足、粗糙度圖雜訊大時調高，把正確判定的粗糙度往周圍傳播;預設 0 = 不啟用。"},
            # --- group: 前景遮罩（物件去背重建） ---
            {"key": "masks", "flag": "--masks", "type": "str", "default": "",
             "group": "前景遮罩（物件去背）", "label": "前景遮罩目錄 (--masks)", "placeholder": "masks 或絕對路徑",
             "hint": "物件去背重建用。填放遮罩 PNG 的資料夾名（相對 source，例如 masks）或絕對路徑;可先用 scripts/mask.py 產生。留空 = 不使用。"},
            {"key": "mask_gt", "flag": "--mask_gt", "type": "bool", "default": False,
             "group": "前景遮罩（物件去背）", "label": "遮罩 GT 影像 (--mask_gt)",
             "hint": "搭配 --masks 使用。把背景從 GT 影像挖掉、只擬合前景物件;背景雜亂、要乾淨單一物件 mesh 時很有用。"},
        ],
        # --- mesh extraction (render.py --extract_mesh) ---
        "mesh_script": "render.py",
        "mesh_args": "-m {out} --extract_mesh --skip_test {args} {extra}",
        "mesh_params": [
            {"key": "mesh_only", "flag": "--mesh_only", "type": "bool", "default": True,
             "label": "只抽 mesh（加速）",
             "hint": "跳過逐視角 PNG（render/normal/depth/PBR）輸出，只做 TSDF 融合抽 mesh，大幅加速。只想要 mesh 時建議開。"},
            {"key": "auto_voxel", "flag": "--auto_voxel", "type": "bool", "default": True,
             "label": "自動體素大小（推薦）",
             "hint": "從高斯點雲密度自動估 voxel_size（作者推薦,跨尺度最穩）。需要 scipy;不可用時自動 fallback 到啟發式。勾選時下方 voxel_size / sdf_trunc 會停用,由程式自行決定。"},
            {"key": "voxel_size", "flag": "--voxel_size", "type": "float", "default": "0.006",
             "label": "體素大小 voxel_size",
             "hint": "TSDF 取樣格大小（場景座標單位,非公分）。預設 0.006,取消上方「自動」才生效。越小 → 越密越細,但越慢、越吃記憶體。注意:此值是相對場景尺度(≈ cameras_extent/1250),換差很多的拍攝距離時要重估。"},
            {"key": "sdf_trunc", "flag": "--sdf_trunc", "type": "float", "default": "",
             "label": "SDF 截斷 sdf_trunc",
             "hint": "越小 → 表面越銳利但對深度雜訊越敏感;越大 → 越穩但細節變糊。留空 = 自動（4 × voxel_size,例如 voxel 0.006 → 0.024）。"},
            {"key": "max_depth", "flag": "--max_depth", "type": "float", "default": "",
             "label": "最大融合深度 max_depth",
             "hint": "depth fusion 時超過此距離的點丟棄。留空 = 自動（2 × 場景半徑）;背景雜訊多可調小。"},
            {"key": "num_clusters", "flag": "--num_clusters", "type": "int", "default": "1",
             "label": "保留聚類數 num_clusters",
             "hint": "後處理保留幾個連通塊。1 = 只取最大物件（預設）;場景含多個分離物件時再調高。"},
            {"key": "filter_depth", "flag": "--filter_depth", "type": "bool", "default": False,
             "label": "融合前過濾深度",
             "hint": "TSDF 融合前先過濾深度圖，可去除部分雜訊浮點。"},
        ],
    },
}


def build_cli(params: list, values: dict) -> str:
    """Assemble CLI tokens from a param schema (list of {key,flag,type,...}) +
    submitted `values` (keyed by each param's `key`). Bools emit just the flag
    when truthy; others emit `flag value` when non-empty. Values are shell-quoted
    so paths with spaces survive the round-trip through shlex.split in train.py.
    Reused for both training (`params`) and mesh extraction (`mesh_params`)."""
    import shlex
    toks: list[str] = []
    for pr in params:
        v = values.get(pr["key"])
        if pr.get("type") == "bool":
            if v:
                toks.append(pr["flag"])
        else:
            s = ("" if v is None else str(v)).strip()
            if s:
                toks += [pr["flag"], s]
    return " ".join(shlex.quote(t) for t in toks)


# --------------------------------------------------------------------------- #
# Environment / path resolution
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def conda_envs_dir() -> Path | None:
    """Directory that holds conda environments (.../envs), or None if unknown."""
    root = os.environ.get("CONDA_ROOT")
    if root and (Path(root) / "envs").is_dir():
        return Path(root) / "envs"
    # We run inside <root>/envs/<panel-env>, so our sys.prefix's parent is envs/.
    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs" and prefix.parent.is_dir():
        return prefix.parent
    try:
        base = subprocess.check_output(
            ["conda", "info", "--base"], text=True, stderr=subprocess.DEVNULL).strip()
        if base and (Path(base) / "envs").is_dir():
            return Path(base) / "envs"
    except Exception:
        pass
    for d in (Path.home() / "miniconda3", Path.home() / "anaconda3", Path("/opt/conda")):
        if (d / "envs").is_dir():
            return d / "envs"
    return None


def env_python(spec: dict) -> Path | None:
    """Absolute python for a backend's conda env, or None if not found."""
    if spec.get("python"):
        py = Path(spec["python"])
        return py if py.is_file() else None
    envs = conda_envs_dir()
    if not envs:
        return None
    py = envs / spec["conda_env"] / "bin" / "python"
    return py if py.is_file() else None


def repo_path(spec: dict) -> Path:
    r = Path(spec.get("repo", "."))
    return r if r.is_absolute() else (BASE / r).resolve()


# --------------------------------------------------------------------------- #
# Backend listing (built-ins merged with per-machine backends.json)
# --------------------------------------------------------------------------- #
def load_backends() -> dict[str, dict]:
    out = {k: dict(v) for k, v in BUILTIN_BACKENDS.items()}
    if BACKENDS_FILE.is_file():
        try:
            user = json.loads(BACKENDS_FILE.read_text())
            for name, spec in (user or {}).items():
                out[name] = {**out.get(name, {}), **spec}     # shallow override
        except Exception:
            pass            # a broken backends.json must not take the panel down
    return out


def get_backend(name: str) -> dict | None:
    return load_backends().get(name)


def available_backends() -> list[dict]:
    """Light summary for the UI form: each backend with a resolved `ready` flag."""
    res = []
    for name, spec in load_backends().items():
        py = env_python(spec)
        repo = repo_path(spec)
        ready = bool(py) and (repo / spec.get("train_script", "train.py")).is_file()
        res.append({
            "name": name,
            "label": spec.get("label", name),
            "ready": ready,
            "params": spec.get("params", []),         # training-param schema
            # mesh extraction is backend-specific (e.g. GS-2M's render.py); a backend
            # supports it only if it declares mesh_args. gsplat & co. won't.
            "mesh": bool(spec.get("mesh_args")),
            "mesh_params": spec.get("mesh_params", []),
        })
    return res


# --------------------------------------------------------------------------- #
# GPU discovery (never hardcode the GPU count — machines differ)
# --------------------------------------------------------------------------- #
def list_gpus() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            gpus.append({"index": int(parts[0]), "name": parts[1],
                         "mem_mb": int(parts[2]) if parts[2].isdigit() else 0})
    return gpus


# --------------------------------------------------------------------------- #
# Preflight /doctor — turn "deploy to a new machine" into a checklist
# --------------------------------------------------------------------------- #
def _colmap_check() -> dict:
    path = shutil.which(COLMAP_BIN)
    version = None
    if path:
        try:
            version = subprocess.check_output(
                [COLMAP_BIN, "-h"], text=True, stderr=subprocess.STDOUT).splitlines()[0].strip()
        except Exception:
            pass
    return {"bin": COLMAP_BIN, "path": path, "version": version, "ok": bool(path)}


def _probe_env(py: Path, imports: list[str]) -> dict:
    """Run a tiny script in the trainer env: torch+CUDA + import the compiled
    submodules. Importing e.g. diff_gaussian_rasterization here catches the most
    common move-to-new-machine failure (extension built for another GPU arch)."""
    code = (
        "import json, sys\n"
        "r = {'python': sys.version.split()[0]}\n"
        "try:\n"
        "    import torch\n"
        "    r['torch'] = torch.__version__\n"
        "    r['cuda'] = bool(torch.cuda.is_available())\n"
        "    r['device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None\n"
        "except Exception as e:\n"
        "    r['torch_error'] = repr(e)\n"
        "fail = {}\n"
        "for m in %r:\n"
        "    if m == 'torch':\n"
        "        continue\n"
        "    try:\n"
        "        __import__(m)\n"
        "    except Exception as e:\n"
        "        fail[m] = repr(e)\n"
        "r['import_fail'] = fail\n"
        "print(json.dumps(r))\n"
    ) % (list(imports),)
    try:
        out = subprocess.check_output([str(py), "-c", code], text=True,
                                      stderr=subprocess.STDOUT, timeout=180)
        return json.loads(out.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"error": "probe timed out (CUDA init >180s?)"}
    except Exception as e:
        return {"error": repr(e)}


def doctor(deep: bool = True) -> dict:
    """Full preflight report. deep=True imports torch in each env (slow, ~seconds
    per backend, triggers CUDA init); deep=False only checks paths exist."""
    report: dict = {
        "colmap": _colmap_check(),
        "conda_envs_dir": str(conda_envs_dir() or ""),
        "gpus": list_gpus(),
        "backends": {},
    }
    for name, spec in load_backends().items():
        py = env_python(spec)
        repo = repo_path(spec)
        script_ok = (repo / spec.get("train_script", "train.py")).is_file()
        item = {
            "label": spec.get("label", name),
            "conda_env": spec.get("conda_env"),
            "env_python": str(py) if py else None,
            "repo": str(repo),
            "repo_ok": script_ok,
            "ready": bool(py) and script_ok,
        }
        if deep and py:
            item["probe"] = _probe_env(py, spec.get("probe_imports", ["torch"]))
        report["backends"][name] = item
    return report
