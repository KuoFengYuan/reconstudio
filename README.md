# Recon Studio

A local web panel for the full reconstruction pipeline — **video → 清晰幀 → COLMAP →
3D Gaussian Splatting 訓練 → Mesh** — implemented as pure-Python modules under
[`pipeline/`](pipeline/). The panel itself is **torch-free**: heavy tools (`ffmpeg`,
`colmap`) and trainers (GS-2M, …) are invoked as **subprocesses**, so progress,
cancellation, live logs, an in-browser 3D viewer, and a mesh download are first-class.

Single-machine local tool, binds to `127.0.0.1` by default. Runs up to **`MAX_JOBS`
(default 4) jobs concurrently**; GPU is chosen manually per job.

```
 ① 抽幀                  ② COLMAP                ③ 訓練                  ④ Mesh
 video ──run_frames──► images ──run_colmap──► sparse + 去畸變dense ──run_train──► 3DGS model ──run_mesh──► tsdf_post.ply
 (NVDEC, 並行)         (+FullHD等比縮放)        (GS-2M, 自家 conda env)            (render.py)   ⬇ 可下載
```

Each finished stage offers a one-click button to prefill the next (frames→COLMAP→訓練→Mesh).

---

## Stages

### ① 抽幀 (frames) — videos → sharp frames
Extracts `fps` frames/sec, scores blur per frame (ffmpeg `blurdetect`), keeps the
sharpest `keep_pct`%. Decoding uses the GPU (NVDEC) automatically, per-video CPU
fallback. Output mirrors the input tree, one dir per video, sharp frames only:

```
IN   <root>/<group>/<video>.MOV          e.g.  FY115/FY115_0518/A/IMG_3600.MOV
OUT  <out>/<group>/frames_<video>/*.jpg  e.g.  FY115/0518_colmap/A/frames_IMG_3600/*.jpg
```

### ② COLMAP — images → reconstruction
Layout auto-detected (or forced via the `layout` selector); a workspace nested inside
`image_root` is ignored when detecting groups:

```
single  XXX/*.jpg                -> 1 shared camera (also when the root itself holds images)
multi   ROOT/<group>/*.jpg       -> 1 camera per group   (CAMERA_MODE=per_folder)
nested  ROOT/<group>/<vid>/*.jpg -> staged, 1 camera per group   (① feeds this)
```

**影像解析度 (resize, 預設 FullHD)** — unlike a max-size cap, FullHD **physically
downscales** every input to a real FullHD copy under `workspace/images_fullhd/`
(longest side ≤ 1920, **aspect-ratio preserving, never upscaled**), and the whole
COLMAP run (features / mapping / undistort) operates on those.
- Quality: **Lanczos** downscaling + max-quality encode (JPEG `q=1`, 4:4:4 no chroma
  subsampling; PNG/TIFF kept lossless), so 4K → FullHD stays sharp.
- Speed: many `ffmpeg` workers in parallel (CPU-bound; NVDEC can't accelerate JPEG
  stills). ~18× over serial on a many-core box. Tune with `COLMAP_PANEL_RESIZE_WORKERS`.
- Idempotent (sentinel), resumable (skips done files), cancellable.

Output (workspace): `database.db`, `image_list.txt`, `images_fullhd/` (when FullHD),
`sparse/0/{cameras,images,points3D}.bin`, `<dataset>_<mapper>_mapper/` (undistorted
**PINHOLE** dense input — what the trainer consumes), `pipeline.log`, sentinels, and
`sparse/points.ply` (cached for the viewer).

### ③ 訓練 (training) — reconstruction → 3DGS model
Runs a Gaussian-splatting trainer in **its own conda env** as a subprocess. Backends
are **data, not code** (see [Backends](#backends)); built-in default is **GS-2M**.

- **Scene adaptation**: a COLMAP workspace is exposed to the trainer non-destructively
  as `<model>_scene/` (symlinks: `sparse/0/{cameras,images,points3D}.bin` + `images/`).
  The undistorted **PINHOLE** model is used; a distorted (OPENCV) model is rejected up
  front with a clear error (the #1 way this integration goes wrong).
- **Tunable params** are rendered from the backend schema (no hard-coding): `--iterations`,
  `-r` (1 = full res), `--data_device` (cpu for big sets), `--sh_degree`, `--material`
  (PBR; auto-starts at iter 5000 — open it for reflective objects / relighting / texture
  export), `--metallic`/`--gamma`/`--lambda_smooth` (材質群組), `--lambda_normal` (補洞),
  `--reflection_threshold`, `--masks`/`--mask_gt` (前景去背), `--eval`, plus a free `extra` field.
- **GPU**: chosen manually per job → `CUDA_VISIBLE_DEVICES` (no auto GPU selection).

### ④ Mesh — model → triangle mesh
Backend-specific (only backends that declare `mesh_args`, e.g. GS-2M's
`render.py --extract_mesh --skip_test`). Tunable: `--mesh_only` (加速), `--auto_voxel`
(推薦), `voxel_size` / `sdf_trunc` / `max_depth` / `num_clusters` / `filter_depth`.
Output `…/train/ours_<iter>/mesh/tsdf_post.ply` — downloadable from the done view
(**⬇ 下載 tsdf_post.ply**).

---

## Concurrency

`MAX_JOBS` asyncio workers pull from one queue; extra jobs queue and backfill. The
history view shows `同時執行 X/N`. GPU jobs (訓練 / Mesh) pick the GPU in their form
field — **no auto-assignment**, so spread heavy jobs across cards yourself (#0 / #1).

```bash
COLMAP_PANEL_MAX_JOBS=4   # how many jobs run at once (default 4)
```

---

## Run

```bash
./run.sh                 # conda env rec; open http://127.0.0.1:8077
```

`run.sh` auto-detects sensible defaults; override via the environment or a **`local.env`**
file (copy `local.env.example` → `local.env`; gitignored). **Backend changes (`app.py` /
`pipeline/` / `jobs.py`) need a restart; template (`templates/`) changes just need a refresh.**

| env | default | why |
|-----|---------|-----|
| `COLMAP_BIN` | `colmap` (PATH) | colmap binary; set for non-standard installs |
| `FFMPEG_BIN` | NVDEC build if present, else PATH `ffmpeg` | needs `blurdetect`; NVDEC = GPU decode; also used for FullHD resize |
| `FFMPEG_HWACCEL` | `cuda` | set `none` to force CPU decode |
| `COLMAP_PANEL_DATA` | `/mnt/ssd1/colmap_panel/data` else `~/.colmap_panel` | job state + logs (keep off a small root fs) |
| `TMPDIR` | `$COLMAP_PANEL_DATA/tmp` | keep ffmpeg/colmap scratch off root `/tmp` |
| `COLMAP_PANEL_BROWSE_ROOT` | `/mnt/ssd1` if present, else `/` | directory-picker root |
| `COLMAP_PANEL_MAX_JOBS` | `4` | concurrent jobs (shared across all stages) |
| `COLMAP_PANEL_RESIZE_WORKERS` | CPU count (≤32) | parallel ffmpeg workers for FullHD resize |
| `COLMAP_PANEL_BACKENDS` | `./backends.json` | per-machine training backends file |
| `CONDA_ROOT` / `CONDA_ENV` | `conda info --base` / `rec` | which conda env to launch the panel in |
| `HOST` / `PORT` | `127.0.0.1` / `8077` | bind address |

Remote access (binds to localhost): `ssh -L 8077:127.0.0.1:8077 user@host`, or `HOST=0.0.0.0 ./run.sh`.

---

## Backends

Trainers/mesh tools are declared as data, merged from the built-in defaults
(`pipeline/backends.py`) and a per-machine **`backends.json`** (gitignored; copy from
[`backends.example.json`](backends.example.json)). Adding a trainer on a new machine is
a config entry, not a code change. Each backend specifies its conda env, repo, command
templates (`train_args` / `mesh_args`), and the form param schema.

Interpreter resolution (most portable first): explicit `python` path → `$CONDA_ROOT/envs/<env>/bin/python`
→ derived from the panel's own `sys.prefix` → `conda info --base`. So if the trainer env
names match (e.g. `gs2m`), it works zero-config.

**`/doctor`** — preflight page (also `/api/doctor`). Per backend it checks: env python,
repo/script present, and (deep) imports `torch` + the compiled CUDA submodule (e.g.
`diff_gaussian_rasterization`) inside that env — catching the most common
move-to-new-machine failure (extension built for another GPU arch). Also reports COLMAP
version and GPUs.

### GS-2M — install & wiring

GS-2M ([github.com/ndming/GS-2M](https://github.com/ndming/GS-2M)) is the built-in
training + mesh backend. Recon Studio expects it as a **sibling of this repo** (`../GS-2M`)
in a conda env named **`gs2m`**.

**1) Get the code** (as a sibling of the panel repo):
```bash
cd /home/will/repo            # the dir that holds the panel repo; GS-2M sits alongside it
git clone https://github.com/ndming/GS-2M.git
```

**2) Create the env + build the CUDA submodules** (needs a C++ compiler; CUDA 12.8 is
fetched into the env). This step is GPU/arch-specific and must be redone per machine:
```bash
cd GS-2M
conda env create --file environment.yml   # -> env "gs2m": python 3.10, pytorch 2.7, cuda-toolkit 12.8
conda activate gs2m
pip install -r requirements.txt           # python deps + builds the 5 submodules:
                                          # diff-gaussian-rasterization, simple-knn,
                                          # fused-ssim, nvdiffrast, render-utils
```

**3) Wiring — usually nothing to do.** The built-in backend in `pipeline/backends.py`
already maps it; if the env name and sibling path match, it works zero-config:

| backend `gs2m` | value |
|---|---|
| `conda_env` | `gs2m` → env python = `<conda envs>/gs2m/bin/python` |
| `repo` | `../GS-2M` (relative to the panel dir) |
| train | `python train.py -s <scene> -m <model> <params>` |
| mesh | `python render.py -m <model> --extract_mesh --skip_test <params>` |

The `<scene>` is built automatically from your COLMAP workspace (`<model>_scene/` with
`sparse/0` + `images` symlinks; PINHOLE only). The envs dir is found via `$CONDA_ROOT`
or the panel's own `sys.prefix`.

**Override** only if the env name or repo location differs — add to `backends.json`
(copy from `backends.example.json`); keys shallow-merge over the built-in:
```json
{
  "gs2m": {
    "conda_env": "gs2m",
    "repo": "/abs/path/to/GS-2M",
    "python": "/opt/conda/envs/gs2m/bin/python"
  }
}
```

**4) Verify** at `/doctor` — the `gs2m` row should be all green: env python ✓, repo/`train.py` ✓,
`torch` + CUDA ✓, `diff_gaussian_rasterization` import ✓. Then it appears (un-greyed) in the
訓練 / Mesh backend selectors.

### Deploy to another machine
The panel is torch-free, so deploying it is trivial; the heavy CUDA env is a per-machine
prerequisite that backends merely locate and `/doctor` verifies:

```bash
# 0) get Recon Studio
git clone https://github.com/KuoFengYuan/reconstudio.git
cd reconstudio
# 1) panel (lightweight; conda env name is configurable via CONDA_ENV, default rec)
conda create -n rec python=3.10 -y
conda run -n rec pip install -r requirements.txt
# 2) prerequisites: install colmap (+ ffmpeg with blurdetect), and set up the trainer
#    env(s) — for GS-2M follow "GS-2M — install & wiring" above (env create + pip build)
# 3) per-machine config (optional if names/paths match)
cp local.env.example local.env          # paths/ports
cp backends.example.json backends.json  # trainer envs/repos
# 4) verify, then run
./run.sh   # open /doctor, fix anything red, then use the panel
```

---

## 操作教學 (workflow)

**啟動**：`./run.sh` → 瀏覽器開 `http://127.0.0.1:8077`（遠端用上面的 SSH tunnel）。

**① 抽幀**：選影片資料夾 → `out_dir` 自動帶出 → 設 `fps` 與去模糊（百分位 / 閾值）→
**▶ 抽幀 + 去模糊** → 完成後 **▶ 接著跑 COLMAP**（自動帶入路徑、layout 自動偵測）。

**② COLMAP**：設 `image_root` / `workspace`；**影像解析度預設 FullHD**（4K 會先等比例縮成
FullHD 實體檔再進 COLMAP）；進階區可調 Camera / Matching / Mapping。**▶ 啟動 COLMAP** →
log 顯示 layout、各階段 banner、stage stepper。完成後可 **🧊 檢視 3D 結果** 或 **🧠 接著訓練**。

**③ 訓練**：選 backend（環境未就緒會灰掉，旁邊有 `/doctor` 連結）、`source`（COLMAP workspace）、
`model_path`（輸出）、**GPU**（手動選 #0 / #1）。參數依 backend schema 顯示，材質/遮罩收在摺疊區。
**▶ 啟動訓練** → 右側狀態列顯示階段（載入相機 / 訓練中 / 存檔…）與 `iter N/total`、loss；
log 的進度條原地更新不洗版。完成後 **🧩 接著抽 Mesh**。

**④ Mesh**：選 backend（僅支援者出現）、`model_path`、GPU、TSDF 參數 → **▶ 抽取 Mesh** →
完成後 **⬇ 下載 tsdf_post.ply**。

**檢視 3D 結果**（COLMAP 完成後內嵌右側，**← 回 log** 返回）：拖曳=360°旋轉、滾輪=縮放、
右鍵=平移；**雙擊相機**看影像名/分數/縮圖並高亮；座標 gizmo、旋轉物體對齊、點/相機大小滑桿；
品質指標 `reproj err`（<1 綠 / <2 黃）、`track len`。

**歷史**：狀態每 3 秒自動更新（保留勾選與捲動），勾選 → **🗑 刪除選取**（進行中先取消）。

---

## How it works

```
form ─POST /ui/{frames,jobs,train,mesh}─► JobManager (asyncio queue, N=MAX_JOBS workers)
                                             │  run_{frames,colmap,train,mesh} in a thread
                                             ▼  (Runner shells out; train/mesh into the backend's conda env)
                                         console.log ─SSE─► browser <pre>  (tqdm bar updates in-place)
                                               │
        frames: "[j/N] …" / "K kept / D dropped"   colmap: "=== feature_extractor …" / "skip <stage>"
        train:  "[ITER n]" / "Training:" / phase    mesh: "TSDF config" / "Num vertices post"  ─► progress / stepper
```

- **Job state** persists under `COLMAP_PANEL_DATA/jobs/<id>/` (`job.json` + `console.log`);
  jobs left running when the server stops are marked failed on reload.
- **Cancel** sets a flag and `SIGTERM`s every child process group the Runner started.
- **Idempotency**: COLMAP & resize use sentinels / output checks — tick `FORCE` to re-run.
- **Torch-free panel**: trainers run via the backend env's absolute `python` (no `conda activate`).

## Files

| file | role |
|------|------|
| `app.py` | FastAPI routes: page, htmx `/ui/*`, JSON `/api/*`, SSE logs, `/viz/<id>`, `/doctor`, mesh.ply download |
| `jobs.py` | `JobManager` (queue, N workers, cancel/delete) + per-kind log parsers + `MAX_JOBS` |
| `pipeline/runner.py` | log emitter + subprocess streaming + cancel (multi-child) |
| `pipeline/frames.py` | `run_frames`: fps extraction (NVDEC + CPU fallback), blur cutoff, parallel, flatten |
| `pipeline/colmap.py` | `run_colmap`: layout detect, stages, sentinels, NESTED staging, **parallel Lanczos FullHD resize** |
| `pipeline/train.py` | `run_train` (COLMAP→trainer scene + PINHOLE guard) and `run_mesh` (mesh extraction) |
| `pipeline/backends.py` | backend registry, env/GPU resolution, CLI builder, `doctor` preflight |
| `pipeline/model.py` | parse sparse model (poses/cameras/points/per-image score) + PLY export |
| `templates/` | `index.html` (4 forms) + htmx partials + `viz.html` (three.js) + `doctor.html` |
| `backends.example.json` | template for per-machine `backends.json` (gitignored) |
| `run.sh` + `local.env.example` | launcher with auto-detected defaults; per-machine overrides |
