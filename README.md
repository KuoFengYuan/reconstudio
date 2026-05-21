# Recon Studio

A local web panel for the full reconstruction pipeline — **video → 清晰幀 → COLMAP →
3D Gaussian Splatting 訓練 → Mesh** — implemented as pure-Python modules under
[`pipeline/`](pipeline/). The panel itself is **torch-free**: heavy tools (`ffmpeg`,
`colmap`) and trainers (GS-2M, …) are invoked as **subprocesses**, so progress,
cancellation, live logs, in-browser 3D viewers (point cloud **and** mesh, with a
mm ruler), and mesh downloads are first-class.

Single-machine local tool, binds to `127.0.0.1` by default. Runs up to **`MAX_JOBS`
(default 4) jobs concurrently**; GPU is chosen manually per job.

```
 ① 抽幀                  ② COLMAP                ③ 訓練                  ④ Mesh
 video ──run_frames──► images ──run_colmap──► sparse + 去畸變dense ──run_train──► 3DGS model ──run_mesh──► tsdf_post.ply
 (NVDEC, 並行)         (+FullHD等比縮放)        (GS-2M, 自家 conda env)            (render.py)   ⬇ 可下載
```

Each finished stage offers a one-click button to prefill the next (frames→COLMAP→訓練→Mesh).
訓練完還可選擇在瀏覽器內用內嵌的 **SuperSplat** 編輯器去背(刪掉背景 splat),一鍵送回後對乾淨
點雲抽 mesh — **非破壞性**,原模型保留(見 [🧹 去背](#-去背-background-removal-optional--supersplat))。
The Mesh stage can optionally rescale the output to **real-world millimetres** from a
ChArUco marker board, and ships an **in-browser mesh viewer with a mm ruler**.

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

> **其他後端 (選用):LichtFeld Studio（MR-NF / iGS+）** — 編譯型 C++ trainer,以
> `"launch": "binary"` 後端宣告(直接跑 binary,不經 conda;見 [Backends](#backends)）。
> 策略預設來自 `--config`（`configs/lichtfeld/*.json`，可調),GUI 露出精選常調項
> （`--headless`/`--no-splash` 由後端自動帶入;MR-NF 另有 `--use-error-map`/`--use-edge-map`，預設開、可關）。
> **這兩個 backend 不支援 mesh**(不宣告 `mesh_args` → 無 Mesh 按鈕),但訓練完一樣能
> **🧹 在 SuperSplat 去背景**（送回後改為「下載乾淨點雲」,不抽 mesh）。

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

### 🧹 去背 (background removal, optional) — SuperSplat
訓練完成面板的 **🧹 在 SuperSplat 去背景** 會在右側內嵌一個自架的
[SuperSplat](https://github.com/playcanvas/supersplat) 編輯器（`static/supersplat/`，MIT），
載入訓練好的雲（GS-2M `point_cloud/iteration_*/point_cloud.ply`、LichtFeld `splat_*.ply`）。
常用流程：框選物件 → **Ctrl+I** 反選 → **Delete** 刪背景（誤刪 **Ctrl+Z** 還原）→ **✅ 送回去背點雲**。

送回時瀏覽器把去背後的雲序列化成 PLY，POST 到 `/api/jobs/<id>/edited_ply`。**原模型完全不動**；
依後端分流(`/api/doctor` 看得到哪個有 mesh)，輸出落點如下：

| 後端 | 送回後動作 | 去背雲輸出路徑 |
|------|-----------|---------------|
| **GS-2M**（有 mesh） | 衍生非破壞性兄弟目錄 + 自動帶入 Mesh 表單 | `<model>_edited_<時間>/point_cloud/iteration_<N>/point_cloud.ply`（symlink 原 `cfg_args`）;重抽的 mesh → `<model>_edited_<時間>/train/ours_<N>/mesh/tsdf_post.ply` |
| **LichtFeld**（無 mesh） | 存檔 + 瀏覽器下載 `cleaned.ply` | `<model>/edited/cleaned_<時間>.ply` |

GS-2M 因此可用乾淨的雲重抽 mesh：`render.py -m <兄弟目錄>` 從原 `source_path`（記在 `cfg_args`）
讀相機、把結果寫進兄弟目錄;去背搞砸就重來，或把 Mesh 的去背欄位留空用原始點雲。

- **去背 ≠ 即時更新 mesh**：編輯只改點雲，要重跑 Mesh 階段才會產出乾淨 mesh（僅 GS-2M）。
- **GS-2M / 後端皆不需改動**：靠兄弟目錄 + symlink，trainer 零修改就能吃去背後的雲。
- **build**：`static/supersplat/` 是建置產物（已 gitignore）。第一次或更新版本時跑
  `./tools/build_supersplat.sh`（需要 node ≥18 + npm + git；會 clone 釘版 SuperSplat、套
  `tools/supersplat-reconstudio.patch`〔滾輪縮放修正 + 送回 API〕、用 `BASE_HREF=/static/supersplat/`
  build、部署並去掉 sourcemap）。

### ④ Mesh — model → triangle mesh
Backend-specific (only backends that declare `mesh_args`, e.g. GS-2M's
`render.py --extract_mesh --skip_test`). Tunable: `--mesh_only` (加速), `--auto_voxel`
(推薦), `voxel_size` (預設 `0.006`，勾自動時停用) / `sdf_trunc` (留空 = 4×voxel) /
`max_depth` / `num_clusters` / `filter_depth`. Output
`…/train/ours_<iter>/mesh/tsdf_post.ply`.

**實際尺寸 (mm) — 選用 ChArUco marker**：拍攝時在場景放一塊標定板，Mesh 表單勾
**提供 marker** 即可。抽完 mesh 後會自動偵測標定板、估算 recon→mm 尺度
(`tools/estimate_marker_scale.py`)，再把 mesh 縮放成實際毫米
(`tools/scale_mesh.py` → `tsdf_post_scaled_mm.ply`)。板子規格寫在後端的
`marker_defaults`（預設 9×6 格、方格 28.806mm、marker 21.12mm、`DICT_5X5_100`），
可在 `backends.json` 覆寫，**不需在介面手填**。

完成後：**🧊 檢視 Mesh**（內嵌 3D 檢視器）、**⬇ 非實際尺寸 mesh**（recon 單位）、
**⬇ 實際尺寸 mesh (mm)**（有 marker 時）。

**🧊 線上 Mesh 檢視器** (`/viz/mesh/<id>`)：打光的實體 mesh，360° 旋轉/縮放/平移；
切換 **實際尺寸 (mm) / 原始 (recon)**（各顯示原生單位）；**📏 量尺**（單擊兩點量
距離，單位隨版本 mm 或 units）；亮度滑桿、白底、線框、頂點色開關、旋轉對齊。

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
**去背編輯器**：`static/supersplat/` 是建置產物，第一次要先 `./tools/build_supersplat.sh`
（node + npm；見 [🧹 去背](#-去背-background-removal-optional--supersplat)）。

| env | default | why |
|-----|---------|-----|
| `COLMAP_BIN` | `colmap` (PATH) | colmap binary; set for non-standard installs |
| `FFMPEG_BIN` | NVDEC build if present, else PATH `ffmpeg` | needs `blurdetect`; NVDEC = GPU decode; also used for FullHD resize |
| `FFMPEG_HWACCEL` | `cuda` | set `none` to force CPU decode |
| `RECON_STUDIO_DATA` | `/mnt/ssd1/recon_studio/data` else `~/.recon_studio` | job state + logs (keep off a small root fs) |
| `TMPDIR` | `$RECON_STUDIO_DATA/tmp` | keep ffmpeg/colmap scratch off root `/tmp` |
| `RECON_STUDIO_BROWSE_ROOT` | `/mnt/ssd1` if present, else `/` | directory-picker root |
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

**Compiled (non-Python) trainers** — set `"launch": "binary"` + `"exec"` (path to the built
executable) instead of `conda_env`/`repo`. The panel runs the binary directly (no conda/torch);
readiness just checks it's executable. A `"config"` (path, resolved relative to the panel)
is passed as `--config` for defaults, and the `params` schema overrides via CLI flags. Example:
**LichtFeld Studio** (`lichtfeld-mrnf` / `lichtfeld-igs+` in `backends.example.json`) — declares
no `mesh_args`, so it's training-only; 去背 still works (送回 → 下載乾淨點雲). Build it per-machine
([wiki](https://github.com/MrNeRF/LichtFeld-Studio/wiki/)) and point `"exec"` at `build/LichtFeld-Studio`.
The panel auto-adds `--headless --no-splash`; if the binary can't find its shared libs when
launched from the panel, set `LD_LIBRARY_PATH` in `local.env` (see `local.env.example`).

**`/doctor`** — preflight page (also `/api/doctor`). Per backend it checks: env python,
repo/script present, and (deep) imports `torch` + the compiled CUDA submodule (e.g.
`diff_gaussian_rasterization`) inside that env — catching the most common
move-to-new-machine failure (extension built for another GPU arch). For `"launch": "binary"`
backends it instead reports whether the `exec` exists and is executable. Also reports COLMAP
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

> **實際尺寸 (mm) 量測（選用）** 用 panel 的 `tools/` 腳本，但在這個 trainer env 內執行，
> 需要 `opencv-contrib-python`（`cv2.aruco`）、`open3d`、`plyfile`、`scipy`
> （`open3d`/`scipy` 已是 GS-2M 依賴）。缺的話補裝：
> ```bash
> conda run -n gs2m pip install opencv-contrib-python plyfile
> ```

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

### LichtFeld Studio — install & wiring (選用;MR-NF / iGS+,訓練 only)

LichtFeld Studio ([github.com/MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio),
**GPL-3.0**) is a compiled C++/CUDA trainer. Unlike GS-2M it runs as a **built binary**
(no conda env), exposing two strategies — **MR-NF** and **iGS+**. It declares no `mesh_args`,
so it's **training-only**; 去背 still works (送回 → 下載乾淨點雲). Recon Studio expects it as a
**sibling of this repo** (`../LichtFeld-Studio`).

**1) Get the code** (as a sibling of the panel repo):
```bash
cd /home/will/repo            # the dir that holds the panel repo
git clone https://github.com/MrNeRF/LichtFeld-Studio.git
```

**2) Build from source** (GPU/arch-specific; per-machine). Needs **CUDA Toolkit 12.8+**, a recent
NVIDIA driver, and **vcpkg** (`VCPKG_ROOT` set). Authoritative steps:
[Wiki](https://github.com/MrNeRF/LichtFeld-Studio/wiki/) + `LichtFeld-Studio/docs/building_and_distribution.md`.
Gist on Ubuntu:
```bash
sudo apt install git curl unzip cmake gcc-14 g++-14 ccache ninja-build zip tar pkg-config python3 python3-dev
cd LichtFeld-Studio
cmake -B build                 # configures (vcpkg pulls deps; first run is slow)
cmake --build build -j"$(nproc)"
./build/LichtFeld-Studio --help # sanity check -> the binary lives at build/LichtFeld-Studio
```
> The default build's binary needs CUDA + vcpkg present at runtime. A self-contained `dist/`
> (`-DBUILD_PORTABLE=ON` + `cmake --install`) is also possible — see the build doc.

**3) Wiring** — the two backends are pre-declared in [`backends.example.json`](backends.example.json)
(copy to `backends.json`); per machine you only set the binary path:

| backend `lichtfeld-mrnf` / `lichtfeld-igs+` | value |
|---|---|
| `launch` | `binary` (run the executable directly, no conda/torch) |
| `exec` | abs path to `…/LichtFeld-Studio/build/LichtFeld-Studio` |
| `config` | `configs/lichtfeld/{mrnf,igsplus}.json` (策略預設,shipped + 可調) |
| train | `LichtFeld-Studio -d <colmap> -o <model> --strategy {mrnf\|igs+} --headless --no-splash --config … <params>` |
| mesh | — (不支援) |

`<colmap>` is your COLMAP workspace's undistorted PINHOLE dir (`sparse/` + `images/`), resolved
automatically. `--headless --no-splash` are auto-added. If the binary can't find its shared libs
when launched from the panel, set `LD_LIBRARY_PATH` in `local.env` (see `local.env.example`).

**4) Verify** at `/doctor` — each `lichtfeld-*` row should show `launch: binary`, `exec_ok ✓`,
`ready ✓`. Then they appear (un-greyed) in the **訓練** selector (not Mesh).

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
#    env(s) — for GS-2M follow "GS-2M — install & wiring" above (env create + pip build);
#    for LichtFeld (optional) follow "LichtFeld Studio — install & wiring" (cmake build)
# 2b) (optional) build the in-browser 去背 editor into static/ (needs node>=18 + npm + git)
./tools/build_supersplat.sh
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
log 的進度條原地更新不洗版。完成後 **🧩 接著抽 Mesh**（或先 **🧹 在 SuperSplat 去背景**）。

**🧹 去背（選用）**：訓練完成面板按 **🧹 在 SuperSplat 去背景** → 右側內嵌編輯器載入訓練雲。
**滾輪=縮放、左鍵拖=旋轉、右鍵=平移**；框選物件 → **Ctrl+I** 反選 → **Delete** 刪背景
（**Ctrl+Z** 還原）→ **✅ 送回去背點雲 → 抽 Mesh**（自動帶入 `_edited_` 路徑到 Mesh 表單，原模型不動）。

**④ Mesh**：選 backend（僅支援者出現）、`model_path`、GPU、TSDF 參數（`voxel_size` 預設 0.006，
勾「自動體素大小」會自動估）。要實際尺寸就勾 **提供 marker**（板子規格已在 `backends.json`，免手填）→
**▶ 抽取 Mesh** → 完成後 **🧊 檢視 Mesh**、**⬇ 非實際尺寸 mesh**、**⬇ 實際尺寸 mesh (mm)**。

**檢視 3D 結果**（COLMAP 完成後內嵌右側，**← 回 log** 返回）：拖曳=360°旋轉、滾輪=縮放、
右鍵=平移；**雙擊相機**看影像名/分數/縮圖並高亮；座標 gizmo、旋轉物體對齊、點/相機大小滑桿；
品質指標 `reproj err`（<1 綠 / <2 黃）、`track len`。

**檢視 Mesh**（Mesh 完成後 **🧊 檢視 Mesh** 內嵌右側）：打光的實體 mesh，可切
**實際尺寸 (mm) / 原始 (recon)**、用 **📏 量尺** 點兩點量距離（單位 mm/units）、
調亮度與白底、切線框 / 頂點色（關＝純色看幾何）。

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

- **Job state** persists under `RECON_STUDIO_DATA/jobs/<id>/` (`job.json` + `console.log`);
  jobs left running when the server stops are marked failed on reload.
- **Cancel** sets a flag and `SIGTERM`s every child process group the Runner started.
- **Idempotency**: COLMAP & resize use sentinels / output checks — tick `FORCE` to re-run.
- **Torch-free panel**: trainers run via the backend env's absolute `python` (no `conda activate`).

## Files

| file | role |
|------|------|
| `app.py` | FastAPI routes: page, htmx `/ui/*`, JSON `/api/*`, SSE logs, `/viz/<id>` + `/viz/mesh/<id>` (mesh viewer), `/doctor`, `mesh.ply` / `mesh_scaled.ply` downloads |
| `jobs.py` | `JobManager` (queue, N workers, cancel/delete) + per-kind log parsers + `MAX_JOBS` |
| `pipeline/runner.py` | log emitter + subprocess streaming + cancel (multi-child) |
| `pipeline/frames.py` | `run_frames`: fps extraction (NVDEC + CPU fallback), blur cutoff, parallel, flatten |
| `pipeline/colmap.py` | `run_colmap`: layout detect, stages, sentinels, NESTED staging, **parallel Lanczos FullHD resize** |
| `pipeline/train.py` | `run_train` (COLMAP→trainer scene + PINHOLE guard) and `run_mesh` (mesh extraction + optional ChArUco marker → mm scaling) |
| `pipeline/backends.py` | backend registry, env/GPU resolution, CLI builder, `doctor` preflight |
| `pipeline/model.py` | parse sparse model (poses/cameras/points/per-image score) + PLY export |
| `templates/` | `index.html` (4 forms) + htmx partials + `viz.html` / `mesh_viz.html` (three.js viewers) + `doctor.html` |
| `tools/` | panel-owned scripts run in the trainer env: `estimate_marker_scale.py` (ChArUco → recon→mm scale), `scale_mesh.py` (scale mesh to mm), vendored `colmap_read_write_model.py` |
| `backends.example.json` | template for per-machine `backends.json` (gitignored) |
| `run.sh` + `local.env.example` | launcher with auto-detected defaults; per-machine overrides |

## License

Recon Studio's own code is released for **non-commercial, research and evaluation
use** ([`LICENSE`](LICENSE)), mirroring its core trainer dependency. It integrates
third-party tools under their own licenses ([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)) —
notably **GS-2M / 3D Gaussian Splatting (Inria, non-commercial)**, COLMAP (BSD),
and FFmpeg (LGPL/GPL). The training & mesh stages are therefore non-commercial;
for commercial use of the Gaussian-Splatting technology, contact Inria.
