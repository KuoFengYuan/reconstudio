# Recon Studio

本機網頁面板,跑完整重建流程 **影片 → 清晰幀 → COLMAP → 3DGS 訓練 → Mesh**。面板本身
**不含 torch**:`ffmpeg` / `colmap` / 訓練器(GS-2M…)都以**子行程**呼叫,所以進度、取消、
即時 log、瀏覽器內 3D/Mesh 檢視器(含 mm 量尺)、Mesh 下載都是內建。單機工具,預設綁
`127.0.0.1`,同時最多跑 `MAX_JOBS`(預設 4)個 job,GPU 每個 job 手動指定。

```
 ① 抽幀                  ② COLMAP                ③ 訓練                  ④ Mesh
 video ──run_frames──► images ──run_colmap──► sparse + 去畸變dense ──run_train──► 3DGS model ──run_mesh──► tsdf_post.ply
 (NVDEC, 並行)         (+FullHD等比縮放)        (GS-2M, 自家 conda env)            (render.py)   ⬇ 可下載
```

每個階段完成後有一鍵帶入下一步的按鈕。訓練完可選擇在瀏覽器內用內嵌 **SuperSplat** 去背
(刪背景 splat)再對乾淨點雲抽 mesh —— **非破壞性**,原模型保留(見 [🧹 去背](#-去背-background-removal-optional--supersplat))。
Mesh 階段可選用 ChArUco 標定板把輸出縮放成**實際毫米**,並附 mm 量尺的 mesh 檢視器。

---

## 🚀 快速開始(新手 5 步)

> 第一次部署照這 5 步就能跑起來,每步的細節在下方對應章節。面板本身很輕(不含 torch);
> 重的是 GPU 訓練環境 —— 那是每台機器各自要裝的前置,裝好後 `/doctor` 會幫你檢查。

```bash
# 1) 抓專案 + 裝面板（輕量）
git clone https://github.com/KuoFengYuan/reconstudio.git
cd reconstudio
conda create -n rec python=3.10 -y
conda run -n rec pip install -r requirements.txt

# 2) 裝外部工具：colmap 與 ffmpeg（ffmpeg 需含 blurdetect）

# 3) 裝訓練後端（需要 GPU）：GS-2M  ← 見下方「GS-2M — install & wiring」
#    LichtFeld Studio、瀏覽器去背、從 GCS 下載 都是選用，需要再裝

# 4) 啟動
./run.sh          # 然後開瀏覽器 http://127.0.0.1:8077
```

**5) 開 `/doctor` 頁面,把紅燈修成綠燈**(它會檢查 colmap、各後端的 env + CUDA)。全綠後,
就照 **① 抽幀 → ② COLMAP → ③ 訓練 → ④ Mesh** 的順序操作(見 [Stages](#stages))。

> 需要時再看:[per-machine 設定 (local.env)](#run) · [GS-2M 安裝](#gs-2m--install--wiring) ·
> 從雲端拉資料見下方「☁️ 從 GCS 下載資料」 · [部署到另一台機器(完整 checklist)](#deploy-to-another-machine)

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
Layout auto-detected (or forced via `layout`); a workspace nested in `image_root` is
ignored when detecting groups:

```
single  XXX/*.jpg                -> 1 shared camera (also when the root holds images)
multi   ROOT/<group>/*.jpg       -> 1 camera per group   (CAMERA_MODE=per_folder)
nested  ROOT/<group>/<vid>/*.jpg -> staged, 1 camera per group   (① feeds this)
```

**影像解析度 (預設 FullHD)** — 把每張輸入**實體**等比縮成長邊 ≤1920 的副本(存
`workspace/images_fullhd/`),整條 COLMAP 跑這些縮圖;Lanczos + 高品質編碼使 4K→FullHD 仍銳利,
可續跑/可取消。選「保持原樣」則不縮。

**GPS / 大場景** — 讀輸入的 EXIF GPS(影片幀無 GPS)。勾任一 GPS 選項需**每張**都有 GPS,
開跑前會檢查 100% 覆蓋;FullHD 縮圖會把原圖 GPS 接回(否則 ffmpeg 重編會洗掉)。可解鎖:

| 選項 | COLMAP 指令 | 作用 | 主要參數 |
|------|------------|------|---------|
| `MATCHER=spatial` | `spatial_matcher` | 用 GPS 鄰近度限制比對候選,大場景比 vocab/sequential 快且穩 | `SPATIAL_MAX_NEIGHBORS` · `SPATIAL_MAX_DISTANCE`(m) · `SPATIAL_IGNORE_Z` |
| `MAPPER=pose_prior` | `pose_prior_mapper` | GPS 先驗進 BA,抗漂移、輸出**直接公制+地理對齊** | `PRIOR_STD_X/Y/Z`(GPS 精度 m;消費級 3~5、RTK ~0.02) |
| `GPS_ALIGN`(選填) | `model_aligner` | 事後把稀疏模型對齊到 ENU 公尺 | `GPS_ALIGN_MAX_ERROR`(m) |

> `GPS_ALIGN` 與 Mesh 的 ChArUco mm 校正擇一;用 `pose_prior` 已對齊則免。

**Output** (workspace):`database.db`、`sparse/0/{cameras,images,points3D}.bin`、
`<dataset>_<mapper>_mapper/`(去畸變 **PINHOLE** dense,訓練吃這個)、`sparse/points.ply`(viewer 快取)。

### ③ 訓練 (training) — reconstruction → 3DGS model
Runs a Gaussian-splatting trainer in **its own conda env** as a subprocess. Backends
are **data, not code** (see [Backends](#backends)); built-in default is **GS-2M**。

> **其他後端 (選用):LichtFeld Studio（MR-NF / iGS+）** — 編譯型 C++ trainer,以
> `"launch": "binary"` 後端宣告(直接跑 binary,不經 conda)。**不支援 mesh**,但訓練完一樣能
> **🧹 在 SuperSplat 去背景**(送回後改為「下載乾淨點雲」)。見 [Backends](#backends)。

- **Scene adaptation**:COLMAP workspace 以非破壞性的 `<model>_scene/`(symlink)給訓練器;
  只用去畸變的 **PINHOLE** 模型,遇到 OPENCV(distorted)會在開跑前直接報錯(最常見的接錯點)。
- **可調參數**由 backend schema 動態產生(非寫死):迭代數、解析度、`--material`(PBR 材質)、
  法線正則、前景遮罩…各欄位附 hint;另有自由 `extra` 欄位塞未列出的旗標。
- **GPU**:每個 job 手動選 → `CUDA_VISIBLE_DEVICES`(**無自動分配**,重任務請自己分散到不同卡)。

### 🧹 去背 (background removal, optional) — SuperSplat
訓練完按 **🧹 在 SuperSplat 去背景**,右側內嵌 [SuperSplat](https://github.com/playcanvas/supersplat)
編輯器(`static/supersplat/`,MIT)載入訓練雲。框選物件 → **Ctrl+I** 反選 → **Delete** 刪背景
(**Ctrl+Z** 還原)→ **✅ 送回去背點雲**。**原模型完全不動**,依後端分流輸出:

| 後端 | 送回後 | 去背雲輸出 |
|------|-------|-----------|
| **GS-2M**(有 mesh) | 衍生兄弟目錄 + 自動帶入 Mesh 表單 | `<model>_edited_<時間>/…/point_cloud.ply`(可用乾淨雲重抽 mesh) |
| **LichtFeld**(無 mesh) | 存檔 + 瀏覽器下載 | `<model>/edited/cleaned_<時間>.ply` |

> 去背只改點雲,要乾淨 mesh 需重跑 Mesh 階段(僅 GS-2M);靠兄弟目錄 + symlink,trainer 零修改。
> `static/supersplat/` 是建置產物(已 gitignore),第一次先跑 `./tools/build_supersplat.sh`(node ≥18 + npm + git)。

### ④ Mesh — model → triangle mesh
Backend-specific(只有宣告 `mesh_args` 的後端,如 GS-2M 的 `render.py --extract_mesh`)。可調:
`--mesh_only`(加速)、`--auto_voxel`(推薦)、`voxel_size`(預設 `0.006`)/ `sdf_trunc` / `max_depth` /
`num_clusters` / `filter_depth`。輸出 `…/train/ours_<iter>/mesh/tsdf_post.ply`。

**實際尺寸 (mm) — 選用 ChArUco marker**:拍攝時在場景放標定板,Mesh 表單勾 **提供 marker** 即可。
抽完 mesh 後自動偵測標定板、估 recon→mm 尺度,再縮放成毫米(`tsdf_post_scaled_mm.ply`)。板子規格寫在
後端的 `marker_defaults`(可在 `backends.json` 覆寫),**不需在介面手填**。

完成後:**🧊 檢視 Mesh**(內嵌 3D 檢視器,可切 mm / recon、**📏 量尺** 量距離、亮度/白底/線框/頂點色)、
**⬇ 非實際尺寸 mesh**、**⬇ 實際尺寸 mesh (mm)**(有 marker 時)。

---

## Run

```bash
./run.sh                 # conda env rec; open http://127.0.0.1:8077
```

`run.sh` auto-detects sensible defaults; override via the environment or a **`local.env`**
file (copy `local.env.example` → `local.env`; gitignored). **改 `app.py` / `pipeline/` / `jobs.py`
需重啟;改 `templates/` 只需重整頁面。**

> **絕大多數變數都會自動偵測或有可用預設,留空即可。** 真正**必須手動設定**(沒有可用預設)的
> 只有一個:用到 **GCS bucket 瀏覽**時的 `CLOUDSDK_CORE_PROJECT`(不設,`gsutil ls` 會報
> 「requires a project id」)。其餘只在**非標準安裝**或想**調效能 / 換落地磁碟**時才需要動。

| env | default | 手動? | why |
|-----|---------|------|-----|
| `CLOUDSDK_CORE_PROJECT` | — (無預設) | **用 GCS 時必填** | GCS 瀏覽器 `gsutil ls` 的預設 GCP project;被 gsutil 子行程繼承(`gcloud config set project …` 亦可) |
| `COLMAP_BIN` | `colmap` (PATH) | 不在 PATH 才設 | colmap binary |
| `FFMPEG_BIN` | NVDEC build if present, else PATH `ffmpeg` | 不在 PATH / 缺 `blurdetect` 才設 | needs `blurdetect`; NVDEC = GPU decode; also FullHD resize |
| `CONDA_ROOT` / `CONDA_ENV` | `conda info --base` / `rec` | 偵測不到才設 | which conda env to launch the panel in |
| `RECON_STUDIO_DATA` | `/mnt/ssd1/recon_studio/data` else `~/.recon_studio` | 建議設 | job state + logs (keep off a small root fs) |
| `RECON_STUDIO_DEST_ROOT` | `/` | 建議設 | local root that GCS downloads land under |
| `GSUTIL_BIN` | `gsutil` (PATH) | 不在 PATH 才設 | gsutil binary (GCS browse / download) |
| `RECON_STUDIO_GCS_ROOT` | (空 = 列全部 bucket) | 選填 | GCS 瀏覽器起始 `gs://` 前綴 |
| `RECON_STUDIO_BROWSE_ROOT` | `/mnt/ssd1` if present, else `/` | 選填 | directory-picker root (local file browser) |
| `FFMPEG_HWACCEL` | `cuda` | 選填 | set `none` to force CPU decode |
| `TMPDIR` | `$RECON_STUDIO_DATA/tmp` | 自動 | keep ffmpeg/colmap scratch off root `/tmp` |
| `COLMAP_PANEL_MAX_JOBS` | `4` | 選填(調效能) | concurrent jobs (shared across all stages) |
| `COLMAP_PANEL_RESIZE_WORKERS` | CPU count (≤32) | 選填(調效能) | parallel ffmpeg workers for FullHD resize |
| `COLMAP_PANEL_BACKENDS` | `./backends.json` | 選填 | per-machine training backends file |
| `HOST` / `PORT` | `127.0.0.1` / `8077` | 選填 | bind address |

> `run.sh` 用 `set -a` 載入 `local.env`,所以每個變數都會匯出到面板、並被它呼叫的外部工具
> (`ffmpeg` / `colmap` / `gsutil`)繼承 —— 這就是為何 `CLOUDSDK_CORE_PROJECT` 寫在 `local.env` 就能讓 `gsutil` 吃到。

Remote access (binds to localhost): `ssh -L 8077:127.0.0.1:8077 user@host`, or `HOST=0.0.0.0 ./run.sh`.

---

## ☁️ 從 GCS 下載資料 (optional)

面板內建 **☁️ GCS** 分頁(「從 GCS 下載到本機」),把 Google Cloud Storage 的資料同步到本機後
再進流程。它**與重建流程解耦** —— 只把 bytes 從 `gs://` 搬到本機;下載完到任一分頁用「瀏覽」
選那個資料夾即可,**不會自動帶你去下一步**。

- **來源**:填 `gs://bucket/path`,或按 **☁️ 瀏覽** 點選 bucket / 資料夾(起點可用 `RECON_STUDIO_GCS_ROOT` 設定)。
- **下載到**:本機路徑;留空 = 自動放到 `RECON_STUDIO_DEST_ROOT` 底下的 `<bucket>/<路徑>`。
- 底層 `gsutil -m rsync -r`:**可續傳、只補差異**。勾 **鏡像模式 `-d`** 會**刪除**本機多餘檔(有風險)。

### 設定 GCS 帳號(一次性)

需要本機裝好 **Google Cloud SDK**(提供 `gcloud` + `gsutil`)並登入:

```bash
# 1) 安裝 Google Cloud SDK：https://cloud.google.com/sdk/docs/install

# 2) 登入 Google 帳號（會開瀏覽器；遠端無頭機改用 --no-launch-browser）
gcloud auth login
#    無人值守 server 也可用 service account：gcloud auth activate-service-account --key-file=KEY.json

# 3) 設定預設 GCP project（⚠ 列 bucket 必需）—— 二選一：
gcloud config set project <YOUR_PROJECT_ID>     # 全域預設
#    或只給面板用，寫進 local.env：CLOUDSDK_CORE_PROJECT=<YOUR_PROJECT_ID>

# 4) 驗證（應列出 buckets）
gsutil ls
```

> 找 `PROJECT_ID`:`gcloud projects list`。相關環境變數見上面 [Run](#run) 的表(只有 `CLOUDSDK_CORE_PROJECT` 必填)。

---

## Backends

Trainers/mesh tools are declared as data, merged from the built-in defaults
(`pipeline/backends.py`) and a per-machine **`backends.json`** (gitignored; copy from
[`backends.example.json`](backends.example.json)). 在新機器加一個訓練器是改設定、不是改程式。
每個 backend 指定它的 conda env、repo、指令樣板(`train_args` / `mesh_args`)、表單參數 schema。

Interpreter resolution (most portable first): explicit `python` path → `$CONDA_ROOT/envs/<env>/bin/python`
→ derived from the panel's own `sys.prefix` → `conda info --base`. 名字對得上(如 `gs2m`)就 zero-config。

**Compiled (non-Python) trainers** — 設 `"launch": "binary"` + `"exec"`(built executable 路徑)取代
`conda_env`/`repo`。面板直接跑 binary(無 conda/torch),readiness 只檢查可執行,`params` schema 變成 CLI 旗標。
例:**LichtFeld Studio**(`backends.example.json` 的 `lichtfeld-mrnf` / `lichtfeld-igs+`)—— 不宣告 `mesh_args`,
training-only;去背仍可用。面板自動帶 `--headless --no-splash`;binary 找不到 shared libs 時在 `local.env` 設 `LD_LIBRARY_PATH`。

**`/doctor`** — preflight 頁(亦 `/api/doctor`)。每個 backend 檢查:env python、repo/script 在不在、
(deep)在該 env 內 import `torch` + 編譯的 CUDA submodule(如 `diff_gaussian_rasterization`)—— 抓最常見的
換機失敗(extension 為別的 GPU arch 編的)。`"launch": "binary"` 後端則報 `exec` 是否可執行。也報 COLMAP 版本與 GPU。

### GS-2M — install & wiring

GS-2M ([github.com/ndming/GS-2M](https://github.com/ndming/GS-2M)) 是內建的訓練 + mesh 後端,
預期放在本 repo 的**兄弟目錄** `../GS-2M`、conda env 名為 **`gs2m`**。

```bash
# 1) 抓 code（與面板 repo 同層）
cd /home/will/repo && git clone https://github.com/ndming/GS-2M.git

# 2) 建 env + 編 CUDA submodules（需 C++ compiler；GPU/arch-specific，每台機器要重做）
cd GS-2M
conda env create --file environment.yml   # -> env "gs2m": python 3.10, pytorch 2.7, cuda-toolkit 12.8
conda activate gs2m
pip install -r requirements.txt           # python deps + 編 5 個 submodule（diff-gaussian-rasterization 等）
```

> **3) Wiring — 通常不用動。** env 名與兄弟路徑對得上就 zero-config(內建 backend 已映射 `conda_env=gs2m`、
> `repo=../GS-2M`、train=`train.py`、mesh=`render.py --extract_mesh`)。env 名或路徑不同才在 `backends.json`
> 覆寫(`conda_env` / `repo` / `python`,shallow-merge)。
>
> **實際尺寸 (mm) 量測(選用)** 的 `tools/` 腳本在此 env 執行,需 `opencv-contrib-python`、`plyfile`
> (`open3d`/`scipy` 已是 GS-2M 依賴):`conda run -n gs2m pip install opencv-contrib-python plyfile`。

**4) Verify** at `/doctor` — `gs2m` 一列全綠(env python ✓、`train.py` ✓、`torch`+CUDA ✓、
`diff_gaussian_rasterization` ✓),就會在 訓練 / Mesh 選單中出現。

### LichtFeld Studio — install & wiring (選用;MR-NF / iGS+,訓練 only)

LichtFeld Studio ([github.com/MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio),
**GPL-3.0**) 是編譯型 C++/CUDA trainer,以**已編譯 binary** 執行(無 conda env),提供 MR-NF / iGS+ 兩種策略。
不宣告 `mesh_args` → **training-only**(去背仍可用)。預期放在兄弟目錄 `../LichtFeld-Studio`。

```bash
# 1) 抓 code（與面板 repo 同層）
cd /home/will/repo && git clone https://github.com/MrNeRF/LichtFeld-Studio.git

# 2) 從原始碼 build（GPU/arch-specific，每台機器；需 CUDA Toolkit 12.8+、新 NVIDIA 驅動、vcpkg）
#    權威步驟見 Wiki + docs/building_and_distribution.md。Ubuntu gist：
sudo apt install git curl unzip cmake gcc-14 g++-14 ccache ninja-build zip tar pkg-config python3 python3-dev
cd LichtFeld-Studio
cmake -B build                  # vcpkg 拉依賴，第一次很慢
cmake --build build -j"$(nproc)"
./build/LichtFeld-Studio --help # binary 在 build/LichtFeld-Studio
```

**3) Wiring** — 兩個 backend 已在 [`backends.example.json`](backends.example.json) 預宣告(複製成
`backends.json`),每台機器只需把 `"exec"` 指到 `…/LichtFeld-Studio/build/LichtFeld-Studio`。`<colmap>`
(去畸變 PINHOLE 的 `sparse/` + `images/`)自動帶入,`--headless --no-splash` 自動加;找不到 shared libs
就在 `local.env` 設 `LD_LIBRARY_PATH`。

**4) Verify** at `/doctor` — 每個 `lichtfeld-*` 顯示 `exec_ok ✓` / `ready ✓`,就會在 **訓練** 選單出現(不在 Mesh)。

### Deploy to another machine
面板 torch-free,部署很簡單;重的 CUDA env 是每台機器的前置,backend 只負責定位、`/doctor` 驗證:

```bash
git clone https://github.com/KuoFengYuan/reconstudio.git && cd reconstudio
# 1) 面板（輕量；env 名可用 CONDA_ENV 改，預設 rec）
conda create -n rec python=3.10 -y
conda run -n rec pip install -r requirements.txt
# 2) 前置：裝 colmap (+ 含 blurdetect 的 ffmpeg)，並依上面章節裝訓練後端
#    GS-2M（env create + pip build）/ LichtFeld（cmake build，選用）
# 2b) （選用）build 瀏覽器去背編輯器到 static/（需 node>=18 + npm + git）
./tools/build_supersplat.sh
# 3) （選用，名字/路徑都對就免）per-machine 設定
cp local.env.example local.env          # 路徑 / ports
cp backends.example.json backends.json  # 訓練器 env / repo
# 4) 驗證後啟動
./run.sh   # 開 /doctor 把紅的修掉，再開始用
```

---

## How it works

```
form ─POST /ui/{frames,jobs,train,mesh}─► JobManager (asyncio queue, N=MAX_JOBS workers)
                                             │  run_{frames,colmap,train,mesh} in a thread
                                             ▼  (Runner shells out; train/mesh into the backend's conda env)
                                         console.log ─SSE─► browser <pre>  (tqdm bar updates in-place)
```

- **Job state** persists under `RECON_STUDIO_DATA/jobs/<id>/` (`job.json` + `console.log`);
  jobs left running when the server stops are marked failed on reload.
- **Cancel** sets a flag and `SIGTERM`s every child process group the Runner started.
- **Idempotency**: COLMAP & resize use sentinels / output checks — tick `FORCE` to re-run.
- **Torch-free panel**: trainers run via the backend env's absolute `python` (no `conda activate`).

## Files

| file | role |
|------|------|
| `app.py` | FastAPI routes: page, htmx `/ui/*`, JSON `/api/*`, SSE logs, `/viz/<id>` + `/viz/mesh/<id>`, `/doctor`, mesh downloads |
| `jobs.py` | `JobManager` (queue, N workers, cancel/delete) + per-kind log parsers + `MAX_JOBS` |
| `pipeline/runner.py` | log emitter + subprocess streaming + cancel (multi-child) |
| `pipeline/frames.py` | `run_frames`: fps extraction (NVDEC + CPU fallback), blur cutoff, parallel, flatten |
| `pipeline/colmap.py` | `run_colmap`: layout detect, stages, sentinels, NESTED staging, parallel Lanczos FullHD resize |
| `pipeline/train.py` | `run_train` (COLMAP→trainer scene + PINHOLE guard) and `run_mesh` (mesh + optional ChArUco mm scaling) |
| `pipeline/backends.py` | backend registry, env/GPU resolution, CLI builder, `doctor` preflight |
| `pipeline/gcs.py` | GCS browse (`gsutil ls`) + download (`gsutil -m rsync -r`) |
| `pipeline/model.py` | parse sparse model (poses/cameras/points/per-image score) + PLY export |
| `templates/` | `index.html` (forms) + htmx partials + three.js viewers + `doctor.html` |
| `tools/` | trainer-env scripts: `estimate_marker_scale.py`、`scale_mesh.py`、vendored `colmap_read_write_model.py` |
| `backends.example.json` / `local.env.example` | per-machine 設定範本(實檔 gitignored) |

## License

Recon Studio's own code is released for **non-commercial, research and evaluation
use** ([`LICENSE`](LICENSE)), mirroring its core trainer dependency. It integrates
third-party tools under their own licenses ([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)) —
notably **GS-2M / 3D Gaussian Splatting (Inria, non-commercial)**, COLMAP (BSD),
and FFmpeg (LGPL/GPL). The training & mesh stages are therefore non-commercial;
for commercial use of the Gaussian-Splatting technology, contact Inria.
