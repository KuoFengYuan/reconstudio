# Recon Studio

本機網頁面板,一條龍跑完整 3D 重建流程:**影片 → 清晰幀 → COLMAP → 3DGS 訓練 → Mesh**。
面板本身 **不含 torch** —— `ffmpeg` / `colmap` / 訓練器(GS-2M…)都以**子行程**呼叫,所以即時
log、取消、瀏覽器內 3D / Mesh 檢視器(含 mm 量尺)、Mesh 下載都是內建。單機工具,預設綁
`127.0.0.1`,同時最多跑 `MAX_JOBS`(預設 4)個 job,GPU 每個 job 手動指定。

```
 ① 抽幀                  ② COLMAP                ③ 訓練                  ④ Mesh
 video ──run_frames──► images ──run_colmap──► sparse + 去畸變dense ──run_train──► 3DGS model ──run_mesh──► tsdf_post.ply
 (NVDEC, 並行)         (+FullHD等比縮放)        (GS-2M, 自家 conda env)            (render.py)   ⬇ 可下載
```

> 三大段:**[一、環境設定](#一環境設定)**(第一次部署看這)→ **[二、使用教學](#二使用教學)**
> (每天操作看這)→ **[三、參考](#三環境變數)**(環境變數 / 架構 / 開發)。

---

# 一、環境設定

> 面板很輕(純 Python,不含 torch);重的是 **GPU 訓練環境**,那是每台機器各自要裝的前置。
> 裝完跑 `/doctor` 會逐項幫你檢查。最小可動 = 步驟 1 + 2 + 3(GS-2M)。

## 1. 安裝面板

```bash
git clone https://github.com/KuoFengYuan/reconstudio.git
cd reconstudio
conda create -n rec python=3.10 -y
conda run -n rec pip install -r requirements.txt
```

## 2. 外部工具(系統層)

| 工具 | 用途 | 備註 |
|------|------|------|
| **colmap** (3.x / 4.x) | 重建核心 | 不在 `PATH` 就設 `COLMAP_BIN` |
| **ffmpeg** | 抽幀去模糊 + FullHD 縮圖 | **需含 `blurdetect` filter**;NVDEC build 可 GPU 解碼。不在 `PATH` 就設 `FFMPEG_BIN` |

## 3. 訓練後端(需要 GPU,每台機器各自編譯)

後端是**資料不是程式**:加一個訓練器是改 `backends.json`,不是改 code。內建預設是 GS-2M。

### GS-2M(內建,訓練 + mesh)
放在面板的**兄弟目錄** `../GS-2M`、conda env 名 **`gs2m`**:

```bash
cd ..                       # 面板 repo 的上一層
git clone https://github.com/ndming/GS-2M.git
cd GS-2M
conda env create --file environment.yml          # 建 env "gs2m"(python 3.10 / pytorch / cuda-toolkit)
conda activate gs2m
pip install -r requirements.txt                  # 編 5 個 CUDA submodule(GPU/arch-specific)
```

env 名與兄弟路徑對得上就 **zero-config**;不同才在 `backends.json` 覆寫 `conda_env` / `repo` / `python`。
> 要量**實際尺寸 (mm)** 才需補:`conda run -n gs2m pip install opencv-contrib-python plyfile`。

### LichtFeld Studio(選用;MR-NF / iGS+,只訓練、不 mesh)
編譯型 C++/CUDA binary(無 conda),需 CUDA Toolkit 12.8+ 與 vcpkg:

```bash
cd ..
git clone https://github.com/MrNeRF/LichtFeld-Studio.git
cd LichtFeld-Studio
cmake -B build && cmake --build build -j"$(nproc)"   # binary 在 build/LichtFeld-Studio
```

在 `backends.json` 把 `exec` 指到 `…/LichtFeld-Studio/build/LichtFeld-Studio`(範本見
`backends.example.json`);找不到 shared libs 時在 `local.env` 設 `LD_LIBRARY_PATH`。

## 4. 選用元件

- **☁️ 從 GCS 下載資料**:需 Google Cloud SDK + 登入 + 設定 project — 見下方 [☁️ GCS](#-從-gcs-下載資料選用)。
- **🧹 SuperSplat 去背編輯器**:`./tools/build_supersplat.sh`(需 node ≥18 + npm + git;產物在
  `static/supersplat/`,已 gitignore)。

## 5. per-machine 設定(選用 — 名字 / 路徑都對就免)

```bash
cp local.env.example local.env          # 路徑 / port / 非標準 binary 位置
cp backends.example.json backends.json  # 後端 env / repo / exec
```

所有設定集中在 `pipeline/config.py`(一個 typed `Settings`,讀環境變數;見 [三、環境變數](#三環境變數))。

## 6. 啟動 + 自我檢查

```bash
./run.sh                  # → 開瀏覽器 http://127.0.0.1:8077
```

第一次先開 **`/doctor`**:它會檢查 colmap、每個後端的 env python / CUDA / repo,把紅燈修成綠燈
再開始用。遠端機器:`ssh -L 8077:127.0.0.1:8077 user@host`,或 `HOST=0.0.0.0 ./run.sh`。

---

# 二、使用教學

照 **①→②→③→④** 順序,每個階段完成後都有「接著跑下一步」的按鈕,會自動帶入路徑。

## ① 抽幀(影片 → 清晰幀)

選影片資料夾 → `out_dir` 自動帶出 → 設 `fps` 與去模糊(百分位 `keep%` 或固定閾值)→
**▶ 抽幀 + 去模糊**。GPU(NVDEC)自動解碼,失敗則該支影片退回 CPU。輸出鏡射輸入結構、每支
影片一個資料夾、只留最清晰的幀;完成後按 **▶ 接著跑 COLMAP**。

```
輸入  <root>/<group>/<video>.MOV          例  FY115/FY115_0518/A/IMG_3600.MOV
輸出  <out>/<group>/frames_<video>/*.jpg  例  FY115/0518_colmap/A/frames_IMG_3600/*.jpg
```

## ② COLMAP(影像 → 重建)

設 `image_root` / `workspace`。版面(layout)自動偵測,也可手動指定:

```
single  XXX/*.jpg                -> 共用 1 台相機(根目錄直接放圖也算)
multi   ROOT/<group>/*.jpg       -> 每個子資料夾 1 台相機
nested  ROOT/<group>/<vid>/*.jpg -> 先 stage 再每群 1 台相機(① 的輸出就是這種)
```

**影像解析度預設 FullHD**:把每張輸入**實體**等比縮成長邊 ≤1920 的副本(存 `workspace/images_fullhd/`),
整條 COLMAP 跑這些縮圖;Lanczos + 高品質編碼讓 4K→FullHD 仍銳利,可續跑 / 可取消。選「保持原樣」則不縮。

**▶ 啟動 COLMAP** → log 顯示 layout、各階段 banner 與 stepper。完成後可 **🧊 檢視 3D 結果** 或
**🧠 接著訓練**。輸出在 workspace:`database.db`、`sparse/0/{cameras,images,points3D}.bin`、
`<dataset>_<mapper>_mapper/`(**去畸變 PINHOLE** dense,訓練吃這個)、`sparse/points.ply`(檢視快取)。

**GPS / 大場景(進階)** — 輸入若每張都有 EXIF GPS(影片幀沒有),可解鎖更快更穩的比對 / 對齊
(勾選時開跑前會檢查 100% 覆蓋;FullHD 縮圖會把原圖 GPS 接回):

| 選項 | 作用 | 主要參數 |
|------|------|---------|
| `MATCHER=spatial` | 用 GPS 鄰近度限制比對候選,大場景比 vocab/sequential 快且穩 | `SPATIAL_MAX_NEIGHBORS` · `SPATIAL_MAX_DISTANCE`(m) · `SPATIAL_IGNORE_Z` |
| `MAPPER=pose_prior` | GPS 先驗進 BA,抗漂移、輸出**直接公制 + 地理對齊** | `PRIOR_STD_X/Y/Z`(GPS 精度 m;消費級 3~5、RTK ~0.02) |
| `GPS_ALIGN`(選填) | 事後把稀疏模型對齊到 ENU 公尺 | `GPS_ALIGN_MAX_ERROR`(m) |

## ③ 訓練(重建 → 3DGS 模型)

選 backend(環境未就緒會灰掉,旁邊有 `/doctor` 連結)、`source`(COLMAP workspace)、
`model_path`(輸出)、**GPU**(手動選 #0 / #1)。可調參數依 backend schema 動態顯示(迭代數、
解析度、`--material` PBR 材質、法線正則、前景遮罩…每個欄位都有 hint),另有自由 `extra` 欄位。
**▶ 啟動訓練** → 右側狀態列顯示階段(載入相機 / 訓練中 / 存檔)與 `iter N/total`、loss。
完成後 **🧩 接著抽 Mesh**,或先 **🧹 在 SuperSplat 去背景**。

> 非破壞性:COLMAP workspace 以 `<model>_scene/`(symlink)給訓練器,原始不動;只吃去畸變的
> **PINHOLE** 模型,遇到 OPENCV(distorted)會在開跑前直接報錯(最常見的接錯點)。

## 🧹 去背(選用,SuperSplat)

訓練完按 **🧹 在 SuperSplat 去背景** → 右側內嵌編輯器載入訓練雲。**滾輪縮放、左鍵旋轉、右鍵平移**;
框選物件 → **Ctrl+I** 反選 → **Delete** 刪背景(**Ctrl+Z** 還原)→ **✅ 送回去背點雲**。
**原模型完全不動**,依後端分流輸出:

| 後端 | 送回後 | 去背雲輸出 |
|------|-------|-----------|
| **GS-2M**(有 mesh) | 衍生兄弟目錄 + 自動帶入 Mesh 表單 | `<model>_edited_<時間>/…/point_cloud.ply`(可用乾淨雲重抽 mesh) |
| **LichtFeld**(無 mesh) | 存檔 + 瀏覽器下載 | `<model>/edited/cleaned_<時間>.ply` |

> 去背只改點雲;要乾淨 mesh 需重跑 Mesh 階段(僅 GS-2M)。靠兄弟目錄 + symlink,trainer 零修改。

## ④ Mesh(模型 → 三角網格)

選 backend(僅宣告 `mesh_args` 的後端,如 GS-2M)、`model_path`、GPU、TSDF 參數
(`voxel_size` 預設 `0.006`、可勾「自動體素大小」、`sdf_trunc` / `max_depth` / `num_clusters` …)。
要**實際尺寸 (mm)** 就勾 **提供 marker**(ChArUco 板規格已寫在 `backends.json` 的 `marker_defaults`,
免手填)。**▶ 抽取 Mesh** → 完成後 **🧊 檢視 Mesh**、**⬇ 非實際尺寸 mesh**、**⬇ 實際尺寸 mesh (mm)**。

## 檢視與歷史

- **🧊 3D 結果**(COLMAP 後):拖曳旋轉 / 滾輪縮放 / 右鍵平移;**雙擊相機**看影像名、分數、縮圖;
  品質指標 `reproj err`、`track len`;可手動勾掉壞相機 → 寫出**非破壞性**的 `cleaned/<時間>/` 副本。
- **🧊 Mesh 檢視器**:打光實體 mesh,切 **mm / recon**、**📏 量尺**點兩點量距離、亮度 / 白底 / 線框 / 頂點色。
- **歷史**:狀態每 3 秒自動更新;勾選 → **🗑 刪除**(進行中的會先取消)。

---

# 三、環境變數

> **絕大多數變數都會自動偵測或有可用預設,留空即可。** 真正**必須手動設定**(沒有可用預設)的
> 只有一個:用到 **GCS bucket 瀏覽**時的 `CLOUDSDK_CORE_PROJECT`。其餘只在**非標準安裝**或想
> **調效能 / 換落地磁碟**時才需要動。`run.sh` 用 `set -a` 載入 `local.env`,所以每個變數都會匯出到
> 面板、並被它呼叫的外部工具(`ffmpeg` / `colmap` / `gsutil`)繼承。

| env | default | 手動? | 用途 |
|-----|---------|------|-----|
| `CLOUDSDK_CORE_PROJECT` | —(無預設) | **用 GCS 時必填** | GCS 瀏覽器 `gsutil ls` 的預設 GCP project(`gcloud config set project …` 亦可) |
| `COLMAP_BIN` | `colmap` (PATH) | 不在 PATH 才設 | colmap binary |
| `FFMPEG_BIN` | NVDEC build 否則 PATH `ffmpeg` | 不在 PATH / 缺 `blurdetect` 才設 | 需 `blurdetect`;NVDEC = GPU 解碼;也用於 FullHD resize |
| `CONDA_ROOT` / `CONDA_ENV` | `conda info --base` / `rec` | 偵測不到才設 | 面板自己跑在哪個 conda env |
| `RECON_STUDIO_DATA` | `/mnt/ssd1/recon_studio/data` 否則 `~/.recon_studio` | 建議設 | job 狀態 + log(放大磁碟、別放小的 root fs) |
| `RECON_STUDIO_DEST_ROOT` | `/` | 建議設 | GCS 下載落地的本機根目錄 |
| `GSUTIL_BIN` | `gsutil` (PATH) | 不在 PATH 才設 | gsutil binary(GCS 瀏覽 / 下載) |
| `RECON_STUDIO_GCS_ROOT` | (空 = 列全部 bucket) | 選填 | GCS 瀏覽器起始 `gs://` 前綴 |
| `RECON_STUDIO_BROWSE_ROOT` | 有 `/mnt/ssd1` 用它,否則 `/` | 選填 | 本機資料夾選擇器根目錄 |
| `FFMPEG_HWACCEL` | `cuda` | 選填 | 設 `none` 強制 CPU 解碼 |
| `TMPDIR` | `$RECON_STUDIO_DATA/tmp` | 自動 | ffmpeg/colmap scratch(避開 root `/tmp`) |
| `COLMAP_PANEL_MAX_JOBS` | `4` | 選填(調效能) | 同時跑幾個 job(跨階段共用) |
| `COLMAP_PANEL_RESIZE_WORKERS` | CPU 數(≤32) | 選填(調效能) | FullHD resize 的並行 ffmpeg 數 |
| `COLMAP_PANEL_BACKENDS` | `./backends.json` | 選填 | per-machine 後端設定檔路徑 |
| `HOST` / `PORT` | `127.0.0.1` / `8077` | 選填 | 綁定位址 |

---

# ☁️ 從 GCS 下載資料(選用)

面板內建 **☁️ GCS** 分頁,把 Google Cloud Storage 的資料同步到本機後再進流程。它**與重建流程解耦**
—— 只把 bytes 從 `gs://` 搬到本機;下載完到任一分頁用「瀏覽」選那個資料夾即可,**不會自動帶你去下一步**。

- **來源**:填 `gs://bucket/path`,或按 **☁️ 瀏覽** 點選 bucket / 資料夾。
- **下載到**:本機路徑;留空 = 自動放到 `RECON_STUDIO_DEST_ROOT` 底下的 `<bucket>/<路徑>`。
- 底層 `gsutil -m rsync -r`:**可續傳、只補差異**。勾 **鏡像模式 `-d`** 會**刪除**本機多餘檔(有風險)。

**設定 GCS 帳號(一次性)** — 需本機裝好 Google Cloud SDK(提供 `gcloud` + `gsutil`)並登入:

```bash
# 1) 安裝 Google Cloud SDK：https://cloud.google.com/sdk/docs/install

# 2) 登入 Google 帳號（會開瀏覽器；遠端無頭機改用 --no-launch-browser）
gcloud auth login
#    無人值守 server 也可用 service account：gcloud auth activate-service-account --key-file=KEY.json

# 3) 設定預設 GCP project（⚠ 列 bucket 必需，不設 gsutil ls 會報 "requires a project id"）—— 二選一：
gcloud config set project <YOUR_PROJECT_ID>     # 全域預設
#    或只給面板用，寫進 local.env：CLOUDSDK_CORE_PROJECT=<YOUR_PROJECT_ID>

# 4) 驗證（應列出 buckets）
gsutil ls
```

> 找 `PROJECT_ID`:`gcloud projects list`。

---

# 四、架構與開發

三層,依賴單向(`app → web/ → pipeline/`,無循環):

```
app.py        app factory:建 FastAPI、mount static、include routers（~30 行)
└─ web/       HTTP 層
   ├─ routers/  pages · browse · create · jobs · viz · doctor（APIRouter）
   ├─ services/ models（job→路徑解析）· forms（表單→參數驗證）
   └─ shared.py templates / _page / UI 常數
jobs.py       JobManager:asyncio 佇列、N=MAX_JOBS workers、狀態存檔 + log 解析
pipeline/     領域層（torch-free,shell out 到外部工具）
   config.py    Settings:所有設定的單一來源
   runner.py    子行程執行 + 取消        backends.py  後端登錄 + /doctor preflight
   frames / colmap / train / gcs        model.py     解析 COLMAP 稀疏模型
```

請求流程:`form ─POST /ui/*─► JobManager(asyncio 佇列)─► run_* 在 thread 內 shell out
─► console.log ─SSE─► 瀏覽器`。Job 狀態存在 `RECON_STUDIO_DATA/jobs/<id>/`;取消會 `SIGTERM`
整個子行程群組;COLMAP / resize 用 sentinel 做 idempotent(勾 `FORCE` 重跑)。

## 主要檔案

| path | 角色 |
|------|------|
| `app.py` | app factory:建 FastAPI、mount static、include routers、啟動 job manager |
| `web/routers/` | APIRouter:`pages` · `browse` · `create`(建 job 表單)· `jobs`(查詢/取消/SSE log)· `viz`(3D/mesh 檢視 + 下載 + cull + 去背)· `doctor` |
| `web/services/` | `models.py`(job→路徑解析、去背模型衍生)· `forms.py`(表單→參數驗證) |
| `web/shared.py` | Jinja templates、`_page` render helper、UI 表單常數 |
| `jobs.py` | `JobManager`(佇列、N workers、cancel/delete)+ 各類 log 解析 + `MAX_JOBS` |
| `pipeline/config.py` | `Settings`:所有設定的單一來源(pydantic-settings) |
| `pipeline/runner.py` | log emitter + 子行程串流 + 取消(multi-child) |
| `pipeline/frames.py` | `run_frames`:抽幀(NVDEC + CPU fallback)、去模糊、並行、flatten |
| `pipeline/colmap.py` | `run_colmap`:layout 偵測、stages、sentinels、NESTED staging、並行 Lanczos FullHD resize |
| `pipeline/train.py` | `run_train`(COLMAP→trainer scene + PINHOLE guard)、`run_mesh`(mesh + 選用 ChArUco mm 縮放) |
| `pipeline/backends.py` | 後端登錄、env/GPU 解析、CLI builder、`doctor` preflight |
| `pipeline/gcs.py` | GCS 瀏覽(`gsutil ls`)+ 下載(`gsutil -m rsync -r`) |
| `pipeline/model.py` | 解析 COLMAP 稀疏模型(poses/cameras/points/分數)+ PLY 匯出 |
| `templates/` | `index.html`(表單)+ htmx partials + three.js 檢視器 + `doctor.html` |
| `tools/` | trainer-env 腳本:`estimate_marker_scale.py`、`scale_mesh.py`、vendored `colmap_read_write_model.py` |
| `tests/` · `pyproject.toml` · `.github/` | pipeline 純函式測試 · 工具設定(ruff/mypy/pytest)· CI |
| `backends.example.json` · `local.env.example` | per-machine 設定範本(實檔 gitignored) |

## 開發

```bash
pip install -e ".[dev]"   # 面板 + ruff / mypy / pytest
pytest                    # pipeline 純函式測試
ruff check .              # lint
mypy pipeline/config.py   # 型別檢查（目前嚴格守住 config.py）
```

CI(`.github/workflows/ci.yml`)在每次 push / PR 跑 ruff + mypy + pytest。**擴充慣例**:加 endpoint →
在 `web/routers/` 加 handler;加訓練後端 → 改 `backends.json`(不必動 code);加設定 → `pipeline/config.py` 加一個欄位。

> 改任何 `.py`(`app` / `jobs` / `web/` / `pipeline/`)需重啟;改 `templates/` 只需重整頁面。

---

# License

Recon Studio 自身的 code 以**非商業、研究與評估用途**釋出([`LICENSE`](LICENSE)),與其核心訓練器
依賴一致。整合的第三方工具各依其授權([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md))——
特別是 **GS-2M / 3D Gaussian Splatting(Inria,非商業)**、COLMAP(BSD)、FFmpeg(LGPL/GPL)。
因此訓練與 mesh 階段為非商業用途;Gaussian-Splatting 技術的商業使用請洽 Inria。
