# AI Agent 實驗元件快速索引

[文件首頁](README.md) · [Agent 協議](agent-guide.md) · [English](../en/context-router.md)

先讀 Agent 協議、再讀本索引表，最後只開啟相關檔案。禁止全專案遞迴掃描、廣泛遍歷
與全域 grep/find。除執行期／外部路徑外，程式路徑皆相對於專案根目錄；連結直接指向原始碼。

本專案負責實驗流程編排，沒有集中式的 `experiments/`、`models/` 或 `data/`
原始碼目錄。資料與模型路徑由個別任務指定，模型實作位於設定的外部專案；
不可為了蒐集背景而掃描這些專案或資料目錄樹。

<a id="agent-context-router"></a>

| 實驗模組 | 核心路徑 | 實驗用途與範疇 | 輸入與產出 |
| :--- | :--- | :--- | :--- |
| 啟動與設定 | [run.sh](../../run.sh), [setup.sh](../../setup.sh), [local.env.example](../../local.env.example), [pipeline/config.py](../../pipeline/config.py) | 面板環境、工具、儲存與併行設定 | `local.env` → 執行期設定 |
| 請求與實驗紀錄 | [app.py](../../app.py), [web/services/forms.py](../../web/services/forms.py), [jobs.py](../../jobs.py) | HTTP 入口、參數驗證、佇列與進度 | 表單 → `<RECON_STUDIO_DATA>/jobs/<job-id>/{job.json,console.log}` |
| 資料搬移與抽幀 | [pipeline/gcs.py](../../pipeline/gcs.py), [pipeline/frames.py](../../pipeline/frames.py) | GCS 雲端搬檔、ffmpeg 抽幀與模糊篩選 | 雲端、檔案、影片 → 本機素材、`frames_<video>/`、品質報告 |
| 遮罩與多次拍攝融合 | [pipeline/matte.py](../../pipeline/matte.py), [tools/sam_matte.py](../../tools/sam_matte.py), [pipeline/fusion.py](../../pipeline/fusion.py) | 外部 SAM 推論與遮罩素材整備 | 照片與提示 → `no_bg/{masks,cutout}/`；融合 `images/`、`masks/` |
| 相機與點雲重建 | [pipeline/colmap/_run.py](../../pipeline/colmap/_run.py), [pipeline/colmap/](../../pipeline/colmap/), [pipeline/large_scene.py](../../pipeline/large_scene.py) | COLMAP 階段；按需定位 rig、GPS、版面輔助檔 | 照片與遮罩 → `database.db`、稀疏模型與去畸變照片 |
| 深度與法線 | [pipeline/depth.py](../../pipeline/depth.py), [pipeline/moge3.py](../../pipeline/moge3.py), [tools/moge3_preprocess.py](../../tools/moge3_preprocess.py) | LichtFeld MoGe-2 或獨立 MoGe-3 環境 | 照片 → 與 `images/` 並列的 `depth/`、`normals/` |
| 模型與後端整合 | [pipeline/backends.py](../../pipeline/backends.py), [backends.example.json](../../backends.example.json) | 後端指令、參數定義與環境解析 | 選用 `backends.json` → 外部預設 `../GS-2M`、`../gsplat`、`../LichtFeld-Studio`；可覆蓋 |
| 訓練與網格 | [pipeline/train.py](../../pipeline/train.py) | 整備 COLMAP 場景，呼叫支援的訓練／網格後端 | 去畸變場景 → 指定 `model_path`，依後端產生模型、權重與網格 |
| 大場景分塊 | [pipeline/blocksplit.py](../../pipeline/blocksplit.py) | 將去畸變場景切成可訓練子塊 | COLMAP 場景 → `block_<ix>_<iy>/`、`manifest.json`、選用裁切池 `_tiles/` |
| 驗收與指標 | [pipeline/verify.py](../../pipeline/verify.py), [tools/verify_recon.py](../../tools/verify_recon.py), [tools/verify_recon.README.md](../../tools/verify_recon.README.md) | COLMAP 觀測、畸變、rig／EO 及對極驗收 | 稀疏模型及選用的對應資料庫／manifest → 文字指標與結束碼 |
| 檢視器與工作區介面 | [templates/index.html](../../templates/index.html), [static/css/workspace.css](../../static/css/workspace.css), [static/js/workspace.js](../../static/js/workspace.js), [static/js/supersplat.js](../../static/js/supersplat.js), [web/routers/viewer.py](../../web/routers/viewer.py) | 工作區排版、模型載入與檢視 | 任務與模型路徑 → 瀏覽器視覺化 |
| SuperSplat 編輯器建置 | [tools/build_supersplat.sh](../../tools/build_supersplat.sh) | 版本選擇與腳本引用的整合修補 | 上游版本與修補 → 產生的 `static/supersplat/`；不掃描套件 |
| 驗證與部署 | [pyproject.toml](../../pyproject.toml), [.github/workflows/ci.yml](../../.github/workflows/ci.yml), [tests/](../../tests/), [infra/systemd/reconstudio.service](../../infra/systemd/reconstudio.service), [scripts/deploy-nginx-lan.sh](../../scripts/deploy-nginx-lan.sh) | 測試設定與區網代理；只讀任務相關測試 | 程式與設定 → 檢查結果或代理設定 |

`RECON_STUDIO_DATA` 保存任務中繼資料／日誌，不等於照片或訓練模型所在處。
`run.sh` 自動選擇儲存位置，可用設定覆蓋；直接使用 `pipeline.config.Settings`
時預設為 `~/.recon_studio`。以指定任務的中繼資料定位成果，不遍歷整個儲存根目錄。

元件移動時同步維護兩種語言索引。單次任務只讀取一種語言，避免重複載入上下文。
