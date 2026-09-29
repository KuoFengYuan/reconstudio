# Recon Studio — 快速開始

[English](../en/README.md) · [專案首頁](../../README.md)

把影片或照片轉成 COLMAP 重建成果、3D Gaussian Splatting 模型，並透過支援的後端
產生網格。本機網頁面板提供表單、進度日誌、取消任務、模型檢視、去背與 GCS 搬檔。

- [啟動後完整教學](usage.md)：欄位填寫範例、預期畫面、檢視器操作、瀏覽器驗收及常見問題。
- [選用工具逐步教學](tools.md)：GCS 搬檔、去背、融合、深度／法線、分塊與量測。
- [開發、部署與日常使用流程](workflows.md)。
- [Agent 檢索協議](agent-guide.md)：執行環境、基準、實驗紀錄與開發規則。
- [實驗元件索引](context-router.md)：直接相關的原始碼路徑及輸入產出。
- [完整操作手冊](user-guide.md)：安裝、日常使用及進階設定。

## 環境設定

先安裝 Git 與 Conda，再於專案根目錄執行：

```bash
./setup.sh
./run.sh --doctor
./run.sh
```

安裝程式建立輕量面板環境並保留既有 `local.env`。ffmpeg、COLMAP 與選用 GPU 後端
需依[完整安裝說明](user-guide.md#一安裝第一次部署)另行安裝。
解決該階段的健檢警告後，開啟啟動時印出的網址；連接埠可由設定調整。

## 面板啟動後怎麼操作

開啟印出的網址，先看右上角「環境檢查」，再依素材選「我有影片」、「我有照片」或既有模型。
第一次使用請跟著[啟動後操作指南](usage.md)，完成填表、提交任務、查看進度及開啟／下載成果。

## 執行實驗

在面板選擇本機照片，或先從影片抽幀。指定新工作區執行 COLMAP，再將去畸變結果
交給訓練。選擇可用後端與獨立模型輸出路徑；僅支援網格的後端可接 Mesh。
記錄任務 ID、實際參數、後端版本與基準差異，並由[元件索引](context-router.md)
定位各階段預設值。本專案負責編排外部訓練器，沒有統一訓練 CLI 或內建模型目錄。

## 評估

在含 `pycolmap`、`numpy` 的獨立環境執行 COLMAP 驗收；YAML manifest 另需 `PyYAML`。
將以下範例替換成該次實驗的 Python、稀疏模型與對應的原始資料庫路徑：

```bash
/path/to/evaluation-env/bin/python tools/verify_recon.py \
  /path/to/workspace/sparse/0 --db /path/to/workspace/database.db
```

個案率定／EO 驗收加上 `--manifest /path/to/case.yaml`。
去畸變模型需加 `--undistorted`，因原始資料庫座標不同而略過對極檢查。
未提供資料庫／manifest 時，依賴它們的檢查會略過。結束碼 `0` 表示已執行項目通過，
`1` 表示驗收失敗；執行錯誤需另行排查。詳見[驗收範圍與限制](../../tools/verify_recon.README.md)。

將指標片段與門檻和基準比較，列出略過項目。此工具驗證幾何品質；
渲染指標應使用所選訓練後端的評估流程。

## 驗證程式修改

依 [AGENT.md](../../AGENT.md)，在開發環境、專案根目錄執行：

```bash
pip install -e '.[dev]'
ruff check .
mypy pipeline/config.py
pytest
```

這些是軟體驗證，不是實驗品質指標。
