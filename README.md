# Recon Studio

A local web workspace for image-based 3D reconstruction: video or photos →
COLMAP → 3D Gaussian Splatting → mesh, with job tracking, live logs, viewers and
measurement tools.

本機網頁工作區：把影片或照片轉成 COLMAP 重建、3DGS 模型與 Mesh，並提供任務追蹤、
即時紀錄、檢視器與量測工具。

## Pipeline / 流程

```
影片 ─ 抽幀 ─┐
             ├─ (去背 · 融合 · 深度/法線) ─ COLMAP ─ (分塊) ─ 3DGS 訓練 ─ Mesh
照片 ────────┘                                                     │
雲端 (GCS) 下載 / 上傳                         檢視：點雲與相機 · Mesh · 量測 · SuperSplat
```

The panel only orchestrates. ffmpeg, COLMAP and each trainer (LichtFeld-Studio,
gsplat, GS-2M) run as subprocesses in their own environments, so the panel itself
installs with `pip` and needs no GPU stack.

面板只負責調度；ffmpeg、COLMAP 與各訓練後端在各自的環境以子程序執行，面板本身
不需要 GPU 套件。

## Interface / 介面

| | |
| :--- | :--- |
| **＋ 建立任務** | Opens the settings column. 「切換功能」 picks a form: 功能 (☁️ 資料 · 抽幀 · COLMAP · 融合 · 訓練 · Mesh), 工具 (🌊 深度/法線 · ✂️ 影像去背 · 🧱 分塊), 檢視 (🖼 圖片 · 👁 Mesh Viewer · 📏 量測 · ✨ SuperSplat). 開啟設定欄並選擇功能表單。 |
| **工作區首頁** | Quick starts: 我有影片 / 我有照片 / 我有 3D 模型. 依手上的素材直接開始。 |
| **目前工作** | Status, COLMAP stages, next-step buttons (train → mesh), downloads and the live log. 狀態、階段、下一步、下載與即時紀錄。 |
| **任務紀錄** | Counts, search by project, folder, name or ID, filter by type and status, bulk delete. 搜尋、依類型與狀態篩選、批次刪除。 |

The header also shows the live-update connection and links to 環境檢查
(`/doctor`). Jobs run inside the panel process, so a change to the panel's Python
code applies only after a restart — and a restart stops running jobs.

上方列顯示即時更新連線狀態與「環境檢查」。任務在面板程序內執行：修改面板的 Python
程式要重新啟動才生效，重啟會中止執行中的任務。

## Quick start / 快速開始

Linux with Conda, ffmpeg and COLMAP installed; an NVIDIA GPU for training.

```bash
./setup.sh          # panel env + local.env (keeps an existing local.env)
./run.sh --doctor   # environment report; fix the stage you need first
./run.sh            # prints the URL (127.0.0.1:8077 unless local.env says otherwise)
```

The panel binds 127.0.0.1 and has no login of its own; put it on the LAN through
the bundled nginx setup (`scripts/deploy-nginx-lan.sh`), not `HOST=0.0.0.0`.

面板只綁 127.0.0.1、沒有自己的登入；區網存取請用內附的 nginx 設定。

## Documentation / 文件

| English | 繁體中文 |
| :--- | :--- |
| [Quick start](docs/en/README.md) | [快速開始](docs/zh-TW/README.md) |
| [After-startup walkthrough](docs/en/usage.md) | [啟動後操作指南](docs/zh-TW/usage.md) |
| [Optional tools](docs/en/tools.md) | [選用工具教學](docs/zh-TW/tools.md) |
| [Development, deployment, daily use](docs/en/workflows.md) | [開發、部署、日常使用](docs/zh-TW/workflows.md) |
| [Agent guide](docs/en/agent-guide.md) · [Context router](docs/en/context-router.md) | [Agent 指引](docs/zh-TW/agent-guide.md) · [元件索引](docs/zh-TW/context-router.md) |
| — | [完整操作手冊](docs/zh-TW/user-guide.md) |

## Development / 開發

```bash
pip install -e '.[dev]'
ruff check . && mypy pipeline/config.py && pytest   # offline, no GPU needed
```

Coding agents start at [AGENTS.md](AGENTS.md) (Claude Code reads it through
[CLAUDE.md](CLAUDE.md)). It covers the context-routing rules — no repository-wide
scans — plus runtime traps, conventions, UI rules and the branch/PR workflow.

<a id="agent-context-router"></a>
Agent 請先讀 [AGENTS.md](AGENTS.md)，再依所選語言的元件索引定位檔案；禁止全專案遞迴檢索。

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
