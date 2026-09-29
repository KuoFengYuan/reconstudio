# 開發、部署與日常使用流程

[文件首頁](README.md) · [元件索引](context-router.md) · [English](../en/workflows.md)

除非另有註明，指令皆在專案根目錄執行。單次任務只讀一種語言，透過元件索引定位相關檔案。

## 先選擇流程

| 流程 | 起點 | 產出與驗證 |
| :--- | :--- | :--- |
| 開發 | 獨立任務分支／工作區及開發環境 | PR 完成審查，合併的 HEAD 通過 `lint-type-test` |
| 部署 | 已驗證 `main` 的乾淨簽出、機器設定與外部工具 | 面板啟動、選用常駐服務及有驗證的 HTTPS 代理 |
| 日常使用 | 已部署服務及可用的後端 | 可追蹤的實驗成果與品質檢查 |

FastAPI 面板（`app.py`、`web/`、`jobs.py`）管理表單、任務狀態與日誌；
`pipeline/` 呼叫外部工具／訓練器，CUDA 訓練依賴放在後端自己的環境。
nginx 負責代理，選用的 systemd user service 負責讓面板常駐。
CI 驗證程式碼，**不會自動部署伺服器**。

## 1. 開發修改

1. 先讀 [claude.md](../../claude.md)、[Agent 指引](agent-guide.md) 與[元件索引](context-router.md)，
   並依 [AGENT.md](../../AGENT.md) 執行。使用短期 `Feature/`、`Bugfix/` 或 `Enhance/` 分支；
   保留無關修改，原簽出不乾淨時使用獨立 worktree。
2. 建立或選用 Python 3.10 以上的開發環境，安裝：

   ```bash
   python -m pip install -e '.[dev]'
   ```

   需要啟動面板測試時，設定該工作區的 `local.env`：指定 `CONDA_ROOT`／`CONDA_ENV`、
   未使用的 `PORT`、獨立 `RECON_STUDIO_DATA` 與實驗輸出路徑。
   `run.sh` 使用設定的 Python，只啟用另一個 shell 環境不會改變其選擇。
   按需執行 `./run.sh --doctor` 及 `./run.sh`；第二個面板程序不可共用正式服務的任務目錄。
3. 僅修改索引定位到的檔案，保留端點、表單與日誌契約。迭代時執行相關檢查，合併前完成：

   ```bash
   ruff check .
   mypy pipeline/config.py
   pytest
   ```

   這些檢查不需執行 GPU 訓練。介面修改另需驗證桌面／窄畫面、鍵盤及受影響操作，
   並明確回報尚未測試的實體裝置項目。
4. 以具體英文 `feat:`、`fix:`、`enhance:` 或 `docs:` 主旨提交，推送任務分支並建立詳細 PR。
   先寫英文行為、取捨、驗證與待測裝置，再附繁體中文摘要。
   CI 在 PR 與 `main` 推送時執行，檢查名稱為 `lint-type-test`。
5. 檢查必要審查與檢查結果，通常以 Squash 合併已驗證 HEAD，主旨保留 `(#PR)`。
   禁止直接推送 `main` 或繞過保護。確認合併後才刪除此任務的本機／遠端分支，
   fast-forward 更新 `main` 並 prune。部署依下方獨立流程進行。

## 2. 部署或更新工作站

### 首次安裝

1. 取得專案、Git、Conda 與預計使用階段所需的外部工具（ffmpeg、COLMAP、選用 GPU 後端）。
   執行 `./setup.sh` 建立／沿用預設 `rec` 面板環境並安裝依賴。
   已有 `local.env` 會保留，建議設定可能寫入 `local.env.detected`。
   Setup 不會安裝訓練器、nginx 或常駐服務。
2. 核對 `local.env`：工具路徑、`CONDA_ROOT`／`CONDA_ENV`、`HOST=127.0.0.1`、
   固定 `PORT`、儲存／瀏覽根目錄與併行數。僅在需覆蓋機器設定時使用 `backends.json`；
   內建後端位於 `pipeline/backends.py`。
3. 執行 `./run.sh --doctor`。結束碼 `0` 代表必要檢查通過，但可能仍有選用後端警告，
   所以還要確認**本次選用的後端**可用。`--fast` 會略過後端 CUDA 探測。
   健檢不啟動伺服器，也不建置 SuperSplat。
4. 執行 `./run.sh` 前景試跑，開啟印出的網址。啟用同連接埠的常駐服務前，先以 Ctrl+C 停止試跑。

### 常駐面板與區網入口

以執行服務的使用者安裝範本：

```bash
mkdir -p ~/.config/systemd/user
cp infra/systemd/reconstudio.service ~/.config/systemd/user/
```

若部署位置不是 `~/repo/reconstudio`，啟用前先修改已安裝 unit 的 `WorkingDirectory`
與 `ExecStart`。兩者應指向部署簽出，不能指向臨時開發 worktree。接著執行：

```bash
systemctl --user daemon-reload
systemctl --user enable --now reconstudio
sudo loginctl enable-linger "$USER"
systemctl --user status reconstudio
```

區網 HTTPS 與帳密代理另外設定：

```bash
sudo scripts/deploy-nginx-lan.sh
./run.sh --doctor
```

腳本安裝／更新 nginx 設定、TLS 與基本驗證，通過 `nginx -t` 後 reload nginx；
**不會啟動或安裝面板常駐服務**。網域預設 `recon.venraas.tw`、HTTPS 連接埠 `443`、
IP 備援入口 `8443`；其他部署使用 `--domain` 指定自己的網域並設定 DNS。
憑證預設 `auto`，優先使用已存在的 Let's Encrypt 憑證，否則使用 mkcert；
使用 mkcert 的客戶端需信任其 CA。詳見[憑證設定](user-guide.md#網址列的紅色不安全)。

面板維持 loopback。代理讀取 `local.env` 的數字 `PORT`（預設 `8077`），
或用 `--panel-port` 指定；修改連接埠後重跑部署腳本。
僅在 shell 覆蓋面板連接埠不會同步修改 nginx。

### 更新既有部署

1. 等執行中任務結束，或在面板明確取消並確認停止；重啟前清空待執行佇列。
   `jobs.py` 啟動時會將保存的 `queued`／`running` 任務視為失敗，不會自動續跑。
   記錄目前 commit，保留機器設定、任務紀錄與成果。
2. 在**乾淨的部署簽出**先停止服務（或前景程序），再從已驗證的 `main` 更新：

   ```bash
   systemctl --user stop reconstudio &&
   git switch main &&
   git pull --ff-only origin main &&
   conda run -n rec python -m pip install -r requirements.txt &&
   ./run.sh --doctor &&
   systemctl --user start reconstudio
   ```

   `rec` 改成實際設定的面板環境；先修復相關失敗項目再啟動。
   原本工作區的未提交修改應原地保留，不可 reset 或覆寫。
   使用前景模式時省略 systemctl，先停舊程序，再以 `./run.sh` 啟動。
3. 代理、網域或連接埠有變動時重跑部署腳本。驗證本機頁面、需登入的區網入口、
   任務即時更新及目標客戶端上的代表模型；失敗時查看相關服務／建置日誌。
4. 需要回復版本時，先排空任務再停止服務，以記錄的已知可用 commit 建立乾淨簽出，
   安裝該版本依賴並將 user service 指向此簽出。明確確認機器設定與資料相容，
   驗證後才恢復使用；保留失敗版本日誌與成果。不可 reset 未提交工作或改寫已發布的 `main` 歷史。

### 修改後如何生效

| 修改項目 | 套用與驗證 |
| :--- | :--- |
| Python 或 `local.env` | 排空任務、重啟面板、重新健檢 |
| 面板依賴 | 在設定的環境安裝 requirements，再重啟 |
| 範本、`static/js/`、`static/css/` | 重新整理瀏覽器並驗證修改的操作 |
| SuperSplat 修補／版本 | 建置編輯器、重新整理、重開模型並核對版本 |
| nginx 範本、網域、面板連接埠 | 重跑部署腳本，驗證／重載代理，測試需登入的入口 |
| systemd unit 路徑／設定 | 更新已安裝 unit、`daemon-reload`、重啟並檢查狀態 |

面板啟動時預設在背景檢查 SuperSplat 最新穩定版本，需有 git／Node／npm，
未快取的依賴需連線下載。建置成功前仍使用既有套件，失敗也保留舊版。
查看 `<RECON_STUDIO_DATA>/supersplat_build.log` 與 `static/supersplat/.version`。
重現部署可固定 `SUPERSPLAT_VER`；`SUPERSPLAT_AUTOUPDATE=0` 停用啟動同步。
手動建置最新穩定版可執行 `SUPERSPLAT_VER=latest ./tools/build_supersplat.sh`；
此腳本不會載入 `local.env`，應明確傳入預計使用的版本與環境設定。

## 3. 日常使用面板

需要逐步按鈕操作與第一次照片重建範例，請看[啟動後操作指南](usage.md)。

1. 常駐服務已啟動時直接開啟其網址，不要在同連接埠再跑一份 `run.sh`。
   臨時前景使用則執行 `./run.sh` 並保留終端機；缺少必要工具／後端時先健檢。
2. 選本機照片／影片，或先把 GCS 素材搬到本機。影片先抽幀；選用的去背／融合在
   COLMAP 前執行。訓練使用的深度／法線須對應同一組訓練照片，比較實驗使用獨立輸出。
3. 執行 COLMAP、查看重建成果，再將去畸變輸出交給可用後端訓練。
   大場景可選擇先分塊，僅支援的後端可產生網格；使用面板下一步按鈕傳遞路徑。
4. 追蹤任務 ID、參數、狀態、輸出位置與相關日誌。任務完成後仍需品質評估：
   幾何檢查依[評估流程](README.md#評估)，渲染指標依後端評估程序。
5. 用對應檢視器開啟成果。SuperSplat v3 需安全來源與可用的 WebGPU adapter；
   面板可讓使用者明確選擇舊版 WebGL 相容模式。
   瀏覽器／GPU 與載入錯誤應分開記錄，不等同模型品質問題。

## 依問題層級排查

| 症狀 | 限定範圍的檢查 |
| :--- | :--- |
| 連接埠已占用 | 核對既有服務／程序與設定，不重複啟動面板 |
| 區網 502 或 WebSocket 502 | `systemctl --user status reconstudio`、`./run.sh --doctor`、代理目標及 `journalctl --user -u reconstudio -n 80` |
| 登入／憑證錯誤 | 確認預計使用的網域、nginx 帳密、DNS 與憑證信任 |
| Viewer 顯示舊版本 | 查看背景建置日誌、部署的 `.version`，再重整／重開 |
| SuperSplat 啟動失敗 | 瀏覽器 Console、安全來源／WebGPU adapter、明確選擇相容版 |
| 任務失敗 | 僅查看該任務 `job.json`／`console.log`、輸入輸出路徑及所選後端 |

面板 `HOST`／`PORT` 優先順序為 `local.env` → `RECON_STUDIO_HOST/PORT`
→ 符合檢查的既有一般環境值 → `127.0.0.1:8077`。
`run.sh` 允許符合位址檢查的 HOST 與數字 PORT，會忽略編譯器產生的不合法 HOST。
操作手冊中的 `8078` 是這台工作站的覆蓋範例；部署值應固定在 `local.env`，
實際位址以啟動輸出與健檢結果為準。
