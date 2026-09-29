# Agent 檢索與 Token 節省協議

[文件首頁](README.md) · [開發／部署／使用流程](workflows.md) · [English](../en/agent-guide.md)

## 嚴格規則

先讀 Markdown，再讀限定範圍的原始碼。禁止全專案遞迴掃描、廣泛遍歷目錄樹，
以及全域 grep/find 搜尋。禁止在專案根目錄執行 `rg --files`、`rg PATTERN .`、
`grep -R`、`find .`、`tree` 或等效腳本。不可為了蒐集背景而列舉資料集、模型權重、
產生的前端套件或相鄰訓練專案。

## 必須遵循的三步流程

1. 先讀 [claude.md](../../claude.md) 與本指引，確認執行環境、基準設定及實驗規則。
   僅在需要時讀取[詳細實作參考](../en/agent-guide.md#implementation-reference)的相關段落。
2. 從 [README.md](../../README.md) 的文件入口，讀取[實驗元件索引表](context-router.md#agent-context-router)，
   定位本次實驗階段、腳本或模型整合元件。
3. 只開啟直接相關的檔案或已定位的元件目錄。限定行數及明確檔名，例如
   `rg -n 'TRAIN_DEFAULTS' pipeline/train.py`，僅按需要追蹤直接引用。
   索引不足時，只可非遞迴列出一個已定位的目錄，或詢問缺少的路徑，再更新兩種語言的索引；
   不可改用全專案掃描。

## 最小上下文

沿用已讀取的上下文，不重新載入未修改的檔案，也不重複讀取兩種語言版本。
不輸出完整日誌、資料集、產生檔案或環境機密；僅回報相關修改、實驗／設定差異、
必要的指標或錯誤片段與驗證結果。既定的 lint、型別檢查及測試仍可執行；
它們用於驗證，不是蒐集原始碼背景。只摘要結果，失敗時再讀取相關案例。

## 執行環境、基準設定與實驗規則

- `run.sh` 載入 `local.env`，可用設定以 `local.env.example` 與
  `pipeline/config.py` 為準。面板不載入 torch，GPU 訓練由外部程序執行。
- 各階段基準分別為 `pipeline/frames.py` 的 `FRAMES_DEFAULTS`、
  `pipeline/colmap/_run.py` 的 `COLMAP_DEFAULTS` 與 `pipeline/train.py` 的
  `TRAIN_DEFAULTS`。目前訓練預設為 `lichtfeld-mrnf`、GPU `0`、`force=False`。
  後端參數定義位於 `pipeline/backends.py`，`backends.json` 以淺層合併覆蓋機器設定。
  修改參數前先確認該階段，不從舊任務推測預設，也不為單次實驗偷偷更動基準。
- 保留來源資料與基準成果；比較實驗使用獨立工作區／模型輸出，明確決定是否續跑或強制重跑。
  記錄任務 ID、輸入輸出路徑、後端版本、實際參數、設定差異及可用的隨機種子。
  `jobs.py` 在 `<RECON_STUDIO_DATA>/jobs/<job-id>/` 保存含參數／中繼資料的
  `job.json` 與 `console.log`；未自動保存的來源資訊需另行記錄，只讀取本次任務相關片段。
- COLMAP 驗收使用 `tools/verify_recon.py`，按需要提供對應資料庫／manifest，
  回報門檻及略過項目。渲染指標依各後端的評估流程執行；`pytest` 通過不代表重建品質合格。
- 分支、詳細雙語 PR、驗證、合併與清理依 [AGENT.md](../../AGENT.md)。
  文件異動應同步維護英文與繁體中文版本；實作時保留端點、表單欄位與日誌解析契約。

## 開發與實作參考

以下指令皆在專案根目錄、開發環境中執行：

```bash
pip install -e '.[dev]'
ruff check .
mypy pipeline/config.py
pytest
```

修改 Python 後需重啟面板；修改範本、JavaScript 或 CSS 後重新整理頁面。
SuperSplat 修補需重新建置。GPU、COLMAP 與其他重型依賴保留在外部環境中。

既有完整實作參考保留為英文，請只閱讀任務所需的段落：

- [指令與驗證方式](../en/agent-guide.md#commands)
- [架構、COLMAP 參數、後端與遮罩契約](../en/agent-guide.md#architecture)
- [測試慣例](../en/agent-guide.md#testing)
- [參數與新任務類型的開發慣例](../en/agent-guide.md#conventions)

中文操作細節請見[完整操作手冊](user-guide.md)。程式路徑與指令皆相對於專案根目錄。
