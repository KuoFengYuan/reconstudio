# verify_recon.py — COLMAP 多鏡頭重建驗收

```bash
PY=/home/will/miniconda3/envs/gsplat/bin/python      # 需要 pycolmap
$PY tools/verify_recon.py <model_dir> [--manifest case.yaml] [--db DB] [--undistorted]
```

回傳 0 = 全過，1 = 有問題（可直接接 `set -e` 的流程）。

## 為什麼需要

`model_aligner` 報的 EO RMS **只驗證參考鏡頭的位置**。實測遇過 EO RMS 1.83 m 看似正常，
實際上 forward 鏡頭 0 個觀測、rig 偏心量錯 100 倍、重建裂成三個互不相連的塊。
下面每一項都是那次沒被抓到的東西。

## 六項檢核

| # | 內容 | 需要 manifest |
|---|---|---|
| 1 | 每感測器觀測數 / 重投影 / track 長度 / 跨感測器連結率 | ✗ |
| 1b | 內方位與**畸變往返可逆性**（自率定後參數退化會在這裡現形） | ✗ |
| 2 | rig 外參 vs 率定證書偏心量 | 需要 `rig` |
| 3 | 相機中心 vs 廠商 EO | 需要 `eo` |
| 4 | 同幀各感測器中心位移（轉進參考相機座標系） | 部分 |
| 5 | **跨感測器對極 Sampson 誤差** — 最靈敏的一項 | 需要 `--db` |

**不給 manifest 也能跑 1 / 1b / 4 / 5**，任何 COLMAP 模型直接吃。光是「某感測器觀測數為 0」
和「跨感測器對極幾百 px」兩個訊號，就足以判斷一組重建是不是壞的。

## 幾個實作上的要點

- **幀分組優先用模型自帶的 `frames.bin`**（COLMAP 4.x），沒有 rig 的模型才退回 manifest 的檔名 regex。
  不要靠 `rig_configurator` 的檔名慣例 —— 各鏡頭尾碼不同時它會 SIGABRT。
- **檢核 4 必須先轉進參考相機座標系**。航帶是往返飛的（κ 差 180°），世界座標系下的位移
  在平均時會互相抵銷，跟證書的相機系偏心量不能直接比。
- **檢核 5 排除同幀配對**。rig 基線只有 0.1–0.2 m，同幀的對極幾何是退化的。
- **去畸變後的模型要加 `--undistorted`**。座標系與 DB keypoints 不同，對極測試會失效。

## manifest

見 `/Disk0/詮華國土測繪/gs1223_fix/case.yaml`。重點欄位：

- `rig.nodal_points` — 率定證書的偏心量表，`units` 宣告單位（證書常用 **mm**，不是 m）
- `rig.axes` — 機體軸 → 參考相機軸的對應。COLMAP 相機系是 `+X=影像右, +Y=影像下, +Z=光軸`。
  這個對應**要逐 case 驗證**：用證書宣告的掛載角（例如「left rolled +42°」）去核對解出來的傾斜方向。
- `eo.schema` — CSV 欄位對應。欄名各家不同（`ELLIPSOID HEIGHT` / `Z` / `Alt` / `H`）。
- `eo.local_origin` — **逐 case 計算**。TWD97 的 NORTHING 2,652,686 在 float32 的解析度是
  0.250 m，是 6 cm GSD 的 4 倍。

## 已知限制

- 檢核 4 對「rig 約束過的模型」是恆等式（散布必為 0），只有對**無 rig 的模型**才有鑑別力。
- 檢核 2/4 只支援單一 rig（`rec.rigs` 取第一個）。
- EO 角度（ω/φ/κ）不參與驗證。實測那組扣掉常數偏心角後散布中位仍有 0.17°
  （1428 m 航高 = 4.2 m 地面），精度不足以當真值。
