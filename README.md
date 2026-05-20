# COLMAP Panel

A web front-end for a two-stage photogrammetry pipeline, implemented as pure-Python
modules under [`pipeline/`](pipeline/) (ports of `extract_frames.sh` / `colmap_pipeline.sh`).
`ffmpeg` and `colmap` are invoked as subprocesses (resolved from `PATH`, overridable via
`FFMPEG_BIN` / `COLMAP_BIN`); the orchestration is Python so progress, cancellation, and an
in-browser 3D result viewer are first-class.

Single-machine local tool. One job runs at a time (ffmpeg and COLMAP are heavy);
the Frames stage decodes on the GPU and runs multiple videos in parallel internally.

## Input / output spec

**① Frames** — videos → sharp frames. Extracts `fps` frames/sec, computes a blur
score per frame (ffmpeg `blurdetect`), keeps the sharpest `keep_pct`%. Decoding uses
the GPU (NVDEC) automatically, falling back to CPU per-video. Output mirrors the input
tree, one dir per video, sharp frames only (flattened):

```
IN   <root>/<group>/<video>.MOV          e.g.  FY115/FY115_0518/A/IMG_3600.MOV
OUT  <out>/<group>/frames_<video>/*.jpg  e.g.  FY115/0518_colmap/A/frames_IMG_3600/*.jpg
```

**② COLMAP** — images → reconstruction. Layout is auto-detected (or forced via the
`layout` selector); a workspace nested inside `image_root` is ignored when detecting groups:

```
single  XXX/*.jpg              -> 1 shared camera (also when the root itself holds images)
multi   ROOT/<group>/*.jpg     -> 1 camera per group   (CAMERA_MODE=per_folder)
nested  ROOT/<group>/<vid>/*.jpg -> staged, 1 camera per group   (① feeds this)
```

**影像解析度 (resize)**: `保持原樣` or `FullHD` (≤1920 longest side). FullHD applies
`--FeatureExtraction.max_image_size 1920` and `image_undistorter --max_image_size 1920`
→ faster, smaller training output. Aspect-preserving; already-≤1920 images are unchanged.

Output (workspace): `database.db`, `image_list.txt`, `sparse/0/{cameras,images,points3D}.bin`,
`<dataset>_<mapper>_mapper/` (undistorted dense input), `pipeline.log`, sentinels
(`.stage.done` …), and `sparse/points.ply` (cached for the 3D viewer).

```
 ① Frames                                ② COLMAP
 video dir ──run_frames (NVDEC,並行)──► <out>/<group>/frames_<video>/*.jpg ──run_colmap──► sparse + dense
   FY115/FY115_0518/A/IMG_3600.MOV         FY115/0518_colmap/A/frames_IMG_3600/*.jpg     FY115/0518_colmap_ws/
```

The Frames tab auto-suggests the output name (`FY115_0518` → `0518_colmap`) and, when a
job finishes, a **▶ 接著跑 COLMAP** button prefills the COLMAP form (`image_root` = output,
`workspace` = `<out>_ws`, `layout` = auto, which detects the nested frames layout).

## Run

```bash
./run.sh                 # conda env colmap_panel, GPU ffmpeg, data on /mnt/ssd1
# open http://127.0.0.1:8077
```

`run.sh` auto-detects sensible defaults; override any of them via the environment or a
**`local.env`** file (copy `local.env.example` → `local.env`; gitignored). **Backend changes
(`app.py` / `pipeline/` / `jobs.py`) need a restart; template (`templates/`) changes just need
a browser refresh.**

| env | default | why |
|-----|---------|-----|
| `COLMAP_BIN` | `colmap` (PATH) | colmap binary; set for non-standard installs |
| `FFMPEG_BIN` | `/mnt/ssd1/bin/ffmpeg-nvdec` if present, else PATH `ffmpeg` | needs the `blurdetect` filter; NVDEC build = GPU decode |
| `FFMPEG_HWACCEL` | `cuda` | set to `none` to force CPU decode |
| `COLMAP_PANEL_DATA` | `/mnt/ssd1/colmap_panel/data` (else `~/.colmap_panel`) | job state + logs (keep off a small root fs) |
| `TMPDIR` | `$COLMAP_PANEL_DATA/tmp` | keep ffmpeg/colmap scratch off root `/tmp` |
| `COLMAP_PANEL_BROWSE_ROOT` | `/mnt/ssd1` if present, else `/` | directory-picker root |
| `CONDA_ROOT` / `CONDA_ENV` | `conda info --base` / `colmap_panel` | which conda env to launch |
| `HOST` / `PORT` | `127.0.0.1` / `8077` | bind address |

Remote access (binds to localhost): `ssh -L 8077:127.0.0.1:8077 user@host`, or `HOST=0.0.0.0 ./run.sh`.

### Deploy to another machine

`colmap` and `ffmpeg` resolve from `PATH` by default, so no code edits are needed —
just install both (ffmpeg must have the `blurdetect` filter) and the Python deps:

```bash
conda create -n colmap_panel python=3.10 -y
conda run -n colmap_panel pip install -r requirements.txt
cp local.env.example local.env      # then edit paths/ports for that machine (optional)
./run.sh
```

If `colmap` / `ffmpeg` live in non-standard locations, set `COLMAP_BIN` / `FFMPEG_BIN`
in `local.env`.

One-time env setup (already done on this box):

```bash
conda create -n colmap_panel python=3.10 -y
/home/aibox/miniconda3/envs/colmap_panel/bin/pip install -r requirements.txt
```

## 操作教學 (workflow)

**啟動**：`./run.sh` → 瀏覽器開 `http://127.0.0.1:8077`（遠端用上面的 SSH tunnel）。

**① 影片 → 清晰幀**（左側分頁「① 影片→清晰幀」）
1. `input`：按「瀏覽」選影片資料夾（會遞迴掃描;顯示底下影片數）。
2. `out_dir`：自動帶出（`FY115_0518` → `0518_colmap`），可改。
3. `fps`：每秒抽幀數。解碼自動用 GPU(NVDEC),失敗退回 CPU。
4. 去模糊：百分位（保留最清晰 X%）或絕對閾值。
5. 按 **▶ 抽幀 + 去模糊** → 右側看即時 log 與「清晰/丟棄」計數。
6. 完成後按 **▶ 接著跑 COLMAP**（自動帶 image_root / workspace,layout 自動偵測）。

**② COLMAP**（分頁「② COLMAP」;只有圖片時可直接從這裡開始）
1. `image_root`：圖片根目錄。layout 預設自動偵測（single / multi / nested 都會自動判斷,
   workspace 放在 image_root 內也會自動排除）。手動指定 layout / subfolders 在「輸入格式」摺疊區。
2. `workspace`：輸出工作區（建議放 `/mnt/ssd1` 或 `/mnt/ssd2`,別放 `/home`）。
3. **影像解析度**：保持原樣，或等比縮小到 FullHD（≤1920;加速、輸出較小、適合訓練）。
4. 進階(Camera / Matching / Mapping / Pipeline / Advanced)為摺疊區塊,需要再展開;`FORCE` 重跑。
5. 按 **▶ 啟動 COLMAP** → log 顯示偵測到的 layout、各階段 banner、stage stepper。

**檢視 3D 結果**（COLMAP 完成後按 **🧊 檢視 3D 結果**,內嵌在右側,非新分頁;**← 回 log** 返回）
- 操作：拖曳=自由 360° 旋轉(trackball,不卡死)、滾輪=縮放、右鍵=平移。
- **雙擊相機** → 顯示影像名稱、索引、分數(registered / features)與**原圖縮圖**,並把該相機**高亮**(不移動視角)。
- **座標 gizmo**（右下角）：顯示朝向、點軸對齊視角;可用「座標 gizmo」開關。
- **🔄 旋轉物體(對齊)**：勾選後出現旋轉環,拖環轉動物體本身去對齊;**歸零旋轉**復原。
- 面板：點大小 / 相機大小滑桿、相機開關、重置視角。
- **品質指標**：`reproj err`(平均重投影誤差,px;<1 綠 / <2 黃 / 更高紅)、`track len`(每點平均觀測數)。
- 旋轉樞紐/gizmo 用穩健中心(中位數),不被離群點帶歪;相機 ≤ 8 台顯示色塊圖例,更多則單色。

**歷史**（分頁「歷史」）：狀態**每 3 秒自動更新**（勾選與捲動會保留）。勾選多筆 → **🗑 刪除選取**
（進行中的先取消,已完成的連 log 一併移除）。

## How it works

```
form ──POST /ui/{frames,jobs}──► JobManager (asyncio queue, 1 worker)
                                     │  run_frames / run_colmap  in a thread
                                     ▼  (Runner shells out to ffmpeg / colmap)
                                 console.log ──SSE──► browser <pre>  (replay = 1 event)
                                       │
            frames:  "######## [j/N] …" + "-> K kept / D dropped"  ─┐
            colmap:  "=== […] feature_extractor …" + "skip <stage>" ─┴─► progress / stepper

done (colmap) ──► /viz/<id> (iframe): model_converter -> points.ply + parsed poses/scores
                  three.js: TrackballControls + point cloud + camera frustums
                  (COLMAP Y-down → three Y-up flip via 180° about X)
```

- **Job state** persists under `COLMAP_PANEL_DATA/jobs/<id>/`: `job.json` + `console.log`.
  Jobs left running when the server stops are marked failed on reload (old-schema job.json still loads).
- **Frames decode**: `-hwaccel cuda` when ffmpeg supports it; on failure a video retries on CPU.
- **Cancel** sets a flag and `SIGTERM`s every child process group the Runner started.
- **Camera pick** (viewer): nearest camera centre in screen space (robust to zoom / overlap / rotation).
- **History**: auto-refreshes every 3s (selection + scroll preserved); multi-select delete.
- **Idempotency**: COLMAP uses sentinels / output checks — tick `FORCE` to re-run.

## Files

| file | role |
|------|------|
| `app.py` | FastAPI routes: page, htmx `/ui/*`, JSON `/api/*` (jobs, delete, scene, image, imagefile), SSE logs, `/viz/<id>` |
| `jobs.py` | generic `JobManager` (queue, thread worker, cancel/delete) + per-kind log parsers |
| `pipeline/runner.py` | log emitter + subprocess streaming + cancel (multi-child) |
| `pipeline/frames.py` | `run_frames`: fps extraction (NVDEC + CPU fallback), blur cutoff, parallel, flatten |
| `pipeline/colmap.py` | `run_colmap`: layout detect, stages, sentinels, NESTED staging, FullHD resize |
| `pipeline/model.py` | parse sparse model (poses/cameras/points/per-image score) + PLY export |
| `templates/` | `index.html` (both forms) + htmx partials + `viz.html` (three.js viewer) |
| `static/three/` | vendored three.js + PLYLoader + Trackball/Transform controls + ViewHelper |
| `run.sh` + `local.env.example` | launcher with auto-detected defaults; per-machine overrides |

## Not yet (easy next steps)

- **Concurrent jobs**: `COLMAP_PANEL_WORKERS=N`, or pin each COLMAP job to a GPU
  (`--*.gpu_index` / `-hwaccel_device`) to use both RTX 4090s. Currently 1 worker (serial).
- Multi-input frames jobs (`run_frames` already accepts several paths/dirs).
- Re-run a single COLMAP stage (uncheck others + FORCE) as one click.
