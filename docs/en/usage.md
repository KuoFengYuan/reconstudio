# Using Recon Studio After Startup

[Documentation](README.md) · [Development/deployment/use](workflows.md) · [繁體中文](../zh-TW/usage.md)

Start here once the panel is running. For installation or service setup, see
[Deployment](workflows.md#2-deploy-or-update-the-workstation). Quoted Chinese text
below identifies the actual interface buttons; the instructions are in English.

## 1. Open the panel and check readiness

1. Open the URL printed by `./run.sh`. Use the local URL on the server itself.
   From another computer, use the configured LAN HTTPS URL and deployment credentials.
   That computer's `127.0.0.1` points to itself, not to the server.
2. If the persistent service is already running, open its URL without starting
   another `run.sh`. For a foreground launch, keep the server terminal open.
3. Click the top-right environment check, “環境檢查”, and verify the tools needed
   for your operation: ffmpeg, COLMAP, or the selected training backend. Resolve a
   backend marked “環境未就緒” before submitting training work.
4. The header connection indicator concerns live updates. Check the job's actual
   status and logs in the current-work tab to follow execution.

## 2. Find your way around

| Interface entry | Purpose |
| :--- | :--- |
| 工作區首頁 | Workspace home: choose video, photos, or an existing 3D model |
| ＋ 建立任務 | Expand task settings; use “切換功能” to select a stage or tool |
| 目前工作 | Current job progress, logs, results, or the model viewer |
| 任務紀錄 | Search old jobs, filter by type/status, then click “開啟” |
| 收合設定 | Collapse settings to give the logs or model more space |

Switching workspace tabs preserves the form and current work. After refreshing,
find submitted jobs in history; use “複製任務連結” to save a link to a specific job.
An unsubmitted form is not a saved job.

## 3. Choose the entry for your input

| What you have | First action | Next stages |
| :--- | :--- | :--- |
| Video | Home → “我有影片” | Extract frames → COLMAP → training → optional Mesh |
| Photo folder | Home → “我有照片” | COLMAP → training → optional Mesh |
| GCS data | Create task → switch function → “資料” | Download to the server, then choose frames/COLMAP |
| Finished COLMAP workspace | Enter its workspace in COLMAP and click “讀取既有結果” | View cloud/cameras; training also needs usable undistorted images/model |
| Mesh file | Home → “我有 3D 模型”, or Mesh Viewer | View directly without retraining |
| 3DGS or point cloud | Create task → switch function → SuperSplat | Open directly, or from a completed training job |

**Training and reconstruction paths refer to the server running the panel.** Their
“瀏覽” picker does not upload photos from your current computer. Put inputs on
readable server storage first, or use the GCS transfer tool. After a GCS download,
manually choose the downloaded folder in the relevant function; reconstruction
does not start automatically.

Only viewers with an explicit local-file/drag-and-drop entry can open models from
your current computer directly. A `.ply` file can contain a mesh or a point cloud;
choose the viewer according to its content.

## 4. First walkthrough: photos to a viewable model

These are example paths. Replace them with readable/writable locations on your
server; `/Disk0` is not required. Have the photos in place first, and use distinct
output directories for comparison runs.

| Purpose | Example |
| :--- | :--- |
| Source photos | `/Disk0/my-scan/images` |
| This COLMAP workspace | `/Disk0/my-scan/colmap-run-01` |
| This model output | `/Disk0/my-scan/model-run-01` |

### A. Reconstruct the photos

1. Click “我有照片”, or create a task and switch to COLMAP.
2. Select the source in “照片資料夾” and the output in “重建工作目錄”. For a first run,
   keep the default resolution/stage selection; retain undistortion for training.
3. Click “啟動 COLMAP” once. In “目前工作”, confirm a job was created. “排隊中” means
   waiting for an execution slot; “執行中” means stage execution has started.
4. When the status becomes “已完成”, click “檢視 3D 結果” to inspect the cloud and
   cameras. Investigate missing geometry or obviously misplaced cameras before
   investing in the training stage.

### B. Train the result

1. Click “接著訓練” on the completed COLMAP job; it fills the source workspace.
2. Confirm `source` points to this workspace or its undistorted output. Choose a
   ready backend (“訓練後端”), model output folder (“模型輸出資料夾”), and GPU.
   Parameters vary by backend; start with the selected backend's defaults.
3. Click “啟動訓練”. **Next-stage buttons fill a form; you still confirm and submit
   the next job yourself.**
4. Follow status/logs in “目前工作”; iteration and loss appear when supplied by the
   backend. On completion, record the model output path. “在 SuperSplat 去背景” opens
   the trained model for viewing; deleting background is optional. “下載點雲” downloads
   the training result.

### C. Extract Mesh only when needed

A completed training job offers “接著抽 Mesh” only when the backend supports it.
Click it, review the Mesh form, then click “抽取 Mesh”. On completion, use
“檢視 Mesh” or the available artifact download links. Textured, GLB, and calibrated
size downloads appear only when those artifacts exist.

Without the Mesh action, you can still view/download the point cloud. Do not assume
all trainers support mesh extraction. Uncalibrated model coordinates are not
physical millimetres; confirm scale and units before measuring.

## 5. Start from video instead

Click “我有影片”, set the video source and frame output directory, review sampling
rate and blur filtering, then click “抽幀並移除模糊影格”. Check the kept/rejected counts
on completion; use the image viewer (“圖片”) to inspect frames if needed.
“接著跑 COLMAP” fills the frame result into the next form. Verify its image/workspace
paths and follow the photo walkthrough above. If no frames survive or extraction
fails, inspect the filter report/error before continuing.

## 6. Open an existing model without running a pipeline

- **Mesh:** “我有 3D 模型” opens Mesh Viewer. Select a server `.ply/.obj/.stl/.glb`
  and click “開啟 Viewer”, or leave the path empty and select/drop a local file in
  the viewer.
- **Cloud/3DGS:** switch to SuperSplat and choose a server file, then click
  “開啟 SuperSplat”. Alternatively leave the path empty and open/drop a local file
  in the editor. Completed training jobs also open their own results directly.
- **Large training results:** when opened from a training job, wait for loading
  to finish. Use “編輯畫質” → “流暢（大場景）” to reduce display load.
  “停止載入／關閉模型” closes this viewer; it is not the training-job cancel action.
- **Loading failure:** read the displayed error and version. SuperSplat v3 needs
  HTTPS/localhost and a usable WebGPU adapter. Follow the environment guidance,
  then use “重試目前版本”; explicitly choose “使用相容版” when a legacy bundle is
  available. Loading the web page does not establish GPU availability or prove
  that the model is corrupt.

## 7. Monitor, cancel, and revisit jobs

| Status or need | Action |
| :--- | :--- |
| 排隊中 | Wait for a slot; do not resubmit the same work |
| 執行中 | Follow stages/logs; scroll up to pause following, “移至最新” to return to the tail |
| 已完成 | Inspect output paths and view/download actions, then decide the next stage |
| 失敗 | Record job ID, displayed error and logs; fix the cause before resubmitting |
| Stop work | Click “取消任務” on a queued/running job and wait for cancellation status |
| Revisit a job | Search “任務紀錄” by name, ID or path, filter type/status, then click “開啟” |
| Full logs | Click “下載完整紀錄”; the on-screen view retains at most 5,000 recent lines |
| Connection fails after submission | Check job history for an existing job before submitting again |

Use “清除篩選” to reveal jobs hidden by filters. Switching tabs or closing a browser
is not server-job cancellation; the server must remain running. Closing its
foreground terminal or restarting the service affects work, and jobs do not
resume automatically after restart. Follow the
[update procedure](workflows.md#update-an-existing-deployment) before stopping it.

## 8. Confirm a successful first run

- The job is completed; retain its link, parameters, and actual output location.
- Frames, COLMAP cloud/cameras, or the trained model open in the appropriate
  viewer and correspond to the intended input.
- Keep/download important results as needed. Feed the actual new output into the
  next stage, rather than a similarly named old directory.
- Completion means the process finished; quality still needs acceptance checks.
  See [Evaluate](README.md#evaluate) for geometry and use the trainer's workflow
  for rendering metrics.

For detailed parameters see the [operator manual (Traditional Chinese)](../zh-TW/user-guide.md#二使用).
For connectivity, version, or service issues see [layered diagnosis](workflows.md#diagnose-the-correct-layer).
