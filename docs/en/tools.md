# Optional Tools: Step-by-Step Tutorials

[Documentation](README.md) · [First-run workflow](usage.md) · [繁體中文](../zh-TW/tools.md)

Choose tools by the result you need; running every tool is unnecessary. Sources
refer to server files unless a viewer explicitly offers local file opening.
Chinese captions below identify actual interface controls.

| Goal | Tool | When to use it |
| :--- | :--- | :--- |
| Bring in cloud inputs or upload results | [GCS transfer](#gcs) | Before reconstruction / after output completion |
| Inspect photo quality or extracted frames | [Images](#images) | Before/after image processing |
| Mask photo backgrounds | [Image masking](#matte) | Before COLMAP/training that needs foreground masks |
| Reconstruct upright/flipped captures of one object together | [Fusion](#fusion) | After masking both passes, before COLMAP |
| Supply depth/normal supervision | [Depth/normals](#depth) | Once training images are fixed, before training |
| Train a large reconstruction as smaller scenes | [Partitioning](#blocks) | After COLMAP undistortion |
| Measure mesh dimensions or distance | [Measurement](#measure) | After a usable mesh exists |

`/Disk0/my-scan` and `gs://my-bucket` are placeholders; replace them with actual
authorized locations. Disabled tools need their environment configured via the
administrator, not unrelated backend settings.

<a id="gcs"></a>

## 1. GCS: download inputs to the server and upload results

**Entry:** “＋ 建立任務” → “切換功能” → “資料”. Download and upload forms are on this
function's page. The server needs configured gsutil/cloud access; web-login
credentials do not confer GCS permissions.

### Download example

| Field | Example | Check |
| :--- | :--- | :--- |
| 來源（GCS） | `gs://my-bucket/my-scan/images` | Actual bucket path, not a Cloud Console web URL |
| 下載到（本機） | `/Disk0/my-scan/images` | Local means the server, with writable parent storage |

1. Paste the source or choose it through “☁️ 瀏覽”.
2. Enter an explicit destination for a first run so you know where data will land.
   If left empty, follow the automatic-destination rule shown in the form.
3. Click “下載” and watch transfer logs in current work. Do not reconstruct partially
   downloaded inputs while transfer is still running.
4. Wait for completion and record the “已下載到” path.
5. Inspect photos through “圖片”, or choose extraction/COLMAP and use that server path.

**Common failures:** bucket listing can require project/authentication configuration;
permission-denied errors need the administrator to confirm cloud access. A destination
outside the browse root might require pasting its full path later. A repeated sync
normally transfers differences, but verify this run's logs and status rather than
assuming an old successful job means current files are synchronized.

### Upload example

1. Confirm the model/mesh job finished and select the actual outputs to retain.
2. In “上傳到 GCS”, enter e.g. `/Disk0/my-scan/model-run-01`. For multiple sources,
   click “瀏覽”, tick files/folders, then “套用所選”. Entering a directory and selecting
   it are separate actions.
3. Set “目的地(GCS)”, e.g. `gs://my-bucket/results/my-scan/run-01`; give experiments
   separate destinations to avoid confusing versions.
4. Click “上傳到 GCS”, wait for completion, and inspect the actual cloud destination,
   filenames and counts through cloud browsing/management tools.
5. Verify the cloud copy before deciding local retention. Upload completion does
   not prove you selected every dependency needed to use the artifact.

Single-directory transfers use sync; files/multiple selections use copy. Confirm
layout in logs and the cloud result. Textured OBJ needs MTL/images; datasets or
symlinks depending on source photos are not complete backups just because a
similarly named directory was uploaded.

<a id="images"></a>

## 2. Images: inspect inputs and processed results

1. Select “圖片” and enter the server image directory.
2. Keep “含子資料夾” enabled to include extracted-frame subfolders; disable to view one level.
3. Click “開啟圖片牆”, open a thumbnail, use arrows or keyboard left/right to move,
   and Esc to close the enlarged view.
4. Check project identity, orientation, sharp detail and shared content between
   neighboring views.
5. Recheck the actual extraction/resizing output, not only the original images.

**No images:** verify the photo directory rather than a workspace parent, nested
folders and supported decoding. HEIC may need additional decoding/conversion support;
report the format and error to the administrator. For transparent masking edges,
use the dedicated “檢視去背結果” view instead of judging a generic thumbnail background.

<a id="matte"></a>

## 3. Image masking: select the foreground in photos first

**Use for:** reconstructing a subject independently of its surroundings, especially
when it changes orientation across captures. This produces photo masks; deleting
points after training in SuperSplat is a [different workflow](usage.md#viewers).

### Field choices

| Field | Value/decision |
| :--- | :--- |
| images | Actual photos or a COLMAP workspace containing `images/` |
| 提示方式 | Ordered moving-camera sequence: box+tracking; fixed position: fixed box; text/exemplar requires a compatible model |
| 模型 | SAM 2 for tracking; choose compatible text/exemplar modes for SAM 3 according to the interface |
| 解析度 | Keep the selected size consistent across passes and downstream COLMAP images |
| 輸出 | COLMAP/fusion needs `masks/`; cutout+masks gives visual previews as well |
| GPU | Ready GPU assigned to the task |
| Edge/advanced options | Keep defaults initially; adjust for an observed boundary problem |

### First box-and-tracking run

1. Set `images`, choose SAM 2 and tracking mode. Inputs must be an ordered capture sequence.
2. Click “匡選物體”. Drag a box around the complete subject to retain, not its background.
3. Click to add a foreground point; Shift+click adds a background point. Click an
   existing marker to remove it, or use “復原” after a mistake.
4. Use previous/next images to check changing views and add anchors when needed.
   A seed can influence the whole sequence, not only the annotated photo; one box
   is not guaranteed to handle every change correctly.
5. Confirm mask output and click “開始去背”. Resolve missing prompts, incompatible
   modes or unavailable environment messages first.
6. Watch live previews; after completion open “檢視去背結果”. Switch checkerboard,
   white, black or green backgrounds; Space compares the original.
7. Inspect thin protrusions, edges, transparent/reflective areas. Open bad images
   for individual repair through the result view before feeding masks into reconstruction.

### Outputs and downstream handoff

- `no_bg/cutout/`: RGBA images for visual inspection or compatible external tools.
- `no_bg/masks/`: single-channel masks for COLMAP/training; do not substitute RGBA.
- With resizing enabled, results follow the `images_<size>/` copy and its `no_bg/`,
  not the original photo directory. Follow the job's reported destination.

For COLMAP, choose the matching photo version and set `MASKS_DIR` to `no_bg/masks`.
Enable “遮罩也用在抽特徵” to exclude background during camera solving. Separately,
“輸出的照片也去背” paints the **undistorted output dataset's** backgrounds black;
this modifies those output images, not just a preview or the original captures.
For already consistently resized inputs, “保持原樣” avoids introducing a size mismatch.

**Check:** image/mask relative names and dimensions must match. Fixed boxes drift
when subjects move around the frame; a bad tracking seed can spoil a sequence.
Use separate outputs for comparison settings, and overwrite only deliberately.

<a id="fusion"></a>

## 4. Fusion: reconstruct upright and flipped captures of one object

Fusion stages **two sets of photos/masks** for a joint COLMAP solve; it does not
merge two arbitrary 3D models.

1. Mask both captures separately at the same resolution and verify each has usable `no_bg/masks/`.
2. Select “融合”; enter e.g. `/Disk0/my-scan/up/images_2560/no_bg` and
   `/Disk0/my-scan/down/images_2560/no_bg` for the two groups.
3. Set fusion output to `/Disk0/my-scan/fusion-run-01`, or let the form derive it
   from the groups' common parent.
4. Click “整理並檢查”; inspect each group's image/mask counts and dimensions. Repair gaps first.
5. Click “帶入 COLMAP 表單”, then check the staged images/masks, workspace and options.
   Do not enable a synchronized multi-camera rig for separate passes; keep consistent
   dimensions. Review before submitting “啟動 COLMAP”.
6. Inspect whether both passes contribute registered cameras/images, rather than
   only one. Continue training from this workspace only after checking the result.

Output contains `images/<group>`, `masks/<group>` and `colmap/`. The first two link
back to sources rather than copy photos; do not move/delete the masking sources
while downstream work needs them. If passes stay separate, check masks and common
views. Missing connectivity may require additional input; staging success is not
proof that joint reconstruction succeeded.

<a id="depth"></a>

## 5. Depth/normals: prepare additional training supervision

**Prerequisite:** the selected trainer uses these inputs and the chosen generator
is available. A first basic training run can leave supervision disabled.

1. Decide which images training actually uses. For an undistorted COLMAP dataset,
   choose its `images/`, not the original distorted/differently sized captures.
2. Select “深度/法線” and an available engine. LichtFeld MoGe-2 and standalone
   MoGe-3 run in different environments.
3. Set `images`, choose depth, normals or both under “產生內容”, and select GPU.
4. Keep model/bit-depth defaults initially and click “產生深度/法向量圖”. A first run
   may download a model; follow logs rather than submitting duplicates.
5. Confirm selected `depth/`/`normals/` outputs alongside `images/`, with matching filenames.
6. Return to training with the same source. Expand the backend's depth/normal
   supervision options, enable the required losses, review weights, then submit.

**Common misunderstanding:** generating files alone does not enable the training
loss; enabling a loss does not establish that this generator produced its inputs.
Recheck alignment after changing images/resolution. Existing outputs are skipped
by default; decide on separate output or deliberate overwrite for new settings
instead of mixing old maps into a new experiment.

<a id="blocks"></a>

## 6. Partition a reconstructed scene into trainable blocks

1. Finish COLMAP/undistortion and verify the overall scene before choosing “分塊”.
2. Set `source` to that workspace/dataset. An explicit first-run destination such
   as `/Disk0/my-scan/blocks-run-01` is easy to track; leaving it blank follows the
   displayed `source/blocks/<time>` rule.
3. For a first attempt, enable automatic partitioning and keep advanced thresholds.
   Image tiling creates a crop pool; disabling it links original images. Choose
   according to training requirements.
4. Leave `region` blank for the whole scene. For a subset, use the COLMAP viewer's
   training-region selection and verify coordinates refer to this model, not another scene.
5. Click “開始分塊”. Once completed, select a block and “檢視這塊” to inspect cameras/coverage.
6. Click “帶入訓練”, confirm `source` names the intended `block_<ix>_<iy>`, choose a
   separate model output for that block, and submit. Partitioning does not automatically
   train every block or merge them into a final model.

Outputs include each block's `images/` and `sparse/`, an overall `manifest.json`,
and `_tiles/` when tiling. Retain dependencies and linked sources. Block size and
buffer use model coordinates; interpret them as metres only after confirming metric
alignment. Use “只裁切不分塊” when you need just a cropped region without a grid.

<a id="measure"></a>

## 7. Measurement: distance, dimensions, and actual millimetres

### Measure two surface points

Open Mesh Viewer → enable “量尺” → click two surface points → read distance →
“清除” to measure again. This measures only the selected points. The viewer's unit
buttons change labels, not calibration.

### Generate width, height, depth, and three views

1. Use “量測高寬深” on a completed Mesh job, or choose “量測”, select a server mesh
   and click “開始量測”.
2. Confirm front, side and top views and the dimension table appear.
3. For wrong orientation, inspect “PCA 主軸” versus “原始 XYZ”: PCA finds principal
   directions while XYZ follows original coordinates.
4. Select the height axis and adjust width/depth swap or flips until the views
   match your intended orientation. Tilted objects can otherwise yield dimensions
   along an unintended direction.
5. Keep units when scale is unknown. Enter a known `mm_per_unit` in “實際尺寸換算”
   and click “套用” only when a reliable conversion exists.

**Scale example:** if 2 model units are known to correspond to 100 mm, the factor
is `100 / 2 = 50` mm/unit. Enter `50`, not `100`, the text `mm`, or a viewer unit label.
Enter `1` only if file coordinates are already millimetres. This illustrates conversion,
not automatic accuracy: establish scale from reliable physical references/calibration.

The page converts displayed measurements; generating a separately scaled mesh uses
the Mesh calibration/scaling workflow. Background fragments can enlarge the bounds,
so confirm/clean the measured model first.

<a id="handoff"></a>

## 8. Which path goes into the next tool?

| Finished operation | Next input | Avoid confusing it with |
| :--- | :--- | :--- |
| GCS download | Reported server destination | Original `gs://` path passed directly to COLMAP |
| Frame extraction | Prefilled photo output root | Original video file |
| Photo masking | Matching images plus `no_bg/masks/` | RGBA cutouts used as single-channel masks |
| Fusion | Staged common `images/`/`masks/` via form handoff | One capture's old workspace |
| COLMAP | Workspace/dataset with undistorted outputs | Sparse PLY alone or original photo folder |
| Depth/normals | Same training dataset matching the maps | Differently sized photo set |
| Partitioning | Selected `block_<ix>_<iy>` | Parent containing all blocks treated as one scene |
| Training | Completed model directory/result actions | Incompatible output from another backend |

Recheck source, destination and selected function before each submission, then
verify paths reported at completion. Return to the [full startup-to-result guide](usage.md)
for the main workflow, status interpretation and troubleshooting.
