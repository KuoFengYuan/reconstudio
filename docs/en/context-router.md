# AI Agent Context Router

[Documentation](README.md) · [Agent protocol](agent-guide.md) · [繁體中文](../zh-TW/context-router.md)

Read the agent protocol, this table, then only the relevant files. Full-codebase
recursive scans, broad traversal, and global grep/find are forbidden.
Paths in code spans are relative to the repository root unless marked runtime or
external; links point to the actual source files.

This project orchestrates experiments; it has no central `experiments/`, `models/`,
or `data/` source directory. Dataset/model paths come from each job; trainer
implementations live in configured external repositories. Do not scan those data
or repository trees to discover context.

<a id="agent-context-router"></a>

| Component | Core Path | Purpose & Scope | Inputs & Artifacts |
| :--- | :--- | :--- | :--- |
| Runtime & configuration | [run.sh](../../run.sh), [setup.sh](../../setup.sh), [local.env.example](../../local.env.example), [pipeline/config.py](../../pipeline/config.py) | Panel environment, tools, storage, concurrency | `local.env` → effective runtime settings |
| Requests & experiment records | [app.py](../../app.py), [web/services/forms.py](../../web/services/forms.py), [jobs.py](../../jobs.py) | HTTP entry, parameter validation, queue and progress | Form values → `<RECON_STUDIO_DATA>/jobs/<job-id>/{job.json,console.log}` |
| Data transfer & frame selection | [pipeline/gcs.py](../../pipeline/gcs.py), [pipeline/frames.py](../../pipeline/frames.py) | GCS transfer, ffmpeg extraction and blur filtering | Bucket/files/video → local files, `frames_<video>/`, quality reports |
| Masking & capture fusion | [pipeline/matte.py](../../pipeline/matte.py), [tools/sam_matte.py](../../tools/sam_matte.py), [pipeline/fusion.py](../../pipeline/fusion.py) | External SAM inference and masked capture staging | Photos/prompts → `no_bg/{masks,cutout}/`; fusion `images/`, `masks/` |
| Reconstruction | [pipeline/colmap/_run.py](../../pipeline/colmap/_run.py), [pipeline/colmap/](../../pipeline/colmap/), [pipeline/large_scene.py](../../pipeline/large_scene.py) | COLMAP stages; follow only the needed rig/GPS/layout helper | Photos/masks → `database.db`, sparse models, undistorted images |
| Depth & normals | [pipeline/depth.py](../../pipeline/depth.py), [pipeline/moge3.py](../../pipeline/moge3.py), [tools/moge3_preprocess.py](../../tools/moge3_preprocess.py) | LichtFeld MoGe-2 or separate MoGe-3 environment | Images → `depth/`, `normals/` beside `images/` |
| Model/backend integration | [pipeline/backends.py](../../pipeline/backends.py), [backends.example.json](../../backends.example.json) | Backend commands, parameter schemas and environment resolution | Optional `backends.json` → external `../GS-2M`, `../gsplat`, `../LichtFeld-Studio` defaults; overridable |
| Training & mesh | [pipeline/train.py](../../pipeline/train.py) | Adapt COLMAP scenes; invoke supported trainer/mesh backend | Undistorted scene → configured `model_path`, backend-specific splats/checkpoints/mesh |
| Scene partitioning | [pipeline/blocksplit.py](../../pipeline/blocksplit.py) | Split undistorted scenes into trainable blocks | COLMAP scene → `block_<ix>_<iy>/`, `manifest.json`, optional `_tiles/` |
| Evaluation & metrics | [pipeline/verify.py](../../pipeline/verify.py), [tools/verify_recon.py](../../tools/verify_recon.py), [tools/verify_recon.README.md](../../tools/verify_recon.README.md) | COLMAP observations, distortion, rig/EO and epipolar checks | Sparse model + optional matching DB/manifest → text metrics and exit status |
| Viewer & workspace UI | [templates/index.html](../../templates/index.html), [static/css/workspace.css](../../static/css/workspace.css), [static/js/workspace.js](../../static/js/workspace.js), [static/js/supersplat.js](../../static/js/supersplat.js), [web/routers/viewer.py](../../web/routers/viewer.py) | Workspace layout, model loading and viewing | Job/model paths → browser visualization |
| SuperSplat build | [tools/build_supersplat.sh](../../tools/build_supersplat.sh) | Version selection and integration patches referenced by the script | Upstream version + patches → generated `static/supersplat/`; do not scan bundle |
| Validation & deployment | [pyproject.toml](../../pyproject.toml), [.github/workflows/ci.yml](../../.github/workflows/ci.yml), [tests/](../../tests/), [infra/systemd/reconstudio.service](../../infra/systemd/reconstudio.service), [scripts/deploy-nginx-lan.sh](../../scripts/deploy-nginx-lan.sh) | Test configuration and LAN proxy setup; open only task-related tests | Code/config → check results or deployed proxy configuration |

`RECON_STUDIO_DATA` stores job metadata/logs, not necessarily images or trained
models. `run.sh` selects storage unless overridden; direct
`pipeline.config.Settings` defaults to `~/.recon_studio`. Read only the chosen
job's metadata to locate artifacts, not the whole storage root.

Maintain both language indexes when moving components. Read one language only
for a task; avoid loading duplicate context.
