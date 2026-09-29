# Recon Studio — Quick Start

[繁體中文](../zh-TW/README.md) · [Project home](../../README.md)

Turn photos or video into a COLMAP reconstruction, a 3D Gaussian Splatting model,
and, with a supported backend, a mesh. The local web panel provides forms,
progress logs, cancellation, model viewers, masking, and GCS transfer.

- [Detailed first-use guide](usage.md): field-by-field examples, expected results, viewer controls, browser acceptance, and troubleshooting.
- [Optional-tool tutorials](tools.md): GCS, masking, fusion, depth/normals, partitioning, and measurement.
- [Development, deployment, and daily-use workflows](workflows.md).
- [Agent routing protocol](agent-guide.md): runtime, baselines, experiment records, development rules.
- [Component context router](context-router.md): exact source paths and artifacts.
- [Detailed operator manual (Traditional Chinese)](../zh-TW/user-guide.md): installation, operation, and advanced configuration.

## Environment setup

Run from the repository root with Git and Conda installed:

```bash
./setup.sh
./run.sh --doctor
./run.sh
```

Setup installs the lightweight panel environment and preserves existing
`local.env`. Install ffmpeg, COLMAP, and the selected GPU backend separately;
resolve that stage's doctor warnings before running. Open the URL printed by
`run.sh`; the port is configurable. See the
[detailed installation instructions (Traditional Chinese)](../zh-TW/user-guide.md#一安裝第一次部署).

## After the panel starts

Open the printed URL, check “環境檢查”, then choose “我有影片”, “我有照片”, or an
existing model. For a first run, follow the [after-startup walkthrough](usage.md)
through submitting a job, checking progress, and opening/downloading its result.

## Run an experiment

In the panel, select local photos or extract video frames. Run COLMAP into a new
workspace, then pass its undistorted result to Training. Choose a ready backend
and a distinct model output path. Run Mesh only with a supporting backend.
Record the job ID, effective parameters, backend revision, and differences from
the baseline. Use the [context router](context-router.md) to locate stage defaults.
The project orchestrates external trainers; it does not provide a universal
training CLI or an in-repository model directory.

## Evaluate

Use a separate environment containing `pycolmap` and `numpy`; YAML manifests
also require `PyYAML`. Replace the example paths with the selected run's Python
interpreter, sparse model, and matching original database:

```bash
/path/to/evaluation-env/bin/python tools/verify_recon.py \
  /path/to/workspace/sparse/0 --db /path/to/workspace/database.db
```

Add `--manifest /path/to/case.yaml` for case-specific calibration/EO checks.
For undistorted models, add `--undistorted`: the epipolar check is skipped because
the database uses original image coordinates. Checks requiring absent DB/manifest
inputs are skipped. Exit `0` means executed checks passed; `1` means acceptance
failures. Investigate execution errors separately. See
[checker details (Traditional Chinese)](../../tools/verify_recon.README.md).

Compare metric excerpts and thresholds against the baseline and report skipped
checks. This checker covers geometry; rendering metrics use the selected
trainer's own evaluation workflow.

## Validate code changes

Follow [AGENT.md](../../AGENT.md). In the development environment:

```bash
pip install -e '.[dev]'
ruff check .
mypy pipeline/config.py
pytest
```

These checks validate software; they do not measure reconstruction quality.
