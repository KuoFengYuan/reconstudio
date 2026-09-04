"""Shared web-layer pieces: the Jinja templates handle, the `_page` render
helper, and the UI/form constants the routers seed forms with.

Kept dependency-light (only `pipeline.config`, no jobs/pipeline-function imports)
so any router or service can import it without an import cycle.
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pipeline.config import settings

BASE = Path(__file__).resolve().parent.parent          # reconstudio/

# settings-derived shortcuts used across routers
BROWSE_ROOT = settings.browse_root   # input directory-browser root
DEST_ROOT = settings.dest_root       # where GCS downloads may land
FFMPEG_BIN = settings.ffmpeg_bin
GSUTIL_BIN = settings.gsutil_bin
GCS_ROOT = settings.gcs_root         # GCS browser start prefix ('' = list all buckets)

templates = Jinja2Templates(directory=str(BASE / "templates"))
templates.env.add_extension("jinja2.ext.do")   # enables {% do dict.update(...) %} in templates


def _fmt_dt(epoch, fmt="%Y-%m-%d %H:%M:%S"):
    """Jinja filter: epoch seconds -> local 'YYYY-MM-DD HH:MM:SS' (blank if unset)."""
    try:
        return time.strftime(fmt, time.localtime(float(epoch))) if epoch else ""
    except (TypeError, ValueError):
        return ""


templates.env.filters["dt"] = _fmt_dt


def _page(request: Request, name: str, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name=name, context=ctx)


# --- form pre-fill defaults / enums (mirror the pipeline module defaults) --- #
COLMAP_DEFAULTS = {
    "CAMERA_MODEL": "OPENCV", "MAX_FEATURES": "4096", "CAMERA_MODE": "per_folder",
    "MATCHER": "vocab", "SEQ_OVERLAP": "10", "NUM_MATCHES": "50",
    # sequential_matcher loop detection: ignore retrieval hits closer than this many
    # images in sequence order, so the candidate budget goes to real revisits.
    # "0" = COLMAP default (no restriction).
    "SEQ_LOOP_MIN_INDEX_DIST": "0",
    # multi-camera rig: off by default; RIG_MODE picks how frames are grouped.
    "RIG_ENABLE": "0", "RIG_MODE": "auto", "RIG_REGEX": "",
    "RIG_REF_CAMERA": "", "RIG_GPS_TOL": "0.5",
    "GUIDED_MATCHING": "1", "MAPPER": "global", "DATASET_NAME": "training_dataset",
    "FORCE": "0", "NESTED_LAYOUT": "1", "VOCAB_TREE": "", "VOCAB_TREE_URL": "",
    # 影像解析度 is a single dropdown (its <option> values carry the cap or "keep"), so the
    # resize cap no longer needs its own form default here; forms.py derives resize_max.
    # spatial_matcher (MATCHER=spatial)
    "SPATIAL_MAX_NEIGHBORS": "50", "SPATIAL_MAX_DISTANCE": "100", "SPATIAL_IGNORE_Z": "1",
    # pose_prior_mapper (MAPPER=pose_prior): GPS position std in metres
    "PRIOR_STD_X": "auto", "PRIOR_STD_Y": "auto", "PRIOR_STD_Z": "auto",
    # Surveyor exterior-orientation CSV (optional): positions overwrite the EXIF pose
    # priors; ω/φ/κ can only enter the DB as the gravity column (global_mapper's
    # rotation averaging). Rig matching + gravity default on, RA_USE_GRAVITY opt-in.
    "POSE_PRIOR_CSV": "", "POSE_PRIOR_CRS": "twd97_tm2_121",
    "POSE_PRIOR_RIG_MATCH": "1", "POSE_PRIOR_GRAVITY": "1",
    "RA_USE_GRAVITY": "0", "RA_MAX_ROTATION_ERROR_DEG": "10",
    # global_mapper multi-component (COLMAP 4.3): every connected component becomes its
    # own sparse/N. That is COLMAP's default, so the knob is the opt-out — ticking it
    # reconstructs only the largest component. GM_MIN_MODEL_SIZE blank = COLMAP's 3.
    "GM_SINGLE_MODEL": "0", "GM_MIN_MODEL_SIZE": "",
    # GPU bundle adjustment (incremental / pose_prior / hierarchical) — on by default.
    # global_mapper isn't in this set: its Ceres BA defaults to GPU on its own.
    "MAPPER_BA_GPU": "1",
    # BA solver backend (all four mappers): "ceres" or "caspar" (GPU, ~1-2 orders
    # faster; needs SIMPLE_RADIAL / PINHOLE cameras)
    "BA_BACKEND": "ceres",
    # Which GPU(s) all COLMAP stages use (CUDA_VISIBLE_DEVICES). "" = every GPU (default);
    # e.g. "0", "1", "0,1".
    "COLMAP_GPU": "",
    # GPS metric alignment (model_aligner) — off by default, needs GPS in the inputs
    "GPS_ALIGN": "0", "GPS_ALIGN_MAX_ERROR": "3.0",
    # --- hierarchical-3d-gaussians large-scene method (MATCHER=custom, MAPPER=hierarchical) ---
    "CM_N_SEQ": "0", "CM_N_QUAD": "10", "CM_N_LOOP": "5", "CM_N_GPS": "25",
    "CM_LOOP_MATCHES": "", "FOCAL_FACTOR": "", "SIFT_MAX_IMAGE_SIZE": "auto",
    "MAX_IMAGE_SIZE": "", "MASKS_DIR": "", "MASK_FEATURES": "0", "CUTOUT_IMAGES": "0",
    "SIMPLIFY": "0", "SIMPLIFY_MULT_MIN_DIST": "10",
    "REORIENT": "0", "REORIENT_TARGET_MED_DIST": "20", "REORIENT_UPSCALE": "0",
    "HM_LEAF_MAX_NUM_IMAGES": "", "HM_IMAGE_OVERLAP": "", "HM_NUM_WORKERS": "",
}
FRAMES_DEFAULTS = {"FPS": "1", "MODE": "percentile", "KEEP_PCT": "70", "THRESHOLD": ""}
ENUMS = {
    "CAMERA_MODE": ["per_folder", "single"],
    "MATCHER": ["sequential", "vocab", "both", "spatial", "custom", "exhaustive"],
    "RIG_MODE": ["auto", "folder", "regex", "gps"],
    "MAPPER": ["global", "incremental", "pose_prior", "hierarchical"],
    "BA_BACKEND": ["ceres", "caspar"],
}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".mkv", ".avi"}
# formats the standalone mesh viewer can load (three.js loaders vendored under
# static/three/addons/loaders). .gltf is excluded: its external .bin/texture refs
# wouldn't resolve through the single-file /api/viewer/meshfile endpoint.
MESH_EXTS = {".ply", ".obj", ".stl", ".glb"}
# formats the self-hosted SuperSplat viewer opens (3DGS point clouds; .ply here is
# a gaussian cloud like point_cloud.ply, not a mesh)
SPLAT_EXTS = {".ply", ".splat", ".ksplat", ".spz", ".sog"}
# images the standalone gallery (/gallery) shows. Browsers render these natively...
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
# ...plus TIFF, which has no native browser support -> transcoded to a JPEG preview
# (the COLMAP/aerial inputs are often TIFF), like the viz photo panel does.
IMAGE_EXTS_ALL = IMAGE_EXTS | {".tif", ".tiff"}
