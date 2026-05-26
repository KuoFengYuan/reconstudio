"""Form → validated pipeline params. Each raises ValueError with a user-facing
(zh) message that the routers turn into the _error.html fragment.
"""
from __future__ import annotations

from pathlib import Path

from web.shared import ENUMS


def scene_label(path: str) -> str:
    """Short, dataset-distinguishing label = '<workspace>/<leaf>'.

    Train/mesh inputs share generic leaf names (training_dataset_global_mapper,
    or a model dir named after the backend), so the leaf alone can't tell two
    datasets apart. The parent (the COLMAP workspace) is what differs, e.g.
    /Disk0/.../0520_iron_13mm_colmap/gs2m -> '0520_iron_13mm_colmap/gs2m'.
    """
    p = Path(path)
    return f"{p.parent.name}/{p.name}" if p.parent.name else p.name


def parse_marker(form: dict, spec: dict) -> dict | None:
    """Build the marker-scaling config when「提供 marker」is ticked, else None.

    The board geometry is taken from the backend's configured `marker_defaults`
    (so the user never types numbers — that's the whole point). A per-job text
    field may override any value, but blank means "use the configured default"."""
    if not form.get("marker_enable"):
        return None
    d = dict(spec.get("marker_defaults") or {})
    if not d:
        raise ValueError("此 backend 未設定標定板規格 (marker_defaults);請在 backends.json 補上後再用。")

    def pick(key):  # per-job override (text field) wins over the configured default
        v = (form.get("marker_" + key) or "").strip()
        return v if v else d.get(key)
    try:
        sx, sy = int(pick("squares_x")), int(pick("squares_y"))
        sq, mk = float(pick("square_mm")), float(pick("marker_mm"))
    except (TypeError, ValueError):
        raise ValueError("marker 覆寫值需為數字 (方格數為整數、邊長 mm 為數值),或留空用預設規格。") from None
    if min(sx, sy) < 2 or min(sq, mk) <= 0:
        raise ValueError("marker 參數不合理:方格數需 ≥ 2,邊長 (mm) 需 > 0。")
    if mk >= sq:
        raise ValueError("marker 邊長應小於方格邊長 (ChArUco marker 嵌在方格內)。")
    return {"enable": True, "squares_x": sx, "squares_y": sy,
            "square_mm": sq, "marker_mm": mk,
            "dict": str(pick("dict") or "DICT_5X5_100").strip()}


def build_colmap_params(image_root: str, workspace: str, folders: list[str],
                        stages: list[str], form: dict) -> dict:
    def g(key: str, default: str) -> str:
        v = (form.get(key) or "").strip()
        return v if v else default

    params = {
        "image_root": image_root, "workspace": workspace, "folders": folders,
        "stages": stages,
        "camera_model": g("CAMERA_MODEL", "OPENCV"),
        "camera_mode": g("CAMERA_MODE", "per_folder"),
        "max_features": g("MAX_FEATURES", "4096"),
        "matcher": g("MATCHER", "both"),
        "seq_overlap": g("SEQ_OVERLAP", "10"),
        "num_matches": g("NUM_MATCHES", "50"),
        "guided_matching": "1" if form.get("GUIDED_MATCHING") else "0",
        "mapper": g("MAPPER", "global"),
        "dataset_name": g("DATASET_NAME", "training_dataset"),
        "vocab_tree": (form.get("VOCAB_TREE") or "").strip() or None,
        "vocab_tree_url": (form.get("VOCAB_TREE_URL") or "").strip() or None,
        "force": bool(form.get("FORCE")),
        "layout": form.get("layout") or "auto",
        "resize": form.get("resize") or "fullhd",
        # spatial_matcher (MATCHER=spatial)
        "spatial_max_neighbors": g("SPATIAL_MAX_NEIGHBORS", "50"),
        "spatial_max_distance": g("SPATIAL_MAX_DISTANCE", "100"),
        "spatial_ignore_z": "1" if form.get("SPATIAL_IGNORE_Z") else "0",
        # pose_prior_mapper (MAPPER=pose_prior): GPS position std in metres
        "prior_std_x": g("PRIOR_STD_X", "3.0"),
        "prior_std_y": g("PRIOR_STD_Y", "3.0"),
        "prior_std_z": g("PRIOR_STD_Z", "5.0"),
        "prior_robust_loss": "1",
        "ba_gpu": bool(form.get("MAPPER_BA_GPU")),
        # GPS metric alignment (model_aligner) — off unless ticked AND GPS is present
        "gps_align": bool(form.get("GPS_ALIGN")),
        "gps_align_type": g("GPS_ALIGN_TYPE", "enu"),
        "gps_align_max_error": g("GPS_ALIGN_MAX_ERROR", "3.0"),
    }
    for key, allowed in (("camera_mode", ENUMS["CAMERA_MODE"]),
                         ("matcher", ENUMS["MATCHER"]), ("mapper", ENUMS["MAPPER"]),
                         ("layout", ["auto", "single", "multi", "nested"]),
                         ("resize", ["keep", "fullhd"]),
                         ("gps_align_type", ["enu", "ecef", "enu-plane", "plane"])):
        if params[key] not in allowed:
            raise ValueError(f"{key} must be one of {allowed}")
    for key in ("max_features", "seq_overlap", "num_matches", "spatial_max_neighbors"):
        if not str(params[key]).isdigit():
            raise ValueError(f"{key} must be an integer")
    for key in ("spatial_max_distance", "prior_std_x", "prior_std_y", "prior_std_z",
                "gps_align_max_error"):
        try:
            float(params[key])
        except ValueError:
            raise ValueError(f"{key} must be a number") from None
    return params
