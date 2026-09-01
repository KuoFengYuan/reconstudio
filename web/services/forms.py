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


def build_blocksplit_params(source: str, out_dir: str, form: dict) -> dict:
    """Validate the 分塊 (block-split) form. Numbers stay strings (house style:
    the pipeline converts); `tile` is the one real bool."""
    def g(key: str, default: str) -> str:
        v = (form.get(key) or "").strip()
        return v if v else default

    params = {
        "source": source, "out_dir": out_dir,
        "block_size": g("block_size", "500"), "buffer": g("buffer", "120"),
        "region": (form.get("region") or "").strip(),
        "tile": bool(form.get("tile")),
        "max_tile_px": g("max_tile_px", "8192"),
        "jpeg_quality": g("jpeg_quality", "95"),
        "min_tile_obs": g("min_tile_obs", "8"),
        "min_images": g("min_images", "15"), "workers": g("workers", "4"),
        "vis_thresh": g("vis_thresh", "0.1667"),
        "auto": form.get("auto") is not None,
        "rebalance": bool(form.get("rebalance")),
        "max_images": g("max_images", "200"),
    }
    for key in ("block_size", "buffer"):
        try:
            float(params[key])
        except ValueError:
            raise ValueError(f"{key} 需為數字") from None
    if float(params["block_size"]) <= 0:
        raise ValueError("block_size 需 > 0")
    if float(params["buffer"]) < 0:
        raise ValueError("buffer 需 ≥ 0")
    for key in ("max_tile_px", "min_tile_obs", "min_images",
                "jpeg_quality", "workers", "max_images"):
        if not str(params[key]).isdigit():
            raise ValueError(f"{key} 需為正整數")
    if not 1 <= int(params["jpeg_quality"]) <= 100:
        raise ValueError("jpeg_quality 需在 1–100")
    if int(params["max_tile_px"]) < 256:
        raise ValueError("max_tile_px 需 ≥ 256")
    if not 1 <= int(params["workers"]) <= 32:
        raise ValueError("workers 需在 1–32")
    try:
        vt = float(params["vis_thresh"])
    except ValueError:
        raise ValueError("vis_thresh 需為數字") from None
    if not 0 < vt <= 1:
        raise ValueError("vis_thresh 需在 0–1（論文用 1/6 ≈ 0.1667）")
    if int(params["max_images"]) < int(params["min_images"]):
        raise ValueError("max_images 需 ≥ min_images（自適應分塊的每塊相機上限）")
    if params["region"]:
        from pipeline.blocksplit import parse_region  # zh errors, shared with pipeline
        parse_region(params["region"])
    return params


def build_colmap_params(image_root: str, workspace: str, folders: list[str],
                        stages: list[str], form: dict) -> dict:
    def g(key: str, default: str) -> str:
        v = (form.get(key) or "").strip()
        return v if v else default

    params = {
        "image_root": image_root, "workspace": workspace, "folders": folders,
        "stages": stages,
        "camera_model": g("CAMERA_MODEL", "SIMPLE_RADIAL"),
        "camera_mode": g("CAMERA_MODE", "per_folder"),
        "max_features": g("MAX_FEATURES", "4096"),
        "matcher": g("MATCHER", "vocab"),
        # --- multi-camera rig (see pipeline/colmap/_rig.py) ---
        "rig_enable": bool(form.get("RIG_ENABLE")),
        "rig_mode": g("RIG_MODE", "auto"),
        "rig_regex": g("RIG_REGEX", ""),
        "rig_ref_camera": g("RIG_REF_CAMERA", ""),
        "rig_gps_tol": g("RIG_GPS_TOL", "0.5"),
        "seq_overlap": g("SEQ_OVERLAP", "10"),
        "num_matches": g("NUM_MATCHES", "50"),
        "guided_matching": "1" if form.get("GUIDED_MATCHING") else "0",
        "mapper": g("MAPPER", "global"),
        "dataset_name": g("DATASET_NAME", "training_dataset"),
        "vocab_tree": (form.get("VOCAB_TREE") or "").strip() or None,
        "vocab_tree_url": (form.get("VOCAB_TREE_URL") or "").strip() or None,
        "force": bool(form.get("FORCE")),
        "layout": form.get("layout") or "auto",
        # 影像解析度 is ONE dropdown carrying "keep" or a numeric longest-side cap; it's
        # split into the pipeline's two params (mode + cap) just after this dict is built.
        "resize": form.get("resize") or "1920",
        "resize_max": "1920",
        # spatial_matcher (MATCHER=spatial)
        "spatial_max_neighbors": g("SPATIAL_MAX_NEIGHBORS", "50"),
        "spatial_max_distance": g("SPATIAL_MAX_DISTANCE", "100"),
        "spatial_ignore_z": "1" if form.get("SPATIAL_IGNORE_Z") else "0",
        # pose_prior_mapper (MAPPER=pose_prior): GPS position std in metres
        "prior_std_x": g("PRIOR_STD_X", "3.0"),
        "prior_std_y": g("PRIOR_STD_Y", "3.0"),
        "prior_std_z": g("PRIOR_STD_Z", "5.0"),
        "prior_robust_loss": "1",
        # Surveyor exterior-orientation CSV (ID,E,N,H,OMEGA,PHI,KAPPA): positions overwrite
        # the EXIF pose priors, ω/φ/κ enter only as the gravity column (global_mapper).
        "pose_prior_csv": (form.get("POSE_PRIOR_CSV") or "").strip(),
        "pose_prior_crs": g("POSE_PRIOR_CRS", "twd97_tm2_121"),
        "pose_prior_rig_match": bool(form.get("POSE_PRIOR_RIG_MATCH")),
        "pose_prior_gravity": bool(form.get("POSE_PRIOR_GRAVITY")),
        "ra_use_gravity": bool(form.get("RA_USE_GRAVITY")),
        "ra_max_rotation_error_deg": g("RA_MAX_ROTATION_ERROR_DEG", "10"),
        "ba_gpu": bool(form.get("MAPPER_BA_GPU")),
        "ba_backend": g("BA_BACKEND", "ceres"),
        "colmap_gpu": (form.get("COLMAP_GPU") or "").strip(),
        # GPS metric alignment (model_aligner) — off unless ticked AND GPS is present
        "gps_align": bool(form.get("GPS_ALIGN")),
        "gps_align_type": g("GPS_ALIGN_TYPE", "enu"),
        "gps_align_max_error": g("GPS_ALIGN_MAX_ERROR", "3.0"),
        # --- h3dgs large-scene method (MATCHER=custom, MAPPER=hierarchical) ---
        "cm_n_seq": g("CM_N_SEQ", "0"), "cm_n_quad": g("CM_N_QUAD", "10"),
        "cm_n_loop": g("CM_N_LOOP", "5"), "cm_n_gps": g("CM_N_GPS", "25"),
        "cm_loop_matches": (form.get("CM_LOOP_MATCHES") or "").strip(),
        "focal_factor": (form.get("FOCAL_FACTOR") or "").strip(),
        "sift_max_image_size": (form.get("SIFT_MAX_IMAGE_SIZE") or "auto").strip(),
        "max_image_size": (form.get("MAX_IMAGE_SIZE") or "").strip(),
        "masks_dir": (form.get("MASKS_DIR") or "").strip().rstrip("/"),
        "simplify": bool(form.get("SIMPLIFY")),
        "simplify_mult_min_dist": g("SIMPLIFY_MULT_MIN_DIST", "10"),
        "reorient": bool(form.get("REORIENT")),
        "reorient_target_med_dist": g("REORIENT_TARGET_MED_DIST", "20"),
        "reorient_upscale": g("REORIENT_UPSCALE", "0"),
        # hierarchical_mapper partition knobs (blank = COLMAP default)
        "hm_leaf_max_num_images": (form.get("HM_LEAF_MAX_NUM_IMAGES") or "").strip(),
        "hm_image_overlap": (form.get("HM_IMAGE_OVERLAP") or "").strip(),
        "hm_num_workers": (form.get("HM_NUM_WORKERS") or "").strip(),
        # region subset (⬚ 選訓練範圍 → 重跑 COLMAP): blank = whole dataset. The rectangle
        # is measured on region_model's frame, so the two travel together.
        "region": (form.get("region") or "").strip(),
        "region_model": (form.get("region_model") or "").strip(),
        "region_buffer": g("region_buffer", "0"),
        "region_vis_thresh": g("region_vis_thresh", "0.1667"),
    }
    # Split the single 影像解析度 selection into the pipeline's mode + cap. "keep" = no
    # resize; a number = downscale to that longest side. "fullhd" is the pre-merge legacy
    # value -> treat as the 1920 default. The digit/enum checks below then validate both.
    sel = "1920" if params["resize"] == "fullhd" else params["resize"]
    if sel == "keep":
        params["resize"], params["resize_max"] = "keep", "1920"
    else:
        params["resize"], params["resize_max"] = "fullhd", sel
    for key, allowed in (("camera_mode", ENUMS["CAMERA_MODE"]),
                         ("matcher", ENUMS["MATCHER"]), ("mapper", ENUMS["MAPPER"]),
                         ("ba_backend", ENUMS["BA_BACKEND"]),
                         ("layout", ["auto", "single", "multi", "nested"]),
                         ("resize", ["keep", "fullhd"]),
                         ("gps_align_type", ["enu", "ecef", "enu-plane", "plane"])):
        if params[key] not in allowed:
            raise ValueError(f"{key} must be one of {allowed}")
    for key in ("max_features", "seq_overlap", "num_matches", "spatial_max_neighbors",
                "cm_n_seq", "cm_n_quad", "cm_n_loop", "cm_n_gps", "resize_max"):
        if not str(params[key]).isdigit():
            raise ValueError(f"{key} must be an integer")
    for key in ("spatial_max_distance", "prior_std_x", "prior_std_y", "prior_std_z",
                "ra_max_rotation_error_deg",
                "gps_align_max_error", "simplify_mult_min_dist",
                "reorient_target_med_dist", "reorient_upscale"):
        try:
            float(params[key])
        except ValueError:
            raise ValueError(f"{key} must be a number") from None
    # optional numerics: validated only when provided (blank = "use COLMAP default")
    for key, label in (("max_image_size", "MAX_IMAGE_SIZE"),
                       ("hm_leaf_max_num_images", "HM_LEAF_MAX_NUM_IMAGES"),
                       ("hm_image_overlap", "HM_IMAGE_OVERLAP")):
        if params[key] and not params[key].isdigit():
            raise ValueError(f"{label} must be a positive integer (or blank)")
    # SIFT size also takes "auto" (follow the undistort cap) and -1 (COLMAP's own
    # default). Without this the "auto" the panel now pre-fills would be rejected
    # here before the pipeline ever saw it.
    sift = params["sift_max_image_size"].lower()
    if sift and sift not in ("auto", "-1") and not sift.isdigit():
        raise ValueError(
            "SIFT_MAX_IMAGE_SIZE must be a positive integer, 'auto', -1, or blank")
    if params["hm_num_workers"]:  # -1 = auto, so allow a leading minus
        try:
            int(params["hm_num_workers"])
        except ValueError:
            raise ValueError("HM_NUM_WORKERS must be an integer (-1 = auto, or blank)") from None
    if params["focal_factor"]:
        try:
            float(params["focal_factor"])
        except ValueError:
            raise ValueError("FOCAL_FACTOR must be a number (or blank)") from None
    # loop closure anchors: space/comma-separated integers, in (a b) pairs
    try:
        loop_ints = [int(x) for x in params["cm_loop_matches"].replace(",", " ").split()]
    except ValueError:
        raise ValueError("CM_LOOP_MATCHES must be space/comma-separated integers") from None
    if len(loop_ints) % 2 != 0:
        raise ValueError("CM_LOOP_MATCHES needs an even count of integers (a b pairs)")
    # region subset: reject a half-filled pair here (the pipeline checks again, but the
    # form is where the user can still fix it), and parse the rectangle for its zh error.
    if params["region"] or params["region_model"]:
        if not params["region"]:
            raise ValueError("region_model 有設但 region 是空的 — 請框選一個範圍,或把兩者都清空。")
        if not params["region_model"]:
            raise ValueError("region 有設但 region_model 是空的 — 需要指定這個範圍是在哪個 "
                             "COLMAP 模型的座標系上框的。")
        from pipeline.blocksplit import parse_region  # zh errors, shared with pipeline
        parse_region(params["region"])
        for key, label in (("region_buffer", "外擴"), ("region_vis_thresh", "vis_thresh")):
            try:
                float(params[key])
            except ValueError:
                raise ValueError(f"{label} 需為數字") from None
        if float(params["region_buffer"]) < 0:
            raise ValueError("外擴需 ≥ 0")
        if not 0 < float(params["region_vis_thresh"]) <= 1:
            raise ValueError("vis_thresh 需在 0–1（論文用 1/6 ≈ 0.1667）")
    return params
