"""COLMAP orchestrator (`run_colmap`).

Stages (skippable via `stages`, re-runnable via force):
  stage -> extract -> match -> calibrate(global only) -> mapper -> simplify
        -> align -> undistort -> reorient
Idempotency uses sentinel files / output checks; the banners match `log()` so the
panel's stage parser is unchanged.

`run_colmap` resolves a `_Ctx` (all config + the mutable shared state: `img_root`,
which NESTED staging and the FullHD resize rebase; `lines`/`gps_present`/`gps_opts`,
filled mid-flight) then calls the stage functions in order. Each `_stage_*` is a
thin wrapper around one COLMAP sub-command with its own skip/sentinel guard.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from ..config import settings
from ..runner import PipelineError, Runner
from ._eo import inject_eo_priors, map_names_to_eo, parse_eo_csv, resolve_crs
from ._gps import gps_coverage, image_gps_latlonalt, inject_pose_priors
from ._intrinsics import (
    apply_to_database,
    camera_folders,
    discover_calibrations,
    match_to_cameras,
)
from ._layout import IMAGE_EXTS, list_image_names, list_images, resolve_layout
from ._resize import resize_to_fullhd, resize_workers
from ._rig import (
    RIG_DEFAULTS,
    build_staging,
    group_auto,
    group_images,
    recommend_ref_camera,
    summarize,
    write_rig_config,
)

COLMAP_STAGES = ["stage", "extract", "rig", "gps_inject", "match", "calibrate", "mapper",
                 "simplify", "align", "undistort", "reorient"]
# colmap / ffmpeg binaries (PATH by default; override via COLMAP_BIN / FFMPEG_BIN).
COLMAP_BIN = settings.colmap_bin

COLMAP_DEFAULTS = {
    "vocab_tree": str(Path.home() / ".cache/colmap/vocab_tree_faiss_flickr100K_words256K.bin"),
    "vocab_tree_url": "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin",
    # OPENCV (k1,k2,p1,p2) describes a real lens far better than SIMPLE_RADIAL's single
    # parameter, which matters for metric/aerial work. Note the Caspar BA backend only
    # accepts SIMPLE_RADIAL/PINHOLE and silently skips every image otherwise — _setup
    # warns when that combination is selected.
    "camera_model": "OPENCV", "max_features": "4096", "camera_mode": "per_folder",
    "matcher": "vocab", "seq_overlap": "10", "num_matches": "50",
    "guided_matching": "1", "mapper": "global", "dataset_name": "training_dataset",
    "force": False, "nested_layout": False, "resize": "fullhd",
    # resize longest-side cap for the "fullhd" option (default 1920). Raise for higher-res
    # training images; the re-encode also produces clean TIFFs that dodge the undistort OIIO
    # TIFF-writer segfault the raw aerial TIFFs trigger.
    "resize_max": "1920",
    # spatial_matcher (large GPS scenes): match only GPS-near images.
    "spatial_max_neighbors": "50", "spatial_max_distance": "100", "spatial_ignore_z": "1",
    # GPS metric alignment via model_aligner (optional, off by default): rewrite the
    # sparse model into a local ENU frame in real-world metres.
    "gps_align": False, "gps_align_type": "enu", "gps_align_max_error": "3.0",
    # pose_prior_mapper: GPS position uncertainty (metres). Consumer GPS ~3-5 m; RTK ~0.02.
    # "auto" picks the uncertainty from the source of the priors: an EO CSV is
    # post-processed airborne POS, so metre-level values would tell BA to ignore it.
    # A number pins it. See _resolve_prior_std.
    "prior_std_x": "auto", "prior_std_y": "auto", "prior_std_z": "auto",
    "prior_robust_loss": "1",
    # --- surveyor-supplied exterior orientation (EO) CSV ---
    # Path to a CSV of ID,EASTING,NORTHING,ELLIPSOID HEIGHT,OMEGA,PHI,KAPPA (the adjusted
    # EO an aerial vendor ships alongside the imagery). Positions overwrite the EXIF-derived
    # pose priors in the DB; ω/φ/κ can only enter as the `gravity` column (COLMAP has no
    # rotation prior), which global_mapper uses via POSE_PRIOR_GRAVITY below. Empty = off.
    "pose_prior_csv": "",
    # CRS the CSV's EASTING/NORTHING are in — see _eo.TM_PRESETS ("twd97_tm2_121" =
    # EPSG:3826). Inverse-projected to WGS84 so COLMAP builds the local ENU frame itself.
    # "cartesian" writes the projected metres straight in instead (bakes in the grid scale
    # factor and curvature; only sensible for small blocks).
    "pose_prior_crs": "twd97_tm2_121",
    # Propagate each CSV station to the other heads of a multi-camera rig, matched by
    # nearest EXIF GPS. Needed for full prior coverage when the CSV covers one head only.
    "pose_prior_rig_match": True,
    # Write ω/φ/κ as a gravity (down-in-camera) prior for the exactly-name-matched images.
    # Only global_mapper consumes it, via --GlobalMapper.ra_use_gravity (RA_USE_GRAVITY).
    "pose_prior_gravity": True,
    # global_mapper: use the DB gravity priors in rotation averaging.
    "ra_use_gravity": False, "ra_max_rotation_error_deg": "10",
    # GPU bundle adjustment for the incremental / pose_prior mappers (big speedup; on by default).
    "ba_gpu": True,
    # BA solver backend: "ceres" (default; the ba_gpu flag above then chooses CPU vs
    # cuDSS-GPU Ceres for incremental/pose_prior) or "caspar" (COLMAP's SymForce GPU
    # solver, ~1-2 orders of magnitude faster — but only supports the SIMPLE_RADIAL /
    # PINHOLE camera models; on any other model Caspar skips every image). Applies to
    # incremental / pose_prior / global / hierarchical (all four take --Mapper.ba_*_backend
    # or --GlobalMapper.ba_backend since COLMAP main@2fe2b41 / #4484).
    "ba_backend": "ceres",
    # Which GPU(s) COLMAP uses across ALL stages (extract / match / mapper / BA / Caspar
    # / aligner), applied via CUDA_VISIBLE_DEVICES. "" = COLMAP default (-1 = every GPU).
    # Set to e.g. "0", "1", or "0,1" to pin specific device(s).
    "colmap_gpu": "",
    # --- hierarchical-3d-gaussians large-scene method (MATCHER=custom, MAPPER=hierarchical) ---
    # custom matcher (make_colmap_custom_matcher): per-view match counts + loop anchors.
    "cm_n_seq": "0", "cm_n_quad": "10", "cm_n_loop": "5", "cm_n_gps": "25",
    "cm_loop_matches": "",
    # feature_extractor focal seed for uncalibrated large scenes (h3dgs uses 0.5); "" = COLMAP default.
    "focal_factor": "",
    # feature_extractor SIFT extraction longest-side cap. "auto" (default) follows the
    # undistort cap, bounded by SIFT_MAX_PX — COLMAP's own default silently extracts at
    # 3200 px, which throws away the resolution a high-res run deliberately keeps. Use a
    # number to pin it (lower = coarser, more viewpoint-robust), or -1 for COLMAP's default.
    "sift_max_image_size": "auto",
    # image_undistorter longest-side cap (h3dgs uses 2048); "" = no cap.
    "max_image_size": "",
    # optional foreground masks: undistort them through the same cameras as the images.
    "masks_dir": "",
    # --- region subset: re-run the whole pipeline over an area picked in the viewer ---
    # `region` is the "minx,miny,maxx,maxy" rectangle the ⬚ 選訓練範圍 tool produces,
    # measured on `region_model`'s own horizontal axes. When both are set the image list
    # is filtered to the images that area needs (see _region.py) and every stage from
    # extraction to undistortion runs on that subset — unlike blocksplit, which crops an
    # already-reconstructed model, this re-solves the geometry for the area, so align
    # pins it back into the same reference frame and the result stays comparable.
    # `region_buffer` (外擴) grows only the point mask feeding the visibility test.
    "region": "", "region_model": "", "region_buffer": "0", "region_vis_thresh": "0.1667",
    # simplify_images (drop pose-outlier cameras + trim observations) and auto_reorient
    # (PCA gravity-align + scale). Both off unless ticked; h3dgs runs both.
    "simplify": False, "simplify_mult_min_dist": "10",
    "reorient": False, "reorient_target_med_dist": "20", "reorient_upscale": "0",
    # hierarchical_mapper partition knobs (COLMAP 4.2 --HierarchicalMapper.*); "" = COLMAP default
    # (leaf_max_num_images 500, image_overlap 50, num_workers -1 = auto).
    "hm_leaf_max_num_images": "", "hm_image_overlap": "", "hm_num_workers": "",
    # --- multi-camera rig (see _rig.py) ---
    **RIG_DEFAULTS,
}


def _download(url: str, dest: Path, r: Runner) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last: Exception | None = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(url) as resp, tmp.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            tmp.unlink(missing_ok=True)
            time.sleep(2)
    raise RuntimeError(f"vocab tree download failed: {last}")


@dataclass
class _Ctx:
    """Resolved config + mutable shared state for one run_colmap invocation.

    `img_root` is rebased by the NESTED staging and the FullHD resize; `lines`,
    `gps_present` and `gps_opts` are filled by the image-list / GPS-coverage steps
    before the COLMAP stages consume them.
    """
    r: Runner
    d: dict
    ws: Path
    img_root: str
    folders: list[str]
    stages: list[str]
    nested: bool
    layout_name: str
    force: bool
    mapper: str
    camera_mode: str
    matcher: str
    vocab_tree: Path
    fullhd: bool
    resize_max: str
    gps_align: bool
    gps_align_type: str
    gps_align_max_error: str
    ba_gpu: bool
    ba_backend: str
    gpu: str
    focal_factor: str
    sift_max_image_size: str
    max_image_size: str
    masks_dir: str
    region: str
    region_model: str
    region_buffer: float
    region_vis_thresh: float
    simplify: bool
    reorient: bool
    rig_enable: bool
    rig_mode: str
    rig_regex: str
    rig_ref_camera: str
    rig_gps_tol: float
    cm_n_seq: str
    cm_n_quad: str
    cm_n_loop: str
    cm_n_gps: int
    cm_loop_ints: list[int]
    hm_leaf: str
    hm_overlap: str
    hm_workers: str
    eo_csv: str
    eo_crs: str
    eo_rig_match: bool
    eo_gravity: bool
    ra_use_gravity: bool
    db: Path
    lst: Path
    dense_dir: Path
    lines: list[str] = field(default_factory=list)
    gps_present: bool = False
    gps_opts: list[str] = field(default_factory=list)
    # image name -> (EO row, exact-name-match?) from the EO CSV; empty when no CSV.
    eo_map: dict = field(default_factory=dict)
    # img_root before the FullHD resize rebases it — the originals still carry EXIF GPS,
    # which the resized copies don't, so gps_inject reads priors from here.
    orig_root: str = ""
    # rig_config.json written by the rig staging stage; consumed by rig_configurator.
    rig_config: Path | None = None
    # staged relative name -> original relative name (rig restaging renames
    # files, and the EO CSV is keyed by the vendor's original filenames).
    rig_orig_names: dict[str, str] = field(default_factory=dict)
    # camera id -> (model, params) taken from a vendor calibration, when the
    # dataset shipped one. Empty means nothing was calibrated.
    rig_intrinsics: dict[str, tuple[str, list[float]]] = field(default_factory=dict)
    # True once a vendor calibration has been written into the database.
    intrinsics_applied: bool = False

    def stage_on(self, s: str) -> bool:
        return s in self.stages

    def need(self, path: Path | str) -> bool:
        return self.force or not Path(path).exists()


def _setup(p: dict, r: Runner) -> _Ctx:
    """Validate inputs, resolve layout, ensure the vocab tree / colmap binary, make
    the workspace dirs, and return the populated `_Ctx`. Raises before any stage."""
    d = {**COLMAP_DEFAULTS, **{k: v for k, v in p.items() if v is not None}}
    img_root = p["image_root"]
    ws = Path(p["workspace"])
    folders = list(p.get("folders") or [])
    stages = list(p.get("stages") or COLMAP_STAGES)
    mapper, camera_mode, matcher = d["mapper"], d["camera_mode"], d["matcher"]
    force = bool(d["force"])
    vocab_tree = Path(d["vocab_tree"])
    # FullHD resize: cap longest side at 1920 for feature extraction + undistorted output.
    fullhd = str(d.get("resize")) == "fullhd"

    if mapper not in ("global", "incremental", "pose_prior", "hierarchical"):
        raise ValueError(f"MAPPER must be 'global', 'incremental', 'pose_prior', or "
                         f"'hierarchical' (got: {mapper})")
    if camera_mode not in ("per_folder", "single"):
        raise ValueError(f"CAMERA_MODE must be 'per_folder' or 'single' (got: {camera_mode})")
    ba_backend = str(d["ba_backend"]).lower()
    if ba_backend not in ("ceres", "caspar"):
        raise ValueError(f"BA_BACKEND must be 'ceres' or 'caspar' (got: {ba_backend})")
    if ba_backend == "caspar" and str(d["camera_model"]).upper() not in ("SIMPLE_RADIAL", "PINHOLE"):
        r.log(f"WARNING: Caspar only supports SIMPLE_RADIAL / PINHOLE cameras, but "
              f"CAMERA_MODEL={d['camera_model']} — Caspar will skip every image "
              f"(no BA happens). Use SIMPLE_RADIAL or PINHOLE.")
    if matcher not in ("sequential", "vocab", "both", "spatial", "custom"):
        raise ValueError(f"MATCHER must be 'sequential', 'vocab', 'both', 'spatial', or "
                         f"'custom' (got: {matcher})")

    # EO CSV (optional): validate path + CRS up front — a wrong projection would put every
    # prior kilometres away, and we'd rather fail before extraction than after.
    eo_csv = str(d["pose_prior_csv"]).strip()
    eo_crs = ""
    if eo_csv:
        if not Path(eo_csv).is_file():
            raise FileNotFoundError(f"pose_prior_csv not found: {eo_csv}")
        eo_crs = resolve_crs(str(d["pose_prior_crs"]))
        if bool(d["ra_use_gravity"]) and mapper != "global":
            r.log(f"WARNING: RA_USE_GRAVITY is on but MAPPER={mapper} — only global_mapper "
                  "uses gravity priors (in rotation averaging); it will be ignored.")
    elif bool(d["ra_use_gravity"]):
        r.log("WARNING: RA_USE_GRAVITY is on but no POSE_PRIOR_CSV is set — nothing writes "
              "the gravity column, so rotation averaging will fall back to no gravity.")

    # Region subset (optional): validate the rectangle and the model it was drawn on
    # before extraction. Both must be present — a region without a model has no frame to
    # be measured in, and a model without a region would silently filter nothing.
    region = str(d["region"]).strip()
    region_model = str(d["region_model"]).strip()
    region_buffer = region_vis_thresh = 0.0
    if region or region_model:
        if not region:
            raise ValueError("region_model 有設但 region 是空的 — 請框選一個範圍,或把兩者都清空。")
        if not region_model:
            raise ValueError("region 有設但 region_model 是空的 — 需要指定這個範圍是在哪個 "
                             "COLMAP 模型的座標系上框的。")
        if not (Path(region_model) / "images.bin").is_file():
            raise FileNotFoundError(
                f"region_model 不是 COLMAP 模型目錄（缺 images.bin）: {region_model}")
        from ._region import parse_region  # numpy path: import only when used
        parse_region(region)                     # raises zh ValueError on a bad rectangle
        # A region run MUST get its own workspace. Pointed at the original one it would
        # find that database.db / sparse/0 already exist, skip extraction and mapping on
        # the sentinel checks, and hand back the FULL-dataset model as if the region had
        # been honoured — or with force=1, overwrite the original reconstruction. Both are
        # silent, so refuse here rather than let either happen.
        if (ws / "database.db").exists():
            raise FileNotFoundError(
                f"workspace 已經有 database.db（{ws}）— 框選範圍重跑必須用一個新的 workspace。"
                "沿用舊的會讓抽特徵/mapper 因為 sentinel 直接跳過,結果拿到的是「整個資料集」"
                "的舊模型(看起來像 region 沒生效);開 FORCE 則會覆蓋掉原本的重建。"
                "請改一個新的 workspace 路徑(例如原本的加上 _region 後綴)。")
        if Path(region_model).resolve() == ws.resolve() or ws.resolve() in Path(
                region_model).resolve().parents:
            raise ValueError(
                f"region_model（{region_model}）在 workspace（{ws}）底下 — "
                "重跑會覆寫它,而它正是這次選片要讀的參考模型。請把 workspace 換成別的路徑。")
        try:
            region_buffer = float(str(d["region_buffer"]) or 0)
            region_vis_thresh = float(str(d["region_vis_thresh"]) or 0.1667)
        except ValueError:
            raise ValueError("region_buffer / region_vis_thresh 需為數字") from None
        if region_buffer < 0:
            raise ValueError("region_buffer 需 ≥ 0")
        if not 0 < region_vis_thresh <= 1:
            raise ValueError("region_vis_thresh 需在 0–1（論文用 1/6 ≈ 0.1667）")

    if not Path(img_root).is_dir():
        raise FileNotFoundError(f"image_root not found: {img_root}")
    folders, nested, layout_name = resolve_layout(
        img_root, folders, str(p.get("layout") or "auto"), bool(d["nested_layout"]),
        workspace=p.get("workspace"))
    shown = "<root>" if folders == [""] else " ".join(folders)
    r.log(f"layout={layout_name}  folders={shown}  nested={int(nested)}")
    for f in folders:
        if f and not (Path(img_root) / f).is_dir():
            raise FileNotFoundError(f"subfolder not found: {Path(img_root) / f}")

    if not vocab_tree.exists():
        r.log(f"vocab tree not found, downloading: {d['vocab_tree_url']} -> {vocab_tree}")
        _download(d["vocab_tree_url"], vocab_tree, r)
    if not shutil.which(COLMAP_BIN):
        raise RuntimeError(f"colmap not found: '{COLMAP_BIN}' (set COLMAP_BIN or add to PATH)")

    # GPU selection: pin all COLMAP stages to the requested device(s) via
    # CUDA_VISIBLE_DEVICES on the runner (applied to every child process). Empty =
    # COLMAP default (every visible GPU).
    gpu = str(d["colmap_gpu"]).strip()
    if gpu:
        r.default_env["CUDA_VISIBLE_DEVICES"] = gpu
        r.log(f"COLMAP GPU: CUDA_VISIBLE_DEVICES={gpu} (all stages)")

    dense_dir = ws / f"{d['dataset_name']}_{mapper}_mapper"
    (ws / "sparse").mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)
    db = ws / "database.db"
    lst = ws / "image_list.txt"
    r.log(f"=== run started {time.strftime('%F %T')} | folders: {' '.join(folders)} ===")

    return _Ctx(
        r=r, d=d, ws=ws, img_root=img_root, folders=folders, stages=stages,
        nested=nested, layout_name=layout_name, force=force, mapper=mapper,
        camera_mode=camera_mode, matcher=matcher, vocab_tree=vocab_tree, fullhd=fullhd,
        resize_max=(str(d["resize_max"]).strip() or "1920"),
        gps_align=bool(d["gps_align"]), gps_align_type=str(d["gps_align_type"]),
        gps_align_max_error=str(d["gps_align_max_error"]),
        ba_gpu=bool(d["ba_gpu"]),   # GPU bundle adjustment (incremental / pose_prior only)
        ba_backend=str(d["ba_backend"]).lower(),  # "ceres" or "caspar" (all mapper modes)
        gpu=gpu,                    # CUDA_VISIBLE_DEVICES for all COLMAP stages ("" = all)
        # h3dgs large-scene method options
        focal_factor=str(d["focal_factor"]).strip(),
        sift_max_image_size=str(d["sift_max_image_size"]).strip(),
        max_image_size=str(d["max_image_size"]).strip(),
        masks_dir=str(d["masks_dir"]).strip(),
        region=region, region_model=region_model,
        region_buffer=region_buffer, region_vis_thresh=region_vis_thresh,
        simplify=bool(d["simplify"]), reorient=bool(d["reorient"]),
        rig_enable=bool(d["rig_enable"]), rig_mode=str(d["rig_mode"]).strip() or "folder",
        rig_regex=str(d["rig_regex"]).strip(),
        rig_ref_camera=str(d["rig_ref_camera"]).strip(),
        rig_gps_tol=float(str(d["rig_gps_tol"]) or 0.5),
        cm_n_seq=str(d["cm_n_seq"]), cm_n_quad=str(d["cm_n_quad"]), cm_n_loop=str(d["cm_n_loop"]),
        cm_n_gps=int(str(d["cm_n_gps"]) or 0),
        cm_loop_ints=[int(x) for x in str(d["cm_loop_matches"]).replace(",", " ").split()],
        hm_leaf=str(d["hm_leaf_max_num_images"]).strip(),
        hm_overlap=str(d["hm_image_overlap"]).strip(),
        hm_workers=str(d["hm_num_workers"]).strip(),
        eo_csv=eo_csv, eo_crs=eo_crs,
        eo_rig_match=bool(d["pose_prior_rig_match"]),
        eo_gravity=bool(d["pose_prior_gravity"]),
        ra_use_gravity=bool(d["ra_use_gravity"]),
        db=db, lst=lst, dense_dir=dense_dir,
    )


def _stage_nested(c: _Ctx) -> None:
    # 0. stage (NESTED_LAYOUT): flatten <group>/<video>/*.jpg -> staging/<group>/<video>_<file> symlinks
    if c.stage_on("stage") and not c.nested:
        c.r.log("skip stage (not a nested layout)")
    if c.stage_on("stage") and c.nested:
        stage_root = c.ws / "staging"
        sentinel = c.ws / ".stage.done"
        abs_root = Path(os.path.realpath(c.img_root))
        if c.need(sentinel):
            c.r.banner(f"stage nested layout -> {stage_root} (NESTED_LAYOUT=1)")
            if c.force:
                shutil.rmtree(stage_root, ignore_errors=True)
            for grp in c.folders:
                (stage_root / grp).mkdir(parents=True, exist_ok=True)
                grp_total = 0
                for vdir in sorted([x for x in (abs_root / grp).iterdir() if x.is_dir()]):
                    vcount = 0
                    for img in sorted([f for f in vdir.iterdir()
                                       if f.is_file() and f.suffix.lower() in IMAGE_EXTS]):
                        link = stage_root / grp / f"{vdir.name}_{img.name}"
                        if link.is_symlink() or link.exists():
                            link.unlink()
                        link.symlink_to(img)
                        vcount += 1
                    if vcount > 0:
                        c.r.log(f"  {grp}/{vdir.name}: {vcount} images")
                        grp_total += vcount
                c.r.log(f"  {grp} total: {grp_total}")
                if grp_total == 0:
                    raise FileNotFoundError(f"no images found under group: {grp}")
            sentinel.touch()
        else:
            c.r.log("skip stage (sentinel exists; set FORCE=1 to redo)")
        c.img_root = str(stage_root)
        c.r.log(f"rebased IMG_ROOT={c.img_root}")


def _region_subset(c: _Ctx, lines: list[str]) -> list[str]:
    """Narrow `lines` to the images a picked viewer region needs.

    Names come out of `region_model`, so they must be the names this run's image list
    uses. They agree when both point at the same folder layout; when they don't, a bare
    name (basename) match is tried before giving up, because the common mismatch is a
    model built from a differently-nested staging of the same files. Either way the
    counts are logged rather than silently applied — a wholesale name mismatch would
    otherwise read as "the region is empty", which sends you hunting the wrong bug.
    """
    from ._region import parse_region, select_region_images  # numpy path: import when used

    names, _stats = select_region_images(
        Path(c.region_model), parse_region(c.region), buffer=c.region_buffer,
        vis_thresh=c.region_vis_thresh, log=c.r.log)
    want = set(names)
    by_base: dict[str, str] = {}
    for n in names:
        by_base.setdefault(PurePosixPath(n).name, n)

    kept: list[str] = []
    exact = base = 0
    for ln in lines:
        if ln in want:
            kept.append(ln)
            exact += 1
        elif PurePosixPath(ln).name in by_base:
            kept.append(ln)
            base += 1
    if base:
        c.r.log(f"region 選片: {base} 張是以檔名（非完整路徑）對上的 — "
                "region_model 與這次的資料夾結構不同,已照檔名採用。")
    missing = len(want) - exact - base
    if missing > 0:
        c.r.log(f"region 選片: {missing} 張在 region_model 裡選中,但這次的影像清單裡找不到"
                "（來源資料夾不同或檔案已移除）。")
    if not kept:
        raise FileNotFoundError(
            f"region 選中 {len(want)} 張影像,但沒有一張出現在這次的影像清單裡 — "
            f"region_model（{c.region_model}）與 image_root（{c.img_root}）"
            "很可能不是同一批影像。")
    c.r.log(f"region 選片: 影像清單 {len(lines)} → {len(kept)} 張")
    return kept


def _build_image_list(c: _Ctx) -> None:
    # image_list.txt (paths relative to img_root)
    lines: list[str] = []
    for f in c.folders:
        lines += list_images(Path(c.img_root) / f, f)
    if c.region:
        lines = _region_subset(c, lines)
    c.lst.write_text("\n".join(lines) + ("\n" if lines else ""))
    c.r.log(f"image_list: {len(lines)} images across {len(c.folders)} folder(s): {' '.join(c.folders)}")
    if not lines:
        raise FileNotFoundError("no images found")
    c.lines = lines


def _resolve_eo_csv(c: _Ctx) -> None:
    """Parse the EO CSV and map its rows onto the image list. Runs BEFORE the coverage
    check (which counts EO-covered images as covered) and before the FullHD resize, so
    the rig matching reads EXIF off the originals."""
    if not c.eo_csv:
        return
    rows = parse_eo_csv(Path(c.eo_csv))
    c.r.log(f"EO CSV: {len(rows)} rows from {c.eo_csv} (CRS={c.eo_crs})")
    c.eo_map = map_names_to_eo(c.lines, rows, c.img_root, c.eo_crs,
                               c.eo_rig_match, c.r,
                               orig_names=c.rig_orig_names)
    if not c.eo_map:
        raise RuntimeError(
            f"EO CSV {c.eo_csv} matched none of the {len(c.lines)} images. The CSV `ID` "
            "column must equal the image filename stem (rig-mate images are then matched "
            "by EXIF GPS). Check the ID format or set POSE_PRIOR_CSV='' to disable.")


def _check_gps_coverage(c: _Ctx) -> None:
    # EXIF GPS coverage. COLMAP reads EXIF GPS into the DB pose priors that
    # spatial_matcher / pose_prior_mapper / model_aligner consume. The GPS pipeline
    # requires GPS on EVERY image — a frame without a prior can't be spatially matched
    # or anchored — so any GPS option needs FULL coverage, checked here before any work
    # runs. The FullHD resize still runs (so we keep the downscale); it just preserves
    # the EXIF GPS through the re-encode (see preserve_exif below).
    # An EO CSV supplies priors directly, so its images count as covered without needing
    # EXIF at all (and its positions overwrite any EXIF-derived prior). Only the images
    # the CSV missed still have to fall back to EXIF, so scan just those.
    rest = [ln for ln in c.lines if ln not in c.eo_map]
    n_rest, _ = gps_coverage(c.img_root, rest, c.r, resize_workers()) if rest else (0, 0)
    n_total = len(c.lines)
    n_gps = len(c.eo_map) + n_rest
    c.gps_present = (n_total > 0 and n_gps == n_total)   # GPS flow valid only at 100%
    c.gps_opts = [name for name, on in (("MATCHER=spatial", c.matcher == "spatial"),
                                        ("MAPPER=pose_prior", c.mapper == "pose_prior"),
                                        ("GPS_ALIGN", c.gps_align)) if on]
    if c.gps_opts:
        if not c.gps_present:
            raise RuntimeError(
                f"GPS option(s) selected ({', '.join(c.gps_opts)}) but only {n_gps}/{n_total} "
                "inputs have a pose prior (EO CSV or EXIF GPS) — the GPS pipeline needs one "
                "on EVERY image, so it aborts before any work runs. Provide GPS-tagged photos "
                "for all inputs (or an EO CSV covering them), or switch to a non-GPS setup "
                "(MATCHER=vocab/both, MAPPER=global/incremental, GPS 對齊 off). Note: video "
                "frames carry no per-frame GPS — it lives in the container, not the frames.")
        c.r.log(f"pose priors on all {n_total} inputs -> GPS flow enabled "
                f"({', '.join(c.gps_opts)}); "
                + (f"{len(c.eo_map)} from the EO CSV, {n_rest} from EXIF"
                   if c.eo_map else
                   "JPEG priors read by COLMAP, TIFF priors via gps_inject")
                + (" (FullHD resize keeps JPEG EXIF; TIFF GPS is read from the originals)"
                   if c.fullhd else ""))
    elif n_gps:
        c.r.log(f"note: {n_gps}/{n_total} inputs have EXIF GPS but no GPS option selected; "
                "running normally" + (" (the FullHD resize will drop the GPS)" if c.fullhd else ""))


def _maybe_resize_fullhd(c: _Ctx) -> None:
    # FullHD: physically downscale the inputs to FullHD copies and run the entire
    # pipeline on those (image_list paths stay relative, so they remain valid). When the
    # GPS flow is on, the resize grafts each JPEG original's EXIF back so its GPS survives;
    # TIFF GPS can't be grafted (offset-based IFD) and is injected separately from the
    # originals by gps_inject, so the EXIF-stripped TIFF copies are fine.
    if c.fullhd:
        # keep EXIF GPS through the re-encode when any GPS flow needs it — including
        # the custom matcher's optional GPS-neighbour pairs (not a hard-fail option,
        # so it's not in gps_opts, but it still reads GPS off the resized images).
        preserve = bool(c.gps_opts) or (c.matcher == "custom" and c.cm_n_gps > 0)
        c.img_root = resize_to_fullhd(c.img_root, c.lines, c.ws, c.force, c.r,
                                      preserve_exif=preserve, max_size=c.resize_max)


# What the database's features are tied to. Change any of these and the stored
# keypoints no longer describe the images the later stages will read: the image
# path (a resize rebases it), the SIFT size the coordinates were detected at, and
# the camera model/mode baked into the cameras table.
def _extract_fingerprint(c: _Ctx) -> dict[str, str]:
    return {
        "image_path": str(c.img_root),
        "sift_max_image_size": _resolve_sift_size(c) or "default",
        "camera_model": str(c.d["camera_model"]).upper(),
        "camera_mode": c.camera_mode,
        "layout": c.layout_name,
        "max_features": str(c.d["max_features"]),
    }


def _extract_is_stale(c: _Ctx) -> list[str]:
    """Which fingerprint fields changed since the database was built.

    Without this, flipping 影像解析度 away from "keep" silently reuses features
    detected on the full-resolution originals while every later stage reads the
    resized copies — the keypoint coordinates then describe a different image and
    nothing in the run says so. Skipping extraction is only safe when the inputs
    it depends on are unchanged.
    """
    stamp = c.ws / ".extract.json"
    if not stamp.exists():
        return []                       # nothing recorded: leave the old behaviour
    try:
        was = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    now = _extract_fingerprint(c)
    return [f"{k}: {was.get(k)!r} -> {now[k]!r}"
            for k in now if str(was.get(k)) != now[k]]


def _stage_extract(c: _Ctx) -> None:
    # 1. feature_extractor
    if c.stage_on("extract"):
        stale = _extract_is_stale(c)
        if stale and not c.need(c.db):
            c.r.log("re-extracting: the database was built with different inputs, so "
                    "its keypoints do not describe the images this run will read")
            for line in stale:
                c.r.log(f"  {line}")
        if c.need(c.db) or stale:
            c.r.banner(f"feature_extractor (CAMERA_MODE={c.camera_mode}, layout={c.layout_name}) -> {c.db}")
            if c.layout_name == "single":
                cam = ["--ImageReader.single_camera", "1"]            # one flat folder = 1 camera
            elif c.camera_mode == "per_folder":
                cam = ["--ImageReader.single_camera_per_folder", "1"]
            else:
                cam = ["--ImageReader.single_camera", "1"]
            # h3dgs seeds the focal length for uncalibrated large scenes (factor 0.5);
            # only passed when set, so normal runs keep COLMAP's default behavior.
            if c.focal_factor:
                cam += ["--ImageReader.default_focal_length_factor", c.focal_factor]
            # Feature-extraction size: COLMAP downscales the longest side to this before
            # detecting SIFT. Its default (-1) behaves like a 3200 cap, so the downscale is
            # invisible unless set. Option is FeatureExtraction.max_image_size on COLMAP
            # 4.x (was SiftExtraction.* on older builds). Pass only when set; higher = more
            # detail but slower and can OOM on huge aerials, lower = coarser/viewpoint-robust.
            sift_px = _resolve_sift_size(c)
            sift_size = ["--FeatureExtraction.max_image_size", sift_px] if sift_px else []
            if sift_px:
                c.r.log(f"FeatureExtraction.max_image_size={sift_px} "
                        "(COLMAP default -1 behaves like 3200)")
            c.r.run([COLMAP_BIN, "feature_extractor", "--database_path", str(c.db),
                     "--image_path", c.img_root, "--image_list_path", str(c.lst), *cam,
                     "--ImageReader.camera_model", str(c.d["camera_model"]),
                     "--SiftExtraction.max_num_features", str(c.d["max_features"]),
                     *sift_size])
            (c.ws / ".extract.json").write_text(
                json.dumps(_extract_fingerprint(c), indent=2), encoding="utf-8")
        else:
            c.r.log("skip extract (database.db exists; set FORCE=1 to redo)")


# SiftGPU is handed EffMaxImageSize() * (1 << -min(0, first_octave)) as its -maxd
# (feature/sift.cc), and first_octave defaults to -1, so whatever we ask for is
# DOUBLED before it reaches the GPU. 8192 therefore already means -maxd 16384,
# which is the practical ceiling; above it extraction fails on large aerials.
SIFT_MAX_PX = 8192


# Position-prior uncertainty in metres, by where the priors came from. A surveyor's
# adjusted EO is post-processed airborne POS; declaring consumer-GPS metres for it
# makes BA weight it to nothing, which silently wastes the priors.
PRIOR_STD_EO = (0.05, 0.05, 0.10)
PRIOR_STD_GPS = (3.0, 3.0, 5.0)


def _resolve_prior_std(c: _Ctx) -> tuple[float, float, float]:
    """(x, y, z) metres for the pose-position priors.

    Each axis is taken from the form when it is a number, and otherwise ("auto",
    the default) from the prior source: EO CSV -> centimetre-level, bare EXIF GPS
    -> metre-level. Mixing is allowed, so a user can pin only Z.
    """
    src = PRIOR_STD_EO if c.eo_csv else PRIOR_STD_GPS
    out = []
    for key, fallback in zip(("prior_std_x", "prior_std_y", "prior_std_z"), src,
                             strict=True):
        raw = str(c.d[key]).strip().lower()
        if raw in ("", "auto"):
            out.append(fallback)
        else:
            out.append(float(raw))
    return (out[0], out[1], out[2])


def _resolve_sift_size(c: _Ctx) -> str:
    """The --FeatureExtraction.max_image_size to pass, or "" to leave it default.

    "auto" ties feature extraction to the resolution the run actually keeps: it is
    incoherent to train on 8192 px imagery while solving the poses at COLMAP's
    silent 3200 px default. It follows the undistort cap when there is one, else
    goes as high as is safe, and never exceeds SIFT_MAX_PX.

    Deliberately NOT the default: for a wide-baseline oblique rig, coarser
    features are often the more robust choice across the nadir/oblique viewpoint
    gap, so raising this is a judgement call rather than a free win.
    """
    want = (c.sift_max_image_size or "").strip().lower()
    if not want:
        return ""
    if want != "auto":
        return c.sift_max_image_size
    cap = SIFT_MAX_PX
    if c.max_image_size:
        try:
            undistort_px = int(c.max_image_size)
        except ValueError:
            undistort_px = 0
        if undistort_px > 0:
            cap = min(undistort_px, SIFT_MAX_PX)
    c.r.log(f"SIFT_MAX_IMAGE_SIZE=auto -> {cap} px "
            f"(undistort cap={c.max_image_size or 'none'}, ceiling={SIFT_MAX_PX})")
    return str(cap)


def _stage_rig_stage(c: _Ctx) -> None:
    """0b. Multi-camera rig: restage images so COLMAP can group them into frames.

    Runs BEFORE the image list / extraction because it rebases `img_root`: the
    names COLMAP records in the database must already be `<camera>/<frame_key>`,
    which is the only thing `rig_configurator` groups by. The FullHD resize that
    may follow keeps relative paths, so the layout survives it.
    """
    if not c.rig_enable:
        return
    # Same listing as _build_image_list: names must be "<folder>/<file>", because
    # the folder IS the camera. list_image_names() would return bare filenames
    # from img_root itself, which for a multi layout is empty (images live one
    # level down) and would leave every mode with nothing to group.
    names: list[str] = []
    for folder in c.folders:
        names += list_images(Path(c.img_root) / folder, folder)
    # No folder-count check here: auto also splits cameras by filename prefix, so
    # a flat dataset can still be a rig. Whether the split worked is judged by the
    # grouping below, which aborts when no exposure is covered by every camera.
    c.r.log(f"rig: {len(names)} images across {len(c.folders)} folder(s): "
            f"{' '.join(f or '.' for f in c.folders)}")

    gps_map = None
    if c.rig_mode == "gps":
        gps_map = {}
        for n in names:
            hit = image_gps_latlonalt(Path(c.img_root) / n)
            if hit:
                gps_map[n] = (hit[0], hit[1])
        c.r.log(f"rig: EXIF GPS found for {len(gps_map)}/{len(names)} images")

    try:
        if c.rig_mode == "auto":
            # auto reports which field it picked; that guess must be visible.
            grouping, notes = group_auto(names)
            for line in notes:
                c.r.log(line)
        else:
            grouping = group_images(names, c.rig_mode, regex=c.rig_regex,
                                    gps=gps_map, gps_tol=c.rig_gps_tol)
    except ValueError as exc:
        raise PipelineError(f"rig grouping failed: {exc}") from exc

    for line in summarize(grouping):
        c.r.log(line)
    if not grouping.complete_frames():
        raise PipelineError(
            "rig: no frame is covered by every camera — the grouping is wrong, so a "
            "rig would constrain nothing. Check 遮罩/regex 設定 (rig_mode="
            f"{c.rig_mode!r}, regex={c.rig_regex!r}) or turn the rig off.")

    # Reference sensor: honour an explicit choice, else recommend one. The EO CSV
    # is the strongest available signal (the head it measures is the head whose
    # angles may become gravity priors), so read its IDs when there is one.
    ref_camera = c.rig_ref_camera
    if not ref_camera:
        eo_stems: set[str] = set()
        if c.eo_csv:
            try:
                eo_stems = {row.stem for row in parse_eo_csv(Path(c.eo_csv))}
            except Exception as exc:                    # noqa: BLE001
                # A bad CSV is _resolve_eo_csv's problem to report; here it only
                # costs us the better recommendation.
                c.r.log(f"rig: could not read the EO CSV for the reference-sensor "
                        f"recommendation ({exc}); falling back to exposure count")
        ref_camera, why = recommend_ref_camera(grouping, eo_stems)
        c.r.log(f"rig: auto-selected reference sensor '{ref_camera}' — {why}")
        if not eo_stems:
            c.r.log("rig: that choice is a weak guess; set 參考鏡頭 (RIG_REF_CAMERA) "
                    "to the nadir/most stable head if you know which it is")

    # Known interior orientation, if the dataset shipped a calibration certificate.
    # Scanned for rather than configured: vendors drop it in with the imagery under
    # arbitrary names, and a document either parses as a calibration or it does not.
    intrinsics: dict[str, tuple[str, list[float]]] = {}
    cals, src = discover_calibrations(Path(c.orig_root or c.img_root))
    if cals:
        model = str(c.d["camera_model"]).upper()
        hit, missed = match_to_cameras(cals, grouping.cameras)
        c.r.log(f"intrinsics: {len(cals)} calibrated head(s) found in {Path(src).name}")
        for cam, cal in sorted(hit.items()):
            params = cal.colmap_params(model)
            intrinsics[cam] = (model, params)
            cx, cy = cal.principal_point_px()
            c.r.log(f"intrinsics:   {cam} <- '{cal.name}' f={cal.focal_px:.1f} px "
                    f"(C={cal.c_mm} mm / {cal.pixel_size_mm * 1000:.2f} um), "
                    f"pp=({cx:.1f}, {cy:.1f}) vs centre "
                    f"({cal.width / 2:.1f}, {cal.height / 2:.1f}) — {cal.detail}")
        if missed:
            c.r.log(f"intrinsics: no calibration matched {missed} — those heads keep "
                    "COLMAP's EXIF-derived guess")
        c.r.log("intrinsics: distortion is left at zero for the bundle to solve; only "
                "focal length and principal point come from the certificate")

    stage_root = c.ws / "rig_images"
    stage_root.mkdir(parents=True, exist_ok=True)
    c.rig_orig_names = build_staging(grouping, Path(c.img_root), stage_root)
    n = len(c.rig_orig_names)
    c.rig_config = c.ws / "rig_config.json"
    c.rig_intrinsics = intrinsics
    ref = write_rig_config(grouping, c.rig_config, ref_camera, intrinsics)
    c.r.banner(f"rig staging: {n} symlinks -> {stage_root} (ref sensor = {ref})")

    # The staged tree IS the dataset from here on, and its shape is always one
    # folder per camera — whatever the input layout was. The rest of the run has
    # to be told, or _build_image_list looks for images in the wrong place and
    # extraction assigns one camera to the whole rig.
    c.img_root = str(stage_root)
    c.folders = grouping.cameras
    c.layout_name = "multi"
    c.camera_mode = "per_folder"   # rig_configurator requires one camera per prefix


def _stage_apply_intrinsics(c: _Ctx) -> None:
    """1b'. Write a vendor calibration into the database, rig or no rig.

    The rig path can also carry intrinsics through rig_config.json, but that only
    exists when a rig is configured — and a certificate matters just as much
    without one. It arguably matters MORE: with no rig constraint, nothing stops a
    head's self-calibration from running away. On the block this was written for,
    a run with the rig off let the nadir focal diverge to 57k px against a
    certificate value of 24k (+138%), which pushed that head's cameras 1600 units
    away from the rest and made the aligned model useless.

    Runs after extraction because it needs the cameras table, and before matching
    so every later stage sees the calibrated values.
    """
    if not c.stage_on("extract") or not c.db.exists():
        return
    cals, src = discover_calibrations(Path(c.orig_root or c.img_root))
    if not cals:
        return
    folders = sorted(set(camera_folders(c.db).values()))
    hit, missed = match_to_cameras(cals, folders)
    if not hit:
        c.r.log(f"intrinsics: {Path(src).name} has {len(cals)} head(s) but none match "
                f"the image folders {folders} — leaving COLMAP's EXIF estimate alone")
        return

    model = str(c.d["camera_model"]).upper()
    changed = apply_to_database(c.db, model, hit)
    c.r.banner(f"intrinsics: applied {len(changed)} calibrated camera(s) from "
               f"{Path(src).name} -> {c.db}")
    for folder, old_f, new_f in changed:
        cal = hit[folder]
        drift = (old_f - new_f) / new_f * 100.0 if new_f else 0.0
        c.r.log(f"  {folder:<12} f {old_f:9.1f} -> {new_f:9.1f} px "
                f"(EXIF estimate was {drift:+.1f}% off)  {cal.detail}")
    if missed:
        c.r.log(f"  no calibration for {missed}; those keep COLMAP's estimate")
    c.r.log("  distortion stays at zero for the bundle to solve; the certificate "
            "supplies focal length and principal point only")
    c.intrinsics_applied = True


def _stage_rig_configure(c: _Ctx) -> None:
    """1c. `colmap rig_configurator`: write rigs/frames into the database.

    Must run after feature extraction (it reads the images table) and before
    matching/mapping, so the mapper sees the rig constraint from the start.
    Extrinsics are left for the mapper to refine (--*.refine_sensor_from_rig,
    on by default); supplying an --input_path reconstruction would only be
    needed to seed them from an existing rig-less solve.
    """
    if not c.rig_enable or not c.stage_on("rig"):
        return
    if c.mapper != "global":
        # The incremental family refuses to start while sensor_from_rig is unknown
        # (incremental_pipeline.cc:563), so those mappers get their rig from
        # _stage_rig_calibrate instead, which can pass --input_path.
        c.r.log(f"rig: deferring rig_configurator — MAPPER={c.mapper} needs known "
                "extrinsics, so they are derived from an initial reconstruction first")
        return
    c.r.banner(f"rig_configurator -> {c.db}")
    c.r.run([COLMAP_BIN, "rig_configurator",
             "--database_path", str(c.db),
             "--rig_config_path", str(c.rig_config)])


def _stage_gps_inject(c: _Ctx) -> None:
    # 1b. GPS pose priors for non-JPEG inputs. COLMAP's feature_extractor reads EXIF GPS
    # only from JPEG (verified on 4.0.4: a GPS-tagged TIFF yields zero pose_priors rows
    # while the identical JPEG yields one per image), so for TIFF — or any image COLMAP
    # missed — we read the EXIF GPS off the ORIGINALS (c.orig_root, since the FullHD copies
    # are EXIF-stripped) and write the priors into the DB ourselves, in COLMAP's own row
    # layout. Must run AFTER extract (the DB + images table exist) and BEFORE match, since
    # spatial_matcher reads priors too. Idempotent: only fills images lacking a prior.
    if not c.stage_on("gps_inject"):
        return
    # 1a. EO CSV priors first: the surveyor's adjusted exterior orientation beats whatever
    # EXIF fix COLMAP already read at extraction, so these OVERWRITE existing rows; the
    # EXIF pass below then only fills images the CSV didn't cover. Runs whenever a CSV is
    # given, independent of the GPS-option gate — the CSV is an explicit user request.
    if c.eo_map and c.db.exists():
        std = _resolve_prior_std(c)
        if std == PRIOR_STD_EO:
            c.r.log(f"PRIOR_STD=auto -> {std} m (EO CSV = post-processed airborne POS)")
        elif min(std) >= 1.0:
            c.r.log(f"note: PRIOR_STD={std} m is a consumer-GPS uncertainty, but an EO CSV "
                    "is post-processed airborne POS (typically 0.03-0.20 m). Leaving it "
                    "this loose lets BA largely ignore the priors you just supplied.")
        n_eo, n_grav, n_miss = inject_eo_priors(
            c.db, c.eo_map, c.eo_crs, std, c.eo_gravity, c.r)
        c.r.banner(f"EO pose priors: wrote {n_eo} from {Path(c.eo_csv).name} "
                   f"(CRS={c.eo_crs}, std={std} m) -> {c.db}")
        c.r.log(f"  gravity written for {n_grav} image(s)"
                + ("" if c.ra_use_gravity else
                   " — set RA_USE_GRAVITY=1 with MAPPER=global to actually use it")
                + f"; {n_miss} image(s) left to the EXIF pass")

    # same trigger as the resize EXIF-preserve: any hard GPS option, or the custom
    # matcher's GPS-neighbour pairs. JPEG-only GPS runs need nothing here (COLMAP already
    # populated them), and the call no-ops in that case anyway.
    if not (c.gps_opts or (c.matcher == "custom" and c.cm_n_gps > 0)):
        return
    if not c.db.exists():
        c.r.log("skip gps_inject (no database.db yet)")
        return
    # write a real covariance from the form's GPS uncertainty (metres) so the in-BA prior
    # alignment works for any mapper, not just pose_prior with --overwrite_priors_covariance.
    std = _resolve_prior_std(c)
    n_inj, n_have = inject_pose_priors(c.db, c.orig_root or c.img_root, c.lines, c.r, std)
    if n_inj:
        c.r.banner(f"GPS pose priors: injected {n_inj} from EXIF (TIFF/non-JPEG; "
                   f"std={std} m) -> {c.db}")
        c.r.log(f"  {n_have} priors already present (JPEG GPS read by COLMAP at extract)")
    else:
        c.r.log(f"gps_inject: nothing to add ({n_have} priors already present; "
                "JPEG GPS is read by COLMAP itself)")


def _stage_match(c: _Ctx) -> None:
    # 2. matcher
    if c.stage_on("match"):
        sentinel = c.ws / ".match.done"
        if c.need(sentinel):
            if c.matcher in ("sequential", "both"):
                loop = 1 if c.matcher == "sequential" else 0
                c.r.banner(f"sequential_matcher (overlap={c.d['seq_overlap']}, loop_detection={loop})")
                seq = ["--SequentialMatching.overlap", str(c.d["seq_overlap"]),
                       "--SequentialMatching.quadratic_overlap", "1"]
                if c.matcher == "sequential":
                    seq += ["--SequentialMatching.loop_detection", "1",
                            "--SequentialMatching.loop_detection_period", str(c.d["seq_overlap"]),
                            "--SequentialMatching.loop_detection_num_images", str(c.d["num_matches"]),
                            "--SequentialMatching.vocab_tree_path", str(c.vocab_tree)]
                c.r.run([COLMAP_BIN, "sequential_matcher", "--database_path", str(c.db),
                         "--FeatureMatching.guided_matching", str(c.d["guided_matching"]), *seq])
            if c.matcher in ("vocab", "both"):
                c.r.banner(f"vocab_tree_matcher (num_images={c.d['num_matches']})")
                c.r.run([COLMAP_BIN, "vocab_tree_matcher", "--database_path", str(c.db),
                         "--FeatureMatching.guided_matching", str(c.d["guided_matching"]),
                         "--VocabTreeMatching.vocab_tree_path", str(c.vocab_tree),
                         "--VocabTreeMatching.num_images", str(c.d["num_matches"])])
            if c.matcher == "spatial":
                c.r.banner(f"spatial_matcher (GPS priors; max_neighbors="
                           f"{c.d['spatial_max_neighbors']}, max_distance="
                           f"{c.d['spatial_max_distance']}m, ignore_z={c.d['spatial_ignore_z']})")
                c.r.run([COLMAP_BIN, "spatial_matcher", "--database_path", str(c.db),
                         "--FeatureMatching.guided_matching", str(c.d["guided_matching"]),
                         "--SpatialMatching.max_num_neighbors", str(c.d["spatial_max_neighbors"]),
                         "--SpatialMatching.max_distance", str(c.d["spatial_max_distance"]),
                         "--SpatialMatching.ignore_z", str(c.d["spatial_ignore_z"])])
            if c.matcher == "custom":
                # h3dgs custom matcher: build an explicit match list (sequential +
                # quadratic frame-steps + loop closure + GPS neighbours) then import it
                # via matches_importer — far fewer pairs than exhaustive for large,
                # ordered multi-camera captures. Names come from the same folder→sorted
                # grouping used for extraction, so they match the DB exactly.
                from .. import large_scene
                groups = [(f, list_image_names(Path(c.img_root) / f)) for f in c.folders]
                matching_txt = c.ws / "matching.txt"
                c.r.banner(f"matches_importer (custom matcher: seq={c.cm_n_seq} quad={c.cm_n_quad} "
                           f"loop={c.cm_n_loop} gps={c.cm_n_gps}) -> {matching_txt}")
                npairs = large_scene.make_custom_matcher(
                    groups, c.img_root, str(matching_txt),
                    n_seq=int(c.cm_n_seq), n_quad=int(c.cm_n_quad), n_loop=int(c.cm_n_loop),
                    loop_matches=c.cm_loop_ints, n_gps=c.cm_n_gps)
                c.r.log(f"  custom match list: {npairs} pairs")
                c.r.run([COLMAP_BIN, "matches_importer", "--database_path", str(c.db),
                         "--match_list_path", str(matching_txt)])
            sentinel.touch()
        else:
            c.r.log("skip match (sentinel exists; set FORCE=1 to redo)")


def _stage_calibrate(c: _Ctx) -> None:
    # 3. view_graph_calibrator (global only)
    if c.stage_on("calibrate"):
        sentinel = c.ws / ".calibrate.done"
        if c.mapper != "global":
            c.r.log(f"skip calibrate (MAPPER={c.mapper}; only required for global)")
        elif c.need(sentinel):
            c.r.banner(f"view_graph_calibrator -> {c.db}")
            c.r.run([COLMAP_BIN, "view_graph_calibrator", "--database_path", str(c.db)])
            sentinel.touch()
        else:
            c.r.log("skip calibrate (sentinel exists; set FORCE=1 to redo)")


def _stage_rig_calibrate(c: _Ctx) -> None:
    """3b. Derive sensor_from_rig for the mappers that need it known in advance.

    Our rig config carries no extrinsics, and only the global pipeline calibrates
    them itself; `mapper` / `pose_prior_mapper` / `hierarchical_mapper` abort with
    UNKNOWN_SENSOR_FROM_RIG (incremental_pipeline.cc:563). COLMAP's own remedy is
    the two-pass route named in that error: reconstruct once WITHOUT rigs, then
    hand that reconstruction to rig_configurator, which averages the relative
    poses between registered sensors into sensor_from_rig and writes them to the
    database (scene/rig.cc UpdateRigAndCameraCalibsFromReconstruction).

    The first pass is throwaway — it only has to register enough frames for that
    average — so it runs a bare global_mapper rather than the configured one. It
    lands in its own directory so the real mapper's sparse/ is untouched.
    """
    if not c.rig_enable or c.mapper == "global" or not c.stage_on("rig"):
        return
    init = c.ws / "sparse_rig_init"
    if c.need(init / "0" / "cameras.bin"):
        c.r.banner(f"rig calibration pass 1/2: global_mapper without rigs -> {init}")
        init.mkdir(parents=True, exist_ok=True)
        c.r.run([COLMAP_BIN, "global_mapper", "--database_path", str(c.db),
                 "--image_path", c.img_root, "--output_path", str(init)])
    else:
        c.r.log(f"skip rig init pass ({init.name}/0 exists; set FORCE=1 to redo)")
    if not (init / "0" / "cameras.bin").exists():
        raise PipelineError(
            f"the rig calibration pass produced no model in {init}/0, so "
            "sensor_from_rig cannot be derived. Reconstruction is failing before the "
            "rig is even involved — check the matches, or set MAPPER=global to let it "
            "calibrate the rig itself in one pass.")

    c.r.banner(f"rig calibration pass 2/2: rig_configurator --input_path {init / '0'} "
               f"-> {c.db}")
    c.r.run([COLMAP_BIN, "rig_configurator",
             "--database_path", str(c.db),
             "--rig_config_path", str(c.rig_config),
             "--input_path", str(init / "0")])


def _stage_mapper(c: _Ctx) -> None:
    # 4. mapper -> sparse/0
    if c.stage_on("mapper"):
        cameras = c.ws / "sparse" / "0" / "cameras.bin"
        if c.need(cameras):
            extra: list[str] = []
            if c.mapper == "global":
                sub, label = "global_mapper", "colmap global_mapper"
                if c.ra_use_gravity:
                    # Rotation averaging can use the DB's per-image gravity (down in the
                    # camera frame, written by the EO CSV inject) to pin 2 of the 3
                    # rotation DOF. Heading is still solved from the view graph — the DB
                    # has no rotation prior. COLMAP solves the mixed case in strata, so
                    # only some images needing gravity is fine.
                    extra = ["--GlobalMapper.ra_use_gravity", "1",
                             "--GlobalMapper.ra_max_rotation_error_deg",
                             str(c.d["ra_max_rotation_error_deg"])]
                    label += " [gravity-aided RA]"
                if c.ba_backend == "caspar":
                    # Caspar GPU solver, exposed to global_mapper since COLMAP main@2fe2b41
                    # (#4484). Same SIMPLE_RADIAL/PINHOLE-only restriction as below (warned
                    # about in _setup); Caspar always runs on GPU, no separate toggle needed.
                    extra += ["--GlobalMapper.ba_backend", "CASPAR"]
                    label += " [Caspar GPU BA]"
            elif c.mapper == "pose_prior":
                # GPS priors folded into BA -> the output is already georeferenced
                # and metric (model_aligner is then redundant). overwrite_priors_
                # covariance=1 means the std_* below set the GPS uncertainty (metres).
                sub, label = "pose_prior_mapper", "colmap pose_prior_mapper (GPS)"
                extra = ["--use_robust_loss_on_prior_position", str(c.d["prior_robust_loss"]),
                         "--overwrite_priors_covariance", "1",
                         *[a for pair in zip(("--prior_position_std_x",
                                              "--prior_position_std_y",
                                              "--prior_position_std_z"),
                                             map(str, _resolve_prior_std(c)),
                                             strict=True) for a in pair]]
            elif c.mapper == "hierarchical":
                # h3dgs large-scene mapper: partition into overlapping sub-models,
                # reconstruct in parallel, then merge — scales where global/incremental
                # choke. Tight global-BA tolerance matches the h3dgs recipe. Partition
                # knobs (leaf_max_num_images / image_overlap / num_workers) are passed
                # only when set, else COLMAP's defaults (500 / 50 / auto) apply.
                sub, label = "hierarchical_mapper", "colmap hierarchical_mapper"
                extra = ["--Mapper.ba_global_function_tolerance", "0.000001"]
                # hierarchical_mapper reconstructs each cluster with the incremental
                # pipeline, so it inherits the same --Mapper.ba_*_backend flags below.
                # COLMAP 4.2 moved these three under the HierarchicalMapper.*
                # namespace; the old bare names are a hard parse error, not a
                # warning, so the stage would die before doing any work.
                if c.hm_leaf:
                    extra += ["--HierarchicalMapper.leaf_max_num_images", c.hm_leaf]
                if c.hm_overlap:
                    extra += ["--HierarchicalMapper.image_overlap", c.hm_overlap]
                if c.hm_workers:
                    extra += ["--HierarchicalMapper.num_workers", c.hm_workers]
            else:
                sub, label = "mapper", "colmap mapper (incremental)"
            # GPU bundle adjustment: BA dominates incremental/pose_prior/hierarchical
            # runtime, so offloading it to CUDA is a big speedup. global_mapper uses its
            # own --GlobalMapper.ba_backend switch (handled above) instead of these
            # --Mapper.ba_* flags, which it doesn't take.
            if c.mapper in ("incremental", "pose_prior", "hierarchical"):
                if c.ba_backend == "caspar":
                    # Caspar GPU solver: ~1-2 orders of magnitude faster than Ceres for the
                    # repeated local/global BA. Requires SIMPLE_RADIAL/PINHOLE cameras
                    # (warned about in _setup); takes over both BA stages.
                    extra += ["--Mapper.ba_local_backend", "CASPAR",
                              "--Mapper.ba_global_backend", "CASPAR"]
                    label += " [Caspar GPU BA]"
                elif c.ba_gpu:
                    extra += ["--Mapper.ba_use_gpu", "1"]
                    label += " [GPU BA]"
            if c.rig_intrinsics or c.intrinsics_applied:
                prefix = "GlobalMapper" if c.mapper == "global" else "Mapper"
                # Hold the focal length. Seeding it is NOT enough: on the block this
                # was written for, the nadir head was seeded from its certificate at
                # 23884 px and the bundle still ran away to 56972 (+138%), which put
                # that head's cameras 1600 units from the rest. Focal length trades
                # off against depth, and nothing in the global pipeline constrains
                # that — it does not read the position priors. A surveyed certificate
                # is a better estimate than anything the bundle can recover here.
                extra += [f"--{prefix}.ba_refine_focal_length", "0"]
                # The principal point stays refinable: it is seeded too, but its y
                # sign is the one convention this pipeline has to guess (certificate
                # y up vs image y down), so a wrong guess must be able to correct
                # itself. Distortion is written as zero and so must be solved.
                extra += [f"--{prefix}.ba_refine_principal_point", "1"]
                label += " [calibrated intrinsics: f held, PP+distortion refined]"
            c.r.banner(f"{label} -> {c.ws / 'sparse'}")
            c.r.run([COLMAP_BIN, sub, "--database_path", str(c.db),
                     "--image_path", c.img_root, "--output_path", str(c.ws / "sparse"), *extra])
            # the mapper just rebuilt sparse/0 from scratch — any simplify backup from a
            # previous run describes a *different* model. If kept, the simplify stage would
            # delete this run's outliers out of that stale backup (different image_ids), so
            # drop it (and any legacy *_heavy.bin files) and let simplify re-back-up *this*
            # model.
            shutil.rmtree(c.ws / "sparse" / "0_heavy", ignore_errors=True)
            for stale in ("images_heavy.bin", "points3D_heavy.bin"):
                (c.ws / "sparse" / "0" / stale).unlink(missing_ok=True)
        else:
            c.r.log("skip mapper (sparse/0/cameras.bin exists; set FORCE=1 to redo)")


def _stage_intrinsics_report(c: _Ctx) -> None:
    """4a'. Compare what the bundle converged to against the certificate.

    This is the check on the one convention the calibration import has to guess:
    the principal-point y sign. If the seed was right the bundle should barely
    move it; a shift of roughly twice the certificate's offset, in the opposite
    direction, is the signature of a flipped sign. The focal length is reported
    for the same reason — it says whether a 2-year-old certificate still matches
    the camera.
    """
    if not c.rig_intrinsics:
        return
    model_dir = c.ws / "sparse" / "0"
    if not (model_dir / "cameras.bin").exists():
        return
    try:
        from ..vendor.read_write_model import read_cameras_binary
        solved = read_cameras_binary(str(model_dir / "cameras.bin"))
    except Exception as exc:                            # noqa: BLE001
        c.r.log(f"intrinsics check: could not read {model_dir}/cameras.bin ({exc})")
        return

    # The rig config assigns cameras in the order written: ref first, then the rest
    # alphabetically — the same order rig_configurator walks.
    ref = next((cam for cam in sorted(c.rig_intrinsics) if cam == c.rig_ref_camera), "")
    order = ([ref] if ref else []) + [x for x in sorted(c.rig_intrinsics) if x != ref]
    by_id = {cid: cam for cid, cam in zip(sorted(solved), order, strict=False)}

    c.r.banner("intrinsics check: bundle vs certificate")
    c.r.log("  camera      f (cert -> solved)          cx (cert -> solved)   "
            "cy (cert -> solved)")
    for cid in sorted(solved):
        cam = by_id.get(cid)
        seed = c.rig_intrinsics.get(cam or "")
        if not seed:
            continue
        s = solved[cid].params
        _, want = seed
        # OPENCV order: fx fy cx cy ...; the shorter models put f first then cx cy
        wf, wcx, wcy = ((want[0], want[2], want[3]) if len(want) >= 4
                        else (want[0], want[1], want[2]))
        gf, gcx, gcy = ((s[0], s[2], s[3]) if len(s) >= 4 else (s[0], s[1], s[2]))
        c.r.log(f"  {cam:<11} {wf:9.1f} -> {gf:9.1f} ({gf - wf:+7.1f})  "
                f"{wcx:8.1f} -> {gcx:8.1f} ({gcx - wcx:+6.1f})  "
                f"{wcy:8.1f} -> {gcy:8.1f} ({gcy - wcy:+6.1f})")
    c.r.log("  a small drift is expected (the certificate predates the mission); a cy "
            "shift near twice the certificate's own offset, with the sign reversed, "
            "would instead mean the principal-point y convention was read backwards")


def _stage_simplify(c: _Ctx) -> None:
    # 4a. simplify_images (h3dgs): drop stray mis-localized (pose-outlier) cameras so the
    # model is cheaper to read and free of bad views. The full pre-simplify model is kept
    # in sparse/0_heavy/. COLMAP's image_deleter does the actual removal — editing the
    # model by hand strands frames.bin's data_ids (COLMAP 4.x rig/frame) and breaks the
    # point2D/point3D track invariants, aborting image_undistorter. Only runs when SIMPLIFY
    # is on.
    if c.stage_on("simplify"):
        sentinel = c.ws / ".simplify.done"
        model0 = c.ws / "sparse" / "0"
        heavy_dir = c.ws / "sparse" / "0_heavy"
        if not c.simplify:
            c.r.log("skip simplify (SIMPLIFY off)")
        elif c.need(sentinel):
            if not (model0 / "cameras.bin").is_file():
                raise RuntimeError("sparse model missing, cannot simplify")
            from .. import large_scene
            # Back the full model up once, then always delete *from that backup* so the
            # stage is idempotent: image_deleter errors on already-missing ids, and a
            # second run must not re-measure outliers on an already-trimmed model.
            if not (heavy_dir / "cameras.bin").is_file():
                shutil.rmtree(heavy_dir, ignore_errors=True)
                shutil.copytree(model0, heavy_dir)
            ids, n_total = large_scene.outlier_image_ids(
                str(heavy_dir), mult_min_dist=float(c.d["simplify_mult_min_dist"]))
            c.r.banner(f"simplify_images (mult_min_dist={c.d['simplify_mult_min_dist']}): "
                       f"drop {len(ids)}/{n_total} pose-outlier cameras -> {model0}")
            if ids:
                ids_file = c.ws / "simplify_delete_ids.txt"
                ids_file.write_text("\n".join(map(str, ids)) + "\n")
                c.r.run([COLMAP_BIN, "image_deleter", "--input_path", str(heavy_dir),
                         "--output_path", str(model0), "--image_ids_path", str(ids_file)])
            else:  # nothing to drop — refresh model0 from the backup so it's pristine
                for f in heavy_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, model0 / f.name)
            c.r.log(f"  images: {n_total} -> {n_total - len(ids)} "
                    f"(full model backed up in {heavy_dir.name}/)")
            sentinel.touch()
        else:
            c.r.log("skip simplify (sentinel exists; set FORCE=1 to redo)")


def _stage_align(c: _Ctx) -> None:
    # 4b. model_aligner (optional GPS metric alignment): rewrite sparse/0 in place
    # into a local ENU frame in real-world metres, using the DB's GPS pose priors.
    # Only runs when explicitly enabled AND GPS was actually found (otherwise it has
    # nothing to align to). Independent of MAPPER; complements the mesh ChArUco
    # mm-scaling — don't enable both on one dataset.
    if c.stage_on("align"):
        sentinel = c.ws / ".align.done"
        if not c.gps_align:
            c.r.log("skip align (GPS_ALIGN off)")
        elif not c.gps_present:
            c.r.log("skip align (GPS_ALIGN on but no EXIF GPS detected in inputs)")
        elif c.need(sentinel):
            if not (c.ws / "sparse" / "0" / "cameras.bin").is_file():
                raise RuntimeError("sparse model missing, cannot GPS-align")
            c.r.banner(f"model_aligner (GPS -> {c.gps_align_type} metres, "
                       f"max_error={c.gps_align_max_error}m) -> {c.ws / 'sparse' / '0'}")
            c.r.run([COLMAP_BIN, "model_aligner",
                     "--input_path", str(c.ws / "sparse" / "0"),
                     "--output_path", str(c.ws / "sparse" / "0"),
                     "--database_path", str(c.db), "--ref_is_gps", "1",
                     "--alignment_type", c.gps_align_type,
                     "--alignment_max_error", c.gps_align_max_error])
            sentinel.touch()
        else:
            c.r.log("skip align (sentinel exists; set FORCE=1 to redo)")


def _stage_undistort(c: _Ctx) -> None:
    # 5. image_undistorter -> dense_dir
    if c.stage_on("undistort"):
        uextra = ["--max_image_size", c.max_image_size] if c.max_image_size else []
        if c.need(c.dense_dir / "sparse" / "cameras.bin"):
            if not (c.ws / "sparse" / "0" / "cameras.bin").is_file():
                raise RuntimeError("sparse model missing, cannot undistort")
            c.r.banner(f"image_undistorter -> {c.dense_dir}"
                       + (f" (max_image_size={c.max_image_size})" if c.max_image_size else ""))
            c.r.run([COLMAP_BIN, "image_undistorter", "--image_path", c.img_root,
                     "--input_path", str(c.ws / "sparse" / "0"),
                     "--output_path", str(c.dense_dir), "--output_type", "COLMAP", *uextra])
        else:
            c.r.log(f"skip undistort ({c.dense_dir}/sparse/cameras.bin exists; set FORCE=1 to redo)")

        # 5b. masks (h3dgs, optional): undistort the foreground masks through the SAME
        # cameras as the images (a model copy with .png image names), then clean them
        # to uint8 0/255 under dense_dir/masks. Only when MASKS_DIR is set.
        if c.masks_dir:
            if not Path(c.masks_dir).is_dir():
                raise FileNotFoundError(f"masks_dir not found: {c.masks_dir}")
            mask_done = c.ws / ".masks.done"
            if c.need(mask_done):
                from .. import large_scene
                src0 = c.ws / "sparse" / "0"
                mask_model = src0 / "masks"
                mask_model.mkdir(parents=True, exist_ok=True)
                shutil.copy(src0 / "cameras.bin", mask_model / "cameras.bin")
                shutil.copy(src0 / "points3D.bin", mask_model / "points3D.bin")
                large_scene.replace_images_by_masks(src0 / "images.bin",
                                                    mask_model / "images.bin")
                tmp_masks = c.ws / "tmp_masks"
                shutil.rmtree(tmp_masks, ignore_errors=True)
                c.r.banner(f"image_undistorter (masks) -> {c.dense_dir / 'masks'}")
                c.r.run([COLMAP_BIN, "image_undistorter", "--image_path", c.masks_dir,
                         "--input_path", str(mask_model), "--output_path", str(tmp_masks),
                         "--output_type", "COLMAP", *uextra])
                n = large_scene.make_mask_uint8(str(tmp_masks / "images"),
                                                str(c.dense_dir / "masks"))
                c.r.log(f"  wrote {n} uint8 masks -> {c.dense_dir / 'masks'}")
                shutil.rmtree(tmp_masks, ignore_errors=True)
                mask_done.touch()
            else:
                c.r.log("skip masks (sentinel exists; set FORCE=1 to redo)")


def _stage_reorient(c: _Ctx) -> None:
    # 6. reorient (h3dgs): make "up" usable in the viewer. Two modes, picked by whether
    # GPS already metric-aligned the model:
    #   * GPS on  -> model_aligner already gave a metric, gravity-correct ENU frame (up=+Z);
    #     just rotate Z-up -> Y-up (fixed +90° about X), preserving GPS's metric scale.
    #   * GPS off -> heuristic PCA gravity-align + rescale (auto_reorient).
    # Either way poses+points rotate (images on disk untouched, so they stay valid).
    # Force-safe: the pre-reorient model is snapshotted to sparse_unaligned/ and always
    # used as the input, so a redo never double-rotates. Only runs when REORIENT is on.
    if c.stage_on("reorient"):
        sentinel = c.ws / ".reorient.done"
        dense_sparse = c.dense_dir / "sparse"
        backup = c.dense_dir / "sparse_unaligned"
        if not c.reorient:
            c.r.log("skip reorient (REORIENT off)")
        elif c.need(sentinel):
            if not (dense_sparse / "cameras.bin").is_file():
                raise RuntimeError("undistorted model missing, cannot reorient")
            from .. import large_scene
            if not backup.exists():            # snapshot the un-reoriented model once
                shutil.copytree(dense_sparse, backup)
            if c.gps_align and c.gps_present:
                c.r.banner(f"reorient (GPS-metric: fixed Z-up -> Y-up, scale preserved) "
                           f"-> {dense_sparse}")
                ni, npts = large_scene.zup_to_yup(str(backup), str(dense_sparse))
                c.r.log(f"  rotated {ni} cams / {npts} points to Y-up (GPS metric scale kept)")
            else:
                c.r.banner(f"auto_reorient (gravity align + scale, target_med_dist="
                           f"{c.d['reorient_target_med_dist']}, upscale={c.d['reorient_upscale']}) "
                           f"-> {dense_sparse}")
                scale, ni, npts = large_scene.auto_reorient(
                    str(backup), str(dense_sparse),
                    upscale=float(c.d["reorient_upscale"]),
                    target_med_dist=float(c.d["reorient_target_med_dist"]))
                c.r.log(f"  reoriented {ni} cams / {npts} points, scale={scale:.5g}")
            sentinel.touch()
        else:
            c.r.log("skip reorient (sentinel exists; set FORCE=1 to redo)")


def run_colmap(p: dict, r: Runner) -> None:
    c = _setup(p, r)
    _stage_nested(c)            # 0. NESTED staging (rebases img_root)
    _stage_rig_stage(c)         # 0b. rig staging (rebases img_root; groups frames)
    _build_image_list(c)        # image_list.txt + c.lines
    _resolve_eo_csv(c)          # optional EO CSV -> c.eo_map (name -> EO row)
    _check_gps_coverage(c)      # c.gps_present / c.gps_opts (may abort)
    c.orig_root = c.img_root    # capture pre-resize originals (intact EXIF) for gps_inject
    _maybe_resize_fullhd(c)     # FullHD downscale (rebases img_root)
    _stage_extract(c)           # 1. feature_extractor
    _stage_apply_intrinsics(c)  # 1b'. vendor calibration -> cameras table
    _stage_rig_configure(c)     # 1c. rig_configurator -> rigs/frames in the DB
    _stage_gps_inject(c)        # 1b. inject TIFF/non-JPEG EXIF GPS into DB pose_priors
    _stage_match(c)             # 2. matcher
    _stage_calibrate(c)         # 3. view_graph_calibrator (global only)
    _stage_rig_calibrate(c)     # 3b. derive sensor_from_rig for non-global mappers
    _stage_mapper(c)            # 4. mapper -> sparse/0
    _stage_intrinsics_report(c) # 4a'. calibrated vs solved intrinsics
    _stage_simplify(c)          # 4a. simplify_images (optional)
    _stage_align(c)             # 4b. model_aligner (optional GPS)
    _stage_undistort(c)         # 5. image_undistorter -> dense_dir (+ optional masks)
    _stage_reorient(c)          # 6. reorient (optional)
    c.r.banner(f"done. workspace={c.ws}")
