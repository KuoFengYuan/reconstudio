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

import os
import shutil
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ..config import settings
from ..runner import Runner
from ._gps import gps_coverage
from ._layout import IMAGE_EXTS, list_image_names, list_images, resolve_layout
from ._resize import resize_to_fullhd, resize_workers

COLMAP_STAGES = ["stage", "extract", "match", "calibrate", "mapper", "simplify",
                 "align", "undistort", "reorient"]
# colmap / ffmpeg binaries (PATH by default; override via COLMAP_BIN / FFMPEG_BIN).
COLMAP_BIN = settings.colmap_bin

COLMAP_DEFAULTS = {
    "vocab_tree": str(Path.home() / ".cache/colmap/vocab_tree_faiss_flickr100K_words256K.bin"),
    "vocab_tree_url": "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin",
    "camera_model": "OPENCV", "max_features": "4096", "camera_mode": "per_folder",
    "matcher": "both", "seq_overlap": "10", "num_matches": "50",
    "guided_matching": "1", "mapper": "global", "dataset_name": "training_dataset",
    "force": False, "nested_layout": False, "resize": "fullhd",
    # spatial_matcher (large GPS scenes): match only GPS-near images.
    "spatial_max_neighbors": "50", "spatial_max_distance": "100", "spatial_ignore_z": "1",
    # GPS metric alignment via model_aligner (optional, off by default): rewrite the
    # sparse model into a local ENU frame in real-world metres.
    "gps_align": False, "gps_align_type": "enu", "gps_align_max_error": "3.0",
    # pose_prior_mapper: GPS position uncertainty (metres). Consumer GPS ~3-5 m; RTK ~0.02.
    "prior_std_x": "3.0", "prior_std_y": "3.0", "prior_std_z": "5.0", "prior_robust_loss": "1",
    # GPU bundle adjustment for the incremental / pose_prior mappers (big speedup; on by default).
    "ba_gpu": True,
    # --- hierarchical-3d-gaussians large-scene method (MATCHER=custom, MAPPER=hierarchical) ---
    # custom matcher (make_colmap_custom_matcher): per-view match counts + loop anchors.
    "cm_n_seq": "0", "cm_n_quad": "10", "cm_n_loop": "5", "cm_n_gps": "25",
    "cm_loop_matches": "",
    # feature_extractor focal seed for uncalibrated large scenes (h3dgs uses 0.5); "" = COLMAP default.
    "focal_factor": "",
    # image_undistorter longest-side cap (h3dgs uses 2048); "" = no cap.
    "max_image_size": "",
    # optional foreground masks: undistort them through the same cameras as the images.
    "masks_dir": "",
    # simplify_images (drop pose-outlier cameras + trim observations) and auto_reorient
    # (PCA gravity-align + scale). Both off unless ticked; h3dgs runs both.
    "simplify": False, "simplify_mult_min_dist": "10",
    "reorient": False, "reorient_target_med_dist": "20", "reorient_upscale": "0",
    # hierarchical_mapper partition knobs (COLMAP 4.0.4); "" = COLMAP default
    # (leaf_max_num_images 500, image_overlap 50, num_workers -1 = auto).
    "hm_leaf_max_num_images": "", "hm_image_overlap": "", "hm_num_workers": "",
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
    gps_align: bool
    gps_align_type: str
    gps_align_max_error: str
    ba_gpu: bool
    focal_factor: str
    max_image_size: str
    masks_dir: str
    simplify: bool
    reorient: bool
    cm_n_seq: str
    cm_n_quad: str
    cm_n_loop: str
    cm_n_gps: int
    cm_loop_ints: list[int]
    hm_leaf: str
    hm_overlap: str
    hm_workers: str
    db: Path
    lst: Path
    dense_dir: Path
    lines: list[str] = field(default_factory=list)
    gps_present: bool = False
    gps_opts: list[str] = field(default_factory=list)

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
    if matcher not in ("sequential", "vocab", "both", "spatial", "custom"):
        raise ValueError(f"MATCHER must be 'sequential', 'vocab', 'both', 'spatial', or "
                         f"'custom' (got: {matcher})")

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
        gps_align=bool(d["gps_align"]), gps_align_type=str(d["gps_align_type"]),
        gps_align_max_error=str(d["gps_align_max_error"]),
        ba_gpu=bool(d["ba_gpu"]),   # GPU bundle adjustment (incremental / pose_prior only)
        # h3dgs large-scene method options
        focal_factor=str(d["focal_factor"]).strip(),
        max_image_size=str(d["max_image_size"]).strip(),
        masks_dir=str(d["masks_dir"]).strip(),
        simplify=bool(d["simplify"]), reorient=bool(d["reorient"]),
        cm_n_seq=str(d["cm_n_seq"]), cm_n_quad=str(d["cm_n_quad"]), cm_n_loop=str(d["cm_n_loop"]),
        cm_n_gps=int(str(d["cm_n_gps"]) or 0),
        cm_loop_ints=[int(x) for x in str(d["cm_loop_matches"]).replace(",", " ").split()],
        hm_leaf=str(d["hm_leaf_max_num_images"]).strip(),
        hm_overlap=str(d["hm_image_overlap"]).strip(),
        hm_workers=str(d["hm_num_workers"]).strip(),
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


def _build_image_list(c: _Ctx) -> None:
    # image_list.txt (paths relative to img_root)
    lines: list[str] = []
    for f in c.folders:
        lines += list_images(Path(c.img_root) / f, f)
    c.lst.write_text("\n".join(lines) + ("\n" if lines else ""))
    c.r.log(f"image_list: {len(lines)} images across {len(c.folders)} folder(s): {' '.join(c.folders)}")
    if not lines:
        raise FileNotFoundError("no images found")
    c.lines = lines


def _check_gps_coverage(c: _Ctx) -> None:
    # EXIF GPS coverage. COLMAP reads EXIF GPS into the DB pose priors that
    # spatial_matcher / pose_prior_mapper / model_aligner consume. The GPS pipeline
    # requires GPS on EVERY image — a frame without a prior can't be spatially matched
    # or anchored — so any GPS option needs FULL coverage, checked here before any work
    # runs. The FullHD resize still runs (so we keep the downscale); it just preserves
    # the EXIF GPS through the re-encode (see preserve_exif below).
    n_gps, n_total = gps_coverage(c.img_root, c.lines, c.r, resize_workers())
    c.gps_present = (n_total > 0 and n_gps == n_total)   # GPS flow valid only at 100%
    c.gps_opts = [name for name, on in (("MATCHER=spatial", c.matcher == "spatial"),
                                        ("MAPPER=pose_prior", c.mapper == "pose_prior"),
                                        ("GPS_ALIGN", c.gps_align)) if on]
    if c.gps_opts:
        if not c.gps_present:
            raise RuntimeError(
                f"GPS option(s) selected ({', '.join(c.gps_opts)}) but only {n_gps}/{n_total} "
                "inputs carry EXIF GPS — the GPS pipeline needs GPS on EVERY image, so it "
                "aborts before any work runs. Provide GPS-tagged photos for all inputs, or "
                "switch to a non-GPS setup (MATCHER=vocab/both, MAPPER=global/incremental, "
                "GPS 對齊 off). Note: video frames carry no per-frame GPS — it lives in the "
                "container, not the frames.")
        c.r.log(f"EXIF GPS on all {n_total} inputs -> GPS flow enabled ({', '.join(c.gps_opts)})"
                + ("; FullHD resize will preserve the GPS EXIF" if c.fullhd else ""))
    elif n_gps:
        c.r.log(f"note: {n_gps}/{n_total} inputs have EXIF GPS but no GPS option selected; "
                "running normally" + (" (the FullHD resize will drop the GPS)" if c.fullhd else ""))


def _maybe_resize_fullhd(c: _Ctx) -> None:
    # FullHD: physically downscale the inputs to FullHD copies and run the entire
    # pipeline on those (image_list paths stay relative, so they remain valid). When the
    # GPS flow is on, the resize grafts each original's EXIF back so GPS survives.
    if c.fullhd:
        # keep EXIF GPS through the re-encode when any GPS flow needs it — including
        # the custom matcher's optional GPS-neighbour pairs (not a hard-fail option,
        # so it's not in gps_opts, but it still reads GPS off the resized images).
        preserve = bool(c.gps_opts) or (c.matcher == "custom" and c.cm_n_gps > 0)
        c.img_root = resize_to_fullhd(c.img_root, c.lines, c.ws, c.force, c.r, preserve_exif=preserve)


def _stage_extract(c: _Ctx) -> None:
    # 1. feature_extractor
    if c.stage_on("extract"):
        if c.need(c.db):
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
            c.r.run([COLMAP_BIN, "feature_extractor", "--database_path", str(c.db),
                     "--image_path", c.img_root, "--image_list_path", str(c.lst), *cam,
                     "--ImageReader.camera_model", str(c.d["camera_model"]),
                     "--SiftExtraction.max_num_features", str(c.d["max_features"])])
        else:
            c.r.log("skip extract (database.db exists; set FORCE=1 to redo)")


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


def _stage_mapper(c: _Ctx) -> None:
    # 4. mapper -> sparse/0
    if c.stage_on("mapper"):
        cameras = c.ws / "sparse" / "0" / "cameras.bin"
        if c.need(cameras):
            extra: list[str] = []
            if c.mapper == "global":
                sub, label = "global_mapper", "colmap global_mapper"
            elif c.mapper == "pose_prior":
                # GPS priors folded into BA -> the output is already georeferenced
                # and metric (model_aligner is then redundant). overwrite_priors_
                # covariance=1 means the std_* below set the GPS uncertainty (metres).
                sub, label = "pose_prior_mapper", "colmap pose_prior_mapper (GPS)"
                extra = ["--use_robust_loss_on_prior_position", str(c.d["prior_robust_loss"]),
                         "--overwrite_priors_covariance", "1",
                         "--prior_position_std_x", str(c.d["prior_std_x"]),
                         "--prior_position_std_y", str(c.d["prior_std_y"]),
                         "--prior_position_std_z", str(c.d["prior_std_z"])]
            elif c.mapper == "hierarchical":
                # h3dgs large-scene mapper: partition into overlapping sub-models,
                # reconstruct in parallel, then merge — scales where global/incremental
                # choke. Tight global-BA tolerance matches the h3dgs recipe. Partition
                # knobs (leaf_max_num_images / image_overlap / num_workers) are passed
                # only when set, else COLMAP's defaults (500 / 50 / auto) apply.
                sub, label = "hierarchical_mapper", "colmap hierarchical_mapper"
                extra = ["--Mapper.ba_global_function_tolerance", "0.000001"]
                if c.hm_leaf:
                    extra += ["--leaf_max_num_images", c.hm_leaf]
                if c.hm_overlap:
                    extra += ["--image_overlap", c.hm_overlap]
                if c.hm_workers:
                    extra += ["--num_workers", c.hm_workers]
            else:
                sub, label = "mapper", "colmap mapper (incremental)"
            # GPU bundle adjustment: BA dominates incremental/pose_prior runtime, so
            # offloading it to CUDA is a big speedup (global_mapper / hierarchical_mapper
            # don't take this flag — they'd reject it — so only the two incremental
            # mappers get it).
            if c.ba_gpu and c.mapper in ("incremental", "pose_prior"):
                extra += ["--Mapper.ba_use_gpu", "1"]
                label += " [GPU BA]"
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
    _build_image_list(c)        # image_list.txt + c.lines
    _check_gps_coverage(c)      # c.gps_present / c.gps_opts (may abort)
    _maybe_resize_fullhd(c)     # FullHD downscale (rebases img_root)
    _stage_extract(c)           # 1. feature_extractor
    _stage_match(c)             # 2. matcher
    _stage_calibrate(c)         # 3. view_graph_calibrator (global only)
    _stage_mapper(c)            # 4. mapper -> sparse/0
    _stage_simplify(c)          # 4a. simplify_images (optional)
    _stage_align(c)             # 4b. model_aligner (optional GPS)
    _stage_undistort(c)         # 5. image_undistorter -> dense_dir (+ optional masks)
    _stage_reorient(c)          # 6. reorient (optional)
    c.r.banner(f"done. workspace={c.ws}")
