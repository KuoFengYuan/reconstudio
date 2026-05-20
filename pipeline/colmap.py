"""Python port of colmap_pipeline.sh.

Stages (skippable via `stages`, re-runnable via force):
  stage -> extract -> match -> calibrate(global only) -> mapper -> undistort
Idempotency uses the same sentinels / output checks as the shell script, and the
banners it emits match `log()` so the panel's stage parser is unchanged.
"""
from __future__ import annotations

import os
import shutil
import time
import urllib.request
from pathlib import Path

from .runner import Runner

COLMAP_STAGES = ["stage", "extract", "match", "calibrate", "mapper", "undistort"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
# colmap binary: PATH by default, override with COLMAP_BIN for non-standard installs.
COLMAP_BIN = os.environ.get("COLMAP_BIN", "colmap")

COLMAP_DEFAULTS = {
    "vocab_tree": str(Path.home() / ".cache/colmap/vocab_tree_faiss_flickr100K_words256K.bin"),
    "vocab_tree_url": "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin",
    "camera_model": "OPENCV", "max_features": "4096", "camera_mode": "per_folder",
    "matcher": "both", "seq_overlap": "10", "num_matches": "50",
    "guided_matching": "1", "mapper": "global", "dataset_name": "training_dataset",
    "force": False, "nested_layout": False, "resize": "keep",
}
FULLHD_MAX = "1920"   # longest side cap for the "fullhd" resize option


def _download(url: str, dest: Path, r: Runner) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last: Exception | None = None
    for attempt in range(3):
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


def _has_images(folder: Path) -> bool:
    return folder.is_dir() and any(
        c.is_file() and c.suffix.lower() in IMAGE_EXTS for c in folder.iterdir())


def _list_images(folder: Path, prefix: str) -> list[str]:
    """Image files directly in `folder` (symlinks followed), relative as prefix/name
    (or just name when prefix is '' — a single flat folder = image_root itself)."""
    out: list[str] = []
    if folder.is_dir():
        for entry in folder.iterdir():
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
                out.append(f"{prefix}/{entry.name}" if prefix else entry.name)
    return sorted(out)


def _resolve_layout(img_root: str, folders: list[str], layout: str,
                    force_nested: bool, workspace: str | None = None) -> tuple[list[str], bool, str]:
    """Return (folders, nested, layout_name) for the three fixed input formats:
      single  : XXX/*.jpg              -> folders=[''] (image_root itself), 1 camera
      multi   : ROOT/<group>/*.jpg     -> folders=[groups], flat, camera per group
      nested  : ROOT/<group>/<vid>/*.jpg -> staged into groups, camera per group
    """
    root = Path(img_root)
    # Ignore the workspace dir if the user nested it inside image_root, so it isn't
    # mistaken for a camera group.
    skip = set()
    if workspace:
        try:
            wp = Path(workspace)
            if wp.resolve().parent == root.resolve():
                skip.add(wp.name)
        except OSError:
            pass
    subdirs = sorted([x.name for x in root.iterdir()
                      if x.is_dir() and not x.name.startswith(".") and x.name not in skip])

    # Explicit single, or auto when the root itself holds images (subdirs are then
    # likely junk such as the workspace) -> single flat folder.
    if layout == "single" or (layout == "auto" and _has_images(root)):
        return ([""] if not folders else folders), False, "single"

    chosen = folders or subdirs
    if not chosen:
        if _has_images(root):
            return [""], False, "single"
        raise FileNotFoundError(f"no images or subfolders found under: {img_root}")

    if layout == "nested" or force_nested:
        return chosen, True, "nested"
    if layout == "multi":
        return chosen, False, "multi"

    # auto: nested if the first group has no direct images but does have subdirs
    nested = False
    for f in chosen:
        g = root / f
        if g.is_dir() and not _has_images(g) and any(c.is_dir() for c in g.iterdir()):
            nested = True
        break
    return chosen, nested, ("nested" if nested else "multi")


def run_colmap(p: dict, r: Runner) -> None:
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

    if mapper not in ("global", "incremental"):
        raise ValueError(f"MAPPER must be 'global' or 'incremental' (got: {mapper})")
    if camera_mode not in ("per_folder", "single"):
        raise ValueError(f"CAMERA_MODE must be 'per_folder' or 'single' (got: {camera_mode})")
    if matcher not in ("sequential", "vocab", "both"):
        raise ValueError(f"MATCHER must be 'sequential', 'vocab', or 'both' (got: {matcher})")

    if not Path(img_root).is_dir():
        raise FileNotFoundError(f"image_root not found: {img_root}")
    folders, nested, layout_name = _resolve_layout(
        img_root, folders, str(p.get("layout") or "auto"), bool(d["nested_layout"]),
        workspace=p.get("workspace"))
    shown = "<root>" if folders == [""] else " ".join(folders)
    r.log(f"layout={layout_name}  folders={shown}  nested={int(nested)}")
    for f in folders:
        if f and not (Path(img_root) / f).is_dir():
            raise FileNotFoundError(f"subfolder not found: {Path(img_root) / f}")

    def stage_on(s: str) -> bool:
        return s in stages

    def need(path: "Path | str") -> bool:
        return force or not Path(path).exists()

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

    # 0. stage (NESTED_LAYOUT): flatten <group>/<video>/*.jpg -> staging/<group>/<video>_<file> symlinks
    if stage_on("stage") and not nested:
        r.log("skip stage (not a nested layout)")
    if stage_on("stage") and nested:
        stage_root = ws / "staging"
        sentinel = ws / ".stage.done"
        abs_root = Path(os.path.realpath(img_root))
        if need(sentinel):
            r.banner(f"stage nested layout -> {stage_root} (NESTED_LAYOUT=1)")
            if force:
                shutil.rmtree(stage_root, ignore_errors=True)
            for grp in folders:
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
                        r.log(f"  {grp}/{vdir.name}: {vcount} images")
                        grp_total += vcount
                r.log(f"  {grp} total: {grp_total}")
                if grp_total == 0:
                    raise FileNotFoundError(f"no images found under group: {grp}")
            sentinel.touch()
        else:
            r.log("skip stage (sentinel exists; set FORCE=1 to redo)")
        img_root = str(stage_root)
        r.log(f"rebased IMG_ROOT={img_root}")

    # image_list.txt (paths relative to img_root)
    lines: list[str] = []
    for f in folders:
        lines += _list_images(Path(img_root) / f, f)
    lst.write_text("\n".join(lines) + ("\n" if lines else ""))
    r.log(f"image_list: {len(lines)} images across {len(folders)} folder(s): {' '.join(folders)}")
    if not lines:
        raise FileNotFoundError("no images found")

    # 1. feature_extractor
    if stage_on("extract"):
        if need(db):
            r.banner(f"feature_extractor (CAMERA_MODE={camera_mode}, layout={layout_name}) -> {db}")
            if layout_name == "single":
                cam = ["--ImageReader.single_camera", "1"]            # one flat folder = 1 camera
            elif camera_mode == "per_folder":
                cam = ["--ImageReader.single_camera_per_folder", "1"]
            else:
                cam = ["--ImageReader.single_camera", "1"]
            size_opt = ["--FeatureExtraction.max_image_size", FULLHD_MAX] if fullhd else []
            r.run([COLMAP_BIN, "feature_extractor", "--database_path", str(db),
                   "--image_path", img_root, "--image_list_path", str(lst), *cam,
                   "--ImageReader.camera_model", str(d["camera_model"]),
                   "--SiftExtraction.max_num_features", str(d["max_features"]), *size_opt])
        else:
            r.log("skip extract (database.db exists; set FORCE=1 to redo)")

    # 2. matcher
    if stage_on("match"):
        sentinel = ws / ".match.done"
        if need(sentinel):
            if matcher in ("sequential", "both"):
                loop = 1 if matcher == "sequential" else 0
                r.banner(f"sequential_matcher (overlap={d['seq_overlap']}, loop_detection={loop})")
                seq = ["--SequentialMatching.overlap", str(d["seq_overlap"]),
                       "--SequentialMatching.quadratic_overlap", "1"]
                if matcher == "sequential":
                    seq += ["--SequentialMatching.loop_detection", "1",
                            "--SequentialMatching.loop_detection_period", str(d["seq_overlap"]),
                            "--SequentialMatching.loop_detection_num_images", str(d["num_matches"]),
                            "--SequentialMatching.vocab_tree_path", str(vocab_tree)]
                r.run([COLMAP_BIN, "sequential_matcher", "--database_path", str(db),
                       "--FeatureMatching.guided_matching", str(d["guided_matching"]), *seq])
            if matcher in ("vocab", "both"):
                r.banner(f"vocab_tree_matcher (num_images={d['num_matches']})")
                r.run([COLMAP_BIN, "vocab_tree_matcher", "--database_path", str(db),
                       "--FeatureMatching.guided_matching", str(d["guided_matching"]),
                       "--VocabTreeMatching.vocab_tree_path", str(vocab_tree),
                       "--VocabTreeMatching.num_images", str(d["num_matches"])])
            sentinel.touch()
        else:
            r.log("skip match (sentinel exists; set FORCE=1 to redo)")

    # 3. view_graph_calibrator (global only)
    if stage_on("calibrate"):
        sentinel = ws / ".calibrate.done"
        if mapper != "global":
            r.log(f"skip calibrate (MAPPER={mapper}; only required for global)")
        elif need(sentinel):
            r.banner(f"view_graph_calibrator -> {db}")
            r.run([COLMAP_BIN, "view_graph_calibrator", "--database_path", str(db)])
            sentinel.touch()
        else:
            r.log("skip calibrate (sentinel exists; set FORCE=1 to redo)")

    # 4. mapper -> sparse/0
    if stage_on("mapper"):
        cameras = ws / "sparse" / "0" / "cameras.bin"
        if need(cameras):
            sub = "global_mapper" if mapper == "global" else "mapper"
            label = "colmap global_mapper" if mapper == "global" else "colmap mapper (incremental)"
            r.banner(f"{label} -> {ws / 'sparse'}")
            r.run([COLMAP_BIN, sub, "--database_path", str(db),
                   "--image_path", img_root, "--output_path", str(ws / "sparse")])
        else:
            r.log("skip mapper (sparse/0/cameras.bin exists; set FORCE=1 to redo)")

    # 5. image_undistorter -> dense_dir
    if stage_on("undistort"):
        if need(dense_dir / "sparse" / "cameras.bin"):
            if not (ws / "sparse" / "0" / "cameras.bin").is_file():
                raise RuntimeError("sparse model missing, cannot undistort")
            r.banner(f"image_undistorter -> {dense_dir}" + (f" (max_image_size={FULLHD_MAX})" if fullhd else ""))
            undist_size = ["--max_image_size", FULLHD_MAX] if fullhd else []
            r.run([COLMAP_BIN, "image_undistorter", "--image_path", img_root,
                   "--input_path", str(ws / "sparse" / "0"),
                   "--output_path", str(dense_dir), "--output_type", "COLMAP", *undist_size])
        else:
            r.log(f"skip undistort ({dense_dir}/sparse/cameras.bin exists; set FORCE=1 to redo)")

    r.banner(f"done. workspace={ws}")
