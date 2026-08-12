"""Job → filesystem resolution + non-destructive edited-model derivation.

Pure path/file logic shared by the viz + create routers. HTTP concerns (404s)
stay in the routers; this layer only resolves and builds paths.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path


def _is_model(d: Path) -> bool:
    """A readable sparse model = the two binaries the viewer parses."""
    return (d / "images.bin").exists() and (d / "points3D.bin").exists()


def _block_model(d: Path) -> Path | None:
    """A blocksplit block dir holds a flat sparse/ (no /0)."""
    md = d / "sparse"
    return md if _is_model(md) else None


def workspace_model(ws: Path) -> Path | None:
    """The sparse model under a workspace. This panel's COLMAP writes sparse/0, but
    a model reconstructed elsewhere (GLOMAP, a COLMAP GUI export, an undistorted
    copy) is just as often a flat sparse/ — and a mapper that dropped its first
    submodel leaves sparse/1 instead. Probe in that order rather than 404-ing on a
    reconstruction that is sitting right there."""
    if _is_model(sub := ws / "sparse" / "0"):
        return sub
    if _is_model(flat := ws / "sparse"):
        return flat
    for d in sorted((p for p in (ws / "sparse").glob("*") if p.is_dir() and p.name.isdigit()),
                    key=lambda p: int(p.name)):
        if _is_model(d):
            return d
    return None


def model_dir(job) -> Path | None:
    if getattr(job, "kind", "") == "blocksplit":
        out = (job.meta or {}).get("out_dir")
        if not out:
            return None
        # default view = the first block (older jobs have no block list in meta)
        for d in sorted(Path(out).glob("block_*")):
            if md := _block_model(d):
                return md
        return None
    ws = (job.meta or {}).get("workspace")
    return workspace_model(Path(ws)) if ws else None


def dense_dir(job) -> Path | None:
    """The undistorted dataset dir (sparse/ + images/) that 3DGS training reads."""
    ws = (job.meta or {}).get("workspace")
    if not ws:
        return None
    p = job.params or {}
    d = Path(ws) / f"{p.get('dataset_name', 'training_dataset')}_{p.get('mapper', 'global')}_mapper"
    return d if (d / "sparse" / "cameras.bin").is_file() and (d / "images").is_dir() else None


def resolve_model(job, sub: str | None) -> Path | None:
    """Sparse model to visualize: the default sparse/0, or — when `sub` is given —
    a camera-culled variant under the workspace's cleaned/ tree (e.g.
    'cleaned/20260522_103000/sparse/0'). `sub` is confined to <ws>/cleaned to keep
    it from escaping the workspace via .. ."""
    if not sub:
        return model_dir(job)
    if getattr(job, "kind", "") == "blocksplit":
        out = (job.meta or {}).get("out_dir")
        if not out:
            return None
        root = Path(out).resolve()
        cand = (root / sub).resolve()         # sub = 'block_<x>_<y>', confined to out_dir
        if not str(cand).startswith(str(root) + os.sep):
            return None
        return _block_model(cand)
    ws = (job.meta or {}).get("workspace")
    if not ws:
        return None
    cleaned = (Path(ws) / "cleaned").resolve()
    cand = (Path(ws) / sub).resolve()
    if cand != cleaned and not str(cand).startswith(str(cleaned) + os.sep):
        return None
    return cand if (cand / "images.bin").exists() and (cand / "points3D.bin").exists() else None


_MAPPER_DIR_RE = re.compile(r"^(?P<name>.+)_(?P<mapper>global|hierarchical)_mapper$")
_RESIZE_DIR_RE = re.compile(r"^images_(?P<max>\d+)$")


def inspect_workspace(ws: Path) -> dict | None:
    """Read an existing COLMAP workspace off disk and recover the few fields a job
    record needs to serve it: dataset_name/mapper (dense_dir rebuilds the
    undistorted dir name from them), resize_max and image_root (_source_image walks
    those to find an image file). Everything is inferred from directory names, so a
    workspace produced outside the panel — or one whose job record is gone — can
    still be viewed. None when the workspace holds no readable sparse model."""
    md = workspace_model(ws)
    if not md:
        return None
    dataset_name, mapper, dataset = "training_dataset", "global", ""
    for d in sorted(ws.glob("*_mapper")):
        m = _MAPPER_DIR_RE.match(d.name)
        if m and (d / "sparse" / "cameras.bin").is_file() and (d / "images").is_dir():
            dataset_name, mapper, dataset = m["name"], m["mapper"], str(d)
            break
    # image_root: the resized copy the pipeline fed COLMAP is the best guess (it is
    # what the undistorted model was built from); fall back to a plain images/ dir.
    resize_max, image_root = "1920", ""
    sized = sorted((d for d in ws.glob("images_*") if _RESIZE_DIR_RE.match(d.name)),
                   key=lambda d: int(_RESIZE_DIR_RE.match(d.name)["max"]), reverse=True)
    if sized:
        resize_max = _RESIZE_DIR_RE.match(sized[0].name)["max"]
        image_root = str(sized[0])
    elif (ws / "images").is_dir():
        image_root = str(ws / "images")
    return {"model": str(md), "dataset": dataset, "dataset_name": dataset_name,
            "mapper": mapper, "resize_max": resize_max, "image_root": image_root}


def block_list(job) -> list[dict]:
    """[{name, views}] of a blocksplit job's blocks for the viewer's switcher.
    manifest.json is the source of truth (covers jobs that predate the
    meta.blocks log parser); fall back to globbing valid block dirs. Empty for
    other job kinds."""
    if getattr(job, "kind", "") != "blocksplit":
        return []
    out = (job.meta or {}).get("out_dir")
    if not out:
        return []
    root = Path(out)
    try:
        mani = json.loads((root / "manifest.json").read_text())
        blocks = [{"name": str(b["name"]), "views": b.get("train_views", "")}
                  for b in mani.get("blocks", [])]
        if blocks:
            return blocks
    except (OSError, ValueError, KeyError):
        pass
    return [{"name": d.name, "views": ""} for d in sorted(root.glob("block_*"))
            if (d / "sparse" / "images.bin").is_file()]


def trained_model_dir(job) -> Path | None:
    """The trained-model output dir, from a train job's meta. Validated by the
    presence of a trained 3DGS .ply (backend-agnostic). Distinct from model_dir,
    which is the COLMAP sparse reconstruction."""
    mp = (job.meta or {}).get("model_path")
    if not mp:
        return None
    d = Path(mp)
    return d if (d.is_dir() and trained_ply(d)) else None


def trained_ply(model_dir: Path) -> Path | None:
    """Highest-iteration trained 3DGS .ply, across trainer layouts:
       GS-2M     → point_cloud/iteration_<N>/point_cloud.ply
       LichtFeld → splat_<N>.ply (in the output dir)."""
    cands = list(model_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    cands += list(model_dir.glob("splat_*.ply"))
    if not cands:
        return None

    def it(p: Path) -> int:
        token = p.parent.name if p.name == "point_cloud.ply" else p.stem
        m = re.search(r"(\d+)", token)
        return int(m.group(1)) if m else -1
    return max(cands, key=it)


def make_edited_model(model: Path) -> tuple[Path, Path]:
    """Create a non-destructive sibling model dir for a SuperSplat-edited (去背)
    cloud and return (edited_root, dest_ply_path). The caller writes the bytes.

    Mirrors the `<model>_scene` symlink trick (train.py): the original model dir
    is never touched. The sibling gets its own
    point_cloud/iteration_<N>/point_cloud.ply (the edited cloud goes there) and a
    symlink to the original cfg_args. GS-2M's render.py then loads the edited
    cloud, reads cameras from the original source_path (recorded in cfg_args),
    and writes the mesh under the sibling — so a bad去背 can't destroy the source."""
    src_ply = trained_ply(model)
    if not src_ply:
        raise ValueError(f"原模型找不到 point_cloud.ply（{model}）。")
    it_dir = src_ply.parent                       # .../point_cloud/iteration_<N>
    edited = model.parent / f"{model.name}_edited_{time.strftime('%Y%m%d_%H%M%S')}"
    dst_it = edited / "point_cloud" / it_dir.name
    dst_it.mkdir(parents=True, exist_ok=True)
    (edited / "cfg_args").symlink_to((model / "cfg_args").resolve())
    light = it_dir / "lighting.pth"               # only present if trained with --material
    if light.is_file():
        (dst_it / "lighting.pth").symlink_to(light.resolve())
    return edited, dst_it / "point_cloud.ply"


def prepare_edited_model(model: Path, upload) -> Path:
    """Build the sibling edited model dir from an uploaded file (Mesh form)."""
    edited, dest = make_edited_model(model)
    with open(dest, "wb") as out:
        shutil.copyfileobj(upload.file, out)
    return edited
