"""GCS sync: pull data from Google Cloud Storage to local disk.

COLMAP / training do heavy random IO over thousands of images, so they must run
on a LOCAL copy — fusing a bucket would be slow and fragile. This module shells
out to `gsutil` (already on PATH + authed); no new Python deps, same "drive
external tools as subprocesses" approach as ffmpeg / colmap.

Deliberately decoupled from the pipeline: it only moves bytes gs:// -> local.
What you then do with the result (COLMAP, train, nothing) is chosen separately
via the normal local folder picker — there is no forced "next step".
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .runner import Runner

GSUTIL_BIN = "gsutil"


def _gsutil(bin_: Optional[str]) -> str:
    return (bin_ or GSUTIL_BIN).strip() or GSUTIL_BIN


def gcs_parent(prefix: str) -> Optional[str]:
    """One level up from a gs:// prefix. '' = the bucket list; None = already
    at the bucket list (nothing above it)."""
    p = (prefix or "").strip().rstrip("/")
    if not p or p in ("gs:/", "gs:"):
        return None
    body = p[len("gs://"):] if p.startswith("gs://") else p
    if "/" not in body:           # gs://bucket -> up to the bucket list
        return ""
    return "gs://" + body.rsplit("/", 1)[0] + "/"


def default_dest(src: str, dest_root: str) -> str:
    """Mirror the gs:// path under the local staging root, preserving structure
    so two prefixes that share a leaf name don't collide:
        gs://bucket/a/c  ->  <dest_root>/bucket/a/c
    """
    body = (src or "").strip()
    if body.startswith("gs://"):
        body = body[len("gs://"):]
    body = body.strip("/")
    return str(Path(dest_root) / body) if body else str(Path(dest_root))


def gcs_ls(prefix: str = "", *, gsutil_bin: Optional[str] = None,
           timeout: float = 30.0) -> dict:
    """List one level under a gs:// prefix (or every bucket when prefix is "").

    Returns {"prefix", "dirs": [{"name","path"}], "nfiles", "parent"}.
    `dirs` are object-name prefixes (end in '/') the picker navigates into;
    files at this level are only counted (you select a folder, not a file).
    """
    g = _gsutil(gsutil_bin)
    if shutil.which(g) is None:
        raise RuntimeError(f"找不到 {g};請確認 Google Cloud SDK 已安裝且在 PATH。")
    prefix = (prefix or "").strip()
    if prefix in ("gs://", "gs:/"):
        prefix = ""
    if prefix and not prefix.startswith("gs://"):
        raise ValueError("prefix 必須是 gs:// 開頭")
    # `gsutil ls` with no arg lists buckets; `gsutil ls gs://b/p/` lists contents.
    target = (prefix.rstrip("/") + "/") if prefix else None
    argv = [g, "ls"] + ([target] if target else [])
    res = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        msg = (res.stderr or res.stdout or "gsutil ls failed").strip()
        raise RuntimeError(msg.splitlines()[-1] if msg else "gsutil ls failed")

    dirs, nfiles = [], 0
    for ln in res.stdout.splitlines():
        ln = ln.strip()
        if not ln.startswith("gs://"):
            continue
        if ln.endswith("/"):
            dirs.append({"name": ln.rstrip("/").rsplit("/", 1)[-1], "path": ln})
        else:
            nfiles += 1
    dirs.sort(key=lambda d: d["name"].lower())
    return {"prefix": prefix, "dirs": dirs, "nfiles": nfiles,
            "parent": gcs_parent(prefix)}


def run_gcs_sync(params: dict, runner: Runner) -> None:
    """Download a gs:// folder to a local dir via `gsutil -m rsync -r`."""
    src = (params.get("src") or "").strip()
    dest = (params.get("dest") or "").strip()
    g = _gsutil(params.get("gsutil_bin"))
    delete = bool(params.get("delete"))

    if not src.startswith("gs://"):
        raise ValueError(f"來源必須是 gs:// 路徑:{src!r}")
    if not dest:
        raise ValueError("下載目的 (本機路徑) 必填")
    if shutil.which(g) is None:
        raise RuntimeError(f"找不到 {g};請確認 Google Cloud SDK 已安裝且在 PATH。")

    runner.banner(f"GCS sync  {src}  →  {dest}")
    Path(dest).mkdir(parents=True, exist_ok=True)
    argv = [g, "-m", "rsync", "-r"]
    if delete:                       # mirror: remove local files absent from source
        argv.append("-d")
    argv += [src.rstrip("/"), dest]
    runner.log("$ " + " ".join(argv))
    runner.log("")
    runner.run(argv)                 # streams gsutil progress into the job log
    runner.log(f"\n[gcs] 完成 → {dest}")
