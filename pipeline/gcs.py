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

import json
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .runner import Runner

GSUTIL_BIN = "gsutil"

# --- fast browser listing: cached gcloud token + JSON API ----------------- #
# `gsutil ls` pays ~1.5–2.8s of Python-CLI startup on *every* folder click,
# which is what makes the picker feel laggy. We instead list in-process via the
# GCS JSON API (stdlib urllib — no new deps), authenticated with an access token
# fetched once from `gcloud` and cached. A short result cache makes navigating
# back up / revisiting a folder instant. gsutil stays as the fallback so a stale
# token or any API hiccup still lists correctly (just slower).
_TOKEN_TTL = 2700.0          # refresh the access token well before its ~1h expiry
_LS_TTL = 30.0               # how long a folder listing stays fresh in the cache
_token_lock = threading.Lock()
_token: dict = {"value": None, "exp": 0.0}
_ls_lock = threading.Lock()
_ls_cache: dict[str, tuple[float, dict]] = {}


def _gsutil(bin_: str | None) -> str:
    return (bin_ or GSUTIL_BIN).strip() or GSUTIL_BIN


def gcs_parent(prefix: str) -> str | None:
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


def _gcloud_bin(gsutil_bin: str | None) -> str:
    """`gcloud` sits next to `gsutil` in the Cloud SDK; derive it from the
    configured gsutil path, else fall back to `gcloud` on PATH."""
    p = Path(_gsutil(gsutil_bin))
    if p.parent != Path(".") and p.name.startswith("gsutil"):
        cand = p.with_name("gcloud" + p.name[len("gsutil"):])
        if cand.exists():
            return str(cand)
    return "gcloud"


def _access_token(gcloud_bin: str, *, timeout: float = 30.0) -> str:
    """A cached OAuth access token from `gcloud auth print-access-token`.
    Fetched at most once per _TOKEN_TTL so listing stays a single HTTP call."""
    now = time.monotonic()
    with _token_lock:
        if _token["value"] and now < _token["exp"]:
            return _token["value"]
        res = subprocess.run([gcloud_bin, "auth", "print-access-token"],
                             capture_output=True, text=True, timeout=timeout)
        if res.returncode != 0:
            msg = (res.stderr or res.stdout or "gcloud auth print-access-token failed").strip()
            raise RuntimeError(msg.splitlines()[-1] if msg else "no access token")
        tok = res.stdout.strip()
        if not tok:
            raise RuntimeError("gcloud 回傳空的 access token")
        _token.update(value=tok, exp=now + _TOKEN_TTL)
        return tok


def _ls_cache_get(key: str) -> dict | None:
    with _ls_lock:
        hit = _ls_cache.get(key)
        return hit[1] if hit and time.monotonic() < hit[0] else None


def _ls_cache_put(key: str, data: dict) -> None:
    with _ls_lock:
        _ls_cache[key] = (time.monotonic() + _LS_TTL, data)


def _gcs_ls_api(prefix: str, gcloud_bin: str, *, timeout: float) -> dict:
    """List one level under a gs://bucket/... prefix via the GCS JSON API.

    In-process (urllib + a cached token) so it skips the gsutil subprocess
    startup. Same return shape as gcs_ls; raises on auth/API errors so the
    caller can fall back to gsutil."""
    body = prefix[len("gs://"):]
    bucket, _, obj = body.partition("/")
    if not bucket:
        raise ValueError("缺少 bucket 名稱")
    if obj and not obj.endswith("/"):
        obj += "/"
    token = _access_token(gcloud_bin, timeout=timeout)
    base = f"https://storage.googleapis.com/storage/v1/b/{urllib.parse.quote(bucket)}/o"

    dirs, nfiles, page = [], 0, None
    while True:
        params = {"delimiter": "/", "maxResults": "1000",
                  "fields": "prefixes,items/name,nextPageToken"}
        if obj:
            params["prefix"] = obj
        if page:
            params["pageToken"] = page
        req = urllib.request.Request(base + "?" + urllib.parse.urlencode(params),
                                     headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
        for sub in payload.get("prefixes", []):          # sub-"folders"
            dirs.append({"name": sub.rstrip("/").rsplit("/", 1)[-1],
                         "path": f"gs://{bucket}/{sub}"})
        for item in payload.get("items", []):            # files (skip dir placeholder)
            if not item.get("name", "").endswith("/"):
                nfiles += 1
        page = payload.get("nextPageToken")
        if not page:
            break
    dirs.sort(key=lambda d: d["name"].lower())
    return {"prefix": prefix, "dirs": dirs, "nfiles": nfiles,
            "parent": gcs_parent(prefix)}


def _gcs_ls_gsutil(prefix: str, *, gsutil_bin: str | None,
                   timeout: float) -> dict:
    """Fallback lister: shell out to `gsutil ls`. Robust but slow (pays the
    full gsutil startup on every call)."""
    g = _gsutil(gsutil_bin)
    if shutil.which(g) is None:
        raise RuntimeError(f"找不到 {g};請確認 Google Cloud SDK 已安裝且在 PATH。")
    # `gsutil ls` with no arg lists buckets; `gsutil ls gs://b/p/` lists contents.
    target = (prefix.rstrip("/") + "/") if prefix else None
    argv = [g, "ls"] + ([target] if target else [])
    res = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        msg = (res.stderr or res.stdout or "gsutil ls failed").strip()
        raise RuntimeError(msg.splitlines()[-1] if msg else "gsutil ls failed")

    self_obj = target.rstrip("/") if target else None   # the dir's own placeholder
    dirs, nfiles = [], 0
    for ln in res.stdout.splitlines():
        ln = ln.strip()
        if not ln.startswith("gs://"):
            continue
        if ln.rstrip("/") == self_obj:    # zero-byte folder placeholder == this dir
            continue
        if ln.endswith("/"):
            dirs.append({"name": ln.rstrip("/").rsplit("/", 1)[-1], "path": ln})
        else:
            nfiles += 1
    dirs.sort(key=lambda d: d["name"].lower())
    return {"prefix": prefix, "dirs": dirs, "nfiles": nfiles,
            "parent": gcs_parent(prefix)}


def gcs_ls(prefix: str = "", *, gsutil_bin: str | None = None,
           timeout: float = 30.0) -> dict:
    """List one level under a gs:// prefix (or every bucket when prefix is "").

    Returns {"prefix", "dirs": [{"name","path"}], "nfiles", "parent"}.
    `dirs` are object-name prefixes (end in '/') the picker navigates into;
    files at this level are only counted (you select a folder, not a file).

    Inside a bucket we list via the fast in-process JSON API and cache the
    result; the bucket-list case (prefix == "") and any API/auth failure fall
    back to `gsutil ls`.
    """
    prefix = (prefix or "").strip()
    if prefix in ("gs://", "gs:/"):
        prefix = ""
    if prefix and not prefix.startswith("gs://"):
        raise ValueError("prefix 必須是 gs:// 開頭")

    cached = _ls_cache_get(prefix)
    if cached is not None:
        return cached

    data = None
    if prefix:                       # fast path only applies inside a bucket
        try:
            data = _gcs_ls_api(prefix, _gcloud_bin(gsutil_bin), timeout=timeout)
        except Exception:            # auth/API/network error -> gsutil fallback
            data = None
    if data is None:
        data = _gcs_ls_gsutil(prefix, gsutil_bin=gsutil_bin, timeout=timeout)

    _ls_cache_put(prefix, data)
    return data


def gcs_multi_plan(srcs: list[str], dest_root: str) -> list[tuple[str, str]]:
    """Map selected gs:// folders to LOCAL dests, keeping only the last folder
    the user navigated into (their shared parent) — not the whole gs:// path:

        in gs://bucket/project/2026/0520_ITRI/  pick CAM_a, CAM_b
        -> <dest_root>/0520_ITRI/CAM_a , <dest_root>/0520_ITRI/CAM_b

    The shared parent is the deepest folder common to every selection, so a
    cross-level pick keeps the structure *below* that folder. Returns
    [(src, dest)] with trailing slashes stripped (cp copies <parent>/<leaf>).
    """
    srcs = [s.strip().rstrip("/") for s in srcs if s and s.strip()]
    if not srcs:
        return []
    parts = [s[len("gs://"):].split("/") for s in srcs]   # [bucket, .., leaf]
    parents = [p[:-1] for p in parts]                     # each folder's parent path
    common: list[str] = []                                # deepest shared parent
    for tup in zip(*parents):
        if len(set(tup)) == 1:
            common.append(tup[0])
        else:
            break
    container = common[-1] if common else ""              # the "last folder entered"
    base = Path(dest_root) / container if container else Path(dest_root)
    return [(s, str(base.joinpath(*p[len(common):]))) for s, p in zip(srcs, parts)]


def _run_gcs_cp_multi(params: dict, runner: Runner) -> None:
    """Download several gs:// folders in ONE job via `gsutil -m cp -r`.

    Dests come from gcs_multi_plan (flattened to the shared folder). Sources are
    grouped by local parent dir so a single multi-source `cp -r` per group drops
    each folder at its planned path — one parallel (-m) transfer instead of N
    queued jobs. Trailing slashes are stripped so a folder copies as
    <parent>/<leaf> (not its flattened contents); re-runs are idempotent
    (cp merges into the existing folder, no <leaf>/<leaf> nesting).
    """
    dest_root = (params.get("dest_root") or "").strip()
    g = _gsutil(params.get("gsutil_bin"))
    plan = gcs_multi_plan(params.get("srcs") or [], dest_root)

    if not plan:
        raise ValueError("沒有要下載的來源")
    bad = next((s for s, _ in plan if not s.startswith("gs://")), None)
    if bad:
        raise ValueError(f"來源必須是 gs:// 路徑:{bad!r}")
    if not dest_root:
        raise ValueError("下載目的根目錄 (dest_root) 必填")
    if shutil.which(g) is None:
        raise RuntimeError(f"找不到 {g};請確認 Google Cloud SDK 已安裝且在 PATH。")

    groups: dict[str, list[str]] = {}          # local parent dir -> sources to drop in it
    for src, dest in plan:
        groups.setdefault(str(Path(dest).parent), []).append(src)

    runner.banner(f"GCS 多選下載  {len(plan)} 個資料夾 → {dest_root}")
    for parent, group in groups.items():
        Path(parent).mkdir(parents=True, exist_ok=True)   # ensure dest exists -> cp puts each at parent/<leaf>
        argv = [g, "-m", "cp", "-r", *group, parent + "/"]
        runner.log("$ " + " ".join(argv))
        runner.log("")
        runner.run(argv)                                  # streams gsutil progress into the job log
    runner.log(f"\n[gcs] 完成 → {len(plan)} 個資料夾已下載到 {dest_root}")


def run_gcs_sync(params: dict, runner: Runner) -> None:
    """Download a gs:// folder to a local dir via `gsutil -m rsync -r`.

    Multi-select downloads pass a `srcs` list instead and are handled in one
    parallel `gsutil -m cp -r` by _run_gcs_cp_multi (same job, same parser)."""
    if params.get("srcs"):
        return _run_gcs_cp_multi(params, runner)
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
