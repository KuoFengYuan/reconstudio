"""System-level preflight checks — the "is this *machine* ready?" half of /doctor.

`backends.py` checks the **trainers** (conda envs, compiled binaries, CUDA arch).
This module checks everything else a fresh machine needs, and specifically the
things that fail *silently* or *late*:

  * ffmpeg exists — and actually has the `blurdetect` filter we depend on
    (missing from plenty of distro builds; without it 抽幀 dies mid-job).
  * the state/scratch disks are really writable and not nearly full.
  * libcudnn9 is resolvable, since LichtFeld's onnxruntime dlopens it at runtime.
  * the optional toolchains (gsutil, node) are present or knowingly absent.
  * local.env itself: `Settings` is ``extra="ignore"``, so a typo'd or long-dead
    variable is swallowed and you silently get the default instead.

Every check returns the same shape so both renderers (doctor.html and
doctor_cli.py) stay generic and new checks need no renderer changes:

    {"key", "label", "status", "value", "detail", "hint"}

  status  ok    ready to use
          warn  an OPTIONAL feature is unavailable, or a setting looks wrong but
                the code degrades gracefully (e.g. GPU decode falls back to CPU)
          err   a REQUIRED piece is missing — something will fail at runtime
          skip  deliberately disabled/not applicable on this machine
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import REPO_ROOT, Settings, settings
from .frames import _ffmpeg_bin
from .gcs import _gcloud_bin

_TIMEOUT = 15          # every probe is a fast local binary; /doctor must never hang
_LOW_DISK_GB = 20      # below this a COLMAP/training run can plausibly fill the disk

STATUSES = ("ok", "warn", "err", "skip")

_BLURDETECT_RE = re.compile(r"^\s*\S+\s+blurdetect\s", re.M)
# A variable assignment in local.env / local.env.example (optionally commented out).
_VAR_RE = re.compile(r"^\s*(?:#\s*)?(?:export\s+)?([A-Za-z_]\w*)=", re.M)
_SET_VAR_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_]\w*)=", re.M)


def make_check(key: str, label: str, status: str,
               value: str = "", detail: str = "", hint: str = "") -> dict:
               # Validating here rather than only in the test covers every branch at its
               # construction point, including the ones a given machine never reaches.
               assert status in STATUSES, f"unknown status {status!r} for check {key!r}"
               return {"key": key, "label": label, "status": status,
               "value": value, "detail": detail, "hint": hint}

def probe(argv: list[str], timeout: float = _TIMEOUT) -> str:
    """Run a short diagnostic command and return stdout+stderr (ffmpeg and friends
    print to stderr). Returns '' on ANY failure — missing binary, non-zero exit,
    timeout — because a probe must never raise into the report, and a report must
    never hang. Shared with backends.py so every probe gets the same timeout."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# colmap: the reconstruction core
# --------------------------------------------------------------------------- #
def colmap_check() -> dict:
    """Same shape as every other system check, so neither renderer needs a special
    case for it. (`backends.doctor()` still derives the older
    `report["colmap"]` dict from this for `/api/doctor` consumers.)"""
    path = shutil.which(settings.colmap_bin)
    if not path:
        return make_check("colmap", "colmap", "err", settings.colmap_bin, "找不到執行檔",
                          "安裝 colmap,或在 local.env 設 COLMAP_BIN=/絕對/路徑")
    return make_check("colmap", "colmap", "ok", path,
                      probe([path, "-h"]).partition("\n")[0].strip())


# --------------------------------------------------------------------------- #
# exiftool: strips Canon's empty-string Artist/Copyright before undistort so
# OIIO 2.4.17's IPTC-IIM encoder doesn't SIGABRT (see colmap._run._sanitize_exif)
# --------------------------------------------------------------------------- #
def exiftool_check() -> dict:
    path = shutil.which("exiftool")
    if not path:
        return make_check(
            "exiftool", "exiftool (Canon EXIF 消毒)", "warn", "找不到執行檔",
            "Canon 機身常把 Artist/Copyright 寫成空字串;image_undistorter 重新編碼這類影像時,"
            "OIIO 的 IPTC 編碼器會 assert 崩潰(SIGABRT)。沒有 exiftool 就不會自動修,"
            "只有踩到才會發現,而且會讓整個 undistort 階段失敗",
            "sudo apt install libimage-exiftool-perl")
    return make_check("exiftool", "exiftool (Canon EXIF 消毒)", "ok", path,
                      probe([path, "-ver"]).strip())


# --------------------------------------------------------------------------- #
# ffmpeg: the binary, the one filter we require, and the configured hwaccel
# --------------------------------------------------------------------------- #
def ffmpeg_checks() -> list[dict]:
    # Resolve the binary EXACTLY the way the pipeline does. frames._ffmpeg_bin()
    # falls back to PATH `ffmpeg` when FFMPEG_BIN points at something unusable, so
    # checking settings.ffmpeg_bin directly would report a hard failure on a
    # machine where 抽幀 actually works — e.g. a local.env copied from another box
    # whose /mnt/ssd1/bin/ffmpeg-nvdec doesn't exist here. It would also probe the
    # wrong binary for blurdetect/hwaccel.
    configured, resolved = settings.ffmpeg_bin, _ffmpeg_bin(None)
    path = shutil.which(resolved)
    if not path:
        return [make_check("ffmpeg", "ffmpeg", "err", configured, "找不到執行檔",
                           "安裝 ffmpeg,或在 local.env 設 FFMPEG_BIN=/絕對/路徑")]
    version = probe([path, "-version"]).partition("\n")[0].strip()
    if resolved == configured:
        out = [make_check("ffmpeg", "ffmpeg", "ok", path, version)]
    else:
        # Works, but the setting is dead — worth saying out loud, since the user
        # believes they configured a specific (probably NVDEC) build.
        out = [make_check("ffmpeg", "ffmpeg", "warn", path, f"{version}\n"
                      f"設定的 FFMPEG_BIN={configured} 無法執行,已退回 PATH 的 ffmpeg",
                      "抽幀照樣能跑;要用原本指定的 build 就修好路徑,否則從 local.env 刪掉這行")]

    # blurdetect drives 抽幀's sharpness scoring — a hard requirement, and the
    # single most common "worked on the old machine" surprise on a new box.
    if _BLURDETECT_RE.search(probe([path, "-hide_banner", "-filters"])):
        out.append(make_check("ffmpeg_blurdetect", "ffmpeg blurdetect filter", "ok", "有"))
    else:
        out.append(make_check(
            "ffmpeg_blurdetect", "ffmpeg blurdetect filter", "err", "缺",
            "這個 build 沒編 blurdetect,抽幀的去模糊評分會失敗",
            "換一個有 blurdetect 的 ffmpeg build(多數 distro 的完整版有;"
            "自行編譯要開 --enable-filter=blurdetect),或設 FFMPEG_BIN 指向它"))

    out.append(_hwaccel_check(path))
    return out


def _hwaccel_check(path: str) -> dict:
    """GPU decode is optional — frames.py probes -hwaccels and falls back to CPU.

    Mirrors frames._supports_cuda deliberately: that helper treats FFMPEG_HWACCEL
    purely as an on/off switch and only ever tests for the string "cuda", because
    frames.py hardcodes `-hwaccel cuda`. Reporting "ok, vaapi available" would be
    promising a mode the pipeline will never actually request.
    """
    want = settings.ffmpeg_hwaccel.strip().lower()
    if want in ("", "none", "0", "cpu"):
        return make_check("ffmpeg_hwaccel", "ffmpeg GPU 解碼", "skip", want or "none",
                          "已在設定中關閉,抽幀用 CPU 解碼")

    listed = [ln.strip() for ln in probe([path, "-hide_banner", "-hwaccels"]).splitlines()[1:]
              if ln.strip()]
    if want != "cuda":
        return make_check(
            "ffmpeg_hwaccel", "ffmpeg GPU 解碼", "warn", want,
            f"抽幀只會送 -hwaccel cuda,{want} 不會被使用(這個 build 可用: "
            f"{', '.join(listed) or '無'})",
            "要 GPU 解碼就設 FFMPEG_HWACCEL=cuda;不要 GPU 解碼就設 none")
    if "cuda" in listed:
        return make_check("ffmpeg_hwaccel", "ffmpeg GPU 解碼", "ok", "cuda",
                          f"可用: {', '.join(listed)}")
    return make_check(
        "ffmpeg_hwaccel", "ffmpeg GPU 解碼", "warn", "cuda",
        f"這個 build 不支援 cuda(可用: {', '.join(listed) or '無'});"
        f"抽幀會自動退回 CPU 解碼(較慢,但功能正常)",
        "要 GPU 解碼就換一個 NVDEC build;不需要的話在 local.env 設 "
        "FFMPEG_HWACCEL=none 讓這項變成「已關閉」")


# --------------------------------------------------------------------------- #
# storage: job state + scratch must be writable, and roomy enough to matter
# --------------------------------------------------------------------------- #
def _dir_check(key: str, label: str, p: Path) -> dict:
    """Writability + free space, WITHOUT creating anything.

    A preflight check must only observe: `./run.sh --doctor` is run on machines
    you're merely inspecting, and creating the tree here would also mask the very
    failure being diagnosed. So probe the deepest EXISTING ancestor instead — on a
    fresh machine the directory legitimately doesn't exist yet (run.sh creates it
    at startup), and "not there yet, but the parent is writable" is an ok.
    """
    probe_dir = p
    while not probe_dir.exists() and probe_dir != probe_dir.parent:
        probe_dir = probe_dir.parent
    if not probe_dir.is_dir():
        return make_check(key, label, "err", str(p), f"{probe_dir} 不是資料夾",
                          "改設一個可寫的路徑")
    try:
        # A real write, not an os.access() guess — that's what catches a
        # read-only NFS mount or a root-owned directory.
        with tempfile.NamedTemporaryFile(dir=probe_dir):
            pass
    except OSError as e:
        return make_check(key, label, "err", str(p), f"無法寫入 {probe_dir}: {e}",
                          "改設一個可寫的路徑,或修掉目錄權限")

    free_gb = shutil.disk_usage(probe_dir).free / 2 ** 30
    pending = "" if p.exists() else "(尚未建立,啟動時自動產生)"
    if free_gb < _LOW_DISK_GB:
        return make_check(key, label, "warn", str(p), f"可寫{pending},但只剩 {free_gb:.0f} GB",
                          "COLMAP 去畸變 / 訓練的中間檔很大,建議換到更空的磁碟")
    return make_check(key, label, "ok", str(p), f"可寫{pending},剩餘 {free_gb:.0f} GB")


def storage_checks() -> list[dict]:
    tmp = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    return [
        _dir_check("data_dir", "job 狀態 + log (RECON_STUDIO_DATA)", settings.data_dir),
        _dir_check("tmpdir", "暫存空間 (TMPDIR)", tmp),
    ]


# --------------------------------------------------------------------------- #
# libcudnn9 — LichtFeld's onnxruntime CUDA provider dlopens it at runtime
# --------------------------------------------------------------------------- #
def cudnn_check() -> dict:
    """The lichtfeld-* trainers and the 🌊 深度 preprocess tool both load
    libcudnn.so.9 through onnxruntime's CUDA execution provider. A missing one
    surfaces as an opaque provider-init failure *after* the job starts, so it's
    worth a red light here instead."""
    # One label, so a typo can't silently mislabel one branch's row.
    label = "libcudnn9 (LichtFeld onnxruntime)"
    for line in probe(["ldconfig", "-p"]).splitlines():
        if "libcudnn.so.9" in line:
            return make_check("cudnn", label, "ok", line.split("=>")[-1].strip(),
                              "由 ldconfig 解析到")

    # Not in the linker cache, but LD_LIBRARY_PATH may still supply it.
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if not d:
            continue
        try:
            hit = min(Path(d).glob("libcudnn.so.9*"), default=None)
        except OSError:
            continue
        if hit:
            return make_check("cudnn", label, "ok", str(hit), "由 LD_LIBRARY_PATH 提供")

    return make_check("cudnn", label, "warn", "找不到",
                      "LichtFeld 訓練 / 🌊 深度 會在 onnxruntime 初始化 CUDA provider 時失敗"
                      "(只用 GS-2M 和 CPU 模式不受影響)",
                      "裝系統套件(如 apt install libcudnn9-cuda-12),或在 local.env 用 "
                      "LD_LIBRARY_PATH 指向現成的 libcudnn.so.9")


# --------------------------------------------------------------------------- #
# optional: GCS (☁️ 資料 tab) and the SuperSplat bundle (去背 / 點雲檢視)
# --------------------------------------------------------------------------- #
def gcs_checks(deep: bool = True) -> list[dict]:
    path = shutil.which(settings.gsutil_bin)
    if not path:
        return [make_check("gsutil", "gsutil (☁️ 資料 分頁)", "warn", settings.gsutil_bin,
                           "找不到,雲端搬檔分頁不能用(其餘功能不受影響)",
                           "裝 Google Cloud SDK 再 `gcloud auth login`;不用雲端就忽略這項")]
    out = [make_check("gsutil", "gsutil (☁️ 資料 分頁)", "ok", path)]
    project = (os.environ.get("CLOUDSDK_CORE_PROJECT") or "").strip()
    if project:
        out.append(make_check("gcs_project", "GCP project", "ok", project))
    else:
        out.append(make_check(
            "gcs_project", "GCP project", "warn", "未設定",
            "`gsutil ls` 不帶參數(列出所有 bucket)需要 project id,會直接報錯",
            "在 local.env 設 CLOUDSDK_CORE_PROJECT=你的-gcp-project"))

    # Locate gcloud the way gcs.py does — the Cloud SDK often isn't on PATH as a
    # whole, so it derives gcloud from the *configured* gsutil's directory. Using
    # a bare which("gcloud") would silently drop this check on exactly the
    # machines where gcs._access_token()'s fast listing path is about to break.
    gcloud = _gcloud_bin(settings.gsutil_bin)
    if deep and shutil.which(gcloud):
        accounts = [ln.strip() for ln in probe(
            [gcloud, "auth", "list", "--filter=status:ACTIVE",
             "--format=value(account)"]).splitlines() if ln.strip()]
        if accounts:
            out.append(make_check("gcs_auth", "GCP 登入", "ok", accounts[0]))
        else:
            out.append(make_check("gcs_auth", "GCP 登入", "warn", "沒有 active 帳號",
                                  "gsutil 會因為沒有憑證而失敗", "跑 `gcloud auth login`"))
    return out


def supersplat_checks() -> list[dict]:
    """The bundle is a gitignored build artifact that run.sh syncs in the
    background at startup, so a fresh clone legitimately has none yet — that's a
    warn (feature unavailable until the first build finishes), never an error."""
    ver_file = REPO_ROOT / "static" / "supersplat" / ".version"
    version = ver_file.read_text().strip() if ver_file.is_file() else ""
    missing = [t for t in ("node", "npm", "git") if not shutil.which(t)]
    toolchain = ("缺 " + ", ".join(missing)) if missing else "node/npm/git 都有"

    if version:
        out = [make_check("supersplat", "SuperSplat bundle (去背 / 點雲檢視)", "ok", version)]
    else:
        out = [make_check(
            "supersplat", "SuperSplat bundle (去背 / 點雲檢視)", "warn", "尚未建置",
            "去背編輯器和 ✨ SuperSplat 檢視器要等 bundle 建好才會動",
            "run.sh 啟動時會在背景自動建置(需要網路);或手動跑 "
            "./tools/build_supersplat.sh")]

    if missing:
        out.append(make_check(
            "supersplat_toolchain", "SuperSplat 建置工具鏈", "warn", toolchain,
            "沒有工具鏈就無法建置 / 自動更新 bundle" + ("" if version else ",而且目前也沒有現成的 bundle"),
            "裝 node>=18 + npm + git,或從別台機器把 static/supersplat/ 整個複製過來"))
    else:
        out.append(make_check("supersplat_toolchain", "SuperSplat 建置工具鏈", "ok", toolchain))
    return out


# --------------------------------------------------------------------------- #
# local.env sanity: typo'd / stale variables are otherwise silently ignored
# --------------------------------------------------------------------------- #
def known_env_vars() -> set[str]:
    """Every variable something actually reads, derived from the two places that
    already know: `Settings`' validation aliases, and every `VAR=` documented in
    local.env.example (including commented-out ones — that file is where the
    shell-only knobs run.sh consumes live, e.g. HOST/PORT/SUPERSPLAT_*).

    Deriving instead of hand-listing means adding a Settings field or documenting
    a new knob needs no edit here, and it makes local.env.example load-bearing
    rather than free to rot.
    """
    known: set[str] = set()
    for field in Settings.model_fields.values():
        alias = field.validation_alias
        if isinstance(alias, str):
            known.add(alias.upper())
        else:   # AliasChoices — every spelling it accepts is legitimate
            known |= {c.upper() for c in getattr(alias, "choices", ()) if isinstance(c, str)}
    example = REPO_ROOT / "local.env.example"
    if example.is_file():
        known |= {v.upper() for v in _VAR_RE.findall(example.read_text())}
    return known


def env_var_check() -> dict:
    """`Settings` uses extra="ignore", so `COLMAP_BNI=…` (one transposition)
    doesn't raise — it quietly uses the default, which is the hardest class of
    misconfiguration to spot on a new machine. Same for variables left over from a
    feature that has since been rewritten.

    Reads the local.env FILE rather than os.environ: the file is what a human
    edits and what run.sh sources from a fixed path, so every key can be checked
    (not just prefixed ones — that would miss `COLMAP_BNI`) with no false alarms
    from whatever the parent shell, CI or systemd happens to export.
    """
    local = REPO_ROOT / "local.env"
    if not local.is_file():
        return make_check("env_vars", "local.env 變數", "skip", "沒有 local.env",
                          "全部走 run.sh 的自動偵測預設值")

    known = known_env_vars()
    unknown = sorted({v for v in _SET_VAR_RE.findall(local.read_text())
                      if v.upper() not in known})
    if not unknown:
        return make_check("env_vars", "local.env 變數", "ok", "沒有無效變數")
    return make_check(
        "env_vars", "local.env 變數", "warn", ", ".join(unknown),
        "沒有任何程式讀這些變數,設了也不會生效(打錯字,或功能改寫後留下的舊設定)",
        "檢查拼字,或從 local.env 刪掉;對照 local.env.example 的完整清單")


# --------------------------------------------------------------------------- #
# Reachability. "The port is up but nobody can connect" is almost never a port
# that isn't up — it is one of these three, none of which the panel itself can
# see from the inside, so they belong on /doctor next to the disks and binaries.
_NGINX_SITE = Path("/etc/nginx/sites-enabled/reconstudio.conf")
_PROXY_RE = re.compile(r"proxy_pass\s+http://127\.0\.0\.1:(\d+)")
_LISTEN_RE = re.compile(r"^\s*listen\s+([\d.]+):(\d+)\s+ssl", re.M)
_NAME_RE = re.compile(r"^\s*server_name\s+([^;\s]+)", re.M)
_INCLUDE_RE = re.compile(r"^\s*include\s+(/[^;\s]+);", re.M)


def _shell_var(name: str, default: str) -> str:
    """A run.sh-only knob (HOST/PORT live in the shell, not in `Settings`).

    Read from `RECON_STUDIO_<NAME>`, which run.sh exports once it has resolved
    the value, and NOT from a bare `HOST`/`PORT`: conda's compiler activation
    exports `HOST=x86_64-conda-linux-gnu` and this check would then cheerfully
    report a build triplet as the panel's bind address. local.env is the fallback
    for a hand-rolled `uvicorn app:app`, where the value would otherwise be
    reported as the default while the process bound something else entirely.
    """
    if os.environ.get(f"RECON_STUDIO_{name}"):
        return os.environ[f"RECON_STUDIO_{name}"]
    local = REPO_ROOT / "local.env"
    if local.is_file():
        found = re.findall(rf"^\s*(?:export\s+)?{name}=([^\s#]+)", local.read_text(), re.M)
        if found:
            return found[-1]
    return default


def _nginx_site() -> dict | None:
    """The proxy's own view of itself: which port it forwards to, and where it
    answers. Parsed from the enabled site rather than asked of nginx, so the
    check needs no root and no running nginx to say something useful.

    The `include`d snippet has to be followed: the two vhosts share one
    proxy_pass, so it lives there and reading only the site file would report
    "no proxy port" — i.e. permanent drift — on a perfectly good install."""
    try:
        conf = _NGINX_SITE.read_text()
    except OSError:                      # missing, or unreadable — same answer either way
        return None
    for inc in _INCLUDE_RE.findall(conf):
        try:
            conf += Path(inc).read_text()
        except OSError:                  # a broken include is nginx's to complain about
            pass
    proxy = _PROXY_RE.search(conf)
    listen = _LISTEN_RE.search(conf)
    name = _NAME_RE.search(conf)
    port = listen.group(2) if listen else ""
    return {"proxy_port": proxy.group(1) if proxy else "",
            "listen_port": port,
            "host": name.group(1) if name else (listen.group(1) if listen else ""),
            # 443 is implied by https://, and a URL people are meant to type
            # should look like the one in the browser's address bar.
            "url_port": "" if port == "443" else f":{port}"}


def bind_check() -> dict:
    """What the panel is reachable ON — which is not what run.sh used to print."""
    host, port = _shell_var("HOST", "127.0.0.1"), _shell_var("PORT", "8077")
    if host in ("0.0.0.0", "::"):
        extra = (f"nginx 站台也在,所以區網上有兩個入口,而 :{port} 這個沒有密碼"
                 if _nginx_site() else "面板沒有任何登入機制,又能瀏覽檔案、開子行程")
        return make_check("bind", "面板綁定位址", "warn", f"{host}:{port}",
                          f"整個區網都連得到 http://<本機IP>:{port},{extra}",
                          "改回 HOST=127.0.0.1,用 sudo scripts/deploy-nginx-lan.sh "
                          "走 nginx(TLS + 帳密)給別人連")
    return make_check("bind", "面板綁定位址", "ok", f"{host}:{port}",
                      "只收本機連線;外面要連走下面的 nginx 代理或 ssh -L")


def lan_proxy_check() -> dict:
    """The drift this exists for: change PORT in local.env, forget the proxy, and
    every bookmark 502s while `ss` still shows the panel listening — which reads
    exactly like "I did expose the port and people still can't connect"."""
    site = _nginx_site()
    port = _shell_var("PORT", "8077")
    if site is None:
        return make_check("lan_proxy", "區網代理 (nginx)", "skip", "沒有安裝",
                          "現在只有本機 / ssh -L 轉發連得到",
                          "要給同事一個固定網址:sudo scripts/deploy-nginx-lan.sh")
    url = f"https://{site['host']}{site['url_port']}/"
    if site["proxy_port"] != port:
        return make_check(
            "lan_proxy", "區網代理 (nginx)", "err", url,
            f"代理轉給 127.0.0.1:{site['proxy_port'] or '?'},但面板綁的是 :{port} —— "
            "從區網連只會拿到 502",
            "重跑 sudo scripts/deploy-nginx-lan.sh(它會直接讀 local.env 的 PORT)")
    if probe(["systemctl", "is-active", "nginx"]).strip() != "active":
        return make_check("lan_proxy", "區網代理 (nginx)", "err", url,
                          "站台設定在,但 nginx 沒有在跑",
                          "sudo systemctl start nginx")
    return make_check("lan_proxy", "區網代理 (nginx)", "ok", url,
                      f"轉給 127.0.0.1:{port};連線要 basic auth 帳密")


def network_checks() -> list[dict]:
    return [bind_check(), lan_proxy_check()]


# --------------------------------------------------------------------------- #
def system_report(deep: bool = True) -> list[dict]:
    """All system-level checks, in the order they matter when standing up a box."""
    return [colmap_check(), exiftool_check(), *ffmpeg_checks(), *storage_checks(), cudnn_check(),
            *gcs_checks(deep), *supersplat_checks(), *network_checks(), env_var_check()]
