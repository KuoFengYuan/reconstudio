"""System-level preflight checks — the deploy-to-a-new-machine half of /doctor.

Fully offline (README's rule for tests/): every probe is monkeypatched, so these
also cover the branches a given machine can never reach — an ffmpeg build without
`blurdetect`, a dead FFMPEG_BIN, an unwritable data disk — which is precisely
where a preflight tool earns its keep.
"""
import pytest

from pipeline import preflight
from pipeline.preflight import (
    STATUSES,
    _dir_check,
    env_var_check,
    exiftool_check,
    ffmpeg_checks,
    known_env_vars,
    make_check,
    storage_checks,
    system_report,
)

_KEYS = {"key", "label", "status", "value", "detail", "hint"}

# `ffmpeg -filters` output shape; the flags column varies per filter.
_FILTERS_WITH_BLUR = " ... blurdetect        V->V       Blurdetect filter.\n"
_FILTERS_WITHOUT = " ... boxblur           V->V       Blur the input.\n"


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """Make ffmpeg_checks() deterministic: `ffmpeg` resolves, and each probe
    returns canned output. Returns a dict the test can mutate per branch."""
    out = {"filters": _FILTERS_WITH_BLUR, "hwaccels": "Hardware acceleration methods:\ncuda\nvaapi\n"}
    monkeypatch.setattr("shutil.which", lambda b: f"/usr/bin/{b.split('/')[-1]}")

    def probe(argv, timeout=None):
        if "-filters" in argv:
            return out["filters"]
        if "-hwaccels" in argv:
            return out["hwaccels"]
        return "ffmpeg version 6.1.1-3ubuntu5 Copyright (c)\nconfiguration: …"

    monkeypatch.setattr(preflight, "probe", probe)
    return out


# --------------------------------------------------------------------------- #
# the uniform shape both renderers iterate blindly
# --------------------------------------------------------------------------- #
def test_check_shape_is_exactly_what_the_renderers_read():
    assert set(make_check("k", "l", "ok")) == _KEYS


def test_check_rejects_an_unknown_status():
    # doctor.html and doctor_cli.py both look the status up in a badge table; a
    # typo would render as "?" instead of failing loudly. Catch it at construction.
    with pytest.raises(AssertionError):
        make_check("k", "l", "OK")          # must be lowercase, from STATUSES


def test_system_report_entries_all_have_valid_status(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)     # no local.env here
    monkeypatch.setattr(preflight, "probe", lambda argv, timeout=None: "")
    monkeypatch.setattr(preflight.settings, "data_dir", tmp_path / "data")
    for c in system_report(deep=False):
        assert set(c) == _KEYS, c
        assert c["status"] in STATUSES, c
        assert c["label"], c


# --------------------------------------------------------------------------- #
# exiftool: optional, but its absence lets a Canon SIGABRT through unfixed
# --------------------------------------------------------------------------- #
def test_missing_exiftool_warns_not_fails(monkeypatch):
    # Unlike colmap/ffmpeg this can't be "err": most datasets never hit the
    # Canon empty-tag bug, so a machine without exiftool still works most of
    # the time — it just won't self-heal the one camera model that needs it.
    monkeypatch.setattr("shutil.which", lambda b: None)
    chk = exiftool_check()
    assert chk["status"] == "warn"
    assert chk["hint"]


def test_exiftool_found_is_ok(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/exiftool" if b == "exiftool" else None)
    monkeypatch.setattr(preflight, "probe", lambda argv, timeout=None: "13.25\n")
    chk = exiftool_check()
    assert chk["status"] == "ok"
    assert chk["value"] == "/usr/bin/exiftool"


# --------------------------------------------------------------------------- #
# ffmpeg: resolve it the way the pipeline does, and check the feature we need
# --------------------------------------------------------------------------- #
def test_missing_ffmpeg_is_a_hard_failure(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda b: None)
    checks = {c["key"]: c for c in ffmpeg_checks()}
    assert checks["ffmpeg"]["status"] == "err"
    assert checks["ffmpeg"]["hint"]


def test_missing_blurdetect_filter_is_a_hard_failure(fake_ffmpeg):
    # ffmpeg runs fine but this build can't score blur -> 抽幀 dies mid-job.
    fake_ffmpeg["filters"] = _FILTERS_WITHOUT
    checks = {c["key"]: c for c in ffmpeg_checks()}
    assert checks["ffmpeg"]["status"] == "ok"
    assert checks["ffmpeg_blurdetect"]["status"] == "err"


def test_dead_ffmpeg_bin_warns_instead_of_failing(monkeypatch, fake_ffmpeg):
    # A local.env copied from another machine points FFMPEG_BIN at a path that
    # doesn't exist here. frames._ffmpeg_bin falls back to PATH, so 抽幀 WORKS —
    # reporting err would send someone debugging a non-problem.
    monkeypatch.setattr(preflight.settings, "ffmpeg_bin", "/mnt/ssd1/bin/ffmpeg-nvdec")
    chk = {c["key"]: c for c in ffmpeg_checks()}["ffmpeg"]
    assert chk["status"] == "warn"
    assert "/mnt/ssd1/bin/ffmpeg-nvdec" in chk["detail"]      # name the dead setting


def test_hwaccel_disabled_is_skip_not_warn(monkeypatch, fake_ffmpeg):
    # "I don't have NVDEC and that's fine" must not look like a problem.
    monkeypatch.setattr(preflight.settings, "ffmpeg_hwaccel", "none")
    assert {c["key"]: c for c in ffmpeg_checks()}["ffmpeg_hwaccel"]["status"] == "skip"


def test_hwaccel_cuda_available_is_ok(monkeypatch, fake_ffmpeg):
    monkeypatch.setattr(preflight.settings, "ffmpeg_hwaccel", "cuda")
    assert {c["key"]: c for c in ffmpeg_checks()}["ffmpeg_hwaccel"]["status"] == "ok"


def test_hwaccel_other_than_cuda_warns_because_frames_only_sends_cuda(monkeypatch, fake_ffmpeg):
    # vaapi is in -hwaccels, but frames.py hardcodes `-hwaccel cuda`. Reporting
    # "ok, vaapi available" would promise something the pipeline never requests.
    monkeypatch.setattr(preflight.settings, "ffmpeg_hwaccel", "vaapi")
    chk = {c["key"]: c for c in ffmpeg_checks()}["ffmpeg_hwaccel"]
    assert chk["status"] == "warn"
    assert "cuda" in chk["detail"]


def test_hwaccel_cuda_unsupported_by_this_build_warns(monkeypatch, fake_ffmpeg):
    fake_ffmpeg["hwaccels"] = "Hardware acceleration methods:\nvaapi\n"
    monkeypatch.setattr(preflight.settings, "ffmpeg_hwaccel", "cuda")
    assert {c["key"]: c for c in ffmpeg_checks()}["ffmpeg_hwaccel"]["status"] == "warn"


# --------------------------------------------------------------------------- #
# storage: writable + roomy, and WITHOUT creating anything
# --------------------------------------------------------------------------- #
def test_dir_check_does_not_create_the_directory(tmp_path):
    # A preflight run must only observe — `./run.sh --doctor` gets run on machines
    # you're merely inspecting.
    target = tmp_path / "made" / "on" / "demand"
    chk = _dir_check("d", "測試目錄", target)
    assert not target.exists()
    assert chk["status"] in ("ok", "warn")           # warn only if the disk is full
    assert "尚未建立" in chk["detail"]                # …but say it isn't there yet


def test_dir_check_reports_an_existing_writable_dir(tmp_path):
    chk = _dir_check("d", "測試目錄", tmp_path)
    assert chk["status"] in ("ok", "warn")
    assert "尚未建立" not in chk["detail"]


def test_dir_check_fails_on_unwritable_dir(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        chk = _dir_check("d", "測試目錄", ro / "sub")
    finally:
        ro.chmod(0o700)                              # so pytest can clean up
    if chk["status"] != "err":
        pytest.skip("running as root: unwritable directories aren't unwritable")
    assert chk["hint"]                               # a failure must say what to do


def test_storage_checks_follow_tmpdir(monkeypatch, tmp_path):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert any(c["key"] == "tmpdir" and c["value"] == str(tmp_path)
               for c in storage_checks())


# --------------------------------------------------------------------------- #
# local.env sanity — the check that exists because Settings swallows typos
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_repo(monkeypatch, tmp_path):
    """Point preflight at a throwaway repo root so local.env / local.env.example
    can be written per test."""
    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)
    (tmp_path / "local.env.example").write_text(
        "# HOST=127.0.0.1\n# PORT=8077\n# SUPERSPLAT_AUTOUPDATE=0\n")
    return tmp_path


def test_no_local_env_is_skip_not_a_problem(fake_repo):
    assert env_var_check()["status"] == "skip"


def test_unprefixed_typo_is_flagged(fake_repo):
    # The case scanning os.environ by prefix could never catch: Settings ignores
    # COLMAP_BNI and silently uses `colmap` from PATH.
    (fake_repo / "local.env").write_text("COLMAP_BNI=/opt/colmap/colmap\n")
    chk = env_var_check()
    assert chk["status"] == "warn"
    assert "COLMAP_BNI" in chk["value"]


def test_stale_var_from_a_rewritten_feature_is_flagged(fake_repo):
    # DEPTH_* died when 🌊 深度 moved to LichtFeld's preprocess subcommand.
    (fake_repo / "local.env").write_text("DEPTH_CONDA_ENV=da2\nDEPTH_MODEL=x\n")
    assert env_var_check()["status"] == "warn"


def test_recognised_vars_are_not_flagged(fake_repo):
    (fake_repo / "local.env").write_text(
        "COLMAP_BIN=colmap\nexport RECON_STUDIO_DATA=/d   # 註解\n"
        "COLMAP_PANEL_MAX_JOBS=8\nHOST=0.0.0.0\nSUPERSPLAT_AUTOUPDATE=0\n")
    assert env_var_check()["status"] == "ok"


def test_commented_out_lines_are_not_treated_as_set(fake_repo):
    (fake_repo / "local.env").write_text("# DEPTH_MODEL=whatever\n")
    assert env_var_check()["status"] == "ok"


def test_known_env_vars_derives_from_settings_and_the_example_file():
    # Adding a Settings field or documenting a knob must not require editing
    # preflight.py — that hand-maintained allowlist is what this replaced.
    known = known_env_vars()
    assert {"COLMAP_BIN", "FFMPEG_BIN", "RECON_STUDIO_DATA"} <= known   # aliases
    assert {"HOST", "PORT", "SUPERSPLAT_AUTOUPDATE", "TMPDIR"} <= known  # example


def test_every_settings_alias_is_documented_in_the_example_file():
    # local.env.example is now load-bearing (known_env_vars reads it), so a new
    # Settings field that nobody documented would be reported as invalid.
    from pipeline.config import Settings
    aliases = {f.validation_alias.upper() for f in Settings.model_fields.values()
               if isinstance(f.validation_alias, str)}
    example = (preflight.REPO_ROOT / "local.env.example").read_text()
    documented = {v.upper() for v in preflight._VAR_RE.findall(example)}
    assert aliases <= documented, f"未寫進 local.env.example: {sorted(aliases - documented)}"


# --------------------------------------------------------------------------- #
# per-backend rows: the hints must live in the DATA, not in a renderer
# --------------------------------------------------------------------------- #
def test_backend_checks_carry_remedies_for_every_failure():
    """doctor.html used to show a red light with no "how do I fix it" text
    because the hints were hardcoded in doctor_cli.py only. Both renderers now
    iterate these rows, so a failing row without a hint is the regression."""
    from pathlib import Path

    from pipeline.backends import _backend_checks

    checks = _backend_checks(
        {"conda_env": "gs2m", "train_script": "train.py"}, None, Path("/nope/GS-2M"), False,
        {"torch": "2.1.0", "cuda": False,
         "import_fail": {"diff_gaussian_rasterization": "ImportError(no kernel image)"}})
    assert all(c["status"] in STATUSES for c in checks)
    # Backends are opt-in: never a hard failure, only warn.
    assert not [c for c in checks if c["status"] == "err"]
    failing = [c for c in checks if c["status"] == "warn"]
    assert len(failing) == 4                      # env python / repo / torch / submodules
    assert all(c["hint"] for c in failing), [c["label"] for c in failing if not c["hint"]]


def test_ready_backend_rows_are_all_ok(tmp_path):
    from pipeline.backends import _backend_checks

    (tmp_path / "train.py").touch()
    checks = _backend_checks({"conda_env": "gs2m"}, tmp_path / "python", tmp_path, True,
                             {"torch": "2.11.0", "cuda": True, "import_fail": {}})
    assert [c["status"] for c in checks] == ["ok"] * 4


# --------------------------------------------------------------------------- #
# Reachability — "the port is up but nobody can connect"
#
# All three failures here look identical from inside the panel (it is listening,
# it answers on loopback), which is why they are checks and not comments.
# --------------------------------------------------------------------------- #
_SITE = """
map $http_upgrade $rs_connection_upgrade { default upgrade; '' close; }
server {
    listen 192.168.90.146:8443 ssl http2;
    server_name recon.example.tw 192.168.90.146 localhost;
    location / { proxy_pass http://127.0.0.1:%s; }
}
"""


@pytest.fixture
def site(monkeypatch, tmp_path):
    """Write an enabled-site file and point preflight at it. Call with no port to
    stand for "no proxy installed"."""
    conf = tmp_path / "reconstudio.conf"
    monkeypatch.setattr(preflight, "_NGINX_SITE", conf)
    monkeypatch.setattr(preflight, "probe", lambda argv, timeout=None: "active\n")

    def write(panel_port: str | None):
        if panel_port is not None:
            conf.write_text(_SITE % panel_port)
        return conf
    return write


def test_a_bare_HOST_from_conda_is_not_read_as_a_bind_address(monkeypatch, fake_repo):
    """conda's compiler activation exports HOST=x86_64-conda-linux-gnu. Reading
    the bare name reported a build triplet as the panel's address — and, before
    run.sh stopped inheriting it, tried to bind it."""
    monkeypatch.setenv("HOST", "x86_64-conda-linux-gnu")
    monkeypatch.delenv("RECON_STUDIO_HOST", raising=False)
    assert preflight._shell_var("HOST", "127.0.0.1") == "127.0.0.1"


def test_run_sh_s_resolved_value_wins(monkeypatch, fake_repo):
    (fake_repo / "local.env").write_text("PORT=8077\n")
    monkeypatch.setenv("RECON_STUDIO_PORT", "9099")
    assert preflight._shell_var("PORT", "8077") == "9099"


def test_local_env_is_the_fallback_for_a_hand_rolled_uvicorn(monkeypatch, fake_repo):
    monkeypatch.delenv("RECON_STUDIO_PORT", raising=False)
    (fake_repo / "local.env").write_text("PORT=8078\n")
    assert preflight._shell_var("PORT", "8077") == "8078"


def test_loopback_only_is_ok_and_says_so(monkeypatch, fake_repo, site):
    site(None)
    monkeypatch.setenv("RECON_STUDIO_HOST", "127.0.0.1")
    monkeypatch.setenv("RECON_STUDIO_PORT", "8077")
    chk = preflight.bind_check()
    assert chk["status"] == "ok" and chk["value"] == "127.0.0.1:8077"


def test_binding_the_world_is_a_warning_because_the_panel_has_no_login(
        monkeypatch, fake_repo, site):
    site(None)
    monkeypatch.setenv("RECON_STUDIO_HOST", "0.0.0.0")
    monkeypatch.setenv("RECON_STUDIO_PORT", "8077")
    chk = preflight.bind_check()
    assert chk["status"] == "warn"
    assert "deploy-nginx-lan.sh" in chk["hint"]


def test_no_proxy_installed_is_skip_with_the_way_to_get_one(monkeypatch, fake_repo, site):
    site(None)
    chk = preflight.lan_proxy_check()
    assert chk["status"] == "skip" and "deploy-nginx-lan.sh" in chk["hint"]


def test_a_matching_proxy_reports_the_url_to_hand_out(monkeypatch, fake_repo, site):
    site("8078")
    monkeypatch.setenv("RECON_STUDIO_PORT", "8078")
    chk = preflight.lan_proxy_check()
    assert chk["status"] == "ok"
    assert chk["value"] == "https://recon.example.tw:8443/"


def test_a_proxy_pointing_at_the_old_port_is_an_error(monkeypatch, fake_repo, site):
    """Change PORT in local.env, forget the proxy: every bookmark 502s while `ss`
    still shows the panel listening. This is the whole reason the check exists."""
    site("8077")
    monkeypatch.setenv("RECON_STUDIO_PORT", "8078")
    chk = preflight.lan_proxy_check()
    assert chk["status"] == "err"
    assert "8077" in chk["detail"] and "8078" in chk["detail"]


def test_a_stopped_nginx_is_an_error_even_though_the_site_is_installed(
        monkeypatch, fake_repo, site):
    site("8078")
    monkeypatch.setenv("RECON_STUDIO_PORT", "8078")
    monkeypatch.setattr(preflight, "probe", lambda argv, timeout=None: "inactive\n")
    chk = preflight.lan_proxy_check()
    assert chk["status"] == "err" and "systemctl start nginx" in chk["hint"]


def test_network_checks_have_the_standard_shape(fake_repo, site):
    site("8077")
    for chk in preflight.network_checks():
        assert set(chk) == _KEYS and chk["status"] in STATUSES


def test_the_reachability_checks_reach_the_report(monkeypatch):
    monkeypatch.setattr(preflight, "probe", lambda argv, timeout=None: "")
    keys = {c["key"] for c in system_report(deep=False)}
    assert {"bind", "lan_proxy"} <= keys


_SPLIT_SITE = """
server {
    listen 192.168.90.146:443 ssl http2;
    server_name recon.example.tw;
    include %s;
}
server {
    listen 192.168.90.146:8443 ssl http2;
    server_name 192.168.90.146 localhost;
    include %s;
}
"""
_SPLIT_COMMON = "location / { proxy_pass http://127.0.0.1:%s; }\n"


def test_the_proxy_port_is_found_through_the_included_snippet(monkeypatch, tmp_path, fake_repo):
    """Both vhosts share one proxy_pass, so it lives in the include. Reading only
    the site file would report permanent drift on a perfectly good install."""
    common = tmp_path / "reconstudio-common.conf"
    common.write_text(_SPLIT_COMMON % "8078")
    conf = tmp_path / "reconstudio.conf"
    conf.write_text(_SPLIT_SITE % (common, common))
    monkeypatch.setattr(preflight, "_NGINX_SITE", conf)
    monkeypatch.setattr(preflight, "probe", lambda argv, timeout=None: "active\n")
    monkeypatch.setenv("RECON_STUDIO_PORT", "8078")

    chk = preflight.lan_proxy_check()
    assert chk["status"] == "ok"
    # 443 is implied by https://; the reported URL is the one people type.
    assert chk["value"] == "https://recon.example.tw/"


def test_a_broken_include_does_not_crash_the_report(monkeypatch, tmp_path, fake_repo):
    conf = tmp_path / "reconstudio.conf"
    conf.write_text(_SPLIT_SITE % ("/nonexistent/a.conf", "/nonexistent/a.conf"))
    monkeypatch.setattr(preflight, "_NGINX_SITE", conf)
    monkeypatch.setattr(preflight, "probe", lambda argv, timeout=None: "active\n")
    assert preflight.lan_proxy_check()["status"] == "err"     # unknown port = drift
