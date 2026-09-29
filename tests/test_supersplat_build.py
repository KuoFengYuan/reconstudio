"""The editor rebuild must preserve local work and the last working deployment."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[1] / "tools" / "build_supersplat.sh"


@pytest.fixture
def build_tree(tmp_path):
    if not shutil.which("git") or not shutil.which("flock"):
        pytest.skip("git and flock required")
    project = tmp_path / "panel"
    tools = project / "tools"
    tools.mkdir(parents=True)
    (project / "static").mkdir()
    shutil.copy2(BUILD, tools / BUILD.name)
    src = tmp_path / "source"
    src.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(src), *args], check=True, capture_output=True)
    git("init", "-q")
    (src / "a").write_text("original\n")
    (src / "b").write_text("original\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    git("tag", "v2.32.5")
    git("tag", "v2.32.6")
    git("tag", "v3.4.2")
    # The source checkout has real uncommitted work that must survive a rebuild.
    (src / "a").write_text("user changes\n")
    for filename, target in [("supersplat-reconstudio.patch", "a"), ("supersplat-performance.patch", "b")]:
        (tools / filename).write_text(f"--- a/{target}\n+++ b/{target}\n@@ -1 +1 @@\n-original\n+patched\n")
    (tools / "supersplat-v3.patch").write_text("--- a/a\n+++ b/a\n@@ -1 +1 @@\n-original\n+v3-patched\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "node").write_text("#!/bin/sh\nexit 0\n")
    npm = bin_dir / "npm"
    npm.write_text("""#!/bin/sh
if [ "$1" = ci ]; then
  printf 'install\n' >> "$BUILD_CALLS"
else
  mkdir -p dist
  printf 'editor\n' > dist/index.js
  printf '<html>editor</html>\n' > dist/index.html
  printf 'worker\n' > dist/splat-loader-worker.js
fi
""")
    npm.chmod(0o755)
    (bin_dir / "node").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "SUPERSPLAT_SRC": str(src),
           "SUPERSPLAT_VER": "v2.32.5", "BUILD_CALLS": str(tmp_path / "calls"), "TMPDIR": str(tmp_path)}
    # Intercept only remote discovery; all archive/tag/deployment operations use real git.
    git_wrapper = bin_dir / "git"
    git_wrapper.write_text(
        '#!/bin/sh\nif [ "$1" = ls-remote ]; then\n'
        '  [ "${REMOTE_FAIL:-0}" = 0 ] || exit 1\n'
        '  cat "$REMOTE_TAGS"\nelse\n'
        f'  exec "{shutil.which("git")}" "$@"\nfi\n'
    )
    git_wrapper.chmod(0o755)
    tags = tmp_path / "remote-tags"
    tags.write_text("hash\trefs/tags/v2.32.6\nhash\trefs/tags/v9.0.0-rc.1\nhash\trefs/tags/v2.32.5\n")
    env["REMOTE_TAGS"] = str(tags)
    env.pop("FORCE", None)
    return project, src, env


def run_build(tree):
    project, _, env = tree
    return subprocess.run(["bash", str(project / "tools" / BUILD.name)], env=env,
                          capture_output=True, text=True, timeout=15)


def test_rebuild_preserves_source_and_skips_only_matching_patches(build_tree):
    project, src, env = build_tree
    first = run_build(build_tree)
    assert first.returncode == 0, first.stderr
    assert (src / "a").read_text() == "user changes\n"
    dest = project / "static" / "supersplat"
    old_revision = (dest / ".patch-version").read_text()
    assert (dest / "splat-loader-worker.js").is_file()
    assert run_build(build_tree).returncode == 0
    assert Path(env["BUILD_CALLS"]).read_text().count("install") == 1
    patch = project / "tools" / "supersplat-performance.patch"
    patch.write_text(patch.read_text().replace("+patched", "+new-patch"))
    rebuilt = run_build(build_tree)
    assert rebuilt.returncode == 0, rebuilt.stderr
    assert (dest / ".patch-version").read_text() != old_revision
    assert Path(env["BUILD_CALLS"]).read_text().count("install") == 2


def test_incompatible_patch_keeps_deployed_editor(build_tree):
    project, src, _ = build_tree
    assert run_build(build_tree).returncode == 0
    dest = project / "static" / "supersplat"
    before = (dest / ".patch-version").read_text()
    patch = project / "tools" / "supersplat-performance.patch"
    patch.write_text(patch.read_text().replace("-original", "-missing-upstream-line"))
    assert run_build(build_tree).returncode != 0
    assert (dest / ".patch-version").read_text() == before
    assert (dest / "index.js").read_text() == "editor\n"
    assert (src / "a").read_text() == "user changes\n"


def test_default_discovers_latest_stable_and_rechecks_on_each_build(build_tree):
    project, _, env = build_tree
    env.pop("SUPERSPLAT_VER")
    assert run_build(build_tree).returncode == 0
    dest = project / "static" / "supersplat"
    assert (dest / ".version").read_text().strip() == "v2.32.6"
    Path(env["REMOTE_TAGS"]).write_text("hash\trefs/tags/v3.4.2\nhash\trefs/tags/v4.0.0-beta.1\n")
    assert run_build(build_tree).returncode == 0
    assert (dest / ".version").read_text().strip() == "v3.4.2"
    assert Path(env["BUILD_CALLS"]).read_text().count("install") == 2
    assert run_build(build_tree).returncode == 0
    assert Path(env["BUILD_CALLS"]).read_text().count("install") == 2


def test_failed_latest_lookup_keeps_last_working_bundle(build_tree):
    project, _, env = build_tree
    assert run_build(build_tree).returncode == 0
    env["SUPERSPLAT_VER"] = "latest"
    env["REMOTE_FAIL"] = "1"
    assert run_build(build_tree).returncode != 0
    assert (project / "static/supersplat/.version").read_text().strip() == "v2.32.5"
    assert Path(env["BUILD_CALLS"]).read_text().count("install") == 1


def test_pin_works_offline_and_missing_bundle_is_rebuilt(build_tree):
    project, _, env = build_tree
    env["REMOTE_FAIL"] = "1"
    assert run_build(build_tree).returncode == 0
    (project / "static/supersplat/index.js").unlink()
    assert run_build(build_tree).returncode == 0
    assert Path(env["BUILD_CALLS"]).read_text().count("install") == 2


def test_unknown_major_keeps_current_editor(build_tree):
    project, _, env = build_tree
    assert run_build(build_tree).returncode == 0
    env["SUPERSPLAT_VER"] = "latest"
    Path(env["REMOTE_TAGS"]).write_text("hash\trefs/tags/v4.0.0\n")
    result = run_build(build_tree)
    assert result.returncode != 0
    assert "unsupported SuperSplat release v4.0.0" in result.stderr
    assert (project / "static/supersplat/.version").read_text().strip() == "v2.32.5"


def test_v3_patch_conflict_does_not_replace_v2(build_tree):
    project, _, env = build_tree
    assert run_build(build_tree).returncode == 0
    env["SUPERSPLAT_VER"] = "v3.4.2"
    patch = project / "tools/supersplat-v3.patch"
    patch.write_text(patch.read_text().replace("-original", "-incompatible"))
    assert run_build(build_tree).returncode != 0
    assert (project / "static/supersplat/.version").read_text().strip() == "v2.32.5"


@pytest.mark.parametrize("pin", [None, "v2.32.5"])
def test_startup_requests_latest_unless_explicitly_pinned(tmp_path, pin):
    """Exercise run.sh so it cannot silently override the builder's new default."""
    if not shutil.which("flock"):
        pytest.skip("flock required")
    project = tmp_path / "panel"
    (project / "tools").mkdir(parents=True)
    (project / "static/supersplat").mkdir(parents=True)
    run_script = BUILD.parents[1] / "run.sh"
    shutil.copy2(run_script, project / "run.sh")
    (project / "tools/detect.sh").write_text("detect_data_disk() { echo /tmp; }\n")
    build = project / "tools/build_supersplat.sh"
    build.write_text('#!/bin/sh\nprintf "%s\\n" "$SUPERSPLAT_VER" > "$REQUESTED_VERSION"\n')
    build.chmod(0o755)
    fake_python = tmp_path / "conda/envs/test/bin/python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/bin/sh\nexit 0\n")
    fake_python.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "ss").write_text("#!/bin/sh\nexit 0\n")
    (fake_bin / "ss").chmod(0o755)
    (project / "local.env").write_text(
        f"CONDA_ROOT={tmp_path}/conda\nCONDA_ENV=test\n"
        f"RECON_STUDIO_DATA={tmp_path}/data\nTMPDIR={tmp_path}/tmp\n"
        "HOST=127.0.0.1\nPORT=8077\nFFMPEG_BIN=ffmpeg\nRECON_STUDIO_BROWSE_ROOT=/\n"
        "SUPERSPLAT_AUTOUPDATE=1\n"
        + (f"SUPERSPLAT_VER={pin}\n" if pin else "unset SUPERSPLAT_VER\n")
    )
    requested = tmp_path / "requested"
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}",
           "REQUESTED_VERSION": str(requested)}
    result = subprocess.run(["bash", "run.sh"], cwd=project, env=env,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert requested.read_text().strip() == (pin or "latest")
    assert f"requested {pin or 'latest'}" in result.stdout
