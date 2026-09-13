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
    # The source checkout has real uncommitted work that must survive a rebuild.
    (src / "a").write_text("user changes\n")
    for filename, target in [("supersplat-reconstudio.patch", "a"), ("supersplat-performance.patch", "b")]:
        (tools / filename).write_text(f"--- a/{target}\n+++ b/{target}\n@@ -1 +1 @@\n-original\n+patched\n")
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
  printf 'worker\n' > dist/splat-loader-worker.js
fi
""")
    npm.chmod(0o755)
    (bin_dir / "node").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "SUPERSPLAT_SRC": str(src),
           "SUPERSPLAT_VER": "v2.32.5", "BUILD_CALLS": str(tmp_path / "calls"), "TMPDIR": str(tmp_path)}
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
