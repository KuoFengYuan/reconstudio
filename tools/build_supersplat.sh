#!/usr/bin/env bash
# Build the latest stable editor plus Recon Studio patches in an isolated directory.
# The sibling checkout is a read-only source cache; never discard local changes.
set -euo pipefail
SUPERSPLAT_VER="${SUPERSPLAT_VER:-latest}"
REPO_URL="https://github.com/playcanvas/supersplat.git"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${SUPERSPLAT_SRC:-$(dirname "$HERE")/supersplat}"
DEST="$HERE/static/supersplat"
for bin in git node npm sha256sum flock; do
  command -v "$bin" >/dev/null || { echo "ERROR: '$bin' is required" >&2; exit 1; }
done
# Concurrent startup/build requests must not race the deployment rename.
LOCK_KEY="$(printf '%s' "$HERE" | sha256sum | cut -c1-16)"
exec 9>"${TMPDIR:-/tmp}/reconstudio-supersplat-$LOCK_KEY.lock"
flock 9
if [[ "$SUPERSPLAT_VER" == "latest" ]]; then
  SUPERSPLAT_VER="$(git ls-remote --tags --refs "$REPO_URL" 'v*' | awk -F/ '$NF ~ /^v[0-9]+\.[0-9]+\.[0-9]+$/ {print $NF}' | sort -V | tail -1)"
  [[ -n "$SUPERSPLAT_VER" ]] || { echo "ERROR: could not resolve latest tag" >&2; exit 1; }
fi
# v3 uses a different native chunk loader and WebGPU renderer; do not apply v2 internals.
case "$SUPERSPLAT_VER" in
  v2.*) PATCHES=("$HERE/tools/supersplat-reconstudio.patch" "$HERE/tools/supersplat-performance.patch") ;;
  v3.*) PATCHES=("$HERE/tools/supersplat-v3.patch") ;;
  *) echo "ERROR: unsupported SuperSplat release $SUPERSPLAT_VER; keeping current bundle" >&2; exit 1 ;;
esac
echo "target: $SUPERSPLAT_VER"
RECON_BUILD_ID="$(cat "$HERE/tools/build_supersplat.sh" "${PATCHES[@]}" | sha256sum | cut -c1-16)"
RECON_BUILD_ID="$SUPERSPLAT_VER-$RECON_BUILD_ID"
export RECON_BUILD_ID
if [[ -z "${FORCE:-}" && -s "$DEST/index.html" && -s "$DEST/index.js" &&
      ( "$SUPERSPLAT_VER" != v2.* || -s "$DEST/splat-loader-worker.js" ) && -f "$DEST/.version" && -f "$DEST/.patch-version" &&
      "$(cat "$DEST/.version")" == "$SUPERSPLAT_VER" && "$(cat "$DEST/.patch-version")" == "$RECON_BUILD_ID" ]]; then
  echo "already at $SUPERSPLAT_VER + $RECON_BUILD_ID"
  exit 0
fi
WORK="$(mktemp -d "${TMPDIR:-/tmp}/reconstudio-supersplat.XXXXXX")"
STAGE=""
cleanup() {
  rm -rf "$WORK"
  [[ -z "$STAGE" ]] || rm -rf "$STAGE"
}
trap cleanup EXIT
if git -C "$SRC" rev-parse --verify "refs/tags/$SUPERSPLAT_VER" >/dev/null 2>&1; then
  git -C "$SRC" archive "refs/tags/$SUPERSPLAT_VER" | tar -x -C "$WORK"
else
  git clone --depth 1 --branch "$SUPERSPLAT_VER" "$REPO_URL" "$WORK"
fi
cd "$WORK"
echo "[1/4] applying Recon Studio patches to $SUPERSPLAT_VER"
for patch in "${PATCHES[@]}"; do git apply "$patch"; done
echo "[2/4] installing pinned dependencies"
npm ci
echo "[3/4] building editor"
BASE_HREF=/static/supersplat/ npm run build
[[ -s dist/index.html && -s dist/index.js &&
   ( "$SUPERSPLAT_VER" != v2.* || -s dist/splat-loader-worker.js ) ]] || { echo 'ERROR: incomplete editor build' >&2; exit 1; }
echo "[4/4] deploying editor"
STAGE="$(mktemp -d "$HERE/static/.supersplat.XXXXXX")"
cp -a dist/. "$STAGE/"
find "$STAGE" -name '*.map' -delete
printf '%s\n' "$SUPERSPLAT_VER" > "$STAGE/.version"
printf '%s\n' "$RECON_BUILD_ID" > "$STAGE/.patch-version"
# Keep the previous bundle until the new one has been moved successfully.
BACKUP="$WORK/previous"
[[ ! -d "$DEST" ]] || mv "$DEST" "$BACKUP"
if ! mv "$STAGE" "$DEST"; then
  [[ ! -d "$BACKUP" ]] || mv "$BACKUP" "$DEST"
  exit 1
fi
STAGE=""
echo "done: $SUPERSPLAT_VER + $RECON_BUILD_ID"
