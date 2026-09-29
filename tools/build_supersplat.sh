#!/usr/bin/env bash
# Build the latest stable editor plus Recon Studio patches in an isolated directory.
# The sibling checkout is a read-only source cache; never discard local changes.
set -euo pipefail
SUPERSPLAT_VER="${SUPERSPLAT_VER:-latest}"
REPO_URL="https://github.com/playcanvas/supersplat.git"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${SUPERSPLAT_SRC:-$(dirname "$HERE")/supersplat}"
DEST="${SUPERSPLAT_DEST:-$HERE/static/supersplat}"
for bin in git node npm sha256sum flock; do
  command -v "$bin" >/dev/null || { echo "ERROR: '$bin' is required" >&2; exit 1; }
done
# Concurrent startup/build requests must not race the deployment rename.
LOCK_KEY="$(printf '%s' "$HERE" | sha256sum | cut -c1-16)"
if [[ "${SUPERSPLAT_SKIP_LOCK:-0}" != 1 ]]; then
  exec 9>"${TMPDIR:-/tmp}/reconstudio-supersplat-$LOCK_KEY.lock"
  flock 9
fi
if [[ "$SUPERSPLAT_VER" == "latest" ]]; then
  SUPERSPLAT_VER="$(git ls-remote --tags --refs "$REPO_URL" 'v*' | awk -F/ '$NF ~ /^v[0-9]+\.[0-9]+\.[0-9]+$/ {print $NF}' | sort -V | tail -1)"
  [[ -n "$SUPERSPLAT_VER" ]] || { echo "ERROR: could not resolve latest tag" >&2; exit 1; }
fi
# v3 uses a different native chunk loader and WebGPU renderer; do not apply v2 internals.
case "$SUPERSPLAT_VER" in
  v2.*) PATCHES=("$HERE/tools/supersplat-reconstudio.patch" "$HERE/tools/supersplat-performance.patch" "$HERE/tools/supersplat-wheel.patch") ;;
  v3.*) PATCHES=("$HERE/tools/supersplat-v3.patch" "$HERE/tools/supersplat-wheel.patch") ;;
  *) echo "ERROR: unsupported SuperSplat release $SUPERSPLAT_VER; keeping current bundle" >&2; exit 1 ;;
esac
# WebGPU is unavailable or unstable on some browser/GPU combinations. Keep a
# patched WebGL editor ready before deploying v3 so the panel can retry there.
if [[ "$SUPERSPLAT_VER" == v3.* && "${SUPERSPLAT_BUILD_LEGACY:-1}" == 1 ]]; then
  SUPERSPLAT_VER=v2.32.5 SUPERSPLAT_DEST="$HERE/static/supersplat-legacy" \
    SUPERSPLAT_BUILD_LEGACY=0 SUPERSPLAT_SKIP_LOCK=1 "$HERE/tools/build_supersplat.sh"
fi
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
if [[ "$DEST" == "$HERE/static/supersplat-legacy" ]]; then
  BASE_HREF=/static/supersplat-legacy/ npm run build
else
  BASE_HREF=/static/supersplat/ npm run build
fi
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
