#!/usr/bin/env bash
# Build the self-hosted SuperSplat editor (去背 + 點雲 viewer) and deploy it under
# static/supersplat/. Reproducible: clones a pinned SuperSplat, applies the
# Recon Studio patches (inline send-back API + mouse-wheel zoom fix), builds with
# the right base path, and strips sourcemaps.
#
#   ./tools/build_supersplat.sh                         # pinned default version
#   SUPERSPLAT_VER=latest ./tools/build_supersplat.sh   # newest upstream tag
#   FORCE=1 ./tools/build_supersplat.sh                 # rebuild even if up to date
#
# The deployed version is stamped in static/supersplat/.version: when it already
# matches the requested version the script exits immediately (run.sh calls this
# with SUPERSPLAT_VER=latest at every startup, so the no-change path must cost
# one `git ls-remote` only). Deployment is ATOMIC (build lands in a sibling dir,
# then one mv) — a failed build, or one racing with a running server, never
# leaves /static/supersplat/ half-written.
#
# Needs: node (>=18) + npm + git. SuperSplat is MIT (see THIRD_PARTY_NOTICES.md).
set -euo pipefail

SUPERSPLAT_VER="${SUPERSPLAT_VER:-v2.26.2}"
REPO_URL="https://github.com/playcanvas/supersplat.git"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"        # reconstudio repo root
SRC="${SUPERSPLAT_SRC:-$(dirname "$HERE")/supersplat}"          # sibling ../supersplat
PATCH="$HERE/tools/supersplat-reconstudio.patch"
DEST="$HERE/static/supersplat"

for bin in git node npm; do
  command -v "$bin" >/dev/null || { echo "ERROR: '$bin' not found (needed to build SuperSplat)" >&2; exit 1; }
done

if [[ "$SUPERSPLAT_VER" == "latest" ]]; then
  SUPERSPLAT_VER="$(git ls-remote --tags --refs "$REPO_URL" 'v*' \
                    | awk -F/ '{print $NF}' | sort -V | tail -1)"
  [[ -n "$SUPERSPLAT_VER" ]] || { echo "ERROR: could not resolve latest tag (offline?)" >&2; exit 1; }
  echo "latest upstream tag: $SUPERSPLAT_VER"
fi

if [[ -z "${FORCE:-}" && -f "$DEST/.version" && "$(cat "$DEST/.version")" == "$SUPERSPLAT_VER" ]]; then
  echo "already at $SUPERSPLAT_VER — nothing to do (FORCE=1 to rebuild)"
  exit 0
fi

if [[ ! -d "$SRC/.git" ]]; then
  echo "[1/5] cloning SuperSplat $SUPERSPLAT_VER -> $SRC"
  git clone --depth 1 --branch "$SUPERSPLAT_VER" "$REPO_URL" "$SRC"
  cd "$SRC"
else
  echo "[1/5] checking out $SUPERSPLAT_VER in $SRC"
  cd "$SRC"
  git fetch --depth 1 origin "refs/tags/$SUPERSPLAT_VER:refs/tags/$SUPERSPLAT_VER" 2>/dev/null || true
  # -f discards a previously applied patch so [2/5] always starts from pristine upstream
  git checkout -f "$SUPERSPLAT_VER" 2>/dev/null || git checkout -f "tags/$SUPERSPLAT_VER"
fi

echo "[2/5] applying Recon Studio patch (idempotent)"
if git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "      already applied — skipping"
else
  # a hard failure here usually means a NEW upstream version moved the patched
  # code; the caller (run.sh) keeps serving the previously deployed bundle.
  git apply "$PATCH"
fi

echo "[3/5] npm ci"
npm ci

echo "[4/5] building (BASE_HREF=/static/supersplat/)"
BASE_HREF=/static/supersplat/ npm run build

echo "[5/5] deploying -> $DEST (atomic swap, sourcemaps stripped)"
rm -rf "$DEST.new" "$DEST.old"
cp -r dist "$DEST.new"
find "$DEST.new" -name '*.map' -delete
echo "$SUPERSPLAT_VER" > "$DEST.new/.version"
[[ -d "$DEST" ]] && mv "$DEST" "$DEST.old"
mv "$DEST.new" "$DEST"
rm -rf "$DEST.old"
echo "done. $SUPERSPLAT_VER served at /static/supersplat/index.html ($(du -sh "$DEST" | cut -f1))"
