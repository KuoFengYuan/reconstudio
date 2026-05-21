#!/usr/bin/env bash
# Build the self-hosted SuperSplat editor (去背) and deploy it under
# static/supersplat/. Reproducible: clones a pinned SuperSplat, applies the
# Recon Studio patches (inline send-back API + mouse-wheel zoom fix), builds with
# the right base path, and strips sourcemaps. Re-run after bumping SUPERSPLAT_VER.
#
#   ./tools/build_supersplat.sh            # uses ../supersplat (sibling of this repo)
#
# Needs: node (>=18) + npm + git. SuperSplat is MIT (see THIRD_PARTY_NOTICES.md).
set -euo pipefail

SUPERSPLAT_VER="${SUPERSPLAT_VER:-v2.26.2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"        # reconstudio repo root
SRC="${SUPERSPLAT_SRC:-$(dirname "$HERE")/supersplat}"          # sibling ../supersplat
PATCH="$HERE/tools/supersplat-reconstudio.patch"
DEST="$HERE/static/supersplat"

if [[ ! -d "$SRC/.git" ]]; then
  echo "[1/5] cloning SuperSplat $SUPERSPLAT_VER -> $SRC"
  git clone --depth 1 --branch "$SUPERSPLAT_VER" https://github.com/playcanvas/supersplat.git "$SRC"
fi

echo "[2/5] applying Recon Studio patch (idempotent)"
cd "$SRC"
if git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "      already applied — skipping"
else
  git apply "$PATCH"
fi

echo "[3/5] npm ci"
npm ci

echo "[4/5] building (BASE_HREF=/static/supersplat/)"
BASE_HREF=/static/supersplat/ npm run build

echo "[5/5] deploying -> $DEST (sourcemaps stripped)"
rm -rf "$DEST"
cp -r dist "$DEST"
find "$DEST" -name '*.map' -delete
echo "done. served at /static/supersplat/index.html ($(du -sh "$DEST" | cut -f1))"
