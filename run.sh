#!/usr/bin/env bash
# Launch Recon Studio (https://github.com/KuoFengYuan/reconstudio) from a conda env.
#
# Per-machine config: copy local.env.example -> local.env and edit it (gitignored).
# Anything in local.env (or the environment) overrides the defaults below.
#
# One-time setup (env name configurable via CONDA_ENV; default rec):
#   conda create -n rec python=3.10 -y
#   conda run -n rec pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"

# Per-machine overrides (paths to colmap/ffmpeg, conda location, ports, …).
[[ -f local.env ]] && set -a && . ./local.env && set +a

: "${HOST:=127.0.0.1}"
: "${PORT:=8077}"
: "${CONDA_ENV:=rec}"
# conda base: explicit CONDA_ROOT, else `conda info --base`, else common locations.
if [[ -z "${CONDA_ROOT:-}" ]]; then
  CONDA_ROOT="$(conda info --base 2>/dev/null || true)"
  [[ -z "$CONDA_ROOT" ]] && for d in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [[ -d "$d" ]] && CONDA_ROOT="$d" && break
  done
fi

# Binaries: PATH by default; override for non-standard installs.
: "${COLMAP_BIN:=colmap}"; export COLMAP_BIN
# Prefer an NVDEC ffmpeg if present, then any FFMPEG_BIN, else PATH ffmpeg.
if [[ -z "${FFMPEG_BIN:-}" ]]; then
  for f in /mnt/ssd1/bin/ffmpeg-nvdec "$(command -v ffmpeg || true)"; do
    [[ -x "$f" ]] && FFMPEG_BIN="$f" && break
  done
fi
: "${FFMPEG_BIN:=ffmpeg}"; export FFMPEG_BIN

# Job state + scratch on a roomy disk (not the small root fs), with sensible fallbacks.
: "${RECON_STUDIO_DATA:=$( [[ -d /mnt/ssd1 ]] && echo /mnt/ssd1/recon_studio/data || echo "$HOME/.recon_studio" )}"
export RECON_STUDIO_DATA; mkdir -p "$RECON_STUDIO_DATA"
: "${TMPDIR:=$RECON_STUDIO_DATA/tmp}"; export TMPDIR; mkdir -p "$TMPDIR"
# directory-picker root (defaults to the data disk if present, else filesystem root)
: "${RECON_STUDIO_BROWSE_ROOT:=$( [[ -d /mnt/ssd1 ]] && echo /mnt/ssd1 || echo / )}"
export RECON_STUDIO_BROWSE_ROOT

export CONDA_ROOT   # so backends.py / doctor can locate sibling trainer envs (gs2m, …)
ENVPY="$CONDA_ROOT/envs/$CONDA_ENV/bin/python"
[[ -x "$ENVPY" ]] || { echo "conda env '$CONDA_ENV' python not found at: $ENVPY" >&2
                       echo "set CONDA_ROOT/CONDA_ENV in local.env" >&2; exit 1; }
export PYTHONNOUSERSITE=1   # keep ~/.local out of the env

# --- SuperSplat auto-update: sync to the newest upstream release at startup --- #
# Runs in the BACKGROUND so the server starts immediately; the build script swaps
# the bundle atomically when done, so the current version keeps serving meanwhile.
# Fail-soft: offline / node missing / patch conflict on a new release just keeps
# the existing bundle (details in the log). Up-to-date check costs one ls-remote.
# Disable with SUPERSPLAT_AUTOUPDATE=0 (e.g. in local.env), or pin SUPERSPLAT_VER.
: "${SUPERSPLAT_AUTOUPDATE:=1}"
if [[ "$SUPERSPLAT_AUTOUPDATE" == "1" ]]; then
  SS_LOG="$RECON_STUDIO_DATA/supersplat_build.log"
  (
    flock -n 9 || exit 0          # another startup is already syncing
    if SUPERSPLAT_VER="${SUPERSPLAT_VER:-latest}" ./tools/build_supersplat.sh >>"$SS_LOG" 2>&1; then
      echo "supersplat: $(cat static/supersplat/.version 2>/dev/null || echo '?') (synced)"
    else
      echo "supersplat: update failed — keeping current bundle (see $SS_LOG)" >&2
    fi
  ) 9>"$RECON_STUDIO_DATA/supersplat_build.lock" &
fi

echo "Recon Studio -> http://$HOST:$PORT  (env=$CONDA_ENV, ffmpeg=$FFMPEG_BIN, colmap=$COLMAP_BIN)"
exec "$ENVPY" -m uvicorn app:app --host "$HOST" --port "$PORT"
