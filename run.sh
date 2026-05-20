#!/usr/bin/env bash
# Launch the COLMAP Panel from a conda env.
#
# Per-machine config: copy local.env.example -> local.env and edit it (gitignored).
# Anything in local.env (or the environment) overrides the defaults below.
#
# One-time setup:
#   conda create -n colmap_panel python=3.10 -y
#   conda run -n colmap_panel pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"

# Per-machine overrides (paths to colmap/ffmpeg, conda location, ports, …).
[[ -f local.env ]] && set -a && . ./local.env && set +a

: "${HOST:=127.0.0.1}"
: "${PORT:=8077}"
: "${CONDA_ENV:=colmap_panel}"
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
: "${COLMAP_PANEL_DATA:=$( [[ -d /mnt/ssd1 ]] && echo /mnt/ssd1/colmap_panel/data || echo "$HOME/.colmap_panel" )}"
export COLMAP_PANEL_DATA; mkdir -p "$COLMAP_PANEL_DATA"
: "${TMPDIR:=$COLMAP_PANEL_DATA/tmp}"; export TMPDIR; mkdir -p "$TMPDIR"
# directory-picker root (defaults to the data disk if present, else filesystem root)
: "${COLMAP_PANEL_BROWSE_ROOT:=$( [[ -d /mnt/ssd1 ]] && echo /mnt/ssd1 || echo / )}"
export COLMAP_PANEL_BROWSE_ROOT

ENVPY="$CONDA_ROOT/envs/$CONDA_ENV/bin/python"
[[ -x "$ENVPY" ]] || { echo "conda env '$CONDA_ENV' python not found at: $ENVPY" >&2
                       echo "set CONDA_ROOT/CONDA_ENV in local.env" >&2; exit 1; }
export PYTHONNOUSERSITE=1   # keep ~/.local out of the env

echo "COLMAP Panel -> http://$HOST:$PORT  (env=$CONDA_ENV, ffmpeg=$FFMPEG_BIN, colmap=$COLMAP_BIN)"
exec "$ENVPY" -m uvicorn app:app --host "$HOST" --port "$PORT"
