#!/usr/bin/env bash
# Launch Recon Studio (https://github.com/KuoFengYuan/reconstudio) from a conda env.
#
#   ./run.sh              start the panel
#   ./run.sh --doctor     print the environment report to the terminal and exit
#                         (same checks as the /doctor page — no browser or port
#                          forward needed; exit 0 = ready, 1 = something missing)
#
# Per-machine config: `./setup.sh` writes a local.env with this machine's paths
# already filled in; anything in local.env (or the environment) overrides the
# auto-detected defaults below. See local.env.example for every knob.
set -euo pipefail
cd "$(dirname "$0")"

MODE=serve
[[ "${1:-}" == "--doctor" ]] && { MODE=doctor; shift; }

# Per-machine overrides (paths to colmap/ffmpeg, conda location, ports, …).
[[ -f local.env ]] && set -a && . ./local.env && set +a

# Shared machine detection (conda root / data disk / ffmpeg) — setup.sh uses the
# same functions, so a fresh install and every later startup agree on the paths.
. ./tools/detect.sh

: "${HOST:=127.0.0.1}"
: "${PORT:=8077}"
: "${CONDA_ENV:=rec}"
: "${CONDA_ROOT:=$(detect_conda_root)}"

# Roomy disk for job state + scratch: biggest local non-root mount, else $HOME.
# (No machine-specific path baked in here — override RECON_STUDIO_DATA in
# local.env if this machine wants a different one.)
DATA_DISK="$(detect_data_disk)"

# Binaries: PATH by default; override for non-standard installs.
: "${COLMAP_BIN:=colmap}"; export COLMAP_BIN
: "${FFMPEG_BIN:=$(detect_ffmpeg "$DATA_DISK")}"; export FFMPEG_BIN

: "${RECON_STUDIO_DATA:=$(default_data_dir "$DATA_DISK")}"; export RECON_STUDIO_DATA
: "${TMPDIR:=$(default_tmp_dir "$DATA_DISK")}"; export TMPDIR
: "${RECON_STUDIO_BROWSE_ROOT:=$(default_browse_root "$DATA_DISK")}"
export RECON_STUDIO_BROWSE_ROOT

export CONDA_ROOT   # so backends.py / doctor can locate sibling trainer envs (gs2m, …)
ENVPY="$CONDA_ROOT/envs/$CONDA_ENV/bin/python"
[[ -x "$ENVPY" ]] || { echo "conda env '$CONDA_ENV' python not found at: $ENVPY" >&2
                       echo "set CONDA_ROOT/CONDA_ENV in local.env" >&2; exit 1; }
export PYTHONNOUSERSITE=1   # keep ~/.local out of the env

# --doctor: report and exit. Deliberately placed BEFORE the mkdir and SuperSplat
# blocks below. A preflight check must only observe: it shouldn't create state on
# a machine you're merely inspecting, and it must not kick off a background build.
# Ordering also makes the report reachable in the case it exists to diagnose — a
# read-only or root-owned data disk would abort the mkdir under `set -e` and print
# a raw bash error instead of preflight's own diagnosis of that exact failure.
if [[ "$MODE" == "doctor" ]]; then
  exec "$ENVPY" -m pipeline.doctor_cli "$@"
fi

mkdir -p "$RECON_STUDIO_DATA" "$TMPDIR"

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
