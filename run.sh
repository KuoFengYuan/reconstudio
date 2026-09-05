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

# `HOST` and `PORT` are the two knob names this project shares with the rest of
# the world — conda's compiler activation exports HOST=x86_64-conda-linux-gnu, and
# plenty of tooling exports PORT. Inheriting either silently is how you get a panel
# that refuses to bind, or one listening on a port nobody was told about while `ss`
# happily shows "the port is up". So remember whether they arrived from the ambient
# environment BEFORE local.env is read, and only honour them if they look sane.
ENV_HOST="${HOST:-}"; ENV_PORT="${PORT:-}"
unset HOST PORT

# Per-machine overrides (paths to colmap/ffmpeg, conda location, ports, …).
[[ -f local.env ]] && set -a && . ./local.env && set +a

# Shared machine detection (conda root / data disk / ffmpeg) — setup.sh uses the
# same functions, so a fresh install and every later startup agree on the paths.
. ./tools/detect.sh

# Precedence: local.env (just sourced) > RECON_STUDIO_HOST/PORT > a bare HOST/PORT
# that actually looks like an address > the default. The documented one-off
# `HOST=0.0.0.0 ./run.sh` still works; `HOST=x86_64-conda-linux-gnu` no longer does.
if [[ -z "${HOST:-}" ]]; then
  HOST="${RECON_STUDIO_HOST:-}"
  if [[ -z "$HOST" && "$ENV_HOST" =~ ^([0-9.]+|::|localhost)$ ]]; then HOST="$ENV_HOST"
  elif [[ -z "$HOST" && -n "$ENV_HOST" ]]; then
    echo "忽略環境變數 HOST=$ENV_HOST(不是位址,多半是 conda 編譯器設的);改用 127.0.0.1" >&2
  fi
fi
if [[ -z "${PORT:-}" ]]; then
  PORT="${RECON_STUDIO_PORT:-}"
  if [[ -z "$PORT" && "$ENV_PORT" =~ ^[0-9]+$ ]]; then
    PORT="$ENV_PORT"
    echo "注意:PORT=$PORT 來自環境變數,不是 local.env。" >&2
  fi
fi
: "${HOST:=127.0.0.1}"
: "${PORT:=8077}"          # keep in step with setup.sh + local.env.example
# Exported under unambiguous names: /doctor must report the address we really
# bound, and reading a bare $HOST there would pick up conda's triplet instead.
export RECON_STUDIO_HOST="$HOST" RECON_STUDIO_PORT="$PORT"
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

# --- refuse to start on a port something else already holds ----------------- #
# uvicorn's own failure here is a Python traceback ending in "address already in
# use", which is easy to scroll past — and then the panel you reach on that port
# is the OLD process, serving old code, which is indistinguishable from "my
# change did nothing".
if busy="$(ss -ltnH "sport = :$PORT" 2>/dev/null)" && [[ -n "$busy" ]]; then
  echo "PORT=$PORT 已經被佔用了,沒有啟動:" >&2
  echo "$busy" | sed 's/^/    /' >&2
  echo "  舊的面板還開著就直接用它,或關掉它再跑;不然改 local.env 的 PORT。" >&2
  echo "  (前一個面板的 pid: $(pgrep -f "uvicorn app:app" | tr '\n' ' ' || true))" >&2
  exit 1
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

# --- what URL actually works, from where ------------------------------------ #
# Printing "http://$HOST:$PORT" was actively misleading: with HOST=0.0.0.0 it
# printed a URL nobody can open, and with the default 127.0.0.1 it said nothing
# about the fact that no one else on the network can reach it at all.
lan_ips() {
  ip -4 -o addr show scope global 2>/dev/null \
    | grep -vE ' (docker|br-|veth|virbr)' | awk '{print $4}' | cut -d/ -f1
}

echo "Recon Studio  (env=$CONDA_ENV, ffmpeg=$FFMPEG_BIN, colmap=$COLMAP_BIN)"
echo "  本機          http://127.0.0.1:$PORT"

NGINX_SITE=/etc/nginx/sites-enabled/reconstudio.conf
if [[ -r "$NGINX_SITE" ]]; then
  proxied="$(sed -nE 's|.*proxy_pass +http://127\.0\.0\.1:([0-9]+).*|\1|p' "$NGINX_SITE" | head -1)"
  hport="$(sed -nE 's|^ *listen +[0-9.]+:([0-9]+) +ssl.*|\1|p' "$NGINX_SITE" | head -1)"
  hname="$(sed -nE 's|^ *server_name +([^ ;]+).*|\1|p' "$NGINX_SITE" | head -1)"
  if [[ "$proxied" == "$PORT" ]]; then
    echo "  區網(nginx)   https://$hname:$hport/   ← 給同事的就是這個(要帳密)"
  else
    echo "  區網(nginx)   ✗ 代理指到 :$proxied,但面板綁的是 :$PORT —— 從外面一定連不到。" >&2
    echo "                 修:sudo scripts/deploy-nginx-lan.sh" >&2
  fi
elif [[ "$HOST" == "0.0.0.0" || "$HOST" == "::" ]]; then
  for ip in $(lan_ips); do
    echo "  區網          http://$ip:$PORT   ⚠ 無密碼、可瀏覽 $RECON_STUDIO_BROWSE_ROOT"
  done
  echo "                 要給別人用請改走 nginx:sudo scripts/deploy-nginx-lan.sh"
else
  echo "  區網          未開放(HOST=$HOST 只收本機)。要給同事連:"
  echo "                 sudo scripts/deploy-nginx-lan.sh   # 固定網址 + 帳密 + TLS"
fi

exec "$ENVPY" -m uvicorn app:app --host "$HOST" --port "$PORT"
