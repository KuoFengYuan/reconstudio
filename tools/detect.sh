#!/usr/bin/env bash
# Machine detection shared by run.sh (every startup) and setup.sh (first install),
# so the two can never disagree about where conda / the data disk / ffmpeg live.
#
# Meant to be SOURCED, not executed. Every function only echoes its answer and
# exports nothing, so callers stay in control of which vars get set.

# --- conda base: `conda info --base` if the CLI is on PATH, else usual installs --- #
detect_conda_root() {
  local d root
  root="$(conda info --base 2>/dev/null || true)"
  if [[ -z "$root" ]]; then
    for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
             "$HOME/mambaforge" /opt/conda; do
      [[ -d "$d" ]] && root="$d" && break
    done
  fi
  echo "$root"
}

# --- the roomiest local disk that isn't the root fs (job state + scratch) --- #
# Root is excluded deliberately: on these boxes it's the OS disk, and a COLMAP or
# training run that fills it takes the whole machine down rather than just the job.
# Network/virtual mounts are skipped (`--local` + -x) so a slow NFS can't win on
# size alone. Prints NOTHING when nothing qualifies — callers fall back to $HOME.
detect_data_disk() {
  local min_gb="${DATA_DISK_MIN_GB:-50}" avail mnt
  # avail first so the mount point can be the rest of the line (paths with spaces).
  df --output=avail,target --local -x tmpfs -x devtmpfs -x squashfs -x overlay 2>/dev/null \
    | awk -v min="$((min_gb * 1024 * 1024))" \
          'NR > 1 && $1 >= min && $2 != "/" { print $1, substr($0, index($0, $2)) }' \
    | sort -rn \
    | while read -r avail mnt; do
        if [[ -d "$mnt" && -w "$mnt" ]]; then echo "$mnt"; break; fi
      done
  # "no roomy disk here" is a normal answer (empty output), NOT an error: without
  # this the while loop's own exit status leaks out — 1 when the last mount it
  # looked at wasn't writable — and `set -e` in the caller would kill run.sh.
  return 0
}

# --- paths DERIVED from the data disk ------------------------------------- #
# These live here rather than in each caller because both run.sh (every startup)
# and setup.sh (writing local.env) need the same answer — deriving them twice is
# exactly the drift tools/detect.sh exists to prevent. $1 = data disk, may be "".
default_data_dir()    { [[ -n "${1:-}" ]] && echo "$1/recon_studio/data" || echo "$HOME/.recon_studio"; }
default_tmp_dir()     { echo "$(default_data_dir "${1:-}")/tmp"; }
default_browse_root() { echo "${1:-/}"; }   # picker starts at the data disk, else fs root

# --- ffmpeg: prefer an NVDEC build (GPU decode for 抽幀), else whatever's on PATH --- #
# $1 (optional): data disk, so a hand-built ffmpeg under <disk>/bin is found too.
detect_ffmpeg() {
  local disk="${1:-}" f
  for f in "$(command -v ffmpeg-nvdec || true)" \
           ${disk:+"$disk/bin/ffmpeg-nvdec"} \
           "$(command -v ffmpeg || true)"; do
    [[ -n "$f" && -x "$f" ]] && { echo "$f"; return; }
  done
  echo ffmpeg      # not found: let the pipeline report it (and /doctor flag it)
}
