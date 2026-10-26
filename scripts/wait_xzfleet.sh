#!/usr/bin/env bash
# SuperDec xzfleet waiter v2 (2026-10-26) -- same dead-man pattern as wait_sd.sh (GPU0/npz).
# Fires train when the xz launcher's status line says LAUNCH with no matching 'exit rc=', and
# relaunches (capped) only on a torch.OutOfMemoryError retry at bs16 or a dead-man window.
set -uo pipefail
REPO="${VT3D_SUPERDEC_REPO:-/mnt/shared/ubuntu-sidecar-pod/superdec}"
cd "${REPO}"
GPU="${VT3D_SD_GPU:-0}"
MIN_FREE_MB="${VT3D_SD_MIN_FREE_MB:-30000}"
MAX_RELAUNCHES="${VT3D_SD_MAX_RELAUNCHES:-3}"
LOG="${REPO}/logs/superdec_xzfleet_waiter.log"
LOCK="${REPO}/logs/.superdec_xzfleet_waiter.lock"
STATUS="${REPO}/logs/launch_status_xz.log"
RUNLOG="${REPO}/logs/superdec_fleet_xz.log"
mkdir -p "${REPO}/logs"
exec 9>"${LOCK}"
flock -n 9 || { echo "[xz-waiter-v2] $(date -u) another already running; exit." >> "${LOG}"; exit 0; }

sd_running()  { pgrep -f "train/train.py.*train_fleet_xz" >/dev/null 2>&1; }
lch_running() { pgrep -f "run_superdec_xzfleet.sh" >/dev/null 2>&1; }
free_mb()     { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" | tr -d ' '; }
launch_count() { [ -f "${STATUS}" ] && { grep -c 'LAUNCH' "${STATUS}" || true; } || echo 0; }
rc_lines()     { [ -f "${STATUS}" ] && { grep -c 'exit rc=' "${STATUS}" || true; } || echo 0; }

echo "[xz-waiter-v2] $(date -u) watching GPU${GPU} for >= ${MIN_FREE_MB} MB free; relaunch cap ${MAX_RELAUNCHES}" >> "${LOG}"
if sd_running; then
  echo "[xz-waiter-v2] $(date -u) xz run already running; exit." >> "${LOG}"; exit 0
fi

# Wait until the GPU has enough free VRAM for an xz step (bs32 ~ 2GB + env slack; 30GB gate is generous
# because ollama sits on GPU1, not GPU0, so this gate only catches a stray second process).
while true; do
  F=$(free_mb)
  if [ "${F}" -ge "${MIN_FREE_MB}" ]; then
    echo "[xz-waiter-v2] $(date -u) GPU${GPU} free=${F}MB >= ${MIN_FREE_MB}MB -> launching (so_far=$(launch_count))" >> "${LOG}"
    break
  fi
  echo "[xz-waiter-v2] $(date -u) GPU${GPU} free=${F}MB (< ${MIN_FREE_MB}); sleep 60s." >> "${LOG}"
  sleep 60
done

do_launch() {
  STATUS_FILE="${STATUS}" nohup bash "${REPO}/scripts/run_superdec_xzfleet.sh" "$@" > /dev/null 2>&1 &
}

L0=$(launch_count); S0=$(rc_lines); T0=$(date +%s)
do_launch

CAP=$((L0 + MAX_RELAUNCHES))
while true; do
  sleep 60
  S1=$(rc_lines)
  if [ "${S1}" -gt "${S0}" ]; then
    RC_L=$(grep 'exit rc=' "${STATUS}" | tail -1)
    echo "[xz-waiter-v2] $(date -u) run finished: ${RC_L}" >> "${LOG}"
    case "${RC_L}" in
      *'exit rc=0'*)
        echo "[xz-waiter-v2] $(date -u) xz run EXITED 0 -- done." >> "${LOG}"; exit 0;;
      *)
        if tail -c 200000 "${RUNLOG}" 2>/dev/null | tr '\r' '\n' | grep -qa 'torch.OutOfMemoryError'; then
          if [ "$(launch_count)" -ge "${CAP}" ]; then
            echo "[xz-waiter-v2] $(date -u) OOM but relaunch cap ${MAX_RELAUNCHES} reached; STOPPING." >> "${LOG}"; exit 3
          fi
          echo "[xz-waiter-v2] $(date -u) NEW torch.OutOfMemoryError -> retry once at bs16." >> "${LOG}"
          S0=${S1}; T0=$(date +%s)
          do_launch --batch-size 16
        else
          echo "[xz-waiter-v2] $(date -u) non-zero exit, no OOM in recent log -> code bug. tail of training log:" >> "${LOG}"
          { [ -f "${RUNLOG}" ] && tail -c 4000 "${RUNLOG}" | tr '\r' '\n' | tail -30; } >> "${LOG}"
          exit 9
        fi
        ;;
    esac
  fi
  # dead-man: launcher AND trainer both gone >20 min without a new rc line.
  if ! lch_running && ! sd_running && [ $(( $(date +%s) - T0 )) -gt 1200 ]; then
    if [ "$(launch_count)" -ge "${CAP}" ]; then
      echo "[xz-waiter-v2] $(date -u) dead-man but relaunch cap ${MAX_RELAUNCHES} reached; STOPPING." >> "${LOG}"; exit 4
    fi
    echo "[xz-waiter-v2] $(date -u) dead-man: launcher+trainer gone >20min, no rc; relaunching (launches=$(launch_count)/${CAP})." >> "${LOG}"
    S0=${S1}; T0=$(date +%s)
    do_launch
  fi
done
