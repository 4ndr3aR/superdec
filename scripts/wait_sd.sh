#!/usr/bin/env bash
# SuperDec fleet-training GPU1 waiter v2 (2026-10-24).
#
# v1 bug (caused 27 phantom dead-man relaunches overnight): the launcher appended
# its 'exit rc=' line to the tqdm training log, where the last progress-bar \r left
# it mid-line, so 'grep -c "^\[run_superdec\].*exit rc="' matched 0 forever. The
# dead-man branch then concluded "not running + no rc" and relaunched the already
# finished run, repeating every ~79 min for 30 h.
#
# v2: the launcher writes its status to a line-oriented file (logs/launch_status.log;
# it appends to STATUS_FILE itself, so nohup output goes nowhere). Dead-man fires
# only when launcher AND trainer are both gone, and is capped at MAX_RELAUNCHES.
set -euo pipefail
REPO="${VT3D_SUPERDEC_REPO:-/mnt/shared/ubuntu-sidecar-pod/superdec}"
PY="${VT3D_VENV_PY:-/mnt/shared/ubuntu-sidecar-pod/venvs/py-3.13/bin/python}"
cd "${REPO}"
GPU=1
MIN_FREE_MB=${VT3D_SD_MIN_FREE_MB:-36000}
MAX_RELAUNCHES=${VT3D_SD_MAX_RELAUNCHES:-3}
LOG="${REPO}/logs/superdec_gpu1_waiter.log"
LOCK="${REPO}/logs/.superdec_waiter.lock"
STATUS="${REPO}/logs/launch_status.log"
RUNLOG="${REPO}/logs/superdec_fleet_small.log"
mkdir -p "${REPO}/logs"
exec 9>"${LOCK}"
flock -n 9 || { echo "[sd-waiter-v2] $(date -u) another waiter already running; exit." >> "${LOG}"; exit 0; }

sd_running()    { pgrep -f "train/train.py.*train_fleet" >/dev/null 2>&1; }
lch_running()   { pgrep -f "run_superdec_fleet.sh" >/dev/null 2>&1; }
free_mb()       { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" | tr -d ' '; }
launch_count()  { [ -f "${STATUS}" ] && { grep -c 'LAUNCH' "${STATUS}" || echo 0; } || echo 0; }
rc_lines()      { [ -f "${STATUS}" ] && { grep -c 'exit rc=' "${STATUS}" || echo 0; } || echo 0; }

echo "[sd-waiter-v2] $(date -u) watching GPU${GPU} for >= ${MIN_FREE_MB} MB free; relaunch cap ${MAX_RELAUNCHES}" >> "${LOG}"
if sd_running; then
  echo "[sd-waiter-v2] $(date -u) superdec already running; exit." >> "${LOG}"; exit 0
fi

# wait for ollama to free the GPU
while true; do
  F=$(free_mb)
  if [ "${F}" -ge "${MIN_FREE_MB}" ]; then
    echo "[sd-waiter-v2] $(date -u) GPU${GPU} free=${F}MB >= ${MIN_FREE_MB}MB -> launching (launches_so_far=$(launch_count))" >> "${LOG}"
    break
  fi
  echo "[sd-waiter-v2] $(date -u) GPU${GPU} free=${F}MB (< ${MIN_FREE_MB}); sleep 60s." >> "${LOG}"
  sleep 60
done

do_launch() {
  STATUS_FILE="${STATUS}" nohup bash "${REPO}/scripts/run_superdec_fleet.sh" "$@" \
      >/dev/null 2>&1 &
}

L0=$(launch_count)
S0=$(rc_lines)
T0=$(date +%s)
do_launch

CAP=$((L0 + MAX_RELAUNCHES))
while true; do
  sleep 60
  S1=$(rc_lines)
  if [ "${S1}" -gt "${S0}" ]; then
    RC_L=$(grep 'exit rc=' "${STATUS}" | tail -1)
    echo "[sd-waiter-v2] $(date -u) run finished: ${RC_L}" >> "${LOG}"
    case "${RC_L}" in
      *'exit rc=0'*)
        echo "[sd-waiter-v2] $(date -u) superdec EXITED 0 -- done." >> "${LOG}"; exit 0;;
      *)
        if tail -c 200000 "${RUNLOG}" 2>/dev/null | tr '\r' '\n' | grep -qa 'torch.OutOfMemoryError'; then
          if [ "$(launch_count)" -ge "${CAP}" ]; then
            echo "[sd-waiter-v2] $(date -u) OOM but relaunch cap ${MAX_RELAUNCHES} reached; STOPPING." >> "${LOG}"; exit 3
          fi
          echo "[sd-waiter-v2] $(date -u) NEW torch.OutOfMemoryError -> retry once at bs16." >> "${LOG}"
          S0=${S1}; T0=$(date +%s)
          do_launch --batch-size 16
        else
          echo "[sd-waiter-v2] $(date -u) non-zero exit, no OOM in recent log -> code bug. tail of training log:" >> "${LOG}"
          { [ -f "${RUNLOG}" ] && tail -c 4000 "${RUNLOG}" | tr '\r' '\n' | tail -30; } >> "${LOG}"
          exit 9
        fi
        ;;
    esac
  fi
  # dead-man: launcher AND trainer both gone for >20 min without an rc line.
  if ! lch_running && ! sd_running && [ $(( $(date +%s) - T0 )) -gt 1200 ]; then
    if [ "$(launch_count)" -ge "${CAP}" ]; then
      echo "[sd-waiter-v2] $(date -u) dead-man but relaunch cap ${MAX_RELAUNCHES} reached; STOPPING loudly." >> "${LOG}"; exit 4
    fi
    echo "[sd-waiter-v2] $(date -u) dead-man: launcher+trainer gone >20min, no rc line; relaunching (launches=$(launch_count)/${CAP})." >> "${LOG}"
    S0=${S1}; T0=$(date +%s)
    do_launch
  fi
done
