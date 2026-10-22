#!/usr/bin/env bash
# SuperDec fleet-training GPU1 waiter (2026-10-22): ollama holds GPU1 (~40GB, deallocs
# ~5 min after it idles). Poll for >= MIN_FREE_MB free on GPU1, then launch the fleet run
# (configs/train_fleet.yaml, bs32) with clean outcome detection (line-snapshot: new
# 'torch.OutOfMemoryError' since launch = OOM; other failure = code bug; never re-reads
# history). flock against double-arming.
set -euo pipefail
REPO="${VT3D_SUPERDEC_REPO:-/mnt/shared/ubuntu-sidecar-pod/superdec}"
PY="${VT3D_VENV_PY:-/mnt/shared/ubuntu-sidecar-pod/venvs/py-3.13/bin/python}"
cd "${REPO}"
GPU=1
MIN_FREE_MB=${VT3D_SD_MIN_FREE_MB:-36000}
LOG="${REPO}/logs/superdec_gpu1_waiter.log"
LOCK="${REPO}/logs/.superdec_waiter.lock"
mkdir -p "${REPO}/logs"
exec 9>"${LOCK}"
flock -n 9 || { echo "[sd-waiter] $(date) another superdec waiter already running; exit." >> "${LOG}"; exit 0; }
RUNLOG="${REPO}/logs/superdec_fleet_small.log"
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" | tr -d ' '; }
sd_running() { pgrep -f "train/train.py.*train_fleet" >/dev/null 2>&1; }

echo "[sd-waiter] $(date) watching GPU${GPU} for >= ${MIN_FREE_MB} MB free (ollama); then launch superdec_fleet_small -> ${RUNLOG}" >> "${LOG}"
if sd_running; then
  echo "[sd-waiter] $(date) superdec_fleet_small already running; exit." >> "${LOG}"
  exit 0
fi

while true; do
  F=$(free_mb)
  if [ "${F}" -ge "${MIN_FREE_MB}" ]; then
    echo "[sd-waiter] $(date) GPU${GPU} free=${F}MB >= ${MIN_FREE_MB}MB -> launching." >> "${LOG}"
    break
  fi
  echo "[sd-waiter] $(date) GPU${GPU} free=${F}MB (< ${MIN_FREE_MB}); sleep 60s." >> "${LOG}"
  sleep 60
done

N0=0; [ -f "${RUNLOG}" ] && N0=$(grep -c '^\[run_superdec\].*exit rc=' "${RUNLOG}" 2>/dev/null || echo 0)
O0=0; [ -f "${RUNLOG}" ] && O0=$(grep -c 'torch.OutOfMemoryError' "${RUNLOG}" 2>/dev/null || echo 0)
echo "[sd-waiter] $(date) launch (rc baseline=${N0}, OOM baseline=${O0})" >> "${LOG}"
nohup bash "${REPO}/scripts/run_superdec_fleet.sh" > "${RUNLOG}" 2>&1 &
T0=$(date +%s)
while true; do
  sleep 60
  N1=$(grep -c '^\[run_superdec\].*exit rc=' "${RUNLOG}" 2>/dev/null || echo 0)
  O1=$(grep -c 'torch.OutOfMemoryError' "${RUNLOG}" 2>/dev/null || echo 0)
  if [ "${N1}" -gt "${N0}" ]; then
    RC_L=$(grep '^\[run_superdec\].*exit rc=' "${RUNLOG}" | tail -1)
    echo "[sd-waiter] $(date) run finished: ${RC_L}" >> "${LOG}"
    if echo "${RC_L}" | grep -q 'exit rc=0'; then
      echo "[sd-waiter] $(date) superdec_fleet_small EXITED 0 -- done." >> "${LOG}"; exit 0
    fi
    if [ "${O1}" -gt "${O0}" ]; then
      echo "[sd-waiter] $(date) NEW torch.OutOfMemoryError (OOM lines ${O0}->${O1}); reducing batch size and retrying once." >> "${LOG}"
      N0=${N1}; T0=$(date +%s)
      nohup bash "${REPO}/scripts/run_superdec_fleet.sh" --batch-size 16 > "${RUNLOG}" 2>&1 &
    else
      echo "[sd-waiter] $(date) non-OOM failure. code bug; last 40 lines:" >> "${LOG}"
      tail -40 "${RUNLOG}" >> "${LOG}" 2>/dev/null
      exit 9
    fi
  fi
  if ! sd_running && [ $(( $(date +%s) - T0 )) -gt 1800 ]; then
    echo "[sd-waiter] $(date) dead-man: no rc line after 30min and not running; relaunching once." >> "${LOG}"
    N0=${N1}; O0=${O1}; T0=$(date +%s)
    nohup bash "${REPO}/scripts/run_superdec_fleet.sh" > "${RUNLOG}" 2>&1 &
  fi
done
