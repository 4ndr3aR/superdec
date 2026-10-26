#!/usr/bin/env bash
# Launch the SuperDec xzfleet run (configs/train_fleet_xz.yaml) on GPU0. 2026-10-26.
# Usage: bash scripts/run_superdec_xzfleet.sh [--batch-size N]
set -uo pipefail
REPO="${VT3D_SUPERDEC_REPO:-/mnt/shared/ubuntu-sidecar-pod/superdec}"
PY="${VT3D_VENV_PY:-/mnt/shared/ubuntu-sidecar-pod/venvs/py-3.13/bin/python}"
cd "${REPO}"
BS=32
while [ $# -gt 0 ]; do
  case "$1" in
    --batch-size) BS="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
mkdir -p logs
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF}}"
# train/*.py import the `superdec` package; make the repo root importable.
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
STATUS="${STATUS_FILE:-${REPO}/logs/launch_status_xz.log}"
RUNLOG="${RUNLOG_FILE:-${REPO}/logs/superdec_fleet_xz.log}"
echo "[run_xz] $(date -u) LAUNCH bs=${BS} -> ${RUNLOG}" >> "${STATUS}"
set +e
CUDA_VISIBLE_DEVICES="${VT3D_SD_GPU:-0}" "${PY}" train/train.py --config-name train_fleet_xz trainer.batch_size="${BS}" > "${RUNLOG}" 2>&1
rc=$?
set -e
printf '[run_xz] exit rc=%s at %s\n' "${rc}" "$(date -u)" >> "${STATUS}"
exit "${rc}"
