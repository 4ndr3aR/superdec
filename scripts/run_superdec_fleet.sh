#!/usr/bin/env bash
# Launch the SuperDec fleet run (configs/train_fleet.yaml) on GPU1. 2026-10-24 v2.
# Usage: bash scripts/run_superdec_fleet.sh [--batch-size N]
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
STATUS="${STATUS_FILE:-${REPO}/logs/launch_status.log}"
echo "[run_superdec] $(date -u) LAUNCH bs=${BS} -> logs/superdec_fleet_small.log" >> "${STATUS}"
set +e
CUDA_VISIBLE_DEVICES=1 "${PY}" train/train.py --config-name train_fleet trainer.batch_size="${BS}"
rc=$?
set -e
# v2 fix: the rc line must NEVER be appended to the tqdm log (the bar's trailing
# \r leaves it mid-line, so '^\[run_superdec\].*exit rc=' can grep it in the
# training log). Status lives in its own line-oriented file the waiter polls.
printf '[run_superdec] exit rc=%s at %s\n' "${rc}" "$(date -u)" >> "${STATUS}"
exit "${rc}"
