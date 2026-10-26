#!/usr/bin/env bash
set -uo pipefail
P=/mnt/shared/ubuntu-sidecar-pod/superdec
cd "${P}"
export CUDA_VISIBLE_DEVICES=0
exec /mnt/shared/ubuntu-sidecar-pod/venvs/py-3.13/bin/python scripts/diag_c.py 2>&1 | grep -vE 'Warning|warn'
