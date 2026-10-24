#!/usr/bin/env bash
# Launch the fast SuperDec fleet eval on GPU0. No nested-quoting: called as
#   screen -dmS sdeval bash <this-script>
set -uo pipefail
P=/mnt/shared/ubuntu-sidecar-pod/superdec
V=/mnt/shared/ubuntu-sidecar-pod/venvs/py-3.13/bin/python
cd "$P"
export CUDA_VISIBLE_DEVICES=0
exec env CUDA_VISIBLE_DEVICES=0 "$V" scripts/eval_fast.py 2>&1 | tee logs/eval_fast.log
