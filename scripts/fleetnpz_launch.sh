#!/usr/bin/env bash
# Launch (or resume) the fleet PLY->NPZ converter. Run ON THE POD:  bash fleetnpz_launch.sh
set -euo pipefail
REPO="${VT3D_SUPERDEC_REPO:-/mnt/shared/ubuntu-sidecar-pod/superdec}"
PY="${VT3D_VENV_PY:-/mnt/shared/ubuntu-sidecar-pod/venvs/py-3.13/bin/python}"
FLEET="${VT3D_FLEET_ROOT:-/mnt/shared/ubuntu-sidecar-pod/hermes-video-to-3d/output/randx64_fleet_64x64}"
OUT="${REPO}/data/fleet"
cd "$REPO"
exec "$PY" superdec/data/build_fleet_npz.py --fleet "$FLEET" --out "$OUT" --val-frac 0.1 --threads 12
