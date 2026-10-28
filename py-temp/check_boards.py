#!/usr/bin/env python
# quick check of the r18v18-e0 boards: dims + nonblank fraction. Runs on the pod, py-3.13 venv.
import glob, os
import numpy as np
import cv2
base = "/mnt/shared/ubuntu-sidecar-pod/hermes-3d-learning-from-monocular-video/png/r18v18-e0-board"
for f in sorted(glob.glob(os.path.join(base, "*", "frame_*.png"))):
    im = cv2.imread(f)
    assert im is not None, f
    nz = float((im > 0).any(-1).mean())
    print(f, im.shape, "nonzero_frac=%.3f" % nz)
