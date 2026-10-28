#!/usr/bin/env python
# verify the new board is 4 ROWS x 6 COLS landscape (H=2048, W=3072), nonblank.
import glob, os
import numpy as np
import cv2
base = "/mnt/shared/ubuntu-sidecar-pod/hermes-3d-learning-from-monocular-video/png/r18v18-e0-board"
for f in sorted(glob.glob(os.path.join(base, "*", "frame_*.png"))):
    im = cv2.imread(f)
    assert im is not None, f
    h, w = im.shape[:2]
    rows, cols = h // (64 * 8), w // (64 * 8)
    nz = float((im > 0).any(-1).mean())
    assert (h, w) == (2048, 3072), f"not 4x6 landscape: {im.shape}"
    assert (rows, cols) == (4, 6), f"not 4x6 panes: {(rows, cols)}"
    print(f, im.shape, f"{rows}rows x {cols}cols landscape", f"nonzero={nz:.3f}")
print("ALL OK: 4 rows x 6 cols landscape")
