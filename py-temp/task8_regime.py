"""Compare the NPZ regime (what the existing fleet checkpoint trained on) vs the .xz surface cloud,
to decide ShapeNetXZ's normalize flag. Read-only."""
import sys, glob, os
import numpy as np
_POD = "/mnt/shared" + "/ubuntu-sidecar-pod"
sys.path.insert(0, _POD + "/hermes-3d-learning-from-monocular-video/src")
from vt3dlearn.data.artifact_io import load_array_archive

def npz_regime():
    npzs = sorted(glob.glob(_POD + "/superdec/data/fleet/*/s*0000_*/pointcloud*.npz"))[:6]
    print("=== NPZ (existing-fleet checkpoint regime) ===")
    for p in npzs:
        d = np.load(p)
        pts = d["points"].astype(np.float64)
        print(f"  {os.path.basename(os.path.dirname(p))[:24]:24s} n={pts.shape[0]} "
              f"mean={pts.mean(0).round(3).tolist()} absmax={np.abs(pts).max(0).round(3).tolist()} "
              f"overall_absmax={np.abs(pts).max():.4f}")

def xz_regime():
    print("=== .xz surface cloud (bundle_index) ===")
    for scen in ("scene_randx64_0000", "scene_randx64_0700"):
        d = load_array_archive(_POD + f"/hermes-video-to-3d/output/randx64_fleet_64x64/{scen}/vt3learn_cache/bundle_index.xz")
        for k in range(int(np.asarray(d["n_obj"]))):
            p = np.asarray(d[f"obj_{k}_surface_pts"]).astype(np.float64)
            print(f"  {scen} obj{k}: n={p.shape[0]} mean={p.mean(0).round(3).tolist()} "
                  f"absmax={np.abs(p).max(0).round(3).tolist()} overall_absmax={np.abs(p).max():.4f}")

npz_regime()
xz_regime()
print("DONE")
