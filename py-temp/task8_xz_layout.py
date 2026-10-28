"""Task8 standalone backport -- lock the .xz layout & object counts before writing ShapeNetXZ.
Reads ONLY via the main repo's load_array_archive (no new data). Runs on the pod, py-3.13 venv."""
import glob, os, sys
_POD = "/mnt/shared" + "/ubuntu-sidecar-pod"
sys.path.insert(0, _POD + "/hermes-3d-learning-from-monocular-video/src")
import numpy as np
from vt3dlearn.data.artifact_io import load_array_archive

FLEET = _POD + "/hermes-video-to-3d/output/randx64_fleet_64x64"
SPLITS = _POD + "/superdec/data/xzfleet_splits"

# 1) object count per split (n_obj in each scene's bundle_index.xz)
def scene_objs(scenename, side):
    p = os.path.join(FLEET, scenename, "vt3learn_cache", "bundle_index.xz")
    if not os.path.isfile(p):
        return None
    d = load_array_archive(p)
    n = int(np.asarray(d["n_obj"]))
    return n

for side in ("train", "val", "test"):
    names = [l.strip() for l in open(os.path.join(SPLITS, side + ".txt")) if l.strip()]
    tot = 0; miss = 0
    for nm in names:
        n = scene_objs(nm, side)
        if n is None:
            miss += 1; continue
        tot += n
    print(f"{side:5s}: {len(names)} scenes, {tot} objects, {miss} missing-bundle")

# 2) key layout: a K=3 scene (train[~80] is 3-obj) + a K=1 scene
def show(scene):
    p = os.path.join(FLEET, scene, "vt3learn_cache", "bundle_index.xz")
    d = load_array_archive(p)
    n = int(np.asarray(d["n_obj"]))
    keys = sorted(d.keys())
    print(f"\n== {scene}  n_obj={n}")
    print("   keys:", keys)
    for k in range(n):
        sk = f"obj_{k}_surface_pts"
        if sk in d:
            a = np.asarray(d[sk]).astype(np.float32)
            print(f"   {sk}: shape={a.shape} mean={a.mean(0).round(3).tolist()} "
                  f"absmax={np.abs(a).max().round(3):.3f}")
    # pose key(s)
    pose_keys = [k for k in keys if "pose" in k]
    for pk in pose_keys:
        print(f"   POSE {pk}: shape={np.asarray(d[pk]).shape}")
    sps = np.asarray(d["obj_0_surface_pts"]).astype(np.float32)
    print(f"   obj_0 surface: mean={sps.mean(0).round(3).tolist()} std={sps.std(0).round(3).tolist()}"
          f" absmax per-axis={np.abs(sps).max(0).round(3).tolist()}")

# find a 3-obj scene from train split
names = [l.strip() for l in open(os.path.join(SPLITS, "train.txt")) if l.strip()]
shown3 = False
for nm in names:
    n = scene_objs(nm, "train")
    if n == 3:
        show(nm); shown3 = True; break
# and the first train scene (usually 1 obj)
show(names[0])
print("\nDONE")
