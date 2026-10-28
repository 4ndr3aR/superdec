"""De-risk: kNN-PCA per-point normals from a single .xz object surface cloud. Validates the
normal-source for the standalone xz backport (cuboid loss needs them). Timed."""
import sys, time
_POD = "/mnt/shared" + "/ubuntu-sidecar-pod"
sys.path.insert(0, _POD + "/hermes-3d-learning-from-monocular-video/src")
import numpy as np
from vt3dlearn.data.artifact_io import load_array_archive

def knn_pca_normals(p, k=16, w=0.3):
    from scipy.spatial import cKDTree
    p = np.ascontiguousarray(p, dtype=np.float64)
    tr = cKDTree(p)
    idx = tr.query(p, k=k + 1, workers=-1)[1]
    nb = p[idx]                                  # (N, k+1, 3)
    rel = nb[:, 1:] - p[:, None, :]              # (N, k, 3) exclude self
    cov = np.einsum("nki,nkj->nij", rel, rel)      # (N,3,3) per-point neighbor covariance
    ev, evv = np.linalg.eigh(cov)
    n = evv[:, :, 0]                             # smallest eigvec = surface normal
    d = n * p
    d = d.sum(-1, keepdims=True)
    d[d == 0] = 1.0
    n = n * (1.0 / d)
    return n.astype(np.float32)

S = _POD + "/hermes-video-to-3d/output/randx64_fleet_64x64"
# scene 0700 obj0 (cyl-ish), obj2 (pyr, absmax 0.322), and scene 0000 obj0 (box)
for scen, k in (("scene_randx64_0700", 0), ("scene_randx64_0700", 2), ("scene_randx64_0000", 0)):
    d = load_array_archive(S + "/" + scen + "/vt3learn_cache/bundle_index.xz")
    p = np.asarray(d[f"obj_{k}_surface_pts"]).astype(np.float64)
    t0 = time.time()
    n = knn_pca_normals(p, k=16)
    dt = time.time() - t0
    print(f"{scen} obj{k}: shape={p.shape} |n| mean={np.linalg.norm(n,axis=1).mean():.4f} "
          f"min={np.linalg.norm(n,axis=1).min():.4f}  ({dt*1e3:.0f} ms, {k+1}th try)")
    # sanity: mean normal over the cloud should be ~0 for a symmetric shape
    print(f"   mean normal = {n.mean(0).round(3).tolist()}")
print("DONE normal-est")
