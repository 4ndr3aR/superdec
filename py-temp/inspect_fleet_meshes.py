"""Inspect the randx64 fleet PLY meshes: verts/faces/watertight/bounds/area per class.

Run ON THE POD with the py-3.13 venv. No args needed.
"""
import glob
import sys

import trimesh

ROOT = "/mnt/shared/ubuntu-sidecar-pod/hermes-video-to-3d/output/randx64_fleet_64x64"
CLASSES = ["box", "sphere", "torus", "cone", "capsule", "cylinder", "pyramid"]

sampled = {}
for cls in CLASSES:
    f = sorted(glob.glob(f"{ROOT}/scene_*/meshes/mesh_o*_{cls}.ply"))
    assert f, f"MISSING class {cls}"
    sampled[cls] = f[0]
    m = trimesh.load(f[0], process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = list(m.geometry.values())[0]
    ext = m.extents
    print(
        f"{cls:9s} {f[0].split('/')[-1]:28s} v={len(m.vertices):5d} f={len(m.faces):5d} "
        f"watertight={m.is_watertight:5} ext={ext.round(3)} area={m.area:8.3f}"
    )

# a second random box (different scene) to check size/aspect variability
boxes = sorted(glob.glob(f"{ROOT}/scene_*/meshes/mesh_o*_box.ply"))
for f in [boxes[:1][0], boxes[len(boxes)//2], boxes[-1]]:
    m = trimesh.load(f, process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = list(m.geometry.values())[0]
    print(f"box-variety {f.split('/')[-1]:28s} ext={m.extents.round(3)} center={m.centroid.round(3)}")

# sample one object to see point/normals quality
f = sampled["sphere"]
m = trimesh.load(f, process=False)
if not isinstance(m, trimesh.Trimesh):
    m = list(m.geometry.values())[0]
pts, fi = trimesh.sample.sample_surface(m, 4096)
nrm = m.face_normals[fi]
print("\nsphere sample: pts", pts.shape, "range", pts.min(-1).round(3), pts.max(-1).round(3),
      "normals unit?", (abs((nrm**2).sum(-1) - 1) < 1e-5).all())

# objects-per-scene histogram over ALL scenes (counts mesh PLYs)
import os
from collections import Counter
c = Counter()
for scene_dir in sorted(glob.glob(f"{ROOT}/scene_randx64_*")):
    c[len(glob.glob(os.path.join(scene_dir, "meshes", "mesh_o*.ply")))] += 1
print("\nobjects-per-scene histogram:", dict(sorted(c.items())))
print("total objects:", sum(c * k for k in c))
sys.exit(0)
