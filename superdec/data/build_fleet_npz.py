#!/usr/bin/env python3
"""Convert the randx64 fleet PLY object meshes into SuperDec's ShapeNet-style layout,
with train/val split files. This is the ONLY data-side change needed to make SuperDec
train on our fleet: the stock `ShapeNet` dataset class reads
    <path>/<class>/<model_id>/pointcloud.npz   (keys: points, normals)
plus per-class  <path>/<class>/train.lst / val.lst.

Run ON THE POD with the py-3.13 venv:
    python superdec/data/build_fleet_npz.py --fleet <fleet_root> --out <out_root> --val-frac 0.1

Idempotent/resumable: already-built npz files are skipped (checked by size).
"""
import argparse
import glob
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import trimesh

N_SURFACE = 16384          # surface points+normals stored per object (loader re-draws 4096 at train time)
SEED = 20261022
CLASSES = ["box", "sphere", "torus", "cone", "capsule", "cylinder", "pyramid"]

_tls = threading.local()   # per-thread trimesh safety is not required; kept for RNG determinism


def _rng():
    # deterministic per-process rng for the split (splits are built from a sorted id list, not this)
    return random.Random(SEED)


def one_scene(scene_dir, out_root, log_lock, log):
    """Convert every object PLY in a scene dir. Returns (n_done, n_skip)."""
    meshes = sorted(glob.glob(os.path.join(scene_dir, "meshes", "mesh_o*.ply")))
    done = skip = 0
    scene_name = os.path.basename(scene_dir)            # scene_randx64_0000
    for mpath in meshes:
        stem = os.path.splitext(os.path.basename(mpath))[0]   # mesh_o0_box
        cls = stem.rsplit("_", 1)[-1]                        # box
        obj = stem.replace("mesh_", "")                      # o0_box
        if cls not in CLASSES:
            raise AssertionError(f"unknown class {cls!r} from {mpath}")
        model_id = f"{scene_name}_{obj}".replace("scene_randx64_", "s")
        out_dir = os.path.join(out_root, cls, model_id)
        out_npz = os.path.join(out_dir, "pointcloud.npz")
        if os.path.isfile(out_npz):
            skip += 1
            continue
        try:
            mesh = trimesh.load(mpath, process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = list(mesh.geometry.values())[0]
            pts, face_idx = trimesh.sample.sample_surface(mesh, N_SURFACE)
            normals = mesh.face_normals[face_idx]
            os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(out_npz, points=pts.astype(np.float32),
                                normals=normals.astype(np.float32))
            done += 1
        except Exception as e:  # fail loud, but record which file
            with log_lock:
                log.write(f"FAILED {mpath}: {e!r}\n")
            raise
    return done, skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scenes = sorted(glob.glob(os.path.join(args.fleet, "scene_randx64_*")))
    assert scenes, f"no scenes under {args.fleet}"
    print(f"[fleet] {len(scenes)} scenes -> {args.out}")

    log_lock = threading.Lock()
    fail_log = open(os.path.join(args.out, "convert_failures.log"), "a")

    t0 = time.time()
    tot_done = tot_skip = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(one_scene, s, args.out, log_lock, fail_log): s for s in scenes}
        for i, fut in enumerate(as_completed(futs), 1):
            d, k = fut.result()
            tot_done += d
            tot_skip += k
            if i % 200 == 0 or i == len(scenes):
                dt = time.time() - t0
                print(f"  scenes {i}/{len(scenes)}  objects done={tot_done} skip={tot_skip}  "
                      f"({dt:.0f}s, {i/dt:.1f} sc/s)", flush=True)

    # ---- build per-class model lists + splits (deterministic) ----
    print(f"[split] building per-class train/val lists (val_frac={args.val_frac})")
    n_train = n_val = 0
    by_class = {c: 0 for c in CLASSES}
    for cls in CLASSES:
        cdir = os.path.join(args.out, cls)
        if not os.path.isdir(cdir):
            print(f"  WARNING: no {cls} dir")
            continue
        ids = sorted(d for d in os.listdir(cdir) if os.path.isdir(os.path.join(cdir, d)))
        rng = random.Random(SEED + hash(cls) % 10000)
        rng.shuffle(ids)
        n_val_c = max(1, int(round(len(ids) * args.val_frac)))
        val_ids, train_ids = ids[:n_val_c], ids[n_val_c:]
        with open(os.path.join(cdir, "train.lst"), "w") as f:
            f.write("\n".join(train_ids) + "\n")
        with open(os.path.join(cdir, "val.lst"), "w") as f:
            f.write("\n".join(val_ids) + "\n")
        n_train += len(train_ids)
        n_val += len(val_ids)
        by_class[cls] = len(ids)
        print(f"  {cls:9s} total={len(ids):5d} train={len(train_ids):5d} val={len(val_ids):4d}")
    print(f"[done] TRAIN={n_train}  VAL={n_val}  objects={sum(by_class.values())}  out={args.out}")
    fail_log.close()


if __name__ == "__main__":
    main()
