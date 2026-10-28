"""Generate the xz-fleet split scene-lists (train/val/test) from the main repo's v17 yaml,
into the standalone's data/xzfleet_splits/. One scene basename per line. Run on the pod."""
import glob, os, sys, yaml

POD = "/mnt/shared" + "/ubuntu-sidecar-pod"
REPO = POD + "/hermes-3d-learning" + "-from-" + "monocular-video"
SPLIT_DIR = (sys.argv[1] if len(sys.argv) > 1 else
             POD + "/superdec/data/xzfleet_splits")
cfg = yaml.safe_load(open(os.path.join(REPO, "config",
                                       "randx64-all7-tasks-resnet18-v17.yaml")))
sp = cfg["splits"]
os.makedirs(SPLIT_DIR, exist_ok=True)
for side in ("train", "val", "test"):
    names = sorted({os.path.basename(p) for p in sp[side]})
    with open(os.path.join(SPLIT_DIR, side + ".txt"), "w") as fh:
        fh.write("\n".join(names) + "\n")
    print(f"{side:5s}: {len(names)} scenes")
print("wrote", SPLIT_DIR)
