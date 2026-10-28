"""Pin the exact rank where the render board's path through tasks/superdec.py diverges (20 vs 4 at dim 2
in sq_surface_points.fexp). Reproduces the CLI board chain on ONE val window with a real checkpoint. CPU-GPU."""
import os, sys, json
import torch

REPO = "/mnt/shared/ubuntu-sidecar-pod/hermes-3d-learning-from-monocular-video"
sys.path.insert(0, os.path.join(REPO, "src"))
os.chdir(REPO)
os.environ["PYTHONPATH"] = "src"

RUN = "output/randx64-all7-20261027-r18v18-convlstm"
CKPT = "checkpoints/20261027-082903-e00-best-val-loss.pt"

from vt3dlearn.tools.load_model import load_run
from vt3dlearn.data.dataset import VideoDataset
from vt3dlearn.train.cli import heads_modules_l as _hml_fn

res = json.load(open(os.path.join(RUN, "resolved_config.json")))
bl_p = os.path.join(RUN, "board_3d_bundles.txt")
with open(bl_p, "w") as fh:
    for p in res["bundles_from_env"]:
        fh.write(p + "\n")
os.environ.pop("VT3D_BUNDLES", None)
os.environ["VT3D_BUNDLES_FILE"] = bl_p

dev = "cuda" if torch.cuda.is_available() else "cpu"
model_b = load_run(RUN, ckpt_rel_s=CKPT, dev_v=dev, strict_b=True)
cfg_v = dict(model_b["cfg"])
ds = VideoDataset(cfg_v, "val")
mode_s = str(getattr(ds, "mode_s_", "frame"))
print("mode:", mode_s, "n_rows:", len(ds), flush=True)

sd = __import__("vt3dlearn.tasks", fromlist=["load_task"]).load_task("superdec")
smp = dict(ds[0])
print("row0 keys: surf", smp["surf_pts_canon"].shape, "pose", smp["obj_poses"].shape, "T_wc", smp["T_wc_cam"].shape, "n_obj", int(smp["n_obj_real"].view(-1)[0]) if smp["n_obj_real"].dim()>=1 else -1, flush=True)

# what _extract_superdec actually does:
from vt3dlearn.tools.render_3d_board import _extract_superdec
try:
    out = _extract_superdec(model_b, smp)
    print("extract OK:", None if out is None else len(out), flush=True)
except Exception as e:
    print("extract FAILED:", type(e).__name__, e, flush=True)
    # step-by-step
    surf = smp["surf_pts_canon"].to(dev).float()
    print("surf dim", surf.dim(), surf.shape)
    if int(surf.dim()) == 3:
        surf = surf.unsqueeze(0)
    n_obj = smp["n_obj_real"].to(dev)
    if int(n_obj.dim()) == 0:
        n_obj = n_obj.view(1)
    pts, objs = sd._object_rows(surf, n_obj, int(sd._HB["max_pts"]))
    print("pts", pts.shape, "n_objs", len(objs))
    pc_norm, c0, s0 = sd.normalize_cloud(pts)
    with torch.enable_grad():
        od = sd._HB["head"](pc_norm)
    print("od scale", od["scale"].shape, "rotate", od["rotate"].shape, "trans", od["trans"].shape, "exist", od["exist"].shape, "assign", od["assign_matrix"].shape, flush=True)
    od_dn = sd.denormalize_outdict(od, c0.to(dev), s0.to(dev))
    print("od_dn scale", od_dn["scale"].shape, "shape", od_dn["shape"].shape, "rotate", od_dn["rotate"].shape, "trans", od_dn["trans"].shape, flush=True)
    try:
        sq = sd.sq_surface_points(od_dn["scale"], od_dn["shape"], od_dn["rotate"], od_dn["trans"])
        print("sq OK", sq.shape, flush=True)
    except Exception as e2:
        print("sq_surface_points FAILED:", type(e2).__name__, e2, flush=True)
