"""Nail the sq_surface_points broadcast: call the REAL outdict_to_cam on the live emit (window val row 0),
and print every sq_surface_points operand shape. CPU-GPU, single scene."""
import os, sys, json
import torch
REPO = "/mnt/shared/ubuntu-sidecar-pod/hermes-3d-learning-from-monocular-video"
sys.path.insert(0, os.path.join(REPO, "src")); os.chdir(REPO); os.environ["PYTHONPATH"] = "src"
RUN = "output/randx64-all7-20261027-r18v18-convlstm"
CKPT = "checkpoints/20261027-082903-e00-best-val-loss.pt"
from vt3dlearn.tools.load_model import load_run
from vt3dlearn.data.dataset import VideoDataset
res = json.load(open(os.path.join(RUN, "resolved_config.json")))
bl_p = os.path.join(RUN, "board_3d_bundles.txt")
with open(bl_p, "w") as fh:
    for p in res["bundles_from_env"]:
        fh.write(p + "\n")
os.environ.pop("VT3D_BUNDLES", None); os.environ["VT3D_BUNDLES_FILE"] = bl_p
dev = "cuda" if torch.cuda.is_available() else "cpu"
model_b = load_run(RUN, ckpt_rel_s=CKPT, dev_v=dev, strict_b=True)
cfg_v = dict(model_b["cfg"])
ds = VideoDataset(cfg_v, "val")
smp = dict(ds[0])
print("n_obj_real:", smp["n_obj_real"].shape, smp["n_obj_real"], "| surf", smp["surf_pts_canon"].shape,
      "| poses", smp["obj_poses"].shape, "| Twc", smp["T_wc_cam"].shape, flush=True)

sd = __import__("vt3dlearn.tasks", fromlist=["load_task"]).load_task("superdec")
with torch.no_grad():
    emit = sd.run({}, dict(smp))
print("emit.sq scale", tuple(emit["sq"]["scale"].shape), "shape", tuple(emit["sq"]["shape"].shape),
      "rotate", tuple(emit["sq"]["rotate"].shape), "trans", tuple(emit["sq"]["trans"].shape),
      "exist", tuple(emit["sq"]["exist"].shape), flush=True)
print("emit _pc", tuple(emit["_pc"].shape), "_c0", tuple(emit["_c0"].shape), "_s0", tuple(emit["_s0"].shape),
      "n_obs", len(emit["_objs"]), emit["_objs"], flush=True)

od_dn = sd.denormalize_outdict(emit["sq"], emit["_c0"].to(dev), emit["_s0"].to(dev))
sc, sh, ro, tr = od_dn["scale"], od_dn["shape"], od_dn["rotate"], od_dn["trans"]
print("dn scale", tuple(sc.shape), "shape", tuple(sh.shape), "rotate", tuple(ro.shape), "trans", tuple(tr.shape), flush=True)
# operand shapes for fexp
a1 = sc[..., 0:1]; e1 = sh[..., 0:1]; e2 = sh[..., 1:2]
n_theta, n_phi = 20, 48
th = torch.linspace(-1.5708, 1.5708, n_theta); ph = torch.linspace(0, 6.2832, n_phi)
TH, PH = torch.meshgrid(th, ph, indexing="ij")
TH = TH.view(1, 1, *TH.shape).expand(sc.shape[0], sc.shape[1], *TH.shape)
print("TH", tuple(TH.shape), "| a1", tuple(a1.shape), "e1", tuple(e1.shape), "e2", tuple(e2.shape), flush=True)
try:
    sq = sd.sq_surface_points(sc, sh, ro, tr, n_theta=20, n_phi=48, dev=dev)
    print("sq OK", tuple(sq.shape), flush=True)
except Exception as e:
    print("SQ FAIL:", type(e).__name__, e, flush=True)
# and the FULL outdict_to_cam
try:
    poses = smp["obj_poses"]; Twc = smp["T_wc_cam"]
    print("calls outdict_to_cam with poses", tuple(poses.shape), "Twc", tuple(Twc.shape), flush=True)
    out = sd.outdict_to_cam(emit, poses, Twc, n_theta=20, n_phi=48)
    print("outdict_to_cam OK ->", len(out), "clouds; first", tuple(out[0][0].shape) if out else None, flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print("OUTDICT_TO_CAM FAIL:", type(e).__name__, e, flush=True)
