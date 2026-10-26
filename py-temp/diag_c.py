import os, sys
import numpy as np
import torch

sys.path.insert(0, "/mnt/shared/ubuntu-sidecar-pod/superdec")
from omegaconf import OmegaConf
from superdec.superdec import SuperDec
from superdec.data.dataloader import ShapeNet

SNAP = ("/mnt/shared/ubuntu-sidecar-pod/superdec/"
        "checkpoints/superdec_fleet_small_run-20261024-1219")
torch.set_grad_enabled(False)
dev = "cuda"
cfg = OmegaConf.load(os.path.join(SNAP, "config.yaml"))
ds = ShapeNet(split="val", cfg=cfg)
ld = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
ckpt = torch.load(os.path.join(SNAP, "epoch_80.pt"), map_location=dev, weights_only=False)
m = SuperDec(cfg.superdec).to(dev).eval()
m.load_state_dict(ckpt["model_state_dict"])


def sq_surf_points(a, e, r, t, nt=64, np_=128):
    a1, a2, a3 = a
    e1, e2 = e

    def fexp(x, p):
        return torch.sign(x) * (torch.abs(x) ** p)

    th = torch.linspace(-np.pi / 2, np.pi / 2, nt, device=dev)
    ph = torch.linspace(-np.pi, np.pi, np_, device=dev)
    TH, PH = torch.meshgrid(th, ph, indexing="ij")
    x = a1 * fexp(torch.cos(TH), e1) * fexp(torch.cos(PH), e2)
    y = a2 * fexp(torch.cos(TH), e1) * fexp(torch.sin(PH), e2)
    z = a3 * fexp(torch.sin(TH), e1)
    return torch.stack([x, y, z], -1).reshape(-1, 3) @ r.T + t


WANT = ["box", "sphere", "capsule"]
have = set()
for b in ld:
    pc = b["points"].to(dev).float()
    o = m(pc)
    for i in range(pc.shape[0]):
        c = b["model_id"][i].rsplit("_", 1)[-1]
        if c not in WANT:
            continue
        g = pc[i].cpu().numpy()
        k = torch.where(o["exist"][i] > 0.5)[0]
        s = o["scale"][i, k[0]].cpu().numpy()
        e = o["shape"][i, k[0]].cpu().numpy()
        r = o["rotate"][i, k[0]].cpu().numpy()
        t = o["trans"][i, k[0]].cpu().numpy()
        rng = g.max(0) - g.min(0)
        ctr = (g.max(0) + g.min(0)) / 2
        rad = np.linalg.norm(g - ctr, axis=1)
        # face-center offset for box: GT points near +x / +y / +z face centers
        sp = sq_surf_points(torch.tensor(s, device=dev), torch.tensor(e, device=dev),
                            torch.tensor(r, device=dev), torch.tensor(t, device=dev))
        print(f"== {c}  n_prims={len(k)}")
        print(f"   GT bounds x/y/z: {np.round((g.min(0), g.max(0)), 3).tolist()}")
        print(f"   GT range (x,y,z): {np.round(rng, 3).tolist()}")
        print(f"   GT radius from center: min={rad.min():.3f} "
              f"mean={rad.mean():.3f} max={rad.max():.3f} "
              f"std={rad.std():.3f}")
        print(f"   PRED scale={np.round(s,3).tolist()} eps={np.round(e,3).tolist()}")
        have.add(c)
    if have == set(WANT):
        break
print("[done]", flush=True)
