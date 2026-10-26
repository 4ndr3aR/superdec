import os, sys
import numpy as np
import torch

sys.path.insert(0, "/mnt/shared/ubuntu-sidecar-pod/superdec")
from omegaconf import OmegaConf
from superdec.superdec import SuperDec
from superdec.data.dataloader import ShapeNet

SNAP = ("/mnt/shared/ubuntu-sidecar-pod/superdec/"
        "checkpoints/superdec_fleet_small_run-20261024-1219")
CK = 80
NSUB_GT, NSUB_SQ = 1024, 512
torch.set_grad_enabled(False)


def sq_surface(a, e, r, t, nt=20, np_=48, dev="cpu"):
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


dev = "cuda"
cfg = OmegaConf.load(os.path.join(SNAP, "config.yaml"))
ds = ShapeNet(split="val", cfg=cfg)
ld = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
ckpt = torch.load(os.path.join(SNAP, f"epoch_{CK}.pt"), map_location=dev, weights_only=False)
m = SuperDec(cfg.superdec).to(dev).eval()
m.load_state_dict(ckpt["model_state_dict"])

D = {}
for b in ld:
    pc = b["points"].to(dev).float()
    o = m(pc)
    for i in range(pc.shape[0]):
        c = b["model_id"][i].rsplit("_", 1)[-1]
        k = torch.where(o["exist"][i] > 0.5)[0]
        if len(k) == 0:
            continue
        s, e, r, t = o["scale"][i, k], o["shape"][i, k], o["rotate"][i, k], o["trans"][i, k]
        surf = torch.cat([sq_surface(s[j], e[j], r[j], t[j], dev=dev) for j in range(len(k))], 0)
        g = pc[i]
        gp = g[torch.randperm(g.shape[0])[:NSUB_GT]]
        sp = surf[torch.randperm(surf.shape[0])[:NSUB_SQ]]
        d1 = torch.cdist(gp, sp).min(1)[0]
        d2 = torch.cdist(sp, gp).min(1)[0]
        d = torch.cat([d1, d2])
        R = D.setdefault(c, dict(n=0, P=0.0, e1=0.0, e2=0.0, d1=0.0, d2=0.0,
                                 p95=0.0, s3=0.0, g3=0.0))
        R["n"] += 1
        R["P"] += len(k)
        R["e1"] += float(e[0, 0])
        R["e2"] += float(e[0, 1])
        R["d1"] += float(d1.mean())
        R["d2"] += float(d2.mean())
        R["p95"] += float(d.quantile(0.95))
        R["s3"] += float(s[0, :].max())
        R["g3"] += float(g.max(dim=0).values.max())

out = {}
for c, R in D.items():
    n = R["n"]
    out[c] = dict(n=n, P=R["P"] / n, e1=R["e1"] / n, e2=R["e2"] / n,
                  d1=R["d1"] / n, d2=R["d2"] / n, p95=R["p95"] / n,
                  s3=R["s3"] / n, g3=R["g3"] / n)
torch.save(out, "/mnt/shared/superdec-pngs/diag_e80.pt")
print("saved diag_e80.pt", flush=True)
