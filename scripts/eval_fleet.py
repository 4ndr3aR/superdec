#!/usr/bin/env python3
"""Evaluate the trained SuperDec fleet run: per-class Chamfer (normalized space) +
a 7-class reconstruction comparison PNG (GT surface vs predicted superquadric surface).
Run ON THE POD after training:
    python scripts/eval_fleet.py --ckpt <ckpt_dir/epoch_XX.pt> [png out path]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf

from superdec.superdec import SuperDec
from superdec.data.dataloader import ShapeNet

CLASSES = ["box", "sphere", "torus", "cone", "capsule", "cylinder", "pyramid"]


def sq_surface_points(scale, eps, rot, trans, n_theta=40, n_phi=80):
    """Sample the predicted superquadric(s); inputs in NORMALIZED object space (B,1,...)."""
    a1, a2, a3 = scale[0, 0].view(3)
    e1, e2 = eps[0, 0].view(2)

    def fexp(x, p):
        return torch.sign(x) * (torch.abs(x) ** p)

    th = torch.linspace(-np.pi / 2, np.pi / 2, n_theta)
    ph = torch.linspace(-np.pi, np.pi, n_phi)
    TH, PH = torch.meshgrid(th, ph, indexing="ij")
    x = a1 * fexp(torch.cos(TH), e1) * fexp(torch.cos(PH), e2)
    y = a2 * fexp(torch.cos(TH), e1) * fexp(torch.sin(PH), e2)
    z = a3 * fexp(torch.sin(TH), e1)
    pts = torch.stack([x, y, z], -1).reshape(-1, 3)
    pts = pts @ rot[0, 0].T + trans[0, 0]
    return pts


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="output png (default: next to script repo /output/)")
    ap.add_argument("--max-objects-per-class", type=int, default=119)
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model = SuperDec(cfg.superdec).cuda().eval()
    model.load_state_dict(ck["model_state_dict"])
    print(f"[eval] loaded {args.ckpt} (epoch {ck.get('epoch')})")

    ds = ShapeNet(split="val", cfg=cfg)
    ld = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, drop_last=False)

    by_class = {c: {"cd": [], "n": 0} for c in CLASSES}
    viz = {}
    for batch in ld:          # full val set: 833 objects (~26 batches) -- cheap at eval speed
        pc, normals = batch["points"].cuda().float(), batch["normals"].cuda().float()
        out = model(pc)
        b = pc.shape[0]
        for i in range(b):
            mid = batch["model_id"][i]           # str: e.g. "s0000_o0_box" (the ShapeNet ds returns model['model_id'])
            cls = mid.rsplit("_", 1)[-1]
            assert cls in CLASSES, f"unparseable model_id {mid!r}"
            scale, eps = out["scale"][i:i + 1], out["shape"][i:i + 1]
            rot, trans, exist = out["rotate"][i:i + 1], out["trans"][i:i + 1], out["exist"][i:i + 1].squeeze(-1)
            keep = torch.where(exist > 0.5)[0]
            if len(keep) == 0:
                continue
            s = scale[:, keep]; e = eps[:, keep]; r = rot[:, keep]; t = trans[:, keep]
            surf = [sq_surface_points(si, ei, ri, ti).cpu().numpy() for si, ei, ri, ti in
                    zip(s, e, r, t)]
            surf = np.concatenate(surf, 0)
            d1 = np.min(((pc[i].cpu().numpy()[:, None, :] - surf[None, :, :]) ** 2).sum(-1), 1)
            d2 = np.min(((surf[:, None, :] - pc[i].cpu().numpy()[None, :, :]) ** 2).sum(-1), 1)
            cd = float(np.sqrt(d1.mean() + d2.mean()))
            by_class[cls]["cd"].append(cd)
            by_class[cls]["n"] += 1
            if cls not in viz:
                viz[cls] = (pc[i].cpu().numpy(), surf, s, exist.cpu().numpy())

    print(f"[eval] per-class Chamfer (normalized obj space, n = #val objects):")
    for c in CLASSES:
        v = by_class[c]
        print(f"  {c:9s} cd={np.mean(v['cd']):.4f}  (n={v['n']})")

    # ---- 7-class comparison PNG ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={"projection": "3d"})
    axes = axes.reshape(-1)
    for k, c in enumerate(CLASSES):
        ax = axes[k]
        if c not in viz:
            ax.set_title(f"{c} (no prediction)")
            continue
        gt, surf, s, exist = viz[c]
        ax.scatter(gt[:, 0], gt[:, 2], gt[:, 1], s=0.4, c="#888888", label="GT")
        ax.scatter(surf[:, 0], surf[:, 2], surf[:, 1], s=0.8, c="#d62728", alpha=0.5,
                   label=f"SqPred (n={len(s)})")
        ax.set_title(f"{c}  cd={np.mean(by_class[c]['cd']):.4f}")
        ax.set_box_aspect((1, 1, 1))
        if k == 0:
            ax.legend(fontsize=6)
    axes[-1].axis("off")
    axes[-1].text(0.05, 0.5,
                  "rows: GT (grey) vs predicted superquadric surface (red); "
                  "unit-cube normalized object space", fontsize=8)
    out_png = args.out or os.path.join(repo, "output", "superdec_fleet_recon.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.suptitle(f"SuperDec fleet recon — {os.path.basename(args.ckpt)}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"[eval] saved {out_png}")


if __name__ == "__main__":
    main()
