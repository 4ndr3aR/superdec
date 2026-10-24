#!/usr/bin/env python3
"""FAST eval of the clean full-run SuperDec fleet checkpoints (e30..e80).
  * per-checkpoint val Chamfer (per-class + overall), GT subsampled -> speed on CPU
  * per-checkpoint 7-class recon figure (same 7 val objects each epoch)
  * summary curve: stored val_loss (eval cub_loss) + overall chamfer
PNGs -> /mnt/shared/superdec-pngs/.  Run ON THE POD (GPU0).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/shared/ubuntu-sidecar-pod/superdec")
from omegaconf import OmegaConf
from superdec.superdec import SuperDec
from superdec.data.dataloader import ShapeNet

SNAP = "/mnt/shared/ubuntu-sidecar-pod/superdec/checkpoints/superdec_fleet_small_run-20261024-1219"
OUT = "/mnt/shared/superdec-pngs"
CKPT = [30, 40, 50, 60, 70, 80]          # clean full-run (e20 was overwritten by a restart)
CLASSES = ["box", "sphere", "torus", "cone", "capsule", "cylinder", "pyramid"]
NSUB_GT = 1024                            # subsample GT for fast chamfer
NSUB_SQ = 512
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def chamfer(p, s, sub_gt=NSUB_GT, sub_sq=NSUB_SQ):
    idx = torch.randperm(p.shape[0])[:sub_gt]
    jdx = torch.randperm(s.shape[0])[:sub_sq]
    p = p[idx]; s = s[jdx]
    d1 = torch.cdist(p, s).min(1)[0].mean()
    d2 = torch.cdist(s, p).min(1)[0].mean()
    return float((d1 + d2).item())


def main():
    dev = "cuda"
    cfg = OmegaConf.load(os.path.join(SNAP, "config.yaml"))
    ds = ShapeNet(split="val", cfg=cfg)
    ld = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    n = len(ds)
    print(f"[fast] val objects={n}", flush=True)

    # one val object per class (first occurrence) -> same 7 every epoch
    viz = {}
    for idx, s in enumerate(ds):
        c = s["model_id"].rsplit("_", 1)[-1]
        if c not in viz:
            viz[c] = idx
        if len(viz) == len(CLASSES):
            break
    viz_idx = list(viz.values())
    print("[fast] viz: " + ", ".join(f"{c}={i}" for c, i in viz.items()), flush=True)

    cds, valv, rec = {}, {}, {}
    for ep in CKPT:
        ckp = torch.load(os.path.join(SNAP, f"epoch_{ep}.pt"), map_location=dev, weights_only=False)
        m = SuperDec(cfg.superdec).to(dev).eval()
        m.load_state_dict(ckp["model_state_dict"])
        per_c, vizout = {}, {i: (None, None, None) for i in viz_idx}
        for bi, b in enumerate(ld):
            pc = b["points"].to(dev).float()
            o = m(pc)
            for i in range(pc.shape[0]):
                flat = bi * pc.shape[0] + i
                c = b["model_id"][i].rsplit("_", 1)[-1]
                k = torch.where(o["exist"][i] > 0.5)[0]
                if len(k) == 0:
                    continue
                s = o["scale"][i, k]; e = o["shape"][i, k]
                r = o["rotate"][i, k]; t = o["trans"][i, k]
                surf = torch.cat([sq_surface(s[j], e[j], r[j], t[j], dev=dev)
                                  for j in range(len(k))], 0)
                per_c.setdefault(c, []).append(chamfer(pc[i], surf))
                if flat in viz_idx and vizout[flat][0] is None:
                    vizout[flat] = (c, pc[i].cpu().numpy(), surf.cpu().numpy())
        overall = float(np.mean([v for vv in per_c.values() for v in vv]))
        cds[ep] = overall
        valv[ep] = float(ckp["val_loss"])
        rec[ep] = per_c
        print(f"[fast] e{ep:02d} val_cub={valv[ep]:.6f} overallCD={overall:.5f} | "
              + " ".join(f"{c}={np.mean(rec[ep][c]):.4f}" for c in CLASSES), flush=True)
        # one recon figure per checkpoint: 7 explicit 3D axes + a 2D text panel (fig.text)
        fig = plt.figure(figsize=(18, 8.5))
        for kk, c in enumerate(CLASSES):
            gi = fig.add_subplot(2, 4, kk + 1, projection="3d")
            gt, surf = vizout[viz[c]][1], vizout[viz[c]][2]
            gi.scatter(gt[:, 0], gt[:, 2], gt[:, 1], s=0.6, c="#888888")
            si = surf[::max(1, len(surf) // 4000)]
            gi.scatter(si[:, 0], si[:, 2], si[:, 1], s=0.9, c="#d62728", alpha=0.6)
            gi.set_title(f"{c}  cd={np.mean(per_c[c]):.4f}", fontsize=9)
            gi.set_box_aspect((1, 1, 1))
        fig.text(0.985, 0.012,
                 f"epoch {ep}   grey = GT  red = predicted superquadric\n"
                 f"stored eval cub_loss = {valv[ep]:.6f}   overall val Chamfer = {overall:.5f}",
                 ha="right", fontsize=11, family="monospace")
        fig.suptitle(f"SuperDec fleet recon — {ep} epochs (7 val objects, one per class)")
        fig.tight_layout()
        p = os.path.join(OUT, f"fleet_recon_e{ep:02d}.png")
        fig.savefig(p, dpi=100)
        print(f"[fast] saved {p}", flush=True)
        del m
        torch.cuda.empty_cache()

    # summary curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(CKPT, [valv[e] for e in CKPT], marker="o", lw=2)
    axes[0].set_title("stored val cub_loss (full-run, eval @833)")
    axes[0].set_xlabel("epoch"); axes[0].grid(alpha=0.3)
    for c in CLASSES:
        axes[1].plot(CKPT, [np.mean(rec[e][c]) for e in CKPT], marker="o", ms=4, label=c)
    axes[1].plot(CKPT, [cds[e] for e in CKPT], marker="s", ms=5, lw=2, color="k", label="overall")
    axes[1].set_title("val Chamfer per class (subsampled)")
    axes[1].set_xlabel("epoch"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fleet_curves.png"), dpi=120)
    print(f"[fast] saved {OUT}/fleet_curves.png", flush=True)

    # final per-class table
    print("\n=== FINAL (e80) val Chamfer per class ===")
    for c in CLASSES:
        print(f"  {c:9s} {np.mean(rec[80][c]):.5f}")
    print("  OVERALL   %.5f" % cds[80])
    print("[fast] DONE", flush=True)


if __name__ == "__main__":
    main()
