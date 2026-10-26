#!/usr/bin/env python3
"""Eval the completed SuperDec fleet run (snapshot dir with epoch_20..80.pt):
  1. parse the single-line tqdm training log -> per-epoch train/eval loss table + curves png
  2. per-checkpoint full-val (833 objects) Chamfer per class + overall -> cd-table + curve png
  3. per-checkpoint 7-class GT-vs-reconstruction figure (SAME 7 objects every epoch)
Run ON THE POD. All PNGs -> /mnt/shared/superdec-pngs/.
"""
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/shared/ubuntu-sidecar-pod/superdec")
from omegaconf import OmegaConf

from superdec.superdec import SuperDec
from superdec.data.dataloader import ShapeNet

SNAP = "/mnt/shared/ubuntu-sidecar-pod/superdec/checkpoints/superdec_fleet_small_run-20261024-1219"
RUNLOG = "/mnt/shared/ubuntu-sidecar-pod/superdec/logs/superdec_fleet_small.log"
OUT = "/mnt/shared/superdec-pngs"
N_CKPT = [20, 30, 40, 50, 60, 70, 80]
CLASSES = ["box", "sphere", "torus", "cone", "capsule", "cylinder", "pyramid"]
os.makedirs(OUT, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- loss curves from log
def parse_log():
    txt = open(RUNLOG, "rb").read().decode("utf-8", "replace")
    upds = txt.replace("\r", "\n").split("\n")
    pat = re.compile(
        r"^(Epoch|Eval)\s+(\d+)/80:.*?"
        r"cub_loss=([\d.]+), cd_loss=([\d.]+), exist_loss=([\d.]+), "
        r"sparsity_loss=([\d.]+), expected_prim_num=([\d.]+), all=([\d.]+)\]$")
    last = {}
    for line in upds:
        m = pat.match(line.strip())
        if m:
            last[(m.group(1), int(m.group(2)))] = dict(
                cub=float(m.group(3)), cd=float(m.group(4)), exist=float(m.group(5)),
                sparsity=float(m.group(6)), prim=float(m.group(7)), total=float(m.group(8)))
    return last


def loss_curves():
    last = parse_log()
    tr_e = sorted(e for ph, e in last if ph == "Epoch")
    va_e = sorted(e for ph, e in last if ph == "Eval")
    if not tr_e and not va_e:
        print("[loss] no parseable updates (log may contain only the latest truncated run)")
        return
    cols = ["cub", "cd", "exist", "sparsity", "prim", "total"]
    print(f"[loss] train epochs in log: {len(tr_e)}; eval epochs: {len(va_e)}"
          "  (NOTE: on-disk log = last relaunch's; full-run curve comes from .pt val_loss)")
    print("\n=== per-epoch LOSS (tqdm running averages)  train / eval ===")
    for e in sorted(set(tr_e) | set(va_e)):
        t = last.get(("Epoch", e), {c: float("nan") for c in cols})
        v = last.get(("Eval", e), {c: float("nan") for c in cols})
        print(f"e{e:02d} | " + " | ".join(f"{t[c]:.4f}/{v[c]:.4f}" for c in cols))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, keys, title in (
        (axes[0], ("cub", "cd"), "cub + cd loss"),
        (axes[1], ("exist", "sparsity", "prim", "total"),
         "exist + sparsity + #prim + total"),
    ):
        for k in keys:
            if tr_e:
                ax.plot(tr_e, [last[("Epoch", e)][k] for e in tr_e], marker="o", ms=3, lw=1,
                        label=f"train {k}")
            if va_e:
                ax.plot(va_e, [last[("Eval", e)][k] for e in va_e], marker="s", ms=3, lw=1.6,
                        color="tab:red", label=f"eval {k}")
        ax.set_title(title)
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "superdec_fleet_loss_curves.png"), dpi=120)
    print(f"[loss] saved {OUT}/superdec_fleet_loss_curves.png")


# ---------------------------------------------------------------- model + geometry
@torch.no_grad()
def load_model(ckpt_path, cfg):
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model = SuperDec(cfg.superdec).cuda().eval()
    model.load_state_dict(ck["model_state_dict"])
    return model, ck


def sq_surface(scale_i, eps_i, rot_i, trans_i, n_theta=24, n_phi=64):
    a1, a2, a3 = scale_i
    e1, e2 = eps_i

    def fexp(x, p):
        return torch.sign(x) * (torch.abs(x) ** p)

    th = torch.linspace(-np.pi / 2, np.pi / 2, n_theta)
    ph = torch.linspace(-np.pi, np.pi, n_phi)
    TH, PH = torch.meshgrid(th, ph, indexing="ij")
    x = a1 * fexp(torch.cos(TH), e1) * fexp(torch.cos(PH), e2)
    y = a2 * fexp(torch.cos(TH), e1) * fexp(torch.sin(PH), e2)
    z = a3 * fexp(torch.sin(TH), e1)
    return torch.stack([x, y, z], -1).reshape(-1, 3) @ rot_i.T + trans_i


def chamfer(p, surf):
    d1 = np.min(((p[:, None, :] - surf[None, :, :]) ** 2).sum(-1), 1)
    d2 = np.min(((surf[:, None, :] - p[None, :, :]) ** 2).sum(-1), 1)
    return float(np.sqrt(d1.mean() + d2.mean()))


def eval_ckpt(model, ld, viz_pick_idx):
    """Full-val per-class CD + captured (gt, surf) for the 7 viz objects."""
    per_class = {}
    viz = {i: (None, None, None) for i in viz_pick_idx}
    for bi, batch in enumerate(ld):
        pc = batch["points"].cuda().float()
        out = model(pc)
        for i in range(pc.shape[0]):
            flat = bi * pc.shape[0] + i
            mid = batch["model_id"][i]
            cls = mid.rsplit("_", 1)[-1]
            exist = out["exist"][i].squeeze(-1)
            keep = torch.where(exist > 0.5)[0]
            if len(keep) == 0:
                continue
            s, e = out["scale"][i, keep], out["shape"][i, keep]
            r, t = out["rotate"][i, keep], out["trans"][i, keep]
            p = pc[i].cpu().numpy()
            surf = None
            for j in range(len(keep)):
                si = sq_surface(s[j].cpu(), e[j].cpu(), r[j].cpu(), t[j].cpu()).cpu().numpy()
                surf = si if surf is None else np.concatenate([surf, si], 0)
            per_class.setdefault(cls, []).append(chamfer(p, surf))
            if i in viz and viz[i][0] is None:
                viz[i] = (cls, p, surf)
    overall = float(np.sqrt(np.mean([cd for v in per_class.values() for cd in v])))
    return overall, {c: float(np.mean(v)) for c, v in per_class.items()}, viz


def main():
    torch.set_grad_enabled(False)   # eval only: no grad graph (heads would still build one)
    cfg = OmegaConf.load(os.path.join(SNAP, "config.yaml"))
    ds = ShapeNet(split="val", cfg=cfg)
    ld = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"[eval] val objects: {len(ds)}  (bs32 -> {(len(ds)+31)//32} batches)")

    loss_curves()

    # one representative val object per class (first occurrence), same for all epochs
    viz_pick = {}
    for idx, s in enumerate(ds):
        cls = s["model_id"].rsplit("_", 1)[-1]
        if cls not in viz_pick:
            viz_pick[cls] = idx
        if len(viz_pick) == len(CLASSES):
            break
    print(f"[eval] viz objects (same 7 every epoch): "
          + ", ".join(f"{c}=@{i}" for c, i in viz_pick.items()))
    viz_idx = list(viz_pick.values())

    cds = {}
    val_loss_rows = []
    for ep in N_CKPT:
        model, ck = load_model(os.path.join(SNAP, f"epoch_{ep}.pt"), cfg)
        val_loss_rows.append((ep, float(ck["val_loss"])))
        overall, per_c, viz = eval_ckpt(model, ld, viz_idx)
        cds[ep] = (overall, per_c, viz)
        print(f"[eval] epoch {ep:2d}: overall {overall:.5f} | "
              + " ".join(f"{c}={per_c.get(c, float('nan')):.4f}" for c in CLASSES))
        del model
        torch.cuda.empty_cache()

    # ---- chamfer table + curves
    print("\n=== per-CLASS val CHAMFER (unit-cube normalized, full 833 val set) ===")
    print("class    " + " ".join(f"e{ep:>3d}" for ep in N_CKPT) + "   | final")
    for c in CLASSES:
        row = " ".join(f"{cds[ep][1].get(c, float('nan')):.4f}" for ep in N_CKPT)
        print(f"{c:8s} {row}   | {cds[N_CKPT[-1]][1].get(c, float('nan')):.4f}")
    row = " ".join(f"{cds[ep][0]:.4f}" for ep in N_CKPT)
    print(f"{'OVERALL':8s} {row}   | {cds[N_CKPT[-1]][0]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(N_CKPT, [cds[e][0] for e in N_CKPT], marker="o")
    axes[0].set_title("overall val Chamfer (all classes)")
    axes[0].set_xlabel("epoch"); axes[0].grid(alpha=0.3)
    for c in CLASSES:
        axes[1].plot(N_CKPT, [cds[e][1].get(c, float("nan")) for e in N_CKPT],
                     marker="o", ms=4, label=c)
    axes[1].set_title("per-class val Chamfer")
    axes[1].set_xlabel("epoch"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "superdec_fleet_cd_curves.png"), dpi=120)
    print(f"[eval] saved {OUT}/superdec_fleet_cd_curves.png")

    # ---- per-epoch 7-class recon figures (3D scatter, GT grey vs predicted red)
    for ep in N_CKPT:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8.5),
                                 subplot_kw={"projection": "3d"})
        grid = axes.reshape(-1)
        overall = cds[ep][0]
        for k, c in enumerate(CLASSES[:7]):
            ax = grid[k]
            cls, gt, surf = cds[ep][2][viz_pick[c]]
            ax.scatter(gt[:, 0], gt[:, 2], gt[:, 1], s=0.5, c="#777777")
            ax.scatter(surf[:, 0], surf[:, 2], surf[:, 1], s=1.0, c="#d62728", alpha=0.55)
            ax.set_title(f"{c}  ({cds[ep][1].get(c, float('nan')):.4f})", fontsize=9)
            ax.set_box_aspect((1, 1, 1))
        ax_extra = grid[7]
        ax_extra.axis("off")
        ax_extra.text(0.05, 0.55,
                      f"epoch {ep}:  grey = GT surface (4096 pts)\nred = predicted superquadric surface\n"
                      f"overall val CD = {overall:.5f}",
                      fontsize=10, family="monospace")
        fig.suptitle(f"SuperDec fleet reconstruction — {ep} epochs  (7 val objects, one per class)")
        fig.tight_layout()
        p = os.path.join(OUT, f"superdec_fleet_recon_e{ep:02d}.png")
        fig.savefig(p, dpi=110)
        print(f"[eval] saved {p}")
    print("[eval] ALL DONE")


if __name__ == "__main__":
    main()
