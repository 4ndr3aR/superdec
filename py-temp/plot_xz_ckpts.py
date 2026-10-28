"""SuperDec xz-fleet (completed) -> checkpoint convergence graphs. Robust tqdm parser.
Reads per-epoch FINAL train (222/222) + eval (26/26) postfix from the run log, plus the
val_loss stored in each saved epoch_*.pt. Two figures. CPU."""
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/mnt/shared/ubuntu-sidecar-pod/superdec"
LOG = os.path.join(REPO, "logs", "superdec_fleet_xz.log")
CKPT = os.path.join(REPO, "checkpoints", "superdec_fleet_xz")
OUT = os.path.join(REPO, "png")
os.makedirs(OUT, exist_ok=True)

KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([0-9.eE+-]+|nan)")
LINE = re.compile(r"^\s*(Epoch|Eval)\s+(\d+)/\d+:\s*(\d+)%.*?(\d+)/(\d+)\s+\[")

def parse_log(path, comp):
    """Return dict: (kind) -> {epoch: {comp: val}} taking the LAST 100% line per epoch."""
    final = {"train": {}, "eval": {}}
    for line in open(path, errors="replace"):
        m = LINE.match(line)
        if not m:
            continue
        kind = "train" if m.group(1) == "Epoch" else "eval"
        ep = int(m.group(2))
        pct = int(m.group(3))
        kv = {k: float(v) for k, v in KV.findall(line)}
        if pct < 100 or "all" not in kv or np.isnan(kv["all"]):
            continue
        final[kind][ep] = kv          # last 100% line for the epoch wins
    return final

d = parse_log(LOG, None)
tr = d["train"]; vl = d["eval"]
te = sorted(tr); ve = sorted(vl)
print(f"train epochs {te[0]}..{te[-1]} ({len(te)}), eval epochs {ve[0]}..{ve[-1]} ({len(ve)})")
print(f"eval e1 all={vl[ve[0]]['all']:.4f}  e80 all={vl[ve[-1]]['all']:.4f}")

# val_loss straight from the checkpoints
cp_e, cp_v = [], []
for f in sorted(glob.glob(os.path.join(CKPT, "epoch_*.pt"))):
    m = re.match(r"epoch_(\d+)\.pt$", os.path.basename(f))
    import torch
    c = torch.load(f, map_location="cpu", weights_only=False)
    cp_e.append(int(m.group(1))); cp_v.append(float(c["val_loss"]))
cp_e = np.array(cp_e); cp_v = np.array(cp_v)
print("checkpoint val_loss:", {e: round(v, 5) for e, v in zip(cp_e, cp_v)})

def ys(dct, ekeys, comp):
    return [dct[e][comp] for e in ekeys if comp in dct.get(e, {})]

# ============ FIGURE 1: convergence (train+val total, ckpt dots) ============
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(te, ys(tr, te, "all"), label="train (epoch mean)", lw=2, color="#1f77b4")
ax[0].plot(ve, ys(vl, ve, "all"), label="val", lw=2, color="#d62728")
ax[0].plot(cp_e, cp_v, "o", ms=5, color="#444", label="ckpt val_loss field (= val cub_loss)")
ax[0].set_title("SuperDec xz-fleet: total loss over 80 epochs")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].annotate(f"val e80 = {vl[ve[-1]]['all']:.4f}", xy=(ve[-1], vl[ve[-1]]['all']),
               xytext=(ve[-1]-30, vl[ve[-1]]['all']-0.006), arrowprops=dict(arrowstyle="->"))

for comp, c in [("cd_loss", "#d62728"), ("cub_loss", "#2ca02c"), ("exist_loss", "#9467bd"), ("sparsity_loss", "#ff7f0e")]:
    ax[1].plot(ve, ys(vl, ve, comp), label=comp.replace("_loss", ""), lw=1.8, color=c)
ax[1].set_title("val per-component")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("loss"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout()
p1 = os.path.join(OUT, "superdec_xz_convergence.png")
fig.savefig(p1, dpi=110); plt.close(fig)
print("wrote", p1)

# ============ FIGURE 2: #primitives + sparsity (the quadric-count story) ============
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
prim_t = ys(tr, te, "expected_prim_num"); prim_v = ys(vl, ve, "expected_prim_num")
ax[0].plot(te[:len(prim_t)], prim_t, label="train", lw=2, color="#1f77b4")
ax[0].plot(ve[:len(prim_v)], prim_v, label="val", lw=2, color="#d62728")
ax[0].axhline(1.0, color="gray", ls="--", lw=1)
ax[0].set_title("expected #superquadrics / object  (-> 1 = the fleet is monoprimitive)")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("n_prim"); ax[0].legend(); ax[0].grid(alpha=.3)
ex_v = ys(vl, ve, "exist_loss")
ax[1].plot(ve[:len(ex_v)], ex_v, lw=2, color="#9467bd")
ax[1].set_title("existence BCE (val)")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("loss"); ax[1].grid(alpha=.3)
fig.tight_layout()
p2 = os.path.join(OUT, "superdec_xz_primcount.png")
fig.savefig(p2, dpi=110); plt.close(fig)
print("wrote", p2)
