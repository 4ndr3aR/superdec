"""Plot the standalone xz-SuperDec run curves from its tqdm log, + an vs-NPZ-baseline compare. CPU."""
import re, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/mnt/shared/ubuntu-sidecar-pod/superdec"
XZ_LOG = os.path.join(REPO, "logs", "superdec_fleet_xz.log")
NPZ_LOG = os.path.join(REPO, "logs", "superdec_fleet_small.log")
OUT = os.path.join(REPO, "png")
os.makedirs(OUT, exist_ok=True)

COMP = ["cd_loss", "cub_loss", "sparsity_loss", "exist_loss"]
KV = re.compile(r"([a-z_]+)=([0-9.]+|\d+\.nan)")

def parse(path, kind):
    """per-epoch: for train, MEAN of all iterations' value; for eval, the final iter (= epoch mean).
    kind in {'train','eval'}; returns dict comp->list over epochs (in epoch order)."""
    key = "Epoch" if kind == "train" else "Eval "
    per_epoch = {}
    for line in open(path, errors="replace"):
        m = re.match(r"^(\s*)(Epoch|Eval)\s+(\d+)/", line)
        if not m:
            continue
        if (m.group(2) == "Epoch") != (kind == "train"):
            continue
        ep = int(m.group(3))
        kv = {k: float(v) for k, v in KV.findall(line)}
        if not kv or "all" not in kv or np.isnan(kv.get("all", np.nan)):
            continue
        per_epoch.setdefault(ep, []).append(kv)
    return per_epoch

def series(per_epoch, comp, kind):
    out = {}
    for ep, lst in per_epoch.items():
        vals = [kv[comp] for kv in lst if comp in kv and kv[comp] == kv[comp]]
        if not vals:
            continue
        out[ep] = float(np.mean(vals)) if kind == "train" else float(lst[-1][comp])
    return out

def ekeys(s):
    return sorted(s)

# ---- xz run ----
xz_tr = parse(XZ_LOG, "train")
xz_vl = parse(XZ_LOG, "eval")
xz_tr_all = series(xz_tr, "all", "train")
xz_vl_all = series(xz_vl, "all", "eval")
xz_vl_cd = series(xz_vl, "cd_loss", "eval")
xz_vl_cub = series(xz_vl, "cub_loss", "eval")
xz_vl_sps = series(xz_vl, "sparsity_loss", "eval")
xz_vl_ex = series(xz_vl, "exist_loss", "eval")
xz_vl_prim = series(xz_vl, "expected_prim_num", "eval")

print("xz train epochs:", ekeys(xz_tr_all)[0], "..", ekeys(xz_tr_all)[-1], f"({len(ekeys(xz_tr_all))} ep)")
print("xz val  epochs:", ekeys(xz_vl_all)[0], "..", ekeys(xz_vl_all)[-1], f"({len(ekeys(xz_vl_all))} ep)")
print("xz val all  e80 =", round(xz_vl_all[80], 5))

# ---- npz baseline ----
npz = None
if os.path.exists(NPZ_LOG):
    npz_vl_all = series(parse(NPZ_LOG, "eval"), "all", "eval")
    if npz_vl_all:
        npz = npz_vl_all
        print("npz val epochs:", ekeys(npz)[0], "..", ekeys(npz)[-1], f"({len(ekeys(npz))} ep)")

def _xy(s):
    ks, vs = ekeys(s), [s[k] for k in ekeys(s)]
    return ks, vs

# ================= FIG 1: xz run full curves (2x2) =================
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("SuperDec on randx64 FLEET via RAW .xz surface clouds  (7098 train / 821 val objects, 80 epochs)", fontsize=13)
kt, vt = _xy(xz_tr_all); kv, vv = _xy(xz_vl_all)
ax[0,0].plot(kt, vt, label="train", lw=2)
ax[0,0].plot(kv, vv, label="val", lw=2.2)
ax[0,0].set_title("Total loss (train avg vs val epoch-mean)")
ax[0,0].set_xlabel("epoch"); ax[0,0].set_ylabel("loss"); ax[0,0].legend(); ax[0,0].grid(alpha=.3)
for s_,lab,c in ((xz_vl_cd,"cd","#1f77b4"),(xz_vl_cub,"cub","#d62728"),(xz_vl_sps,"sparsity","#2ca02c"),(xz_vl_ex,"exist","#9467bd")):
    ks,vs = _xy(s_); ax[0,1].plot(ks,vs,label=lab,color=c,lw=1.8)
ax[0,1].set_title("Val per-component loss"); ax[0,1].set_xlabel("epoch"); ax[0,1].set_ylabel("loss"); ax[0,1].legend(); ax[0,1].grid(alpha=.3)
ks,vs = _xy(xz_vl_all); ax[1,0].plot(ks,vs,color="#1f77b4",lw=2.2)
if npz is not None:
    ks2,vs2 = _xy(npz); ax[1,0].plot(ks2,vs2,color="gray",lw=1.6,ls="--",label="prior NPZ baseline (833 val, npz src)")
ax[1,0].set_title("Val total loss (xz vs prior NPZ)"); ax[1,0].set_xlabel("epoch"); ax[1,0].set_ylabel("loss"); ax[1,0].legend(); ax[1,0].grid(alpha=.3)
ks,vs = _xy(xz_vl_prim); ax[1,1].plot(ks,vs,color="#ff7f0e",lw=2)
ax[1,1].set_title("Val expected #primitives/object"); ax[1,1].set_xlabel("epoch"); ax[1,1].set_ylabel("n_prim"); ax[1,1].grid(alpha=.3)
fig.tight_layout(rect=[0,0,1,0.96]); p1=os.path.join(OUT,"superdec_xz_curves.png"); fig.savefig(p1,dpi=110); plt.close(fig)
print("wrote", p1)

print("DONE")
