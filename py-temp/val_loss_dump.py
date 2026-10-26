import os, torch
snaps = {
 "run-20261024-1219 (snapshot)": "/mnt/shared/ubuntu-sidecar-pod/superdec/checkpoints/superdec_fleet_small_run-20261024-1219",
 "live dir": "/mnt/shared/ubuntu-sidecar-pod/superdec/checkpoints/superdec_fleet_small",
}
for name, d in snaps.items():
    print(f"== {name}")
    rows = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".pt"):
            continue
        ck = torch.load(os.path.join(d, f), map_location="cpu", weights_only=False)
        rows.append((int(f.split("_")[1].split(".")[0]), ck["epoch"], ck["val_loss"]))
    for ep, ck_ep, vl in sorted(rows):
        print(f"  {f'epoch_{ep:02d}.pt':16s} ckpt.epoch={ck_ep:2d}  val_loss={vl:.6f}")
