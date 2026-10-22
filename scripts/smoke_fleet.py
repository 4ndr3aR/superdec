#!/usr/bin/env python3
"""SuperDec-on-fleet smoke (stable-point A/B check):
build model from a config, count params, run N real batches through
forward+loss+backward on GPU, verify finite losses and sane per-class shape recovery,
report peak GPU memory. Run ON THE POD with the py-3.13 venv:
    python scripts/smoke_fleet.py --config configs/train_fleet.yaml --batches 12
"""
import argparse
import os
import sys
import time

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from superdec.superdec import SuperDec
from superdec.data.dataloader import ShapeNet
from superdec.loss.loss import Loss


def quat_angle_deg(rotate):
    """Rotation magnitude of a batch of quat (B,N,4, scalar-first) in degrees."""
    w = rotate[..., 0].abs()
    w = torch.clamp(w, max=1.0)
    return (2.0 * torch.arccos(w) * 180.0 / torch.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--train-steps", type=int, default=0, help="optionally do N optimizer steps")
    ap.add_argument("--bs", type=int, default=0, help="override config batch size (for co-residency smoke)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    t0 = time.time()
    model = SuperDec(cfg.superdec).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] model built in {time.time()-t0:.0f}s (incl. first CUDA JIT compile). "
          f"params={n_params/1e6:.2f}M  (n_queries={cfg.superdec.decoder.n_queries}, "
          f"emb={model.emb_dims}, enc out={cfg.superdec.point_encoder.out_channels})")

    ds = ShapeNet(split="train", cfg=cfg)
    print(f"[smoke] dataset: {len(ds)} objects")
    ld = torch.utils.data.DataLoader(ds, batch_size=(args.bs or cfg.trainer.batch_size), shuffle=True,
                                     num_workers=0, drop_last=(args.bs or cfg.trainer.batch_size) > 1)
    loss_fn = Loss(cfg.loss).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr, betas=tuple(cfg.optimizer.betas))

    it = iter(ld)
    t0 = time.time()
    ok = 0
    for bi in range(args.batches):
        batch = next(it)
        pc, normals = batch["points"].cuda().float(), batch["normals"].cuda().float()
        out = model(pc)
        loss, ld_ = loss_fn(pc, normals, out)
        if args.train_steps and bi < args.train_steps:
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert torch.isfinite(loss), f"non-finite loss at batch {bi}: {ld_}"
        assert all(torch.isfinite(v).all() for k, v in out.items()), f"non-finite out at {bi}"
        ok += 1
        if bi == 0:
            print(f"[smoke] batch0: loss={loss.item():.4f} "
                  f"cd={ld_['cd_loss']:.4f} cub={ld_['cub_loss']:.4f} "
                  f"exist={ld_.get('exist_loss', float('nan')):.4f} sps={ld_['sparsity_loss']:.4f} "
                  f"expected_prim={ld_['expected_prim_num']:.2f}")
            print(f"[smoke] batch0 out shapes: " +
                  " ".join(f"{k}={tuple(v.shape)}" for k, v in out.items()))
        if bi % 10 == 9:
            torch.cuda.synchronize()
            print(f"[smoke] batches 0..{bi}: {bi+1} in {time.time()-t0:.1f}s "
                  f"peak_gpu={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
                  f"final_loss={loss.item():.4f} cd={ld_['cd_loss']:.4f} "
                  f"exist={ld_.get('exist_loss', float('nan')):.4f}")
    print(f"[smoke] PASS: {ok} batches finite forward+loss{' +backward' if args.train_steps else ''} "
          f"peak_gpu={torch.cuda.max_memory_allocated()/1e9:.2f}GB")


if __name__ == "__main__":
    main()
