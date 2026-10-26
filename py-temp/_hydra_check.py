"""Mirror train/train.py's hydra entry (same config_path/naming/overrides) but with a no-op
main, to prove: (a) configs/train_fleet.yaml resolves from the repo root with dotlist
overrides, (b) hydra.run.dir='.' keeps the CWD at the launch dir (so 'data/fleet' resolves).
Run:  cd <superdec-repo> && python scripts/_hydra_check.py trainer.batch_size=16
"""
import os

import hydra
from omegaconf import DictConfig

_here = os.path.dirname(os.path.abspath(__file__))          # <repo>/scripts
repo = os.path.dirname(_here)
cwd_before = os.getcwd()


@hydra.main(config_path="../configs", config_name="train_fleet", version_base=None)
def main(cfg: DictConfig):
    assert os.getcwd() == cwd_before, f"hydra chdir broke: {os.getcwd()} != {cwd_before}"
    assert os.path.isdir(os.path.join(os.getcwd(), cfg.shapenet.path)), "data/fleet not visible from CWD"
    print(f"HYDRA_OK cwd-pinned CWD={os.getcwd()}  bs={cfg.trainer.batch_size}  "
          f"queries={cfg.superdec.decoder.n_queries}  save_path={cfg.trainer.save_path}/{cfg.run_name}")


# train/train.py is launched as `python train/train.py` from the repo root; replicate the
# sys.path + argv contract of that entrypoint for this check.
import sys
sys.path.insert(0, os.path.join(repo, "train"))
sys.argv = ["_hydra_check.py"] + sys.argv[1:]
os.chdir(repo)
main()
