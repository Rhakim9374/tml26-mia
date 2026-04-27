"""Train one shadow model.

Usage:
    python scripts/train_shadow.py <seed>

Submit 16 shadows in one Condor batch:
    condor_submit mia.sub \\
        -append "script=scripts/train_shadow.py" \\
        -append 'args=$(Process)' \\
        -append "tag=shadow" \\
        -queue 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train import train_shadow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("seed", type=int, help="Seed = $(Process) under Condor.")
    p.add_argument("--pool", default="pub", choices=["pub"],
                   help="Pool to sample shadow training data from.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    a = p.parse_args()

    print(f"Training shadow seed={a.seed} pool={a.pool} epochs={a.epochs}", flush=True)
    t0 = time.time()
    train_shadow(seed=a.seed, pool_name=a.pool, epochs=a.epochs,
                 batch_size=a.batch_size)
    print(f"Done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
