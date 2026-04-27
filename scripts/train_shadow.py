"""Train shadow models. One Condor job trains `count` shadows back-to-back.

Usage:
    python scripts/train_shadow.py <job_id> [--count N]

Seeds for job_id k with count C: [k*C, k*C+1, ..., k*C+C-1].

Submit 256 shadows in 8 jobs (32 per GPU, sequential within a job):
    condor_submit mia.sub \\
        -append "script=scripts/train_shadow.py" \\
        -append 'args=$(Process) --count 32' \\
        -append "tag=shadow" \\
        -queue 8
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
    p.add_argument("job_id", type=int, help="Per-job index (= $(Process) under Condor).")
    p.add_argument("--count", type=int, default=1,
                   help="Shadows to train back-to-back in this job.")
    p.add_argument("--pool", default="pub", choices=["pub"],
                   help="Pool to sample shadow training data from.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    a = p.parse_args()

    base = a.job_id * a.count
    print(f"Job {a.job_id}: seeds {base}..{base + a.count - 1} "
          f"(pool={a.pool}, epochs={a.epochs})", flush=True)
    t_total = time.time()
    for i in range(a.count):
        seed = base + i
        print(f"\n=== Shadow {i+1}/{a.count}  seed={seed} ===", flush=True)
        t0 = time.time()
        train_shadow(seed=seed, pool_name=a.pool, epochs=a.epochs,
                     batch_size=a.batch_size)
        print(f"  seed {seed} done in {time.time() - t0:.1f}s", flush=True)
    print(f"\nJob {a.job_id} total: {time.time() - t_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
