"""Train shadow models on the combined pub+priv pool.

One Condor job trains `count` shadows back-to-back. Seeds for job_id k with
count C are [k*C, k*C+1, ..., k*C+C-1] so multiple jobs in parallel produce
disjoint seed ranges.

Two shadow families are needed for the final ensemble:

  baseline (no label smoothing, 100 epochs):
    for k in $(seq 0 15); do
      condor_submit mia.sub \\
        -append "script=scripts/train_shadow.py" \\
        -append "args=$k --count 32 --ckpt_prefix shadow" \\
        -append "tag=shadow_baseline" -queue 1
    done
    # → 16 jobs × 32 shadows = 512 baseline shadows.

  recipe-matched (LS=0.06, 60 epochs):
    for k in $(seq 0 15); do
      condor_submit mia.sub \\
        -append "script=scripts/train_shadow.py" \\
        -append "args=$k --count 32 --label_smoothing 0.06 --epochs 60 --ckpt_prefix lsv1" \\
        -append "tag=shadow_lsv1" -queue 1
    done
    # → 16 jobs × 32 shadows = 512 recipe-matched shadows.
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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Cross-entropy label smoothing. Use 0.06 to roughly match "
                        "target's φ distribution; leave at 0 for the baseline family.")
    p.add_argument("--ckpt_prefix", default="shadow",
                   help="Filename prefix for checkpoints. Use distinct prefixes "
                        "(e.g. 'shadow' and 'lsv1') for different shadow families.")
    a = p.parse_args()

    base = a.job_id * a.count
    print(f"Job {a.job_id}: seeds {base}..{base + a.count - 1}", flush=True)
    print(f"  epochs={a.epochs}  batch={a.batch_size}  "
          f"label_smoothing={a.label_smoothing}  prefix={a.ckpt_prefix}",
          flush=True)
    t_total = time.time()
    for i in range(a.count):
        seed = base + i
        print(f"\n=== Shadow {i+1}/{a.count}  seed={seed} ===", flush=True)
        t0 = time.time()
        train_shadow(
            seed=seed,
            epochs=a.epochs,
            batch_size=a.batch_size,
            label_smoothing=a.label_smoothing,
            ckpt_prefix=a.ckpt_prefix,
        )
        print(f"  seed {seed} done in {time.time() - t0:.1f}s", flush=True)
    print(f"\nJob {a.job_id} total: {time.time() - t_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
