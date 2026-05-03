"""Train shadow models on the combined pub+priv pool. One Condor job trains
`count` shadows back-to-back.

Usage:
    python scripts/train_shadow.py <job_id> [--count N]

Seeds for job_id k with count C: [k*C, k*C+1, ..., k*C+C-1].

Submit 512 shadows in 16 jobs (32 per GPU, sequential within a job):
    condor_submit mia.sub \\
        -append "script=scripts/train_shadow.py" \\
        -append 'args=$(Process) --count 32' \\
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
    p.add_argument("job_id", type=int, help="Per-job index (= $(Process) under Condor).")
    p.add_argument("--count", type=int, default=1,
                   help="Shadows to train back-to-back in this job.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--optimizer", default="sgd",
                   choices=["sgd", "adam", "adamw"])
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--no_nesterov", action="store_true")
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--mixup_alpha", type=float, default=0.0)
    p.add_argument("--aug_strong", action="store_true")
    p.add_argument("--scheduler", default="cosine",
                   choices=["cosine", "step", "linear", "constant"])
    p.add_argument("--ckpt_prefix", default="shadow",
                   help="Filename prefix for checkpoints. Use a non-default value "
                        "(e.g. 'lsv1') so a recipe variant doesn't clobber "
                        "the baseline shadow_NNNN.pt files.")
    a = p.parse_args()

    base = a.job_id * a.count
    print(f"Job {a.job_id}: seeds {base}..{base + a.count - 1}", flush=True)
    print(f"  epochs={a.epochs}  batch={a.batch_size}  optimizer={a.optimizer}  "
          f"lr={a.lr}", flush=True)
    print(f"  momentum={a.momentum}  nesterov={not a.no_nesterov}  "
          f"weight_decay={a.weight_decay}  scheduler={a.scheduler}", flush=True)
    print(f"  label_smoothing={a.label_smoothing}  mixup_alpha={a.mixup_alpha}  "
          f"aug_strong={a.aug_strong}", flush=True)
    print(f"  prefix={a.ckpt_prefix}", flush=True)
    t_total = time.time()
    for i in range(a.count):
        seed = base + i
        print(f"\n=== Shadow {i+1}/{a.count}  seed={seed} ===", flush=True)
        t0 = time.time()
        train_shadow(
            seed=seed,
            epochs=a.epochs,
            batch_size=a.batch_size,
            optimizer_type=a.optimizer,
            lr=a.lr,
            momentum=a.momentum,
            nesterov=not a.no_nesterov,
            weight_decay=a.weight_decay,
            label_smoothing=a.label_smoothing,
            mixup_alpha=a.mixup_alpha,
            aug_strong=a.aug_strong,
            scheduler_type=a.scheduler,
            ckpt_prefix=a.ckpt_prefix,
        )
        print(f"  seed {seed} done in {time.time() - t0:.1f}s", flush=True)
    print(f"\nJob {a.job_id} total: {time.time() - t_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
