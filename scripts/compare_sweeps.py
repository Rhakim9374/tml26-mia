"""Rank all completed shadow-recipe sweeps by how well their φ distribution
matches the target's.

Loads:
  checkpoints/<prefix>_<seed>_phi.npy        (per-sample shadow φ on combined pool)
  checkpoints/<prefix>_<seed>_in_idx.pt      (IN-mask for this shadow)
  checkpoints/logit_features/phi_target.npy  (target φ on combined pool)

For each sweep, splits the shadow's φ into IN / OUT halves and reports the L1
percentile distance to target's distribution at percentiles {1,5,25,50,75,95,99}.
Lower distance = closer match = better candidate for full-batch training.

Usage (any time after at least one sweep finishes AND phi_target.npy exists):
    python3 scripts/compare_sweeps.py [--prefix-pattern sweep_]
or via Condor:
    condor_submit mia_grad.sub \\
        -append "script=scripts/compare_sweeps.py" \\
        -append "tag=compare_sweeps" -queue 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINTS_DIR = ROOT / "checkpoints"
PHI_TARGET_PATH = CHECKPOINTS_DIR / "logit_features" / "phi_target.npy"
QS = (1, 5, 25, 50, 75, 95, 99)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix_pattern", default="sweep_",
                   help="Glob filter on shadow prefix (default: 'sweep_').")
    a = p.parse_args()

    if not PHI_TARGET_PATH.exists():
        sys.exit(f"Missing {PHI_TARGET_PATH}. Run score_online_lira.py with the "
                 "feature-saving patch first (the multi-aug rerun writes this).")

    phi_target = np.load(PHI_TARGET_PATH)
    target_pct = np.percentile(phi_target, QS)
    print(f"target φ percentiles  ({len(phi_target)} samples):", flush=True)
    print(f"  " + "  ".join(f"p{q}={target_pct[i]:+6.2f}" for i, q in enumerate(QS)),
          flush=True)

    phi_files = sorted(CHECKPOINTS_DIR.glob(f"{a.prefix_pattern}*_phi.npy"))
    if not phi_files:
        sys.exit(f"No φ files matching {a.prefix_pattern}*_phi.npy in {CHECKPOINTS_DIR}")
    print(f"\nFound {len(phi_files)} sweep φ files. Ranking by L1 percentile "
          f"distance (IN distribution vs target).\n", flush=True)

    results = []
    for phi_path in phi_files:
        stem = phi_path.stem  # e.g. "sweep_ls005_9000_phi"
        idx_path = CHECKPOINTS_DIR / (stem.replace("_phi", "_in_idx") + ".pt")
        if not idx_path.exists():
            print(f"  skip {stem}: no in_idx file", flush=True)
            continue
        phi_shadow = np.load(phi_path)
        in_idx = torch.load(idx_path, weights_only=True).numpy()
        in_mask = np.zeros(phi_shadow.shape[0], dtype=bool)
        in_mask[in_idx] = True
        phi_in = phi_shadow[in_mask]
        phi_out = phi_shadow[~in_mask]

        in_pct = np.percentile(phi_in, QS)
        out_pct = np.percentile(phi_out, QS)
        match_in_l1 = float(np.abs(in_pct - target_pct).mean())
        match_out_l1 = float(np.abs(out_pct - target_pct).mean())
        match_avg = (match_in_l1 + match_out_l1) / 2.0

        results.append({
            "name": stem.replace("_phi", ""),
            "match_in":  match_in_l1,
            "match_out": match_out_l1,
            "match_avg": match_avg,
            "in_pct":    in_pct,
            "out_pct":   out_pct,
        })

    results.sort(key=lambda r: r["match_avg"])

    print(f"{'rank':>4}  {'name':<35}  {'match_avg':>9}  {'match_in':>9}  {'match_out':>9}")
    for rank, r in enumerate(results, 1):
        print(f"{rank:>4}  {r['name']:<35}  "
              f"{r['match_avg']:>9.3f}  {r['match_in']:>9.3f}  {r['match_out']:>9.3f}",
              flush=True)

    print(f"\nBest match: {results[0]['name']}")
    best_in = results[0]["in_pct"]
    print(f"  shadow IN percentiles: " +
          "  ".join(f"p{q}={best_in[i]:+6.2f}" for i, q in enumerate(QS)))
    print(f"  target percentiles:    " +
          "  ".join(f"p{q}={target_pct[i]:+6.2f}" for i, q in enumerate(QS)))


if __name__ == "__main__":
    main()
