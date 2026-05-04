"""3-way rank-average ensemble of logit-LiRA shadow families.

Loads three saved log-LR arrays from independent score_online_lira.py runs
with different shadow recipes, and combines them via rank-averaging across
a 2-simplex sweep (α + β + γ = 1).

Families:
  baseline  (no LS, ep=100, 512 shadows):  pub TPR 0.0697
  lsv1      (LS=0.06, ep=60, 1024 shadows): pub TPR 0.0664
  lsv2      (LS=0.05, ep=75, 512 shadows):  pub TPR 0.0657

The 2-way ensemble (baseline + lsv1, α=0.3) hit pub 0.0737, priv 0.0697.
Adding a third recipe gives a chance to catch members the 2-way mix misses,
IF lsv2 ranks samples differently enough from both baseline and lsv1.

Strategy:
  - Coarse 0.1-step sweep over the 2-simplex (66 combinations)
  - Fine 0.05-step sweep around the best coarse point (~30 more)

Inputs (must already exist):
  checkpoints/logit_features_baseline/log_lr.npy
  checkpoints/logit_features_lsv1/log_lr.npy
  checkpoints/logit_features_lsv2/log_lr.npy

Outputs:
  submissions/submission_ensemble3.csv

Usage:
    condor_submit mia_grad.sub \\
        -append "script=scripts/score_logit_ensemble3.py" \\
        -append "tag=ensemble3" -queue 1
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_combined, load_pub
from src.eval import tpr_at_fpr

PATHS = {
    "baseline": ROOT / "checkpoints" / "logit_features_baseline" / "log_lr.npy",
    "lsv1":     ROOT / "checkpoints" / "logit_features_lsv1"     / "log_lr.npy",
    "lsv2":     ROOT / "checkpoints" / "logit_features_lsv2"     / "log_lr.npy",
}
OUT_PATH = ROOT / "submissions" / "submission_ensemble3.csv"


def to_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(len(x))
    return rank / max(len(x) - 1, 1)


def simplex_grid(step: float, center: tuple[float, float, float] | None = None,
                 radius: float = 0.0):
    """Yield (a, b, c) tuples on the 2-simplex (a+b+c=1, all ≥ 0)."""
    if center is None:
        # Full simplex sweep
        n = int(round(1.0 / step))
        for i in range(n + 1):
            for j in range(n - i + 1):
                k = n - i - j
                yield round(i * step, 6), round(j * step, 6), round(k * step, 6)
    else:
        # Local sweep around `center` within `radius` (L_inf in α-β-γ space)
        ca, cb, cc = center
        n = int(round(radius / step))
        for di in range(-n, n + 1):
            for dj in range(-n, n + 1):
                a = round(ca + di * step, 6)
                b = round(cb + dj * step, 6)
                c = round(1 - a - b, 6)
                if a < -1e-9 or b < -1e-9 or c < -1e-9:
                    continue
                if a > 1 + 1e-9 or b > 1 + 1e-9 or c > 1 + 1e-9:
                    continue
                yield a, b, c


def main():
    for name, path in PATHS.items():
        if not path.exists():
            sys.exit(f"Missing {name}: {path}")

    arrays = {name: np.load(path) for name, path in PATHS.items()}
    n = arrays["baseline"].shape[0]
    for name, arr in arrays.items():
        if arr.shape[0] != n:
            sys.exit(f"Shape mismatch: {name} has {arr.shape[0]}, expected {n}")
        print(f"loaded {name:9s}  shape={arr.shape}", flush=True)

    combined = load_combined()
    n_pub, n_priv = combined.n_pub, combined.n_priv
    pub_membership = np.asarray(load_pub().membership, dtype=int)
    ids = combined.ids

    # Sanity: per-attack pub TPR.
    print(f"\nIndividual pub TPRs:", flush=True)
    for name, arr in arrays.items():
        print(f"  {name:9s}  {tpr_at_fpr(arr[:n_pub], pub_membership):.4f}",
              flush=True)

    ranks = {name: to_rank(arr) for name, arr in arrays.items()}

    def score_mix(a, b, c):
        return a * ranks["baseline"] + b * ranks["lsv1"] + c * ranks["lsv2"]

    # Coarse 0.05-step sweep over full simplex.
    print(f"\n=== COARSE 0.05-step sweep (231 combos) ===", flush=True)
    print(f"{'baseline':>9}  {'lsv1':>6}  {'lsv2':>6}  {'pub TPR':>9}", flush=True)
    best = (-1.0, (0.0, 0.0, 0.0), None)
    for (a, b, c) in simplex_grid(0.05):
        mix = score_mix(a, b, c)
        tpr = tpr_at_fpr(mix[:n_pub], pub_membership)
        if tpr > best[0]:
            best = (tpr, (a, b, c), mix)
            print(f"{a:>9.2f}  {b:>6.2f}  {c:>6.2f}  {tpr:>9.4f}  ← best",
                  flush=True)

    print(f"\nBest from coarse: α(baseline)={best[1][0]:.2f}  "
          f"β(lsv1)={best[1][1]:.2f}  γ(lsv2)={best[1][2]:.2f}  "
          f"pub TPR={best[0]:.4f}", flush=True)

    # Fine 0.02-step sweep within ±0.10 of coarse winner.
    print(f"\n=== FINE 0.02-step sweep (radius 0.10 around coarse winner) ===",
          flush=True)
    print(f"{'baseline':>9}  {'lsv1':>6}  {'lsv2':>6}  {'pub TPR':>9}", flush=True)
    for (a, b, c) in simplex_grid(0.02, center=best[1], radius=0.10):
        mix = score_mix(a, b, c)
        tpr = tpr_at_fpr(mix[:n_pub], pub_membership)
        if tpr > best[0]:
            best = (tpr, (a, b, c), mix)
            print(f"{a:>9.2f}  {b:>6.2f}  {c:>6.2f}  {tpr:>9.4f}  ← best",
                  flush=True)

    a_best, b_best, c_best = best[1]
    print(f"\n=== FINAL BEST ===", flush=True)
    print(f"  α(baseline) = {a_best:.2f}", flush=True)
    print(f"  β(lsv1)     = {b_best:.2f}", flush=True)
    print(f"  γ(lsv2)     = {c_best:.2f}", flush=True)
    print(f"  pub TPR     = {best[0]:.4f}", flush=True)
    print(f"  (2-way baseline+lsv1 α=0.3 was 0.0737 / priv 0.069691)",
          flush=True)

    score_priv = best[2][n_pub:]
    priv_ids = ids[n_pub:]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "score"])
        for i, s in zip(priv_ids, score_priv):
            w.writerow([str(i), f"{s:.6f}"])
    print(f"\nWrote {OUT_PATH}  rows={len(priv_ids)}  "
          f"score range=[{score_priv.min():.4f}, {score_priv.max():.4f}]",
          flush=True)
    print(f"\nTo submit:")
    print(f"  cp submissions/submission_ensemble3.csv submissions/submission.csv")
    print(f"  python3 -m src.submit --tag ensemble3_a{a_best:.2f}_b{b_best:.2f}_c{c_best:.2f}")


if __name__ == "__main__":
    main()
