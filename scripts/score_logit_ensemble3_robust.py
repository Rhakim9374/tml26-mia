"""3-way ensemble biased toward matched-recipe shadows for better pub→priv generalization.

The unconstrained 3-way sweep found pub-optimum at α(baseline)=0.73, β(lsv1)=0.06,
γ(lsv2)=0.21 with pub TPR 0.0756 — but priv didn't improve over the 2-way
(α=0.3, β=0.7, priv 0.069691). Reason: baseline shadows have a much larger
pub→priv gap (0.009) than matched-recipe shadows (0.004), so over-weighting
baseline overfits pub.

This script constrains the simplex search:
  α(baseline) ∈ [0.0, 0.4]
  β(lsv1) + γ(lsv2) ≥ 0.6
Tests every combo on a 0.02-step grid (much finer than the previous coarse).
Also explicitly tests known reasonable points.

Outputs:
  submissions/submission_ensemble3_robust.csv

Usage:
    condor_submit mia_grad.sub \\
        -append "script=scripts/score_logit_ensemble3_robust.py" \\
        -append "tag=ensemble3_robust" -queue 1
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
OUT_PATH = ROOT / "submissions" / "submission_ensemble3_robust.csv"

ALPHA_MAX = 0.40   # cap baseline weight to limit pub overfit
STEP = 0.02


def to_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(len(x))
    return rank / max(len(x) - 1, 1)


def main():
    arrays = {name: np.load(p) for name, p in PATHS.items()}
    n = arrays["baseline"].shape[0]
    for name, arr in arrays.items():
        print(f"loaded {name:9s}  shape={arr.shape}", flush=True)

    combined = load_combined()
    n_pub = combined.n_pub
    pub_membership = np.asarray(load_pub().membership, dtype=int)
    ids = combined.ids

    print(f"\nIndividual pub TPRs:", flush=True)
    for name, arr in arrays.items():
        print(f"  {name:9s}  {tpr_at_fpr(arr[:n_pub], pub_membership):.4f}",
              flush=True)

    ranks = {name: to_rank(arr) for name, arr in arrays.items()}

    def score_mix(a, b, c):
        return a * ranks["baseline"] + b * ranks["lsv1"] + c * ranks["lsv2"]

    candidates = []  # (a, b, c, pub_tpr)

    # Hand-picked checkpoints we want to evaluate explicitly.
    explicit = [
        (0.3, 0.7, 0.0),   # the previously-winning 2-way
        (0.3, 0.0, 0.7),   # 2-way with lsv2 instead of lsv1
        (0.3, 0.35, 0.35), # equal lsv split
        (0.0, 0.5, 0.5),   # no baseline
        (0.4, 0.3, 0.3),   # slight baseline lean
    ]
    print(f"\n=== explicit candidates ===", flush=True)
    print(f"{'baseline':>9}  {'lsv1':>6}  {'lsv2':>6}  {'pub TPR':>9}", flush=True)
    for (a, b, c) in explicit:
        mix = score_mix(a, b, c)
        tpr = tpr_at_fpr(mix[:n_pub], pub_membership)
        candidates.append((a, b, c, tpr))
        print(f"{a:>9.2f}  {b:>6.2f}  {c:>6.2f}  {tpr:>9.4f}", flush=True)

    # Constrained simplex sweep: α ∈ [0, ALPHA_MAX], full simplex for (β, γ).
    print(f"\n=== constrained sweep (α ≤ {ALPHA_MAX}, step {STEP}) ===",
          flush=True)
    print(f"{'baseline':>9}  {'lsv1':>6}  {'lsv2':>6}  {'pub TPR':>9}",
          flush=True)
    n_alpha = int(round(ALPHA_MAX / STEP))
    n_beta = int(round(1.0 / STEP))
    best = (-1.0, None, None)
    for i in range(n_alpha + 1):
        a = round(i * STEP, 4)
        for j in range(n_beta - i + 1):
            b = round(j * STEP, 4)
            c = round(1.0 - a - b, 4)
            if c < -1e-9:
                continue
            mix = score_mix(a, b, c)
            tpr = tpr_at_fpr(mix[:n_pub], pub_membership)
            candidates.append((a, b, c, tpr))
            if tpr > best[0]:
                best = (tpr, (a, b, c), mix)
                print(f"{a:>9.2f}  {b:>6.2f}  {c:>6.2f}  {tpr:>9.4f}  ← best",
                      flush=True)

    print(f"\n=== TOP 10 BY PUB TPR (any constraint) ===", flush=True)
    candidates.sort(key=lambda x: -x[3])
    for a, b, c, tpr in candidates[:10]:
        print(f"  α={a:.2f}  β={b:.2f}  γ={c:.2f}  pub TPR={tpr:.4f}",
              flush=True)

    a_best, b_best, c_best = best[1]
    print(f"\n=== FINAL CHOICE (constrained best) ===", flush=True)
    print(f"  α(baseline) = {a_best:.2f}", flush=True)
    print(f"  β(lsv1)     = {b_best:.2f}", flush=True)
    print(f"  γ(lsv2)     = {c_best:.2f}", flush=True)
    print(f"  pub TPR     = {best[0]:.4f}", flush=True)
    print(f"  (Reference: 2-way α=0.3 β=0.7 pub 0.0737, priv 0.069691)",
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
    print(f"  cp submissions/submission_ensemble3_robust.csv submissions/submission.csv")
    print(f"  python3 -m src.submit --tag ensemble3robust_a{a_best:.2f}_b{b_best:.2f}_c{c_best:.2f}")


if __name__ == "__main__":
    main()
