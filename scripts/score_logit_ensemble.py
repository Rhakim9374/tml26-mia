"""Rank-average ensemble of two logit-LiRA shadow families.

Loads two saved log-LR arrays from independent score_online_lira.py runs
(different shadow recipes) and combines them. The two families are:

  baseline (no LS, ep=100, 512 shadows): pub TPR 0.0697, priv 0.0607
  lsv1 (LS=0.06, ep=60, 1024 shadows):    pub TPR 0.0664, priv 0.0621

Different shadow training recipes → different LiRA reference distributions
→ different ranking errors. Rank-averaging the two priv scores has a real
chance of catching members one attack misses but the other catches.

Inputs (must already exist):
  checkpoints/logit_features_lsv1/log_lr.npy     from lsv1 run
  checkpoints/logit_features/log_lr.npy          from baseline rerun

Outputs:
  submissions/submission_ensemble_logit.csv

Usage:
    condor_submit mia_grad.sub \\
        -append "script=scripts/score_logit_ensemble.py" \\
        -append "tag=ensemble_logit" -queue 1
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

LSV1_PATH = ROOT / "checkpoints" / "logit_features_lsv1" / "log_lr.npy"
BASELINE_PATH = ROOT / "checkpoints" / "logit_features" / "log_lr.npy"
OUT_PATH = ROOT / "submissions" / "submission_ensemble_logit.csv"


def to_rank(x: np.ndarray) -> np.ndarray:
    """Convert array to per-element rank in [0, 1]. Higher x → higher rank."""
    order = np.argsort(x)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(len(x))
    return rank / max(len(x) - 1, 1)


def main():
    if not LSV1_PATH.exists():
        sys.exit(f"Missing {LSV1_PATH}")
    if not BASELINE_PATH.exists():
        sys.exit(f"Missing {BASELINE_PATH}. Run score_online_lira.py with "
                 "--ckpt_prefix shadow to generate baseline features.")

    lsv1_loglr = np.load(LSV1_PATH)
    baseline_loglr = np.load(BASELINE_PATH)
    print(f"lsv1     log-LR shape: {lsv1_loglr.shape}", flush=True)
    print(f"baseline log-LR shape: {baseline_loglr.shape}", flush=True)
    if lsv1_loglr.shape != baseline_loglr.shape:
        sys.exit("shape mismatch — both should be (n_total,) of same length")

    combined = load_combined()
    n_pub, n_priv = combined.n_pub, combined.n_priv
    n_total = n_pub + n_priv
    pub_membership = np.asarray(load_pub().membership, dtype=int)
    ids = combined.ids

    # Sanity: per-attack pub TPR.
    print(f"\nIndividual pub TPRs:", flush=True)
    print(f"  baseline (no LS):  {tpr_at_fpr(baseline_loglr[:n_pub], pub_membership):.4f}",
          flush=True)
    print(f"  lsv1 (LS=0.06):    {tpr_at_fpr(lsv1_loglr[:n_pub], pub_membership):.4f}",
          flush=True)

    # Rank-average sweep.
    baseline_rank = to_rank(baseline_loglr)
    lsv1_rank = to_rank(lsv1_loglr)

    # Coarse 0.05-step sweep across full range, then fine 0.02-step sweep
    # around the previously-found optimum (α≈0.3 gave 0.0737 on pub).
    coarse = np.round(np.arange(0.0, 1.01, 0.05), 3)
    fine = np.round(np.arange(0.20, 0.46, 0.02), 3)
    alphas = sorted(set(coarse.tolist()) | set(fine.tolist()))

    print(f"\n=== α·baseline + (1−α)·lsv1 (rank-averaged) — pub TPR ===",
          flush=True)
    best_alpha, best_tpr, best_score = 0.5, -1.0, None
    for alpha in alphas:
        mix = alpha * baseline_rank + (1 - alpha) * lsv1_rank
        tpr = tpr_at_fpr(mix[:n_pub], pub_membership)
        marker = ""
        if tpr > best_tpr:
            best_alpha, best_tpr, best_score = float(alpha), tpr, mix
            marker = "  ← best so far"
        print(f"  α={alpha:.2f}  pub TPR={tpr:.4f}{marker}", flush=True)

    print(f"\nBest mix: α={best_alpha:.1f}  pub TPR={best_tpr:.4f}", flush=True)

    # Priv submission CSV — ranks already in [0,1].
    score_priv = best_score[n_pub:]
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
    print(f"  cp submissions/submission_ensemble_logit.csv submissions/submission.csv")
    print(f"  python3 -m src.submit --tag ensemble_logit_a{best_alpha:.1f}")


if __name__ == "__main__":
    main()
