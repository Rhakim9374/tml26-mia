"""Rank-average ensemble of two logit-LiRA shadow families → final submission.

Loads log-LR arrays from two independent score_online_lira.py runs (each
on a shadow family with a different training recipe) and combines them by
ranking. Different recipes produce different reference distributions, so
they rank samples differently and catch different members.

Inputs (must already exist):
  checkpoints/logit_features_shadow/log_lr.npy   (LS=0,    epochs=100)
  checkpoints/logit_features_lsv1/log_lr.npy     (LS=0.06, epochs=60)

Output:
  submissions/submission.csv                     (priv portion, ready to upload)

The α weight (baseline vs lsv1) was chosen by sweeping pub TPR; α=0.30
gave the highest leaderboard priv TPR (0.069691). The script re-prints a
short α sweep around the optimum as a verification — the chosen mix is
the one with highest pub TPR within ±0.05 of α=0.30.

Run on the cluster (no GPU needed; pure NumPy):
    condor_submit mia.sub \\
        -append "script=scripts/score_logit_ensemble.py" \\
        -append "tag=ensemble" -queue 1
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

BASELINE_PATH = ROOT / "checkpoints" / "logit_features_shadow" / "log_lr.npy"
LSV1_PATH = ROOT / "checkpoints" / "logit_features_lsv1"   / "log_lr.npy"
OUT_PATH = ROOT / "submissions" / "submission.csv"

# α weight for the baseline family in the rank-mix (1 − α goes to lsv1).
# Chosen by pub-TPR sweep + leaderboard validation.
TARGET_ALPHA = 0.30


def to_rank(x: np.ndarray) -> np.ndarray:
    """Convert array to per-element rank in [0, 1]. Higher x → higher rank."""
    order = np.argsort(x)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(len(x))
    return rank / max(len(x) - 1, 1)


def main():
    for path in (BASELINE_PATH, LSV1_PATH):
        if not path.exists():
            sys.exit(f"Missing {path}. Run score_online_lira.py with the "
                     "matching --ckpt_prefix first.")

    baseline_loglr = np.load(BASELINE_PATH)
    lsv1_loglr = np.load(LSV1_PATH)
    if baseline_loglr.shape != lsv1_loglr.shape:
        sys.exit(f"Shape mismatch: baseline {baseline_loglr.shape} vs "
                 f"lsv1 {lsv1_loglr.shape}")

    combined = load_combined()
    n_pub = combined.n_pub
    pub_membership = np.asarray(load_pub().membership, dtype=int)
    ids = combined.ids

    print(f"baseline (LS=0)    pub TPR = "
          f"{tpr_at_fpr(baseline_loglr[:n_pub], pub_membership):.4f}", flush=True)
    print(f"lsv1     (LS=0.06) pub TPR = "
          f"{tpr_at_fpr(lsv1_loglr[:n_pub], pub_membership):.4f}", flush=True)

    baseline_rank = to_rank(baseline_loglr)
    lsv1_rank = to_rank(lsv1_loglr)

    # Verification sweep around TARGET_ALPHA. The winner on pub TPR within this
    # narrow band IS what we use for the submission, so this also handles minor
    # drift between training runs.
    print(f"\nα sweep (α·baseline + (1−α)·lsv1) around α={TARGET_ALPHA}:",
          flush=True)
    best_alpha, best_tpr, best_score = None, -1.0, None
    for alpha in np.round(np.arange(TARGET_ALPHA - 0.05, TARGET_ALPHA + 0.051, 0.01), 3):
        mix = alpha * baseline_rank + (1 - alpha) * lsv1_rank
        tpr = tpr_at_fpr(mix[:n_pub], pub_membership)
        marker = ""
        if tpr > best_tpr:
            best_alpha, best_tpr, best_score = float(alpha), tpr, mix
            marker = "  ←"
        print(f"  α={alpha:.2f}  pub TPR={tpr:.4f}{marker}", flush=True)
    print(f"\nChosen mix: α={best_alpha:.2f}  pub TPR={best_tpr:.4f}", flush=True)

    # Priv submission CSV. Ranks are already in [0, 1] so no further mapping
    # is needed; the metric only cares about ordering.
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
    print(f"\nTo upload:")
    print(f"  python -m src.submit --tag ensemble_a{best_alpha:.2f}")


if __name__ == "__main__":
    main()
