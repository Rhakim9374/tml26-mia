"""Post-hoc scoring variants on saved adv_distance features.

The original adv_distance score (loss_before − loss_after) hit pub TPR
0.0451 ≈ random because PGD over-attacked: loss_after blew up to ~8 for
every sample, leaving uniform variance and no signal. This script tries
several alternative scoring functions on the same saved (loss_before,
loss_after) arrays — no model re-evaluation needed:

  * raw_diff       = loss_before − loss_after        (original, baseline)
  * neg_after      = −loss_after                     (just final loss, ignore initial)
  * ratio          = −loss_after / loss_before       (scale-invariant, member-y if attack proportionally weak)
  * log_ratio      = −(log loss_after − log loss_before)
  * rel_increase   = −(loss_after − loss_before) / loss_before
  * log_after      = −log(loss_after)                (smooth tail-handling on the after side)

Higher score = more member-y in every variant. Pick the one with best
pub TPR; if any beats 0.06, it's a useful ensemble candidate.

Usage:
    condor_submit mia_grad.sub \\
        -append "script=scripts/score_adv_variants.py" \\
        -append "tag=adv_variants" -queue 1
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

FEATURES_DIR = ROOT / "checkpoints" / "adv_distance"
SUBMISSIONS_DIR = ROOT / "submissions"
EPS = 1e-6


def write_submission(name: str, score: np.ndarray, n_pub: int, ids: list):
    priv_score = score[n_pub:]
    # Min-max normalize to [0, 1] (rank-preserving). Avoids the loss-of-scale
    # issue some sigmoid clips have on big-magnitude scores.
    s_min, s_max = priv_score.min(), priv_score.max()
    if s_max - s_min < EPS:
        priv_score_01 = np.full_like(priv_score, 0.5)
    else:
        priv_score_01 = (priv_score - s_min) / (s_max - s_min)
    out_path = SUBMISSIONS_DIR / f"submission_{name}.csv"
    priv_ids = ids[n_pub:]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "score"])
        for i, s in zip(priv_ids, priv_score_01):
            w.writerow([str(i), f"{s:.6f}"])
    return out_path


def main():
    loss_before = np.load(FEATURES_DIR / "loss_before.npy")
    loss_after = np.load(FEATURES_DIR / "loss_after.npy")
    print(f"loaded loss_before {loss_before.shape}  loss_after {loss_after.shape}",
          flush=True)
    print(f"loss_before  min={loss_before.min():.4f}  max={loss_before.max():.4f}  "
          f"mean={loss_before.mean():.4f}  std={loss_before.std():.4f}", flush=True)
    print(f"loss_after   min={loss_after.min():.4f}  max={loss_after.max():.4f}  "
          f"mean={loss_after.mean():.4f}  std={loss_after.std():.4f}", flush=True)

    combined = load_combined()
    n_pub = combined.n_pub
    ids = combined.ids
    pub_membership = np.asarray(load_pub().membership, dtype=int)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    lb = np.maximum(loss_before, EPS)
    la = np.maximum(loss_after, EPS)

    variants = {
        "adv_raw_diff":     loss_before - loss_after,
        "adv_neg_after":    -loss_after,
        "adv_ratio":        -la / lb,
        "adv_log_ratio":    -(np.log(la) - np.log(lb)),
        "adv_rel_increase": -(la - lb) / lb,
        "adv_log_after":    -np.log(la),
    }

    print(f"\n=== Pub TPR @ 5%FPR for adversarial-score variants ===", flush=True)
    for name, score in variants.items():
        tpr = tpr_at_fpr(score[:n_pub], pub_membership)
        out = write_submission(name, score, n_pub, ids)
        print(f"  {name:18s}  pub TPR={tpr:.4f}  → {out.name}", flush=True)


if __name__ == "__main__":
    main()
