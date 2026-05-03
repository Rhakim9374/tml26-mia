"""Sweep statistical-test variants on saved logit features.

Loads checkpoints/logit_features/* (φ_target, μ_in, μ_out, σ_in, σ_out per
sample on the combined pool) and computes several membership scores:

  * LiRA fixed-σ  (Carlini eq. 4)
        log N(φ_target | μ_IN, σ_IN_global)
      − log N(φ_target | μ_OUT, σ_OUT_global)
    The current strongest single-attack baseline.

  * RMIA-style    (Zarifzadeh 2023, simplified)
        score = φ_target − μ_OUT     # log Pr(x|target)/Pr(x|population)
    Calibration via population OUT mean only — no IN-Gaussian dependence.

  * Z-score OUT   (Sablayrolles 2019, "Yeom"-style with shadow calibration)
        score = (φ_target − μ_OUT) / σ_OUT_global
    Same direction as RMIA but normalised by OUT spread.

Each variant gets its own submission CSV. Pick the one with best pub TPR
(but mind that ranking-on-pub overfits the choice; gap to logit-LiRA tells
you how much).

Usage:
    condor_submit mia_grad.sub \\
        -append "script=scripts/score_lira_variants.py" \\
        -append "tag=lira_variants" -queue 1
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_combined, load_pub
from src.eval import tpr_at_fpr

FEATURES_DIR = ROOT / "checkpoints" / "logit_features"
SUBMISSIONS_DIR = ROOT / "submissions"
SIGMOID_CLIP = 50.0


def write_submission(name: str, score: np.ndarray, n_pub: int, ids: list):
    priv_score_logits = score[n_pub:]
    priv_ids = ids[n_pub:]
    # Sigmoid the priv portion into [0,1]; rank-preserving.
    priv_score = 1.0 / (1.0 + np.exp(-np.clip(priv_score_logits,
                                              -SIGMOID_CLIP, SIGMOID_CLIP)))
    out_path = SUBMISSIONS_DIR / f"submission_{name}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "score"])
        for i, s in zip(priv_ids, priv_score):
            w.writerow([str(i), f"{s:.6f}"])
    return out_path


def gauss_log_pdf(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def main():
    if not (FEATURES_DIR / "phi_target.npy").exists():
        sys.exit(f"Missing features in {FEATURES_DIR}. Run score_online_lira.py first.")

    phi_target = np.load(FEATURES_DIR / "phi_target.npy")
    mu_in = np.load(FEATURES_DIR / "mu_in.npy")
    mu_out = np.load(FEATURES_DIR / "mu_out.npy")
    sigma_in = float(np.load(FEATURES_DIR / "sigma_in.npy"))
    sigma_out = float(np.load(FEATURES_DIR / "sigma_out.npy"))
    print(f"Loaded features: {phi_target.shape[0]} samples", flush=True)
    print(f"σ_in={sigma_in:.4f}  σ_out={sigma_out:.4f}", flush=True)

    combined = load_combined()
    n_pub = combined.n_pub
    ids = combined.ids
    pub_membership = np.asarray(load_pub().membership, dtype=int)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Variant 1: LiRA fixed-σ (recompute from saved μs/σs).
    lira = (gauss_log_pdf(phi_target, mu_in, sigma_in) -
            gauss_log_pdf(phi_target, mu_out, sigma_out))

    # Variant 2: RMIA-style (mean-shift = φ − μ_OUT).
    rmia = phi_target - mu_out

    # Variant 3: Z-score OUT.
    zscore_out = (phi_target - mu_out) / sigma_out

    # Variant 4: Z-score IN-vs-OUT difference.
    zscore_diff = ((phi_target - mu_in) / sigma_in
                   - (phi_target - mu_out) / sigma_out)

    print(f"\n=== Pub TPR @ 5%FPR for each variant ===", flush=True)
    variants = {
        "lira_multiaug6":  lira,
        "rmia_meanshift":  rmia,
        "zscore_out":      zscore_out,
        "zscore_indiff":   zscore_diff,
    }
    for name, score in variants.items():
        tpr = tpr_at_fpr(score[:n_pub], pub_membership)
        out = write_submission(name, score, n_pub, ids)
        print(f"  {name:18s}  pub TPR={tpr:.4f}  → {out.name}", flush=True)

    print("\nTo submit the best by pub TPR:")
    print("  cp submissions/submission_<name>.csv submissions/submission.csv")
    print("  python3 -m src.submit --tag <name>")


if __name__ == "__main__":
    main()
