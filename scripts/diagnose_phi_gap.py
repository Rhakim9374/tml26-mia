"""Diagnose the shadow-vs-target φ distribution gap.

LiRA assumes shadows produce φ values from the same distribution the target
does (per sample). If shadows overfit more than the target — which our recon
suggests — the per-sample LiRA Gaussians are misaligned and per-sample
log-LR is computed against a wrong reference. This script quantifies the gap
from already-saved features so we know what to fix.

Inputs (from score_online_lira.py with feature saving):
  checkpoints/logit_features/phi_target.npy
  checkpoints/logit_features/mu_in.npy   (per-sample mean φ across IN shadows)
  checkpoints/logit_features/mu_out.npy  (per-sample mean φ across OUT shadows)

What it prints:
  * Percentile comparison overall (target vs μ_IN vs μ_OUT)
  * Same comparison split by pub membership label
  * Per-class breakdown
  * "Effective gap" = mean(target φ for members) − mean(μ_IN for members).
    Negative gap = shadows overconfident on what would be members → overfit.

Usage:
    condor_submit mia_grad.sub \\
        -append "script=scripts/diagnose_phi_gap.py" \\
        -append "tag=phi_gap" -queue 1
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_combined, load_pub
from src.model import NUM_CLASSES

FEATURES_DIR = ROOT / "checkpoints" / "logit_features"


def pct_str(a: np.ndarray, qs=(1, 5, 25, 50, 75, 95, 99)) -> str:
    return "  ".join(f"p{q}={np.nanpercentile(a, q):+6.2f}" for q in qs)


def main():
    phi_target = np.load(FEATURES_DIR / "phi_target.npy")
    mu_in = np.load(FEATURES_DIR / "mu_in.npy")
    mu_out = np.load(FEATURES_DIR / "mu_out.npy")
    print(f"Loaded {phi_target.shape[0]} samples", flush=True)

    combined = load_combined()
    n_pub = combined.n_pub
    pub_labels = np.asarray(combined.labels[:n_pub], dtype=int)
    pub_mem = np.asarray(load_pub().membership, dtype=int)

    # -- 1) Overall distribution comparison (combined pool) --
    print("\n=== φ distribution percentiles (combined pool) ===")
    print(f"  target φ : {pct_str(phi_target)}")
    print(f"  μ_IN     : {pct_str(mu_in)}")
    print(f"  μ_OUT    : {pct_str(mu_out)}")

    # -- 2) Split by pub membership --
    pub_target = phi_target[:n_pub]
    pub_mu_in = mu_in[:n_pub]
    pub_mu_out = mu_out[:n_pub]

    members = pub_mem == 1
    nonmem = pub_mem == 0
    print(f"\n=== pub members ({members.sum()}) ===")
    print(f"  target φ : {pct_str(pub_target[members])}")
    print(f"  μ_IN     : {pct_str(pub_mu_in[members])}")
    print(f"  μ_OUT    : {pct_str(pub_mu_out[members])}")
    print(f"\n=== pub non-members ({nonmem.sum()}) ===")
    print(f"  target φ : {pct_str(pub_target[nonmem])}")
    print(f"  μ_IN     : {pct_str(pub_mu_in[nonmem])}")
    print(f"  μ_OUT    : {pct_str(pub_mu_out[nonmem])}")

    # -- 3) Effective overfitting gap --
    gap_members = pub_target[members].mean() - pub_mu_in[members].mean()
    gap_nonmem = pub_target[nonmem].mean() - pub_mu_out[nonmem].mean()
    print(f"\n=== overfitting gaps (target − shadow average) ===")
    print(f"  members:     mean(target φ) − mean(μ_IN)  = {gap_members:+.3f}")
    print(f"  non-members: mean(target φ) − mean(μ_OUT) = {gap_nonmem:+.3f}")
    print(f"  Negative gap → shadows are MORE overconfident than target")
    print(f"  → shadow training recipe is over-fitting; per-sample LiRA")
    print(f"    Gaussians sit at the wrong location for target's φ.")

    # -- 4) Per-class breakdown --
    print(f"\n=== per-class mean φ comparison (pub) ===")
    print(f"{'class':>5} | {'target_mem':>10} {'μ_IN_mem':>10} {'gap_mem':>8} | "
          f"{'target_non':>10} {'μ_OUT_non':>10} {'gap_non':>8}")
    for c in range(NUM_CLASSES):
        m_mask = members & (pub_labels == c)
        n_mask = nonmem & (pub_labels == c)
        if m_mask.sum() == 0 or n_mask.sum() == 0:
            continue
        t_m = pub_target[m_mask].mean()
        mu_m = pub_mu_in[m_mask].mean()
        t_n = pub_target[n_mask].mean()
        mu_n = pub_mu_out[n_mask].mean()
        print(f"{c:>5} | {t_m:>+10.3f} {mu_m:>+10.3f} {t_m - mu_m:>+8.3f} | "
              f"{t_n:>+10.3f} {mu_n:>+10.3f} {t_n - mu_n:>+8.3f}")

    # -- 5) Suggest direction --
    print(f"\n=== suggested recipe modifications ===")
    if gap_members < -3:
        print("  Large negative gap on members → shadows are SIGNIFICANTLY more")
        print("  overfit than target. Try reducing shadow overfitting via:")
        print("    (a) label smoothing (LS=0.05 or 0.1) — directly caps φ scale")
        print("    (b) fewer epochs (50 instead of 100) — stops earlier")
        print("    (c) heavier weight decay (1e-3 instead of 5e-4)")
        print("    (d) stronger augmentations (RandomErasing, ColorJitter)")
        print("  Run train_shadow_sweep.py with each variant; pick the one")
        print("  whose μ_IN best matches target φ on members.")
    elif gap_members > 3:
        print("  Large positive gap → shadows are UNDER-fit. Train more epochs.")
    else:
        print("  Small gap. Recipe is already well-matched.")


if __name__ == "__main__":
    main()
