"""Rank-average ensemble of logit-LiRA + grad-LiRA top-1 layer.

Combines two attacks that look at different things:
  - Logit-LiRA: φ = log(p / (1 − p)) on the correct class (output confidence)
  - Grad-LiRA top-1: log L2 norm of ∇_θ L for the strongest single layer
                     (parameter-space curvature)

Each attack carries similar overall TPR (logit pub ≈ 0.07, grad top-1 pub
≈ 0.07) but ranks samples slightly differently. If their disagreements
correlate with the membership signal, rank-averaging picks up samples that
ONE attack misses but the OTHER catches.

Inputs (must already exist):
  checkpoints/logit_lira_loglr.npy             from score_online_lira.py
  checkpoints/grad_features_lean/combined.pt   from combine_grad_features.py

Outputs:
  submissions/submission_ensemble.csv

Usage:
    condor_submit mia_grad.sub \\
        -append "script=scripts/score_ensemble.py" \\
        -append "tag=ensemble" -queue 1
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval import tpr_at_fpr

LOGIT_PATH = ROOT / "checkpoints" / "logit_features" / "log_lr.npy"
GRAD_PATH = ROOT / "checkpoints" / "grad_features_lean" / "combined.pt"
OUT_PATH = ROOT / "submissions" / "submission_ensemble.csv"
SIGMA_FLOOR = 1e-6


def gauss_log_pdf(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def to_rank(x: np.ndarray) -> np.ndarray:
    """Convert array to per-element rank in [0, 1]. Higher x → higher rank."""
    order = np.argsort(x)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(len(x))
    return rank / max(len(x) - 1, 1)


def main():
    if not LOGIT_PATH.exists():
        sys.exit(f"Missing {LOGIT_PATH}. Run score_online_lira.py first.")
    if not GRAD_PATH.exists():
        sys.exit(f"Missing {GRAD_PATH}. Run combine_grad_features.py first.")

    logit_loglr = np.load(LOGIT_PATH)
    print(f"logit log-LR shape: {logit_loglr.shape}", flush=True)

    bundle = torch.load(GRAD_PATH, weights_only=False)
    g_target = bundle["g_target"]                   # (n_total, L)
    g_shadow = bundle["g_shadow"]                   # (n_shadows, n_total, L)
    in_masks = bundle["in_masks"]                   # (n_shadows, n_total)
    param_names = bundle["param_names"]
    n_pub, n_priv = bundle["n_pub"], bundle["n_priv"]
    n_total = n_pub + n_priv
    pub_membership = bundle["pub_membership"]
    ids = bundle["ids"]
    print(f"grad bundle: {g_shadow.shape[0]} shadows × {n_total} samples × "
          f"{len(param_names)} layers", flush=True)

    if logit_loglr.shape[0] != n_total:
        sys.exit(f"shape mismatch: logit log-LR has {logit_loglr.shape[0]} "
                 f"entries vs {n_total} from grad bundle")

    # Recompute per-layer log-LR, identify the top-1 layer by pub TPR.
    out_masks = ~in_masks
    in_g = np.where(in_masks[..., None], g_shadow, np.nan)
    out_g = np.where(out_masks[..., None], g_shadow, np.nan)
    mu_in = np.nanmean(in_g, axis=0)
    mu_out = np.nanmean(out_g, axis=0)
    sigma_in = np.maximum(np.nanstd(in_g, axis=(0, 1)), SIGMA_FLOOR)
    sigma_out = np.maximum(np.nanstd(out_g, axis=(0, 1)), SIGMA_FLOOR)
    grad_log_lr = (gauss_log_pdf(g_target, mu_in, sigma_in) -
                   gauss_log_pdf(g_target, mu_out, sigma_out))   # (n_total, L)

    per_layer_tpr = np.array([tpr_at_fpr(grad_log_lr[:n_pub, j], pub_membership)
                              for j in range(grad_log_lr.shape[1])])
    top1_idx = int(np.argmax(per_layer_tpr))
    print(f"grad top-1 layer: {param_names[top1_idx]} "
          f"(pub TPR={per_layer_tpr[top1_idx]:.4f})", flush=True)

    grad_top1 = grad_log_lr[:, top1_idx]            # (n_total,)

    # Sanity: pub TPR for each component alone.
    print(f"\nIndividual pub TPRs:")
    print(f"  logit-LiRA      : {tpr_at_fpr(logit_loglr[:n_pub], pub_membership):.4f}",
          flush=True)
    print(f"  grad-LiRA top-1 : {tpr_at_fpr(grad_top1[:n_pub], pub_membership):.4f}",
          flush=True)

    # Rank-average (parameter-free, robust to scale differences).
    logit_rank = to_rank(logit_loglr)
    grad_rank = to_rank(grad_top1)
    ensemble_rank = (logit_rank + grad_rank) / 2.0

    # Sweep mixing weights too — pick the best by pub TPR.
    print(f"\n=== Pub TPR for mix α·logit + (1−α)·grad_top1 (ranks averaged) ===")
    best_alpha, best_tpr, best_score = 0.5, -1.0, ensemble_rank
    for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        mix = alpha * logit_rank + (1 - alpha) * grad_rank
        tpr = tpr_at_fpr(mix[:n_pub], pub_membership)
        marker = ""
        if tpr > best_tpr:
            best_alpha, best_tpr, best_score = alpha, tpr, mix
            marker = "  ← best so far"
        print(f"  α={alpha:.1f}  pub TPR={tpr:.4f}{marker}", flush=True)

    print(f"\nBest mixing weight: α={best_alpha:.1f}  pub TPR={best_tpr:.4f}",
          flush=True)

    # Priv submission CSV (final scores ∈ [0,1] from the rank already).
    priv_ids = ids[n_pub:]
    score_priv = best_score[n_pub:]
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
    print(f"  cp submissions/submission_ensemble.csv submissions/submission.csv")
    print(f"  python3 -m src.submit --tag ensemble_logit_grad_a{best_alpha:.1f}")


if __name__ == "__main__":
    main()
