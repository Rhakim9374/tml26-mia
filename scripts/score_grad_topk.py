"""Top-K layer combination sweep on saved grad-LiRA features.

Loads checkpoints/grad_features_lean/combined.pt and tries summing the top-K
layers (ranked by individual pub TPR @ 5% FPR) for several K. The combined
sum-over-62-layers score (0.0646) underperforms several individual layers
(>0.07), so naive sum is diluting signal — top-K should recover it.

Writes one submission CSV per K to submissions/submission_grad_topK.csv so
you can A/B-submit. Pure NumPy after loading; no GPU required (run via the
docker container though, since head node has no torch/numpy).

Caveat: ranking by pub TPR and selecting top-K, then re-evaluating on pub,
overfits the layer choice to pub. The priv leaderboard score is the honest
test. Layer ordering is broadly stable (late layers carry MIA signal),
so the overfit penalty should be small relative to the gain.

Usage:
    condor_submit mia_grad.sub \\
        -append "script=scripts/score_grad_topk.py" \\
        -append "tag=grad_topk" -queue 1
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

FEATURES_PATH = ROOT / "checkpoints" / "grad_features_lean" / "combined.pt"
SUBMISSIONS_DIR = ROOT / "submissions"
SIGMA_FLOOR = 1e-6
SIGMOID_CLIP = 50.0
TOP_KS = [1, 3, 5, 10, 15, 20, 30, 62]


def gauss_log_pdf(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def main():
    if not FEATURES_PATH.exists():
        sys.exit(f"Missing {FEATURES_PATH}. Run combine_grad_features.py first.")

    bundle = torch.load(FEATURES_PATH, weights_only=False)
    g_target = bundle["g_target"]                    # (n_total, L)
    g_shadow = bundle["g_shadow"]                    # (n_shadows, n_total, L)
    in_masks = bundle["in_masks"]                    # (n_shadows, n_total)
    param_names = bundle["param_names"]
    n_pub, n_priv = bundle["n_pub"], bundle["n_priv"]
    pub_membership = bundle["pub_membership"]
    ids = bundle["ids"]
    n_layers = len(param_names)
    print(f"Loaded {g_shadow.shape[0]} shadows × {g_shadow.shape[1]} samples "
          f"× {n_layers} layers", flush=True)

    # Recompute per-layer fixed-σ log-LR on combined pool (same as combine script).
    out_masks = ~in_masks
    in_g = np.where(in_masks[..., None], g_shadow, np.nan)
    out_g = np.where(out_masks[..., None], g_shadow, np.nan)
    mu_in = np.nanmean(in_g, axis=0)
    mu_out = np.nanmean(out_g, axis=0)
    sigma_in = np.maximum(np.nanstd(in_g, axis=(0, 1)), SIGMA_FLOOR)
    sigma_out = np.maximum(np.nanstd(out_g, axis=(0, 1)), SIGMA_FLOOR)
    log_lr = (gauss_log_pdf(g_target, mu_in, sigma_in) -
              gauss_log_pdf(g_target, mu_out, sigma_out))     # (n_total, L)

    # Rank layers by individual pub TPR (highest first).
    per_layer_tpr = np.array([tpr_at_fpr(log_lr[:n_pub, j], pub_membership)
                              for j in range(n_layers)])
    order = np.argsort(-per_layer_tpr)
    print("\nLayer ranking by individual pub TPR @ 5%FPR:")
    for rank, j in enumerate(order[:20]):
        print(f"  #{rank+1:2d}  TPR={per_layer_tpr[j]:.4f}  {param_names[j]}",
              flush=True)

    # Top-K sweep + per-K submission CSV.
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n=== top-K sum sweep on pub (sum of top-K layer log-LRs) ===")
    priv_ids = ids[n_pub:]
    for K in TOP_KS:
        if K > n_layers:
            continue
        chosen = order[:K]
        score = log_lr[:, chosen].sum(axis=1)
        pub_tpr = tpr_at_fpr(score[:n_pub], pub_membership)
        score_priv = 1.0 / (1.0 + np.exp(-np.clip(score[n_pub:],
                                                  -SIGMOID_CLIP, SIGMOID_CLIP)))
        out_path = SUBMISSIONS_DIR / f"submission_grad_top{K}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "score"])
            for i, s in zip(priv_ids, score_priv):
                w.writerow([str(i), f"{s:.6f}"])
        print(f"  K={K:2d}  pub TPR={pub_tpr:.4f}  → {out_path.name}",
              flush=True)

    print("\nTo submit one (e.g. the K with best pub TPR):")
    print("  cp submissions/submission_grad_topK.csv submissions/submission.csv")
    print("  python3 -m src.submit --tag grad_lira_topK_n512")


if __name__ == "__main__":
    main()
