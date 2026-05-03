"""Combine shard outputs from score_grad_lira_lean.py and score them.

Loads checkpoints/grad_features_lean/{target.pt, shard_*.pt}, concatenates
shadow features across shards, runs per-layer fixed-σ LiRA, prints pub TPR
diagnostic, writes submissions/submission_grad_lean.csv. Pure NumPy after
loading — no GPU needed (run on the conduit head node).

Usage (after all shard jobs from score_grad_lira_lean.py have written their
files into checkpoints/grad_features_lean/):
    python scripts/combine_grad_features.py
    cp submissions/submission_grad_lean.csv submissions/submission.csv
    python -m src.submit --tag grad_lira_lean_n512
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

FEATURES_DIR = ROOT / "checkpoints" / "grad_features_lean"
TARGET_PATH = FEATURES_DIR / "target.pt"
COMBINED_PATH = FEATURES_DIR / "combined.pt"
OUT_PATH = ROOT / "submissions" / "submission_grad_lean.csv"
SIGMA_FLOOR = 1e-6
SIGMOID_CLIP = 50.0


def gauss_log_pdf(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def main():
    if not TARGET_PATH.exists():
        sys.exit(f"Missing {TARGET_PATH}. Run score_grad_lira_lean.py with --shard 0 N first.")

    tgt = torch.load(TARGET_PATH, weights_only=False)
    g_target = tgt["g_target"]                               # (n_total, L)
    param_names = tgt["param_names"]
    n_pub, n_priv = tgt["n_pub"], tgt["n_priv"]
    ids = tgt["ids"]
    pub_membership = tgt["pub_membership"]
    n_layers = len(param_names)
    n_total = n_pub + n_priv

    shard_files = sorted(FEATURES_DIR.glob("shard_*_of_*.pt"))
    if not shard_files:
        sys.exit(f"No shard files in {FEATURES_DIR}")
    shards = [torch.load(f, weights_only=False) for f in shard_files]

    # Sanity: all shards must agree on shard_n + n_pub/n_priv + param_names.
    shard_n = shards[0]["shard_n"]
    seen_ks = set()
    for s, f in zip(shards, shard_files):
        if s["shard_n"] != shard_n:
            sys.exit(f"{f} reports shard_n={s['shard_n']}, expected {shard_n}")
        if s["n_pub"] != n_pub or s["n_priv"] != n_priv:
            sys.exit(f"{f} pool sizes mismatch target.pt")
        if list(s["param_names"]) != list(param_names):
            sys.exit(f"{f} param_names mismatch target.pt")
        seen_ks.add(s["shard_k"])
    if seen_ks != set(range(shard_n)):
        sys.exit(f"Shard ks {sorted(seen_ks)} != expected {list(range(shard_n))}")

    # Concatenate in shard-K order so seed ordering matches all_ckpts[K::N] interleaving.
    shards.sort(key=lambda s: s["shard_k"])
    g_shadow = np.concatenate([s["g_shadow"] for s in shards], axis=0)
    in_masks = np.concatenate([s["in_masks"] for s in shards], axis=0)
    n_shadows = g_shadow.shape[0]
    print(f"Combined {len(shards)} shards → {n_shadows} shadows × "
          f"{n_total} samples × {n_layers} layers", flush=True)

    # Per-layer fixed-σ LiRA on the combined pool.
    out_masks = ~in_masks
    in_g = np.where(in_masks[..., None], g_shadow, np.nan)
    out_g = np.where(out_masks[..., None], g_shadow, np.nan)
    mu_in = np.nanmean(in_g, axis=0)
    mu_out = np.nanmean(out_g, axis=0)
    sigma_in_global = np.maximum(np.nanstd(in_g, axis=(0, 1)), SIGMA_FLOOR)
    sigma_out_global = np.maximum(np.nanstd(out_g, axis=(0, 1)), SIGMA_FLOOR)

    in_counts = in_masks.sum(axis=0)
    print(f"\nIN-shadow count per sample (combined pool): "
          f"min={in_counts.min()} median={int(np.median(in_counts))} "
          f"max={in_counts.max()}", flush=True)

    log_lr = (gauss_log_pdf(g_target, mu_in, sigma_in_global) -
              gauss_log_pdf(g_target, mu_out, sigma_out_global))   # (N, L)
    score = log_lr.sum(axis=1)

    pub_tpr = tpr_at_fpr(score[:n_pub], pub_membership)
    print(f"\n=== TPR@5%FPR on pub (per-layer fixed-σ LiRA, lean grad-norms) ===",
          flush=True)
    print(f"  combined (sum over {n_layers} layers): {pub_tpr:.4f}", flush=True)
    per_layer_tpr = np.array([tpr_at_fpr(log_lr[:n_pub, j], pub_membership)
                              for j in range(n_layers)])
    top = np.argsort(-per_layer_tpr)[:10]
    print("  top-10 individual layer TPRs:", flush=True)
    for j in top:
        print(f"    {param_names[j]:50s}  TPR={per_layer_tpr[j]:.4f}", flush=True)

    # Save the merged feature tensor for future iteration on combination logic.
    # pickle_protocol=4: g_shadow is (512, 28000, 62) ≈ 3.5GB which serializes
    # to a single string > 4GB under default protocol. Protocol 4 supports it.
    torch.save(
        {
            "g_target": g_target,
            "g_shadow": g_shadow,
            "in_masks": in_masks,
            "param_names": param_names,
            "ids": ids,
            "n_pub": n_pub,
            "n_priv": n_priv,
            "pub_membership": pub_membership,
        },
        COMBINED_PATH,
        pickle_protocol=4,
    )
    print(f"\nSaved merged features → {COMBINED_PATH}", flush=True)

    # Priv submission CSV.
    score_priv = 1.0 / (1.0 + np.exp(-np.clip(score[n_pub:],
                                              -SIGMOID_CLIP, SIGMOID_CLIP)))
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
    print("\nTo submit:", flush=True)
    print("  cp submissions/submission_grad_lean.csv submissions/submission.csv",
          flush=True)
    print("  python3 -m src.submit --tag grad_lira_lean_n512", flush=True)


if __name__ == "__main__":
    main()
