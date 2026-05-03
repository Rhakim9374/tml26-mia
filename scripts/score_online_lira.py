"""Online LiRA scoring with augmentation queries.

For each (sample, model), φ = log(p / (1-p)) on the correct class, averaged
over augmentations: identity + horizontal flip. Computed directly from logits
as φ_y = z_y − logsumexp(z_{j≠y}) — no clamping or EPS, no overflow when
softmax saturates at 1.0.

Shadows trained on the COMBINED pub+priv pool (see src/train.py), so every
sample (pub OR priv) is IN for ~half the shadows and OUT for the other half.
That gives us per-sample IN-Gaussians and OUT-Gaussians on priv at scoring
time — the prerequisite for online LiRA to actually work at submission. Pub
samples get the same per-sample treatment, so the pub TPR is a faithful
estimate of priv TPR (no class-conditional fallback hacks).

Per-layer-of-φ this is just one scalar per sample, so the tensor is small:
  φ matrix: (n_shadows, n_combined) = (512, 28000) ≈ 115 MB at float64.

Scoring (Carlini fixed-σ LiRA, eq. 4):
  μ_IN(x), μ_OUT(x) per-sample from the per-shadow IN/OUT mask.
  σ_IN, σ_OUT global scalars (single σ per IN/OUT pool).
  log-LR(x) = log N(φ_target(x) | μ_IN(x), σ_IN) − log N(φ_target(x) | μ_OUT(x), σ_OUT).

Pub TPR at 5% FPR is reported as a sanity check; the submission CSV uses the
log-LR sigmoid-mapped to [0, 1] over the priv portion of the combined pool.

Run with:
    condor_submit mia.sub -append "script=scripts/score_online_lira.py" \\
                          -append "tag=score_online" -queue 1
    python -m src.submit --tag online_lira_combined_n512_aug2
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_combined, load_pub, predict_collate, MODEL_PATH
from src.eval import tpr_at_fpr
from src.model import build_model, load_target

CHECKPOINTS_DIR = ROOT / "checkpoints"
OUT_PATH = ROOT / "submissions" / "submission.csv"
SIGMA_FLOOR = 1e-6
SIGMOID_CLIP = 50.0  # avoid np.exp overflow on extreme log-LRs
DEFAULT_CKPT_PREFIX = "shadow"


def make_augs(imgs: torch.Tensor) -> list[torch.Tensor]:
    """Identity + hflip. Empirically tested 6-aug variant (id+hflip+4 corner
    crops) on baseline shadows — pub TPR REGRESSED from 0.0697 to 0.0624.
    The corner crops apparently disturb predictions in a way that hurts
    member/non-member ranking on 32x32 inputs, despite Carlini's paper
    suggesting otherwise. Sticking with the 2-aug version that works."""
    return [imgs, torch.flip(imgs, dims=[-1])]


@torch.no_grad()
def collect_phi(model, loader, n: int, device: str) -> np.ndarray:
    """Mean φ across augmentations, in dataset order. Computed from logits as
    z_y − logsumexp(z_{j≠y}) — equivalent to log(p/(1−p)) on the true class but
    numerically stable when softmax saturates."""
    phi = np.zeros(n, dtype=np.float64)
    pos = 0
    for _, imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        augs = make_augs(imgs)
        phi_sum = torch.zeros(imgs.shape[0], device=device, dtype=torch.float64)
        for aug in augs:
            logits = model(aug)
            z_y = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            masked = logits.scatter(1, labels.unsqueeze(1), float("-inf"))
            log_sum_other = torch.logsumexp(masked, dim=1)
            phi_sum += (z_y - log_sum_other).double()
        nb = imgs.shape[0]
        phi[pos:pos + nb] = (phi_sum / len(augs)).cpu().numpy()
        pos += nb
    return phi


def gauss_log_pdf(x, mu, sigma):
    """log N(x; μ, σ), elementwise. σ may be a scalar or array broadcastable to x."""
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_prefix", default=DEFAULT_CKPT_PREFIX,
                   help="Glob CHECKPOINTS_DIR/<prefix>_*.pt for shadow weights "
                        "(and <prefix>_NNNN_in_idx.pt for IN-masks). Use a "
                        "non-default prefix to score against a different shadow "
                        "family without mixing it with the baseline shadow_*.pt.")
    a = p.parse_args()
    prefix = a.ckpt_prefix

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  ckpt_prefix={prefix}", flush=True)

    combined = load_combined()
    n_pub, n_priv, n_total = combined.n_pub, combined.n_priv, len(combined)
    print(f"combined pool: n_pub={n_pub} n_priv={n_priv} total={n_total}",
          flush=True)
    loader = DataLoader(combined, batch_size=512, shuffle=False, num_workers=2,
                        collate_fn=predict_collate)

    print("Forward pass: target on combined pool", flush=True)
    target = load_target(MODEL_PATH, map_location=device).to(device)
    phi_target = collect_phi(target, loader, n_total, device)
    del target
    if device == "cuda":
        torch.cuda.empty_cache()

    ckpts = sorted(c for c in CHECKPOINTS_DIR.glob(f"{prefix}_*.pt")
                   if "_in_idx" not in c.name)
    if not ckpts:
        sys.exit(f"No shadow checkpoints ({prefix}_*.pt) in {CHECKPOINTS_DIR}")
    n_shadows = len(ckpts)
    print(f"Found {n_shadows} shadow checkpoints (prefix={prefix})", flush=True)

    phi_shadow = np.zeros((n_shadows, n_total), dtype=np.float64)
    in_masks = np.zeros((n_shadows, n_total), dtype=bool)

    shadow = build_model().to(device)
    for k, ckpt in enumerate(ckpts):
        seed_str = ckpt.stem.split("_")[-1]
        idx_path = CHECKPOINTS_DIR / f"{prefix}_{seed_str}_in_idx.pt"
        in_idx = torch.load(idx_path, weights_only=True).numpy()
        in_masks[k, in_idx] = True

        shadow.load_state_dict(torch.load(ckpt, map_location=device,
                                          weights_only=True))
        shadow.eval()
        phi_shadow[k] = collect_phi(shadow, loader, n_total, device)
        if (k + 1) % 32 == 0 or k == n_shadows - 1:
            print(f"  scored shadow {k+1}/{n_shadows}", flush=True)

    # Per-sample IN/OUT μ across shadows; global IN/OUT σ across (sample, shadow).
    out_masks = ~in_masks
    in_phi = np.where(in_masks, phi_shadow, np.nan)
    out_phi = np.where(out_masks, phi_shadow, np.nan)
    mu_in = np.nanmean(in_phi, axis=0)                              # (n_total,)
    mu_out = np.nanmean(out_phi, axis=0)                            # (n_total,)
    sigma_in = max(float(np.nanstd(in_phi)), SIGMA_FLOOR)
    sigma_out = max(float(np.nanstd(out_phi)), SIGMA_FLOOR)

    in_counts = in_masks.sum(axis=0)
    print(f"\nIN-shadow count per sample (combined pool): "
          f"min={in_counts.min()} median={int(np.median(in_counts))} "
          f"max={in_counts.max()}", flush=True)
    print(f"σ_in_global={sigma_in:.4f}  σ_out_global={sigma_out:.4f}",
          flush=True)

    log_lr = (gauss_log_pdf(phi_target, mu_in, sigma_in) -
              gauss_log_pdf(phi_target, mu_out, sigma_out))

    # Save the full per-sample feature set so downstream variant scripts
    # (RMIA, Z-score, ensemble) can run without re-doing the ~90-min sweep.
    # Indices in every saved array: [0, n_pub) = pub; [n_pub, end) = priv.
    FEATURES_DIR = ROOT / "checkpoints" / "logit_features"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_DIR / "log_lr.npy", log_lr)
    np.save(FEATURES_DIR / "phi_target.npy", phi_target)
    np.save(FEATURES_DIR / "mu_in.npy", mu_in)
    np.save(FEATURES_DIR / "mu_out.npy", mu_out)
    np.save(FEATURES_DIR / "sigma_in.npy", np.array(sigma_in))
    np.save(FEATURES_DIR / "sigma_out.npy", np.array(sigma_out))
    print(f"Saved logit features → {FEATURES_DIR}", flush=True)

    # Pub portion: TPR sanity check using known membership labels.
    pub_membership = np.asarray(load_pub().membership, dtype=int)
    pub_tpr = tpr_at_fpr(log_lr[:n_pub], pub_membership)
    print(f"\n=== TPR@5%FPR on pub (per-sample fixed-σ LiRA) ===", flush=True)
    print(f"  {pub_tpr:.4f}", flush=True)

    # Priv portion: write submission.
    score_priv = 1.0 / (1.0 + np.exp(-np.clip(log_lr[n_pub:],
                                              -SIGMOID_CLIP, SIGMOID_CLIP)))
    priv_ids = combined.ids[n_pub:]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "score"])
        for i, s in zip(priv_ids, score_priv):
            w.writerow([str(i), f"{s:.6f}"])
    print(f"\nWrote {OUT_PATH}  rows={len(priv_ids)}  "
          f"score range=[{score_priv.min():.4f}, {score_priv.max():.4f}]",
          flush=True)


if __name__ == "__main__":
    main()
