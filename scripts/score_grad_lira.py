"""White-box gradient-norm LiRA scoring (per-layer features).

For each (sample, model), compute the per-sample gradient ∇_θ L(x, y; θ),
then take its L2 norm *per named parameter tensor* (≈60 features for
ResNet-18). Take log of each (grad norms are log-normal-ish); average across
augmentations (identity + horizontal flip). Per-sample features are computed
with `torch.func.vmap(grad(...))` — one fused forward+backward per batch,
no per-sample Python loop.

Shadows trained on the COMBINED pub+priv pool (see src/train.py), so every
sample (pub OR priv) is IN for ~half the shadows and OUT for the other half.
That gives priv per-sample IN/OUT statistics, which is what makes online
LiRA actually generalize from the pub TPR sanity check to the priv submission.

Per-layer fixed-σ LiRA (Carlini eq. 4):
  μ_IN(x), μ_OUT(x) per (sample, layer) from the per-shadow IN/OUT mask.
  σ_IN, σ_OUT global per-layer scalars.
  log-LR_l(x) = log N(g_l(x) | μ_IN_l(x), σ_IN_l) − log N(g_l(x) | μ_OUT_l(x), σ_OUT_l).
  Combined score(x) = sum_l log-LR_l(x) (independence assumption — replaceable
  with a learned combiner once features are saved).

Outputs:
  submissions/submission_grad.csv         (separate from logit-LiRA's
                                           submission.csv, no clobbering)
  checkpoints/grad_features/features.pt   (raw per-layer features so a
                                           learned combiner / different
                                           scoring can be tried without
                                           re-running the 5–6h sweep)

Run with:
    condor_submit mia_grad.sub -queue 1
    # when satisfied with pub TPR:
    cp submissions/submission_grad.csv submissions/submission.csv
    python3 -m src.submit --tag grad_lira_combined_n512_aug2
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_combined, load_pub, MODEL_PATH
from src.eval import tpr_at_fpr
from src.model import build_model, load_target

CHECKPOINTS_DIR = ROOT / "checkpoints"
FEATURES_PATH = CHECKPOINTS_DIR / "grad_features" / "features.pt"
OUT_PATH = ROOT / "submissions" / "submission_grad.csv"
SIGMA_FLOOR = 1e-6
SIGMOID_CLIP = 50.0
GRAD_BATCH = 64  # vmap holds per-sample grads for a whole batch on GPU


def preload(pool, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply standard transform + stack the whole combined pool into
    (N, 3, 32, 32) imgs + (N,) labels on `device`. ~11 MB for 28k samples —
    fits easily and removes DataLoader/worker overhead from the inner loop."""
    imgs = torch.stack([pool[i][1] for i in range(len(pool))])
    labels = torch.tensor([pool[i][2] for i in range(len(pool))],
                          dtype=torch.long)
    return imgs.to(device), labels.to(device)


def collect_grad_log_norms(model, imgs: torch.Tensor, labels: torch.Tensor,
                           param_names: list[str], device: str) -> np.ndarray:
    """(N, L) per-sample per-named-parameter log L2 grad norm, mean over augs.

    Uses vmap(grad) for per-sample gradients in one fused backward — no Python
    per-sample loop. Model must be in eval() mode so BN uses running stats and
    single-sample functional calls are well-defined under vmap.
    """
    n = imgs.shape[0]
    n_layers = len(param_names)
    out = np.zeros((n, n_layers), dtype=np.float32)

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_fn(p, x, y):
        pred = functional_call(model, {**p, **buffers}, (x.unsqueeze(0),))
        return F.cross_entropy(pred, y.unsqueeze(0))

    grad_fn = vmap(grad(loss_fn, argnums=0), in_dims=(None, 0, 0))

    for start in range(0, n, GRAD_BATCH):
        x_batch = imgs[start:start + GRAD_BATCH]
        y_batch = labels[start:start + GRAD_BATCH]
        nb = x_batch.shape[0]
        accum = torch.zeros(nb, n_layers, device=device, dtype=torch.float32)
        for aug in (x_batch, torch.flip(x_batch, dims=[-1])):
            grads = grad_fn(params, aug, y_batch)
            for j, name in enumerate(param_names):
                accum[:, j] += grads[name].flatten(1).norm(dim=1).float() \
                    .clamp_min(1e-12).log()
        accum /= 2.0  # number of augs
        out[start:start + nb] = accum.cpu().numpy()
    return out


def gauss_log_pdf(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def main():
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout,
          flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    combined = load_combined()
    n_pub, n_priv, n_total = combined.n_pub, combined.n_priv, len(combined)
    print(f"combined pool: n_pub={n_pub} n_priv={n_priv} total={n_total}",
          flush=True)
    imgs, labels = preload(combined, device)
    print(f"preloaded {tuple(imgs.shape)} on {device}", flush=True)

    target = load_target(MODEL_PATH, map_location=device).to(device)
    target.eval()
    param_names = [n for n, _ in target.named_parameters()]
    n_layers = len(param_names)
    print(f"per-layer features: {n_layers} named-parameter tensors", flush=True)

    print("Forward+backward: target on combined pool", flush=True)
    g_target = collect_grad_log_norms(target, imgs, labels, param_names, device)
    del target
    if device == "cuda":
        torch.cuda.empty_cache()

    ckpts = sorted(c for c in CHECKPOINTS_DIR.glob("shadow_*.pt")
                   if "_in_idx" not in c.name)
    if not ckpts:
        sys.exit(f"No shadow checkpoints (shadow_*.pt) in {CHECKPOINTS_DIR}")
    n_shadows = len(ckpts)
    print(f"Found {n_shadows} shadow checkpoints", flush=True)

    g_shadow = np.zeros((n_shadows, n_total, n_layers), dtype=np.float32)
    in_masks = np.zeros((n_shadows, n_total), dtype=bool)

    shadow = build_model().to(device)
    for k, ckpt in enumerate(ckpts):
        seed_str = ckpt.stem.split("_")[-1]
        idx_path = CHECKPOINTS_DIR / f"shadow_{seed_str}_in_idx.pt"
        in_idx = torch.load(idx_path, weights_only=True).numpy()
        in_masks[k, in_idx] = True

        shadow.load_state_dict(torch.load(ckpt, map_location=device,
                                          weights_only=True))
        shadow.eval()
        g_shadow[k] = collect_grad_log_norms(shadow, imgs, labels,
                                             param_names, device)
        if (k + 1) % 16 == 0 or k == n_shadows - 1:
            print(f"  scored shadow {k+1}/{n_shadows}", flush=True)

    pub_membership = np.asarray(load_pub().membership, dtype=int)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "g_target": g_target,                # (n_total, L)
            "g_shadow": g_shadow,                # (n_shadows, n_total, L)
            "in_masks": in_masks,                # (n_shadows, n_total)
            "param_names": param_names,
            "ids": list(combined.ids),
            "labels": list(combined.labels),
            "n_pub": n_pub,
            "n_priv": n_priv,
            "pub_membership": pub_membership,
        },
        FEATURES_PATH,
    )
    print(f"Saved raw features → {FEATURES_PATH}", flush=True)

    # ============== Per-layer fixed-σ LiRA across the combined pool ==============
    out_masks = ~in_masks
    in_g = np.where(in_masks[..., None], g_shadow, np.nan)         # (S, N, L)
    out_g = np.where(out_masks[..., None], g_shadow, np.nan)       # (S, N, L)
    mu_in = np.nanmean(in_g, axis=0)                                # (N, L)
    mu_out = np.nanmean(out_g, axis=0)                              # (N, L)
    sigma_in_global = np.maximum(np.nanstd(in_g, axis=(0, 1)),
                                 SIGMA_FLOOR)                       # (L,)
    sigma_out_global = np.maximum(np.nanstd(out_g, axis=(0, 1)),
                                  SIGMA_FLOOR)                      # (L,)

    in_counts = in_masks.sum(axis=0)
    print(f"\nIN-shadow count per sample (combined pool): "
          f"min={in_counts.min()} median={int(np.median(in_counts))} "
          f"max={in_counts.max()}", flush=True)

    log_lr = (gauss_log_pdf(g_target, mu_in, sigma_in_global) -
              gauss_log_pdf(g_target, mu_out, sigma_out_global))    # (N, L)
    score = log_lr.sum(axis=1)                                       # (N,)

    # Pub portion: TPR sanity check.
    pub_tpr_combined = tpr_at_fpr(score[:n_pub], pub_membership)
    print(f"\n=== TPR@5%FPR on pub (per-layer fixed-σ LiRA on log-grad-norms) ===",
          flush=True)
    print(f"  combined (sum over {n_layers} layers): {pub_tpr_combined:.4f}",
          flush=True)
    per_layer_tpr = np.array([tpr_at_fpr(log_lr[:n_pub, j], pub_membership)
                              for j in range(n_layers)])
    top = np.argsort(-per_layer_tpr)[:10]
    print("  top-10 individual layer TPRs:", flush=True)
    for j in top:
        print(f"    {param_names[j]:50s}  TPR={per_layer_tpr[j]:.4f}",
              flush=True)

    # Priv portion: write submission.
    score_priv = 1.0 / (1.0 + np.exp(-np.clip(score[n_pub:],
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
    print("\nTo submit:", flush=True)
    print("  cp submissions/submission_grad.csv submissions/submission.csv",
          flush=True)
    print("  python3 -m src.submit --tag grad_lira_combined_n512_aug2",
          flush=True)


if __name__ == "__main__":
    main()
