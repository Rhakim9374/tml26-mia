"""Online LiRA scoring with augmentation queries.

For each (sample, model), φ = log(p / (1 - p)) on the correct class, averaged
over augmentations: identity + horizontal flip.

Pub: per-sample IN-Gaussian and OUT-Gaussian fit over the 512 shadow φ values.
     score = log N(φ_target | IN) - log N(φ_target | OUT).
Priv: per-sample OUT-Gaussian (all shadows are OUT for priv); IN-Gaussian is
     class-conditional, pooled across all pub IN observations grouped by label.
     Same log-LR formula. Final score is sigmoid(log-LR) ∈ [0, 1] (rank-preserving).

Run after training shadows in `checkpoints/`:
    condor_submit mia.sub -append "script=scripts/score_online_lira.py" \\
                          -append "tag=score_online" -queue 1
    python -m src.submit --tag online_lira_n512_aug2
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_pub, load_priv, predict_collate, MODEL_PATH
from src.eval import tpr_at_fpr
from src.model import NUM_CLASSES, build_model, load_target

CHECKPOINTS_DIR = ROOT / "checkpoints"
OUT_PATH = ROOT / "submissions" / "submission.csv"
SIGMA_FLOOR = 1e-6
SIGMOID_CLIP = 50.0  # avoid np.exp overflow warnings on extreme log-LRs


def make_augs(imgs: torch.Tensor) -> list[torch.Tensor]:
    """Identity + horizontal flip. Cheap, captures most of Carlini's aug-query gain."""
    return [imgs, torch.flip(imgs, dims=[-1])]


@torch.no_grad()
def collect_phi(model, loader, n: int, device: str) -> np.ndarray:
    """Mean φ across augmentations, in dataset order.

    φ_y = z_y - logsumexp(z_{j≠y}) — equivalent to log(p / (1-p)) on the true
    class but computed directly from logits, so it never enters probability
    space. No clamping, no EPS, no overflow when softmax saturates at 1.0.
    """
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


def gauss_log_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """log N(x; μ, σ), elementwise."""
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    pub = load_pub()
    priv = load_priv()
    pub_loader = DataLoader(pub, batch_size=512, shuffle=False, num_workers=2,
                            collate_fn=predict_collate)
    priv_loader = DataLoader(priv, batch_size=512, shuffle=False, num_workers=2,
                             collate_fn=predict_collate)

    print("Forward pass: target", flush=True)
    target = load_target(MODEL_PATH, map_location=device).to(device)
    phi_target_pub = collect_phi(target, pub_loader, len(pub), device)
    phi_target_priv = collect_phi(target, priv_loader, len(priv), device)
    del target
    if device == "cuda":
        torch.cuda.empty_cache()

    ckpts = sorted(c for c in CHECKPOINTS_DIR.glob("pub_shadow_*.pt")
                   if "_in_idx" not in c.name)
    if not ckpts:
        sys.exit(f"No shadow checkpoints in {CHECKPOINTS_DIR}")
    n_shadows = len(ckpts)
    print(f"Found {n_shadows} shadow checkpoints", flush=True)

    phi_shadow_pub = np.zeros((n_shadows, len(pub)), dtype=np.float64)
    phi_shadow_priv = np.zeros((n_shadows, len(priv)), dtype=np.float64)
    in_masks_pub = np.zeros((n_shadows, len(pub)), dtype=bool)

    shadow = build_model().to(device)
    for k, ckpt in enumerate(ckpts):
        seed_str = ckpt.stem.split("_")[-1]
        idx_path = CHECKPOINTS_DIR / f"pub_shadow_{seed_str}_in_idx.pt"
        in_idx = torch.load(idx_path, weights_only=True).numpy()
        in_masks_pub[k, in_idx] = True

        shadow.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        shadow.eval()
        phi_shadow_pub[k] = collect_phi(shadow, pub_loader, len(pub), device)
        phi_shadow_priv[k] = collect_phi(shadow, priv_loader, len(priv), device)
        if (k + 1) % 32 == 0 or k == n_shadows - 1:
            print(f"  scored shadow {k+1}/{n_shadows}", flush=True)

    # Per-pub-sample IN/OUT Gaussians.
    out_masks_pub = ~in_masks_pub
    in_counts = in_masks_pub.sum(axis=0)
    print(f"\npub IN-shadow count per sample: min={in_counts.min()} "
          f"median={int(np.median(in_counts))} max={in_counts.max()}", flush=True)

    in_phi = np.where(in_masks_pub, phi_shadow_pub, np.nan)
    out_phi = np.where(out_masks_pub, phi_shadow_pub, np.nan)
    mu_in_pub = np.nanmean(in_phi, axis=0)
    sigma_in_raw = np.nanstd(in_phi, axis=0, ddof=1)
    mu_out_pub = np.nanmean(out_phi, axis=0)
    sigma_out_raw = np.nanstd(out_phi, axis=0, ddof=1)

    # Diagnostics: are per-sample σs collapsing? (would explain log-LR junk.)
    def pct(a, qs=(1, 5, 50, 95, 99)):
        return "  ".join(f"p{q}={np.nanpercentile(a, q):+.3g}" for q in qs)
    print("\nφ_target_pub  ", pct(phi_target_pub), flush=True)
    print("μ_in_pub      ", pct(mu_in_pub), flush=True)
    print("μ_out_pub     ", pct(mu_out_pub), flush=True)
    print("σ_in_pub raw  ", pct(sigma_in_raw), flush=True)
    print("σ_out_pub raw ", pct(sigma_out_raw), flush=True)
    n_floor_in = int((sigma_in_raw < SIGMA_FLOOR * 10).sum())
    n_floor_out = int((sigma_out_raw < SIGMA_FLOOR * 10).sum())
    print(f"σ near floor: in={n_floor_in}/{len(pub)}  out={n_floor_out}/{len(pub)}",
          flush=True)

    sigma_in_pub = np.maximum(sigma_in_raw, SIGMA_FLOOR)
    sigma_out_pub = np.maximum(sigma_out_raw, SIGMA_FLOOR)

    log_lr_pub = (gauss_log_pdf(phi_target_pub, mu_in_pub, sigma_in_pub) -
                  gauss_log_pdf(phi_target_pub, mu_out_pub, sigma_out_pub))

    # Fixed-variance variant (Carlini eq. 4): pool σ across all samples.
    sigma_in_global = float(np.nanstd(in_phi))
    sigma_out_global = float(np.nanstd(out_phi))
    log_lr_pub_fixed = (gauss_log_pdf(phi_target_pub, mu_in_pub, sigma_in_global) -
                        gauss_log_pdf(phi_target_pub, mu_out_pub, sigma_out_global))

    # Mean-shift baseline: just (φ_target − μ_out). Robust to σ pathologies.
    score_meanshift = phi_target_pub - mu_out_pub

    pub_mem = np.asarray(pub.membership, dtype=int)
    print(f"\n=== TPR@5%FPR on pub ===")
    print(f"  online LiRA per-sample σ: {tpr_at_fpr(log_lr_pub, pub_mem):.4f}", flush=True)
    print(f"  online LiRA fixed σ     : {tpr_at_fpr(log_lr_pub_fixed, pub_mem):.4f}  "
          f"(σ_in={sigma_in_global:.3f} σ_out={sigma_out_global:.3f})", flush=True)
    print(f"  φ_target − μ_out        : {tpr_at_fpr(score_meanshift, pub_mem):.4f}", flush=True)

    # Priv: per-sample OUT-Gaussian (all shadows OUT) + class-conditional IN.
    mu_out_priv = phi_shadow_priv.mean(axis=0)
    sigma_out_priv = np.maximum(phi_shadow_priv.std(axis=0, ddof=1), SIGMA_FLOOR)

    pub_labels = np.asarray(pub.labels, dtype=int)
    priv_labels = np.asarray(priv.labels, dtype=int)
    mu_in_class = np.zeros(NUM_CLASSES, dtype=np.float64)
    sigma_in_class = np.zeros(NUM_CLASSES, dtype=np.float64)
    print("\nClass-conditional IN-Gaussian (pooled from pub):")
    for c in range(NUM_CLASSES):
        sub = in_phi[:, pub_labels == c]
        mu_in_class[c] = np.nanmean(sub)
        sigma_in_class[c] = max(np.nanstd(sub, ddof=1), SIGMA_FLOOR)
        print(f"  class {c}: μ={mu_in_class[c]:+.3f}  σ={sigma_in_class[c]:.3f}", flush=True)

    mu_in_priv = mu_in_class[priv_labels]
    sigma_in_priv = sigma_in_class[priv_labels]
    log_lr_priv = (gauss_log_pdf(phi_target_priv, mu_in_priv, sigma_in_priv) -
                   gauss_log_pdf(phi_target_priv, mu_out_priv, sigma_out_priv))
    score_priv = 1.0 / (1.0 + np.exp(-np.clip(log_lr_priv, -SIGMOID_CLIP, SIGMOID_CLIP)))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ids = [str(i) for i in priv.ids]
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "score"])
        for i, s in zip(ids, score_priv):
            w.writerow([i, f"{s:.6f}"])
    print(f"\nWrote {OUT_PATH}  rows={len(ids)}  "
          f"score range=[{score_priv.min():.4f}, {score_priv.max():.4f}]",
          flush=True)


if __name__ == "__main__":
    main()
