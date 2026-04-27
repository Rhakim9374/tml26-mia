"""Offline LiRA scoring: pub validation + priv submission.

Uses 16 OUT-only shadows trained on random 50% subsets of pub. For each
(sample, shadow), compute φ(p) = log(p/(1-p)) on the correct class. Then:

    For each pub sample:
        OUT shadows = shadows where this sample was NOT in the IN set (~8).
        Fit Gaussian over their φs. Compute z = (φ_target - μ) / σ.
        Score = sigmoid(z).
    For each priv sample:
        All 16 shadows are OUT (priv was never in the shadow pool).
        Same scoring formula.

Reports:
    - TPR@5%FPR on pub using LiRA score (vs the 0.0507 naive φ floor).
    - Writes submissions/submission.csv with priv scores.

Run after training 16 shadows:
    condor_submit mia.sub -append "script=scripts/score_offline_lira.py" \\
                          -append "tag=score_lira"
    # then:
    python -m src.submit --tag offline_lira_n16
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_pub, load_priv, MODEL_PATH
from src.model import build_model, load_target

CHECKPOINTS_DIR = ROOT / "checkpoints"
OUT_PATH = ROOT / "submissions" / "submission.csv"
EPS = 1e-12


def phi(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tpr_at_fpr(scores: np.ndarray, members: np.ndarray, fpr: float = 0.05) -> float:
    order = np.argsort(-scores)
    m = members[order]
    n_neg = int((members == 0).sum())
    n_pos = int((members == 1).sum())
    fp_budget = int(np.floor(fpr * n_neg))
    neg_seen, cutoff = 0, len(m)
    for i, mi in enumerate(m):
        if mi == 0:
            neg_seen += 1
            if neg_seen > fp_budget:
                cutoff = i
                break
    tp = int((m[:cutoff] == 1).sum())
    return tp / max(n_pos, 1)


@torch.no_grad()
def collect_phi(model, loader, n: int, device: str) -> np.ndarray:
    """Return an array of φ values aligned with the dataset's order."""
    out = np.zeros(n, dtype=np.float64)
    pos = 0
    for batch in loader:
        # support both pub (4-tuple) and priv (4-tuple, mem=None)
        _, imgs, labels_b, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        labels_b = labels_b.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        p_correct = probs.gather(1, labels_b.unsqueeze(1)).squeeze(1).cpu().numpy()
        nb = p_correct.shape[0]
        out[pos:pos + nb] = phi(p_correct)
        pos += nb
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    pub = load_pub()
    priv = load_priv()
    pub_loader = DataLoader(pub, batch_size=512, shuffle=False, num_workers=2)
    priv_loader = DataLoader(priv, batch_size=512, shuffle=False, num_workers=2)

    # Target φ over pub and priv.
    print("Forward pass: target", flush=True)
    target = load_target(MODEL_PATH, map_location=device).to(device)
    phi_target_pub = collect_phi(target, pub_loader, len(pub), device)
    phi_target_priv = collect_phi(target, priv_loader, len(priv), device)
    del target
    torch.cuda.empty_cache() if device == "cuda" else None

    # Find shadows.
    ckpts = sorted(CHECKPOINTS_DIR.glob("pub_shadow_*.pt"))
    ckpts = [c for c in ckpts if "_in_idx" not in c.name]
    if not ckpts:
        sys.exit(f"No shadow checkpoints found in {CHECKPOINTS_DIR}")
    print(f"Found {len(ckpts)} shadow checkpoints", flush=True)

    n_shadows = len(ckpts)
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
        print(f"  [{k+1}/{n_shadows}] {ckpt.name}", flush=True)

    # OUT mask for pub: True where shadow did NOT include this sample.
    out_masks_pub = ~in_masks_pub

    # Per-pub-sample OUT-Gaussian fit.
    out_counts_pub = out_masks_pub.sum(axis=0)  # ~ n_shadows / 2
    print(f"pub OUT-shadow count per sample: min={out_counts_pub.min()} "
          f"median={int(np.median(out_counts_pub))} max={out_counts_pub.max()}",
          flush=True)

    masked = np.where(out_masks_pub, phi_shadow_pub, np.nan)
    mu_out_pub = np.nanmean(masked, axis=0)
    sigma_out_pub = np.nanstd(masked, axis=0, ddof=1)
    sigma_out_pub = np.maximum(sigma_out_pub, 1e-6)

    z_pub = (phi_target_pub - mu_out_pub) / sigma_out_pub
    score_pub = sigmoid(z_pub)

    pub_mem = np.asarray(pub.membership, dtype=int)
    naive = tpr_at_fpr(phi_target_pub, pub_mem, 0.05)
    lira = tpr_at_fpr(score_pub, pub_mem, 0.05)
    print(f"\n=== TPR@5%FPR on pub ===")
    print(f"  naive φ       : {naive:.4f}")
    print(f"  offline LiRA  : {lira:.4f}   (Δ = {lira-naive:+.4f})")

    # Priv: all shadows are OUT.
    mu_out_priv = phi_shadow_priv.mean(axis=0)
    sigma_out_priv = phi_shadow_priv.std(axis=0, ddof=1)
    sigma_out_priv = np.maximum(sigma_out_priv, 1e-6)
    z_priv = (phi_target_priv - mu_out_priv) / sigma_out_priv
    score_priv = sigmoid(z_priv)

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
