"""Adversarial-distance MIA (target-only, no shadows).

For each sample x, run K=20 PGD steps that try to push the target's loss
on the true label up. The score is the loss INCREASE achieved:

  score(x) = − (loss_after − loss_before)
             = loss_before − loss_after

Members are more *robust* — the model has already settled comfortably on
them, so the adversarial perturbation has less room to grow the loss.
Non-members are correctly classified too (the target generalizes well)
but with less margin, so PGD increases their loss more easily.

Higher score = smaller attack-induced loss increase = more member-y.

This is the input-space analogue to grad-LiRA (parameter-space) — a
genuinely orthogonal feature family that should decorrelate from the
output-confidence signals (logit-LiRA, grad-LiRA, aug-consistency).
Choquette-Choo et al. 2021 ("Label-Only MIA") show adversarial robustness
attacks match shadow-based MIA in well-generalizing regimes.

Outputs (independent of all other attack paths — no clobbering):
  checkpoints/adv_distance/features.npy           (n_total,) score
  checkpoints/adv_distance/loss_before.npy        (n_total,) initial loss
  checkpoints/adv_distance/loss_after.npy         (n_total,) post-PGD loss
  submissions/submission_adv_distance.csv

Usage:
    condor_submit mia_grad.sub \\
        -append "script=scripts/score_adv_distance.py" \\
        -append "tag=adv_distance" -queue 1
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_combined, load_pub, predict_collate, MODEL_PATH
from src.eval import tpr_at_fpr
from src.model import load_target

CHECKPOINTS_DIR = ROOT / "checkpoints"
FEATURES_DIR = CHECKPOINTS_DIR / "adv_distance"
OUT_PATH = ROOT / "submissions" / "submission_adv_distance.csv"
SIGMOID_CLIP = 50.0
PGD_STEPS = 20
PGD_ALPHA = 0.01   # per-step L_inf budget in normalized image space


def pgd_attack(model, x: torch.Tensor, y: torch.Tensor,
               n_steps: int, alpha: float) -> torch.Tensor:
    """Untargeted L_inf PGD: ascend the loss on the true label. Returns the
    adversarial input after `n_steps` sign-gradient steps."""
    x_adv = x.clone().detach().requires_grad_(True)
    for _ in range(n_steps):
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="sum")
        grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
        with torch.no_grad():
            x_adv = (x_adv + alpha * grad.sign()).detach().requires_grad_(True)
    return x_adv.detach()


def collect_adv_signal(model, loader, n: int, device: str
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample initial and post-PGD cross-entropy loss on the true label."""
    loss_before = np.zeros(n, dtype=np.float64)
    loss_after = np.zeros(n, dtype=np.float64)
    pos = 0
    n_batches = (n + 255) // 256
    for batch_idx, (_, imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        nb = imgs.shape[0]

        with torch.no_grad():
            l0 = F.cross_entropy(model(imgs), labels, reduction="none")
        loss_before[pos:pos + nb] = l0.cpu().numpy()

        x_adv = pgd_attack(model, imgs, labels, PGD_STEPS, PGD_ALPHA)
        with torch.no_grad():
            l1 = F.cross_entropy(model(x_adv), labels, reduction="none")
        loss_after[pos:pos + nb] = l1.cpu().numpy()

        pos += nb
        if (batch_idx + 1) % 16 == 0 or batch_idx == n_batches - 1:
            print(f"  batch {batch_idx + 1}/{n_batches}", flush=True)
    return loss_before, loss_after


def main():
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout,
          flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  PGD_STEPS={PGD_STEPS}  PGD_ALPHA={PGD_ALPHA}",
          flush=True)

    combined = load_combined()
    n_pub, n_priv, n_total = combined.n_pub, combined.n_priv, len(combined)
    print(f"combined pool: n_pub={n_pub} n_priv={n_priv} total={n_total}",
          flush=True)
    loader = DataLoader(combined, batch_size=256, shuffle=False, num_workers=2,
                        collate_fn=predict_collate)

    target = load_target(MODEL_PATH, map_location=device).to(device)
    target.eval()  # eval mode for BN; PGD still flows gradients through inputs.

    print(f"\nPGD attack: {PGD_STEPS} steps × combined pool", flush=True)
    loss_before, loss_after = collect_adv_signal(target, loader, n_total, device)

    # Larger loss_after − loss_before = easier to attack = LESS member-y.
    # Score is the negation so that higher = more member-y, matching the
    # convention used by every other submission CSV.
    score = loss_before - loss_after

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_DIR / "features.npy", score)
    np.save(FEATURES_DIR / "loss_before.npy", loss_before)
    np.save(FEATURES_DIR / "loss_after.npy", loss_after)
    print(f"\nSaved features → {FEATURES_DIR}", flush=True)
    print(f"loss_before  mean={loss_before.mean():.4f}  std={loss_before.std():.4f}",
          flush=True)
    print(f"loss_after   mean={loss_after.mean():.4f}  std={loss_after.std():.4f}",
          flush=True)
    print(f"score (b−a)  mean={score.mean():.4f}  std={score.std():.4f}",
          flush=True)

    pub_membership = np.asarray(load_pub().membership, dtype=int)
    pub_tpr = tpr_at_fpr(score[:n_pub], pub_membership)
    print(f"\n=== TPR@5%FPR on pub (adversarial distance) ===", flush=True)
    print(f"  {pub_tpr:.4f}", flush=True)

    # Map score → (0,1) via sigmoid for submission format compliance. Rank-
    # preserving so the metric is unaffected; the clip avoids overflow.
    score_priv = score[n_pub:]
    score_priv_01 = 1.0 / (1.0 + np.exp(-np.clip(score_priv,
                                                 -SIGMOID_CLIP, SIGMOID_CLIP)))
    priv_ids = combined.ids[n_pub:]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "score"])
        for i, s in zip(priv_ids, score_priv_01):
            w.writerow([str(i), f"{s:.6f}"])
    print(f"\nWrote {OUT_PATH}  rows={len(priv_ids)}  "
          f"score range=[{score_priv_01.min():.4f}, {score_priv_01.max():.4f}]",
          flush=True)


if __name__ == "__main__":
    main()
