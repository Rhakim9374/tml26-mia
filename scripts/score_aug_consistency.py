"""Augmentation-consistency MIA (label-only, target-only, no shadows).

For each sample x, classify it under K random augmentations of itself.
  score(x) = mean confidence(target on true class) across K augs

Members were trained ON augmentations of these images; the model has explicit
invariance baked in. Non-members are correctly classified too (target
generalizes well) but with marginally lower stability under view changes.
The signal is small per sample, so we average over K=20 augs to reduce
variance.

This is a label-only attack from Choquette-Choo et al. 2021 ("Label-Only
Membership Inference Attacks"). Uses ONLY the target model — no shadows,
no parameter gradients, no φ from shadow models. Completely independent
signal source from logit-LiRA and grad-LiRA, which is the whole point: if
it decorrelates, the ensemble lifts beyond the ~0.07 ceiling we've hit
with confidence-based attacks.

Outputs (independent of LiRA / grad-LiRA paths — no clobbering):
  checkpoints/aug_consistency/features.npy   (n_total,) mean confidence
  checkpoints/aug_consistency/correct.npy    (n_total,) integer count of correct preds
  submissions/submission_aug_consistency.csv

Usage:
    condor_submit mia_grad.sub \\
        -append "script=scripts/score_aug_consistency.py" \\
        -append "tag=aug_consistency" -queue 1
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
FEATURES_DIR = CHECKPOINTS_DIR / "aug_consistency"
OUT_PATH = ROOT / "submissions" / "submission_aug_consistency.csv"
SIGMOID_CLIP = 50.0
N_AUGS = 20
PAD = 4
SEED = 12345


def random_aug(imgs: torch.Tensor) -> torch.Tensor:
    """Per-sample random pad-and-crop + random hflip on normalized tensors.

    PAD=4 gives 9 crop offsets in each dim (= 81 spatial views), plus 2 flip
    states, plenty of view diversity for K=20 averaging.
    """
    B, C, H, W = imgs.shape
    padded = F.pad(imgs, (PAD, PAD, PAD, PAD), mode="reflect")
    dy = torch.randint(0, 2 * PAD + 1, (B,), device=imgs.device)
    dx = torch.randint(0, 2 * PAD + 1, (B,), device=imgs.device)
    out = torch.empty_like(imgs)
    for b in range(B):
        out[b] = padded[b, :, dy[b]:dy[b] + H, dx[b]:dx[b] + W]
    flip = torch.rand(B, device=imgs.device) < 0.5
    flipped = torch.flip(out, dims=[-1])
    return torch.where(flip.view(B, 1, 1, 1), flipped, out)


@torch.no_grad()
def collect_aug_signal(model, loader, n: int, device: str
                       ) -> tuple[np.ndarray, np.ndarray]:
    """For each sample: mean softmax-confidence on true class across N_AUGS
    random views, plus integer count of correct top-1 predictions."""
    confidence_sum = torch.zeros(n, device=device, dtype=torch.float64)
    correct_count = torch.zeros(n, device=device, dtype=torch.long)
    for aug_idx in range(N_AUGS):
        pos = 0
        for _, imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            x_aug = random_aug(imgs)
            logits = model(x_aug)
            probs = torch.softmax(logits, dim=1)
            conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            pred = logits.argmax(dim=1)
            nb = imgs.shape[0]
            confidence_sum[pos:pos + nb] += conf.double()
            correct_count[pos:pos + nb] += (pred == labels).long()
            pos += nb
        if (aug_idx + 1) % 5 == 0 or aug_idx == N_AUGS - 1:
            print(f"  pass {aug_idx + 1}/{N_AUGS}", flush=True)
    return (confidence_sum / N_AUGS).cpu().numpy(), correct_count.cpu().numpy()


def main():
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout,
          flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  N_AUGS={N_AUGS}  PAD={PAD}  seed={SEED}", flush=True)
    torch.manual_seed(SEED)

    combined = load_combined()
    n_pub, n_priv, n_total = combined.n_pub, combined.n_priv, len(combined)
    print(f"combined pool: n_pub={n_pub} n_priv={n_priv} total={n_total}",
          flush=True)
    loader = DataLoader(combined, batch_size=256, shuffle=False, num_workers=2,
                        collate_fn=predict_collate)

    target = load_target(MODEL_PATH, map_location=device).to(device)
    target.eval()

    print(f"\nForward passes: {N_AUGS} augs × combined pool", flush=True)
    mean_conf, correct = collect_aug_signal(target, loader, n_total, device)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_DIR / "features.npy", mean_conf)
    np.save(FEATURES_DIR / "correct.npy", correct)
    print(f"\nSaved features → {FEATURES_DIR}", flush=True)

    pub_membership = np.asarray(load_pub().membership, dtype=int)
    print(f"\n=== TPR@5%FPR on pub (aug-consistency) ===", flush=True)
    tpr_conf = tpr_at_fpr(mean_conf[:n_pub], pub_membership)
    tpr_corr = tpr_at_fpr(correct[:n_pub].astype(np.float64), pub_membership)
    print(f"  mean confidence : {tpr_conf:.4f}", flush=True)
    print(f"  # correct of {N_AUGS:2d}: {tpr_corr:.4f}", flush=True)

    # Use mean-confidence as the submitted score (continuous, more granular
    # than the integer count). Already in [0, 1] from softmax.
    score_priv = mean_conf[n_pub:]
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
