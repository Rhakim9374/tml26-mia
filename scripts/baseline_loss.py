"""Baseline submission: per-sample softmax probability on the correct class.

This is the loss-threshold baseline (φ-based, but in [0,1] space). Recon showed
TPR@5%FPR ≈ 0.0507 on pub — near random — so this submission is purely for
validating the leaderboard pipeline (API key, CSV format, server reachability).
The leaderboard score is expected to be ~0.05.

Run on the cluster:
    condor_submit mia.sub -append "script=scripts/baseline_loss.py" -append "tag=baseline"
    # then once the job finishes:
    python -m src.submit --tag baseline_phi
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

from src.data import load_priv, predict_collate, MODEL_PATH
from src.model import load_target

OUT_PATH = ROOT / "submissions" / "submission.csv"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    priv = load_priv()
    print(f"priv size: {len(priv)}", flush=True)

    model = load_target(MODEL_PATH, map_location=device).to(device)
    loader = DataLoader(priv, batch_size=512, shuffle=False, num_workers=2,
                        collate_fn=predict_collate)

    p_correct = np.zeros(len(priv), dtype=np.float64)
    ids: list[str] = []
    pos = 0
    with torch.no_grad():
        for ids_b, imgs, labels_b in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            pc = probs.gather(1, labels_b.unsqueeze(1)).squeeze(1)
            n = pc.shape[0]
            p_correct[pos:pos + n] = pc.cpu().numpy()
            ids.extend(str(i) for i in ids_b)
            pos += n

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "score"])
        for i, s in zip(ids, p_correct):
            w.writerow([i, f"{s:.6f}"])
    print(f"Wrote {OUT_PATH}  rows={len(ids)}  "
          f"score range=[{p_correct.min():.4f}, {p_correct.max():.4f}]",
          flush=True)


if __name__ == "__main__":
    main()
