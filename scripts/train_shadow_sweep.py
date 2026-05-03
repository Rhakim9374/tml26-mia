"""Train ONE shadow with a candidate recipe + score it on pub vs target's φ.

Used to find a shadow recipe whose φ distribution matches the target's. Our
baseline recipe overfits ~3-5x more than the target (target φ p99 ≈ 3.78,
shadow φ p99 ≈ 15.8), which is the leading suspect for the LiRA pub TPR
ceiling at 0.07. Lower-overfit candidates: label smoothing, fewer epochs,
heavier weight decay.

Each invocation trains one shadow (different seed per recipe variant to
avoid overwriting) under a NEW prefix so the baseline 512 shadows stay
intact. The script then runs that shadow on the combined pool and prints
a φ-distribution comparison to target's saved phi_target.npy.

Pick the recipe whose shadow φ distribution best matches target's. Then
train 512 shadows with that recipe overnight via train_shadow.py with
the matching --label_smoothing / --epochs / --weight_decay / --ckpt_prefix
flags.

Usage examples (each ~5–10 min on P100):
    condor_submit mia.sub \\
        -append "script=scripts/train_shadow_sweep.py" \\
        -append "args=--seed 9000 --label_smoothing 0.05 --ckpt_prefix sweep_ls005" \\
        -append "tag=sweep_ls005" -queue 1

    condor_submit mia.sub \\
        -append "script=scripts/train_shadow_sweep.py" \\
        -append "args=--seed 9001 --label_smoothing 0.10 --ckpt_prefix sweep_ls010" \\
        -append "tag=sweep_ls010" -queue 1

    condor_submit mia.sub \\
        -append "script=scripts/train_shadow_sweep.py" \\
        -append "args=--seed 9002 --epochs 50 --ckpt_prefix sweep_ep50" \\
        -append "tag=sweep_ep50" -queue 1

    condor_submit mia.sub \\
        -append "script=scripts/train_shadow_sweep.py" \\
        -append "args=--seed 9003 --weight_decay 1e-3 --ckpt_prefix sweep_wd1e3" \\
        -append "tag=sweep_wd1e3" -queue 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_combined, predict_collate
from src.model import build_model
from src.train import train_shadow

CHECKPOINTS_DIR = ROOT / "checkpoints"
PHI_TARGET_PATH = ROOT / "checkpoints" / "logit_features" / "phi_target.npy"


@torch.no_grad()
def collect_phi(model, loader, n: int, device: str) -> np.ndarray:
    """Single-aug (identity) φ on combined pool. Quick — full multi-aug not needed
    just for distribution comparison."""
    phi = np.zeros(n, dtype=np.float64)
    pos = 0
    for _, imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        z_y = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        masked = logits.scatter(1, labels.unsqueeze(1), float("-inf"))
        log_sum_other = torch.logsumexp(masked, dim=1)
        nb = imgs.shape[0]
        phi[pos:pos + nb] = (z_y - log_sum_other).double().cpu().numpy()
        pos += nb
    return phi


def pct_str(a: np.ndarray, qs=(1, 5, 25, 50, 75, 95, 99)) -> str:
    return "  ".join(f"p{q}={np.nanpercentile(a, q):+6.2f}" for q in qs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--ckpt_prefix", default="sweep",
                   help="Filename prefix so this run doesn't clobber baseline shadows.")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Recipe sweep ===", flush=True)
    print(f"  seed={a.seed}  epochs={a.epochs}  batch={a.batch_size}", flush=True)
    print(f"  label_smoothing={a.label_smoothing}  weight_decay={a.weight_decay}",
          flush=True)
    print(f"  ckpt_prefix={a.ckpt_prefix}", flush=True)

    t0 = time.time()
    ckpt_path = train_shadow(
        seed=a.seed,
        epochs=a.epochs,
        batch_size=a.batch_size,
        label_smoothing=a.label_smoothing,
        weight_decay=a.weight_decay,
        ckpt_prefix=a.ckpt_prefix,
        device=device,
    )
    print(f"\nTraining done in {time.time() - t0:.1f}s → {ckpt_path.name}",
          flush=True)

    # Score on combined pool to compare φ distribution with target.
    combined = load_combined()
    n_pub = combined.n_pub
    loader = DataLoader(combined, batch_size=512, shuffle=False, num_workers=2,
                        collate_fn=predict_collate)
    model = build_model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=True))
    model.eval()
    phi_shadow = collect_phi(model, loader, len(combined), device)

    # Split shadow φ by IN/OUT for THIS shadow.
    idx_path = CHECKPOINTS_DIR / f"{a.ckpt_prefix}_{a.seed:04d}_in_idx.pt"
    in_idx = torch.load(idx_path, weights_only=True).numpy()
    in_mask = np.zeros(len(combined), dtype=bool)
    in_mask[in_idx] = True
    phi_in = phi_shadow[in_mask]
    phi_out = phi_shadow[~in_mask]

    print(f"\n=== shadow φ distribution percentiles ===")
    print(f"  IN  (this shadow trained on these): {pct_str(phi_in)}")
    print(f"  OUT (this shadow did not see):      {pct_str(phi_out)}")

    if PHI_TARGET_PATH.exists():
        phi_target = np.load(PHI_TARGET_PATH)
        print(f"\n=== target φ for reference ===")
        print(f"  target φ (combined pool):           {pct_str(phi_target)}")

        # Match metric: |median(shadow IN) − median(target overall)|
        # Smaller is better — closer scale match.
        match_in = abs(np.median(phi_in) - np.median(phi_target))
        match_out = abs(np.median(phi_out) - np.median(phi_target))
        print(f"\n=== match score (lower = closer to target distribution) ===")
        print(f"  |median(shadow IN)  − median(target)|  = {match_in:.3f}")
        print(f"  |median(shadow OUT) − median(target)|  = {match_out:.3f}")
    else:
        print(f"\n(phi_target.npy not found at {PHI_TARGET_PATH} — skip comparison)")


if __name__ == "__main__":
    main()
