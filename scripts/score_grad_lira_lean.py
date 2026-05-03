"""Lean grad-LiRA feature extraction (P100-friendly + multi-GPU sharding).

Same per-layer log-grad-norm features as score_grad_lira.py, but tuned for
P100 throughput and structured for multi-job parallelism:

  * GRAD_BATCH = 192   — fits ~14GB of the P100's 16GB at vmap+grad. Triples
                         the per-step batch over the original (B=64),
                         amortizing Python/launch overhead.
  * Identity-only augs — drops the horizontal-flip query. Halves work; the
                         original (id+flip) projected ≥24h on P100 here.
  * --shard K N        — process shadows[K::N] only. Each shard writes its
                         own slice; combine_grad_features.py merges them
                         and computes scores. With N=4 on 4 P100s the wall
                         clock drops to ~1.5h.

Coexists with score_grad_lira.py — different output paths, no clobbering.

Single-job usage (no sharding):
    condor_submit mia_grad.sub \\
        -append "script=scripts/score_grad_lira_lean.py" \\
        -append "tag=grad_lean" -queue 1

Multi-GPU usage (4-way shard):
    for k in 0 1 2 3; do
        condor_submit mia_grad.sub \\
            -append "script=scripts/score_grad_lira_lean.py" \\
            -append "args=--shard $k 4" \\
            -append "tag=grad_lean_$k" -queue 1
    done

When all shard jobs complete:
    python scripts/combine_grad_features.py
    cp submissions/submission_grad_lean.csv submissions/submission.csv
    python -m src.submit --tag grad_lira_lean_n512

Outputs (under checkpoints/grad_features_lean/):
    shard_<K>_of_<N>.pt   per-shard shadow features
    target.pt             target features (only shard K=0 writes this)
"""

from __future__ import annotations

import argparse
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
from src.model import build_model, load_target

CHECKPOINTS_DIR = ROOT / "checkpoints"
FEATURES_DIR = CHECKPOINTS_DIR / "grad_features_lean"
GRAD_BATCH = 192


def preload(pool, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.stack([pool[i][1] for i in range(len(pool))])
    labels = torch.tensor([pool[i][2] for i in range(len(pool))],
                          dtype=torch.long)
    return imgs.to(device), labels.to(device)


def collect_grad_log_norms(model, imgs: torch.Tensor, labels: torch.Tensor,
                           param_names: list[str], device: str) -> np.ndarray:
    """(N, L) per-sample per-named-parameter log L2 grad norm. Identity aug only."""
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
        grads = grad_fn(params, x_batch, y_batch)
        layer_norms = torch.zeros(nb, n_layers, device=device, dtype=torch.float32)
        for j, name in enumerate(param_names):
            layer_norms[:, j] = grads[name].flatten(1).norm(dim=1).float() \
                .clamp_min(1e-12).log()
        out[start:start + nb] = layer_norms.cpu().numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", nargs=2, type=int, default=[0, 1],
                   metavar=("K", "N"),
                   help="Process shadows[K::N]. Default 0 1 = all shadows.")
    a = p.parse_args()
    shard_k, shard_n = a.shard
    if not (0 <= shard_k < shard_n):
        sys.exit(f"Invalid --shard {shard_k} {shard_n}: need 0 <= K < N")

    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout,
          flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    print(f"shard: {shard_k} of {shard_n}", flush=True)

    combined = load_combined()
    n_pub, n_priv, n_total = combined.n_pub, combined.n_priv, len(combined)
    print(f"combined pool: n_pub={n_pub} n_priv={n_priv} total={n_total}",
          flush=True)
    imgs, labels = preload(combined, device)

    target = load_target(MODEL_PATH, map_location=device).to(device)
    target.eval()
    param_names = [n for n, _ in target.named_parameters()]
    n_layers = len(param_names)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if shard_k == 0:
        print(f"Forward+backward: target on combined pool", flush=True)
        g_target = collect_grad_log_norms(target, imgs, labels, param_names, device)
        torch.save(
            {
                "g_target": g_target,
                "param_names": param_names,
                "ids": list(combined.ids),
                "n_pub": n_pub,
                "n_priv": n_priv,
                "pub_membership": np.asarray(load_pub().membership, dtype=int),
            },
            FEATURES_DIR / "target.pt",
        )
        print(f"Saved target features → {FEATURES_DIR / 'target.pt'}", flush=True)
    del target
    if device == "cuda":
        torch.cuda.empty_cache()

    all_ckpts = sorted(c for c in CHECKPOINTS_DIR.glob("shadow_*.pt")
                       if "_in_idx" not in c.name)
    if not all_ckpts:
        sys.exit(f"No shadow checkpoints (shadow_*.pt) in {CHECKPOINTS_DIR}")
    shard_ckpts = all_ckpts[shard_k::shard_n]
    n_shard = len(shard_ckpts)
    print(f"Total shadows: {len(all_ckpts)}; this shard processes {n_shard}",
          flush=True)

    g_shard = np.zeros((n_shard, n_total, n_layers), dtype=np.float32)
    in_masks_shard = np.zeros((n_shard, n_total), dtype=bool)
    shard_seeds: list[int] = []

    shadow = build_model().to(device)
    for j, ckpt in enumerate(shard_ckpts):
        seed_str = ckpt.stem.split("_")[-1]
        shard_seeds.append(int(seed_str))
        idx_path = CHECKPOINTS_DIR / f"shadow_{seed_str}_in_idx.pt"
        in_idx = torch.load(idx_path, weights_only=True).numpy()
        in_masks_shard[j, in_idx] = True

        shadow.load_state_dict(torch.load(ckpt, map_location=device,
                                          weights_only=True))
        shadow.eval()
        g_shard[j] = collect_grad_log_norms(shadow, imgs, labels,
                                            param_names, device)
        if (j + 1) % 8 == 0 or j == n_shard - 1:
            print(f"  scored shadow {j+1}/{n_shard} (seed {shard_seeds[-1]})",
                  flush=True)

    shard_path = FEATURES_DIR / f"shard_{shard_k}_of_{shard_n}.pt"
    torch.save(
        {
            "g_shadow": g_shard,           # (n_shard, n_total, L)
            "in_masks": in_masks_shard,    # (n_shard, n_total)
            "shadow_seeds": shard_seeds,
            "shard_k": shard_k,
            "shard_n": shard_n,
            "param_names": param_names,
            "n_pub": n_pub,
            "n_priv": n_priv,
        },
        shard_path,
    )
    print(f"Saved shard features → {shard_path}", flush=True)


if __name__ == "__main__":
    main()
