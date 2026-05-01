"""Phase 0 recon: characterize the data + measure baseline MIA signal.

Run on the cluster (interactive recommended for first run):

    condor_submit -i mia.sub -append "tag=recon"
    # inside the container:
    cd /home/$USER/tml26-mia
    python scripts/recon.py

Or as a batch job (default script in mia.sub is this file):
    condor_submit mia.sub -append "tag=recon"
    # then: cat runlogs/recon.<ClusterId>.0.out

Reports:
  - Dataset sizes; member/non-member counts in pub.pt.
  - Image shape, dtype, value range (raw + after transform).
  - Number of classes (in pub and priv).
  - Target model accuracy on pub members vs non-members.
  - Mean / median logit-scaled loss φ(p) = log(p/(1-p)) on the correct class
    for pub members vs non-members. The Δmean is the "is the attack feasible"
    sanity check — LiRA needs at least a small positive gap to bootstrap from.
  - TPR@5%FPR for the naive φ-as-score attack on pub. This is the floor LiRA
    needs to beat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_pub, load_priv, predict_collate, MODEL_PATH
from src.eval import tpr_at_fpr
from src.model import load_target


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}")

    pub = load_pub()
    priv = load_priv()

    print("\n=== Sizes ===")
    print(f"pub  : {len(pub)}")
    print(f"priv : {len(priv)}")

    pub_mem = np.asarray(pub.membership, dtype=int)
    print(f"pub members     : {int((pub_mem == 1).sum())}")
    print(f"pub non-members : {int((pub_mem == 0).sum())}")

    raw = pub.imgs[0]
    print("\n=== Image format (raw, pre-transform) ===")
    print(f"type  : {type(raw).__name__}")
    print(f"dtype : {getattr(raw, 'dtype', None)}")
    print(f"shape : {getattr(raw, 'shape', None)}")
    if isinstance(raw, torch.Tensor):
        print(f"range : [{raw.min().item():.4f}, {raw.max().item():.4f}]")

    s = pub[0]
    print(
        f"\nSample after transform: id={s[0]}  label={s[2]}  mem={s[3]}  "
        f"img.shape={s[1].shape}  img.dtype={s[1].dtype}"
    )

    pub_labels = np.asarray(pub.labels, dtype=int)
    priv_labels = np.asarray(priv.labels, dtype=int)
    print("\n=== Classes ===")
    print(f"pub  unique labels: {sorted(np.unique(pub_labels).tolist())}")
    print(f"priv unique labels: {sorted(np.unique(priv_labels).tolist())}")
    print(f"pub  label counts : {np.bincount(pub_labels).tolist()}")

    print("\n=== Forward pass: target model on pub ===")
    model = load_target(MODEL_PATH, map_location=device).to(device)
    loader = DataLoader(pub, batch_size=512, shuffle=False, num_workers=2,
                        collate_fn=predict_collate)

    p_correct = np.zeros(len(pub), dtype=np.float64)
    correct = np.zeros(len(pub), dtype=bool)
    pos = 0
    with torch.no_grad():
        for _, imgs, labels_b in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            pc = probs.gather(1, labels_b.unsqueeze(1)).squeeze(1)
            pred = logits.argmax(1)
            n = pc.shape[0]
            p_correct[pos:pos + n] = pc.cpu().numpy()
            correct[pos:pos + n] = (pred == labels_b).cpu().numpy()
            pos += n

    eps = 1e-12
    phi = np.log(np.clip(p_correct, eps, 1 - eps) / np.clip(1 - p_correct, eps, 1 - eps))

    print("\n=== Target accuracy on pub ===")
    print(f"  members     : {correct[pub_mem == 1].mean():.4f}")
    print(f"  non-members : {correct[pub_mem == 0].mean():.4f}")

    print("\n=== Logit-scaled loss  φ(p) = log(p/(1-p))  on correct class ===")
    print(
        f"  members     mean={phi[pub_mem == 1].mean():+.3f}  "
        f"median={np.median(phi[pub_mem == 1]):+.3f}"
    )
    print(
        f"  non-members mean={phi[pub_mem == 0].mean():+.3f}  "
        f"median={np.median(phi[pub_mem == 0]):+.3f}"
    )
    print(
        f"  Δmean (members − non-members): "
        f"{phi[pub_mem == 1].mean() - phi[pub_mem == 0].mean():+.3f}"
    )

    naive = tpr_at_fpr(phi, pub_mem, 0.05)
    print(f"\n=== Naive φ-as-score TPR@5%FPR on pub : {naive:.4f} ===")
    print("(LiRA needs to beat this comfortably to be worth the compute.)")


if __name__ == "__main__":
    main()
