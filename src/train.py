"""Shadow-model training utilities.

Standard CIFAR-style recipe: SGD momentum 0.9, weight_decay 5e-4, cosine LR
peak 0.1, 100 epochs, batch 256. Augmentations: random crop (32, padding=4)
+ horizontal flip. Same Normalize as the target, applied after augs.

Each shadow trains on a deterministic random 50% subset of the COMBINED
pub+priv pool (28k samples), controlled by a seed. Including priv in the
pool is what makes online LiRA work on priv: every priv sample is IN for
~half the shadows and OUT for the other half, giving us per-sample IN/OUT
Gaussians on priv at scoring time. Priv membership is the *target's*
training-set membership; for shadows we just need (image, label) pairs and
both pub.pt and priv.pt provide those.

Saves to checkpoints/shadow_NNNN.pt + shadow_NNNN_in_idx.pt. The in-idx
tensor stores positions in the combined pool: [0, n_pub) → pub samples,
[n_pub, total) → priv samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from src.data import MEAN, STD, CombinedPool, load_combined
from src.model import build_model

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def train_transform(strong: bool = False) -> T.Compose:
    """Standard CIFAR-style augs. With `strong=True`, adds ColorJitter and
    RandomErasing to the pipeline — heavier regularization that prevents
    100%-confident memorization on hard samples."""
    pre_norm = [T.Resize(32), T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if strong:
        pre_norm.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    post_norm = [T.Normalize(mean=MEAN, std=STD)]
    if strong:
        post_norm.append(T.RandomErasing(p=0.25))
    return T.Compose(pre_norm + post_norm)


def eval_transform() -> T.Compose:
    return T.Compose([
        T.Resize(32),
        T.Normalize(mean=MEAN, std=STD),
    ])


class IndexedSubset(Dataset):
    """Wraps a pool (CombinedPool / MembershipDataset), yields (img, label) for
    the given indices, applying a fresh transform. Avoids mutating the
    underlying dataset."""

    def __init__(self, base: CombinedPool, indices: Iterable[int],
                 transform: T.Compose):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        img = self.base.imgs[idx]
        label = self.base.labels[idx]
        img = self.transform(img)
        return img, label


def split_in_out(n: int, seed: int, frac_in: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(frac_in * n))
    return perm[:cut], perm[cut:]


def _maybe_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Standard mixup (Zhang et al. 2017). alpha=0 disables → returns inputs
    unchanged with a sentinel `lam=None`."""
    if alpha <= 0:
        return x, y, y, None
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _make_scheduler(optimizer, scheduler_type: str, epochs: int):
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3),
                                               gamma=0.1)
    if scheduler_type == "linear":
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                 end_factor=0.0, total_iters=epochs)
    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0,
                                                   total_iters=epochs)
    raise ValueError(f"Unknown scheduler_type: {scheduler_type}")


def _make_optimizer(model, optimizer_type: str, lr: float, momentum: float,
                    nesterov: bool, weight_decay: float):
    if optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                               weight_decay=weight_decay, nesterov=nesterov)
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    if optimizer_type == "adamw":
        # AdamW decouples weight decay from gradient update — different from
        # Adam's L2 regularization in the gradient.
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer_type: {optimizer_type}")


def train_shadow(
    seed: int,
    epochs: int = 100,
    batch_size: int = 256,
    optimizer_type: str = "sgd",
    lr: float = 0.1,
    momentum: float = 0.9,
    nesterov: bool = True,
    weight_decay: float = 5e-4,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    aug_strong: bool = False,
    scheduler_type: str = "cosine",
    num_workers: int = 2,
    device: str | None = None,
    verbose: bool = True,
    ckpt_prefix: str = "shadow",
) -> Path:
    """Train one shadow on a 50% subset of the combined pub+priv pool.

    Knobs (all the levers we use to match target's φ distribution):
      label_smoothing : caps maximum confidence (LS=0.05 already matches target
                        in p25–p99; lower tail still needs more help).
      mixup_alpha     : Zhang-style mixup. Likely cause of target's wide lower
                        tail — mixup creates "hard" mixed samples the model
                        can't classify confidently.
      aug_strong      : ColorJitter + RandomErasing on top of crop+flip.
                        Heavier regularization, similar effect to mixup.
      epochs / lr / scheduler_type / weight_decay / momentum / nesterov :
                        standard optimizer knobs.

    ckpt_prefix lets you train recipe variants (e.g., "lsv1") without
    overwriting baseline shadow_NNNN.pt.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pool = load_combined(transform=None)
    in_idx, _ = split_in_out(len(pool), seed=seed, frac_in=0.5)
    train_ds = IndexedSubset(pool, in_idx, train_transform(strong=aug_strong))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device == "cuda"),
                              drop_last=False)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model().to(device)
    optimizer = _make_optimizer(model, optimizer_type, lr, momentum, nesterov,
                                weight_decay)
    scheduler = _make_scheduler(optimizer, scheduler_type, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    model.train()
    for epoch in range(epochs):
        running, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            mixed_x, y_a, y_b, lam = _maybe_mixup(imgs, labels, mixup_alpha)
            logits = model(mixed_x)
            if lam is None:
                loss = criterion(logits, y_a)
            else:
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            # Accuracy reported against the "majority" label only (y_a) when
            # mixup is on — informative enough to monitor convergence.
            correct += (logits.argmax(1) == y_a).sum().item()
            total += imgs.size(0)
        scheduler.step()
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss={running/total:.4f}  "
                  f"acc={correct/total:.4f}  lr={scheduler.get_last_lr()[0]:.4f}",
                  flush=True)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINTS_DIR / f"{ckpt_prefix}_{seed:04d}.pt"
    idx_path = CHECKPOINTS_DIR / f"{ckpt_prefix}_{seed:04d}_in_idx.pt"
    torch.save(model.state_dict(), ckpt_path)
    torch.save(torch.from_numpy(in_idx.astype(np.int64)), idx_path)
    if verbose:
        print(f"Saved {ckpt_path.name} ({len(in_idx)} IN samples "
              f"of {len(pool)} combined)", flush=True)
    return ckpt_path
