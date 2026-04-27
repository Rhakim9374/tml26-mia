"""Shadow-model training utilities.

Standard CIFAR-style recipe: SGD momentum 0.9, weight_decay 5e-4, cosine LR
peak 0.1, 100 epochs, batch 256. Augmentations: random crop (32, padding=4)
+ horizontal flip. Same Normalize as the target, applied after augs.

A "shadow" trains on a deterministic random 50% subset of a pool dataset,
controlled by a seed. The IN-index set is saved alongside the checkpoint so
later scoring scripts can tell who was IN vs OUT for each shadow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from src.data import MEAN, STD, MembershipDataset, load_pub
from src.model import build_model

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def train_transform() -> T.Compose:
    return T.Compose([
        T.Resize(32),
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def eval_transform() -> T.Compose:
    return T.Compose([
        T.Resize(32),
        T.Normalize(mean=MEAN, std=STD),
    ])


class IndexedSubset(Dataset):
    """Wraps a MembershipDataset, yields (img, label) for the given indices,
    applying a fresh transform. Avoids mutating the underlying dataset."""

    def __init__(self, base: MembershipDataset, indices: Iterable[int],
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


def train_shadow(seed: int, pool_name: str = "pub", epochs: int = 100,
                 batch_size: int = 256, num_workers: int = 2,
                 device: str | None = None, verbose: bool = True) -> Path:
    """Train one shadow and save checkpoint + IN-indices to CHECKPOINTS_DIR."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if pool_name == "pub":
        pool = load_pub(transform=None)
    else:
        raise ValueError(f"Unknown pool: {pool_name}")

    in_idx, _ = split_in_out(len(pool), seed=seed, frac_in=0.5)
    train_ds = IndexedSubset(pool, in_idx, train_transform())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device == "cuda"),
                              drop_last=False)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        scheduler.step()
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss={running/total:.4f}  "
                  f"acc={correct/total:.4f}  lr={scheduler.get_last_lr()[0]:.4f}",
                  flush=True)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINTS_DIR / f"{pool_name}_shadow_{seed:04d}.pt"
    idx_path = CHECKPOINTS_DIR / f"{pool_name}_shadow_{seed:04d}_in_idx.pt"
    torch.save(model.state_dict(), ckpt_path)
    torch.save(torch.from_numpy(in_idx.astype(np.int64)), idx_path)
    if verbose:
        print(f"Saved {ckpt_path.name} ({len(in_idx)} IN samples)", flush=True)
    return ckpt_path
