"""Dataset classes + loaders, mirroring the course-provided task_template.py.

The .pt files are pickled instances of these classes from `__main__` (because
the course staff's data-prep script defined them at top level). Loading from
any other module fails with `Can't get attribute 'MembershipDataset' on
<module '__main__' ...>`. We register the classes into __main__ at import
time so plain `torch.load` works no matter which script is the entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PUB_PATH = DATA_DIR / "pub.pt"
PRIV_PATH = DATA_DIR / "priv.pt"
MODEL_PATH = DATA_DIR / "model.pt"

# Normalization constants the target model was trained with.
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]


def standard_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# Register classes into __main__ so torch.load can unpickle .pt files that
# reference __main__.MembershipDataset / __main__.TaskDataset.
sys.modules["__main__"].TaskDataset = TaskDataset
sys.modules["__main__"].MembershipDataset = MembershipDataset


def load_pub(transform=None) -> MembershipDataset:
    ds = torch.load(PUB_PATH, weights_only=False)
    ds.transform = transform if transform is not None else standard_transform()
    return ds


def load_priv(transform=None) -> MembershipDataset:
    ds = torch.load(PRIV_PATH, weights_only=False)
    ds.transform = transform if transform is not None else standard_transform()
    return ds


class CombinedPool(Dataset):
    """pub + priv concatenated. Indices [0, n_pub) are pub; [n_pub, total) are priv.

    Used for both shadow training (each shadow trains on a 50% subset of this
    combined pool, so every sample — pub OR priv — is IN for ~half the shadows
    and OUT for the other half) and for scoring shadows on all 28k samples
    uniformly. This gives priv samples per-sample IN/OUT statistics, which is
    what makes online LiRA work on priv at submission time.
    """

    def __init__(self, transform=None):
        pub = load_pub(transform=None)
        priv = load_priv(transform=None)
        self.imgs = list(pub.imgs) + list(priv.imgs)
        self.labels = list(pub.labels) + list(priv.labels)
        self.ids = list(pub.ids) + list(priv.ids)
        self.n_pub = len(pub.ids)
        self.n_priv = len(priv.ids)
        self.transform = transform if transform is not None else standard_transform()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform is not None:
            img = self.transform(img)
        return self.ids[idx], img, self.labels[idx]


def load_combined(transform=None) -> CombinedPool:
    return CombinedPool(transform=transform)


def predict_collate(batch):
    """Collate for forward-pass loaders. Drops membership so priv (all None) works."""
    ids = [b[0] for b in batch]
    imgs = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return ids, imgs, labels
