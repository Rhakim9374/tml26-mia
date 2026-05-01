"""Shared evaluation utilities."""

from __future__ import annotations

import numpy as np


def tpr_at_fpr(scores: np.ndarray, members: np.ndarray, fpr: float = 0.05) -> float:
    """Highest TPR achievable while keeping FPR ≤ `fpr`. Higher score → more member-y."""
    order = np.argsort(-scores)
    m = members[order]
    n_pos = int((members == 1).sum())
    n_neg = int((members == 0).sum())
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
