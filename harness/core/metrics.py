"""Evaluation metrics, numpy only (no sklearn dependency).

DP and EO are computed from hard predictions at a 0.5 threshold under *uniform*
weighting, whatever phi the training used. Evaluating with the training weights
would let a method look fair by down-weighting the nodes it fails on.
"""
from __future__ import annotations

import numpy as np


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), ties averaged."""
    y = np.asarray(y).astype(int)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)

    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):                      # average ranks within ties
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1

    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def evaluate(prob: np.ndarray, y: np.ndarray, sens: np.ndarray,
             idx: np.ndarray, threshold: float = 0.5) -> dict:
    p, yy, ss = prob[idx], y[idx].astype(int), sens[idx].astype(int)
    pred = (p >= threshold).astype(int)

    def rate(mask):
        return float(pred[mask].mean()) if mask.sum() > 0 else float("nan")

    dp = abs(rate(ss == 0) - rate(ss == 1))
    eo = abs(rate((ss == 0) & (yy == 1)) - rate((ss == 1) & (yy == 1)))

    return dict(
        acc=float((pred == yy).mean()),
        auc=roc_auc(yy, p),
        dp=float(dp),
        eo=float(eo),
        n=int(len(idx)),
    )


def selection_score(m: dict, rho0: float = 0.3) -> float:
    """Validation score for early stopping: accuracy minus weighted disparity."""
    return m["acc"] - rho0 * m["dp"] - (1 - rho0) * m["eo"]
