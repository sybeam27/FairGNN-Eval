"""
Backbone: a 2-layer GCN written in plain torch (CPU-friendly, no PyG).

The graphs from core/datasets.py already carry self-loops and are symmetric, so
GCN propagation is just

    H' = D^-1/2 A D^-1/2 H W

which is exactly what PyG's GCNConv computes. Doing it directly keeps harness/
free of torch_geometric and lets edge weights be perturbed cheaply, which the
structure-level objective needs on every step.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_adj(edge_index: torch.Tensor, n: int,
                   edge_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Sparse D^-1/2 A D^-1/2 for an edge list that already has self-loops."""
    src, dst = edge_index[0], edge_index[1]
    dev = edge_index.device
    w = torch.ones(edge_index.size(1), device=dev) if edge_weight is None else edge_weight

    deg = torch.zeros(n, device=dev).scatter_add_(0, src, w)
    dinv = deg.clamp(min=1e-12).pow(-0.5)
    vals = dinv[src] * w * dinv[dst]

    return torch.sparse_coo_tensor(edge_index, vals, (n, n)).coalesce()


class GCN(nn.Module):
    """2 layers, scalar logit out. Optional uncertainty head for the sigma signal."""

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.5,
                 uncertainty_head: bool = False):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, 1)
        self.unc = nn.Linear(hidden, 1) if uncertainty_head else None
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        h = torch.sparse.mm(adj, self.lin1(x)).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        logit = torch.sparse.mm(adj, self.lin2(h)).squeeze(-1)
        sigma = None
        if self.unc is not None:
            sigma = F.softplus(self.unc(h)).squeeze(-1) + 1e-6
        return logit, h, sigma


def perturb_cross_group(edge_index: torch.Tensor, sens: torch.Tensor,
                        p: float, mode: str, generator=None) -> torch.Tensor:
    """Edge weights for the counterfactual graph used by the structure loss.

    mode='drop'  : delete a fraction p of cross-group edges
    mode='scale' : attenuate every cross-group edge to (1 - p)

    The draft picks the mode by regime (scale where boundary nodes are sparse,
    so that deleting them would fragment the graph; drop where cross-group
    edges are abundant). That choice is a caller argument here, not baked in,
    so it can be held fixed while the ranking signal varies.
    """
    src, dst = edge_index[0], edge_index[1]
    dev = edge_index.device
    is_cross = sens[src] != sens[dst]
    w = torch.ones(edge_index.size(1), device=dev)

    if mode == "scale":
        w[is_cross] = 1.0 - p
    elif mode == "drop":
        # Drawn on CPU on purpose: a CUDA generator yields a different stream
        # for the same seed, and runs must be comparable across devices.
        keep = torch.rand(int(is_cross.sum()), generator=generator) >= p
        w[is_cross] = keep.to(dev).float()
    else:
        raise ValueError(f"mode must be 'drop' or 'scale', got {mode!r}")
    return w


def to_tensors(graph, device: str = "cpu") -> dict:
    """numpy Graph -> the tensors the training loop needs."""
    t = dict(
        x=torch.tensor(graph.x, dtype=torch.float32, device=device),
        y=torch.tensor(graph.y, dtype=torch.float32, device=device),
        sens=torch.tensor(graph.sens, dtype=torch.long, device=device),
        edge_index=torch.tensor(graph.edge_index, dtype=torch.long, device=device),
    )
    for k in ("idx_train", "idx_val", "idx_test", "idx_fair"):
        t[k] = torch.tensor(np.asarray(getattr(graph, k)), dtype=torch.long, device=device)
    t["adj"] = normalized_adj(t["edge_index"], graph.n_nodes)
    return t
