"""X27 gate: numerical equivalence of FMP's propagation against official DGL GraphConv.

FMP propagates with

    self.propa = GraphConv(in_feats, in_feats, weight=False, bias=False, activation=None)
    y = gamma * hh + (1 - gamma) * self.propa(g, feat=x)

on a graph built exactly as `FMP-main/main.py` builds it:

    adj  = load_pokec(...)          # symmetric, already + sp.eye
    g    = dgl.from_scipy(adj)
    g    = dgl.remove_self_loop(g)
    g    = dgl.add_self_loop(g)

With `weight=False, bias=False, norm='both'` that is exactly
`D^-1/2 (A + I) D^-1/2 X`, so the torch-only reproduction can be checked against
the official implementation instead of assumed equal (X27 section 11; a HARD STOP
if equivalence cannot be shown).

This script imports nothing from the project, so the same file runs in the
isolated DGL oracle venv and in the main environment:

    # oracle venv (official DGL)
    /home/sypark/x27_dgl_env/bin/python x27_graphconv_equiv.py --mode dgl   --out /tmp/.../dgl.npz
    # main env (torch-only reproduction)
    /home/sypark/miniconda3/envs/dev/bin/python x27_graphconv_equiv.py --mode torch --out /tmp/.../torch.npz
    # compare
    python x27_graphconv_equiv.py --compare /tmp/.../dgl.npz /tmp/.../torch.npz --tol 1e-5
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import scipy.sparse as sp
import torch


def pokec_adj(csv, rel, max_nodes=None):
    """The adjacency `load_pokec` builds, optionally restricted to the first N nodes."""
    import pandas as pd
    df = pd.read_csv(csv)
    idx = np.array(df["user_id"], dtype=int)
    if max_nodes is not None:
        idx = idx[:max_nodes]
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(rel, dtype=int)
    keep = np.array([(u in idx_map and v in idx_map) for u, v in edges_unordered])
    e = edges_unordered[keep]
    edges = np.array([[idx_map[u], idx_map[v]] for u, v in e], dtype=int).reshape(-1, 2)
    n = len(idx)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])          # load_pokec adds the identity here
    return adj.tocoo(), n


def synthetic_adj(n=64, p=0.08, seed=0):
    rng = np.random.default_rng(seed)
    a = (rng.random((n, n)) < p).astype(np.float32)
    a = np.triu(a, 1)
    a = a + a.T
    adj = sp.coo_matrix(a) + sp.eye(n)
    return adj.tocoo(), n


def features(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float32)


def run_dgl(adj, x):
    import dgl
    from dgl.nn.pytorch import GraphConv
    g = dgl.from_scipy(adj)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    conv = GraphConv(x.shape[1], x.shape[1], weight=False, bias=False, activation=None)
    with torch.no_grad():
        return conv(g, feat=x).numpy()


def run_torch(adj, x):
    """D^-1/2 (A + I) D^-1/2 X, self-loops present exactly once, as DGL's norm='both'."""
    a = adj.tocsr()
    a.setdiag(0)                     # remove_self_loop
    a.eliminate_zeros()
    a = a + sp.eye(a.shape[0], format="csr")   # add_self_loop
    a = a.tocoo()
    deg = np.asarray(a.sum(axis=1)).ravel()
    dinv = np.power(deg, -0.5, where=deg > 0)
    dinv[deg == 0] = 0.0
    vals = dinv[a.row] * a.data * dinv[a.col]
    i = torch.tensor(np.vstack([a.row, a.col]), dtype=torch.long)
    m = torch.sparse_coo_tensor(i, torch.tensor(vals, dtype=torch.float32),
                                (a.shape[0], a.shape[0])).coalesce()
    with torch.no_grad():
        return torch.sparse.mm(m, x).numpy()


def cases(args):
    out = {}
    adj, n = synthetic_adj(seed=0)
    out["synthetic"] = (adj, features(n, 8, 1))
    for name, csv, rel in (("pokec_z", "data/pokec/region_job.csv",
                            "data/pokec/region_job_relationship.txt"),
                           ("pokec_n", "data/pokec/region_job_2.csv",
                            "data/pokec/region_job_2_relationship.txt")):
        if not os.path.exists(csv):
            continue
        if args.subset:
            adj, n = pokec_adj(csv, rel, max_nodes=args.subset)
            out[f"{name}_subset{args.subset}"] = (adj, features(n, 64, 2))
        if args.full:
            adj, n = pokec_adj(csv, rel, max_nodes=None)
            out[f"{name}_full"] = (adj, features(n, 64, 3))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("dgl", "torch"))
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--subset", type=int, default=5000)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()

    if a.compare:
        A, B = (dict(np.load(p)) for p in a.compare)
        keys = sorted(set(A) & set(B))
        worst, ok = 0.0, True
        print(f"{'case':<22}{'max |dgl - torch|':>20}{'rel':>12}   verdict")
        for k in keys:
            d = float(np.max(np.abs(A[k] - B[k])))
            scale = float(np.max(np.abs(A[k]))) or 1.0
            worst = max(worst, d / scale)
            good = d / scale <= a.tol
            ok &= good
            print(f"{k:<22}{d:>20.3e}{d / scale:>12.2e}   {'ok' if good else 'FAIL'}")
        print(f"\nworst relative difference {worst:.2e} vs tolerance {a.tol:g}: "
              f"{'EQUIVALENT' if ok else 'NOT EQUIVALENT (hard stop)'}")
        return 0 if ok else 2

    random.seed(0)
    res = {}
    for name, (adj, x) in cases(a).items():
        y = run_dgl(adj, x) if a.mode == "dgl" else run_torch(adj, x)
        res[name] = y
        print(f"{a.mode}: {name} n={adj.shape[0]} nnz={adj.nnz} -> out {y.shape} "
              f"mean {y.mean():+.6f} std {y.std():.6f}")
    np.savez(a.out, **res)
    print(f"[written] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
