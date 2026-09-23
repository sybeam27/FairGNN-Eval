"""
Graph loading for the harness/ re-experiments.

Why this exists instead of importing utils/data.py
--------------------------------------------------
utils/data.py needs torch + torch_geometric just to read an edge list, and it
rebuilds a 108 MB text file on every call. Everything the allocation study
needs is a numpy array, so this module reads the same raw files, applies the
same transformations, and caches the result as .npz.

It is a *replication*, not a rewrite: the graph half was verified to reproduce
the submitted Tables 5/6 exactly on all nine settings (see
harness/e0_diagnostics/provenance.csv). `verify_against_utils_data()` re-checks
that equivalence in an environment that has torch installed.

The provenance guard
--------------------
`load()` asserts that the graph it just built matches PAPER_STATS. The Credit
edge file in data/ was silently replaced at some point with a graph 9.4x
smaller (E=304,754 vs 2,873,716), which moved Credit out of the degree-skewed
regime and would have invalidated every degree-skewed result. That must never
pass unnoticed again, so a mismatch raises by default.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")

EPS = 1e-8

# n_nodes, n_edges, h, r_b, d_gap  -- from the submitted Tables 5 and 6.
PAPER_STATS = {
    "pokec_z":    (67796, 1303712, 0.953, 0.369, 0.083),
    "pokec_z_g":  (67796, 1303712, 0.479, 0.889, 0.024),
    "pokec_n":    (66569, 1100663, 0.956, 0.307, 0.056),
    "pokec_n_g":  (66569, 1100663, 0.489, 0.886, 0.011),
    "credit":     (30000, 2873716, 0.960, 0.677, 0.315),
    "recidivism": (18876,  642616, 0.536, 0.998, 0.023),
    "income":     (14821,  100483, 0.884, 0.334, 0.123),
    "german":     ( 1000,   44484, 0.809, 0.970, 0.049),
    "nba":        (  403,   21645, 0.729, 0.978, 0.096),
}

REGIME_PAPER = {
    "pokec_z": "clustered", "pokec_n": "clustered", "income": "clustered",
    "german": "saturated", "recidivism": "saturated", "nba": "saturated",
    "credit": "degree-skewed",
    "pokec_z_g": "mixed", "pokec_n_g": "mixed",
}

ALL_SETTINGS = list(PAPER_STATS)


# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------
@dataclass
class Graph:
    name: str
    x: np.ndarray             # (n, d) float32, min-max to [-1,1] except sens col
    y: np.ndarray             # (n,)   int64, binary
    sens: np.ndarray          # (n,)   int64, binary
    edge_index: np.ndarray    # (2, E) int64, symmetric, self-loops included
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    idx_fair: np.ndarray      # the set I over which every fairness term is computed
    stats: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def regime_paper(self) -> str:
        return REGIME_PAPER.get(self.name, "unknown")

    def __repr__(self) -> str:
        return (f"Graph({self.name}: n={self.n_nodes}, E={self.n_edges}, "
                f"h={self.stats['h']:.3f}, r_b={self.stats['r_b']:.3f}, "
                f"d_gap={self.stats['d_gap']:.3f}, |I|={len(self.idx_fair)})")


# ---------------------------------------------------------------------------
# Verbatim ports from utils/data.py
# ---------------------------------------------------------------------------
def _make_adj(edges: np.ndarray, n: int):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj + sp.eye(adj.shape[0])


def _feature_norm(x: np.ndarray, preserve_cols=()) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    lo, hi = x.min(axis=0), x.max(axis=0)
    denom = hi - lo
    denom[denom == 0] = 1.0
    normed = 2 * (x - lo) / denom - 1
    for c in preserve_cols:
        normed[:, c] = x[:, c]
    return normed


def _to_binary(values, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.int64)
        uniq = np.unique(arr)
        if len(uniq) != 2:
            raise ValueError(f"{name} must be binary, got {uniq}")
        return arr if set(uniq.tolist()) == {0, 1} else (arr == uniq.max()).astype(np.int64)
    s = pd.Series(arr).astype(str).str.strip().str.lower()
    uniq = pd.unique(s)
    if len(uniq) != 2:
        raise ValueError(f"{name} must be binary, got {uniq}")
    pos = {"1", "true", "yes", "y", ">50k", ">50k.", "high", "white", "male"}
    if any(u in pos for u in uniq):
        return s.isin(pos).astype(np.int64).to_numpy()
    return (s == sorted(uniq.tolist())[1]).astype(np.int64).to_numpy()


def _balanced_split(y: np.ndarray, label_number: int, seed: int):
    i0, i1 = np.where(y == 0)[0].tolist(), np.where(y == 1)[0].tolist()
    rng = random.Random(seed)
    rng.shuffle(i0)
    rng.shuffle(i1)
    tr = np.append(i0[:min(int(0.5 * len(i0)), label_number // 2)],
                   i1[:min(int(0.5 * len(i1)), label_number // 2)])
    va = np.append(i0[int(0.5 * len(i0)):int(0.75 * len(i0))],
                   i1[int(0.5 * len(i1)):int(0.75 * len(i1))])
    te = np.append(i0[int(0.75 * len(i0)):], i1[int(0.75 * len(i1)):])
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


def _pokec_split(y: np.ndarray, sens: np.ndarray, label_number: int,
                 sens_number: int, seed: int, test_idx: bool):
    rng = random.Random(seed)
    label_idx = np.where(y >= 0)[0].tolist()
    rng.shuffle(label_idx)
    tr = label_idx[:min(int(0.5 * len(label_idx)), label_number)]
    va = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        te = label_idx[label_number:]
        va = te
    else:
        te = label_idx[int(0.75 * len(label_idx)):]
    sens_valid = set(np.where(sens >= 0)[0].tolist())
    te = np.asarray(sorted(sens_valid & set(te)), dtype=np.int64)
    fair = sorted(sens_valid - set(va) - set(te.tolist()))
    rng.shuffle(fair)
    return (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64), te,
            np.asarray(fair[:sens_number], dtype=np.int64))


# ---------------------------------------------------------------------------
# Per-setting raw readers
# ---------------------------------------------------------------------------
def _read_edges(path: str) -> np.ndarray:
    """np.genfromtxt takes minutes on Credit's 108 MB file; pandas takes seconds."""
    raw = pd.read_csv(path, sep=r"\s+", header=None).to_numpy()
    return raw.astype(np.int64)


def _tabular(folder, csv, edge_file, sens_attr, predict_attr, drop_cols,
             label_number, split_seed, gender_fix=False):
    df = pd.read_csv(os.path.join(DATA, folder, csv)).copy()
    if gender_fix:
        g = df["Gender"].astype(str).str.strip()
        df["Gender"] = np.where(g == "Female", 1, np.where(g == "Male", 0, -1)).astype(int)

    header = [c for c in df.columns if c != predict_attr and c not in drop_cols]
    x = df[header].to_numpy(dtype=np.float32)
    y = df[predict_attr].to_numpy()
    y = np.where(y == -1, 0, y).astype(np.int64)          # german encodes -1
    if sens_attr == "race":
        sens = _to_binary(df[sens_attr].values, sens_attr)
    else:
        sens = df[sens_attr].to_numpy().astype(np.int64)

    edges = _read_edges(os.path.join(DATA, folder, edge_file))   # idx_map is identity
    tr, va, te = _balanced_split(y, label_number, split_seed)
    return x, y, sens, edges, header.index(sens_attr), tr, va, te, tr


def _pokec_like(folder, csv, rel_file, sens_attr, predict_attr,
                label_number, sens_number, split_seed, test_idx):
    df = pd.read_csv(os.path.join(DATA, folder, csv)).copy()
    header = [c for c in df.columns if c not in ("user_id", sens_attr, predict_attr)]
    x = df[header].to_numpy(dtype=np.float32)
    y = df[predict_attr].to_numpy().astype(np.int64)
    y = np.where(y > 1, 1, y)
    sens = df[sens_attr].to_numpy().astype(np.int64)

    idx_map = {j: i for i, j in enumerate(np.array(df["user_id"], dtype=np.int64))}
    raw = _read_edges(os.path.join(DATA, folder, rel_file))
    edges = np.array([idx_map[v] for v in raw.flatten()], dtype=np.int64).reshape(raw.shape)

    x = np.concatenate([x, sens.astype(np.float32)[:, None]], axis=1)   # sens as last feature
    tr, va, te, fair = _pokec_split(y, sens, label_number, sens_number, split_seed, test_idx)
    return x, y, sens, edges, x.shape[1] - 1, tr, va, te, fair


READERS = {
    "credit":     lambda s: _tabular("credit", "credit.csv", "credit_edges.txt",
                                     "Age", "NoDefaultNextMonth", {"Single"}, 6000, s),
    "recidivism": lambda s: _tabular("bail", "bail.csv", "bail_edges.txt",
                                     "WHITE", "RECID", set(), 100, s),
    "german":     lambda s: _tabular("german", "german.csv", "german_edges.txt",
                                     "Gender", "GoodCustomer",
                                     {"OtherLoansAtStore", "PurposeOfLoan"}, 100, s,
                                     gender_fix=True),
    "income":     lambda s: _tabular("income", "income.csv", "income_edges.txt",
                                     "race", "income", set(), 6000, s),
    "pokec_z":    lambda s: _pokec_like("pokec", "region_job.csv", "region_job_relationship.txt",
                                        "region", "I_am_working_in_field", 500, 200, s, False),
    "pokec_z_g":  lambda s: _pokec_like("pokec", "region_job.csv", "region_job_relationship.txt",
                                        "gender", "I_am_working_in_field", 500, 200, s, False),
    "pokec_n":    lambda s: _pokec_like("pokec", "region_job_2.csv", "region_job_2_relationship.txt",
                                        "region", "I_am_working_in_field", 500, 200, s, False),
    "pokec_n_g":  lambda s: _pokec_like("pokec", "region_job_2.csv", "region_job_2_relationship.txt",
                                        "gender", "I_am_working_in_field", 500, 200, s, False),
    "nba":        lambda s: _pokec_like("NBA", "nba.csv", "nba_relationship.txt",
                                        "country", "SALARY", 100, 50, s, True),
}


# ---------------------------------------------------------------------------
# Statistics (identical to signal_diagnostics.graph_stats)
# ---------------------------------------------------------------------------
def graph_stats(edge_index: np.ndarray, sens: np.ndarray, n: int) -> dict:
    src, dst = edge_index[0], edge_index[1]
    deg = np.zeros(n, dtype=float)
    np.add.at(deg, src, 1.0)
    d0 = float(deg[sens == 0].mean()) if np.any(sens == 0) else 0.0
    d1 = float(deg[sens == 1].mean()) if np.any(sens == 1) else 0.0

    is_inter = sens[src] != sens[dst]
    has_inter = np.zeros(n, dtype=bool)
    has_inter[src[is_inter]] = True
    n_cross = np.zeros(n, dtype=float)
    np.add.at(n_cross, src[is_inter], 1.0)

    return dict(
        h=float(np.mean(sens[src] == sens[dst])),
        r_b=float(has_inter.mean()),
        d_gap=abs(d0 - d1) / (d0 + d1 + EPS),
        deg=deg,
        r_cross=np.divide(n_cross, deg, out=np.zeros_like(n_cross), where=deg > 0),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load(name: str, split_seed: int = 20, use_cache: bool = True,
         check_provenance: bool = True) -> Graph:
    if name not in READERS:
        raise KeyError(f"unknown setting {name!r}; choose from {ALL_SETTINGS}")

    os.makedirs(CACHE, exist_ok=True)
    cache_path = os.path.join(CACHE, f"{name}_seed{split_seed}.npz")

    if use_cache and os.path.exists(cache_path):
        z = np.load(cache_path)
        x, y, sens, edge_index = z["x"], z["y"], z["sens"], z["edge_index"]
        tr, va, te, fair = z["idx_train"], z["idx_val"], z["idx_test"], z["idx_fair"]
    else:
        x, y, sens, edges, sens_col, tr, va, te, fair = READERS[name](split_seed)
        x = _feature_norm(x, preserve_cols=[sens_col])
        adj = _make_adj(edges, len(y)).tocoo()
        edge_index = np.vstack([adj.row, adj.col]).astype(np.int64)
        if use_cache:
            np.savez_compressed(cache_path, x=x, y=y, sens=sens, edge_index=edge_index,
                                idx_train=tr, idx_val=va, idx_test=te, idx_fair=fair)

    st = graph_stats(edge_index, sens, len(y))
    g = Graph(name=name, x=x, y=y, sens=sens, edge_index=edge_index,
              idx_train=tr, idx_val=va, idx_test=te, idx_fair=fair, stats=st)

    if check_provenance:
        _assert_provenance(g)
    return g


def _assert_provenance(g: Graph, tol: float = 5e-4) -> None:
    """Fail loudly if the graph on disk is not the graph the paper describes."""
    if g.name not in PAPER_STATS:
        return
    n, E, h, rb, dg = PAPER_STATS[g.name]
    problems = []
    if g.n_nodes != n:
        problems.append(f"n_nodes {g.n_nodes} != {n}")
    if g.n_edges != E:
        problems.append(f"n_edges {g.n_edges} != {E} (ratio {g.n_edges / E:.3f})")
    for key, want in (("h", h), ("r_b", rb), ("d_gap", dg)):
        got = g.stats[key]
        if abs(got - want) > tol:
            problems.append(f"{key} {got:.4f} != {want:.3f}")
    if problems:
        raise RuntimeError(
            f"[provenance] {g.name}: the graph on disk does not match Tables 5/6:\n  "
            + "\n  ".join(problems)
            + "\n  The raw data under data/ has changed. Do not run experiments "
              "against it until this is resolved (see data/credit/_quarantine_wrong_graph/)."
        )


def load_all(split_seed: int = 20, **kw) -> dict:
    return {n: load(n, split_seed=split_seed, **kw) for n in ALL_SETTINGS}


def verify_against_utils_data(name: str, split_seed: int = 20) -> dict:
    """Run this in an environment WITH torch to confirm this module is faithful."""
    import torch  # noqa: F401
    import sys
    sys.path.insert(0, REPO)
    from utils.data import get_dataset

    ref, _, _, _ = get_dataset(name, split_seed=split_seed)
    mine = load(name, split_seed=split_seed, use_cache=False)
    ref_ei = ref.edge_index.cpu().numpy()
    out = {
        "edge_index_equal": bool(np.array_equal(np.sort(ref_ei, axis=1),
                                                np.sort(mine.edge_index, axis=1))),
        "sens_equal": bool(np.array_equal(ref.sens.cpu().numpy().astype(int), mine.sens)),
        "y_equal": bool(np.array_equal(ref.y.cpu().numpy().astype(int), mine.y)),
        "x_close": bool(np.allclose(ref.x.cpu().numpy(), mine.x, atol=1e-5)),
        "n_train_equal": int(ref.train_mask.sum()) == len(mine.idx_train),
    }
    return out


if __name__ == "__main__":
    for nm in ALL_SETTINGS:
        print(load(nm))
