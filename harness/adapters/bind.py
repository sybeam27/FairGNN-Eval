"""
BIND (Dong et al., AAAI'23) under our protocol, wrapping the official code.

Why this exists. E4/E5 found that the only allocation criterion that works is
`bind_influence` -- and that signal is our own first-order stand-in for BIND's
influence score, not BIND. It is no longer one signal among eleven; it carries
the entire positive result, so the published estimator has to replace ours
before anything is claimed. This module is that replacement.

What is and is not ours. `BIND-main/` is not modified, and nothing here writes
to it. The estimator is imported and called as published:

    implementations.approximator.grad_z_graph_faircost
    implementations.approximator.s_test_graph_cost      (damp 0.03, scale 60,
                                                         recursion_depth 5000)
    implementations.approximator.grad_z_graph
    implementations.approximator.cal_influence_graph
    implementations.approximator.cal_influence_graph_nodal
    implementations.GNNs.gcn.GCN                        (1-layer GCNConv + fc)

What is reimplemented here is only the *driver*. BIND ships it as three
top-level scripts rather than functions: `2_influence_computation_and_save.py`
loads a Windows-only `caffe2_nvrtc.dll` at import, hard-codes its own dataset
loaders and paths, and communicates with `3_removing_and_testing.py` through a
`final_influence_<dataset>.npy` file on disk. None of that is importable, and
none of it can read our splits. The logic below is transcribed from those two
scripts with the file hand-off replaced by a return value; every line that
computes something is marked with the script and line it came from, so the
transcription is checkable against the original.

The one substantive choice we make is the data: BIND's loaders are replaced by
`utils/data.py::get_dataset`, which is what every other method in the audit
reads, and which was verified index-for-index against `harness/core/datasets.py`
on all nine settings at three split seeds. Without that substitution BIND would
be evaluated on different splits from everything it is compared to.

Cost. The estimator is expensive by construction: 5000 Hessian-vector products
over the whole graph, then one gradient computation per training node with the
adjacency rebuilt each time. It also depends on the training split, so it must
be recomputed for every (setting, split) rather than once per setting.
`influence()` therefore caches to disk keyed by a content hash of the inputs,
not by dataset name -- `algorithms/FairGT_alg.py` caches a derived tensor by
name alone, and `harness/README.md` records why that is the shape of a repeat of
the Credit accident.
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "harness"))
from core.paths import repo as _repo  # noqa: E402  (repositories live under models/)

import hashlib
import os
import warnings
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIND = _repo("BIND-main")
if BIND not in sys.path:
    sys.path.insert(0, BIND)

from implementations.GNNs.gcn import GCN                      # noqa: E402
from implementations.approximator import (                    # noqa: E402
    cal_influence_graph, cal_influence_graph_nodal,
    grad_z_graph, s_test_graph_cost)

CACHE = os.path.join(ROOT, "harness", ".cache", "bind")

# BIND-main/README.md section 2(ii): "The hyper-parameters in approximator.py
# requires parameter-tuning. Example configurations for function
# s_test_graph_cost are provided here: for Income and Recidivism, set the scale
# as 25; for Pokec1, 100; for Pokec2, 60." The code's default is 60, so running
# every setting at the default would be running Income and Recidivism at a value
# the authors explicitly say is wrong for them.
#
# Three of our nine settings are not in BIND's paper at all -- German, NBA and
# Credit -- so no value is given for them and we have to choose one. We take the
# code default, 60, and say so: it is our choice, not BIND's, and any result on
# those three carries that caveat. Pokec1/Pokec2 in the README are the two
# region-attribute Pokec graphs; the gender variants are ours, not theirs, and
# inherit the same graph's value.
SCALE = {"income": 25, "recidivism": 25,
         "pokec_z": 100, "pokec_z_g": 100,
         "pokec_n": 60, "pokec_n_g": 60,
         "german": 60, "nba": 60, "credit": 60}
SCALE_FROM_PAPER = {"income", "recidivism", "pokec_z", "pokec_n"}

# LiSSA approximates H^-1 v by h <- v + (I - H/scale) h. The recursion diverges
# whenever `scale` is below the largest Hessian eigenvalue, and it diverges to
# NaN without raising: `cal_influence_graph` returns a vector of NaNs and
# `_order_and_budget`, scanning it for a sign change, finds none and returns
# max_num = 0. A divergent estimate is therefore indistinguishable, in the
# recorded columns, from a converged estimate that found nothing to delete.
#
# Measured 2026-09-12: at the scales above, Pokec-n and Pokec-n_g return 0 of
# 500 finite values, and NBA 0 of 100 on some cells. The audit read those as
# BIND's budget rule being empty. They were not; the estimator had diverged.
#
# So the scale is escalated along a fixed ladder until the estimate is finite.
# The criterion is finiteness alone -- it cannot depend on, or tune, any
# fairness or accuracy outcome. The rung actually used is returned so a run at
# a scale other than the published one is visible in the results.
SCALE_LADDER = [25, 60, 100, 150, 250, 400, 700, 1000]


def influence_converged(*args, scale=60, **kw):
    """`influence` escalated until finite. Returns (final, scale_used, n_steps).

    Raises if no rung converges: a cell with no usable estimate is reported,
    never silently handed on as a vector of NaNs.
    """
    steps = 0
    for rung in [scale] + [r for r in SCALE_LADDER if r > scale]:
        final = influence(*args, scale=rung, **kw)
        if np.isfinite(final).all():
            return final, rung, steps
        steps += 1
    raise RuntimeError(
        f"LiSSA did not converge at any scale up to {SCALE_LADDER[-1]}")


# --------------------------------------------------------------------------
# transcribed from 2_influence_computation_and_save.py
# --------------------------------------------------------------------------
def _find123Nei(G, node):
    """2_influence_computation_and_save.py:31. Returns [1-hop, 2-hop, 3-hop]
    neighbour lists; the caller uses only the first."""
    import networkx as nx
    nodes = list(nx.nodes(G))
    nei1_li, nei2_li, nei3_li = [], [], []
    for FNs in list(nx.neighbors(G, node)):
        nei1_li.append(FNs)
    for n1 in nei1_li:
        for SNs in list(nx.neighbors(G, n1)):
            nei2_li.append(SNs)
    nei2_li = list(set(nei2_li) - set(nei1_li))
    if node in nei2_li:
        nei2_li.remove(node)
    for n2 in nei2_li:
        for TNs in nx.neighbors(G, n2):
            nei3_li.append(TNs)
    nei3_li = list(set(nei3_li) - set(nei2_li) - set(nei1_li))
    if node in nei3_li:
        nei3_li.remove(node)
    return [nei1_li, nei2_li, nei3_li]


def _del_adj(adj_sp, harmful):
    """2_influence_computation_and_save.py:126 -- drop rows/cols, re-symmetrise,
    re-add self-loops."""
    mask = np.ones(adj_sp.shape[0], dtype=bool)
    mask[harmful] = False
    a = sp.coo_matrix(adj_sp.tocsr()[mask, :][:, mask])
    a = a + a.T.multiply(a.T > a) - a.multiply(a.T > a)
    return a + sp.eye(a.shape[0])


def _train_gcn(x, edge_index, y, idx_train, idx_val, nhid, dropout, lr, wd,
               epochs, device, seed, select="final"):
    """1_training.py:93 -- BCE-with-logits on the training nodes. BIND's GCN is
    one GCNConv plus a linear head, not the two-layer stack the rest of the
    audit uses; that is BIND's design and is kept.

    `select="final"` is the default because it is what BIND actually runs.
    `1_training.py:150-157` saves the checkpoint inside the loop whenever the
    validation loss improves, then saves again unconditionally after the loop,
    so the second write overwrites the selected model and everything downstream
    loads the last epoch. Reproducing the published log requires reproducing
    that. `select="best_val"` is the behaviour the code was evidently trying to
    have; it is offered so the two can be compared.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = GCN(x.shape[1], nhid, 1, dropout).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    best, best_state = np.inf, None
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        out = m(x, edge_index)
        loss = F.binary_cross_entropy_with_logits(
            out[idx_train], y[idx_train].unsqueeze(1).float())
        loss.backward(); opt.step()
        if select == "best_val":
            m.eval()
            with torch.no_grad():
                v = float(F.binary_cross_entropy_with_logits(
                    m(x, edge_index)[idx_val], y[idx_val].unsqueeze(1).float()))
            if v < best:
                best, best_state = v, {k: t.detach().clone()
                                       for k, t in m.state_dict().items()}
    if select == "best_val" and best_state is not None:
        m.load_state_dict(best_state)
    return m


def _key(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(np.ascontiguousarray(p).tobytes() if isinstance(p, np.ndarray)
                 else str(p).encode())
    return h.hexdigest()[:16]


def influence(x, edge_index, adj_sp, y, sens, idx_train, idx_val, idx_test,
              *, scale=60, nhid=128, dropout=0.5, lr=1e-3, wd=0.0, epochs=500,
              device="cuda", seed=27, use_cache=True):
    """BIND's per-training-node influence on the bias cost.

    Returns `final_influence`, one value per node of `idx_train`, positive
    meaning removing the node reduces the bias measure -- the sign convention
    of `3_removing_and_testing.py:243-247`, where the largest values are the
    'harmful' ones deleted first.
    """
    os.makedirs(CACHE, exist_ok=True)
    ck = os.path.join(CACHE, f"infl_{_key(edge_index.cpu().numpy(), y.cpu().numpy(), sens.cpu().numpy(), idx_train.cpu().numpy(), scale, nhid, dropout, lr, wd, epochs, seed)}.npy")
    if use_cache and os.path.exists(ck):
        return np.load(ck)

    dev = torch.device(device)
    x, edge_index, y, sens = (t.to(dev) for t in (x, edge_index, y, sens))
    model = _train_gcn(x, edge_index, y, idx_train, idx_val,
                       nhid, dropout, lr, wd, epochs, dev, seed)

    # 2_influence...:163 -- for each training node, the training nodes inside
    # its 1-hop computation graph
    import networkx as nx
    G = nx.Graph(adj_sp)
    tr = idx_train.cpu().numpy()
    tr_set = set(tr.tolist())
    involving = [list(set(_find123Nei(G, int(v))[0]) & tr_set) for v in tr]

    gpu = 0 if dev.type == "cuda" else -1
    # 2_influence...:173-175
    h_cost = s_test_graph_cost(edge_index, x, idx_train, idx_test, y, sens,
                               model, gpu=gpu, scale=scale)
    grads = grad_z_graph(edge_index, x, idx_train, y, model, gpu=gpu)
    influences, *_ = cal_influence_graph(idx_train, h_cost, grads, gpu=gpu)

    from torch_geometric.utils import convert
    non_iid = []
    for i in range(len(tr)):                                   # 2_influence:180
        if not involving[i]:
            non_iid.append(0.0)
            continue
        ref = list(range(adj_sp.shape[0]))                     # :185 reindex
        for j in range(len(ref) - tr[i]):
            ref[j + tr[i]] -= 1
        keep = np.ones(len(y.cpu()), dtype=bool)
        keep[tr[i]] = False
        idx_tr_d = torch.LongTensor(np.array(ref)[tr[np.arange(len(tr)) != i]])
        inv_d = torch.LongTensor(np.array(ref)[np.array(involving[i])])
        a_d = _del_adj(adj_sp, tr[i])
        ei_d = convert.from_scipy_sparse_matrix(a_d)[0].to(dev)
        g_d = grad_z_graph(ei_d, x[keep], inv_d, y[keep], model, gpu=gpu)
        inf_d, *_ = cal_influence_graph_nodal(idx_tr_d, inv_d, h_cost, g_d,
                                              gpu=gpu)
        non_iid.append(float(sum(inf_d)))

    order = {int(v): k for k, v in enumerate(tr)}              # 2_influence:217
    final = np.array([non_iid[i] - np.array(influences)[
        [order[int(t)] for t in involving[i] + [tr[i]]]].sum()
        for i in range(len(tr))])
    if np.isfinite(final).all():
        np.save(ck, final)
    else:
        # A diverged estimate must not be cached. It is cheap to re-read and
        # impossible to tell from a converged one downstream -- max_num comes
        # back 0 either way -- so caching it would make the divergence
        # permanent and invisible for every later run on this cell.
        warnings.warn(
            f"LiSSA did not converge at scale {scale}: "
            f"{int((~np.isfinite(final)).sum())} of {final.size} values "
            f"non-finite. Not cached. Use influence_converged() to escalate.",
            RuntimeWarning, stacklevel=2)
    return final


# --------------------------------------------------------------------------
# transcribed from 3_removing_and_testing.py
# --------------------------------------------------------------------------
def _metrics(logit, y, sens, idx):
    """3_removing_and_testing.py:272 `fair_metric`, plus the accuracy/AUC/F1 the
    same file records. Predictions are `output > 0`, i.e. a 0.5 threshold on the
    sigmoid, which is what the rest of the audit uses too."""
    from sklearn.metrics import f1_score, roc_auc_score
    o = logit[idx].detach().cpu().numpy().squeeze()
    yy = y[idx].detach().cpu().numpy()
    ss = sens[idx].detach().cpu().numpy()
    p = (o > 0).astype(int)
    s0, s1 = ss == 0, ss == 1
    s0y, s1y = s0 & (yy == 1), s1 & (yy == 1)
    f = lambda m: p[m].mean() if m.any() else 0.0        # noqa: E731
    return {"acc": float((p == yy).mean()),
            "auc": float(roc_auc_score(yy, o)) if len(np.unique(yy)) > 1 else 0.5,
            "f1": float(f1_score(yy, p)),
            "dp": float(abs(f(s0) - f(s1))),
            "eo": float(abs(f(s0y) - f(s1y)))}


def _order_and_budget(final, idx_train, involving):
    """3_removing_and_testing.py:243-268.

    Nodes are ranked most-harmful first, then thinned so that no two selected
    nodes share a training node in their 1-hop computation graph, and the budget
    stops where the influence changes sign -- past that point deletion is
    predicted to help unfairness rather than hurt it.
    """
    helpful_idx = np.argsort(final).tolist()
    harmful_idx = helpful_idx[::-1]

    total, masker = [], np.ones(len(harmful_idx), dtype=bool)
    for i in range(len(harmful_idx) - 1):
        if masker[i]:
            total += involving[harmful_idx[i]]
        if set(total) & set(involving[harmful_idx[i + 1]]):
            masker[i + 1] = False
    harmful_idx = np.array(harmful_idx)[masker]

    max_num = 0
    f = final[harmful_idx]
    for i in range(len(f) - 1):
        if f[i] * f[i + 1] <= 0:
            max_num = i + 1
            break
    return harmful_idx, max_num


def run(x, edge_index, adj_sp, y, sens, idx_train, idx_val, idx_test,
        *, scale=60, nhid=128, dropout=0.5, lr=1e-3, wd=0.0, epochs=500,
        device="cuda", seed=27, percentage_budget=0.3, use_cache=True):
    """One BIND run: influence, thinned ranking, then the deletion sweep.

    `3_removing_and_testing.py` sweeps k from 0 to `0.3 * max_num`, warm-starting
    each retraining from the *pretrained* model rather than from scratch, and
    saves the whole curve. It never picks an operating point -- the paper reports
    curves. A table needs one, so k is chosen on the **validation** split by the
    criterion the rest of the audit uses. That choice is ours and is not BIND's;
    the full curve is returned alongside so it can be checked.
    """
    from torch_geometric.utils import convert
    import networkx as nx

    dev = torch.device(device)
    n = x.shape[0]
    final, scale_used, scale_steps = influence_converged(
                      x, edge_index, adj_sp, y, sens, idx_train, idx_val,
                      idx_test, scale=scale, nhid=nhid, dropout=dropout, lr=lr,
                      wd=wd, epochs=epochs, device=device, seed=seed,
                      use_cache=use_cache)

    tr = idx_train.cpu().numpy()
    tr_set = set(tr.tolist())
    G = nx.Graph(adj_sp)
    involving = [list(set(_find123Nei(G, int(v))[0]) & tr_set) for v in tr]
    harmful_idx, max_num = _order_and_budget(final, idx_train, involving)
    harmful = tr[harmful_idx]

    x0, ei0, y0, s0 = (t.to(dev) for t in (x, edge_index, y, sens))
    base = _train_gcn(x0, ei0, y0, idx_train.to(dev), idx_val.to(dev),
                      nhid, dropout, lr, wd, epochs, dev, seed)
    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    curve = []
    for k in range(int(percentage_budget * max_num) + 1):
        drop = harmful[:k]
        keep = np.ones(n, dtype=bool)
        keep[drop] = False
        ref = np.cumsum(keep) - 1                       # :383 reindex, vectorised
        tr_k = torch.LongTensor(ref[tr[np.isin(np.arange(len(tr)),
                                               harmful_idx[:k], invert=True)]])
        va_k = torch.LongTensor(ref[idx_val.cpu().numpy()])
        te_k = torch.LongTensor(ref[idx_test.cpu().numpy()])

        ei_k = convert.from_scipy_sparse_matrix(
            _del_adj(adj_sp, drop) if k else adj_sp)[0].to(dev)
        xk, yk, sk = x0[keep], y0[keep], s0[keep]

        m = GCN(x.shape[1], nhid, 1, dropout).to(dev)
        m.load_state_dict(base_state)                   # :396 warm start
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
        best, best_state = np.inf, None
        for _ in range(epochs):
            m.train(); opt.zero_grad()
            out = m(xk, ei_k)
            F.binary_cross_entropy_with_logits(
                out[tr_k], yk[tr_k].unsqueeze(1).float()).backward()
            opt.step()
            m.eval()
            with torch.no_grad():
                v = float(F.binary_cross_entropy_with_logits(
                    m(xk, ei_k)[va_k], yk[va_k].unsqueeze(1).float()))
            if v < best:
                best, best_state = v, {a: b.detach().clone()
                                       for a, b in m.state_dict().items()}
        m.load_state_dict(best_state)
        m.eval()
        with torch.no_grad():
            lg = m(xk, ei_k)
        curve.append({"k": k,
                      **{f"val_{a}": b for a, b in _metrics(lg, yk, sk, va_k).items()},
                      **{f"test_{a}": b for a, b in _metrics(lg, yk, sk, te_k).items()}})

    # operating point: our choice, not BIND's -- same validation criterion the
    # rest of the audit selects with
    pick = min(curve, key=lambda r: r["val_dp"] + r["val_eo"])
    return {"selected": pick, "curve": curve, "max_num": int(max_num),
            "n_candidates": int(len(harmful_idx)),
            "scale_used": int(scale_used), "scale_steps": int(scale_steps)}
