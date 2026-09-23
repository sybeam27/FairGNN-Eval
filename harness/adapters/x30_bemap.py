"""X30 BeMap adapter: official `BeMap-main` models and training step, unmodified.

Runs in the isolated DGL environment (/home/sypark/x27_dgl_cuda). `models.py`
and `train_bemap.py` are loaded from BeMap-main with this repository's `utils`
package held aside during the import (BeMap ships its own `utils.py`).

Intervention I: balance-aware sampling. In training, `BeMap_GCN.forward(x,
epoch)` builds a fresh fair subgraph for the epoch (`get_subgraph`); with
`epoch == -1` it uses the full graph, which is what inference always does.

    M+I   train_bemap.train(epoch, ...)   -- the official training step
    M-I   train_bemap.train(-1, ...)      -- the same step through the model's
                                             own full-graph branch

Model, optimiser, loss and the full-graph evaluation are identical; only the
training graph differs. The evaluation after each step is what
`train_bemap.test` computes (`model.eval(); model(features, -1)`), recorded
over all nodes; the official `test` is not called because it selects on the
TEST split (train_bemap.py:136-142), which is why no native cell exists.

Configuration: the README's command `train_bemap.py --dataset <d> --model gcn`
at the script's parser defaults (lr 1e-3, weight_decay 1e-5, hidden 128,
dropout 0.5 [unused: use_dropout=False], beta 0.25, lam 0.5, save_num 4),
features through BeMap's own `feature_norm`. Datasets in its `--dataset`
choices that the study supports: pokec_z, bail, credit (nba excluded study-wide).

Wrappers: `get_subgraph` call counter and edge-count record; parameter hash
after construction; numpy's global generator consumed only by M+I sampling is
reseeded by the runner before each arm, so the paired arms start equal.
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "harness"))
from core.paths import repo as _repo  # noqa: E402  (repositories live under models/)

import argparse
import contextlib
import hashlib
import importlib.util
import io
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BM = _repo("BeMap-main")
DATASETS = ("pokec_z", "bail", "credit")
_MODS: dict = {}
_TEMPLATES: dict = {}


def _load():
    """Import once. train_bemap.py seeds numpy and torch (53) at module level,
    which would overwrite the runner's paired seeding for whichever arm triggers
    the import, so every generator state is saved before and restored after."""
    if _MODS:
        return _MODS["models"], _MODS["train"]
    import random as _r
    rng = (_r.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        return _load_inner()
    finally:
        _r.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])


def _load_inner():
    names = ("utils", "models", "layers")
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k in names or k.startswith("utils.")}
    saved_path, saved_argv = list(sys.path), list(sys.argv)
    try:
        sys.path.insert(0, BM)
        sys.argv = ["train_bemap.py"]          # its module-level parse_known_args
        mods = {}
        for nm, fn in (("models", "models.py"), ("train", "train_bemap.py")):
            spec = importlib.util.spec_from_file_location(f"_x30_bemap_{nm}", os.path.join(BM, fn))
            m = importlib.util.module_from_spec(spec)
            with contextlib.redirect_stdout(io.StringIO()):
                spec.loader.exec_module(m)
            mods[nm] = m
    finally:
        sys.path[:] = saved_path
        sys.argv = saved_argv
        for k in names:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    _MODS.update(mods)
    return mods["models"], mods["train"]


def config(encoder=None):
    """Parser defaults; `--model` from the encoder (X31: 'GAT' -> the script's
    own `--model gat` branch, train_bemap.py:115-118; otherwise the default gcn)."""
    _, T = _load()
    a = T.args
    return dict(lr=a.lr, weight_decay=a.weight_decay, hidden=a.hidden, dropout=a.dropout,
                beta=a.beta, lam=a.lam, save_num=a.save_num,
                model="gat" if encoder == "GAT" else a.model)


def _state_hash(model):
    h = hashlib.sha1()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def train_arm(dataset, encoder, arm, split, seed, epochs, device="cuda"):
    from utils.dataloading import load_data
    M, T = _load()
    if encoder not in (None, "GCN", "GAT"):
        raise SystemExit(f"BeMap has no {encoder!r} model")
    cfg = config(encoder)
    adj, feats, labels, itr, iva, ite, sens, sens_idx = load_data(
        dataset, feature_normalize=False, split_seed=split)
    features = T.feature_norm(feats)
    coo = adj.coalesce()
    i_, v_ = coo.indices().numpy(), coo.values().numpy()
    adj_sp = sp.coo_matrix((v_, (i_[0], i_[1])), shape=adj.shape).tocsr()
    sens_cpu = sens.cpu()

    st = dict(subgraphs=0, sub_edges=[], full_edges=None, init_hash=None)
    orig_sub = M.BeMap.get_subgraph

    def sub(self, epoch):
        g = orig_sub(self, epoch)
        st["subgraphs"] += 1
        if len(st["sub_edges"]) < 3:
            st["sub_edges"].append(int(g.num_edges()))
        return g

    scores = []
    try:
        M.BeMap.get_subgraph = sub
        with contextlib.redirect_stdout(io.StringIO()):
            if cfg["model"] == "gat":       # train_bemap.py:116-118, same arguments
                model = M.BeMap_GAT(nfeat=features.shape[1], nhid=cfg["hidden"], nclass=1,
                                    dropout=cfg["dropout"], graph=adj_sp, attrs=sens_cpu,
                                    lam=cfg["lam"], beta=cfg["beta"], save_num=cfg["save_num"],
                                    use_cuda=True)
            else:
                model = M.BeMap_GCN(nfeat=features.shape[1], nhid=cfg["hidden"], nclass=1,
                                    dropout=cfg["dropout"], graph=adj_sp, attrs=sens_cpu,
                                    lam=cfg["lam"], beta=cfg["beta"], save_num=cfg["save_num"],
                                    use_cuda=True, norm="both")
            st["init_hash"] = _state_hash(model)
            st["full_edges"] = int(model.graph.num_edges())
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            model.cuda()
            x, y, itr_c = features.cuda(), labels.cuda(), itr.cuda()
            for epoch in range(epochs):
                T.train(epoch if arm == "plus" else -1, model, opt, x, y, itr_c)
                model.eval()
                with torch.no_grad():
                    out = model(x, -1)                 # train_bemap.test's forward
                scores.append(out.squeeze().detach().float().cpu().numpy())
    finally:
        M.BeMap.get_subgraph = orig_sub

    tag = "M+I" if arm == "plus" else "M-I"
    if arm == "plus":
        checks = {f"{tag} a balanced subgraph is drawn every training epoch":
                  (st["subgraphs"] == epochs, f"({st['subgraphs']} of {epochs})"),
                  f"{tag} training subgraph is smaller than the full graph":
                  (bool(st["sub_edges"]) and max(st["sub_edges"]) < st["full_edges"],
                   f"(first subgraphs {st['sub_edges']} vs full {st['full_edges']})")}
    else:
        checks = {f"{tag} no subgraph is ever drawn": (st["subgraphs"] == 0, f"({st['subgraphs']})")}

    def pair_checks(other):
        return {"paired arms share initial parameters":
                (st["init_hash"] == other["gate"]["init_hash"], "")}

    n = features.shape[0]
    return dict(scores=scores, code_epoch=-1, gate=st, gate_checks=checks, pair_checks=pair_checks,
                provenance="official-repo (README command, parser defaults)",
                masks=tuple(np.isin(np.arange(n), i.cpu().numpy()) for i in (itr, iva, ite)),
                config=dict(cfg, arm=arm, full_edges=st["full_edges"]))
