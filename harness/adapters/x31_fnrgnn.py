"""X31 FnRGNN adapter: the released `FnRGNN` class, unmodified, in a
harness-completed classification loop (protocol section 3).

FnRGNN (CIKM 2025) releases its model class and best configurations but no
training loop, and its `optimize()` fits every row of `data.y` without a mask.
The harness completes the loop as fixed in X31 section 3, identically in both
arms:

* criterion: `nn.BCEWithLogitsLoss` in place of `nn.MSELoss` (classification);
* every loss term on the training nodes of the common split only;
* `mmd_sample = 500`, the class's own sample size for its other sampled loss;
* H from the runner, one `model.eval()` forward over all nodes after each step.

`train_step` is `FnRGNN.optimize` with exactly those two changes (mask,
criterion); every loss component is the class's own method.

    M+I   use_mmd = use_gwn = use_edge_weight = True
    M-I   all three False  (the same GCN, unit edge weights, supervised loss only)

Configuration: logs/best_configs/<cfg>_best_joint.json (german_g,
region_job_r, region_job_2_r). Features standardised over all nodes with
StandardScaler, as FnRGNN's loader does; degree-0 nodes are kept (common graph).
Environment: /home/sypark/x30_edits_env (geomloss).
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "harness"))
from core.paths import repo as _repo  # noqa: E402

import contextlib
import hashlib
import importlib.util
import io
import json
import os

import numpy as np
import torch
import torch.nn as nn

FNR = _repo("FnRGNN-master")
DATASETS = ("german", "pokec_z", "pokec_n")
CONFIG_NAME = {"german": "german_g", "pokec_z": "region_job_r", "pokec_n": "region_job_2_r"}
MMD_SAMPLE = 500
APPLIED = ("hidden_dim", "dropout", "lambda2", "gamma", "lambda_dist", "lr", "weight_decay")
_MODS: dict = {}


def _load():
    """utils/model.py only, under a private name; its imports are third-party."""
    if "m" not in _MODS:
        spec = importlib.util.spec_from_file_location(
            "_x31_fnrgnn_model", os.path.join(FNR, "utils", "model.py"))
        m = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(m)
        _MODS["m"] = m
    return _MODS["m"]


def config(dataset):
    name = CONFIG_NAME[dataset]
    path = os.path.join(FNR, "logs", "best_configs", f"{name}_best_joint.json")
    p = json.load(open(path))["params"]
    missing = [k for k in APPLIED if k not in p]
    if missing:
        raise SystemExit(f"[fnrgnn] {name}_best_joint.json lacks {missing}")
    return dict({k: p[k] for k in APPLIED}, config_file=os.path.basename(path),
                mmd_sample=MMD_SAMPLE)


def _state_hash(model):
    h = hashlib.sha1()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def train_step(model, data, idx_train, bce, st):
    """FnRGNN.optimize (utils/model.py:902-927) with the loss terms restricted
    to the training nodes and the criterion BCE; nothing else changed."""
    model.train()
    y, h = model.forward(data)
    sensitive_attr = data.sensitive_attr[idx_train]
    y, h = y[idx_train], h[idx_train]
    target = data.y[idx_train].view(-1, 1)

    sup_loss = bce(y, target)
    mmd_loss = model.compute_mmd(h, sensitive_attr) if model.use_mmd \
        else torch.tensor(0.0, device=h.device, requires_grad=True)
    gwn_loss_1 = model.compute_sinkhorn_loss(y, sensitive_attr) if model.use_gwn \
        else torch.tensor(0.0, device=h.device, requires_grad=True)
    gwn_loss_2 = model.compute_dist(y, sensitive_attr) if model.use_gwn \
        else torch.tensor(0.0, device=h.device, requires_grad=True)
    gwn_loss = gwn_loss_1 + gwn_loss_2
    total_loss = sup_loss + model.lambda2 * mmd_loss + model.lambda_dist * gwn_loss

    model.optimizer.zero_grad()
    total_loss.backward()
    model.optimizer.step()

    st["steps"] += 1
    st["loss_rows"] = max(st["loss_rows"], int(target.shape[0]))
    st["mmd_nonzero"] += int(float(mmd_loss.detach()) != 0.0)
    st["gwn_nonzero"] += int(float(gwn_loss.detach()) != 0.0)


def train_arm(dataset, encoder, arm, split, seed, epochs, device="cuda"):
    from sklearn.preprocessing import StandardScaler
    from torch_geometric.data import Data
    from utils.dataloading import load_data
    if dataset not in DATASETS:
        raise SystemExit(f"FnRGNN has no configuration for {dataset!r}")
    M = _load()
    cfg = config(dataset)
    adj, feats, labels, itr, iva, ite, sens, _ = load_data(
        dataset, feature_normalize=False, split_seed=split)
    x = torch.tensor(StandardScaler().fit_transform(feats.cpu().numpy()), dtype=torch.float32)
    ei = adj.coalesce().indices()
    data = Data(x=x, edge_index=ei, y=labels.float(),
                sensitive_attr=(sens.cpu() > 0).long()).to(device)
    idx_train = itr.to(device)
    on = arm == "plus"

    model = M.FnRGNN(nfeat=x.shape[1], hidden_dim=int(cfg["hidden_dim"]), dropout=cfg["dropout"],
                     lm=cfg["lambda2"], gm=cfg["gamma"], ld=cfg["lambda_dist"],
                     mmd_sample=MMD_SAMPLE, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                     use_mmd=on, use_gwn=on, use_edge_weight=on).to(device)
    bce = nn.BCEWithLogitsLoss()
    st = dict(steps=0, loss_rows=0, mmd_nonzero=0, gwn_nonzero=0,
              init_hash=_state_hash(model), edge_w_all_one=None)
    model.eval()
    with torch.no_grad():
        ew = (model.compute_edge(data.x, data.edge_index, data.sensitive_attr)
              if model.use_edge_weight else torch.ones(data.edge_index.size(1), device=device))
    st["edge_w_all_one"] = bool(torch.all(ew == 1).item())

    scores = []
    for _ in range(epochs):
        train_step(model, data, idx_train, bce, st)
        model.eval()
        with torch.no_grad():
            y, _ = model(data)
        scores.append(y.squeeze().detach().float().cpu().numpy())

    n_train = int(idx_train.numel())
    tag = "M+I" if on else "M-I"
    checks = {f"{tag} BCE on exactly the training nodes": (
        st["loss_rows"] == n_train and st["steps"] == epochs,
        f"({st['loss_rows']} rows, {n_train} training nodes, {st['steps']} steps)")}
    if on:
        checks[f"{tag} MMD and GWN terms active every step"] = (
            st["mmd_nonzero"] == epochs and st["gwn_nonzero"] == epochs,
            f"(MMD {st['mmd_nonzero']}, GWN {st['gwn_nonzero']} of {epochs})")
        checks[f"{tag} edge reweighting active"] = (not st["edge_w_all_one"], "")
    else:
        checks[f"{tag} MMD and GWN terms identically zero"] = (
            st["mmd_nonzero"] == 0 and st["gwn_nonzero"] == 0,
            f"(MMD {st['mmd_nonzero']}, GWN {st['gwn_nonzero']})")
        checks[f"{tag} unit edge weights"] = (st["edge_w_all_one"], "")

    def pair_checks(other):
        return {"paired arms share initial parameters":
                (st["init_hash"] == other["gate"]["init_hash"], "")}

    n = x.shape[0]
    return dict(scores=scores, code_epoch=-1, gate=st, gate_checks=checks, pair_checks=pair_checks,
                provenance="official-repo class + best_joint config; harness-completed loop (X31 s3)",
                masks=tuple(np.isin(np.arange(n), i.cpu().numpy()) for i in (itr, iva, ite)),
                config=dict({k: cfg[k] for k in APPLIED}, config_file=cfg["config_file"],
                            mmd_sample=MMD_SAMPLE, arm=arm, criterion="BCEWithLogits",
                            loss_nodes="train"))
