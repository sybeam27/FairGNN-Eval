"""X30 FairEdit adapter: `algorithms/FairEdit.py` unmodified, driven as the
repository wrapper `utils/train_baselines.py` (FairEdit branch) drives it.

Intervention I: gradient-guided fair graph editing, `fair_edit_trainer.
fair_graph_edit`, called once per epoch while `epoch < edit_num`
(FairEdit.py:780). The trainer exposes the switch itself:

    M+I   edit_num = 10   (fit() default; the wrapper never passes it)
    M-I   edit_num = 0    (the loop condition is never true; nothing else moves)

Configuration (repository evidence only): `utils/param.json` FairEdit[dataset]
= {weight_decay, hidden}, both mapped to args by train_baselines; lr 1e-3 (CLI
default); dropout 0.2 and model 'gcn' as the wrapper passes them;
load_data(feature_normalize = dataset == 'german'); fit() then applies its own
feature_norm keeping the sensitive column. `predict()` hard-codes
`trainer.train(epochs=100)`; the controlled horizon H is given to the same
`trainer.train` directly.

Wrappers: `GCN.forward` for per-epoch logits (first eval-mode call after each
training step; the counterfactual, noisy and explainer forwards come later in
the epoch); `fair_graph_edit` call counter; parameter hash before training;
a private working directory per process (checkpoint namespace separation).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = ("german", "bail", "credit")
PKEY = {"bail": "recidivism"}
EDIT_NUM_PLUS = 10


def config(dataset):
    p = json.load(open(os.path.join(ROOT, "utils", "param.json")))["FairEdit"][PKEY.get(dataset, dataset)]
    if not isinstance(p, dict):
        raise KeyError(f"FairEdit has no param.json entry for {dataset}")
    return dict(weight_decay=float(p["weight_decay"]), hidden=int(p["hidden"]), lr=1e-3,
                dropout=0.2, model_name="gcn", feature_normalize=(dataset == "german"))


def train_arm(dataset, encoder, arm, split, seed, epochs, device="cuda"):
    from utils.dataloading import load_data
    import core.paths  # noqa: F401  (puts the repository root on sys.path)
    import models.algorithms.FairEdit as FE
    cfg = config(dataset)
    adj, feats, labels, itr, iva, ite, sens, sens_idx = load_data(
        dataset, feature_normalize=cfg["feature_normalize"], split_seed=split)
    edit_num = EDIT_NUM_PLUS if arm == "plus" else 0

    scores, st = [], dict(armed=False, edits=0, init_hash=None, e0=None, e1=None)
    orig_fwd, orig_edit = FE.GCN.forward, FE.fair_edit_trainer.fair_graph_edit

    def fwd(self, *a, **k):
        out = orig_fwd(self, *a, **k)
        if self.training:
            st["armed"] = True
        elif st["armed"]:
            scores.append(out.squeeze().detach().float().cpu().numpy())
            st["armed"] = False
        return out

    def edit(self):
        st["edits"] += 1
        return orig_edit(self)

    cwd = os.getcwd()
    run_dir = os.path.join(ROOT, "harness", "results", "x30", "cache", "fairedit_cwd", str(os.getpid()))
    os.makedirs(run_dir, exist_ok=True)
    try:
        FE.GCN.forward, FE.fair_edit_trainer.fair_graph_edit = fwd, edit
        os.chdir(run_dir)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            m = FE.FairEdit()
            m.fit(adj, feats, labels, itr, iva, ite, sens, sens_idx, model_name=cfg["model_name"],
                  epochs=epochs, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                  hidden=cfg["hidden"], dropout=cfg["dropout"], edit_num=edit_num, device=device)
            tr = m.trainer
            h = hashlib.sha1()
            for k_, v in tr.model.state_dict().items():
                h.update(k_.encode()); h.update(v.detach().cpu().numpy().tobytes())
            st["init_hash"] = h.hexdigest()[:16]
            st["e0"] = int(tr.edge_index.shape[1])
            tr.train(epochs=epochs)
            st["e1"] = int(tr.edge_index.shape[1])
    finally:
        os.chdir(cwd)
        FE.GCN.forward, FE.fair_edit_trainer.fair_graph_edit = orig_fwd, orig_edit
    if len(scores) != epochs:
        raise SystemExit(f"[fairedit] {len(scores)} evaluations for {epochs} epochs")

    # the trainer's own selector: strict `loss_val < best_loss`, best_loss = 100
    yv = labels.cpu().numpy()[iva.cpu().numpy()].astype(float)
    best, code = 100.0, -1
    for e, v in enumerate(scores):
        z = v[iva.cpu().numpy()].astype(np.float64)
        bce = float((np.logaddexp(0.0, -np.abs(z)) + np.maximum(z, 0.0) - z * yv).mean())
        if bce < best:
            best, code = bce, e

    tag = "M+I" if arm == "plus" else "M-I"
    if arm == "plus":
        checks = {f"{tag} fair_graph_edit called edit_num times":
                  (st["edits"] == min(EDIT_NUM_PLUS, epochs), f"({st['edits']})"),
                  f"{tag} graph edited": (st["e1"] != st["e0"], f"(edges {st['e0']} -> {st['e1']})")}
    else:
        checks = {f"{tag} fair_graph_edit never called": (st["edits"] == 0, f"({st['edits']})"),
                  f"{tag} graph unchanged": (st["e1"] == st["e0"], f"(edges {st['e0']} -> {st['e1']})")}

    def pair_checks(other):
        return {"paired arms share initial parameters":
                (st["init_hash"] == other["gate"]["init_hash"], "")}

    n = feats.shape[0]
    return dict(scores=scores, code_epoch=code, gate=st, gate_checks=checks, pair_checks=pair_checks,
                provenance="local-unverified (utils/param.json)",
                masks=tuple(np.isin(np.arange(n), i.cpu().numpy()) for i in (itr, iva, ite)),
                config=dict(cfg, edit_num=edit_num, edges_before=st["e0"], edges_after=st["e1"]))
