"""X30 EDITS adapter: `algorithms/EDITS.py` unmodified, driven as the repository
wrapper `utils/train_baselines.py` (EDITS branch) drives it, with wrappers only.

Intervention I: EDITS's debiasing stage, `EDITS.fit` -- attribute debiasing
(`X_debaising`, then the `truncation` lowest-weight columns zeroed) and
structural debiasing (`Adj_renew`, then `binarize` at `threshold_proportion`
inside `predict`). EDITS is a pre-processing method: `predict` trains its
downstream GCN on whatever `(self.X_debiased, self.adj1)` the stage produced.

    M+I   fit(...) then predict(...)                         (the wrapper path)
    M-I   the stage is bypassed: X_debiased = the same preprocessed features,
          adj1 = the original adjacency, then the same predict(...). With
          adj1 == adj_ori, `the_con1` is identically zero, so `binarize` edits
          nothing and A_debiased == adj_ori exactly (checked, not assumed).

Configuration (repository evidence only):
    `utils/param.json` EDITS[dataset] = {dropout, threshold_proportion};
    lr 1e-3, weight_decay 1e-5 (train_baselines CLI defaults); fit epochs 100 on
    german, otherwise min(500, 500); features / column norm on german and credit;
    load_data(feature_normalize=False). All as in train_baselines.py:567-587.

One routing repair, recorded: train_baselines maps the param.json key `dropout`
onto `args.dropout` (PARAM_TO_ARGS), which the EDITS branch never reads, so the
remaining key `threshold_proportion` becomes `param1` and is passed as the
debiaser's `dropout`, while `predict` receives `threshold_proportion=param2=1`,
the placeholder default -- under which no edge can ever be edited and the
structural half of I silently never runs. The wrapper's positional intent
(param1 = dropout, param2 = threshold_proportion, the key order in param.json)
is restored. No value is invented; both come from param.json.

Wrappers: `seed_all` injected immediately before `predict` in both arms
(`EDITS.optimize` reseeds the global generators to 10 every epoch, only in M+I,
which would otherwise give every M+I run the same downstream initialisation);
`GCN.forward` wrapped for per-epoch logits; `normalize_scipy` wrapped inside
`predict` to count edited entries; `GCN.__init__` wrapped to hash the generator
state at construction.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os

import numpy as np
import scipy.sparse as sp
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = ("german", "bail", "credit")
PKEY = {"bail": "recidivism"}
LR, WD = 1e-3, 1e-5
TRUNCATION = 4                     # EDITS.fit default, not passed by the wrapper


def config(dataset):
    p = json.load(open(os.path.join(ROOT, "utils", "param.json")))["EDITS"][PKEY.get(dataset, dataset)]
    if not isinstance(p, dict):
        raise KeyError(f"EDITS has no param.json entry for {dataset}")
    return dict(dropout=float(p["dropout"]), threshold_proportion=float(p["threshold_proportion"]),
                lr=LR, weight_decay=WD, fit_epochs=(100 if dataset == "german" else min(500, 500)),
                column_norm=dataset in ("credit", "german"))


def _rng_hash():
    h = hashlib.sha1(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        h.update(torch.cuda.get_rng_state().numpy().tobytes())
    return h.hexdigest()[:16]


def train_arm(dataset, encoder, arm, split, seed, epochs, device="cuda"):
    from core.trajectory import seed_all
    from utils.dataloading import load_data
    import FairGate.models.algorithms.EDITS as E
    cfg = config(dataset)
    adj, feats, labels, itr, iva, ite, sens, sens_idx = load_data(
        dataset, feature_normalize=False, split_seed=split)
    if cfg["column_norm"]:
        feats = feats / feats.norm(dim=0)

    scores, st = [], dict(armed=False, init_rng=None, nfeat=None, edited=None,
                          fit_ran=False, zero_cols=None)
    orig_fwd, orig_init, orig_norm = E.GCN.forward, E.GCN.__init__, E.normalize_scipy
    orig_fit = E.EDITS.fit
    coo = adj.coalesce()
    idx_np, val_np = coo.indices().cpu().numpy(), coo.values().cpu().numpy()
    adj_ori = sp.coo_matrix((val_np, (idx_np[0], idx_np[1])), shape=adj.shape).tocsr()

    def fwd(self, *a, **k):
        out = orig_fwd(self, *a, **k)
        if self.training:
            st["armed"] = True
        elif st["armed"]:
            scores.append(out.squeeze().detach().float().cpu().numpy())
            st["armed"] = False
        return out

    def init(self, nfeat, nhid, nclass, dropout):
        st["init_rng"], st["nfeat"] = _rng_hash(), int(nfeat)
        orig_init(self, nfeat, nhid, nclass, dropout)

    def norm(mx):
        if st["edited"] is None:           # the first call in predict: binarized A
            st["edited"] = int((sp.csr_matrix(mx) != adj_ori).nnz)
        return orig_norm(mx)

    def fit(self, *a, **k):
        st["fit_ran"] = True
        return orig_fit(self, *a, **k)

    cwd = os.getcwd()
    run_dir = os.path.join(ROOT, "harness", "results", "x30", "cache", "edits_cwd", str(os.getpid()))
    os.makedirs(run_dir, exist_ok=True)
    try:
        E.GCN.forward, E.GCN.__init__, E.EDITS.fit = fwd, init, fit
        os.chdir(run_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            model = E.EDITS(feats, dropout=cfg["dropout"], lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
            if arm == "plus":
                model.fit(adj, feats, sens, itr, iva, half=False, device=device,
                          epochs=cfg["fit_epochs"])
            else:
                model.X_debiased = feats.clone().to(device)
                model.adj1 = adj_ori.copy()
            st["zero_cols"] = int((model.X_debiased.abs().sum(0) == 0).sum())
            E.normalize_scipy = norm
            seed_all(seed * 1000 + split + 1)
            model.predict(adj, labels, sens, itr, iva, ite, epochs=epochs, lr=cfg["lr"],
                          weight_decay=cfg["weight_decay"],
                          threshold_proportion=cfg["threshold_proportion"], device=device)
    finally:
        os.chdir(cwd)
        E.GCN.forward, E.GCN.__init__, E.normalize_scipy = orig_fwd, orig_init, orig_norm
        E.EDITS.fit = orig_fit
    if len(scores) != epochs:
        raise SystemExit(f"[edits] {len(scores)} evaluations for {epochs} epochs")

    zero_base = int((feats.abs().sum(0) == 0).sum())
    tag = "M+I" if arm == "plus" else "M-I"
    if arm == "plus":
        checks = {
            f"{tag} debiasing stage ran": (st["fit_ran"], ""),
            f"{tag} debiasing changed features or structure":
                (st["zero_cols"] > zero_base or (st["edited"] or 0) > 0,
                 f"(zeroed columns {st['zero_cols'] - zero_base}, edited adjacency entries {st['edited']})"),
        }
    else:
        checks = {
            f"{tag} debiasing stage bypassed": (not st["fit_ran"], ""),
            f"{tag} adjacency and features are the originals":
                (st["edited"] == 0 and st["zero_cols"] == zero_base,
                 f"(edited entries {st['edited']}, extra zero columns {st['zero_cols'] - zero_base})"),
        }

    def pair_checks(other):
        return {"downstream GCN constructed from the same generator state":
                (st["init_rng"] == other["gate"]["init_rng"],
                 f"(input width M+I {st['nfeat']} vs M-I {other['gate']['nfeat']})")}

    return dict(scores=scores, code_epoch=epochs - 1, gate=st, gate_checks=checks,
                pair_checks=pair_checks, provenance="local-unverified (utils/param.json; routing repaired)",
                masks=tuple(np.isin(np.arange(feats.shape[0]), i.cpu().numpy()) for i in (itr, iva, ite)),
                config=dict(cfg, arm=arm, edited_entries=st["edited"],
                            zeroed_columns=st["zero_cols"] - zero_base))
