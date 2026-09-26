"""Scientific pilot: does τ_pkg differ from τ_int?

One question, and nothing else is read from this run. Not a ranking, not a
component ordering, not a headline.

    τ_pkg   the published configuration against the common baseline, each
            keeping its own selector, with the final outcome recomputed by the
            unified evaluator G_c. The repository's own numbers are kept beside
            it as the reproduction view.

    τ_int   M₁ − M₀ for one method, at the same (dataset, backbone), the same
            split and run, the same actual horizon, the same shared selector
            σ_c^BCE, the same decision rule δ_c and the same G_c. Only the
            claimed intervention changes.

σ_c^AUC runs alongside as a robustness check, fixed before any result.

Outcome vectors, oriented so larger is better in both coordinates:

    primary     [ ΔAUC , −ΔDP ]
    secondary   [ ΔAUC , −ΔEO ]

Three runs give direction and magnitude. They are not a confidence interval and
are not reported as one.

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python harness/experiments/pilot_tau.py
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "harness"))

import core.paths  # noqa: F401,E402  (workspace on sys.path)
from core.evaluator import evaluate                    # noqa: E402
from core.published_config import native_config, published  # noqa: E402
from core.trajectory import (SplitRef, ValidationHistory,  # noqa: E402
                             eval_rng_seed, select_max_auc, select_min_bce)

# M₀ per method: every component its paper claims as fairness, switched off.
# `closure` says whether the components are crossed; FairGB's CAL is nested
# inside CNM and FairVGNN's two interact, so those get full-vs-both-off only.
METHODS = {
    "FairGNN":  dict(closure=True,  off=dict(alpha=0.0, beta=0.0)),
    "NIFTY":    dict(closure=True,  off=dict(sim_coeff=0.0)),
    "FairGB":   dict(closure=False, off=dict(use_cal=False, use_cnm=False)),
    "FairVGNN": dict(closure=False, off=dict(f_mask="no", weight_clip="no")),
}


def load(ds, split_seed, device, feature_normalize=False):
    """Exactly what utils/train_baselines hands each method: CPU tensors.

    The audit does not call to_device for GNN, NIFTY, FairGB or FairVGNN -- only
    FairGNN's branch moves anything, and each constructor handles the rest. A
    harness that tidied this up by moving everything to the device would not be
    running what the audit runs, and FairVGNN's internal get_dataset rejects
    CUDA tensors outright.
    """
    from utils.dataloading import load_data
    return load_data(ds, feature_normalize=feature_normalize,
                     split_seed=split_seed)


def _split_ref(y, idx, sens):
    idx = np.asarray(idx.cpu() if torch.is_tensor(idx) else idx)
    yv = np.asarray(y.detach().cpu() if torch.is_tensor(y) else y)
    sv = np.asarray(sens.detach().cpu() if torch.is_tensor(sens) else sens)
    return SplitRef(node_id=idx, y=yv[idx].astype(int), a=sv[idx].astype(int))


def outcome(y, sens, idx, score):
    """G_c on one split. δ_c is `score > 0` for every method in this pilot."""
    ref = _split_ref(y, idx, sens)
    return evaluate(ref.y, ref.a, raw_score=np.asarray(score), decision="score>0")


def vec(a: dict, b: dict, fair: str = "dp") -> tuple[float, float]:
    """[ΔAUC, −Δfair], oriented so larger is better in both coordinates."""
    return (a["auc"] - b["auc"], -(a[fair] - b[fair]))


def train(method, data, seed, epochs, device, off: dict | None = None,
          cfg: dict | None = None):
    """One trained model plus its validation trajectory.

    `off` switches the claimed intervention off; `None` is the published
    configuration. M₀ and M₁ differ in nothing else -- same data, same seed,
    same horizon, same instrumentation.

    Returns (score_fn, history, code_best_epoch), where `score_fn(state, idx)`
    evaluates raw scores on a split after restoring a checkpoint.
    """
    adj, f, y, itr, iva, ite, s, si = data
    off = dict(off or {})
    cfg = dict(cfg or {})
    cfg.pop("feature_normalize", None)      # applied at load(), not at fit()
    torch.manual_seed(seed); np.random.seed(seed)

    if method == "FairGNN":
        import itertools as it
        from models.algorithms.FairGNN import FairGNN
        from utils.train_baselines import to_device
        adj, f, y, itr, iva, ite, s = to_device(adj, f, y, itr, iva, ite, s, device)
        m = FairGNN(nfeat=f.shape[1],
                    acc=cfg.get("acc", 0.39),
                    alpha=off.get("alpha", cfg.get("alpha", 8)),
                    beta=off.get("beta", cfg.get("beta", 0.005))).to(device)
        m.args.num_hidden = cfg.get("num_hidden", 128); m.args.epochs = epochs
        G = list(it.chain(m.GNN.parameters(), m.classifier.parameters(),
                          m.estimator.parameters()))
        _lr = cfg.get("lr", 1e-3); _wd = cfg.get("weight_decay", 0.0)
        m.optimizer_G = torch.optim.Adam(G, lr=_lr, weight_decay=_wd)
        m.optimizer_A = torch.optim.Adam(m.adv.parameters(), lr=_lr, weight_decay=_wd)
        hist = ValidationHistory(_split_ref(y, iva, s), epochs,
                                 state_fn=lambda: {k: v.detach().clone()
                                                   for k, v in m.state_dict().items()})
        m.fit(adj, f, y, itr, iva, ite, s, itr, device=device, trajectory=hist)

        def score_fn(state, idx):
            if state is not None:
                m.load_state_dict(state)
            m.eval()
            with torch.no_grad():
                # features come from the closure: FairGNN takes them as a fit()
                # argument and does not keep them on the module
                out, _ = m(m.edge_index, f)
            return out.squeeze().detach().cpu().numpy()[np.asarray(idx)]
        return score_fn, hist, getattr(m, "best_epoch", -1)

    if method == "NIFTY":
        from models.algorithms.NIFTY import NIFTY
        _drops = {k: cfg[k] for k in ("drop_edge_rate_1", "drop_edge_rate_2",
                                      "drop_feature_rate_1", "drop_feature_rate_2")
                  if k in cfg}
        m = NIFTY(adj, f, y, itr, iva, ite, s, si,
                  num_hidden=cfg.get("num_hidden", 128),
                  num_proj_hidden=cfg.get("num_proj_hidden", 128),
                  lr=cfg.get("lr", 1e-3),
                  weight_decay=cfg.get("weight_decay", 1e-5), device=device,
                  sim_coeff=off.get("sim_coeff", cfg.get("sim_coeff", 0.5)),
                  **_drops)
        if cfg.get("restore_train_drop_rates"):
            # NIFTY.__init__ zeroes its training drop rates after building the
            # validation views (X13); native NIFTY applies them every epoch.
            # Restored here, after construction, without editing the class.
            for _k, _v in _drops.items():
                setattr(m, _k, _v)
        hist = ValidationHistory(_split_ref(y, iva, s), epochs + 1,
                                 state_fn=lambda: {k: v.detach().clone()
                                                   for k, v in m.state_dict().items()})
        m.fit(epochs=epochs, trajectory=hist)

        def score_fn(state, idx, replay=False):
            """Test scoring uses the clean graph, as NIFTY's own evaluation does.

            replay=True scores what NIFTY's loop actually records as validation
            output: the fixed augmented view (val_x_1, val_edge_index_1), in
            idx_val order. A restored checkpoint can only be checked against
            that view. It is also the input sigma_c reads for NIFTY, in Arm A
            and Arm B alike (X14).
            """
            if state is not None:
                m.load_state_dict(state)
            m.eval()
            with torch.no_grad():
                if replay:
                    out = m.forwarding_predict(m.forward(m.val_x_1, m.val_edge_index_1))
                    iv_ = m.idx_val.cpu() if torch.is_tensor(m.idx_val) else m.idx_val
                    return out.squeeze().detach().cpu().numpy()[np.asarray(iv_)]
                emb = m.forward(m.features.to(device), m.edge_index.to(device))
                out = m.forwarding_predict(emb)
            return out.squeeze().detach().cpu().numpy()[np.asarray(idx)]
        return score_fn, hist, getattr(m, "best_epoch", -1)

    if method in ("FairGB", "FairVGNN"):
        from core.trajectory import inference_state, restore_inference_state
        if method == "FairGB":
            from utils.data import get_dataset
            from models.algorithms.FairGB_alg import FairGB
            from models.algorithms.FairGB.eval import evaluate_ged3 as gb_eval
            d, sens_idx, _, _ = get_dataset(
                DS[0], split_seed=SPLIT[0],
                feature_normalize=cfg.get("fairgb_get_dataset_normalize", True))
            d.sens_idx = sens_idx
            vidx = np.where(d.val_mask.cpu().numpy())[0]
            ref = SplitRef(node_id=vidx,
                           y=d.y.cpu().numpy()[vidx].astype(int),
                           a=d.sens.cpu().numpy()[vidx].astype(int))
            hist = ValidationHistory(ref, epochs)
            m = FairGB()
            m.fit(d, device=device, runs=1, seed=seed, epochs=epochs,
                  trajectory=hist,
                  **{k: v for k, v in cfg.items()
                     if k in ("hidden", "c_lr", "c_wd", "e_lr", "e_wd",
                              "alpha", "eta", "dropout", "encoder", "warmup")},
                  use_cal=off.get("use_cal", True),
                  use_cnm=off.get("use_cnm", True))
            tmask = d.test_mask.cpu().numpy()

            def score_fn(state, idx, _m=m, _d=d, _t=tmask):
                if state is not None:
                    restore_inference_state(state, **_m.inference_modules())
                *_, out = gb_eval(_m._cls_for_inference, _m._enc_for_inference,
                                  _d, return_output=True)
                v = out.squeeze().detach().cpu().numpy()
                return v[np.asarray(idx)]
            return score_fn, hist, getattr(m, "best_epoch", -1)

        from models.algorithms.FairVGNN import FairVGNN, evaluate_ged3 as vg_eval
        iv = np.asarray(iva.cpu() if torch.is_tensor(iva) else iva)
        ref = SplitRef(node_id=iv,
                       y=np.asarray(y.detach().cpu() if torch.is_tensor(y) else y)[iv].astype(int),
                       a=np.asarray(s.detach().cpu() if torch.is_tensor(s) else s)[iv].astype(int))
        import models.algorithms.FairVGNN as _VG
        from core.trajectory import (bundle_history, capture_rng_bundle,
                                     restore_rng_bundle, seed_all)
        # Replay state (X10): FairVGNN's validation call samples a Gumbel mask,
        # on CUDA from the CUDA generator. The method snapshots only the CPU
        # generator, so the harness captures the full bundle immediately before
        # each validation call and the history swaps it into every checkpoint.
        # The method's source is not edited.
        holder = {"bundle": None}
        hist = bundle_history(ref, epochs, holder)
        _orig_eval = _VG.evaluate_ged3

        def _capturing_eval(*a_, **k_):
            holder["bundle"] = capture_rng_bundle()
            return _orig_eval(*a_, **k_)
        if cfg.get("fairvgnn_credit_adapter"):
            # native FairVGNN/credit: the official fairvgnn_credit.py loop, in a
            # separate adapter; the shared wrapper is never edited (X19)
            from adapters.fairvgnn_credit_native import FairVGNNCreditNative
            m = FairVGNNCreditNative(clip_c=cfg["clip_c"])
        else:
            m = FairVGNN()
        # seed must be passed explicitly: FairVGNN.run() calls
        # seed_everything(count + args.seed) and then resets every module, so
        # without it the fit default seed=1 overrides the harness seed and all
        # runs of a split start from the same parameters (X9)
        _orig_norm = _VG.feature_norm
        _VG.evaluate_ged3 = _capturing_eval
        if cfg.get("vg_wrapper_normalize", True) is False:
            # official dataset.py leaves german unnormalized; the wrapper's
            # get_dataset normalizes unconditionally, so for native german the
            # harness makes its feature_norm the identity for this fit (X13)
            _VG.feature_norm = lambda x: x
        try:
            m.fit(adj, f, y, itr, iva, ite, s, si, device=device, runs=1,
                  epochs=epochs, trajectory=hist, seed=seed,
                  **{k: v for k, v in cfg.items()
                     if k in ("hidden", "c_lr", "c_wd", "e_lr", "e_wd", "top_k",
                              "alpha", "clip_e", "d_epochs", "g_epochs",
                              "c_epochs", "ratio", "prop", "dropout", "encoder",
                              "K")},
                  f_mask=off.get("f_mask", "yes"),
                  weight_clip=off.get("weight_clip", "yes"))
        finally:
            _VG.evaluate_ged3 = _orig_eval
            _VG.feature_norm = _orig_norm

        def score_fn(state, idx, xi=None, replay=False, _m=m):
            """Scores after restoring `state`.

            replay=True   restore the checkpoint's own RNG bundle and return the
                          validation scores in the order they were recorded --
                          the replay-fidelity path.
            xi=<int>      restore the parameters, then reseed every generator to
                          the cell's final-evaluation seed -- the test path, so
                          all arms and slots of a cell share one mask draw.
            neither       legacy: whatever generator position results.
            """
            if state is not None:
                restore_inference_state(state, **_m.inference_modules())
            if not replay and xi is not None:
                seed_all(int(xi))
            *_, out = vg_eval(_m._inf_data.x, _m._inf_modules["classifier"],
                              _m._inf_modules["discriminator"],
                              _m._inf_modules["generator"],
                              _m._inf_modules["encoder"], _m._inf_data,
                              _m._inf_args, return_output=True)
            if replay:
                return out[_m._inf_data.val_mask].squeeze().detach().cpu().numpy()
            v = out.squeeze().detach().cpu().numpy()
            return v[np.asarray(idx)]
        return score_fn, hist, getattr(m, "best_epoch", -1)

    raise SystemExit(f"pilot does not know {method!r}")


DS = ["german"]
SPLIT = [20]

# ---------------------------------------------------------------------------
# Cell-level incremental persistence (I/O contract only; X15).
#
# A run used to write every row once, at the end. When /home filled up, a
# finished 30-cell run was lost to that single write. Now each (method,
# dataset, split, run) cell is appended the moment both of its selector rows
# exist, flushed and fsynced. An existing file is validated before anything
# runs, and completed cells are skipped, so a rerun resumes.
# ---------------------------------------------------------------------------
STORE_KEY = ("protocol", "method", "dataset", "split_id", "run_id", "selector")
STORE_SELECTORS = ("common_bce", "common_auc")


class CellStore:
    def __init__(self, path):
        self.path, self.header = path, None
        self.keys, self.done = set(), set()
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            return
        raw = open(path, "rb").read()
        if not raw.endswith(b"\n"):
            raise SystemExit(f"[store] {path}: last row is incomplete (no trailing "
                             "newline); refusing to resume or overwrite")
        cells = {}
        with open(path, newline="") as fh:
            rd = csv.reader(fh)
            header = next(rd)
            missing = [c for c in STORE_KEY if c not in header]
            if missing:
                raise SystemExit(f"[store] {path}: header lacks key columns {missing}")
            for i, row in enumerate(rd, start=2):
                if len(row) != len(header):
                    raise SystemExit(f"[store] {path}: line {i} is an incomplete row "
                                     f"({len(row)} of {len(header)} fields); refusing")
                rec = dict(zip(header, row))
                k = tuple(rec[c] for c in STORE_KEY)
                if k in self.keys:
                    raise SystemExit(f"[store] {path}: duplicate key {k} at line {i}; "
                                     "refusing to overwrite")
                self.keys.add(k)
                cells.setdefault(k[:5], set()).add(rec["selector"])
        for c, sels in cells.items():
            if sels != set(STORE_SELECTORS):
                raise SystemExit(f"[store] {path}: incomplete cell {c} (selectors "
                                 f"{sorted(sels)}); refusing")
        self.header, self.done = header, set(cells)

    @staticmethod
    def _cell(protocol, method, dataset, split, run):
        return (str(protocol), str(method), str(dataset), str(split), str(run))

    def is_done(self, protocol, method, dataset, split, run):
        return self._cell(protocol, method, dataset, split, run) in self.done

    def append_cell(self, rows):
        new = [tuple(str(r[c]) for c in STORE_KEY) for r in rows]
        if len(set(new)) != len(new) or any(k in self.keys for k in new):
            raise SystemExit(f"[store] duplicate key among {new}; refusing to overwrite")
        if {k[5] for k in new} != set(STORE_SELECTORS) or len({k[:5] for k in new}) != 1:
            raise SystemExit(f"[store] refusing to persist an incomplete cell {new}")
        cols = list(rows[0].keys())
        write_header = self.header is None
        if write_header:
            self.header = cols
        elif set(cols) != set(self.header):
            raise SystemExit("[store] row columns differ from the existing file header")
        with open(self.path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.header)
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(r)
            fh.flush()
            os.fsync(fh.fileno())
        self.keys.update(new)
        self.done.add(new[0][:5])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="german")
    ap.add_argument("--splits", nargs="*", type=int, default=[20, 21, 22])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=27)
    ap.add_argument("--epochs", type=int, default=200,
                    help="baseline B horizon; also the method horizon in armA")
    ap.add_argument("--protocol", choices=("armA", "native"), default="armA",
                    help="armA: published() at --epochs; native: native_config() "
                         "at each cell's native horizon, B still at --epochs")
    ap.add_argument("--method_epochs", type=int, default=None,
                    help="override the method horizon (gate smoke tests only; "
                         "recorded per row)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--methods", nargs="*", default=sorted(METHODS))
    ap.add_argument("--out", default=os.path.join(ROOT, "harness", "results",
                                                  "pilot_tau.csv"))
    a = ap.parse_args()
    print(f"\npilot: {a.dataset} / splits {a.splits} / {a.runs} runs / "
          f"{a.epochs} epochs\n")
    print("This run answers one question: do tau_pkg and tau_int differ?\n"
          "No ranking or conclusion is read from it.\n")
    DS[0] = a.dataset
    dev = a.device
    rows = []
    store = CellStore(a.out)
    if store.done:
        print(f"[store] resuming {a.out}: {len(store.done)} completed cell(s) "
              "will be skipped")

    for split in a.splits:
      SPLIT[0] = split
      for run in range(a.runs):
        seed = a.seed0 + run
        if all(store.is_done(a.protocol, m_, a.dataset, split, run)
               for m_ in a.methods):
            print(f"  s{split} r{run}: all methods already persisted, skipped")
            continue
        # One load per (normalize) policy. The methods do not agree on
        # preprocessing -- FairGNN and FairVGNN normalize german, nobody else
        # does -- and that disagreement is part of each package, so it is
        # applied, not smoothed away. B keeps the GNN baseline's own policy.
        _cache = {}

        def _data(flag):
            if flag not in _cache:
                d_ = load(a.dataset, split, dev, feature_normalize=flag)
                if _cache:
                    # normalization must touch features and nothing else, or
                    # the outcome operator below would be reading one split's
                    # labels against another's predictions
                    ref_ = next(iter(_cache.values()))
                    for j, nm in ((2, "labels"), (3, "idx_train"),
                                  (4, "idx_val"), (5, "idx_test"), (6, "sens")):
                        u = np.asarray(ref_[j].cpu() if torch.is_tensor(ref_[j])
                                       else ref_[j])
                        v = np.asarray(d_[j].cpu() if torch.is_tensor(d_[j])
                                       else d_[j])
                        if not np.array_equal(u, v):
                            raise SystemExit(
                                f"feature_normalize changed {nm} on "
                                f"{a.dataset} split {split}; B, M0 and M1 "
                                "would not share a split")
                _cache[flag] = d_
            return _cache[flag]

        b_pub = published("GNN", a.dataset)
        b_cfg = dict(b_pub["config"])
        data = _data(bool(b_cfg.pop("feature_normalize", False)))
        adj, f, y, itr, iva, ite, s, si = data
        ite_np = np.asarray(ite.cpu() if torch.is_tensor(ite) else ite)

        # ---- common baseline B --------------------------------------------
        from models.algorithms.GNN import GNN
        torch.manual_seed(seed); np.random.seed(seed)
        b = GNN(adj, f, y, itr, iva, ite, s, si, device=dev, **b_cfg)
        hb = ValidationHistory(_split_ref(y, iva, s), a.epochs + 1,
                               state_fn=lambda: {k: v.detach().clone()
                                                 for k, v in b.state_dict().items()})
        b.fit(epochs=a.epochs, trajectory=hb)

        def b_score(state, idx):
            if state is not None:
                b.load_state_dict(state)
            b.eval()
            with torch.no_grad():
                emb = b.forward(b.features.to(dev), b.edge_index.to(dev))
                out = b.forwarding_predict(emb)
            return out.squeeze().detach().cpu().numpy()[np.asarray(idx)]

        base_pub = outcome(y, s, ite_np, b_score(b.best_state, ite_np))
        # B under each common selector, read in this process from this very
        # model. A separate reproduction pass is not equivalent: GPU
        # nondeterminism compounds over 200 epochs and diverged the baseline in
        # 16 of 30 cells (worst case DP 0.000 vs 0.336), so tau_base^audit must
        # come from the same B that tau_pkg^pub was measured against.
        base_c = {}
        for _sel in ("common_bce", "common_auc"):
            if _sel not in hb.slots:
                print(f"  s{split} r{run}: baseline slot {_sel} missing")
                continue
            _ep, _st = hb.slots[_sel]
            base_c[_sel] = (_ep, outcome(y, s, ite_np, b_score(_st, ite_np)))

        for meth in a.methods:
            if store.is_done(a.protocol, meth, a.dataset, split, run):
                print(f"  s{split} r{run} {meth}: already persisted, skipped")
                continue
            spec = METHODS[meth]
            mp = (native_config(meth, a.dataset) if a.protocol == "native"
                  else published(meth, a.dataset))
            m_epochs = (a.method_epochs if a.method_epochs is not None
                        else (mp["horizon"] if a.protocol == "native" else a.epochs))
            mcfg = mp["config"]
            mdata = _data(bool(mcfg.get("feature_normalize", False)))
            try:
                # paired RNG: both arms of a cell start the stochastic parts of
                # evaluation from the same state, so a difference between them
                # is the intervention and not the draw
                torch.manual_seed(seed * 1000 + split)
                f1_, h1, e1 = train(meth, mdata, seed, m_epochs, dev, off=None,
                                    cfg=mcfg)
                torch.manual_seed(seed * 1000 + split)
                f0_, h0, e0 = train(meth, mdata, seed, m_epochs, dev,
                                    off=spec["off"], cfg=mcfg)
            except Exception as exc:                              # noqa: BLE001
                print(f"  s{split} r{run} {meth}: FAILED "
                      f"{type(exc).__name__}: {exc}")
                continue

            # FairVGNN's inference is stochastic; every test evaluation of the
            # cell -- native view, M1/M0, BCE/AUC -- uses one pre-fixed seed
            _kw = ({"xi": eval_rng_seed(a.dataset, split, run)}
                   if meth == "FairVGNN" else {})
            m1_pub = outcome(y, s, ite_np, f1_(None, ite_np, **_kw))
            cell_rows = []
            for sel in ("common_bce", "common_auc"):
                if sel not in base_c or sel not in h1.slots or sel not in h0.slots:
                    print(f"  s{split} r{run} {meth}: slot {sel} missing")
                    continue
                e1s, st1 = h1.slots[sel]
                e0s, st0 = h0.slots[sel]
                sc1 = np.asarray(f1_(st1, ite_np, **_kw))
                sc0 = np.asarray(f0_(st0, ite_np, **_kw))
                m1 = outcome(y, s, ite_np, sc1)
                m0 = outcome(y, s, ite_np, sc0)
                # diagnostics only: computed from scores already produced, so
                # no extra forward pass, no RNG, no change to any contract
                _a = np.asarray(s.detach().cpu() if torch.is_tensor(s)
                                else s)[ite_np].astype(int)
                p1 = (sc1 > 0).astype(int)
                p0 = (sc0 > 0).astype(int)
                flip = p1 != p0
                n_a1 = int((_a == 1).sum()); n_a0 = int((_a == 0).sum())
                diag = dict(
                    n_flip=int(flip.sum()),
                    n_flip_a1=int((flip & (_a == 1)).sum()),
                    n_flip_a0=int((flip & (_a == 0)).sum()),
                    n_test_a1=n_a1, n_test_a0=n_a0,
                    # the smallest nonzero |dDP| a single flipped prediction can
                    # produce on this split; a diagnostic, never a metric
                    dp_min_step=min(1.0 / max(n_a1, 1), 1.0 / max(n_a0, 1)))
                cell_rows.append(dict(
                    method=meth, dataset=a.dataset, backbone="GCN",
                    split_id=split, run_id=run, seed=seed, selector=sel,
                    provenance=mp["provenance"], **diag,
                    protocol=a.protocol, method_epochs=int(m_epochs),
                    b_epochs=int(a.epochs),
                    eval_rng_seed=int(_kw.get("xi", -1)),
                    rng_contract=("bundle-replay/xi-eval" if _kw
                                  else "deterministic-inference"),
                    feature_normalize=int(bool(mcfg.get("feature_normalize",
                                                        False))),
                    m1_epoch=e1s, m0_epoch=e0s, code_epoch=e1,
                    m1_auc=m1["auc"], m1_dp=m1["dp"], m1_eo=m1["eo"],
                    m0_auc=m0["auc"], m0_dp=m0["dp"], m0_eo=m0["eo"],
                    b_auc=base_pub["auc"], b_dp=base_pub["dp"], b_eo=base_pub["eo"],
                    bc_epoch=base_c[sel][0],
                    bc_auc=base_c[sel][1]["auc"], bc_dp=base_c[sel][1]["dp"],
                    bc_eo=base_c[sel][1]["eo"],
                    m1pub_auc=m1_pub["auc"], m1pub_dp=m1_pub["dp"],
                    m1pub_eo=m1_pub["eo"],
                    # paired differences, formed in the cell before any
                    # aggregation
                    int_dauc=m1["auc"] - m0["auc"],
                    int_ndp=-(m1["dp"] - m0["dp"]),
                    int_neo=-(m1["eo"] - m0["eo"]),
                    pkg_dauc=m1_pub["auc"] - base_pub["auc"],
                    pkg_ndp=-(m1_pub["dp"] - base_pub["dp"]),
                    pkg_neo=-(m1_pub["eo"] - base_pub["eo"]),
                    eo_defined=bool(m1["eo_defined"] and m0["eo_defined"])))
            if len(cell_rows) != len(STORE_SELECTORS):
                print(f"  s{split} r{run} {meth}: incomplete cell "
                      f"({len(cell_rows)} rows), not persisted")
                continue
            store.append_cell(cell_rows)
            rows.extend(cell_rows)
            r = rows[-1]
            print(f"  s{split} r{run} {meth}: int=[{r['int_dauc']:+.4f}, "
                  f"{r['int_ndp']:+.4f}]  pkg=[{r['pkg_dauc']:+.4f}, "
                  f"{r['pkg_ndp']:+.4f}]  (ep M1={r['m1_epoch']} M0={r['m0_epoch']})")

    import pandas as pd
    # every cell is already on disk; the summary reads what was persisted,
    # including cells resumed from an earlier run
    if not (os.path.exists(a.out) and os.path.getsize(a.out) > 0):
        print("\nno rows"); return 1
    df = pd.read_csv(a.out)
    print(f"\n[persisted incrementally] {a.out}: {len(df)} rows\n")
    print("=" * 88)
    print("Paired effects, formed per (split, run) cell then averaged.")
    print("[dAUC, -dDP] primary; larger is better in both coordinates.")
    print("=" * 88)
    prim = df[df.selector == "common_bce"]
    for meth, g in prim.groupby("method"):
        print(f"\n  {meth}")
        print(f"    {'split':>6s} {'n':>3s} {'tau_int':>20s} {'tau_pkg':>20s}")
        for sp, gg in g.groupby("split_id"):
            print(f"    {sp:6d} {len(gg):3d} "
                  f"[{gg.int_dauc.mean():+.4f}, {gg.int_ndp.mean():+.4f}]".rjust(22)
                  + f"[{gg.pkg_dauc.mean():+.4f}, {gg.pkg_ndp.mean():+.4f}]".rjust(22))
        print(f"    {'ALL':>6s} {len(g):3d} "
              f"[{g.int_dauc.mean():+.4f}, {g.int_ndp.mean():+.4f}]".rjust(22)
              + f"[{g.pkg_dauc.mean():+.4f}, {g.pkg_ndp.mean():+.4f}]".rjust(22))
    print("\nRobustness selector (sigma_c^AUC), tau_int only:")
    rob = df[df.selector == "common_auc"]
    for meth, g in rob.groupby("method"):
        print(f"  {meth:10s} [{g.int_dauc.mean():+.4f}, {g.int_ndp.mean():+.4f}]")
    nud = int((~df.eo_defined).sum())
    print(f"\nEO undefined in {nud} of {len(df)} rows "
          f"({nud / max(len(df), 1):.1%}) -- reported, not dropped from means.")
    print("Per-split effects are shown before the overall mean on purpose.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
