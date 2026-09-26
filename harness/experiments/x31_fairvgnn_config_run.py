"""X31 FairVGNN configuration-variation cells: the other full rows of the
official run scripts, through exactly the path the primary FairVGNN cells took.

The primary FairVGNN cell per dataset uses the official GCN row at the default
propagation (`published_config._fairvgnn_lines`). The same official scripts
publish further full rows -- other encoders and the spmm propagation -- each with
the authors' own both-off ablation row. X31 runs every such row as a
configuration-variation (robustness) cell:

    german  GCN-spmm, GIN, SAGE      bail  GCN-spmm, SAGE      credit  GIN, SAGE

Everything except the configuration is the primary cells' path, imported from
`pilot_tau` unchanged: the loader, B at `published("GNN", dataset)` trained in the
same process, `train("FairVGNN", ...)` with its RNG-bundle replay and the fixed
evaluation seed `eval_rng_seed`, the off-state `f_mask='no', weight_clip='no'`,
H = 200, sigma_c^BCE / sigma_c^AUC, the common evaluator and the CellStore
contract. The configuration is read from the row with the same key list and
parser defaults `published()` uses for the primary row; feature normalisation
follows the primary cell of the same dataset. The credit rows run through the
same wrapper as the primary credit cell; the credit script's classifier
clipping (`clip_c`) belongs to the native procedure and is recorded, not applied.

    CUDA_VISIBLE_DEVICES=2 python harness/experiments/x31_fairvgnn_config_run.py \
        --dataset german --config GIN --out <csv> [--smoke]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))

import pilot_tau as P                                              # noqa: E402
from core.published_config import (PROV, _flags, _read, _fairvgnn_defaults,  # noqa: E402
                                   published)
from core.trajectory import eval_rng_seed                          # noqa: E402

KEYS = ("clip_e", "d_epochs", "g_epochs", "c_epochs", "c_lr", "e_lr", "c_wd", "e_wd",
        "ratio", "top_k", "alpha", "hidden", "dropout", "prop", "K")
CONFIGS = {"german": ("GCNspmm", "GIN", "SAGE"), "bail": ("GCNspmm", "SAGE"),
           "credit": ("GIN", "SAGE")}


def rows(dataset):
    """Full rows (neither f_mask nor weight_clip switched off), keyed by label."""
    out = {}
    for ln in _read(os.path.join(PROV, f"fairvgnn_run_{dataset}.sh")):
        if not ln.strip().startswith("python"):
            continue
        f = _flags(ln)
        if "f_mask" in f or "weight_clip" in f:
            continue
        lab = f["encoder"] + ("spmm" if f.get("prop") == "spmm" else "")
        out.setdefault(lab, f)
    return out


def config(dataset, label):
    f = rows(dataset)[label]
    defs = _fairvgnn_defaults()
    cfg = {k: f.get(k, defs.get(k)) for k in KEYS if f.get(k, defs.get(k)) is not None}
    cfg["encoder"] = f["encoder"]
    cfg["feature_normalize"] = published("FairVGNN", dataset)["config"].get("feature_normalize", False)
    meta = dict(native_epochs=f.get("epochs", defs.get("epochs")), clip_c=f.get("clip_c"))
    return cfg, meta


def native(cfg, meta, dataset):
    """The native changes `native_config("FairVGNN", ds)` applies to the primary
    row, applied to this row: credit trains with fairvgnn_credit.py's loop (the
    row's own --clip_c, else that script's parser default); feature normalisation
    follows the official dataset.py rule, with the harness pre-normalisation off."""
    from core.published_config import (_fairvgnn_credit_clip_c,
                                       _fairvgnn_unnormalized_dataset)
    cfg = dict(cfg)
    if dataset == "credit":
        cfg["fairvgnn_credit_adapter"] = True
        cfg["clip_c"] = meta["clip_c"] if meta["clip_c"] is not None else _fairvgnn_credit_clip_c()
    cfg["feature_normalize"] = False
    cfg["vg_wrapper_normalize"] = dataset != _fairvgnn_unnormalized_dataset()
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=tuple(CONFIGS))
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", type=int, default=[20, 21, 22, 23, 24, 25])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=27)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--native", action="store_true",
                    help="the row at its own horizon with the method's own selection and "
                         "the native changes of the primary FairVGNN native cells; B stays "
                         "at --epochs (the fixed controlled reference)")
    a = ap.parse_args()
    if a.config not in CONFIGS[a.dataset]:
        raise SystemExit(f"{a.config} is not a published full row for {a.dataset}")
    if a.device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise SystemExit("GPU 2 only: set CUDA_VISIBLE_DEVICES=2")
    cfg, meta = config(a.dataset, a.config)
    m_epochs, protocol = a.epochs, "x31"
    if a.native:
        cfg, m_epochs, protocol = native(cfg, meta, a.dataset), int(meta["native_epochs"]), "x31native"
    tag = f"FairVGNN-{a.config}"
    off = P.METHODS["FairVGNN"]["off"]
    P.DS[0] = a.dataset
    store = None if a.smoke else P.CellStore(a.out)
    for split in a.splits:
        P.SPLIT[0] = split
        for run in range(a.runs):
            seed = a.seed0 + run
            if store and store.is_done(protocol, tag, a.dataset, split, run):
                continue
            # data: B at the baseline's policy, the arms at the primary FairVGNN
            # cell's policy; normalisation may touch features only
            b_cfg = dict(published("GNN", a.dataset)["config"])
            bdata = P.load(a.dataset, split, a.device,
                           feature_normalize=bool(b_cfg.pop("feature_normalize", False)))
            mflag = bool(cfg.get("feature_normalize", False))
            mdata = (P.load(a.dataset, split, a.device, feature_normalize=mflag)
                     if mflag != bool(published("GNN", a.dataset)["config"]
                                      .get("feature_normalize", False)) else bdata)
            for j in (2, 3, 4, 5, 6):
                u, v = bdata[j], mdata[j]
                if not np.array_equal(np.asarray(u.cpu() if torch.is_tensor(u) else u),
                                      np.asarray(v.cpu() if torch.is_tensor(v) else v)):
                    raise SystemExit("[x31] normalisation changed labels or split (hard stop)")
            adj, f, y, itr, iva, ite, s, si = bdata
            ite_np = np.asarray(ite.cpu() if torch.is_tensor(ite) else ite)

            # ---- common baseline B, exactly as pilot_tau.main ----------------
            import core.paths  # noqa: F401  (puts the repository root on sys.path)
            from models.algorithms.GNN import GNN
            torch.manual_seed(seed); np.random.seed(seed)
            b = GNN(adj, f, y, itr, iva, ite, s, si, device=a.device, **b_cfg)
            hb = P.ValidationHistory(P._split_ref(y, iva, s), a.epochs + 1,
                                     state_fn=lambda: {k: v.detach().clone()
                                                       for k, v in b.state_dict().items()})
            b.fit(epochs=a.epochs, trajectory=hb)

            def b_score(state, idx):
                if state is not None:
                    b.load_state_dict(state)
                b.eval()
                with torch.no_grad():
                    out = b.forwarding_predict(b.forward(b.features.to(a.device),
                                                         b.edge_index.to(a.device)))
                return out.squeeze().detach().cpu().numpy()[np.asarray(idx)]
            base_pub = P.outcome(y, s, ite_np, b_score(b.best_state, ite_np))
            base_c = {sel: (hb.slots[sel][0], P.outcome(y, s, ite_np, b_score(hb.slots[sel][1], ite_np)))
                      for sel in P.STORE_SELECTORS if sel in hb.slots}

            # ---- paired arms, exactly as pilot_tau.main ----------------------
            torch.manual_seed(seed * 1000 + split)
            f1_, h1, e1 = P.train("FairVGNN", mdata, seed, m_epochs, a.device, off=None, cfg=cfg)
            torch.manual_seed(seed * 1000 + split)
            f0_, h0, e0 = P.train("FairVGNN", mdata, seed, m_epochs, a.device, off=off, cfg=cfg)
            xi = eval_rng_seed(a.dataset, split, run)
            m1_pub = P.outcome(y, s, ite_np, f1_(None, ite_np, xi=xi))
            ok = all(sel in h.slots for h in (h1, h0) for sel in P.STORE_SELECTORS) \
                and set(base_c) == set(P.STORE_SELECTORS)
            sc = {sel: (np.asarray(f1_(h1.slots[sel][1], ite_np, xi=xi)),
                        np.asarray(f0_(h0.slots[sel][1], ite_np, xi=xi)))
                  for sel in P.STORE_SELECTORS} if ok else {}
            fin = ok and all(np.isfinite(v).all() for pr in sc.values() for v in pr)
            print(f"  {tag}/{a.dataset} s{split} r{run}: B and both arms have both selector "
                  f"slots {ok}; finite test scores {fin}; horizon {m_epochs}", flush=True)
            if not (ok and fin):
                raise SystemExit("[x31] gate failed (hard stop; nothing persisted)")
            if a.smoke:
                continue
            _a = np.asarray(s.detach().cpu() if torch.is_tensor(s) else s)[ite_np].astype(int)
            n_a1, n_a0 = int((_a == 1).sum()), int((_a == 0).sum())
            cell = []
            for sel in P.STORE_SELECTORS:
                sc1, sc0 = sc[sel]
                m1 = P.outcome(y, s, ite_np, sc1)
                m0 = P.outcome(y, s, ite_np, sc0)
                flip = (sc1 > 0).astype(int) != (sc0 > 0).astype(int)
                cell.append(dict(
                    method=tag, dataset=a.dataset, backbone=cfg["encoder"], split_id=split,
                    run_id=run, seed=seed, selector=sel, provenance="official-repo",
                    n_flip=int(flip.sum()), n_flip_a1=int((flip & (_a == 1)).sum()),
                    n_flip_a0=int((flip & (_a == 0)).sum()), n_test_a1=n_a1, n_test_a0=n_a0,
                    dp_min_step=min(1.0 / max(n_a1, 1), 1.0 / max(n_a0, 1)),
                    protocol=protocol, method_epochs=m_epochs, b_epochs=a.epochs,
                    eval_rng_seed=int(xi), rng_contract="bundle-replay/xi-eval",
                    feature_normalize=int(mflag),
                    m1_epoch=h1.slots[sel][0], m0_epoch=h0.slots[sel][0], code_epoch=e1,
                    m1_auc=m1["auc"], m1_dp=m1["dp"], m1_eo=m1["eo"],
                    m0_auc=m0["auc"], m0_dp=m0["dp"], m0_eo=m0["eo"],
                    b_auc=base_pub["auc"], b_dp=base_pub["dp"], b_eo=base_pub["eo"],
                    bc_epoch=base_c[sel][0], bc_auc=base_c[sel][1]["auc"],
                    bc_dp=base_c[sel][1]["dp"], bc_eo=base_c[sel][1]["eo"],
                    m1pub_auc=m1_pub["auc"], m1pub_dp=m1_pub["dp"], m1pub_eo=m1_pub["eo"],
                    int_dauc=m1["auc"] - m0["auc"], int_ndp=-(m1["dp"] - m0["dp"]),
                    int_neo=-(m1["eo"] - m0["eo"]),
                    pkg_dauc=m1_pub["auc"] - base_pub["auc"],
                    pkg_ndp=-(m1_pub["dp"] - base_pub["dp"]),
                    pkg_neo=-(m1_pub["eo"] - base_pub["eo"]),
                    eo_defined=bool(m1["eo_defined"] and m0["eo_defined"]),
                    config=repr(dict(cfg, **meta))))
            store.append_cell(cell)
            print(f"  {tag}/{a.dataset} s{split} r{run}: persisted", flush=True)
    print("SMOKE PASS" if a.smoke else f"[x31] done -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
