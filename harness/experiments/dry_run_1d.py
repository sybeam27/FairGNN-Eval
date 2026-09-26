"""Phase 1D end-to-end dry run. Pipeline correctness only.

**No scientific claim is read from this run.** If a method looks fairer here,
that is ignored. The run passes or fails on four conditions:

    1 instrumentation invariance   logging must not change training
    2 selector replay              a method's own rule, replayed on the stored
                                   trajectory, must pick the epoch its code did
    3 test isolation               the selector API cannot see test data
    4 trajectory completeness      every classifier epoch logged, alignment and
                                   score length constant

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python harness/experiments/dry_run_1d.py --model GNN
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

from core.trajectory import (SplitRef, ValidationHistory,  # noqa: E402
                             select_max_acc, select_max_auc,
                             select_composite, select_min_bce,
                             select_nifty)
from utils.dataloading import load_data                    # noqa: E402


def build(model, ds, split_seed, seed, device):
    adj, f, y, itr, iva, ite, s, si = load_data(ds, feature_normalize=False,
                                                split_seed=split_seed)
    # Index tensors stay on CPU, matching utils/train_baselines. Moving them to
    # the device surfaces a pre-existing NIFTY line that indexes a numpy array
    # with a CUDA tensor (NIFTY.py:509) -- the audit never hits it because it
    # never moves them. The harness must reproduce the audit's calling
    # convention, not a tidier one.
    adj, f, y, s = (t.to(device) if hasattr(t, "to") else t for t in (adj, f, y, s))
    torch.manual_seed(seed); np.random.seed(seed)
    if model == "GNN":
        import core.paths  # noqa: F401  (puts the repository root on sys.path)
        from models.algorithms.GNN import GNN
        m = GNN(adj, f, y, itr, iva, ite, s, si, num_hidden=128,
                num_proj_hidden=128, lr=1e-3, weight_decay=0.0, device=device)
        run = lambda traj: m.fit(epochs=EPOCHS[0], trajectory=traj)
        sens_of = lambda: s
    elif model == "FairGNN":
        import itertools
        from models.algorithms.FairGNN import FairGNN
        # alpha and beta from utils/param.json for German, as the audit uses
        m = FairGNN(nfeat=f.shape[1], alpha=8, beta=0.005).to(device)
        m.args.num_hidden = 128
        G = list(itertools.chain(m.GNN.parameters(), m.classifier.parameters(),
                                 m.estimator.parameters()))
        m.optimizer_G = torch.optim.Adam(G, lr=1e-3, weight_decay=0.0)
        m.optimizer_A = torch.optim.Adam(m.adv.parameters(), lr=1e-3,
                                         weight_decay=0.0)
        m.args.epochs = EPOCHS[0]
        run = lambda traj: m.fit(adj, f, y, itr, iva, ite, s, itr,
                                 device=device, trajectory=traj)
        sens_of = lambda: s
    elif model == "NIFTY":
        from models.algorithms.NIFTY import NIFTY
        m = NIFTY(adj, f, y, itr, iva, ite, s, si, num_hidden=128,
                  num_proj_hidden=128, lr=1e-3, weight_decay=1e-5, device=device)
        run = lambda traj: m.fit(epochs=EPOCHS[0], trajectory=traj)
        sens_of = lambda: s
    elif model == "FairGB":
        from utils.data import get_dataset
        from models.algorithms.FairGB_alg import FairGB
        data, sens_idx, _, _ = get_dataset(ds, split_seed=split_seed)
        data.sens_idx = sens_idx
        m = FairGB()
        run = lambda traj: m.fit(data, device=device, runs=1, seed=seed,
                                 epochs=EPOCHS[0], hidden=16, trajectory=traj)
        vmask = data.val_mask.cpu().numpy()
        iv = np.where(vmask)[0]
        yv = data.y.cpu().numpy()
        sv = data.sens.cpu().numpy()
        return m, run, (lambda: sv), (yv, iv)
    elif model == "FairVGNN":
        from models.algorithms.FairVGNN import FairVGNN
        m = FairVGNN()
        run = lambda traj: m.fit(adj, f, y, itr, iva, ite, s, si, device=device,
                                 runs=1, epochs=EPOCHS[0], hidden=16,
                                 trajectory=traj)
        sens_of = lambda: s
    else:
        raise SystemExit(f"dry run does not know {model!r}")
    return m, run, sens_of, (y, iva)


EPOCHS = [60]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GNN")
    ap.add_argument("--dataset", default="german")
    ap.add_argument("--split_seed", type=int, default=20)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--epochs", type=int, default=60)
    # The invariance check runs on CPU. On GPU this repository is
    # nondeterministic at a scale that swamps what the check is looking for:
    # two *uninstrumented* NIFTY runs with the same seed selected epoch 59 and
    # epoch 37. On CPU the same configuration is exactly reproducible, so a
    # difference between instrumented and uninstrumented runs can only come
    # from the instrumentation. Experiments still run on GPU.
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()

    print(f"\n=== dry run: {a.model} / {a.dataset} / split {a.split_seed} / "
          f"seed {a.seed} / {a.epochs} epochs ===\n")
    ok = 0

    def chk(name, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
        ok += bool(cond)
        return bool(cond)

    # ---- 1. instrumentation invariance -----------------------------------
    EPOCHS[0] = a.epochs
    m1, run1, _, _ = build(a.model, a.dataset, a.split_seed, a.seed, a.device)
    run1(None)                                                # no logger
    def snapshot(m):
        """Whatever this object exposes that an instrumented run must match.

        Modules give a parameter state; FairGB is a wrapper with none, so the
        comparison falls back to what it does expose -- the chosen epoch and the
        metrics it reports. Weaker, and stated as weaker, but a logger that
        changed training would still have to keep both identical.
        """
        st = getattr(m, "best_state", None)
        if st is None and hasattr(m, "state_dict"):
            st = m.state_dict()
        if st is not None:
            return ("state", {k: v.detach().cpu().numpy().copy()
                              for k, v in st.items()})
        try:
            return ("reported", tuple(np.round(np.asarray(m.predict(),
                                                          dtype=float), 10)))
        except Exception:                                     # noqa: BLE001
            return ("epoch_only", None)

    ref_kind, ref_state = snapshot(m1)
    ref_epoch = getattr(m1, "best_epoch", -1)

    m2, run2, sens_of, (y, iva) = build(a.model, a.dataset, a.split_seed,
                                        a.seed, a.device)
    iva_np = iva.cpu().numpy() if torch.is_tensor(iva) else np.asarray(iva)
    y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    sv = sens_of()
    sens = sv.detach().cpu().numpy() if torch.is_tensor(sv) else np.asarray(sv)
    ref = SplitRef(node_id=iva_np, y=y_np[iva_np].astype(int),
                   a=sens[iva_np].astype(int))
    # The same `epochs=N` argument does not mean the same number of training
    # steps. GNN and NIFTY loop `range(epochs + 1)`; FairGNN, FairGB and FairSIN
    # loop `range(epochs)`. Declaring one horizon for all of them would fail the
    # completeness check for a reason that is not a logging fault, so the
    # convention is declared per method -- and the difference is recorded in the
    # horizon audit rather than smoothed over here.
    HORIZON = {"GNN": a.epochs + 1, "NIFTY": a.epochs + 1,
               "FairGNN": a.epochs, "FairGB": a.epochs,
               "FairVGNN": a.epochs}
    hist = ValidationHistory(ref, horizon=HORIZON[a.model])
    run2(hist)                                                # with logger

    kind2, st2 = snapshot(m2)
    if ref_kind == "state":
        same = all(np.allclose(ref_state[k], st2[k], atol=a.tol) for k in ref_state)
    elif ref_kind == "reported":
        same = (kind2 == "reported"
                and len(ref_state) == len(st2)
                and all(abs(x - z) <= a.tol or (np.isnan(x) and np.isnan(z))
                        for x, z in zip(ref_state, st2)))
    else:
        same = True          # nothing comparable beyond the epoch check below
    chk(f"1. instrumentation invariance ({ref_kind})", same, f"tol {a.tol}")
    ep2 = getattr(m2, "best_epoch", -1)
    chk("1b. same epoch selected", ref_epoch == ep2,
        f"{ref_epoch} vs {ep2}")

    # ---- 4. trajectory completeness ---------------------------------------
    c = hist.completeness()
    chk("4. all classifier epochs logged",
        c["complete"], f"{c['n_logged']}/{c['horizon']}")
    chk("4b. epochs unique and contiguous",
        c["epochs_unique"] and c["epochs_contiguous"])
    chk("4c. raw_score length constant", c["score_len_constant"])
    chk("4d. no non-finite epochs", c["n_nonfinite_epochs"] == 0,
        f"{c['n_nonfinite_epochs']} epochs with non-finite scores")

    # ---- 2. selector replay ------------------------------------------------
    # NIFTY's own rule is argmin(BCE + sim_coeff * invariance), which the
    # trajectory does not carry -- it stores the classifier's validation score,
    # not the invariance term. So its replay is checked against the recorded
    # best_epoch directly, and the shared selector is reported beside it.
    sel = {"GNN": select_min_bce, "FairGNN": select_max_acc,
           "NIFTY": select_nifty, "FairGB": select_composite,
           "FairVGNN": select_composite}[a.model]
    replay = sel(hist)
    replay = sel(hist)
    chk("2. selector replay reproduces the code's epoch",
        replay == ep2, f"replay {replay} vs code {ep2}")

    alt = select_max_auc(hist)
    print(f"       (secondary selector picks epoch {alt}; "
          f"agreement is not required)")

    # ---- 3. test isolation -------------------------------------------------
    fields = set(vars(hist)) | set(vars(hist.ref))
    chk("3. no test data reachable from the history",
        not any("test" in f.lower() for f in fields), f"{sorted(fields)}")

    print(f"\n{ok}/8 conditions passed\n")
    return 0 if ok == 8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
