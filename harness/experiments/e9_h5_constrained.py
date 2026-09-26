"""
H5 -- does refusing to spend accuracy raise the Pareto dominance rate?

Registered in harness/PREREGISTRATION.md on 2026-09-11. The observation behind it:
re-read on the (AUC, dP) front, `bind_influence` dominates uniform in 23.0% of
cells and is dominated by it in 13.0%. If that 13% is accuracy lost rather than
disparity not reduced, an allocation that declines to spend accuracy should move
part of it into the incomparable or dominating column.

    uniform         phi uniform
    unconstrained   phi from bind_influence
    constrained     phi from bind_influence, fairness term held back whenever
                    validation AUC is more than tau below the uniform arm's

`auc_ref` is the uniform arm's validation AUC in the *same* (setting, split,
init) cell -- not a number carried across cells, and not a number read off the
nine settings.

Two parameters and where they come from
---------------------------------------
`tau` = 0.005 is inherited, not tuned: it is `adaptive_auc_tol`'s value in
FairGate's own published configuration (outputs/ours/exp_fairgate_fiw_v1.csv,
utils/model_fairgate.py:1154). The registration fixed it before any of this ran
and forbids presenting a sweep over it as anything but a sensitivity analysis.

The pull-back is on/off, not a smooth scaling. A smooth version needs a slope,
and the slope would have to come from somewhere -- in practice from the nine
settings, which is the failure mode every other guard in the registration
exists to prevent. On/off adds nothing to choose.

The constraint reads the *previous* epoch's validation AUC, because this
epoch's is not computed until after the step. A run cannot condition on a
measurement it has not made yet.

Falsification, as registered: **H5 fails if the constrained arm's dominance
rate over uniform is not higher than the unconstrained arm's.** The registered
prior also expects the ΔDP-only win rate to *fall*; if that is the trade, it is
the point and not a disappointment.

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python harness/experiments/e9_h5_constrained.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import datasets as D                 # noqa: E402
from core.trainer import Config, train         # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
SIGNAL = "bind_influence"
SPLITS = [20, 21, 22, 23, 24, 25]
INITS = [27, 28, 29, 30, 31]
TAU = 0.005


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", type=int, default=SPLITS)
    ap.add_argument("--inits", nargs="*", type=int, default=INITS)
    ap.add_argument("--signal", default=SIGNAL)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lambda-fair", type=float, default=0.2)
    ap.add_argument("--q-gate", type=float, default=0.7)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=os.path.join(RESULTS, "e9_h5.csv"))
    args = ap.parse_args()

    datasets = args.datasets or D.ALL_SETTINGS
    os.makedirs(RESULTS, exist_ok=True)
    n_cells = len(datasets) * len(args.splits) * len(args.inits)
    print(f"datasets={datasets}\nsplits={args.splits}  inits={args.inits}")
    print(f"signal={args.signal}  tau={args.tau}")
    print(f"cells = {n_cells}  ({n_cells * 3} runs)\n")

    rows, t0, done = [], time.time(), 0
    for ds in datasets:
        for sp in args.splits:
            g = D.load(ds, split_seed=sp)
            for init in args.inits:
                done += 1
                t = time.time()
                base = Config(seed=init, epochs=args.epochs, warmup=args.warmup,
                              lambda_fair=args.lambda_fair, q_gate=args.q_gate,
                              device=args.device)

                def rec(arm, r, cfg):
                    rows.append({"setting": ds, "split_seed": sp, "init_seed": init,
                                 "seed": f"{sp}_{init}", "arm": arm,
                                 "signal": cfg.signal, "auc_ref": cfg.auc_ref,
                                 "n_held": r.n_held, "epochs_run": r.epochs_run,
                                 **{f"test_{k}": v for k, v in r.test.items()},
                                 **{f"val_{k}": v for k, v in r.val.items()},
                                 **{f"phi_{k}": v for k, v in r.phi.items()}})

                try:
                    cu = base.__class__(**{**base.__dict__, "signal": "uniform"})
                    ru = train(g, cu)
                    rec("uniform", ru, cu)

                    cn = base.__class__(**{**base.__dict__, "signal": args.signal})
                    rn = train(g, cn)
                    rec("unconstrained", rn, cn)

                    # the reference is this cell's uniform run, nothing wider
                    cc = base.__class__(**{**base.__dict__, "signal": args.signal,
                                           "auc_ref": float(ru.val["auc"]),
                                           "auc_tol": args.tau})
                    rc = train(g, cc)
                    rec("constrained", rc, cc)
                except Exception as e:                                # noqa: BLE001
                    print(f"  {ds} sp={sp} init={init} FAILED "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue

                print(f"  [{done}/{n_cells}] {ds:<10} sp={sp} init={init} "
                      f"ref={ru.val['auc']:.4f} held={rc.n_held}/{rc.epochs_run} "
                      f"({time.time() - t:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[saved] {args.out}  ({len(df)} runs, {time.time() - t0:.0f}s)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
