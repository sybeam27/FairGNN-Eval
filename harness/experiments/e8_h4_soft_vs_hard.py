"""
H4 -- does influence used as a bounded weight beat the same influence used to
delete nodes?

Registered in harness/PREREGISTRATION.md on 2026-09-11, before the audit finished.
The motivation is Finding 9: BIND's per-node influence estimate is unstable
across initialisations to the point of sign reversal (German, seeds 28 vs 29:
Spearman -0.800, top-30% Jaccard 0.02 against a chance level of 0.18). A
discrete deletion cannot recover from a wrong estimate; a continuous weight
degrades in proportion to it.

The trap this design exists to avoid
------------------------------------
BIND deletes nodes and runs no fairness regulariser. We weight a fairness
regulariser and delete nothing. Comparing the two published methods head to
head would measure "regulariser or not" and report it as "soft or hard" -- the
same class of error this study has already had to retract twice. So both
interventions are applied to *one* backbone, at one fairness budget:

    base   phi uniform,              no deletion,          lambda_fair 0.2
    hard   phi uniform,              top-k harmful deleted, lambda_fair 0.2
    soft   phi from influence,       no deletion,          lambda_fair 0.2

`allocate()` normalises mean(phi) = 1 over the fairness set, so all three arms
spend the same total fairness pressure. The only thing that differs between
`hard` and `soft` is whether the same influence vector acts by removing nodes
or by redistributing that pressure. `base` is carried so each arm can also be
read against a common reference.

What is shared, exactly
-----------------------
One influence vector per (setting, split, init) cell, from `adapters/bind.py`
-- the official estimator under BIND's own published hyper-parameters (16
hidden, wd 1e-4, 1000 epochs, per-dataset scale), not the first-order stand-in
registered as `bind_influence` in signals.py. The estimator's seed is tied to
the init seed, so the instability that motivates H4 is exercised rather than
averaged away by computing influence once.

The deletion budget is BIND's own: `_order_and_budget` thins the ranking so no
two deleted nodes share a training node in their 1-hop computation graph and
stops where the influence changes sign, and k is the endpoint of the sweep
`3_removing_and_testing.py` publishes, round(0.3 * max_num). It is not chosen
on validation -- that would hand the hard arm a selection the soft arm does not
get. k is recorded per cell.

Where H4 cannot be tested
-------------------------
Wherever BIND's budget rule yields k = 0 the hard arm is the base arm and there
is no contrast. Those cells are run and reported, never dropped, but they are
excluded from the test: an arm that did not act is not an arm that failed.

The audit read `max_num` = 0 on Pokec-n and Pokec-n_g as exactly that. It was
not. Measured 2026-09-12, those settings return 0 of 500 finite influence
values at scale 60 -- LiSSA had diverged, and `_order_and_budget` scanning a
vector of NaNs for a sign change finds none and reports max_num = 0. Pokec-n
converges at 150. The escalation now lives in `adapters/bind.py`
(`influence_converged`), so a diverged estimate can no longer be read as an
empty budget here or in the audit.

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python harness/experiments/e8_h4_soft_vs_hard.py \
        --datasets german --splits 20 --inits 27
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import datasets as D                 # noqa: E402
from core import allocate as A                 # noqa: E402
from core.trainer import Config, train         # noqa: E402
from adapters.bind import SCALE as SCALE_LOOKUP  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
ARMS = ["base", "hard", "soft"]
SPLITS = [20, 21, 22, 23, 24, 25]
INITS = [27, 28, 29, 30, 31]
# Income and Credit are excluded: the audit could not compute BIND on them at
# this protocol, so there is no influence vector to share between the arms.
DATASETS = ["german", "nba", "recidivism", "pokec_z", "pokec_z_g",
            "pokec_n", "pokec_n_g"]


def delete_nodes(graph, victims: np.ndarray):
    """A copy of `graph` with `victims` removed and every index remapped.

    BIND deletes the node from the graph, not merely from the training set
    (`2_influence_computation_and_save.py:126` drops the rows and columns and
    re-symmetrises), so dropping it from `idx_train` alone would leave its
    edges carrying its features into its neighbours and would not be the same
    intervention.

    Only training nodes are ever deleted, so idx_val and idx_test survive and
    the test metrics of the three arms are computed on the same nodes.
    """
    n = graph.x.shape[0]
    keep = np.ones(n, dtype=bool)
    keep[victims] = False
    remap = np.full(n, -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))

    ei = graph.edge_index
    m = keep[ei[0]] & keep[ei[1]]
    ei = np.vstack([remap[ei[0][m]], remap[ei[1][m]]])

    def rm(idx):
        v = remap[np.asarray(idx)]
        return v[v >= 0]

    g = D.Graph(name=graph.name, x=graph.x[keep], y=graph.y[keep],
                sens=graph.sens[keep], edge_index=ei,
                idx_train=rm(graph.idx_train), idx_val=rm(graph.idx_val),
                idx_test=rm(graph.idx_test), idx_fair=rm(graph.idx_fair),
                stats=D.graph_stats(ei, graph.sens[keep], int(keep.sum())))
    return g


def influence_for(graph, ds: str, seed: int, device: str):
    """The official BIND influence vector and its deletion budget, for one cell.

    Returns (score_full, victims, max_num) where score_full is length n_nodes
    with the training nodes carrying their influence and every other node the
    minimum -- so `allocate()` can gate on it while nodes BIND says nothing
    about are never promoted.
    """
    import torch
    from torch_geometric.utils import to_scipy_sparse_matrix
    from adapters.bind import influence_converged, _order_and_budget, SCALE

    n = graph.x.shape[0]
    x = torch.tensor(graph.x, dtype=torch.float32)
    ei = torch.tensor(graph.edge_index, dtype=torch.long)
    y = torch.tensor(graph.y, dtype=torch.long)
    s = torch.tensor(graph.sens, dtype=torch.long)
    itr = torch.tensor(np.asarray(graph.idx_train), dtype=torch.long)
    iva = torch.tensor(np.asarray(graph.idx_val), dtype=torch.long)
    ite = torch.tensor(np.asarray(graph.idx_test), dtype=torch.long)
    Asp = to_scipy_sparse_matrix(ei, num_nodes=n).tocsr()
    Asp = ((Asp + Asp.T) > 0).astype(float)

    # LiSSA's recursion h <- v + (I - H/scale) h diverges when `scale` is
    # smaller than the largest Hessian eigenvalue, and it diverges to NaN
    # silently. That is what produced the all-NaN vectors on NBA: scale 60 gave
    # 0 of 100 finite, 150 gave 100 of 100. NBA's 60 was *our* guess -- it is
    # not in SCALE_FROM_PAPER -- so this was our hyper-parameter, not a defect
    # in BIND's estimator, and the earlier reading of it as one was wrong.
    #
    # The fix is mechanical: keep the published scale where it converges, and
    # otherwise take the first rung of a fixed ladder that does. The criterion
    # is finiteness, which is independent of whether `soft` or `hard` wins, so
    # this cannot tune the hypothesis. The scale actually used is recorded per
    # cell, and a cell that never converges is reported rather than dropped.
    final, scale_used, escalations = influence_converged(
        x, ei, Asp, y, s, itr, iva, ite, scale=SCALE[ds], nhid=16, lr=1e-3,
        wd=1e-4, epochs=1000, device=device, seed=seed)

    import networkx as nx
    from adapters.bind import _find123Nei
    G = nx.Graph(Asp)
    tr = itr.numpy()
    tr_set = set(tr.tolist())
    involving = [list(set(_find123Nei(G, int(v))[0]) & tr_set) for v in tr]
    harmful, max_num = _order_and_budget(final, itr, involving)

    k = int(round(0.3 * max_num))               # BIND's own sweep endpoint
    victims = tr[harmful[:k]] if k > 0 else np.array([], dtype=np.int64)

    score = np.full(n, float(final.min()), dtype=np.float64)
    score[tr] = final
    return score, victims, int(max_num), k, int(scale_used), escalations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--splits", nargs="*", type=int, default=SPLITS)
    ap.add_argument("--inits", nargs="*", type=int, default=INITS)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lambda-fair", type=float, default=0.2)
    ap.add_argument("--q-gate", type=float, default=0.7)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=os.path.join(RESULTS, "e8_h4.csv"))
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    n_cells = len(args.datasets) * len(args.splits) * len(args.inits)
    print(f"datasets={args.datasets}\nsplits={args.splits}  inits={args.inits}")
    print(f"arms={args.arms}\ncells = {n_cells}  ({n_cells * len(args.arms)} runs)\n")

    rows, t0 = [], time.time()
    done = 0
    for ds in args.datasets:
        for sp in args.splits:
            g = D.load(ds, split_seed=sp)
            for init in args.inits:
                done += 1
                t = time.time()
                try:
                    (score, victims, max_num, k,
                     scale_used, escalations) = influence_for(g, ds, init,
                                                              args.device)
                except Exception as e:                             # noqa: BLE001
                    print(f"  {ds} sp={sp} init={init} INFLUENCE FAILED "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                t_infl = time.time() - t

                # BIND's estimator can return an all-NaN vector: observed on
                # NBA split 21 init 29, where all 100 training nodes came back
                # NaN and max_num collapsed to 0. That is a property of the
                # estimator on that cell, not a transport error, and it belongs
                # in the table -- a cell that silently loses its soft arm looks
                # like a cell that was compared.
                nonfinite = not np.isfinite(score).all()
                cell = {"setting": ds, "split_seed": sp, "init_seed": init,
                        "seed": f"{sp}_{init}", "max_num": max_num, "k": k,
                        "n_deleted": len(victims), "infl_sec": round(t_infl, 1),
                        "infl_nonfinite": bool(nonfinite),
                        "scale_used": scale_used,
                        "scale_published": SCALE_LOOKUP[ds],
                        "scale_escalations": escalations}
                for arm in args.arms:
                    if arm == "soft" and nonfinite:
                        rows.append({**cell, "arm": arm})     # recorded, not run
                        continue
                    gg = delete_nodes(g, victims) if arm == "hard" and len(victims) else g
                    cfg = Config(signal="uniform", seed=init, epochs=args.epochs,
                                 warmup=args.warmup, lambda_fair=args.lambda_fair,
                                 q_gate=args.q_gate, device=args.device)
                    try:
                        if arm == "soft":
                            r = _train_with_score(gg, cfg, score)
                        else:
                            r = train(gg, cfg)
                    except Exception as e:                         # noqa: BLE001
                        print(f"  {ds} sp={sp} init={init} {arm} FAILED "
                              f"{type(e).__name__}: {e}", flush=True)
                        continue
                    rows.append({**cell, "arm": arm,
                                 **{f"test_{kk}": vv for kk, vv in r.test.items()},
                                 **{f"val_{kk}": vv for kk, vv in r.val.items()},
                                 # so a soft arm that silently collapsed to
                                 # uniform cannot be read as a soft arm that
                                 # was tried and did not help
                                 **{f"phi_{kk}": vv for kk, vv in r.phi.items()},
                                 "best_epoch": r.best_epoch,
                                 "epochs_run": r.epochs_run})
                print(f"  [{done}/{n_cells}] {ds:<10} sp={sp} init={init} "
                      f"k={k}/{max_num} scale={scale_used} infl={t_infl:.0f}s "
                      f"tot={time.time() - t:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[saved] {args.out}  ({len(df)} runs, {time.time() - t0:.0f}s)\n")
    return 0


def _train_with_score(graph, cfg: Config, score: np.ndarray):
    """`train()` with phi taken from a caller-supplied score instead of a signal.

    The trainer resolves phi through signals.get(cfg.signal); the influence
    vector comes from BIND's estimator and is not a member of that library, and
    registering it there would make it look like one of the eleven signals whose
    search was closed by deviation D4. It is injected for this experiment only.
    """
    import core.signals as S

    name = f"_h4_injected_{id(score)}"
    S.register(name)(lambda graph, state=None, _s=score: _s)
    try:
        return train(graph, replace(cfg, signal=name))
    finally:
        S.REGISTRY.pop(name, None)


if __name__ == "__main__":
    raise SystemExit(main())
