"""
Is Finding 14's 75.9% the fairness intervention, or partly the backbone?

Finding 14 reports FairGate Pareto-dominating plain GCN on 75.9% of cells. The
two rows are not the same model in more ways than the fairness term:

    audit `GNN`      the baseline codebase's GCN, its own training loop
    E10 `backbone`   FairGate's architecture and training loop, lambda_fair = 0

If `backbone` already dominates `GNN`, then part of the 75.9% is architecture,
initialisation and optimisation rather than the fairness intervention, and every
rung of the E10 ladder sits on top of that offset. The ladder localises *where*
the fairness effect is; this localises how much of the headline is fairness at
all.

Three comparisons, paired on (setting, split_seed), Pareto on (AUC up, dP down):

    backbone      vs GNN        the backbone offset itself
    all_alloc     vs GNN        should reproduce Finding 14's FairGate row
    all_alloc     vs backbone   the whole fairness stack, against its own backbone

The third is the one that matters. It is Finding 14's claim with the backbone
held fixed, and it is the number the paper can defend as "what the fairness
intervention buys".

Preprocessing: the audit's `GNN` row is the as-submitted arm, which loads
unnormalised, while FairGate normalises. Finding 14 measured that forcing
normalisation costs plain GCN 0.0072 AUC on average and 0.131 on Recidivism, so
the normunified arm's GNN is used when available and both are reported when they
differ.

Usage
-----
    python harness/experiments/analyze_backbone_vs_gnn.py
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_audit import dominates                      # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
CELL = ["setting", "split_seed"]


def paired(x: pd.DataFrame, y: pd.DataFrame, label: str) -> dict | None:
    m = x.merge(y, on=CELL, suffixes=("", "_b"))
    if m.empty:
        return None
    c = [dominates(r.roc_auc_mean, r.dp_mean, r.roc_auc_mean_b, r.dp_mean_b)
         for r in m.itertuples()]
    w, l = c.count(1), c.count(-1)
    return {"label": label, "n": len(c), "wins": w, "losses": l, "inc": c.count(0),
            "net": w - l,
            "p": stats.binomtest(w, w + l, 0.5).pvalue if w + l else 1.0,
            "rate": w / len(c),
            "d_auc": float((m.roc_auc_mean - m.roc_auc_mean_b).mean()),
            "d_dp": float((m.dp_mean - m.dp_mean_b).mean())}


def show(r: dict | None) -> None:
    if r is None:
        print("  (no matched cells)")
        return
    print(f"  {r['label']:<28s} n={r['n']:3d}  win={r['wins']:3d} "
          f"lose={r['losses']:3d} inc={r['inc']:3d}  net={r['net']:+4d}  "
          f"p={r['p']:.4f}  rate={r['rate']:.3f}  "
          f"dAUC={r['d_auc']:+.4f}  ddP={r['d_dp']:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default=os.path.join(RESULTS, "e10_ladder.csv"))
    ap.add_argument("--audit", default=os.path.join(RESULTS, "e7_audit_merged.csv"))
    ap.add_argument("--unified", default=os.path.join(RESULTS,
                                                      "e7_baseline_audit_normunified.csv"))
    a = ap.parse_args()

    L = pd.read_csv(a.ladder)
    if "status" in L:
        L = L[L.status.fillna("ok").eq("ok")]           # NaN == resumed
    L = L.dropna(subset=["roc_auc_mean", "dp_mean"])
    A = pd.read_csv(a.audit)
    gnn_sub = A[A.model == "GNN"]
    gnn_uni = (pd.read_csv(a.unified).query("model == 'GNN'")
               if os.path.exists(a.unified) else None)

    bb = L[L.arm == "backbone"]
    al = L[L.arm == "all_alloc"]
    print(f"\nladder settings: {sorted(L.setting.unique())}")
    print(f"backbone cells {len(bb)}   all_alloc cells {len(al)}   "
          f"audit GNN cells {len(gnn_sub)}\n")

    for name, gnn in (("GNN, as submitted", gnn_sub),
                      ("GNN, normalisation unified", gnn_uni)):
        if gnn is None or gnn.empty:
            continue
        print(f"=== against {name} ===")
        show(paired(bb, gnn, "backbone vs GNN"))
        show(paired(al, gnn, "all_alloc vs GNN"))
        print()

    print("=== the fairness stack against its own backbone ===")
    r = paired(al, bb, "all_alloc vs backbone")
    show(r)
    if r:
        print("\nThis is Finding 14's claim with the backbone held fixed. Whatever")
        print("`all_alloc vs GNN` says, this is what the fairness intervention buys.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
