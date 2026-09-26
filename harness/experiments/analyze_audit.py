"""
E7 read on the utility-fairness front, not on disparity alone.

Ranking fair-GNN methods by ΔDP is the convention and it is not sound: a model
that predicts one class scores ΔDP = 0. That is not a hypothetical -- EDITS
did exactly this on Income and NBA in the table this study is auditing. The
paper's own appendix says so and the table still ranks on ΔDP.

So every comparison here is a Pareto comparison on (AUC up, ΔDP down), counted
per (split, init) cell. Nothing is scalarised: any weighting of accuracy against
disparity is a free parameter, and with one in hand a reader can be given
whichever ranking is wanted. Where two methods trade off, they are reported as
incomparable rather than ordered.

Three questions, in order of what they cost to answer.

  1. Does a fairness method beat plain GCN at all? Not "is its ΔDP lower" but
     "does it dominate", per cell, with the ΔDP-only rate printed beside it so
     the difference between the two readings is visible.
  2. Does the split change the answer? The same comparison at `split_seed = 20`
     alone -- the field's protocol, and the one every published number here was
     computed under -- against all six.
  3. How much of each method's ΔDP is bought with accuracy? Mean ΔAUC against
     GCN on matched cells.

Usage
-----
    python harness/experiments/analyze_audit.py [--csv harness/results/e7_audit_full.csv]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "GNN"                      # plain GCN, no fairness intervention
CELL = ["split_seed"]             # each row already averages the 5 init seeds


def dominates(a_auc, a_dp, b_auc, b_dp) -> int:
    """+1 if A Pareto-dominates B on (AUC up, ΔDP down), -1 if dominated, else 0."""
    if a_auc >= b_auc and a_dp <= b_dp and (a_auc > b_auc or a_dp < b_dp):
        return 1
    if b_auc >= a_auc and b_dp <= a_dp and (b_auc > a_auc or b_dp < a_dp):
        return -1
    return 0


def pairwise(df: pd.DataFrame, a: str, b: str, splits=None) -> dict | None:
    x = df[df.model == a]
    y = df[df.model == b]
    if splits is not None:
        x, y = x[x.split_seed.isin(splits)], y[y.split_seed.isin(splits)]
    m = x.merge(y, on=["setting"] + CELL, suffixes=("", "_b"))
    if m.empty:
        return None
    c = [dominates(r.roc_auc_mean, r.dp_mean, r.roc_auc_mean_b, r.dp_mean_b)
         for r in m.itertuples()]
    w, l = c.count(1), c.count(-1)
    p = stats.binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
    return {"n": len(c), "dominates": w, "dominated": l, "incomparable": c.count(0),
            "net": w - l, "p": p,
            "dp_only_win": float((m.dp_mean < m.dp_mean_b).mean()),
            "d_auc": float((m.roc_auc_mean - m.roc_auc_mean_b).mean()),
            "d_dp": float((m.dp_mean - m.dp_mean_b).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "harness", "results",
                                                  "e7_audit_full.csv"))
    ap.add_argument("--complete-only", action="store_true",
                    help="drop methods that have not finished all cells")
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    n_cells = d.groupby("model").size()
    full = int(n_cells.max())
    if args.complete_only:
        d = d[d.model.isin(n_cells[n_cells == full].index)]
    pd.set_option("display.width", 240)
    print(f"{len(d)} cells   methods: {dict(n_cells)}   "
          f"settings {d.setting.nunique()}  splits {sorted(d.split_seed.unique())}\n")

    methods = [m for m in d.model.unique() if m != BASE]

    print("=" * 108)
    print(f"1. vs {BASE} (plain GCN).  Pareto on (AUC, ΔDP) per cell; "
          f"`dp_only_win` is the conventional reading of the same cells.")
    print("=" * 108)
    rows = []
    for m in methods:
        r = pairwise(d, m, BASE)
        if r:
            rows.append({"model": m, **r})
    t = pd.DataFrame(rows).sort_values("net", ascending=False)
    t["dominance_rate"] = (t["dominates"] / t["n"]).round(3)
    print(t[["model", "n", "dominates", "dominated", "incomparable", "net", "p",
             "dominance_rate", "dp_only_win", "d_auc", "d_dp"]]
          .round(4).to_string(index=False))
    print("\n`net` > 0 with small p: the method is on the better side of the "
          "front more often\nthan the worse. A large `dp_only_win` beside a "
          "small `dominance_rate` means the\nconventional table would rank it "
          "well on disparity it paid for in accuracy.")

    print("\n" + "=" * 108)
    print("2. Does the split change the answer?  split 20 alone (the protocol "
          "every published\n   number here used) against all six.")
    print("=" * 108)
    rows = []
    for m in methods:
        a = pairwise(d, m, BASE, splits=[20])
        b = pairwise(d, m, BASE)
        if a and b:
            rows.append({"model": m,
                         "split20_dominance": a["dominates"] / a["n"],
                         "all6_dominance": b["dominates"] / b["n"],
                         "split20_dp_win": a["dp_only_win"],
                         "all6_dp_win": b["dp_only_win"],
                         "split20_net": a["net"], "all6_net": b["net"]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n" + "=" * 108)
    print("3. Per setting: which methods dominate GCN, and does it survive "
          "re-splitting?")
    print("=" * 108)
    rows = []
    for s, sub in d.groupby("setting"):
        r = {"setting": s}
        for m in methods:
            x = pairwise(sub, m, BASE)
            r[m] = f"{x['dominates']}/{x['n']}" if x else "-"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nEntries are cells (of six splits) in which the method Pareto-"
          "dominates plain GCN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
