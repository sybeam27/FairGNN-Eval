"""
E7, differenced across preprocessing arms.

`analyze_audit.py` reports that three methods Pareto-dominate plain GCN on a
large majority of cells. Those same three raise AUC by 0.043-0.076, and they
are also the methods whose loaders normalise features while GNN's and NIFTY's
do not. Dominance and preprocessing are confounded in that arm, and no amount
of re-reading it separates them.

This script reads both arms and reports the difference.

  as_submitted  each method keeps the preprocessing its own implementation
                shipped: GNN and NIFTY unnormalised everywhere, FairGNN
                normalised on NBA and German only, FairGB/FairGT/FairGate
                normalised everywhere.
  normunified   every method normalised, via --force_feature_normalize 1.
                Only GNN, NIFTY and FairGNN change; the other four already
                normalised, and their as-submitted rows are carried over
                unchanged rather than re-run.

Two things are printed.

  1. Per method, what the override did to it on matched (setting, split) cells:
     ΔAUC and ΔΔP between arms. Normalisation is not uniformly good --
     measured directly it is +0.074 AUC on Income and -0.050 on Credit -- so
     a method losing here is a result, not a bug, and is printed as such.

  2. The dominance-over-GCN table recomputed inside the unified arm, beside
     the as-submitted one. The baseline moves too: GNN is one of the three
     methods the override changes, so this is not the same comparison with a
     tidier control, and both columns are needed to say anything.

What this cannot do: it holds feature normalisation fixed, not preprocessing
in general. BIND keeps its own published pipeline (1-layer GCN, its own
feature_norm) and is excluded from the unified arm entirely -- its row was
never part of the comparison being audited.

Usage
-----
    python harness/experiments/analyze_arms.py
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_audit import BASE, CELL, pairwise          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "harness", "results")
# The override is a no-op for these: their loaders already normalise, so the
# as-submitted rows are the unified rows. Re-running would only add sampling
# noise to a comparison that is supposed to be exact.
UNCHANGED = ["FairGB", "FairGT", "FairGate"]
CHANGED = ["GNN", "NIFTY", "FairGNN"]


def load(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"missing arm CSV: {path}")
    d = pd.read_csv(path)
    d = d[d.get("status", "ok").fillna("ok").eq("ok")] if "status" in d else d
    return d.assign(arm=label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submitted", default=os.path.join(RESULTS, "e7_audit_merged.csv"))
    ap.add_argument("--unified", default=os.path.join(RESULTS,
                                                      "e7_baseline_audit_normunified.csv"))
    a = ap.parse_args()

    sub = load(a.submitted, "as_submitted")
    uni = load(a.unified, "normunified")

    key = ["model", "setting"] + CELL
    carried = sub[sub.model.isin(UNCHANGED)]
    uni_full = pd.concat([uni[uni.model.isin(CHANGED)], carried], ignore_index=True)

    print(f"\nas_submitted : {len(sub)} cells, {sub.model.nunique()} methods")
    print(f"normunified  : {len(uni_full)} cells "
          f"({len(uni[uni.model.isin(CHANGED)])} re-run, {len(carried)} carried over)")

    # ---- 1. what the override did, per method -----------------------------
    m = (sub[key + ["roc_auc_mean", "dp_mean"]]
         .merge(uni[key + ["roc_auc_mean", "dp_mean"]], on=key, suffixes=("_s", "_u")))
    if m.empty:
        print("\nno matched cells between arms yet")
    else:
        print("\n=== effect of forcing normalisation (matched cells) ===")
        print(f"{'model':10s} {'n':>4s} {'d_auc':>8s} {'d_dp':>8s}  "
              f"{'auc_up':>7s} {'dp_down':>8s}")
        for name, g in m.groupby("model"):
            d_auc = (g.roc_auc_mean_u - g.roc_auc_mean_s)
            d_dp = (g.dp_mean_u - g.dp_mean_s)
            print(f"{name:10s} {len(g):4d} {d_auc.mean():+8.4f} {d_dp.mean():+8.4f}  "
                  f"{(d_auc > 0).mean():7.2f} {(d_dp < 0).mean():8.2f}")
        print("\nper setting, averaged over the changed methods:")
        s = m.assign(d_auc=m.roc_auc_mean_u - m.roc_auc_mean_s).groupby("setting").d_auc
        for k, v in s.mean().sort_values().items():
            print(f"  {k:12s} {v:+.4f}")

    # ---- 2. dominance over GCN, inside each arm ---------------------------
    for label, df in (("as_submitted", sub), ("normunified", uni_full)):
        print(f"\n=== dominates plain GCN -- arm: {label} ===")
        print(f"{'model':10s} {'n':>4s} {'dom':>5s} {'domd':>5s} {'net':>5s} "
              f"{'p':>8s} {'rate':>6s} {'dp_only':>8s} {'d_auc':>8s} {'d_dp':>8s}")
        rows = []
        for name in sorted(set(df.model) - {BASE}):
            r = pairwise(df, name, BASE)
            if r:
                rows.append((name, r))
        for name, r in sorted(rows, key=lambda t: -t[1]["net"]):
            print(f"{name:10s} {r['n']:4d} {r['dominates']:5d} {r['dominated']:5d} "
                  f"{r['net']:+5d} {r['p']:8.4f} "
                  f"{r['dominates'] / r['n']:6.3f} {r['dp_only_win']:8.3f} "
                  f"{r['d_auc']:+8.4f} {r['d_dp']:+8.4f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
