"""
E10 read as registered: each rung against the one below it, on the Pareto front.

The ladder is an ablation downward from the published FairGate. Reading it is
therefore a sequence of paired comparisons between adjacent rungs, plus two
comparisons that are not adjacent and exist to catch specific mistakes.

    backbone -> out            what a prediction-level regulariser alone buys
    out -> rep_out             what representation alignment adds
    rep_out -> all_uniform     what structural consistency adds
    all_uniform -> all_alloc   what the allocation adds                 <- H6
    all_alloc  vs all_perm     ranking, or merely a spread of weights
    all_alloc  vs all_uniform1 what the shipped ablation actually varies

Rungs 1-3 all carry `uniform_budget`, so they spend exactly the budget
`all_alloc` spends and each step is attributable to its component. `all_uniform1`
does not -- it is ones(N), 1.4x to 1.75x more total pressure -- which is why it
appears only in the last line and never as a rung.

Every comparison is Pareto on (AUC up, dP down), paired on
(setting, split_seed), sign test over comparable cells, Holm across the family.
Nothing is scalarised: cells where two rungs trade off are counted as
incomparable, because any weighting of accuracy against disparity is a free
parameter and with one in hand the ranking can be chosen.

A note on `status`. Both runners set it only on a fresh subprocess run; a cell
read back from its cache carries NaN. In the audit that was 342 rows of 366.
NaN here means "resumed", not "failed", and is treated as ok -- filtering it out
would silently compute the ladder on whichever cells happened not to be cached.

Usage
-----
    python harness/experiments/analyze_ladder.py
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
LADDER = ["backbone", "out", "rep_out", "all_uniform", "all_alloc"]
STEPS = [(b, a) for a, b in zip(LADDER, LADDER[1:])]      # (upper, lower)
EXTRA = [("all_alloc", "all_perm"), ("all_alloc", "all_uniform1")]


def paired(df: pd.DataFrame, a: str, b: str) -> dict | None:
    x, y = df[df.arm == a], df[df.arm == b]
    m = x.merge(y, on=CELL, suffixes=("", "_b"))
    if m.empty:
        return None
    c = [dominates(r.roc_auc_mean, r.dp_mean, r.roc_auc_mean_b, r.dp_mean_b)
         for r in m.itertuples()]
    w, l = c.count(1), c.count(-1)
    return {"n": len(c), "wins": w, "losses": l, "inc": c.count(0), "net": w - l,
            "p": stats.binomtest(w, w + l, 0.5).pvalue if w + l else 1.0,
            "rate": w / len(c),
            "dp_only": float((m.dp_mean < m.dp_mean_b).mean()),
            "d_auc": float((m.roc_auc_mean - m.roc_auc_mean_b).mean()),
            "d_dp": float((m.dp_mean - m.dp_mean_b).mean()),
            "cells": m}


def holm(rows: list[tuple[str, dict]]) -> dict[str, float]:
    rs = sorted(rows, key=lambda t: t[1]["p"])
    k, prev, out = len(rs), 0.0, {}
    for i, (name, r) in enumerate(rs):
        prev = max(prev, min(1.0, r["p"] * (k - i)))
        out[name] = prev
    return out


def line(tag: str, r: dict | None, adj: float | None = None) -> None:
    if r is None:
        print(f"  {tag:<30s} (no cells)")
        return
    h = f"  Holm={adj:.4f}" if adj is not None else ""
    print(f"  {tag:<30s} n={r['n']:3d}  win={r['wins']:3d} lose={r['losses']:3d} "
          f"inc={r['inc']:3d}  net={r['net']:+4d}  p={r['p']:.4f}{h}  "
          f"dAUC={r['d_auc']:+.4f}  ddP={r['d_dp']:+.4f}  dp_only={r['dp_only']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(RESULTS, "e10_ladder.csv"))
    a = ap.parse_args()
    if not os.path.exists(a.csv):
        sys.exit(f"not run yet: {a.csv}")

    d = pd.read_csv(a.csv)
    if "status" in d:
        d = d[d.status.fillna("ok").eq("ok")]            # NaN == resumed, not failed
    d = d.dropna(subset=["roc_auc_mean", "dp_mean"])

    have = d.groupby("arm").size()
    cells = d.groupby(CELL).ngroups
    full = d.groupby(CELL).arm.nunique().eq(len(LADDER) + 2)
    print(f"\ncells present: {cells}   complete (all 7 arms): {int(full.sum())}")
    print("rows per arm: " + ", ".join(f"{k}={v}" for k, v in have.items()))
    print("\nsettings: " + ", ".join(sorted(d.setting.unique())))

    print("\n" + "=" * 104)
    print("THE LADDER — each rung against the one below it, Pareto on (AUC up, dP down)")
    print("=" * 104)
    rows = [(f"{lo} -> {up}", paired(d, up, lo)) for up, lo in STEPS]
    rows += [(f"{a_} vs {b_}", paired(d, a_, b_)) for a_, b_ in EXTRA]
    rows = [(t, r) for t, r in rows if r]
    adj = holm(rows)
    for t, r in rows:
        line(t, r, adj[t])

    print("\nmean position of each rung (not a comparison, an orientation):")
    print(f"  {'arm':<14s} {'AUC':>8s} {'dP':>8s} {'acc':>8s}")
    for arm in LADDER + ["all_perm", "all_uniform1"]:
        g = d[d.arm == arm]
        if len(g):
            print(f"  {arm:<14s} {g.roc_auc_mean.mean():8.4f} {g.dp_mean.mean():8.4f} "
                  f"{g.acc_mean.mean():8.4f}")

    h6 = dict(rows).get("all_uniform -> all_alloc")
    if h6:
        print("\n" + "=" * 104)
        print("H6 — does the allocation add anything over the same budget spent uniformly?")
        print("=" * 104)
        p = adj["all_uniform -> all_alloc"]
        if p >= 0.05:
            print(f"  FALSIFIED as registered: all_alloc does not beat all_uniform "
                  f"(net {h6['net']:+d}, Holm {p:.4f}).")
        elif h6["net"] > 0:
            print(f"  HOLDS: net {h6['net']:+d}, Holm {p:.4f}.")
        else:
            print(f"  FALSIFIED, and in the other direction: all_uniform beats "
                  f"all_alloc (net {h6['net']:+d}, Holm {p:.4f}).")
        perm = dict(rows).get("all_alloc vs all_perm")
        if perm:
            print(f"  Against the permutation control: net {perm['net']:+d}, "
                  f"Holm {adj['all_alloc vs all_perm']:.4f}.")
            print("  If the allocation beats uniform but not the permutation, what "
                  "helps is having a\n  spread of weights, not the ranking the "
                  "signal produces.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
