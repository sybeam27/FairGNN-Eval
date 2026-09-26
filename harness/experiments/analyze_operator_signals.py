"""
Does a correctly measured structural signal also lose?

Deviation D4 added `prop_cross`, `prop_dev` and `prop_sens`: the same three
structural quantities the library already had, measured on the operator the
backbone is defined with rather than on a one-hop neighbour count. It also fixed
the stopping rule before the answer was known --

    if none of them beats `bind_influence` on any setting with a paired
    interval excluding zero under the enlarged Holm family, the structural line
    is closed and the conclusion is recorded as *topology-based allocation does
    not work on this benchmark; a model-dependent influence estimate does*.

-- and stated the expected outcome, which was that the rule fires.

This script applies exactly that. It merges E5 into E4 so all eleven signals sit
on the same (split, init) cells, corrects over the enlarged family of
9 x 10 = 90 signal-vs-uniform comparisons, and reports the head-to-head against
`bind_influence` that the stopping rule is written in terms of.

Usage
-----
    python harness/experiments/analyze_operator_signals.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "uniform"
REF = "bind_influence"
NEW = ["prop_cross", "prop_dev", "prop_sens"]
CELL = ["split_seed", "init_seed"]


def holm(p: np.ndarray) -> np.ndarray:
    n, order, adj, run = len(p), np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def paired(df: pd.DataFrame, ds: str, a: str, b: str, metric: str) -> dict | None:
    col = f"test_{metric}"
    A = df[(df.setting == ds) & (df.signal == a)][CELL + [col]]
    B = df[(df.setting == ds) & (df.signal == b)][CELL + [col]]
    m = A.merge(B, on=CELL, suffixes=("", "_b"))
    if len(m) < 3:
        return None
    d = (m[col] - m[f"{col}_b"]).to_numpy()
    n = len(d)
    tc = stats.t.ppf(0.975, n - 1)
    se = d.std(ddof=1) / np.sqrt(n)
    try:
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return {"setting": ds, "signal": a, "vs": b, "n": n, "mean_d": d.mean(),
            "ci_lo": d.mean() - tc * se, "ci_hi": d.mean() + tc * se,
            "sd_d": d.std(ddof=1), "wins": int((d < 0).sum()), "p_raw": p}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4", default=os.path.join(ROOT, "results", "e4_signal_swap_split.csv"))
    ap.add_argument("--e5", default=os.path.join(ROOT, "results", "e5_operator_signals.csv"))
    ap.add_argument("--metric", default="dp")
    args = ap.parse_args()

    e4 = pd.read_csv(args.e4)
    e5 = pd.read_csv(args.e5)
    df = pd.concat([e4, e5], ignore_index=True)
    pd.set_option("display.width", 220)

    sigs = sorted(df.signal.unique())
    sets = sorted(set(e4.setting) & set(e5.setting))
    df = df[df.setting.isin(sets)]
    print(f"{len(df)} runs, {len(sets)} settings, {len(sigs)} signals: {sigs}")
    print(f"new in E5: {[s for s in NEW if s in sigs]}\n")

    # ---- vs uniform, corrected over the enlarged family ----
    rows = [r for ds in sets for s in sigs if s != BASE
            for r in [paired(df, ds, s, BASE, args.metric)] if r]
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p_raw"].to_numpy())
    t["verdict"] = np.where((t.ci_hi < 0) & (t.p_holm <= 0.05), "helps",
                     np.where((t.ci_lo > 0) & (t.p_holm <= 0.05), "hurts", "undecided"))
    print("=" * 100)
    print(f"vs {BASE}, Holm over the enlarged family of {len(t)} comparisons "
          f"({len(sets)} settings x {len(sigs) - 1} signals)")
    print("=" * 100)
    piv = t.pivot_table(index="setting", columns="signal", values="mean_d").round(4)
    vd = t.pivot_table(index="setting", columns="signal", values="verdict", aggfunc="first")
    order = [c for c in sigs if c != BASE]
    show = piv[order].astype(str)
    for c in order:
        show[c] = show[c] + vd[c].map({"helps": " H", "hurts": " X"}).fillna("  ")
    print(show.to_string())
    print("\nverdicts:", t.verdict.value_counts().to_dict())

    # ---- the stopping rule ----
    rows = [r for ds in sets for s in NEW if s in sigs
            for r in [paired(df, ds, s, REF, args.metric)] if r]
    h = pd.DataFrame(rows)
    h["p_holm"] = holm(h["p_raw"].to_numpy())
    h["verdict"] = np.where((h.ci_hi < 0) & (h.p_holm <= 0.05), "beats bind",
                     np.where((h.ci_lo > 0) & (h.p_holm <= 0.05), "loses", "tie"))
    print("\n" + "=" * 100)
    print(f"THE STOPPING RULE (deviation D4): does any of {NEW} beat `{REF}` anywhere?")
    print("=" * 100)
    print(h[["setting", "signal", "mean_d", "ci_lo", "ci_hi", "wins", "p_holm", "verdict"]]
            .sort_values("mean_d").round(4).to_string(index=False))

    n_beat = int((h.verdict == "beats bind").sum())
    print("\n" + "-" * 100)
    if n_beat == 0:
        print("The rule FIRES. No operator-matched structural signal beats the "
              "influence estimate\nanywhere. Recorded conclusion: topology-based "
              "allocation does not work on this\nbenchmark; a model-dependent "
              "influence estimate does. No further structural signal\nis "
              "proposed, and the failure of our own draft's central signal is "
              "part of the finding.")
    else:
        print(f"The rule does NOT fire: {n_beat} cell(s) beat `{REF}`. The "
              "structural line stays open;\nthe next question is whether a "
              "pre-training rule can identify those cells, and that\nmust be "
              "evaluated leave-one-setting-out, not read off this table.")
        print(h[h.verdict == "beats bind"].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
