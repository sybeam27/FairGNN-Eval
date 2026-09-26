"""
E4/E5 re-read on the utility-fairness front.

Every conclusion in E2-E5 was taken from ΔDP alone, which is the failure this
study's own appendix warns about: a model that predicts one class scores
ΔDP = 0, and EDITS did exactly that on Income and NBA in the table being
audited. Re-reading the same runs as Pareto comparisons on (AUC up, ΔDP down)
changes which signals look like they work.

This is the script for the numbers quoted in `harness/PREREGISTRATION.md` under
"Design hypotheses"; they were computed ad hoc first and are reproduced here so
they can be regenerated rather than trusted.

Two comparisons, both paired on the (split, init) cell:

  vs `uniform`   does allocating by this signal beat not allocating?
  vs `random`    does allocating by *this* signal beat allocating by nothing in
                 particular? `random` is the control the whole study turns on --
                 a signal that cannot beat it is not carrying information, and
                 the ΔDP-only reading cannot see that.

Usage
-----
    python harness/experiments/analyze_allocation_pareto.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CELL = ["split_seed", "init_seed"]


def dominates(a_auc, a_dp, b_auc, b_dp) -> int:
    if a_auc >= b_auc and a_dp <= b_dp and (a_auc > b_auc or a_dp < b_dp):
        return 1
    if b_auc >= a_auc and b_dp <= a_dp and (b_auc > a_auc or b_dp < a_dp):
        return -1
    return 0


def head_to_head(d: pd.DataFrame, a: str, b: str) -> dict | None:
    out = []
    for _, sub in d.groupby("setting"):
        x = sub[sub.signal == a][CELL + ["test_auc", "test_dp"]]
        y = sub[sub.signal == b][CELL + ["test_auc", "test_dp"]]
        m = x.merge(y, on=CELL, suffixes=("", "_b"))
        out.append(m)
    if not out:
        return None
    m = pd.concat(out, ignore_index=True)
    if m.empty:
        return None
    c = [dominates(r.test_auc, r.test_dp, r.test_auc_b, r.test_dp_b)
         for r in m.itertuples()]
    w, l = c.count(1), c.count(-1)
    return {"n": len(c), "dominates": w, "dominated": l,
            "incomparable": c.count(0), "net": w - l,
            "p": stats.binomtest(w, w + l, 0.5).pvalue if w + l else 1.0,
            "dp_only_win": float((m.test_dp < m.test_dp_b).mean()),
            "d_auc": float((m.test_auc - m.test_auc_b).mean()),
            "d_dp": float((m.test_dp - m.test_dp_b).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4", default=os.path.join(ROOT, "harness", "results",
                                                 "e4_signal_swap_split.csv"))
    ap.add_argument("--e5", default=os.path.join(ROOT, "harness", "results",
                                                 "e5_operator_signals.csv"))
    args = ap.parse_args()

    d = pd.concat([pd.read_csv(args.e4), pd.read_csv(args.e5)], ignore_index=True)
    pd.set_option("display.width", 220)
    sigs = sorted(d.signal.unique())
    print(f"{len(d)} runs  {d.setting.nunique()} settings  {len(sigs)} signals  "
          f"{d.split_seed.nunique()} splits x {d.init_seed.nunique()} inits\n")

    for ref in ("uniform", "random"):
        rows = []
        for s in sigs:
            if s == ref:
                continue
            r = head_to_head(d, s, ref)
            if r:
                rows.append({"signal": s, **r})
        t = pd.DataFrame(rows).sort_values("net", ascending=False)
        t["dominance_rate"] = (t["dominates"] / t["n"]).round(3)
        print("=" * 104)
        print(f"vs `{ref}`   Pareto on (AUC, ΔDP), paired on (split, init)")
        print("=" * 104)
        print(t[["signal", "n", "dominates", "dominated", "incomparable", "net",
                 "p", "dominance_rate", "dp_only_win", "d_auc", "d_dp"]]
              .round(4).to_string(index=False))
        if ref == "random":
            better = t[(t.net > 0) & (t.p < 0.05)]["signal"].tolist()
            worse = t[(t.net < 0) & (t.p < 0.05)]["signal"].tolist()
            print(f"\nbeats random: {better or 'none'}")
            print(f"loses to random: {worse or 'none'}")
            print("\nA signal that cannot beat a random ranking is not carrying "
                  "information about\nwhere fairness pressure should go, whatever "
                  "its ΔDP column says.")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
