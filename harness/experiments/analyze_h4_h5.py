"""
H4 and H5, read exactly as registered.

Both hypotheses were written into harness/PREREGISTRATION.md on 2026-09-11, with
their falsification conditions, before either runner existed. This script
implements those conditions and nothing else. It reports what they say to
report, including when that is a failure.

    H4  soft weighting Pareto-dominates hard deletion of the same influence,
        more often than the reverse.
        Falsified if the soft arm does not win with p < 0.05 -- and equally if
        it wins on dP while losing on the Pareto count, "the same mistake in a
        new place".

    H5  the constrained arm's Pareto dominance rate over uniform is higher than
        the unconstrained arm's.
        Falsified if it is not higher. The registered prior also expects the
        dP-only win rate to fall; that trade is the point, not a disappointment.

Every comparison is Pareto on (AUC up, dP down), per (setting, split, init)
cell, with a sign test over the cells that are comparable. Cells where the two
arms are incomparable are counted and never broken by scalarising: any
weighting of accuracy against disparity is a free parameter, and with one in
hand the ranking can be chosen.

H4 has a prerequisite the data must be checked against. Where BIND's own budget
rule yields k = 0 the hard arm *is* the base arm and there is no contrast; the
audit already measured max_num = 0 on Pokec-n and Pokec-n_g. Those cells are
reported separately and excluded from the H4 test, because including them would
dilute the comparison with cells in which nothing was compared.

Usage
-----
    python harness/experiments/analyze_h4_h5.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_audit import dominates                     # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
CELL = ["setting", "split_seed", "init_seed"]


def paired(df: pd.DataFrame, a: str, b: str) -> dict | None:
    """Pareto outcome of arm `a` against arm `b`, per cell."""
    x, y = df[df.arm == a], df[df.arm == b]
    m = x.merge(y, on=CELL, suffixes=("", "_b"))
    if m.empty:
        return None
    c = [dominates(r.test_auc, r.test_dp, r.test_auc_b, r.test_dp_b)
         for r in m.itertuples()]
    w, l = c.count(1), c.count(-1)
    return {"n": len(c), "wins": w, "losses": l, "incomparable": c.count(0),
            "net": w - l,
            "p": stats.binomtest(w, w + l, 0.5).pvalue if w + l else 1.0,
            "rate": w / len(c),
            "dp_only_win": float((m.test_dp < m.test_dp_b).mean()),
            "d_auc": float((m.test_auc - m.test_auc_b).mean()),
            "d_dp": float((m.test_dp - m.test_dp_b).mean()),
            "cells": m}


def line(tag: str, r: dict | None) -> None:
    if r is None:
        print(f"  {tag:<34s} (no cells)")
        return
    print(f"  {tag:<34s} n={r['n']:4d}  win={r['wins']:3d} lose={r['losses']:3d} "
          f"inc={r['incomparable']:3d}  net={r['net']:+4d}  p={r['p']:.4f}  "
          f"rate={r['rate']:.3f}  dp_only={r['dp_only_win']:.3f}  "
          f"dAUC={r['d_auc']:+.4f}  ddP={r['d_dp']:+.4f}")


def do_h4(path: str) -> None:
    if not os.path.exists(path):
        print(f"\n[H4] not run yet: {path}")
        return
    d = pd.read_csv(path)
    print("\n" + "=" * 100)
    print("H4 -- soft weighting vs hard deletion of the same BIND influence")
    print("=" * 100)

    k = d.groupby("setting")[["k", "max_num"]].mean()
    zero = d.groupby("setting")["k"].apply(lambda s: float((s == 0).mean()))
    print("\ndeletion budget actually realised (BIND's own rule):")
    print(f"  {'setting':<12s} {'mean k':>7s} {'mean max_num':>13s} {'frac k=0':>9s}")
    for s in k.index:
        print(f"  {s:<12s} {k.loc[s, 'k']:7.2f} {k.loc[s, 'max_num']:13.2f} "
              f"{zero[s]:9.2f}")

    # A cell whose soft arm is missing (or present but unscored) lost its
    # influence vector to a non-finite estimate. Counted, never dropped
    # quietly: the difference between "compared and lost" and "never compared"
    # is the whole reason this section exists.
    have = d.dropna(subset=["test_auc"]) if "test_auc" in d else d
    n_cells = have.groupby(CELL).ngroups
    missing = (have[have.arm == "base"].merge(have[have.arm == "soft"], on=CELL,
                                              how="left", suffixes=("", "_s")))
    n_missing = int(missing.test_auc_s.isna().sum()) if "test_auc_s" in missing else 0
    if n_missing:
        print(f"\ncells with no usable soft arm (BIND influence non-finite): "
              f"{n_missing} of {n_cells}")
        if "infl_nonfinite" in d:
            bad = d[d.infl_nonfinite.fillna(False).astype(bool)]
            if len(bad):
                print("  " + ", ".join(sorted(
                    f"{r.setting}/sp{r.split_seed}/init{r.init_seed}"
                    for r in bad.drop_duplicates(CELL).itertuples())))
        print("  These are excluded from every comparison below.")
        print("  They also matter for reading the k = 0 column. A non-finite "
              "influence vector\n  forces max_num to 0 -- the sign-change scan "
              "finds no sign change in NaNs -- so\n  \"max_num = 0\" conflates "
              "two different events: BIND's rule finding nothing to\n  delete, "
              "and BIND's estimator having failed. The audit reported max_num = "
              "0 on\n  Pokec-n and Pokec-n_g as the former; measured, it was "
              "the latter -- 0 of 500\n  finite values at scale 60. "
              "adapters/bind.py now escalates the scale until the\n  estimate "
              "converges, so this column means what it says.")

    live = have[have.n_deleted > 0]
    dead_cells = int((have[have.arm == "hard"].n_deleted == 0).sum())
    n_hard = int((have.arm == "hard").sum())
    print(f"\ncells where the hard arm deleted nothing (hard == base): "
          f"{dead_cells} of {n_hard}")
    print("They are excluded below: an arm that did not act is not an arm that "
          "failed.")

    print("\nregistered test, on cells where deletion happened:")
    line("soft vs hard", paired(live, "soft", "hard"))
    print("\nfor context, each against the common base:")
    line("soft vs base", paired(live, "soft", "base"))
    line("hard vs base", paired(live, "hard", "base"))

    r = paired(live, "soft", "hard")
    if r:
        print("\nverdict:")
        if r["p"] >= 0.05:
            print(f"  H4 FALSIFIED as registered -- soft does not win with "
                  f"p < 0.05 (net {r['net']:+d}, p = {r['p']:.4f}).")
        elif r["net"] > 0 and r["dp_only_win"] > 0.5:
            print(f"  H4 holds (net {r['net']:+d}, p = {r['p']:.4f}).")
        elif r["net"] > 0:
            print(f"  H4 holds on the Pareto count (net {r['net']:+d}, "
                  f"p = {r['p']:.4f}) while losing the dP-only column "
                  f"({r['dp_only_win']:.3f}) -- which is the direction the "
                  f"registration wanted, not the one it warned about.")
        else:
            print(f"  H4 FALSIFIED -- soft loses the Pareto count "
                  f"(net {r['net']:+d}).")
        print("\nper setting:")
        c = r["cells"].assign(
            o=[dominates(x.test_auc, x.test_dp, x.test_auc_b, x.test_dp_b)
               for x in r["cells"].itertuples()])
        for st, g in c.groupby("setting"):
            w, l = int((g.o == 1).sum()), int((g.o == -1).sum())
            p = stats.binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
            print(f"    {st:<12s} n={len(g):3d}  win={w:3d} lose={l:3d} "
                  f"net={w - l:+4d}  p={p:.4f}")


def do_h5(path: str) -> None:
    if not os.path.exists(path):
        print(f"\n[H5] not run yet: {path}")
        return
    d = pd.read_csv(path)
    print("\n" + "=" * 100)
    print("H5 -- allocation under an accuracy constraint vs unconstrained")
    print("=" * 100)

    h = d[d.arm == "constrained"]
    print(f"\nconstraint activity: held {h.n_held.mean():.1f} of "
          f"{h.epochs_run.mean():.1f} epochs on average; "
          f"never fired in {float((h.n_held == 0).mean()):.2f} of cells, "
          f"fired on every fairness epoch in "
          f"{float((h.n_held >= h.epochs_run - 100).mean()):.2f}.")
    print("A constrained arm that never fires is the unconstrained arm; one "
          "that always fires\nhas no fairness term at all. Both are failure "
          "modes and are visible here, not inferred.")

    print("\nregistered test -- dominance over uniform, both arms:")
    ru = paired(d, "unconstrained", "uniform")
    rc = paired(d, "constrained", "uniform")
    line("unconstrained vs uniform", ru)
    line("constrained   vs uniform", rc)
    print("\nhead to head:")
    line("constrained vs unconstrained", paired(d, "constrained", "unconstrained"))

    if ru and rc:
        print("\nverdict:")
        if rc["rate"] > ru["rate"]:
            print(f"  H5 holds on the registered criterion: dominance rate "
                  f"{ru['rate']:.3f} -> {rc['rate']:.3f}.")
        else:
            print(f"  H5 FALSIFIED as registered -- the constrained arm's "
                  f"dominance rate ({rc['rate']:.3f}) is not higher than the "
                  f"unconstrained arm's ({ru['rate']:.3f}).")
        print(f"  dP-only win rate {ru['dp_only_win']:.3f} -> "
              f"{rc['dp_only_win']:.3f}  (the registration expected this to "
              f"fall)")
        print(f"  mean dAUC vs uniform {ru['d_auc']:+.4f} -> {rc['d_auc']:+.4f}"
              f"   (the constraint exists to protect this)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h4", default=os.path.join(RESULTS, "e8_h4.csv"))
    ap.add_argument("--h5", default=os.path.join(RESULTS, "e9_h5.csv"))
    a = ap.parse_args()
    do_h4(a.h4)
    do_h5(a.h5)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
