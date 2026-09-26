"""X30 method-coverage extension analysis, in separate strata.

No new statistic. Everything is imported and used unchanged:

    analyze_armA.build                      derived per-row quantities
    analyze_armA.SIGN_MIN / NEAR_ZERO       the frozen resolved thresholds
    bootstrap_armA.boot / SEED              the frozen paired hierarchical bootstrap
    analyze_x29.per_cell / contracts / completeness / describe /
               selector_agreement           the X29 views, as committed

Strata, never silently merged (X30 protocol section 10):

    CORE                        the frozen 12 controlled cells (X22)
    DATASET EXTENSION           the 4 X29 cells
    METHOD EXTENSION            the X30 primary cells (one backbone per method x dataset)
    METHOD EXTENSION: VARIANTS  FairSIN GIN / SAGE backbones, reported apart
    METHOD EXTENSION: NATIVE    FairSIN at native horizon and native selector
    COMPONENT CASE STUDIES      X24-X27, referenced only; not re-analysed here
    COMBINED DESCRIPTIVE        CORE + DATASET EXTENSION + METHOD EXTENSION

Refuses an incomplete store (every admitted cell must have 30 units, 60 rows),
unless --allow-partial, which is for smoke tests and is never reported.

    python harness/experiments/analyze_x30.py --out harness/results/x30
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))

from analyze_armA import build                                     # noqa: E402
from analyze_x29 import (CORE_CSVS, X29_GLOB, UNITS, ROWS, per_cell,  # noqa: E402
                         contracts, completeness, describe, selector_agreement)

X30_GLOB = f"{ROOT}/harness/results/x30/x30_*.csv"
X30N_GLOB = f"{ROOT}/harness/results/x30/x30native_*.csv"
VARIANT_METHODS = ("FairSIN-GIN", "FairSIN-SAGE", "BIND-10pct")   # protocol 3.2


def _inputs(pattern):
    return [f for f in sorted(glob.glob(pattern))
            if not f.endswith(("_summary.csv", "_cell_table.csv", "_native_view.csv"))]


def per_cell_present(d, fh, tag):
    """`analyze_x29.per_cell` crosses every method with every dataset, which is
    empty for a pair that does not exist (X29 had one method, so this never
    arose). The frozen helper is called once per pair that actually has rows --
    same construction, same bootstrap, same order (dataset, then method).

    One recorded consequence: `per_cell` seeds `default_rng(SEED)` on entry, so
    each cell now draws from a generator seeded with the frozen SEED rather than
    from one stream shared by every cell in the view. Draws stay deterministic
    and reproducible, and a cell's interval no longer depends on how many cells
    precede it (the Monte-Carlo shift X23 recorded). The estimator, the 10,000
    replicates and the resolved rule are unchanged."""
    out = []
    for ds_ in sorted(d.dataset.unique()):
        for m in sorted(d[d.dataset == ds_].method.unique()):
            g = d[(d.method == m) & (d.dataset == ds_)]
            if g.empty:
                continue
            out.append(per_cell(g, fh, tag))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def show(summary, rows, title, fh):
    """describe() a stratum, or say plainly that it has no cell in this view
    (an interim view can exclude a whole stratum: X29's cells are pokec only)."""
    if summary is None or summary.empty:
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", file=fh)
        print("  no cell in this view", file=fh)
        return False
    describe(summary, rows, title, fh)
    return True


def native_view(d, fh):
    """Native package effect as the code reports it: M+I at its own selector
    against B at its own selector (pub_*), plus the controlled estimands at the
    native horizon. Descriptive; bootstrapped with the frozen procedure."""
    print("\n  native package view (M+I at code-native selector vs B at its own "
          "selector), -dDP and dAUC means over units:", file=fh)
    g = d[d.selector == "common_bce"]
    for (m, ds_), gg in g.groupby(["method", "dataset"], sort=True):
        nsel = int(gg.m1pub_auc.isna().sum())
        print(f"    {m:<13} {ds_:<8} pub_auc {gg.pub_auc.mean():+.4f}  pub_ndp "
              f"{gg.pub_ndp.mean():+.4f}  native selector chose nothing in "
              f"{nsel}/{len(gg)} units", file=fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "harness", "results", "x30"))
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="INTERIM view restricted to these datasets (all strata); "
                         "completeness is still enforced for every cell in the view")
    ap.add_argument("--allow-partial", action="store_true",
                    help="smoke-test only; never use for a reported result")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    x30_files, x30n_files = _inputs(X30_GLOB), _inputs(X30N_GLOB)
    if not x30_files:
        raise SystemExit("no X30 controlled result CSV found; nothing to analyse")
    suffix = ("_interim_" + "_".join(a.datasets)) if a.datasets else ""
    txt = os.path.join(a.out, f"x30_analysis{suffix}.txt")
    with open(txt, "w") as fh:
        print("X30 method-coverage extension -- strata kept apart", file=fh)
        print("The frozen resolved rule, bootstrap and cell construction are "
              "imported unchanged.\n", file=fh)
        print(f"controlled inputs: {[os.path.basename(f) for f in x30_files]}", file=fh)
        print(f"native inputs:     {[os.path.basename(f) for f in x30n_files]}", file=fh)

        core = build(CORE_CSVS)
        dext = build(_inputs(X29_GLOB))
        allm = build(x30_files)
        mext = allm[~allm.method.isin(VARIANT_METHODS)].copy()
        var = allm[allm.method.isin(VARIANT_METHODS)].copy()
        nat = build(x30n_files) if x30n_files else None
        if a.datasets:
            print(f"INTERIM VIEW restricted to datasets {a.datasets}; the remaining X30 "
                  "cells are still running and are not part of this view.", file=fh)
            keep = lambda d: d[d.dataset.isin(a.datasets)].copy()      # noqa: E731
            core, dext, allm = keep(core), keep(dext), keep(allm)
            mext, var = keep(mext), keep(var)
            nat = keep(nat) if nat is not None else None

        print("\ncontracts:", file=fh)
        bad = (contracts(core, "CORE", fh) + contracts(dext, "DATASET EXTENSION", fh)
               + contracts(mext, "METHOD EXTENSION", fh))
        if len(var):
            bad += contracts(var, "VARIANTS", fh)
        if nat is not None:
            bad += contracts(nat, "NATIVE", fh)
        print("", file=fh)
        short = (completeness(core, "CORE", fh) + completeness(dext, "DATASET EXTENSION", fh)
                 + completeness(allm, "METHOD EXTENSION (+variants)", fh)
                 + (completeness(nat, "NATIVE", fh) if nat is not None else []))
        if short and not a.allow_partial:
            msg = "; ".join(f"{m}/{ds} has {u}/{UNITS} units, {r}/{ROWS} rows"
                            for m, ds, u, r in short)
            print(f"\nREFUSED: incomplete cells -- {msg}", file=fh)
            fh.flush()
            raise SystemExit(f"[x30] refusing to analyse an incomplete store: {msg}")
        if short:
            print("\n  WARNING: --allow-partial; INCOMPLETE cells, must not be reported:",
                  file=fh)
            for m, ds_, u, r in short:
                print(f"    {m}/{ds_}: {u}/{UNITS} units", file=fh)

        s_core = per_cell_present(core, fh, "CORE")
        s_dext = per_cell_present(dext, fh, "DATASET EXTENSION")
        s_mext = per_cell_present(mext, fh, "METHOD EXTENSION")
        parts = [x for x in (s_core, s_dext, s_mext) if x is not None and not x.empty]
        show(s_core, core, "CORE -- the frozen 12 controlled cells (X22)", fh)
        show(s_dext, dext, "DATASET EXTENSION -- the X29 cells", fh)
        show(s_mext, mext, "METHOD EXTENSION -- the X30 primary cells", fh)
        selector_agreement(mext, fh, "METHOD EXTENSION")
        if len(var):
            s_var = per_cell_present(var, fh, "METHOD EXTENSION: VARIANTS")
            if not s_var.empty:
                parts.append(s_var)
            show(s_var, var, "METHOD EXTENSION: VARIANTS -- FairSIN GIN / SAGE, BIND-10pct "
                 "(not pooled into any count)", fh)
            selector_agreement(var, fh, "VARIANTS")
        if nat is not None:
            s_nat = per_cell_present(nat, fh, "METHOD EXTENSION: NATIVE")
            if not s_nat.empty:
                parts.append(s_nat)
            show(s_nat, nat, "METHOD EXTENSION: NATIVE -- FairSIN at native horizon "
                 "(not pooled into any count)", fh)
            native_view(nat, fh)

        print(f"\n{'=' * 78}\nCOMPONENT CASE STUDIES\n{'=' * 78}", file=fh)
        print("  X24 NIFTY/german factorial, X25 NIFTY trajectory selection, X26 FairGB "
              "selection support, X27 FMP mechanistic.\n  Reported in their own "
              "RESULTS files; not re-analysed and not pooled here.", file=fh)

        comb_rows = pd.concat([core, dext, mext], ignore_index=True)
        comb = pd.concat([x for x in (s_core, s_dext, s_mext) if x is not None and not x.empty],
                         ignore_index=True)
        if not comb.empty:
            comb["view"] = "COMBINED DESCRIPTIVE"
        show(comb, comb_rows, "COMBINED DESCRIPTIVE -- core + dataset extension + "
             "method extension (descriptive only)", fh)
        print(f"\ncontract violations total: {bad}", file=fh)

    pd.concat(parts, ignore_index=True).to_csv(os.path.join(a.out, f"x30_summary{suffix}.csv"),
                                               index=False)
    views = [core.assign(view="CORE"), dext.assign(view="DATASET EXTENSION"),
             mext.assign(view="METHOD EXTENSION")]
    if len(var):
        views.append(var.assign(view="VARIANTS"))
    if nat is not None:
        views.append(nat.assign(view="NATIVE"))
    pd.concat(views, ignore_index=True).to_csv(os.path.join(a.out, f"x30_cell_table{suffix}.csv"),
                                               index=False)
    print(open(txt).read())
    return 0 if bad == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
