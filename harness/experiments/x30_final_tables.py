"""Emit the paper-ready tables for every COMPLETE cell, into final_results/.

Nothing new is computed: the per-row construction (`analyze_armA.build`), the
paired hierarchical bootstrap (`bootstrap_armA.boot`, seed 20260914, 10,000
replicates) and the resolved rule (sign >= 0.75, |mean| >= 0.010, 95% interval
excluding 0) are imported and used unchanged, exactly as the frozen analysis
does. Cells that are not complete (30 units / 60 rows) are excluded and listed
in `coverage.csv` as pending, never partially summarised.

Strata are kept apart and labelled in every file:

    CORE                 the frozen 12 controlled cells (X22)
    DATASET EXTENSION    the X29 cells
    METHOD EXTENSION     X30 primary cells (one backbone per method x dataset)
    VARIANTS             FairSIN GIN / SAGE, BIND-10pct (never pooled)
    NATIVE               FairSIN at its native horizon and native selector

    python harness/experiments/x30_final_tables.py --out final_results
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))

from analyze_armA import build                                   # noqa: E402
from analyze_x29 import CORE_CSVS, X29_GLOB, UNITS, ROWS         # noqa: E402
from analyze_x30 import VARIANT_METHODS, per_cell_present, _inputs  # noqa: E402

COORD = {"dAUC": "auc", "-dDP": "ndp", "-dEO": "neo"}
QUANT = {"tau_I": "int", "tau_nonint": "abase", "tau_pkg": "apkg"}
# Every admitted cell of the frozen protocol, so pending ones are visible.
ADMITTED = (
    [("METHOD EXTENSION", f"FairSIN-GCN", d, "x30") for d in
     ("german", "bail", "credit", "pokec_z", "pokec_n")]
    + [("METHOD EXTENSION", m, d, "x30") for m in ("EDITS", "FairEdit")
       for d in ("german", "bail", "credit")]
    + [("METHOD EXTENSION", "BeMap", d, "x30") for d in ("bail", "credit", "pokec_z")]
    + [("METHOD EXTENSION", "GEAR", "bail", "x30")]
    + [("METHOD EXTENSION", "BIND-1pct", d, "x30") for d in ("bail", "income")]
    + [("VARIANTS", f"FairSIN-{e}", d, "x30") for e in ("GIN", "SAGE")
       for d in ("german", "bail", "credit", "pokec_z", "pokec_n")]
    + [("VARIANTS", "BIND-10pct", d, "x30") for d in ("bail", "income")]
    + [("NATIVE", f"FairSIN-{e}", d, "x30native") for e in ("GCN", "GIN", "SAGE")
       for d in ("german", "bail", "credit", "pokec_z", "pokec_n")]
)


def complete_cells(d):
    """(method, dataset) pairs with the full 30 units and 60 rows."""
    ok = []
    for (m, ds_), g in d.groupby(["method", "dataset"]):
        if g.groupby(["split_id", "run_id"]).ngroups == UNITS and len(g) == ROWS:
            ok.append((m, ds_))
    return ok


def keep_complete(d):
    ok = set(complete_cells(d))
    return d[[(m, ds_) in ok for m, ds_ in zip(d.method, d.dataset)]].copy()


def summarise(d, stratum):
    """The frozen per-cell bootstrap, as a long table."""
    if d.empty:
        return pd.DataFrame()
    s = per_cell_present(d, io.StringIO(), stratum)
    s.insert(0, "stratum", stratum)
    return s


def wide(long_):
    """One row per (stratum, method, dataset): every estimand and coordinate."""
    rows = []
    for (st, m, ds_), g in long_.groupby(["stratum", "method", "dataset"], sort=False):
        r = dict(stratum=st, method=m, dataset=ds_, n_units=int(g.n_cells.iloc[0]))
        for q in QUANT:
            for c in COORD:
                x = g[(g.quantity == q) & (g.coord == c)]
                if x.empty:
                    continue
                x = x.iloc[0]
                k = f"{q}_{c.replace('-', 'neg_')}"
                r[f"{k}_mean"] = x["mean"]; r[f"{k}_lo"] = x.lo; r[f"{k}_hi"] = x.hi
                r[f"{k}_sign"] = x["sign"]; r[f"{k}_resolved"] = bool(x.resolved)
        if "tau_I_neg_dDP_mean" in r and "tau_nonint_neg_dDP_mean" in r:
            r["abs_tau_nonint_gt_abs_tau_I_on_negDP"] = (
                abs(r["tau_nonint_neg_dDP_mean"]) > abs(r["tau_I_neg_dDP_mean"]))
        rows.append(r)
    return pd.DataFrame(rows)


def units_table(d, stratum):
    cols = ["method", "dataset", "backbone", "protocol", "split_id", "run_id", "seed",
            "selector", "m1_epoch", "m0_epoch", "bc_epoch", "code_epoch",
            "m1_auc", "m1_dp", "m1_eo", "m0_auc", "m0_dp", "m0_eo",
            "bc_auc", "bc_dp", "bc_eo", "b_auc", "b_dp", "b_eo",
            "m1pub_auc", "m1pub_dp", "m1pub_eo",
            "int_auc", "int_ndp", "int_neo", "abase_auc", "abase_ndp", "abase_neo",
            "apkg_auc", "apkg_ndp", "apkg_neo", "n_flip", "n_test_a1", "n_test_a0",
            "eo_defined", "provenance", "constsign_plus", "constsign_minus"]
    out = d[[c for c in cols if c in d.columns]].copy()
    out.insert(0, "stratum", stratum)
    return out


def selector_table(d, stratum):
    rows = []
    for (m, ds_), g in d.groupby(["method", "dataset"], sort=False):
        b = g[g.selector == "common_bce"]; a = g[g.selector == "common_auc"]
        rows.append(dict(stratum=stratum, method=m, dataset=ds_,
                         tau_I_neg_dDP_bce=b.int_ndp.mean(), tau_I_neg_dDP_auc=a.int_ndp.mean(),
                         tau_I_dAUC_bce=b.int_auc.mean(), tau_I_dAUC_auc=a.int_auc.mean(),
                         sign_agrees=bool((b.int_ndp.mean() > 0) == (a.int_ndp.mean() > 0))))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "final_results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # same input filter as the frozen analyzer, plus the interim products
    x30 = [f for f in _inputs(f"{ROOT}/harness/results/x30/x30_*.csv") if "_interim_" not in f]
    x30n = [f for f in _inputs(f"{ROOT}/harness/results/x30/x30native_*.csv")
            if "_interim_" not in f]

    core = keep_complete(build(CORE_CSVS))
    dext = keep_complete(build(_inputs(X29_GLOB)))
    allm = build(x30)
    mext = keep_complete(allm[~allm.method.isin(VARIANT_METHODS)])
    var = keep_complete(allm[allm.method.isin(VARIANT_METHODS)])
    nat = keep_complete(build(x30n)) if x30n else pd.DataFrame()

    views = [("CORE", core), ("DATASET EXTENSION", dext), ("METHOD EXTENSION", mext),
             ("VARIANTS", var), ("NATIVE", nat)]
    long_ = pd.concat([summarise(d, st) for st, d in views if not d.empty], ignore_index=True)
    w = wide(long_)
    units = pd.concat([units_table(d, st) for st, d in views if not d.empty], ignore_index=True)
    sel = pd.concat([selector_table(d, st) for st, d in views if not d.empty], ignore_index=True)

    # coverage: every admitted cell, complete or pending
    done = {(st, m, ds_) for st, m, ds_ in
            [(r.stratum, r.method, r.dataset) for r in w.itertuples()]}
    cov = []
    for st, m, ds_, proto in ADMITTED:
        f = (f"{ROOT}/harness/results/x30/{proto}_{m}_{ds_}.csv" if ds_ != "income"
             else f"{ROOT}/harness/results/x30/{proto}_{m}_{ds_}_s20.csv")
        n = 0
        if ds_ == "income":
            for sp in range(20, 26):
                p = f"{ROOT}/harness/results/x30/{proto}_{m}_{ds_}_s{sp}.csv"
                if os.path.exists(p):
                    n += max(0, sum(1 for _ in open(p)) - 1) // 2
        elif os.path.exists(f):
            n = max(0, sum(1 for _ in open(f)) - 1) // 2
        cov.append(dict(stratum=st, method=m, dataset=ds_, protocol=proto,
                        units_done=n, units_required=UNITS,
                        status="complete" if (st, m, ds_) in done else
                               ("pending" if n == 0 else "partial")))
    cov = pd.DataFrame(cov)

    # controlled vs native, for the cells where both exist (FairSIN)
    cn = []
    c_ = w[w.stratum.isin(["METHOD EXTENSION", "VARIANTS"])]
    n_ = w[w.stratum == "NATIVE"]
    for r in n_.itertuples():
        m = c_[(c_.method == r.method) & (c_.dataset == r.dataset)]
        if m.empty:
            continue
        m = m.iloc[0]
        cn.append(dict(method=r.method, dataset=r.dataset,
                       controlled_tau_I_negDP=m.tau_I_neg_dDP_mean,
                       controlled_lo=m.tau_I_neg_dDP_lo, controlled_hi=m.tau_I_neg_dDP_hi,
                       controlled_resolved=m.tau_I_neg_dDP_resolved,
                       native_tau_I_negDP=r.tau_I_neg_dDP_mean,
                       native_lo=r.tau_I_neg_dDP_lo, native_hi=r.tau_I_neg_dDP_hi,
                       native_resolved=r.tau_I_neg_dDP_resolved,
                       same_sign=bool((m.tau_I_neg_dDP_mean > 0) == (r.tau_I_neg_dDP_mean > 0))))
    cn = pd.DataFrame(cn)

    # per-stratum counts, the numbers a paper table reports
    st_rows = []
    for st in w.stratum.unique():
        g = w[w.stratum == st]
        st_rows.append(dict(stratum=st, cells=len(g),
                            resolved_tau_I_negDP=int(g.tau_I_neg_dDP_resolved.sum()),
                            resolved_tau_I_dAUC=int(g.tau_I_dAUC_resolved.sum()),
                            resolved_tau_I_negEO=int(g.tau_I_neg_dEO_resolved.sum()),
                            abs_nonint_gt_abs_I=int(g.abs_tau_nonint_gt_abs_tau_I_on_negDP.sum()),
                            tau_I_negDP_mean=g.tau_I_neg_dDP_mean.mean(),
                            tau_I_negDP_min=g.tau_I_neg_dDP_mean.min(),
                            tau_I_negDP_max=g.tau_I_neg_dDP_mean.max(),
                            tau_nonint_negDP_mean=g.tau_nonint_neg_dDP_mean.mean()))
    st_tab = pd.DataFrame(st_rows)

    for name, df in (("cell_estimates_long.csv", long_), ("cell_estimates_wide.csv", w),
                     ("unit_level_rows.csv", units), ("selector_agreement.csv", sel),
                     ("stratum_summary.csv", st_tab), ("coverage.csv", cov),
                     ("controlled_vs_native.csv", cn)):
        df.to_csv(os.path.join(a.out, name), index=False)
        print(f"[written] {os.path.join(a.out, name)}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
