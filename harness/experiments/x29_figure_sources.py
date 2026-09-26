"""X29 figure source CSVs, compatible with the X28 paper-figure pipeline.

Emits only source data. It draws nothing and computes no new statistic: every
number is read from the analyzer's output (x29_summary.csv, x29_cell_table.csv),
which in turn uses the frozen bootstrap and the frozen resolved rule.

Writes under harness/results/x29/figure_sources/ only. No X22-X27 figure, table or
source file is read for writing or overwritten.

Protocol section 12 lists six figures. Five are emitted here. The sixth --
observed protocol span, native-available cells only -- has no extension content
at all, because native_config() succeeds for exactly the seven pairs already
frozen in X23 and the extension adds none. Emitting an empty file would imply a
measurement that does not exist, so it is not emitted; the existing
harness/results/figures/paper/source_data/figG_protocol_span_source.csv already
covers the frozen cells.

    python harness/experiments/x29_figure_sources.py
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
X29 = os.path.join(ROOT, "harness", "results", "x29")


def wide(summary, quantity):
    """One row per (view, method, dataset) with both coordinates side by side."""
    q = summary[summary.quantity == quantity]
    a = q[q.coord == "dAUC"].set_index(["view", "method", "dataset"])
    d = q[q.coord == "-dDP"].set_index(["view", "method", "dataset"])
    out = pd.DataFrame({
        "dAUC": a["mean"], "dAUC_lo": a["lo"], "dAUC_hi": a["hi"],
        "dAUC_sign": a["sign"], "dAUC_resolved": a["resolved"],
        "ndDP": d["mean"], "ndDP_lo": d["lo"], "ndDP_hi": d["hi"],
        "ndDP_sign": d["sign"], "ndDP_resolved": d["resolved"],
        "n_units": a["n_cells"]})
    return out.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=X29)
    ap.add_argument("--outdir", default=os.path.join(X29, "figure_sources"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    s = pd.read_csv(os.path.join(a.indir, "x29_summary.csv"))
    cells = pd.read_csv(os.path.join(a.indir, "x29_cell_table.csv"))
    outs = []

    def emit(name, df, note):
        p = os.path.join(a.outdir, name)
        df.to_csv(p, index=False)
        outs.append((p, len(df), note))

    # 1. utility-fairness intervention scatter: x = tau_I(dAUC), y = tau_I(-dDP)
    emit("x29_fig1_intervention_scatter_source.csv", wide(s, "tau_I"),
         "x = tau_I dAUC, y = tau_I -dDP, one point per method x dataset cell")

    # 2. package vs intervention magnitude: tau_nonint against tau_I
    ni = wide(s, "tau_nonint").add_prefix("nonint_")
    iv = wide(s, "tau_I").add_prefix("int_")
    both = pd.concat([ni.rename(columns={"nonint_view": "view",
                                         "nonint_method": "method",
                                         "nonint_dataset": "dataset"}),
                      iv.drop(columns=["int_view", "int_method", "int_dataset"])],
                     axis=1)
    both["nonint_larger_on_ndDP"] = both.nonint_ndDP.abs() > both.int_ndDP.abs()
    emit("x29_fig2_package_vs_intervention_source.csv", both,
         "tau_nonint against tau_I, both coordinates, with the magnitude relation")

    # 3. method-wise forest of tau_I(-dDP)
    f = s[(s.quantity == "tau_I") & (s.coord == "-dDP")].copy()
    f = f.sort_values(["view", "method", "dataset"])
    emit("x29_fig3_method_forest_source.csv",
         f[["view", "method", "dataset", "mean", "lo", "hi", "sign",
            "resolved", "n_cells"]],
         "one interval per cell, grouped by method; -dDP under sigma_c^BCE")

    # 4. dataset-wise distribution of tau_I(-dDP)
    emit("x29_fig4_dataset_distribution_source.csv",
         f.sort_values(["view", "dataset", "method"])[
             ["view", "dataset", "method", "mean", "lo", "hi", "resolved"]],
         "same intervals, grouped by dataset")

    # 5. selector agreement: tau_I(-dDP) under BCE against AUC, per cell
    rows = []
    for (view, m, ds_), g in cells.groupby(["view", "method", "dataset"],
                                           sort=True):
        b = g[g.selector == "common_bce"].int_ndp.mean()
        c = g[g.selector == "common_auc"].int_ndp.mean()
        rows.append(dict(view=view, method=m, dataset=ds_, bce=b, auc=c,
                         gap=b - c, sign_agrees=bool((b > 0) == (c > 0)),
                         n_units=g.groupby(["split_id", "run_id"]).ngroups))
    emit("x29_fig5_selector_agreement_source.csv", pd.DataFrame(rows),
         "x = tau_I -dDP under sigma_c^BCE, y = under sigma_c^AUC; y=x is agreement")

    for p, n, note in outs:
        print(f"[written] {os.path.relpath(p, ROOT)}  ({n} rows)  -- {note}")
    print("\n[not emitted] figure 6, observed protocol span: the extension adds "
          "no native cell,\n              so there is nothing to draw. "
          "harness/results/figures/paper/source_data/\n              "
          "figG_protocol_span_source.csv already covers the frozen cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
