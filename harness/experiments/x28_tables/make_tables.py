"""X28 publication tables: build every table from the frozen artifacts.

No training is run and no frozen value is altered. Each table writes
  harness/results/tables/paper/<name>.tex
  harness/results/tables/paper/source_data/<name>_source.csv
  harness/results/tables/paper/<name>_preview.png
and every table is checked before it is written; a scientific mismatch stops
the run rather than overwriting anything.

    python harness/experiments/x28_tables/make_tables.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tcommon as T                                                  # noqa: E402

ORIENT = ("Fairness coordinates are oriented so that positive values indicate "
          "improved fairness.")
PAIRED = ("Effects are paired intervention contrasts under the common "
          "controlled protocol.")
RESRULE = (r"\textbf{R} marks an effect that is \emph{resolved} under the "
           r"pre-registered rule (sign stability $\geq 0.75$, $|$mean$| \geq "
           r"0.010$, and a 95\% interval excluding zero); \emph{u} marks "
           r"unresolved. Bold marks this pre-registered state only, never a "
           r"preferred outcome.")


# ============================================================ controlled inputs
def controlled_cells(ck):
    """The 12 controlled cells, assembled from two frozen artifacts."""
    boot = T.FIG.parse_armA_bootstrap()
    per = T.parse_armA_per_dataset()
    seven = T.FIG.parse_armB_seven()

    rows = []
    for ds in T.DATASETS:
        for m in T.METHODS:
            p = per[(per.dataset == ds) & (per.method == m)].iloc[0]
            r = {}
            for q, tag in (("tau_base^audit", "nonint"), ("tau_int", "int"),
                           ("tau_pkg^audit", "pkg"), ("D_selector", "dsel")):
                for coord, ckey in (("-dDP", "ndp"), ("dAUC", "auc")):
                    b = boot[(boot.dataset == ds) & (boot.method == m)
                             & (boot.quantity == q) & (boot.coord == coord)].iloc[0]
                    r[f"{tag}_{ckey}"] = b["mean"]
                    r[f"{tag}_{ckey}_lo"] = b["lo"]
                    r[f"{tag}_{ckey}_hi"] = b["hi"]
                    ck.interval(f"{m}/{ds} {q} {coord}", b["mean"], b["lo"], b["hi"])
            # the two artifacts must agree on the means they both carry
            ck.close(f"{m}/{ds} tau_int -dDP: section [4] vs bootstrap",
                     p["int_ndp"], r["int_ndp"], 2e-4)
            ck.close(f"{m}/{ds} tau_base -dDP: section [4] vs bootstrap",
                     p["base_ndp"], r["nonint_ndp"], 2e-4)
            # the frozen identity tau_pkg = tau_nonint + tau_int
            for ckey in ("ndp", "auc"):
                ck.close(f"{m}/{ds} tau_pkg = tau_nonint + tau_int ({ckey})",
                         r["nonint_" + ckey] + r["int_" + ckey], r["pkg_" + ckey],
                         3e-4)
            r.update(method=m, dataset=ds,
                     sign_int_ndp=p["sign_int_ndp"],
                     base_gt_int=bool(p["base_gt_int"]))
            r["int_ndp_resolved"] = T.resolved(r["int_ndp"], r["int_ndp_lo"],
                                               r["int_ndp_hi"], p["sign_int_ndp"])
            # -dEO exists for the seven native-validation cells only
            e = seven[(seven.method == m) & (seven.dataset == ds)
                      & (seven.coord == "-dEO") & (seven.quantity == "tau_int")]
            if len(e):
                e = e.iloc[0]
                r.update(int_neo=e["a_mean"], int_neo_lo=e["a_lo"],
                         int_neo_hi=e["a_hi"], int_neo_sign=e["a_sign"],
                         int_neo_resolved=bool(e["a_resolved"]))
            else:
                r.update(int_neo=None, int_neo_lo=None, int_neo_hi=None,
                         int_neo_sign=None, int_neo_resolved=None)
            rows.append(r)
    d = pd.DataFrame(rows)
    ck.unique("controlled cells", d, ["method", "dataset"])
    ck.true("controlled: 12 cells", len(d) == 12)

    # validate the resolved computation against the seven flags armB states
    n = 0
    for _, s in seven[(seven.coord == "-dDP") & (seven.quantity == "tau_int")].iterrows():
        g = d[(d.method == s["method"]) & (d.dataset == s["dataset"])].iloc[0]
        ck.true(f"{s['method']}/{s['dataset']}: Arm A resolved reproduces the "
                f"frozen flag ({s['a_resolved']})",
                bool(g["int_ndp_resolved"]) == bool(s["a_resolved"]))
        ck.close(f"{s['method']}/{s['dataset']}: Arm A sign stability agrees",
                 g["sign_int_ndp"], s["a_sign"], 5e-3)
        n += 1
    ck.true("all seven frozen Arm A resolved flags were checked", n == 7)
    return d


# ================================================================== Table 1 / A1
def table1(d, ck, outs):
    body, csv = [], []
    for _, r in d.iterrows():
        rr = dict(method=r.method, dataset=T.DS_LABEL[r.dataset],
                  tau_nonint=r.nonint_ndp, tau_nonint_lo=r.nonint_ndp_lo,
                  tau_nonint_hi=r.nonint_ndp_hi, tau_I=r.int_ndp,
                  tau_I_lo=r.int_ndp_lo, tau_I_hi=r.int_ndp_hi,
                  tau_I_sign=r.sign_int_ndp, tau_I_resolved=r.int_ndp_resolved,
                  nonint_larger=r.base_gt_int, tau_I_dAUC=r.int_auc)
        csv.append(rr)
        body.append([T.esc(r.method), T.DS_LABEL[r.dataset],
                     T.num(r.nonint_ndp), T.ci(r.nonint_ndp_lo, r.nonint_ndp_hi),
                     T.num(r.int_ndp), T.ci(r.int_ndp_lo, r.int_ndp_hi),
                     T.res(r.int_ndp_resolved),
                     "yes" if r.base_gt_int else "no"])
    df = pd.DataFrame(csv)
    ck.complete("Table 1", df, 12)
    n_larger = int(df.nonint_larger.sum())
    ck.true("Table 1: |tau_nonint| > |tau_I| count matches Arm A J1 (10 of 12)",
            n_larger == 10)
    tex = T.latex_table(
        "llrrrrcc",
        [[r"Method", r"Dataset", r"$\tau_{\mathrm{nonint}}$", r"95\% CI",
          r"$\tau_{I}$", r"95\% CI", r"res.", r"$|\tau_{\mathrm{nonint}}|>|\tau_I|$"]],
        body,
        caption=("Controlled attribution summary on $-\\Delta$DP under the "
                 "$\\sigma_c^{\\mathrm{BCE}}$ selector. " + PAIRED + " " + ORIENT
                 + f" The non-intervention package is larger in magnitude than the "
                 f"intervention in {n_larger} of the 12 cells; this is a count of "
                 f"these cells, not a rate in any population."),
        label="tab:controlled-attribution",
        notes=[RESRULE,
               r"$\tau_{\mathrm{nonint}} = Y(M_{\mathrm{off}})-Y(B)$ and "
               r"$\tau_{I} = Y(M_{\mathrm{on}})-Y(M_{\mathrm{off}})$; intervals "
               r"are 95\% paired hierarchical bootstrap, 6 splits $\times$ 5 runs "
               r"per cell."],
        width="single")
    ck.latex_matches_csv("Table 1", tex, pd.DataFrame(
        dict(a=[T.num(v) for v in df.tau_I],
             b=[T.num(v) for v in df.tau_nonint])), ["a", "b"])
    T.write("table1_controlled_attribution", df, tex,
            "Table 1 -- controlled attribution summary", outs)
    return df


def table1_compact_auc(d, ck, outs):
    """Table 1 variant carrying the AUC intervention effect in one column."""
    body, csv = [], []
    for _, r in d.iterrows():
        csv.append(dict(method=r.method, dataset=T.DS_LABEL[r.dataset],
                        tau_nonint=r.nonint_ndp, tau_I=r.int_ndp,
                        tau_I_resolved=r.int_ndp_resolved,
                        tau_I_dAUC=r.int_auc, tau_I_dAUC_lo=r.int_auc_lo,
                        tau_I_dAUC_hi=r.int_auc_hi,
                        nonint_larger=r.base_gt_int))
        body.append([T.esc(r.method), T.DS_LABEL[r.dataset],
                     T.num(r.nonint_ndp), T.num(r.int_ndp),
                     T.res(r.int_ndp_resolved), T.num(r.int_auc),
                     "yes" if r.base_gt_int else "no"])
    df = pd.DataFrame(csv)
    ck.complete("Table 1 (compact+AUC)", df, 12)
    tex = T.latex_table(
        "llrrcrc",
        [[r"Method", r"Dataset", r"$\tau_{\mathrm{nonint}}$", r"$\tau_{I}$",
          r"res.", r"$\tau_{I}$ ($\Delta$AUC)",
          r"$|\tau_{\mathrm{nonint}}|>|\tau_I|$"]],
        body,
        caption=("Controlled attribution summary, compact form with the utility "
                 "coordinate. " + PAIRED + " " + ORIENT
                 + " Fairness columns are $-\\Delta$DP; intervals are omitted here "
                 "and given in Table~\\ref{tab:controlled-attribution}."),
        label="tab:controlled-attribution-auc",
        notes=[RESRULE], width="single")
    T.write("table1b_controlled_attribution_with_auc", df, tex,
            "Table 1b -- controlled attribution, with AUC", outs)
    return df


def tableA1(d, ck, outs):
    body, csv = [], []
    for _, r in d.iterrows():
        eo_known = r.int_neo is not None and not pd.isna(r.int_neo)
        csv.append(dict(
            method=r.method, dataset=T.DS_LABEL[r.dataset],
            tau_nonint_ndp=r.nonint_ndp, tau_nonint_ndp_lo=r.nonint_ndp_lo,
            tau_nonint_ndp_hi=r.nonint_ndp_hi,
            tau_I_ndp=r.int_ndp, tau_I_ndp_lo=r.int_ndp_lo,
            tau_I_ndp_hi=r.int_ndp_hi, tau_I_ndp_sign=r.sign_int_ndp,
            tau_I_ndp_resolved=r.int_ndp_resolved,
            tau_pkg_ndp=r.pkg_ndp, tau_pkg_ndp_lo=r.pkg_ndp_lo,
            tau_pkg_ndp_hi=r.pkg_ndp_hi,
            tau_nonint_auc=r.nonint_auc, tau_I_auc=r.int_auc,
            tau_I_auc_lo=r.int_auc_lo, tau_I_auc_hi=r.int_auc_hi,
            tau_pkg_auc=r.pkg_auc,
            tau_I_neo=r.int_neo if eo_known else "",
            tau_I_neo_lo=r.int_neo_lo if eo_known else "",
            tau_I_neo_hi=r.int_neo_hi if eo_known else "",
            tau_I_neo_sign=r.int_neo_sign if eo_known else "",
            tau_I_neo_resolved=r.int_neo_resolved if eo_known else "",
            D_selector_ndp=r.dsel_ndp, D_selector_ndp_lo=r.dsel_ndp_lo,
            D_selector_ndp_hi=r.dsel_ndp_hi,
            D_selector_auc=r.dsel_auc))
        body.append([
            T.esc(r.method), T.DS_LABEL[r.dataset],
            T.num(r.nonint_ndp, 4), T.num(r.int_ndp, 4),
            T.ci(r.int_ndp_lo, r.int_ndp_hi, 4), f"{r.sign_int_ndp:.2f}",
            T.res(r.int_ndp_resolved), T.num(r.pkg_ndp, 4),
            T.num(r.int_neo, 4) if eo_known else "--",
            (T.res(r.int_neo_resolved) if eo_known else "--"),
            T.num(r.int_auc, 4), T.ci(r.int_auc_lo, r.int_auc_hi, 4),
            T.num(r.dsel_ndp, 4), T.ci(r.dsel_ndp_lo, r.dsel_ndp_hi, 4)])
    df = pd.DataFrame(csv)
    ck.true("Table A1: 12 rows", len(df) == 12)
    n_eo = int(sum(1 for _, r in d.iterrows()
                   if r.int_neo is not None and not pd.isna(r.int_neo)))
    ck.true("Table A1: -dEO present for exactly the 7 native-validation cells",
            n_eo == 7)
    tex = T.latex_table(
        # 2 + 6 + 2 + 2 + 2 = 14, matching the \multicolumn groups and the
        # 14 cells each body row emits
        "llrrrccrrcrrrr",
        [[r"\multicolumn{2}{c}{Cell}", r"\multicolumn{6}{c}{$-\Delta$DP}",
          r"\multicolumn{2}{c}{$-\Delta$EO}", r"\multicolumn{2}{c}{$\Delta$AUC}",
          r"\multicolumn{2}{c}{Selector $D$}"],
         [r"Method", r"Dataset", r"$\tau_{\mathrm{nonint}}$", r"$\tau_I$",
          r"95\% CI", r"$s$", r"res.", r"$\tau_{\mathrm{pkg}}$",
          r"$\tau_I$", r"res.", r"$\tau_I$", r"95\% CI", r"mean", r"95\% CI"]],
        body,
        caption=("Controlled 12-cell full results. " + PAIRED + " " + ORIENT
                 + " $s$ is sign stability across the 30 cells. "
                 r"$D = \tau_I^{\mathrm{BCE}} - \tau_I^{\mathrm{AUC}}$ is the "
                 r"frozen selector-sensitivity quantity."),
        label="tab:controlled-full",
        notes=[RESRULE,
               r"$-\Delta$EO is shown only for the seven cells where a frozen "
               r"bootstrap of that coordinate exists (the native-validation "
               r"cells); no frozen Arm A artifact bootstraps $-\Delta$EO for the "
               r"remaining five, and none was computed here.",
               r"The $\sigma_c^{\mathrm{AUC}}$ view is given through the frozen "
               r"paired difference $D$ rather than as a separate column, because "
               r"no frozen artifact carries an interval for the "
               r"$\sigma_c^{\mathrm{AUC}}$ effect itself."],
        width="double")
    T.write("tableA1_controlled_full", df, tex,
            "Table A1 -- controlled 12-cell full results", outs)
    return df


# ================================================================== Table 2 / A2
def transfer_rows(ck):
    seven = T.FIG.parse_armB_seven()
    out = []
    for (m, ds), g in seven.groupby(["method", "dataset"], sort=True):
        row = dict(method=m, dataset=ds)
        for coord, key in (("-dDP", "ndp"), ("-dEO", "neo"), ("dAUC", "auc")):
            t = g[(g.coord == coord) & (g.quantity == "tau_int")].iloc[0]
            b = g[(g.coord == coord) & (g.quantity == "tau_base")].iloc[0]
            row.update({
                f"ctrl_{key}": t["a_mean"], f"ctrl_{key}_lo": t["a_lo"],
                f"ctrl_{key}_hi": t["a_hi"], f"ctrl_{key}_sign": t["a_sign"],
                f"ctrl_{key}_resolved": bool(t["a_resolved"]),
                f"nat_{key}": t["n_mean"], f"nat_{key}_lo": t["n_lo"],
                f"nat_{key}_hi": t["n_hi"], f"nat_{key}_sign": t["n_sign"],
                f"nat_{key}_resolved": bool(t["n_resolved"]),
                f"ctrl_base_{key}": b["a_mean"], f"nat_base_{key}": b["n_mean"],
                f"class_{key}": t.get("transfer_class", "")})
            ck.interval(f"{m}/{ds} controlled {coord}", t["a_mean"], t["a_lo"], t["a_hi"])
            ck.interval(f"{m}/{ds} native {coord}", t["n_mean"], t["n_lo"], t["n_hi"])
            ck.klass(f"{m}/{ds} {coord} transfer class", t.get("transfer_class", ""))
            row[f"ctrl_larger_{key}"] = abs(b["a_mean"]) > abs(t["a_mean"])
            row[f"nat_larger_{key}"] = abs(b["n_mean"]) > abs(t["n_mean"])
        out.append(row)
    d = pd.DataFrame(out)
    ck.unique("transfer cells", d, ["method", "dataset"])
    ck.true("transfer: 7 cells", len(d) == 7)
    return d


def table2(d, ck, outs):
    body, csv = [], []
    for _, r in d.iterrows():
        csv.append(dict(method=r.method, dataset=T.DS_LABEL[r.dataset],
                        tau_I_controlled=r.ctrl_ndp, tau_I_native=r.nat_ndp,
                        transfer_class=r.class_ndp))
        body.append([T.esc(r.method), T.DS_LABEL[r.dataset],
                     T.num(r.ctrl_ndp), T.num(r.nat_ndp), r.class_ndp])
    df = pd.DataFrame(csv)
    ck.complete("Table 2", df, 7)
    tex = T.latex_table(
        "llrrl",
        [[r"Method", r"Dataset", r"$\tau_I$ controlled", r"$\tau_I$ native",
          r"Transfer class"]],
        body,
        caption=("Controlled-to-native transfer on $-\\Delta$DP under the "
                 "$\\sigma_c^{\\mathrm{BCE}}$ selector. " + PAIRED + " " + ORIENT
                 + " Transfer classes are the frozen X22/X23 classification and "
                 "are reproduced here unchanged."),
        label="tab:transfer",
        notes=[r"Intervals, sign stability and the magnitude relation are given "
               r"in Table~\ref{tab:transfer-full}."], width="single")
    ck.latex_matches_csv("Table 2", tex, pd.DataFrame(
        dict(a=[T.num(v) for v in df.tau_I_controlled],
             b=[T.num(v) for v in df.tau_I_native])), ["a", "b"])
    T.write("table2_transfer", df, tex,
            "Table 2 -- controlled to native transfer", outs)
    return df


def tableA2(d, ck, outs):
    body, csv = [], []
    for _, r in d.iterrows():
        for coord, key, lab in (("-dDP", "ndp", r"$-\Delta$DP"),
                                ("-dEO", "neo", r"$-\Delta$EO"),
                                ("dAUC", "auc", r"$\Delta$AUC")):
            csv.append(dict(
                method=r.method, dataset=T.DS_LABEL[r.dataset], coordinate=coord,
                tau_I_controlled=r[f"ctrl_{key}"],
                controlled_lo=r[f"ctrl_{key}_lo"], controlled_hi=r[f"ctrl_{key}_hi"],
                controlled_sign=r[f"ctrl_{key}_sign"],
                controlled_resolved=r[f"ctrl_{key}_resolved"],
                tau_I_native=r[f"nat_{key}"],
                native_lo=r[f"nat_{key}_lo"], native_hi=r[f"nat_{key}_hi"],
                native_sign=r[f"nat_{key}_sign"],
                native_resolved=r[f"nat_{key}_resolved"],
                nonint_larger_controlled=r[f"ctrl_larger_{key}"],
                nonint_larger_native=r[f"nat_larger_{key}"],
                transfer_class=r[f"class_{key}"]))
            body.append([
                T.esc(r.method), T.DS_LABEL[r.dataset], lab,
                T.num(r[f"ctrl_{key}"], 4),
                T.ci(r[f"ctrl_{key}_lo"], r[f"ctrl_{key}_hi"], 4),
                T.res(r[f"ctrl_{key}_resolved"]),
                T.num(r[f"nat_{key}"], 4),
                T.ci(r[f"nat_{key}_lo"], r[f"nat_{key}_hi"], 4),
                T.res(r[f"nat_{key}_resolved"]),
                ("yes" if r[f"ctrl_larger_{key}"] else "no") + r"$\rightarrow$"
                + ("yes" if r[f"nat_larger_{key}"] else "no"),
                r[f"class_{key}"]])
    df = pd.DataFrame(csv)
    ck.complete("Table A2", df, 21)
    tex = T.latex_table(
        "lll rrc rrc cl",
        [[r"Method", r"Dataset", r"Coord.", r"$\tau_I$ ctrl.", r"95\% CI",
          r"res.", r"$\tau_I$ nat.", r"95\% CI", r"res.",
          r"$|\tau_{\mathrm{nonint}}|>|\tau_I|$", r"Transfer class"]],
        body,
        caption=("Native-validation full results: seven cells on all three "
                 "coordinates. " + PAIRED + " " + ORIENT
                 + " Transfer classes are the frozen X22/X23 classification."),
        label="tab:transfer-full", notes=[RESRULE], width="double")
    T.write("tableA2_transfer_full", df, tex,
            "Table A2 -- native validation full results", outs)
    return df


# ================================================================= Table 3a / 3b
X25S, X26S = T.FIG.X25_SUMMARY, T.FIG.X26_SUMMARY
SEL_CASES = [
    dict(case="NIFTY / German", src="X25", path=X25S,
         q=dict(short="D1.bce.h200.ndp", cap="D1.bce.c200.ndp",
                full="D1.bce.c1000.ndp", S="D1.bce.S.ndp", T="D1.bce.T.ndp",
                H="D1.bce.H.ndp")),
    dict(case="FairGB / Bail", src="X26", path=X26S,
         q=dict(short="bce.h200.ndp", cap="bce.cap.ndp", full="bce.full.ndp",
                S="bce.S.ndp", T="bce.T.ndp", H="bce.Hshift.ndp")),
]


def table3a(ck, outs):
    body, csv = [], []
    for c in SEL_CASES:
        s = T.FIG.summary(c["path"])
        v = {k: T.FIG.q(s, q) for k, q in c["q"].items()}
        ck.close(f"{c['src']}: S = full - cap",
                 v["full"]["mean"] - v["cap"]["mean"], v["S"]["mean"], 1e-9)
        ck.close(f"{c['src']}: T = cap - short",
                 v["cap"]["mean"] - v["short"]["mean"], v["T"]["mean"], 1e-9)
        ck.close(f"{c['src']}: H_shift = S + T",
                 v["S"]["mean"] + v["T"]["mean"], v["H"]["mean"], 1e-9)
        for k in ("short", "cap", "full", "S", "T", "H"):
            ck.interval(f"{c['src']} {k}", v[k]["mean"], v[k]["lo"], v[k]["hi"])
        direction = ("support expansion makes the effect more harmful"
                     if v["S"]["mean"] < 0 else
                     "support expansion makes the effect less harmful")
        csv.append(dict(case=c["case"], source=c["src"],
                        tau_short=v["short"]["mean"], tau_long_cap=v["cap"]["mean"],
                        tau_long_full=v["full"]["mean"],
                        S=v["S"]["mean"], S_lo=v["S"]["lo"], S_hi=v["S"]["hi"],
                        S_resolved=bool(v["S"]["resolved"]),
                        T=v["T"]["mean"], T_lo=v["T"]["lo"], T_hi=v["T"]["hi"],
                        T_resolved=bool(v["T"]["resolved"]),
                        H_shift=v["H"]["mean"], direction=direction))
        body.append([c["case"], T.num(v["short"]["mean"]), T.num(v["cap"]["mean"]),
                     T.num(v["full"]["mean"]),
                     T.num(v["S"]["mean"]) + " " + T.res(bool(v["S"]["resolved"])),
                     T.num(v["T"]["mean"]) + " " + T.res(bool(v["T"]["resolved"])),
                     direction])
    df = pd.DataFrame(csv)
    ck.complete("Table 3a", df, 2)
    tex = T.latex_table(
        "lrrrrrl",
        [[r"Case", r"$\tau$ short $H$", r"$\tau$ long, cap",
          r"$\tau$ long, full", r"$S$", r"$T$", r"Direction"]],
        body,
        caption=("Selection-support decomposition on $-\\Delta$DP under the "
                 "$\\sigma_c^{\\mathrm{BCE}}$ selector. " + ORIENT
                 + r" $S = \tau_{\mathrm{full}} - \tau_{\mathrm{cap}}$ is the "
                 r"selection-support term and $T = \tau_{\mathrm{cap}} - "
                 r"\tau_{\mathrm{short}}$ the prefix/run term; $S + T$ is the "
                 r"whole horizon shift."),
        label="tab:selection-support",
        notes=[RESRULE,
               r"FairGB/Credit is absent because its pre-registered "
               r"instrumentation-invariance gate did not pass and it was never "
               r"analysed (Table~\ref{tab:excluded}), not because it was negative."],
        width="single")
    T.write("table3a_selection_support", df, tex,
            "Table 3a -- selection-support decomposition", outs)
    return df


def table3b(ck, outs):
    LAM1, LAM2 = (5.0, 30.0), (0.01, 20.0)
    body, csv = [], []
    for ds, path in T.FIG.X27_SUMMARY.items():
        s = T.FIG.summary(path)
        # propagation at the endpoint that carries it
        props = {}
        for l2 in LAM2:
            r = T.FIG.q(s, f"prop.l2{l2:g}.ndp", "last")
            props[l2] = r
            ck.interval(f"{ds} prop l2={l2:g}", r["mean"], r["lo"], r["hi"])
        fair, fres = [], []
        for l1 in LAM1:
            for l2 in LAM2:
                r = T.FIG.q(s, f"fair.l1{l1:g}.l2{l2:g}.ndp", "last")
                p = T.FIG.q(s, f"prop.l2{l2:g}.ndp", "last")
                t = T.FIG.q(s, f"total.l1{l1:g}.l2{l2:g}.ndp", "last")
                ck.close(f"{ds} l1={l1:g} l2={l2:g}: total = prop + fair",
                         p["mean"] + r["mean"], t["mean"], 1e-9)
                ck.interval(f"{ds} fair l1={l1:g} l2={l2:g}",
                            r["mean"], r["lo"], r["hi"])
                fair.append(r["mean"])
                fres.append(bool(r["resolved"]))
        bound = max(abs(v) for v in fair)
        ck.true(f"{ds}: no fairness-correction cell is resolved", not any(fres))
        # selector agreement on the resolution state of tau_fair
        sel_states = []
        for sel in ("last", "bce", "auc"):
            st = [bool(T.FIG.q(s, f"fair.l1{l1:g}.l2{l2:g}.ndp", sel)["resolved"])
                  for l1 in LAM1 for l2 in LAM2]
            sel_states.append(any(st))
        robust = ("unresolved under all three selectors"
                  if not any(sel_states) else "differs across selectors")
        big = max(props.values(), key=lambda r: abs(r["mean"]))
        l2big = [k for k, v in props.items() if v is big][0]
        csv.append(dict(dataset=ds,
                        tau_prop_lambda2=l2big, tau_prop=big["mean"],
                        tau_prop_lo=big["lo"], tau_prop_hi=big["hi"],
                        tau_prop_resolved=bool(big["resolved"]),
                        tau_prop_small_lambda2=min(LAM2),
                        tau_prop_small=props[min(LAM2)]["mean"],
                        tau_prop_small_resolved=bool(props[min(LAM2)]["resolved"]),
                        tau_fair_max_abs=bound,
                        tau_fair_resolved_cells=int(sum(fres)),
                        tau_fair_cells=len(fair),
                        selector_robustness=robust))
        body.append([
            T.esc(ds),
            T.num(big["mean"]) + " " + T.res(bool(big["resolved"]))
            + rf" \scriptsize($\lambda_2={l2big:g}$)",
            T.ci(big["lo"], big["hi"]),
            rf"all {len(fair)} unresolved, $|\tau_{{\mathrm{{fair}}}}|\leq"
            rf"{bound:.4f}$",
            robust])
    df = pd.DataFrame(csv)
    ck.complete("Table 3b", df, 2)
    tex = T.latex_table(
        "lrrll",
        [[r"Dataset", r"$\tau_{\mathrm{prop}}$", r"95\% CI",
          r"$\tau_{\mathrm{fair}}$ over the pre-registered grid",
          r"Selector robustness"]],
        body,
        caption=("FMP component decomposition on $-\\Delta$DP under "
                 "$\\sigma_{\\mathrm{last}}$. " + ORIENT
                 + r" $\tau_{\mathrm{prop}}$ is the propagation component "
                 r"$F00\rightarrow F01(\lambda_2)$ at the endpoint where it is "
                 r"largest in magnitude, and $\tau_{\mathrm{fair}}$ the "
                 r"fairness-correction component $F01\rightarrow F11(\lambda_1,"
                 r"\lambda_2)$ at the same propagation state."),
        label="tab:fmp-components",
        notes=[RESRULE,
               r"FMP has no native $\lambda$, so every value is conditional on "
               r"the stated configuration and FMP is excluded from all benchmark "
               r"aggregates. The fairness-correction entries are a bound over the "
               r"four pre-registered grid cells, never an average: an unresolved "
               r"effect is an absence of resolved change, not a demonstrated zero."],
        width="double")
    T.write("table3b_fmp_components", df, tex,
            "Table 3b -- FMP component decomposition", outs)
    return df


def table3_combined(a, b, ck, outs):
    """One table for both mechanisms; kept for readability comparison."""
    body, csv = [], []
    for _, r in a.iterrows():
        # bracket access: a one-letter column named "T" collides with the
        # pandas transpose attribute, so r.T is not the column
        S, Sr = r["S"], r["S_resolved"]
        Tv, Tr = r["T"], r["T_resolved"]
        csv.append(dict(block="selection support", row=r["case"],
                        term1_name="S", term1=S, term1_resolved=Sr,
                        term2_name="T", term2=Tv, term2_resolved=Tr,
                        note=r["direction"]))
        body.append([r["case"], r"$S$ / $T$",
                     T.num(S) + " " + T.res(Sr),
                     T.num(Tv) + " " + T.res(Tr), r["direction"]])
    for _, r in b.iterrows():
        csv.append(dict(block="FMP components", row=r.dataset,
                        term1_name="tau_prop", term1=r.tau_prop,
                        term1_resolved=r.tau_prop_resolved,
                        term2_name="tau_fair (max |.|)", term2=r.tau_fair_max_abs,
                        term2_resolved=False, note=r.selector_robustness))
        body.append([T.esc(r.dataset),
                     r"$\tau_{\mathrm{prop}}$ / $\tau_{\mathrm{fair}}$",
                     T.num(r.tau_prop) + " " + T.res(r.tau_prop_resolved),
                     rf"$\leq{r.tau_fair_max_abs:.4f}$ u", r.selector_robustness])
    df = pd.DataFrame(csv)
    ck.complete("Table 3 (combined)", df, 4)
    tex = T.latex_table(
        "lllll",
        [[r"Case", r"Terms", r"First term", r"Second term", r"Reading"]],
        body,
        caption=("Mechanistic decomposition summary on $-\\Delta$DP. " + ORIENT
                 + r" The upper block decomposes a horizon shift into the "
                 r"selection-support term $S$ and the prefix/run term $T$; the "
                 r"lower block decomposes the FMP package into its propagation "
                 r"and fairness-correction components. The two blocks are "
                 r"different decompositions and their terms are not comparable "
                 r"across blocks."),
        label="tab:mechanism", notes=[RESRULE], width="double",
        midrules=(len(a),))
    T.write("table3_mechanistic_combined", df, tex,
            "Table 3 -- mechanistic decomposition (combined)", outs)
    return df


# ==================================================================== Table 4
def table4(ck, outs):
    rows = [
        dict(stage="Controlled audit", methods="FairGNN, NIFTY, FairGB, FairVGNN",
             datasets="German, Bail, Credit", protocol="common, $H=200$",
             cells="12 cells, 6 splits $\\times$ 5 runs",
             selector=r"$\sigma_c^{\mathrm{BCE}}$",
             question="Is the claimed intervention the part that moves the outcome?"),
        dict(stage="Native validation", methods="FairGB, FairVGNN, NIFTY",
             datasets="German, Bail, Credit",
             protocol="each method's native horizon",
             cells="7 cells, 6 splits $\\times$ 5 runs",
             selector=r"$\sigma_c^{\mathrm{BCE}}$",
             question="Does the controlled conclusion transfer to the native protocol?"),
        dict(stage="NIFTY factorial (X24)", methods="NIFTY", datasets="German",
             protocol="$2\\times2$: horizon $\\times$ augmentation view",
             cells="4 cells, 6 splits $\\times$ 5 runs",
             selector=r"$\sigma_c^{\mathrm{BCE}}$, $\sigma_c^{\mathrm{AUC}}$",
             question="Which protocol factor carries the attribution shift?"),
        dict(stage="Selection support (X25, X26)", methods="NIFTY, FairGB",
             datasets="German; Bail (Credit stopped)",
             protocol="extended horizon, capped vs full checkpoint support",
             cells="2 cases, 6 splits $\\times$ 5 runs",
             selector=r"$\sigma_c^{\mathrm{BCE}}$, $\sigma_c^{\mathrm{AUC}}$",
             question="Trajectory or checkpoint support?"),
        dict(stage="FMP case study (X27)", methods="FMP",
             datasets="Pokec-z, Pokec-n",
             protocol="7 configurations, $\\lambda$ from the official grid",
             cells="2 datasets $\\times$ 7 configs, 6 splits $\\times$ 5 runs",
             selector=r"$\sigma_{\mathrm{last}}$ (BCE, AUC as robustness)",
             question="Propagation or the claimed fairness correction?"),
    ]
    df = pd.DataFrame(rows)
    ck.complete("Table 4", df, 5)
    body = [[r["stage"], r["methods"], r["datasets"], r["protocol"], r["cells"],
             r["selector"], r["question"]] for r in rows]
    tex = T.latex_table(
        "p{2.4cm}p{2.6cm}p{2.2cm}p{2.6cm}p{2.4cm}p{1.8cm}p{3.4cm}",
        [[r"Stage", r"Methods", r"Datasets", r"Protocol",
          r"Splits $\times$ runs", r"Selector", r"Scientific question"]],
        body,
        caption=("Experiment design and coverage. This table summarises the "
                 "design; it reports no results. FMP is a mechanistic case study "
                 "and is excluded from all benchmark aggregates."),
        label="tab:design", width="double")
    T.write("table4_design_coverage", df, tex,
            "Table 4 -- experiment design and coverage", outs)
    return df


# ================================================================ appendix A3-A7
def tableA3(ck, outs):
    d = T.x24()
    cells = ("P00", "P10", "P01", "P11")
    terms = (("total", r"$\Delta_{\mathrm{total}}$"), ("H", r"$\Delta_H$"),
             ("D", r"$\Delta_D$"), ("I", r"$I_{HD}$"))
    body, csv = [], []
    for sel in ("common_bce", "common_auc"):
        for name, lab in ([(c, c) for c in cells] + list(terms)):
            for coord, clab in (("ndp", r"$-\Delta$DP"), ("neo", r"$-\Delta$EO"),
                                ("auc", r"$\Delta$AUC")):
                r = T.qsel(d, sel, name, coord)
                ck.interval(f"X24 {sel}/{name}/{coord}", r["mean"], r["lo"], r["hi"])
                csv.append(dict(selector=sel, quantity=name, coordinate=coord,
                                mean=r["mean"], lo=r["lo"], hi=r["hi"],
                                sign=r["sign"], resolved=bool(r["resolved"])))
                body.append([sel.replace("common_", ""), lab, clab,
                             T.num(r["mean"], 4), T.ci(r["lo"], r["hi"], 4),
                             f"{r['sign']:.2f}", T.res(bool(r["resolved"]))])
    df = pd.DataFrame(csv)
    ck.complete("Table A3", df, 2 * 8 * 3)
    # the frozen identity of the 2x2
    for sel in ("common_bce", "common_auc"):
        for coord in ("ndp", "neo", "auc"):
            tot = T.qsel(d, sel, "total", coord)["mean"]
            p00 = T.qsel(d, sel, "P00", coord)["mean"]
            p11 = T.qsel(d, sel, "P11", coord)["mean"]
            ck.close(f"X24 {sel}/{coord}: Delta_total = P11 - P00",
                     p11 - p00, tot, 1e-9)
    tex = T.latex_table(
        "lllrrcc",
        [[r"Selector", r"Quantity", r"Coord.", r"Mean", r"95\% CI", r"$s$", r"res."]],
        body,
        caption=("NIFTY/German $2\\times2$ protocol factorial: the four protocol "
                 "cells and the attribution-shift decomposition. " + ORIENT
                 + r" $\Delta_H$ is the horizon main effect, $\Delta_D$ the "
                 r"augmentation/validation-view main effect and $I_{HD}$ their "
                 r"interaction."),
        label="tab:x24-full", notes=[RESRULE], width="double")
    T.write("tableA3_nifty_factorial", df, tex,
            "Table A3 -- NIFTY 2x2 factorial", outs)
    return df


def tableA4(ck, outs):
    s = T.FIG.summary(X25S)
    body, csv = [], []
    for D in ("D0", "D1"):
        for sig in ("bce", "auc"):
            for coord, clab in (("ndp", r"$-\Delta$DP"), ("neo", r"$-\Delta$EO"),
                                ("auc", r"$\Delta$AUC")):
                terms = [("S", f"{D}.{sig}.S.{coord}"),
                         ("T", f"{D}.{sig}.T.{coord}"),
                         ("H_shift", f"{D}.{sig}.H.{coord}")]
                if sig == "bce":
                    terms += [("E_on", f"{D}.tele.m1ext.{coord}"),
                              ("E_off", f"{D}.tele.m0ext.{coord}")]
                for lab, q in terms:
                    r = T.FIG.q(s, q)
                    ck.interval(f"X25 {q}", r["mean"], r["lo"], r["hi"])
                    csv.append(dict(config=D, selector=sig, coordinate=coord,
                                    quantity=lab, mean=r["mean"], lo=r["lo"],
                                    hi=r["hi"], sign=r["sign"],
                                    resolved=bool(r["resolved"])))
                    body.append([D, sig.upper(), clab, lab, T.num(r["mean"], 4),
                                 T.ci(r["lo"], r["hi"], 4), f"{r['sign']:.2f}",
                                 T.res(bool(r["resolved"]))])
                ck.close(f"X25 {D}/{sig}/{coord}: H = S + T",
                         T.FIG.q(s, f"{D}.{sig}.S.{coord}")["mean"]
                         + T.FIG.q(s, f"{D}.{sig}.T.{coord}")["mean"],
                         T.FIG.q(s, f"{D}.{sig}.H.{coord}")["mean"], 1e-9)
    df = pd.DataFrame(csv)
    tex = T.latex_table(
        "llllrrcc",
        [[r"Config", r"Selector", r"Coord.", r"Term", r"Mean", r"95\% CI",
          r"$s$", r"res."]],
        body,
        caption=("X25 NIFTY/German selection-support decomposition, full. "
                 + ORIENT + r" $E_{\mathrm{on}}$ and $E_{\mathrm{off}}$ are the "
                 r"telescoping extension terms of the two arms, defined only for "
                 r"the $\sigma_c^{\mathrm{BCE}}$ selector in the frozen analysis."),
        label="tab:x25-full", notes=[RESRULE], width="double")
    T.write("tableA4_x25_decomposition", df, tex,
            "Table A4 -- X25 selection-support full", outs)
    return df


def tableA5(ck, outs):
    s = T.FIG.summary(X26S)
    body, csv = [], []
    for sig in ("bce", "auc"):
        for coord, clab in (("ndp", r"$-\Delta$DP"), ("neo", r"$-\Delta$EO"),
                            ("auc", r"$\Delta$AUC")):
            for lab, q in (("tau_short (H200)", f"{sig}.h200.{coord}"),
                           ("tau_cap", f"{sig}.cap.{coord}"),
                           ("tau_full", f"{sig}.full.{coord}"),
                           ("S", f"{sig}.S.{coord}"), ("T", f"{sig}.T.{coord}"),
                           ("H_shift", f"{sig}.Hshift.{coord}"),
                           ("E_on", f"{sig}.E1.{coord}"),
                           ("E_off", f"{sig}.E0.{coord}")):
                r = T.FIG.q(s, q)
                ck.interval(f"X26 {q}", r["mean"], r["lo"], r["hi"])
                csv.append(dict(selector=sig, coordinate=coord, quantity=lab,
                                mean=r["mean"], lo=r["lo"], hi=r["hi"],
                                sign=r["sign"], resolved=bool(r["resolved"])))
                body.append([sig.upper(), clab, lab, T.num(r["mean"], 4),
                             T.ci(r["lo"], r["hi"], 4), f"{r['sign']:.2f}",
                             T.res(bool(r["resolved"]))])
            ck.close(f"X26 {sig}/{coord}: S = E_on - E_off",
                     T.FIG.q(s, f"{sig}.E1.{coord}")["mean"]
                     - T.FIG.q(s, f"{sig}.E0.{coord}")["mean"],
                     T.FIG.q(s, f"{sig}.S.{coord}")["mean"], 1e-9)
    df = pd.DataFrame(csv)
    tex = T.latex_table(
        "lllrrcc",
        [[r"Selector", r"Coord.", r"Term", r"Mean", r"95\% CI", r"$s$", r"res."]],
        body,
        caption=("X26 FairGB/Bail selection-support decomposition, full. "
                 + ORIENT + " FairGB/Credit is not included: its pre-registered "
                 "gate did not pass and it was never analysed."),
        label="tab:x26-full", notes=[RESRULE], width="double")
    T.write("tableA5_x26_decomposition", df, tex,
            "Table A5 -- X26 selection-support full", outs)
    return df


def tableA6(ck, outs):
    LAM1, LAM2 = (5.0, 30.0), (0.01, 20.0)
    body, csv = [], []
    for ds, path in T.FIG.X27_SUMMARY.items():
        s = T.FIG.summary(path)
        for sel in ("last", "bce", "auc"):
            for l1 in LAM1:
                for l2 in LAM2:
                    for coord, clab in (("ndp", r"$-\Delta$DP"),
                                        ("neo", r"$-\Delta$EO"),
                                        ("auc", r"$\Delta$AUC")):
                        p = T.FIG.q(s, f"prop.l2{l2:g}.{coord}", sel)
                        f = T.FIG.q(s, f"fair.l1{l1:g}.l2{l2:g}.{coord}", sel)
                        t = T.FIG.q(s, f"total.l1{l1:g}.l2{l2:g}.{coord}", sel)
                        ck.close(f"{ds}/{sel}/l1{l1:g}/l2{l2:g}/{coord}: "
                                 "total = prop + fair",
                                 p["mean"] + f["mean"], t["mean"], 1e-9)
                        csv.append(dict(
                            dataset=ds, selector=sel, lambda1=l1, lambda2=l2,
                            coordinate=coord,
                            tau_prop=p["mean"], tau_prop_lo=p["lo"],
                            tau_prop_hi=p["hi"], tau_prop_resolved=bool(p["resolved"]),
                            tau_fair=f["mean"], tau_fair_lo=f["lo"],
                            tau_fair_hi=f["hi"], tau_fair_resolved=bool(f["resolved"]),
                            tau_total=t["mean"], tau_total_lo=t["lo"],
                            tau_total_hi=t["hi"],
                            tau_total_resolved=bool(t["resolved"])))
                        body.append([
                            T.esc(ds), sel, f"{l1:g}", f"{l2:g}", clab,
                            T.num(p["mean"], 4) + " " + T.res(bool(p["resolved"])),
                            T.num(f["mean"], 4) + " " + T.res(bool(f["resolved"])),
                            T.ci(f["lo"], f["hi"], 4),
                            T.num(t["mean"], 4) + " " + T.res(bool(t["resolved"]))])
    df = pd.DataFrame(csv)
    ck.complete("Table A6", df, 2 * 3 * 2 * 2 * 3)
    tex = T.latex_table(
        "lllll rrrr",
        [[r"Dataset", r"Selector", r"$\lambda_1$", r"$\lambda_2$", r"Coord.",
          r"$\tau_{\mathrm{prop}}$", r"$\tau_{\mathrm{fair}}$",
          r"95\% CI ($\tau_{\mathrm{fair}}$)", r"$\tau_{\mathrm{total}}$"]],
        body,
        caption=("X27 FMP full grid. " + ORIENT
                 + r" $\tau_{\mathrm{total}} = \tau_{\mathrm{prop}} + "
                 r"\tau_{\mathrm{fair}}$ holds exactly per cell and in every "
                 r"bootstrap replicate. FMP has no native $\lambda$; all values "
                 r"are configuration-conditional and excluded from benchmark "
                 r"aggregates."),
        label="tab:x27-full", notes=[RESRULE], width="double")
    T.write("tableA6_fmp_grid", df, tex, "Table A6 -- FMP full grid", outs)
    return df


def tableA7(ck, outs):
    rows = [dict(
        experiment="FairGB / Credit (X26)",
        stage="Selection-support replication",
        status="stopped before any run; not analysed",
        reason=("the pre-registered instrumentation-invariance gate did not pass"),
        scientific_outcome="none; the cross-dataset comparison is incomplete")]
    df = pd.DataFrame(rows)
    ck.complete("Table A7", df, 1)
    body = [[r["experiment"], r["stage"], r["status"], r["reason"],
             r["scientific_outcome"]] for r in rows]
    tex = T.latex_table(
        "p{3.0cm}p{3.0cm}p{3.2cm}p{4.2cm}p{3.6cm}",
        [[r"Experiment", r"Stage", r"Status", r"Reason", r"Scientific outcome"]],
        body,
        caption=("Excluded and stopped experiments. A stopped cell contributes "
                 "no result in either direction: it is reported as unknown, not "
                 "as a negative finding."),
        label="tab:excluded", width="double")
    T.write("tableA7_excluded", df, tex, "Table A7 -- excluded / stopped", outs)
    return df


# ======================================================================== main
def main() -> int:
    T.FIG.rc()
    ck, outs = T.Checks(), []
    d = controlled_cells(ck)
    table1(d, ck, outs)
    table1_compact_auc(d, ck, outs)
    tableA1(d, ck, outs)
    tr = transfer_rows(ck)
    table2(tr, ck, outs)
    tableA2(tr, ck, outs)
    a = table3a(ck, outs)
    b = table3b(ck, outs)
    table3_combined(a, b, ck, outs)
    table4(ck, outs)
    tableA3(ck, outs)
    tableA4(ck, outs)
    tableA5(ck, outs)
    tableA6(ck, outs)
    tableA7(ck, outs)
    ok = ck.report()
    for o in outs:
        print("[written]", o)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
