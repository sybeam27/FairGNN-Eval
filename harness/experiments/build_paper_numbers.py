"""Every number the manuscript states, with where it comes from and whether the rebuild moved it.

One row per reported value:

    location   where it appears -- a section, a table label plus row/column, or a caption
    quantity   what it is
    value      the number as the manuscript should print it
    definition the rule that produces it, stated rather than implied
    source     the file it was computed from
    changed    yes / no / new, against the frozen bundle

The four inline tables (tab:evaluation_sets, tab:controlled_protocol_contract,
tab:native_horizon_selector, tab:aggregate_robustness) are written into the LaTeX body, so they get
rows here keyed by label plus position instead of a .tex file.

    python harness/experiments/build_paper_numbers.py --out results_v2/paper_numbers.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIT = os.path.join(ROOT, "results", "phase0_audit")
V2 = os.path.join(ROOT, "results_v2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

COORDS = ["dAUC", "negDP", "negEO"]
TEX = {"dAUC": "dAUC", "negDP": "-dDP", "negEO": "-dEO"}
R = []


def add(location, quantity, value, definition, source, changed, coordinate="", label=""):
    R.append(dict(location=location, label=label, quantity=quantity, coordinate=coordinate,
                  value=value, definition=definition, source_file=source, changed_vs_frozen=changed))


def prim(path):
    d = pd.read_csv(path)
    return d[d.count_in_primary_summary == True].copy()          # noqa: E712


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(V2, "paper_numbers.csv"))
    a = ap.parse_args()
    import build_results as B

    froz = prim(os.path.join(ROOT, "results", "cell_results.csv"))
    new = prim(os.path.join(V2, "bundle", "cell_results.csv"))
    h1000 = prim(os.path.join(V2, "bundle_B_H1000", "cell_results.csv"))
    rep2 = prim(os.path.join(V2, "bundle_B_rep2", "cell_results.csv"))
    cells_all = pd.read_csv(os.path.join(V2, "bundle", "cell_results.csv"))

    # ---------------------------------------------------------------- Table 1 / body 4.2.1
    for c in COORDS:
        def larger(d):
            return int((d[f"tau_nonint_{c}_mean"].abs() > d[f"tau_I_{c}_mean"].abs()).sum())

        def neg(d):
            return int((d[f"tau_pkg_{c}_mean"] < 0).sum())

        def opp(d):
            return int(((np.sign(d[f"tau_pkg_{c}_mean"]) * np.sign(d[f"tau_I_{c}_mean"])) < 0).sum())

        def res(d):
            return int(d[f"tau_I_{c}_resolved"].astype(str).str.lower().eq("true").sum())

        add("Sec. 4.2.1 body; tab:exp1-summary row 'Package larger'", "package larger",
            f"{larger(new)}/36", "|tau_{B->-I}| > |tau_{-I->+I}|, strict, unrounded cell means",
            "results_v2/bundle/cell_results.csv",
            "yes" if larger(new) != larger(froz) else "no", c, "tab:exp1-summary")
        add("Sec. 4.2.1 body", "tau_pkg < 0", f"{neg(new)}/36", "tau_{B->+I} mean < 0",
            "results_v2/bundle/cell_results.csv",
            "yes" if neg(new) != neg(froz) else "no", c)
        add("Sec. 4.2.1 body; tab:exp1-summary row 'Opposite sign'", "opposite sign",
            f"{opp(new)}/36", "sign(tau_pkg) * sign(tau_I) < 0; exactly 0 counts as neither",
            "results_v2/bundle/cell_results.csv",
            "yes" if opp(new) != opp(froz) else "no", c, "tab:exp1-summary")
        add("tab:exp1-summary row 'Resolved'", "resolved", f"{res(new)}/36",
            "unit-level sign consistency >= 0.75, |tau| >= 0.010, 95% interval excludes 0",
            "results_v2/bundle/cell_results.csv", "no", c, "tab:exp1-summary")
        add("tab:exp1-summary row 'Median |tau_B->-I|'", "median abs tau_nonint",
            f"{new[f'tau_nonint_{c}_mean'].abs().median():.3f}", "median over the 36 primary cells",
            "results_v2/bundle/cell_results.csv",
            "yes" if round(new[f"tau_nonint_{c}_mean"].abs().median(), 3)
            != round(froz[f"tau_nonint_{c}_mean"].abs().median(), 3) else "no", c, "tab:exp1-summary")

    # ---------------------------------------------------------------- Table 20, inline
    for name, d, tag in (("B_rep1", new, "reported"), ("B_H1000", h1000, "sensitivity")):
        for c in COORDS:
            r = B.aggregate_robustness(d, c)
            add(f"tab:aggregate_robustness ({tag}, {name}) row 'cell-weighted'",
                "package larger, cell-weighted", f"{r['cell_weighted']*100:.1f}%",
                "larger cells / 36", f"results_v2/bundle{'' if name=='B_rep1' else '_B_H1000'}/cell_results.csv",
                "yes" if name == "B_rep1" else "new", c, "tab:aggregate_robustness")
            add(f"tab:aggregate_robustness ({tag}, {name}) row 'method-balanced'",
                "package larger, method-balanced", f"{r['method_balanced']*100:.1f}%",
                "per-method share averaged over the 11 methods",
                f"results_v2/bundle{'' if name=='B_rep1' else '_B_H1000'}/cell_results.csv",
                "yes" if name == "B_rep1" else "new", c, "tab:aggregate_robustness")
            add(f"tab:aggregate_robustness ({tag}, {name}) row 'LOMO'",
                "package larger, LOMO range",
                f"{r['lomo_min']*100:.1f}-{r['lomo_max']*100:.1f}%",
                "cell-weighted share with each method dropped in turn",
                f"results_v2/bundle{'' if name=='B_rep1' else '_B_H1000'}/cell_results.csv",
                "yes" if name == "B_rep1" else "new", c, "tab:aggregate_robustness")

    # ---------------------------------------------------------------- Table 8, inline
    cov = pd.read_csv(os.path.join(V2, "bundle", "coverage.csv"))
    nat = pd.read_csv(os.path.join(V2, "bundle", "3b_protocol_native_horizon_selector.csv"))
    sel = pd.read_csv(os.path.join(V2, "bundle", "3a_protocol_selector_bce_vs_auc.csv"))
    proc = pd.read_csv(os.path.join(V2, "bundle", "3c_protocol_native_published_procedure.csv"))
    rc = nat.native_evaluation_role.value_counts().to_dict()
    sets = [("primary controlled cells", len(new), "no"),
            ("configuration-robustness cells",
             int(((cells_all.configuration_role == "robustness")
                  & (cells_all.protocol == "controlled")).sum()), "no"),
            ("selector pairs", len(sel), "no"),
            ("published-horizon, systematic", rc.get("systematic", 0), "yes"),
            ("published-horizon, targeted", rc.get("targeted", 0), "no"),
            ("re-execution only (SFG/German, SFG/Credit)", rc.get("re_execution", 0), "new"),
            ("published-procedure pairs", len(proc), "no"),
            ("FnRGNN classification cells", 3, "no"),
            ("FnRGNN regression cells", 3, "no")]
    for nm, v, ch in sets:
        add(f"tab:evaluation_sets row '{nm}'", "cell count", str(v),
            "count of matched cells or pairs in that evaluation set",
            "results_v2/bundle/coverage.csv, 3a/3b/3c", ch, "", "tab:evaluation_sets")

    # ---------------------------------------------------------------- Table 10, inline
    bd = pd.read_csv(os.path.join(AUDIT, "c2_baseline_sensitivity_by_dataset.csv"))
    for ds in ["german", "bail", "credit", "income", "pokec_z", "pokec_n", "pokec_z_g", "pokec_n_g"]:
        r1 = bd[(bd.baseline == "B_rep1") & (bd.dataset == ds)].iloc[0]
        fz = bd[(bd.baseline == "frozen") & (bd.dataset == ds)].iloc[0]
        add(f"tab:baseline_spec row '{ds}' col 'B horizon'", "B horizon",
            str(int(r1.horizon)), "resolved published horizon where one exists, else 200",
            "results_v2/baselines/B_rep1/", "yes" if ds == "german" else "no", "",
            "tab:baseline_spec")
        add(f"tab:baseline_spec row '{ds}' col 'B test AUC'", "B test AUC",
            f"{r1.auc:.4f}", "B at the validation-BCE checkpoint, mean over 30 units",
            "results_v2/baselines/B_rep1/",
            "yes" if abs(r1.auc - fz.auc) > 5e-4 else "no", "", "tab:baseline_spec")
        add(f"tab:baseline_spec row '{ds}' col 'distinct B draws'",
            "distinct B draws", "1",
            "one B per (dataset, split, run), shared by every method; was one per method",
            "results_v2/baselines/B_rep1/",
            "yes" if int(fz.n_distinct_baselines) != 1 else "no", "",
            "tab:baseline_spec")

    # ---------------------------------------------------------------- Table 14, inline
    for r in nat.itertuples():
        add(f"tab:native_horizon_selector row '{r.method}/{r.dataset}/{r.configuration}'",
            "evaluation role", r.native_evaluation_role,
            "systematic / targeted / re_execution; re_execution = the two protocols resolve to the "
            "same configuration and horizon",
            "results_v2/bundle/3b_protocol_native_horizon_selector.csv",
            "yes" if (r.method, r.dataset) in {("SFG", "german"), ("SFG", "credit")} else "no",
            "", "tab:native_horizon_selector")
    add("tab:native_horizon_selector total", "published-horizon comparisons",
        str(rc.get("systematic", 0) + rc.get("targeted", 0)),
        "systematic + targeted, excluding the two re-execution rows",
        "results_v2/bundle/3b_protocol_native_horizon_selector.csv", "yes", "",
        "tab:native_horizon_selector")

    # ---------------------------------------------------------------- Reproducibility Statement
    nf = pd.read_csv(os.path.join(AUDIT, "noise_floor_delta.csv"))
    c8 = pd.read_csv(os.path.join(AUDIT, "c8_reexecution_delta.csv"))
    add("Reproducibility Statement", "re-execution intervals excluding zero",
        f"{int(nf.ci_excludes_zero.sum())}/{len(nf)} and {int(c8.ci_excludes_zero.sum())}/{len(c8)}",
        "paired hierarchical bootstrap, 10,000 replicates, seed 20260914",
        "results/phase0_audit/noise_floor_delta.csv, c8_reexecution_delta.csv", "new")
    for c in COORDS:
        add("Reproducibility Statement", "re-execution noise floor, cross-cell max |Delta|",
            f"{nf[nf.coordinate == c].abs_delta.max():.5f}",
            "largest cell-level mean |Delta| over the 15 pairings",
            "results/phase0_audit/noise_floor_delta.csv", "new", c)
    add("Appendix C (Seeds and compute environment)", "seeds", "{27, 28, 29, 30, 31}",
        "seed = 27 + run_id, identical across splits; verified on all 3,900 store rows",
        "results_v2/bundle/per_unit_metrics.csv.gz", "no")
    add("Appendix C (Seeds and compute environment)", "compute", ">= 52 GPU-hours",
        "union of every interval the run logs date-stamp; a floor, not a total. 12 of the 36 "
        "primary cells have no timing artifact",
        "results/phase0_audit/T11_compute_environment.md", "yes")
    add("Appendix C (Seeds and compute environment)", "hardware", "a single 48 GB NVIDIA GPU",
        "the runs recorded no device name; 47.50 GiB capacity appears in an OOM traceback",
        "harness/results/x30/logs/x30_FairSIN-GIN_pokec_z.log:22", "yes")

    # ---------------------------------------------------------------- C-9 / C-12, body
    c9 = pd.read_csv(os.path.join(AUDIT, "c9_native_procedure.csv"))
    for fam in ("native", "procedure"):
        f = c9[c9.family == fam]
        n = f.groupby(["method", "dataset", "configuration"]).ngroups
        for c in COORDS:
            q = f[f.coordinate == c]
            add(f"Sec. 4.3 body ({fam})", "Delta_attr interval excludes zero",
                f"{int(q.ci_excludes_zero.sum())}/{n}",
                "paired Delta_attr, 95% interval, same estimator as the configuration pairs",
                "results/phase0_audit/c9_native_procedure.csv",
                "yes" if fam == "native" else "no", c)
    c11 = pd.read_csv(os.path.join(AUDIT, "c11_subsets.csv"))
    SUB = {"single_operation_12": "single-operation subset (12 cells)",
           "single_operation_9_no_FairEdit": "single-operation subset without FairEdit (9 cells)"}
    for r in c11.itertuples():
        if r.subset not in SUB:
            continue
        add(f"tab:aggregate_robustness row '{SUB[r.subset]}'", "package larger, subset",
            f"{r.package_larger}/{r.n_cells}",
            "|tau_{B->-I}| > |tau_{-I->+I}|, counted on that subset; reported as a fraction "
            "because one cell moves a 9-cell share by 11 points",
            "results/phase0_audit/c11_subsets.csv", "new", r.coordinate,
            "tab:aggregate_robustness")

    c12 = pd.read_csv(os.path.join(AUDIT, "c12_inert_excluded.csv"))
    for r in c12[c12.subset.str.startswith("inert_removed")].itertuples():
        add("Appendix, inert-cell sensitivity", "package larger, inert cells removed",
            f"{r.package_larger}/{r.n_cells}",
            "FairEdit/credit and FairEdit/bail removed: all three coordinates at or below the "
            "re-execution threshold",
            "results/phase0_audit/c12_inert_excluded.csv", "new", r.coordinate,
            "tab:aggregate_robustness")

    out = pd.DataFrame(R)
    out.to_csv(a.out, index=False)
    print(f"{len(out)} rows -> {a.out}")
    print(out.changed_vs_frozen.value_counts().to_string())
    print("\nby label:")
    print(out[out.label != ""].label.value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
