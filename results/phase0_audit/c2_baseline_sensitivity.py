"""C-2: how much of the B-referenced headline is baseline nondeterminism, and how much is strength.

Four baselines, one table. The method arms are identical in all four -- only B differs -- so
tau_{-I->+I} is the same everywhere and is not reported here.

    frozen    the published bundle: B re-trained per method, German at H = 200 (the defect)
    B_rep1    one B per unit, published horizon where one resolves (German 1000)   [reported]
    B_rep2    the identical rule, independently re-trained          [nondeterminism]
    B_H1000   H = 1000 on every dataset                                  [strength]

The blocks are the ones fixed in B_rebuild_decision.md rule 6, before any of them existed:
"surrounding package is larger", the count of cells with tau_pkg < 0, the opposite-sign count, and
the family medians of |tau_{B->-I}|. Plus, per dataset, B's own absolute AUC/DP/EO.

Writes results/phase0_audit/c2_baseline_sensitivity.csv and _by_dataset.csv.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
V2 = os.path.join(ROOT, "results_v2")

COORDS = ["dAUC", "negDP", "negEO"]
BUNDLES = {
    "frozen": os.path.join(ROOT, "results", "superseded", "cell_results.csv"),
    "B_rep1": os.path.join(ROOT, "results", "cell_results.csv"),
    "B_rep2": os.path.join(V2, "bundle_B_rep2", "cell_results.csv"),
    "B_H1000": os.path.join(V2, "bundle_B_H1000", "cell_results.csv"),
}
FAMILY = {
    "Modification": ["BIND", "EDITS", "FairEdit"],
    "Constraint": ["FairGNN", "FairVGNN", "SFG"],
    "Regularization": ["BeMap", "FairGB", "FairSIN", "GEAR", "NIFTY"],
}


def primary(path):
    d = pd.read_csv(path)
    return d[d.count_in_primary_summary == True].copy()          # noqa: E712


def blocks(d, name):
    rows = []
    for c in COORDS:
        nonint, tau_i, pkg = (d[f"tau_nonint_{c}_mean"], d[f"tau_I_{c}_mean"],
                              d[f"tau_pkg_{c}_mean"])
        rows.append(dict(
            baseline=name, coordinate=c, n_cells=len(d),
            package_larger=int((nonint.abs() > tau_i.abs()).sum()),
            tau_pkg_negative=int((pkg < 0).sum()),
            opposite_sign=int(((pkg > 0) != (tau_i > 0)).sum()),
            median_abs_tau_nonint=float(nonint.abs().median()),
            median_abs_tau_I=float(tau_i.abs().median()),
            resolved=int(d[f"tau_I_{c}_resolved"].astype(str).str.lower().eq("true").sum())))
        for fam, methods in FAMILY.items():
            g = d[d.method.isin(methods)]
            rows[-1][f"median_abs_tau_nonint_{fam[:3]}"] = float(
                g[f"tau_nonint_{c}_mean"].abs().median())
    return rows


def b_absolute():
    """B's own AUC/DP/EO per dataset, for each rebuilt variant, at the common BCE selector."""
    rows = []
    u = pd.read_csv(os.path.join(ROOT, "results", "per_unit_metrics.csv.gz"))
    f = u[(u.state == "B") & (u.protocol == "controlled") & (u.selector == "common_bce")]
    for ds, g in f.groupby("dataset"):
        rows.append(dict(baseline="frozen", dataset=ds, n_distinct_baselines=g.method.nunique(),
                         auc=float(g.auc.mean()), dp=float(g.dp.mean()), eo=float(g.eo.mean()),
                         horizon="", epoch_median=float("nan")))
    for v in ("B_rep1", "B_rep2", "B_H1000"):
        for p in sorted(glob.glob(os.path.join(V2, "baselines", v, "B_*.csv"))):
            b = pd.read_csv(p)
            b = b[b.selector == "common_bce"]
            rows.append(dict(baseline=v, dataset=os.path.basename(p)[2:-4],
                             n_distinct_baselines=1,
                             auc=float(b.auc.mean()), dp=float(b.dp.mean()),
                             eo=float(b.eo.mean()),
                             horizon=int(b.horizon.iloc[0]),
                             epoch_median=float(b.epoch.median())))
    return pd.DataFrame(rows)


def main():
    rows = []
    for name, path in BUNDLES.items():
        if not os.path.exists(path):
            raise SystemExit(f"[c2] missing bundle: {path}")
        rows += blocks(primary(path), name)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "c2_baseline_sensitivity.csv"), index=False)
    bd = b_absolute()
    bd.to_csv(os.path.join(HERE, "c2_baseline_sensitivity_by_dataset.csv"), index=False)

    print("The four baselines, on the 36 primary cells\n")
    for c in COORDS:
        s = out[out.coordinate == c].set_index("baseline")
        print(f"  {c}")
        print(f"    {'':<10}{'pkg larger':>12}{'tau_pkg<0':>11}{'opp sign':>10}"
              f"{'med|tau_B|':>12}{'resolved':>10}")
        for b in BUNDLES:
            r = s.loc[b]
            print(f"    {b:<10}{r.package_larger:>8}/36{r.tau_pkg_negative:>11}"
                  f"{r.opposite_sign:>10}{r.median_abs_tau_nonint:>12.4f}{r.resolved:>10}")
        print()

    print("B's own absolute metrics, per dataset (validation-BCE selector)\n")
    piv = bd.pivot_table(index="dataset", columns="baseline", values="auc")
    print("  AUC")
    print(piv[["frozen", "B_rep1", "B_rep2", "B_H1000"]].round(4).to_string())
    print("\n  horizon used")
    print(bd[bd.baseline != "frozen"].pivot_table(index="dataset", columns="baseline",
                                                  values="horizon").to_string())

    print("\nSensitivity, read off the table above:")
    for c in COORDS:
        s = out[out.coordinate == c].set_index("baseline")
        nd = abs(int(s.loc["B_rep1"].package_larger) - int(s.loc["B_rep2"].package_larger))
        st = abs(int(s.loc["B_rep1"].package_larger) - int(s.loc["B_H1000"].package_larger))
        fr = abs(int(s.loc["B_rep1"].package_larger) - int(s.loc["frozen"].package_larger))
        print(f"  {c:6} package larger: frozen->B_rep1 moves {fr}, "
              f"nondeterminism (rep1 vs rep2) {nd}, strength (rep1 vs H1000) {st}")


if __name__ == "__main__":
    main()
