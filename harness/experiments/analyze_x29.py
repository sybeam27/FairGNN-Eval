"""X29 coverage-extension analysis: CORE, EXTENSION and COMBINED DESCRIPTIVE.

Nothing here is a new statistic. The cell construction, the bootstrap and the
resolved rule are imported from the frozen modules and used unchanged:

    analyze_armA.build            derived per-cell quantities
    analyze_armA.sign_stability   sign stability
    analyze_armA.SIGN_MIN/NEAR_ZERO   the frozen resolved thresholds
    bootstrap_armA.cell_table/boot/SEED   the frozen paired hierarchical bootstrap

The three views are kept apart and never silently merged:

    CORE                  the frozen 12 controlled cells (X22)
    EXTENSION             the new X29 cells
    COMBINED DESCRIPTIVE  both together, descriptive only

Writes only under harness/results/x29/. No frozen artifact is read for writing or
overwritten.

    python harness/experiments/analyze_x29.py --out harness/results/x29
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

from analyze_armA import build, sign_stability, SIGN_MIN, NEAR_ZERO   # noqa: E402
from bootstrap_armA import cell_table, boot, SEED                     # noqa: E402

CORE_CSVS = [f"{ROOT}/harness/results/armA_german.csv",
             f"{ROOT}/harness/results/armA_bail.csv",
             f"{ROOT}/harness/results/armA_bail_s23_25.csv",
             f"{ROOT}/harness/results/armA_credit.csv",
             f"{ROOT}/harness/results/armA_credit_s23_25.csv",
             f"{ROOT}/harness/results/armA_fairvgnn_german.csv",
             f"{ROOT}/harness/results/armA_fairvgnn_bail.csv",
             f"{ROOT}/harness/results/armA_fairvgnn_credit.csv"]
X29_GLOB = f"{ROOT}/harness/results/x29/x29_*.csv"
COLS = ["abase_auc", "abase_ndp", "int_auc", "int_ndp", "apkg_auc", "apkg_ndp",
        "D_auc", "D_ndp"]
# X29 section 4 registers -dEO as the secondary coordinate, but the frozen
# bootstrap_armA.cell_table hard-codes its column list to dAUC and -dDP, so EO
# never reaches the bootstrap. The frozen helper is used unchanged for its eight
# columns and the three EO columns are appended by the identical pivot, then fed
# to the frozen boot() exactly as the others are. No new statistic: same
# construction, same bootstrap, one more registered coordinate.
COLS_EO = ["abase_neo", "int_neo", "apkg_neo"]
ALL_COLS = COLS + COLS_EO
COORDS = (("auc", "dAUC"), ("ndp", "-dDP"), ("neo", "-dEO"))
SEL = "common_bce"


def cell_table_eo(d, method, dataset):
    """The frozen cell table, plus the EO columns built the same way."""
    out = cell_table(d, method, dataset)
    g = d[(d.method == method) & (d.dataset == dataset)]
    w = g.pivot_table(index=["split_id", "run_id"], columns="selector",
                      values=["abase_neo", "int_neo", "apkg_neo"])
    idx = out.set_index(["split_id", "run_id"]).index
    for q in ("abase", "int", "apkg"):
        out[f"{q}_neo"] = w[(f"{q}_neo", SEL)].reindex(idx).to_numpy()
    return out


def resolved(mean, lo, hi, sign):
    """The frozen rule, unchanged."""
    return bool(sign >= SIGN_MIN and abs(mean) >= NEAR_ZERO and lo * hi > 0)


SPLITS, RUNS, SELECTORS = 6, 5, 2
UNITS = SPLITS * RUNS            # (split, run) units per method x dataset
ROWS = UNITS * SELECTORS


def completeness(d, tag, fh):
    """Which (method, dataset) are short of a full 6 splits x 5 runs?

    A partially written store is the dangerous case: a mean over one unit will
    happily come out "resolved". Nothing partial reaches the summary.
    """
    short = []
    print(f"  [{tag}] completeness, expecting {UNITS} units ({ROWS} rows) each:",
          file=fh)
    for (m, ds_), g in d.groupby(["method", "dataset"], sort=True):
        u = g.groupby(["split_id", "run_id"]).ngroups
        ok = (u == UNITS and len(g) == ROWS)
        print(f"    {m:<9} {ds_:<11} {u:>3}/{UNITS} units, {len(g):>3}/{ROWS} rows"
              f"  {'OK' if ok else 'INCOMPLETE'}", file=fh)
        if not ok:
            short.append((m, ds_, u, len(g)))
    return short


def contracts(d, tag, fh):
    """Per-row checks. A violation is printed and counted, never silently fixed."""
    bad = 0
    key = ["method", "dataset", "split_id", "run_id", "selector"]
    dup = int(d.duplicated(key).sum())
    if dup:
        print(f"  [{tag}] {dup} duplicated cell keys", file=fh)
        bad += dup
    nonfinite = int((~np.isfinite(d[["int_auc", "int_ndp", "abase_auc",
                                     "abase_ndp"]].to_numpy())).sum())
    if nonfinite:
        print(f"  [{tag}] {nonfinite} non-finite outcomes", file=fh)
        bad += nonfinite
    # the registered identity, per row
    for c in ("auc", "ndp", "neo"):
        r = (d[f"abase_{c}"] + d[f"int_{c}"] - d[f"apkg_{c}"]).abs().max()
        print(f"  [{tag}] identity apkg = abase + int ({c}): max |residual| "
              f"{r:.2e}", file=fh)
        if r > 1e-9:
            bad += 1
    if "eo_defined" in d.columns:
        undef = int((~d.eo_defined.astype(bool)).sum())
        print(f"  [{tag}] EO undefined rows: {undef}", file=fh)
    print(f"  [{tag}] rows {len(d)}, (split,run) units {len(d) // SELECTORS}, "
          f"method x dataset cells {d.groupby(['method','dataset']).ngroups}, "
          f"contract violations {bad}", file=fh)
    return bad


def per_cell(d, fh, tag):
    """Bootstrap each (method, dataset) exactly as the frozen script does."""
    rng = np.random.default_rng(SEED)
    out = []
    for ds_ in sorted(d.dataset.unique()):
        for m in sorted(d.method.unique()):
            cells = cell_table_eo(d, m, ds_)
            if cells.empty:
                continue
            reps = boot(cells, ALL_COLS, rng)
            g = d[(d.method == m) & (d.dataset == ds_) & (d.selector == SEL)]
            for q, lab in (("abase", "tau_nonint"), ("int", "tau_I"),
                           ("apkg", "tau_pkg"), ("D", "D_selector")):
                for c, cl in COORDS:
                    if q == "D" and c == "neo":
                        continue      # cell_table defines D on dAUC/-dDP only
                    k = ALL_COLS.index(f"{q}_{c}")
                    lo, hi = np.percentile(reps[:, k], [2.5, 97.5])
                    mu = float(cells[f"{q}_{c}"].mean())
                    sgn = (sign_stability(g[f"{q}_{c}"])
                           if q != "D" else sign_stability(cells[f"{q}_{c}"]))
                    out.append(dict(view=tag, method=m, dataset=ds_, quantity=lab,
                                    coord=cl, mean=mu, lo=float(lo), hi=float(hi),
                                    sign=float(sgn),
                                    resolved=resolved(mu, lo, hi, sgn),
                                    n_cells=len(cells)))
    return pd.DataFrame(out)


def describe(s, d, tag, fh):
    """The descriptive summary the protocol asks for, as counts of these cells."""
    print(f"\n{'=' * 78}\n{tag}\n{'=' * 78}", file=fh)
    cells = s[(s.quantity == "tau_I") & (s.coord == "-dDP")]
    base = s[(s.quantity == "tau_nonint") & (s.coord == "-dDP")]
    n = len(cells)
    if n == 0:
        print("  no cells", file=fh)
        return
    larger = 0
    for _, r in cells.iterrows():
        b = base[(base.method == r.method) & (base.dataset == r.dataset)].iloc[0]
        larger += abs(b["mean"]) > abs(r["mean"])
    res = int(cells.resolved.sum())
    print(f"  method x dataset cells                 {n}", file=fh)
    print(f"  |tau_nonint| > |tau_I| on -dDP         {larger} of {n}", file=fh)
    print(f"  resolved tau_I on -dDP                 {res} of {n}", file=fh)
    print(f"  tau_I  (-dDP)  mean {cells['mean'].mean():+.4f}  "
          f"min {cells['mean'].min():+.4f}  max {cells['mean'].max():+.4f}", file=fh)
    print(f"  tau_nonint (-dDP) mean {base['mean'].mean():+.4f}  "
          f"min {base['mean'].min():+.4f}  max {base['mean'].max():+.4f}", file=fh)
    print("  These are counts of these cells. They are not a rate in any "
          "population of fair-GNN research.", file=fh)

    # utility-fairness quadrant of the intervention
    au = s[(s.quantity == "tau_I") & (s.coord == "dAUC")]
    print("\n  intervention quadrant (x = dAUC, y = -dDP):", file=fh)
    for _, r in cells.iterrows():
        a = au[(au.method == r.method) & (au.dataset == r.dataset)].iloc[0]
        quad = (("utility up" if a["mean"] > 0 else "utility down") + ", " +
                ("fairness up" if r["mean"] > 0 else "fairness down"))
        print(f"    {r.method:<9} {r.dataset:<11} {quad:<28}"
              f" tau_I=[{a['mean']:+.4f}, {r['mean']:+.4f}]"
              f" {'R' if r.resolved else 'u'}", file=fh)


def selector_agreement(d, fh, tag):
    """Does the qualitative reading change between the two selectors?"""
    print(f"\n  selector agreement on -dDP ({tag}):", file=fh)
    for (m, ds_), g in d.groupby(["method", "dataset"], sort=True):
        b = g[g.selector == "common_bce"].int_ndp.mean()
        a = g[g.selector == "common_auc"].int_ndp.mean()
        same = (b > 0) == (a > 0)
        print(f"    {m:<9} {ds_:<11} BCE {b:+.4f}  AUC {a:+.4f}  "
              f"sign {'agrees' if same else 'DIFFERS'}", file=fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "harness", "results", "x29"))
    ap.add_argument("--allow-partial", action="store_true",
                    help="smoke-test only: analyse a store that is still being "
                         "written. Never use for a reported result.")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ext_files = sorted(glob.glob(X29_GLOB))
    ext_files = [f for f in ext_files if not f.endswith(("_summary.csv",
                                                         "_cell_table.csv"))]
    if not ext_files:
        raise SystemExit("no X29 result CSV found; nothing to analyse")

    txt = os.path.join(a.out, "x29_analysis.txt")
    with open(txt, "w") as fh:
        print("X29 coverage extension -- CORE / EXTENSION / COMBINED DESCRIPTIVE",
              file=fh)
        print("The frozen resolved rule, bootstrap and cell construction are "
              "imported unchanged.\n", file=fh)
        print(f"extension inputs: {[os.path.basename(f) for f in ext_files]}",
              file=fh)

        core = build(CORE_CSVS)
        ext = build(ext_files)
        print("\ncontracts:", file=fh)
        bad = contracts(core, "CORE", fh) + contracts(ext, "EXTENSION", fh)
        print("", file=fh)
        short = completeness(core, "CORE", fh) + completeness(ext, "EXTENSION", fh)
        if short and not a.allow_partial:
            msg = "; ".join(f"{m}/{ds} has {u}/{UNITS} units, {r}/{ROWS} rows"
                            for m, ds, u, r in short)
            print(f"\nREFUSED: incomplete cells -- {msg}", file=fh)
            fh.flush()
            raise SystemExit(
                f"[x29] refusing to analyse an incomplete store: {msg}\n"
                f"       a mean over a partial cell can come out 'resolved' and "
                f"mean nothing.\n"
                f"       pass --allow-partial only for a smoke test.")
        if short:
            print("\n  WARNING: --allow-partial is set; the following are "
                  "INCOMPLETE and must not be reported:", file=fh)
            for m, ds_, u, r in short:
                print(f"    {m}/{ds_}: {u}/{UNITS} units", file=fh)

        s_core = per_cell(core, fh, "CORE")
        s_ext = per_cell(ext, fh, "EXTENSION")
        s_all = pd.concat([s_core, s_ext], ignore_index=True)

        describe(s_core, core, "CORE -- the frozen 12 controlled cells (X22)", fh)
        describe(s_ext, ext, "EXTENSION -- the new X29 cells", fh)

        comb = s_all.copy()
        comb["view"] = "COMBINED DESCRIPTIVE"
        describe(comb, pd.concat([core, ext], ignore_index=True),
                 "COMBINED DESCRIPTIVE -- core and extension together", fh)

        selector_agreement(ext, fh, "EXTENSION")
        print(f"\ncontract violations total: {bad}", file=fh)

    s_all.to_csv(os.path.join(a.out, "x29_summary.csv"), index=False)
    pd.concat([core.assign(view="CORE"), ext.assign(view="EXTENSION")],
              ignore_index=True).to_csv(
        os.path.join(a.out, "x29_cell_table.csv"), index=False)
    print(open(txt).read())
    print(f"[written] {txt}")
    print(f"[written] {os.path.join(a.out, 'x29_summary.csv')}")
    print(f"[written] {os.path.join(a.out, 'x29_cell_table.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
