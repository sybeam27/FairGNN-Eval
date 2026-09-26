"""C-11, C-12 and C-9 on the rebuilt bundle.

C-11  "the surrounding package is larger" on the G.2 single-operation subset -- BIND, BeMap,
      FairEdit, GEAR, NIFTY (12 cells) -- and on the 9 that remain once FairEdit's three are
      removed. One rule for all three coordinates, `build_results.nonint_larger`:
      |tau_{B->-I}| > |tau_{-I->+I}|, strict, on unrounded means. Reported as count/denominator.

C-12  the same headline counts with the cells whose two arms are effectively identical removed.
      The rule was fixed in STEP_C_PLAN before the noise floor was measured: per coordinate the
      threshold is the smallest unit-level max |Delta| between two re-executions of one cell across
      the five noise-floor cells, and a cell is excluded only when all three of its coordinates
      fall at or below it.

C-9   the native and procedure pairs against three criteria, reported side by side: the 95%
      interval, the cross-cell re-execution maxima, and -- where the pair's cell is one the noise
      floor was measured on -- that cell's own maximum. Pairs where the primary criterion and the
      cell-matched comparison disagree are listed first.

Reads results/ (the B_rep1 rebuild), the frozen T8 activation table and the two
noise-floor CSVs. Writes c11_subsets.csv, c12_inert_excluded.csv, c9_native_procedure.csv.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
V2 = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))

COORDS = ["dAUC", "negDP", "negEO"]
SINGLE_OP = ["BIND", "BeMap", "FairEdit", "GEAR", "NIFTY"]          # G.2 subset
NOISE_CELLS = {                       # noise-floor cell -> (method, dataset, configuration)
    "SFG_german": ("SFG", "german"), "NIFTY_german": ("NIFTY", "german"),
    "FairGB_bail": ("FairGB", "bail"), "FairSIN-GCN_credit": ("FairSIN", "credit"),
    "FairVGNN_german": ("FairVGNN", "german"),
}

cells = pd.read_csv(os.path.join(V2, "cell_results.csv"))
prim = cells[cells.count_in_primary_summary == True].copy()        # noqa: E712
assert len(prim) == 36, len(prim)
nf = pd.read_csv(os.path.join(HERE, "noise_floor_delta.csv"))
t8 = pd.read_csv(os.path.join(HERE, "T8_activation.csv"))


def larger(d, c):
    """the one rule, applied to a frame of cells"""
    return (d[f"tau_nonint_{c}_mean"].abs() > d[f"tau_I_{c}_mean"].abs())


# ---------------------------------------------------------------- C-11
rows = []
sub12 = prim[prim.method.isin(SINGLE_OP)]
sub9 = sub12[sub12.method != "FairEdit"]
assert len(sub12) == 12 and len(sub9) == 9, (len(sub12), len(sub9))
for name, d in (("all_36", prim), ("single_operation_12", sub12),
                ("single_operation_9_no_FairEdit", sub9)):
    for c in COORDS:
        rows.append(dict(subset=name, coordinate=c, n_cells=len(d),
                         package_larger=int(larger(d, c).sum())))
c11 = pd.DataFrame(rows)
c11.to_csv(os.path.join(HERE, "c11_subsets.csv"), index=False)

# ---------------------------------------------------------------- C-12
rr = nf[nf.pairing == "rep1 vs rep2"]
thr = {c: float(rr[rr.coordinate == c].max_unit_abs_delta.min()) for c in COORDS}
thr_cell = {c: rr.loc[rr[rr.coordinate == c].max_unit_abs_delta.idxmin(), "cell"] for c in COORDS}
t8 = t8.assign(
    ok_dAUC=t8.max_abs_dauc <= thr["dAUC"],
    ok_negDP=t8.max_abs_ddp <= thr["negDP"],
    ok_negEO=t8.max_abs_deo <= thr["negEO"])
t8["excluded"] = t8.ok_dAUC & t8.ok_negDP & t8.ok_negEO
drop = set(zip(t8.loc[t8.excluded, "method"], t8.loc[t8.excluded, "dataset"]))
kept = prim[~prim.set_index(["method", "dataset"]).index.isin(drop)]
rows = []
for name, d in (("all_36", prim), (f"inert_removed_{len(kept)}", kept)):
    for c in COORDS:
        rows.append(dict(subset=name, coordinate=c, n_cells=len(d),
                         package_larger=int(larger(d, c).sum()),
                         tau_pkg_negative=int((d[f"tau_pkg_{c}_mean"] < 0).sum()),
                         resolved=int(d[f"tau_I_{c}_resolved"].astype(str).str.lower()
                                      .eq("true").sum())))
c12 = pd.DataFrame(rows)
c12.to_csv(os.path.join(HERE, "c12_inert_excluded.csv"), index=False)

# ---------------------------------------------------------------- C-9
pairs = pd.read_csv(os.path.join(V2, "3b_protocol_native_horizon_selector.csv"))
proc = pd.read_csv(os.path.join(V2, "3c_protocol_native_published_procedure.csv"))
cross = {c: float(nf[nf.coordinate == c].abs_delta.max()) for c in COORDS}
cellmax = {}
for tag, (m, ds) in NOISE_CELLS.items():
    q = rr[rr.cell == tag]
    cellmax[(m, ds)] = {c: float(q[q.coordinate == c].abs_delta.iloc[0]) for c in COORDS}

print("inputs:", {k: len(v) for k, v in
                  (("3b native", pairs), ("3c procedure", proc))})
print("3b columns:", [c for c in pairs.columns][:14])
print("3c columns:", [c for c in proc.columns][:14])

print("\n=== C-11 ===")
print(c11.pivot_table(index="subset", columns="coordinate", values="package_larger").to_string())
print("\nas count/denominator:")
for _, r in c11.iterrows():
    print(f"  {r.subset:32} {r.coordinate:6} {r.package_larger}/{r.n_cells}")

print("\n=== C-12 ===")
print("thresholds (pre-registered: smallest unit-level max |Delta| over rep1-vs-rep2):")
for c in COORDS:
    print(f"  {c:6} {thr[c]:.6f}   (from {thr_cell[c]})")
print(f"\ncells excluded: {len(drop)} -> {sorted(drop)}")
near = t8[(~t8.excluded) & (t8[['ok_dAUC', 'ok_negDP', 'ok_negEO']].sum(axis=1) == 2)]
if len(near):
    print("passed 2 of 3 (kept by the all-three rule):")
    print(near[["method", "dataset", "max_abs_dauc", "max_abs_ddp", "max_abs_deo"]]
          .to_string(index=False))
print()
print(c12.pivot_table(index="subset", columns="coordinate",
                      values=["package_larger", "tau_pkg_negative", "resolved"]).to_string())
print("\nLimitation to report beside this: the noise floor was measured on "
      + ", ".join(sorted(f"{m}/{d}" for m, d in NOISE_CELLS.values()))
      + " -- neither FairEdit nor FairGNN is among them, so the threshold is transferred.")

print("\n=== C-9 reference values ===")
print("cross-cell re-execution maxima:", {c: round(cross[c], 5) for c in COORDS})
print("cell-matched maxima (rep1 vs rep2):")
for k, v in cellmax.items():
    print(f"  {k[0]}/{k[1]:8} " + "  ".join(f"{c} {v[c]:.5f}" for c in COORDS))
