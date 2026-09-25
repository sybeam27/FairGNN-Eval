"""C-9: the native and procedure pairs against the measured re-execution noise.

Delta_attr = tau_I(published protocol) - tau_I(controlled), formed per matched unit and read with
the project's own paired hierarchical bootstrap (`bootstrap_armA.boot`, seed `bootstrap_armA.SEED`,
10,000 replicates) -- the same estimator A-3 used for the configuration pairs and the noise floor.

Three criteria, reported side by side, as fixed in STEP_C_PLAN before this ran:

    primary        the 95% interval excludes zero              <- what the paper reports
    secondary      |Delta_attr| > the cross-cell re-execution maximum
                   (dAUC 0.01174, negDP 0.06677, negEO 0.04599)
    cell-matched   |Delta_attr| > that same cell's own re-execution maximum, where the noise
                   floor was measured on it. The strongest form: no transfer between methods.

The two SFG rows are reclassified out of the native set (Decision 2, from A-1: a pure re-run), so
the native set is 19 pairs, not 21. They are reported separately as `re_execution`.

Writes results/phase0_audit/c9_native_procedure.csv.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
V2 = os.path.join(ROOT, "results_v2", "bundle")
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))
from bootstrap_armA import SEED, boot  # noqa: E402

REPS = 10_000
COORDS = [("dAUC", "auc", 1), ("negDP", "dp", -1), ("negEO", "eo", -1)]
SELECTOR = "common_bce"
CROSS = {"dAUC": 0.011740, "negDP": 0.066773, "negEO": 0.045986}
CELLMAX = {                                    # rep1-vs-rep2 cell-level |Delta|, per coordinate
    ("SFG", "german"): {"dAUC": 0.00116, "negDP": 0.04890, "negEO": 0.04079},
    ("NIFTY", "german"): {"dAUC": 0.00029, "negDP": 0.00126, "negEO": 0.00438},
    ("FairGB", "bail"): {"dAUC": 0.00251, "negDP": 0.00542, "negEO": 0.00805},
    ("FairSIN", "credit"): {"dAUC": 0.00866, "negDP": 0.01866, "negEO": 0.01576},
    ("FairVGNN", "german"): {"dAUC": 0.00001, "negDP": 0.00015, "negEO": 0.00063},
}
REEXEC = {("SFG", "german"), ("SFG", "credit")}       # Decision 2

u = pd.read_csv(os.path.join(ROOT, "results", "per_unit_metrics.csv.gz"))


def tau(method, dataset, configuration, protocol):
    s = u[(u.method == method) & (u.dataset == dataset) & (u.configuration == configuration)
          & (u.protocol == protocol) & (u.selector == SELECTOR)]
    if s.empty:
        return None
    p = s.pivot_table(index=["split_id", "run_id"], columns="state", values=["auc", "dp", "eo"])
    return pd.DataFrame({c: sgn * (p[(k, "M_plus_I")] - p[(k, "M_minus_I")])
                         for c, k, sgn in COORDS})


def run(frame, family):
    rng = np.random.default_rng(SEED)
    out = []
    for r in frame.itertuples():
        a = tau(r.method, r.dataset, r.configuration, "controlled")
        b = tau(r.method, r.dataset, r.configuration, "native")
        if a is None or b is None:
            raise SystemExit(f"[c9] per-unit rows missing for {r.method}/{r.dataset}/"
                             f"{r.configuration}")
        idx = a.index.intersection(b.index)
        d = b.loc[idx] - a.loc[idx]
        cells = d.reset_index()[["split_id", "run_id"]].copy()
        for c, _, _ in COORDS:
            cells[c] = d[c].to_numpy()
        est = boot(cells, [c for c, _, _ in COORDS], rng, reps=REPS)
        fam = "re_execution" if (r.method, r.dataset) in REEXEC else family
        for j, (c, _, _) in enumerate(COORDS):
            lo, hi = (float(v) for v in np.percentile(est[:, j], [2.5, 97.5]))
            delta = float(d[c].mean())
            cm = CELLMAX.get((r.method, r.dataset), {}).get(c)
            out.append(dict(
                family=fam, method=r.method, dataset=r.dataset,
                configuration=r.configuration,
                native_evaluation_role=getattr(r, "native_evaluation_role", ""),
                coordinate=c, n_matched_units=len(idx),
                tau_controlled=float(a.loc[idx, c].mean()),
                tau_published=float(b.loc[idx, c].mean()),
                delta_attr=delta, abs_delta_attr=abs(delta),
                ci_low=lo, ci_high=hi,
                ci_excludes_zero=bool(lo > 0 or hi < 0),
                rerun_max_cross_cell=CROSS[c],
                exceeds_rerun_max=bool(abs(delta) > CROSS[c]),
                rerun_max_same_cell=cm,
                exceeds_same_cell_max=(None if cm is None else bool(abs(delta) > cm)),
                bootstrap_seed=SEED, bootstrap_reps=REPS))
    return out


nat = pd.read_csv(os.path.join(V2, "3b_protocol_native_horizon_selector.csv"))
proc = pd.read_csv(os.path.join(V2, "3c_protocol_native_published_procedure.csv"))
rows = run(nat, "native") + run(proc, "procedure")
out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "c9_native_procedure.csv"), index=False)

n_nat = out[out.family == "native"].groupby(["method", "dataset", "configuration"]).ngroups
n_re = out[out.family == "re_execution"].groupby(["method", "dataset"]).ngroups
n_pr = out[out.family == "procedure"].groupby(["method", "dataset"]).ngroups
print(f"pairs: native {n_nat}, re_execution {n_re}, procedure {n_pr}  "
      f"(21 native before Decision 2)")

# ---- disagreements first, as the plan requires ------------------------------------------
dis = out[(out.rerun_max_same_cell.notna())
          & (out.ci_excludes_zero != out.exceeds_same_cell_max)]
print("\n" + "=" * 90)
print("DISAGREEMENTS between the primary criterion and the cell-matched comparison")
print("=" * 90)
if dis.empty:
    print("  none: on every pair whose cell has a measured noise floor, the interval and the")
    print("  cell-matched threshold agree.")
else:
    for r in dis.itertuples():
        print(f"  {r.family:12} {r.method}/{r.dataset}/{r.configuration} [{r.coordinate}]")
        print(f"      |Delta_attr| = {r.abs_delta_attr:.5f}   interval "
              f"[{r.ci_low:+.4f},{r.ci_high:+.4f}] excludes 0: {r.ci_excludes_zero}")
        print(f"      that cell's own re-execution max = {r.rerun_max_same_cell:.5f} -> "
              f"exceeds it: {r.exceeds_same_cell_max}")

for fam in ("native", "procedure", "re_execution"):
    f = out[out.family == fam]
    if f.empty:
        continue
    n = f.groupby(["method", "dataset", "configuration"]).ngroups
    print(f"\n=== {fam} ({n} pairs) ===")
    print(f"  {'coordinate':<8}{'interval excl. 0':>18}{'> cross-cell max':>18}"
          f"{'> own-cell max':>16}")
    for c, _, _ in COORDS:
        q = f[f.coordinate == c]
        own = q[q.rerun_max_same_cell.notna()]
        print(f"  {c:<8}{int(q.ci_excludes_zero.sum()):>13}/{len(q):<4}"
              f"{int(q.exceeds_rerun_max.sum()):>13}/{len(q):<4}"
              f"{int(own.exceeds_same_cell_max.sum()) if len(own) else 0:>11}/{len(own):<4}")

print("\nre_execution rows in full (Decision 2 removed these from the native set):")
r = out[out.family == "re_execution"]
for cell in r.groupby(["method", "dataset"]).groups:
    q = r[(r.method == cell[0]) & (r.dataset == cell[1])]
    bits = []
    for c, _, _ in COORDS:
        x = q[q.coordinate == c].iloc[0]
        bits.append(f"{c} {x.delta_attr:+.4f} [{x.ci_low:+.4f},{x.ci_high:+.4f}]"
                    f"{'*' if x.ci_excludes_zero else ' '}")
    print(f"  {cell[0]}/{cell[1]:8} " + "  ".join(bits))
