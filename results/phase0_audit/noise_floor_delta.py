"""The noise floor of tau_{-I->+I}: paired differences between independent realizations of one cell.

For a cell and a realization r, each unit omega = (split, run) gives

    tau_r,omega = Y_omega(M^{+I}; r) - Y_omega(M^{-I}; r)

and for a pair of realizations (r, r') the per-unit difference is d_omega = tau_r',omega - tau_r,omega.
Delta is the mean of d over the 30 matched units. Its interval is the same estimator A-3 used
(`harness/experiments/bootstrap_armA.boot`, seed `bootstrap_armA.SEED`, 10,000 replicates): resample
the 6 splits with replacement, then the 5 runs inside each drawn split, and apply that one resample
to d, so both realizations and both arms inside each are always carried together.

Three pairings per cell: frozen vs rep1, frozen vs rep2, rep1 vs rep2. Nothing here is a
"resolved" test; the only question is whether the 95% interval excludes zero -- that is, whether
re-executing the identical command moves tau_I by more than its own paired uncertainty.

Coordinates follow the paper's orientation (higher is better): AUC as is, DP and EO negated.

Reads results/per_unit_metrics.csv.gz (frozen, read-only) and results_v2/noise_floor/rep*/ and writes
results/phase0_audit/noise_floor_delta.csv.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results")
NF = os.path.join(ROOT, "results_v2", "noise_floor")
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))
from bootstrap_armA import SEED, boot  # noqa: E402  (the frozen bootstrap and its seed)

REPS = 10_000
COORDS = [("dAUC", "auc", 1), ("negDP", "dp", -1), ("negEO", "eo", -1)]
SELECTOR = "common_bce"

# tag -> (method, dataset, configuration) as build_results names them
CELLS = {
    "SFG_german": ("SFG", "german", "default"),
    "NIFTY_german": ("NIFTY", "german", "default"),
    "FairGB_bail": ("FairGB", "bail", "default"),
    "FairSIN-GCN_credit": ("FairSIN", "credit", "GCN"),
    "FairVGNN_german": ("FairVGNN", "german", "default"),
}


def tau_frozen(method, dataset, configuration):
    """per-unit tau_{-I->+I} from the frozen export, indexed by (split, run)."""
    u = tau_frozen.cache
    s = u[(u.method == method) & (u.dataset == dataset) & (u.configuration == configuration)
          & (u.protocol == "controlled") & (u.selector == SELECTOR)]
    if s.empty:
        raise SystemExit(f"frozen per-unit rows missing for {method}/{dataset}/{configuration}")
    p = s.pivot_table(index=["split_id", "run_id"], columns="state", values=["auc", "dp", "eo"])
    return pd.DataFrame({c: sgn * (p[(k, "M_plus_I")] - p[(k, "M_minus_I")]) for c, k, sgn in COORDS})


tau_frozen.cache = pd.read_csv(os.path.join(RESULTS, "per_unit_metrics.csv.gz"))


def tau_rerun(path):
    """per-unit tau_{-I->+I} from a re-run store, read exactly as the frozen stores are read."""
    d = pd.read_csv(path)
    d = d[d.selector == SELECTOR]
    for c in ("m1_auc", "m1_dp", "m1_eo", "m0_auc", "m0_dp", "m0_eo"):
        d[c] = d[c].astype(float)
    d = d.set_index(["split_id", "run_id"]).sort_index()
    return pd.DataFrame({c: sgn * (d[f"m1_{k}"] - d[f"m0_{k}"]) for c, k, sgn in COORDS})


def interval(a: pd.DataFrame, b: pd.DataFrame, rng):
    """paired Delta of tau_I between two realizations, with the frozen hierarchical bootstrap."""
    idx = a.index.intersection(b.index)
    d = b.loc[idx] - a.loc[idx]
    cells = d.reset_index()[["split_id", "run_id"]].copy()
    for c, _, _ in COORDS:
        cells[c] = d[c].to_numpy()
    est = boot(cells, [c for c, _, _ in COORDS], rng, reps=REPS)   # one resample, all coordinates
    out = {}
    for j, (c, _, _) in enumerate(COORDS):
        lo, hi = (float(v) for v in np.percentile(est[:, j], [2.5, 97.5]))
        out[c] = dict(delta=float(d[c].mean()), abs_delta=float(abs(d[c].mean())),
                      max_unit_abs_delta=float(d[c].abs().max()),
                      n_units_changed=int((d[c].abs() > 0).sum()),
                      ci_low=lo, ci_high=hi, ci_excludes_zero=bool(lo > 0 or hi < 0))
    return out, len(idx), int(d.reset_index().split_id.nunique())


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for tag, (method, dataset, configuration) in CELLS.items():
        real = {"frozen": tau_frozen(method, dataset, configuration)}
        for k in (1, 2):
            p = os.path.join(NF, f"rep{k}", f"{tag}.csv")
            if os.path.exists(p):
                real[f"rep{k}"] = tau_rerun(p)
        names = list(real)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                res, n_units, n_splits = interval(real[x], real[y], rng)
                for c, _, _ in COORDS:
                    rows.append(dict(cell=tag, method=method, dataset=dataset,
                                     configuration=configuration, pairing=f"{x} vs {y}",
                                     coordinate=c, n_splits=n_splits, n_matched_units=n_units,
                                     tau_a=float(real[x][c].mean()), tau_b=float(real[y][c].mean()),
                                     **res[c], bootstrap_seed=SEED, bootstrap_reps=REPS))
    out = pd.DataFrame(rows)
    path = os.path.join(HERE, "noise_floor_delta.csv")
    out.to_csv(path, index=False)

    print(f"seed = {SEED}, replicates = {REPS}, selector = {SELECTOR}, rows = {len(out)}")
    print("saved:", path)
    print("\nPer cell and pairing, Delta of tau_I with its 95% paired interval "
          "(* = interval excludes zero):\n")
    for tag in CELLS:
        s = out[out.cell == tag]
        if s.empty:
            continue
        print(f"  {tag}")
        for pr in s.pairing.unique():
            q = s[s.pairing == pr]
            bits = []
            for c, _, _ in COORDS:
                r = q[q.coordinate == c].iloc[0]
                bits.append(f"{c} {r.delta:+.4f} [{r.ci_low:+.4f},{r.ci_high:+.4f}]"
                            f"{'*' if r.ci_excludes_zero else ' '}")
            print(f"    {pr:<20} " + "  ".join(bits))
        print()

    print("Intervals excluding zero, by coordinate:")
    for c, _, _ in COORDS:
        s = out[out.coordinate == c]
        n = int(s.ci_excludes_zero.sum())
        who = ", ".join(f"{r.cell} ({r.pairing})" for r in s[s.ci_excludes_zero].itertuples())
        print(f"  {c}: {n} / {len(s)} pairings" + (f"  -- {who}" if n else ""))

    print("\nRe-run noise floor (max |Delta| over every pairing), for C-9:")
    for c, _, _ in COORDS:
        print(f"  {c}: {out[out.coordinate == c].abs_delta.max():.6f}")


if __name__ == "__main__":
    main()
