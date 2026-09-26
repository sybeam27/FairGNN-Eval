"""T4 addendum -- intervals for the derived quantities of the NIFTY/German 2x2 (tab:nifty_factorial).

Cells (-Delta_DP, sigma_c^BCE), per matched unit (split_id, run_id):

    theta00  H=200,  D0   harness/results/armA_german.csv                 (NIFTY rows)
    theta01  H=200,  D1   harness/results/x24_nifty_german_P01.csv
    theta10  H=1000, D0   frozen: x24_nifty_german_P10.csv | rerun: x25/x25_R10.csv
    theta11  H=1000, D1   frozen: armB_native_NIFTY_german.csv | rerun: x25/x25_R11.csv

No H=200 rerun exists (X25_RESULTS.md:16-17), so both columns share theta00 and theta01; only the
H=1000 side changes. Seeds of every source are compared per unit and reported.

Derived quantities, recomputed inside every bootstrap replicate:

    Delta_H     = 1/2[(theta10 - theta00) + (theta11 - theta01)]      horizon main effect
    Delta_D     = 1/2[(theta01 - theta00) + (theta11 - theta10)]      configuration main effect
    Gamma_HD    = theta11 - theta10 - theta01 + theta00               interaction
    Delta_H|D=1 = theta11 - theta01                                   simple horizon effect at D1

Interval: paired hierarchical bootstrap, 10,000 replicates -- resample the 6 splits, then the 5 runs
inside each resampled split, applying the same draw to all four cells so units stay matched.

Resolved is reported with the frozen rule's three conditions, where sign consistency is the share of
non-zero per-unit values carrying the point estimate's sign.

Reads frozen files only; writes results/phase0_audit/T4_nifty_factorial_ci.csv.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "harness", "results")
SEED, REPS = 20260924, 10_000
MAG, SIGN = 0.010, 0.75

SOURCES = {
    "theta00": ("armA_german.csv", "H=200,  D0"),
    "theta01": ("x24_nifty_german_P01.csv", "H=200,  D1"),
    "theta10_frozen": ("x24_nifty_german_P10.csv", "H=1000, D0 (frozen)"),
    "theta11_frozen": ("armB_native_NIFTY_german.csv", "H=1000, D1 (frozen)"),
    "theta10_rerun": (os.path.join("x25", "x25_R10.csv"), "H=1000, D0 (rerun)"),
    "theta11_rerun": (os.path.join("x25", "x25_R11.csv"), "H=1000, D1 (rerun)"),
}


def cell(fname: str) -> pd.DataFrame:
    """per-unit tau_I on -Delta_DP for NIFTY/German under sigma_c^BCE, indexed by (split, run)."""
    d = pd.read_csv(os.path.join(RES, fname))
    d = d[(d.method == "NIFTY") & (d.selector == "common_bce")]
    d = d.set_index(["split_id", "run_id"]).sort_index()
    out = pd.DataFrame({"tau": -(d.m1_dp - d.m0_dp)})
    out["seed"] = d["seed"] if "seed" in d else np.nan
    return out


cells = {k: cell(f) for k, (f, _) in SOURCES.items()}
idx = cells["theta00"].index
for k, c in cells.items():
    assert c.index.equals(idx), f"{k}: unit index differs"
print(f"units: {len(idx)} = {idx.get_level_values(0).nunique()} splits x "
      f"{idx.get_level_values(1).nunique()} runs")

# seed check across the sources that a column mixes
seeds = pd.DataFrame({k: c["seed"] for k, c in cells.items()})
same = seeds.nunique(axis=1).eq(1)
print(f"seed identical across all six sources in {int(same.sum())}/{len(same)} units")
if not same.all():
    print(seeds[~same].to_string())

DERIVED = {
    "Delta_H": lambda t00, t01, t10, t11: 0.5 * ((t10 - t00) + (t11 - t01)),
    "Delta_D": lambda t00, t01, t10, t11: 0.5 * ((t01 - t00) + (t11 - t10)),
    "Gamma_HD": lambda t00, t01, t10, t11: t11 - t10 - t01 + t00,
    "Delta_H|D=1": lambda t00, t01, t10, t11: t11 - t01,
}
LEVELS = {"theta00": lambda t00, t01, t10, t11: t00, "theta01": lambda t00, t01, t10, t11: t01,
          "theta10": lambda t00, t01, t10, t11: t10, "theta11": lambda t00, t01, t10, t11: t11}

rows = []
for run_set in ("frozen", "rerun"):
    t00 = cells["theta00"]["tau"]
    t01 = cells["theta01"]["tau"]
    t10 = cells[f"theta10_{run_set}"]["tau"]
    t11 = cells[f"theta11_{run_set}"]["tau"]
    arrs = [t.to_numpy() for t in (t00, t01, t10, t11)]
    splits = idx.get_level_values(0).to_numpy()
    groups = [np.flatnonzero(splits == s) for s in np.unique(splits)]

    rng = np.random.default_rng(SEED)                     # one stream per run set, recorded
    draws = np.empty((REPS, sum(len(g) for g in groups)), dtype=int)
    for r in range(REPS):
        pick = [groups[i][rng.integers(0, len(groups[i]), len(groups[i]))]
                for i in rng.integers(0, len(groups), len(groups))]
        draws[r] = np.concatenate(pick)
    means = [a[draws].mean(axis=1) for a in arrs]         # (REPS,) per cell, same resampled units

    for term, f in {**LEVELS, **DERIVED}.items():
        per_unit = f(*arrs)                               # matched per-unit value of the quantity
        mu = float(np.mean(per_unit))
        est = f(*means)
        lo, hi = (float(v) for v in np.percentile(est, [2.5, 97.5]))
        nz = per_unit[per_unit != 0]
        sc = float((np.sign(nz) == np.sign(mu)).mean()) if len(nz) else 0.0
        rows.append(dict(run_set=run_set, term=term, mean=round(mu, 6), lo=round(lo, 6), hi=round(hi, 6),
                         sign_consistency=round(sc, 4), n_units=len(per_unit),
                         resolved=bool(abs(mu) >= MAG and lo * hi > 0 and sc >= SIGN),
                         seed=SEED, reps=REPS,
                         source_H1_D0=SOURCES[f"theta10_{run_set}"][0],
                         source_H1_D1=SOURCES[f"theta11_{run_set}"][0]))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "T4_nifty_factorial_ci.csv"), index=False)
print()
print(out[["run_set", "term", "mean", "lo", "hi", "sign_consistency", "resolved"]].to_string(index=False))
