"""Direct paired uncertainty of the configuration attribution shift, per pair and coordinate.

For each of the 26 primary/alternative configuration comparisons and each unit omega = (split, run):

    tau_primary,omega = Y_omega(M^{+I}; C_primary) - Y_omega(M^{-I}; C_primary)
    tau_alt,omega     = Y_omega(M^{+I}; C_alt)     - Y_omega(M^{-I}; C_alt)
    d_omega           = tau_alt,omega - tau_primary,omega

Delta_attr is the mean of d over the 30 matched units. Its interval is the project's own paired
hierarchical bootstrap (`harness/experiments/bootstrap_armA.boot`, seed `bootstrap_armA.SEED`):
resample the 6 splits with replacement, then the 5 runs within each drawn split, and apply that one
resample to d -- so the primary and the alternative configuration, and both arms inside each, are
always carried together. 10,000 replicates, percentile interval, no magnitude floor and no
"resolved" rule: the only question is whether the 95% interval excludes zero.

Coordinates follow the paper's orientation (higher is better): AUC as is, DP and EO negated.

Reads results/per_unit_metrics.csv.gz and results/2_configuration_variation.csv (read-only) and
writes results/phase0_audit/configuration_delta_attr_bootstrap.csv.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))
from bootstrap_armA import SEED, boot  # noqa: E402  (the frozen bootstrap and its seed)

REPS = 10_000
COORDS = [("dAUC", "auc", 1), ("negDP", "dp", -1), ("negEO", "eo", -1)]

u = pd.read_csv(os.path.join(RESULTS, "per_unit_metrics.csv.gz"))
cfg = pd.read_csv(os.path.join(RESULTS, "2_configuration_variation.csv"))


def factor(v: str) -> str:
    v = str(v).lower()
    if "budget" in v:
        return "budget"
    if "upstream" in v or "spmm" in v:
        return "upstream"
    return "encoder"


def tau_units(method, dataset, configuration):
    """per-unit tau_{-I->+I} for one (method, dataset, configuration), indexed by (split, run)."""
    s = u[(u.method == method) & (u.dataset == dataset) & (u.configuration == configuration)
          & (u.protocol == "controlled") & (u.selector == "common_bce")]
    if s.empty:
        return None
    p = s.pivot_table(index=["split_id", "run_id"], columns="state", values=["auc", "dp", "eo"])
    return pd.DataFrame({c: sgn * (p[(k, "M_plus_I")] - p[(k, "M_minus_I")]) for c, k, sgn in COORDS})


pairs = cfg.drop_duplicates(["method", "dataset", "primary_configuration", "robustness_configuration"])
pairs = pairs.reset_index(drop=True)
print(f"pairs found: {len(pairs)}")
by_factor = pairs.varied_factor.map(factor).value_counts().to_dict()
print("by varied factor:", by_factor)
assert len(pairs) == 26, len(pairs)
assert by_factor == {"encoder": 18, "upstream": 6, "budget": 2}, by_factor

rng = np.random.default_rng(SEED)
rows, missing = [], []
for i, r in pairs.iterrows():
    a = tau_units(r.method, r.dataset, r.primary_configuration)
    b = tau_units(r.method, r.dataset, r.robustness_configuration)
    if a is None or b is None:
        missing.append((r.method, r.dataset, r.primary_configuration, r.robustness_configuration))
        continue
    idx = a.index.intersection(b.index)                      # matched on (split_id, run_id)
    d = (b.loc[idx] - a.loc[idx])
    cells = d.reset_index()[["split_id", "run_id"]].copy()
    n_splits = cells.split_id.nunique()
    for c, _, _ in COORDS:
        cells[c] = d[c].to_numpy()
    est = boot(cells, [c for c, _, _ in COORDS], rng, reps=REPS)   # same resample for all coordinates
    for j, (c, _, _) in enumerate(COORDS):
        lo, hi = (float(v) for v in np.percentile(est[:, j], [2.5, 97.5]))
        rows.append(dict(
            pair_id=f"P{i + 1:02d}", method=r.method, dataset=r.dataset,
            primary_configuration=r.primary_configuration,
            alternative_configuration=r.robustness_configuration,
            varied_factor=factor(r.varied_factor), coordinate=c,
            n_splits=n_splits, n_matched_units=len(idx),
            tau_primary=float(a.loc[idx, c].mean()), tau_alternative=float(b.loc[idx, c].mean()),
            delta_attr=float(d[c].mean()), ci_low=lo, ci_high=hi,
            ci_excludes_zero=bool(lo > 0 or hi < 0), bootstrap_seed=SEED, bootstrap_reps=REPS))

if missing:
    raise SystemExit(f"per-unit data missing for: {missing}")
out = pd.DataFrame(rows)
path = os.path.join(HERE, "configuration_delta_attr_bootstrap.csv")
out.to_csv(path, index=False)

# ---- sanity checks -------------------------------------------------------------------
bad_units = out[out.n_matched_units != 30]
print("\nA. point estimate == alternative - primary:",
      f"max |diff| = {(out.delta_attr - (out.tau_alternative - out.tau_primary)).abs().max():.2e}")
print("B. Table-27 source check (2_configuration_variation.csv), per-unit vs published means:")
chk = cfg.merge(out, left_on=["method", "dataset", "primary_configuration", "robustness_configuration", "coordinate"],
                right_on=["method", "dataset", "primary_configuration", "alternative_configuration", "coordinate"])
print(f"   rows compared {len(chk)};"
      f" max |primary diff| {(chk.primary_tau_I_mean - chk.tau_primary).abs().max():.2e};"
      f" max |alternative diff| {(chk.robustness_tau_I_mean - chk.tau_alternative).abs().max():.2e}")
print("C. pairing: units per pair", sorted(out.n_matched_units.unique()),
      "| splits per pair", sorted(out.n_splits.unique()),
      "| pairs off 30 units:", len(bad_units))
print("D. orientation: AUC as is, DP and EO negated (COORDS =", [(c, k, s) for c, k, s in COORDS], ")")
print("E. bootstrap pairing: d is the per-unit paired difference, so one resample serves both "
      "configurations and both arms (bootstrap_armA.boot)")

print("\nCoordinate      CI excludes zero")
counts = {}
for c, _, _ in COORDS:
    n = int(out[out.coordinate == c].ci_excludes_zero.sum())
    counts[c] = n
    label = {"dAUC": "DeltaAUC ", "negDP": "-DeltaDP ", "negEO": "-DeltaEO "}[c]
    print(f"{label}       {n} / 26")
print(f"\nseed = {SEED}, replicates = {REPS}, rows = {len(out)}")
print("saved:", path)

print("\nPairs whose interval excludes zero:")
for c, _, _ in COORDS:
    sel = out[(out.coordinate == c) & out.ci_excludes_zero]
    print(f"  {c}: " + ", ".join(f"{r.method}/{r.dataset} ({r.primary_configuration}->{r.alternative_configuration})"
                                 for r in sel.itertuples()))

print("\nLaTeX:")
print(f"""Directly estimating the paired shift
$\\Delta_{{\\mathrm{{attr}}}}$ gives intervals excluding zero in
{counts['dAUC']}, {counts['negDP']}, and {counts['negEO']} of the 26 pairs for $\\Delta\\mathrm{{AUC}}$,
$-\\Delta_{{\\mathrm{{DP}}}}$, and $-\\Delta_{{\\mathrm{{EO}}}}$, respectively.""")
