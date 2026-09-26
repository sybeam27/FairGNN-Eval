"""C-8: the two SFG pairs that differ by nothing but the process they ran in.

A-1 established that SFG/german and SFG/credit are a pure re-execution: the configuration diff
between the `controlled` and `native` rows is empty, the loader, preprocessing and seeds are the
same, and the `--native` flag reaches only `m_epochs`, which is 200 on both datasets anyway. The
harness-trained B is identical in 60/60 rows; only SFG's own training loop diverges.

So their Delta_attr is a re-execution difference, measured with the paper's own estimator:

    d_omega = tau_I(native)_omega - tau_I(controlled)_omega          per unit omega = (split, run)
    Delta_attr = mean of d, interval from `bootstrap_armA.boot` (seed `bootstrap_armA.SEED`,
    10,000 replicates): resample the 6 splits, then the 5 runs inside each drawn split.

Identical in form to `noise_floor_delta.py`, which does the same thing for two realizations the
audit produced itself. This is block B of tab:noise_floor; that script produces block A.

If an interval here excludes zero, the estimator does not cover re-execution noise -- a finding
about the bootstrap, not about SFG. That case is reported before anything else (STEP_C_PLAN C-8).

Coordinates follow the paper's orientation: AUC as is, DP and EO negated.

Reads results/per_unit_metrics.csv.gz (frozen, read-only); writes
results/phase0_audit/c8_reexecution_delta.csv.
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
SELECTOR = "common_bce"
PAIRS = [("SFG", "german"), ("SFG", "credit")]

u = pd.read_csv(os.path.join(RESULTS, "per_unit_metrics.csv.gz"))


def tau(method, dataset, protocol):
    """per-unit tau_{-I->+I} for one (cell, protocol), indexed by (split, run)."""
    s = u[(u.method == method) & (u.dataset == dataset) & (u.protocol == protocol)
          & (u.selector == SELECTOR)]
    if s.empty:
        raise SystemExit(f"no per-unit rows for {method}/{dataset}/{protocol}")
    p = s.pivot_table(index=["split_id", "run_id"], columns="state", values=["auc", "dp", "eo"])
    return pd.DataFrame({c: sgn * (p[(k, "M_plus_I")] - p[(k, "M_minus_I")])
                         for c, k, sgn in COORDS})


rng = np.random.default_rng(SEED)
rows = []
for method, ds in PAIRS:
    a, b = tau(method, ds, "controlled"), tau(method, ds, "native")
    idx = a.index.intersection(b.index)
    d = b.loc[idx] - a.loc[idx]
    cells = d.reset_index()[["split_id", "run_id"]].copy()
    for c, _, _ in COORDS:
        cells[c] = d[c].to_numpy()
    est = boot(cells, [c for c, _, _ in COORDS], rng, reps=REPS)   # one resample, all coordinates
    for j, (c, _, _) in enumerate(COORDS):
        lo, hi = (float(v) for v in np.percentile(est[:, j], [2.5, 97.5]))
        rows.append(dict(
            block="B", cell=f"{method}_{ds}", method=method, dataset=ds,
            pairing="controlled vs published horizon (re-execution)", coordinate=c,
            n_splits=int(d.reset_index().split_id.nunique()), n_matched_units=len(idx),
            tau_a=float(a.loc[idx, c].mean()), tau_b=float(b.loc[idx, c].mean()),
            delta=float(d[c].mean()), abs_delta=float(abs(d[c].mean())),
            max_unit_abs_delta=float(d[c].abs().max()),
            n_units_changed=int((d[c].abs() > 0).sum()),
            ci_low=lo, ci_high=hi, ci_excludes_zero=bool(lo > 0 or hi < 0),
            bootstrap_seed=SEED, bootstrap_reps=REPS))

out = pd.DataFrame(rows)
path = os.path.join(HERE, "c8_reexecution_delta.csv")
out.to_csv(path, index=False)

print(f"seed = {SEED}, replicates = {REPS}, selector = {SELECTOR}, rows = {len(out)}")
print("saved:", path)

n_excl = int(out.ci_excludes_zero.sum())
print("\n" + "=" * 78)
if n_excl:
    print(f"*** {n_excl} of {len(out)} BLOCK-B INTERVALS EXCLUDE ZERO ***")
    print("The estimator does not cover re-execution noise on these rows. This is a finding")
    print("about the paired hierarchical bootstrap, not about SFG.")
    print(out[out.ci_excludes_zero][["cell", "coordinate", "delta", "ci_low", "ci_high"]]
          .to_string(index=False))
else:
    print("No block-B interval excludes zero.")
print("=" * 78)

print("\nBlock B, per cell and coordinate:\n")
for cell in out.cell.unique():
    q = out[out.cell == cell]
    bits = []
    for c, _, _ in COORDS:
        r = q[q.coordinate == c].iloc[0]
        bits.append(f"{c} {r.delta:+.4f} [{r.ci_low:+.4f},{r.ci_high:+.4f}]"
                    f"{'*' if r.ci_excludes_zero else ' '}")
    print(f"  {cell:16} " + "  ".join(bits))
    print(f"  {'':16} units changed: "
          + ", ".join(f"{c} {int(q[q.coordinate == c].iloc[0].n_units_changed)}/30"
                      for c, _, _ in COORDS))

print("\nCaption bracket:")
print("  " + ("no interval excludes zero in either block" if not n_excl
              else "no interval excludes zero in the upper block"))

print("\nAgainst the block-A range quoted in the caption (-Delta_DP: 0.001 NIFTY/German "
      "to 0.067 SFG/German):")
for cell in out.cell.unique():
    v = float(out[(out.cell == cell) & (out.coordinate == "negDP")].abs_delta.iloc[0])
    where = "inside" if 0.0013 <= v <= 0.0668 else ("ABOVE" if v > 0.0668 else "below")
    print(f"  {cell:16} |Delta| on -Delta_DP = {v:.4f}   {where} the block-A range")
