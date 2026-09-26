"""How many replicates the study's own rule of thumb asks for, applied to the 36 primary cells.

`e0_noise_floor.py:29` states the rule the design was budgeted with:

    R >= (4 * sd_paired / d)^2

to resolve a paired difference d with a 95% interval of half-width d/2. Here d = 0.010, the
magnitude floor in the resolved rule, and sd_paired is the standard deviation of the 30 per-unit
values of tau_{-I->+I} in that cell.

**This is an approximation that ignores the design.** The 30 units are not 30 independent draws:
they are 6 splits x 5 runs, and a run shares its split with four others. Where split-level variance
dominates, the effective number of independent replicates is nearer 6 than 30, and an R computed as
if the units were independent is optimistic -- it asks for fewer replicates than the nested design
actually needs. The same intuition is why the paper's intervals come from a hierarchical bootstrap
that resamples splits first and runs within them, not from a flat one.

To make that gap visible rather than only stating it, the script also reports, per cell, the share
of variance that sits between splits (a one-way ANOVA on split_id) and the design effect it
implies, 1 + (m - 1) * ICC with m = 5 runs per split.

Reads results/per_unit_metrics.csv.gz and results/cell_results.csv (read-only). Writes
results/phase0_audit/r_needed.csv.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

D = 0.010                 # the magnitude floor in the resolved rule
COORDS = [("dAUC", "auc", 1), ("negDP", "dp", -1), ("negEO", "eo", -1)]
SELECTOR = "common_bce"

u = pd.read_csv(os.path.join(ROOT, "results", "per_unit_metrics.csv.gz"))
cells = pd.read_csv(os.path.join(ROOT, "results", "cell_results.csv"))
prim = cells[cells.count_in_primary_summary == True]          # noqa: E712


def tau_units(method, dataset, configuration):
    s = u[(u.method == method) & (u.dataset == dataset) & (u.configuration == configuration)
          & (u.protocol == "controlled") & (u.selector == SELECTOR)]
    p = s.pivot_table(index=["split_id", "run_id"], columns="state", values=["auc", "dp", "eo"])
    return pd.DataFrame({c: sgn * (p[(k, "M_plus_I")] - p[(k, "M_minus_I")])
                         for c, k, sgn in COORDS})


def icc(v, split):
    """Between-split share of variance, one-way random effects, clipped at 0."""
    g = pd.DataFrame({"v": v, "s": split})
    k = g.groupby("s").v
    n_g, m = k.ngroups, g.groupby("s").size().mean()
    if n_g < 2:
        return np.nan
    grand = g.v.mean()
    msb = sum(len(x) * (x.mean() - grand) ** 2 for _, x in k) / (n_g - 1)
    msw_num = sum(((x - x.mean()) ** 2).sum() for _, x in k)
    dfw = len(g) - n_g
    if dfw <= 0:
        return np.nan
    msw = msw_num / dfw
    var_b = (msb - msw) / m
    tot = var_b + msw
    return float(max(var_b, 0.0) / tot) if tot > 0 else 0.0


rows = []
for r in prim.itertuples():
    t = tau_units(r.method, r.dataset, r.configuration)
    idx = t.reset_index()
    for c, _, _ in COORDS:
        v = t[c].to_numpy()
        sd = float(np.std(v, ddof=1))
        r_need = int(np.ceil((4 * sd / D) ** 2))
        rho = icc(v, idx.split_id.to_numpy())
        deff = 1 + (5 - 1) * rho if not np.isnan(rho) else np.nan
        rows.append(dict(
            method=r.method, dataset=r.dataset, configuration=r.configuration, coordinate=c,
            n_units=len(v), tau_I=float(np.mean(v)), sd_paired=sd,
            resolved=str(getattr(r, f"tau_I_{c}_resolved")).strip().lower() == "true",
            R_needed=r_need, exceeds_30=bool(r_need > 30),
            icc_split=rho, design_effect=deff,
            R_needed_design_adjusted=(int(np.ceil(r_need * deff)) if not np.isnan(deff) else None)))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "r_needed.csv"), index=False)

print(f"R >= (4 * sd_paired / d)^2  with d = {D}, sd from the 30 per-unit tau_I values")
print(f"{len(out)} rows = 36 primary cells x 3 coordinates\n")

print(f"{'coordinate':<9}{'unresolved':>11}{'  of which R>30':>16}{'median R':>10}"
      f"{'min':>7}{'max':>9}")
for c, _, _ in COORDS:
    q = out[out.coordinate == c]
    un = q[~q.resolved]
    print(f"{c:<9}{len(un):>11}{int(un.exceeds_30.sum()):>16}"
          f"{int(un.R_needed.median()):>10}{int(un.R_needed.min()):>7}{int(un.R_needed.max()):>9}")

print("\nresolved cells that still ask for more than 30:")
any_ = False
for c, _, _ in COORDS:
    q = out[(out.coordinate == c) & out.resolved & out.exceeds_30]
    if len(q):
        any_ = True
        print(f"  {c}: {len(q)} of {int(out[(out.coordinate == c) & out.resolved].shape[0])}")
        for x in q.itertuples():
            print(f"      {x.method}/{x.dataset}[{x.configuration}]  "
                  f"tau_I {x.tau_I:+.4f}  sd {x.sd_paired:.4f}  R_needed {x.R_needed}")
if not any_:
    print("  none")

print("\nWhole set, both criteria:")
for c, _, _ in COORDS:
    q = out[out.coordinate == c]
    print(f"  {c:<7} R>30 in {int(q.exceeds_30.sum()):>2}/36 cells; "
          f"median R {int(q.R_needed.median()):>4}; "
          f"median ICC(split) {q.icc_split.median():.2f}; "
          f"median design effect {q.design_effect.median():.2f}")

print("\nThe design-adjusted column multiplies R by 1 + (m-1)*ICC, m = 5 runs per split. It is a "
      "correction for the nesting the formula ignores, not a second rule:")
for c, _, _ in COORDS:
    q = out[out.coordinate == c]
    print(f"  {c:<7} median R {int(q.R_needed.median()):>4} -> "
          f"{int(q.R_needed_design_adjusted.median()):>5} adjusted; "
          f"cells over 30: {int(q.exceeds_30.sum())} -> "
          f"{int((q.R_needed_design_adjusted > 30).sum())}")
print("\nsaved:", os.path.join(HERE, "r_needed.csv"))
