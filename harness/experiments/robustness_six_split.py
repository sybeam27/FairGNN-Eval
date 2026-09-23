"""Appendix robustness: does the resolved classification survive a six-cluster analysis?

The frozen primary analysis resamples the 6 splits and then the 5 runs inside each split (10,000
replicates). The top level has only 6 clusters, so this script re-does the same comparison with two
deliberately conservative alternatives that use nothing but those 6 numbers:

    split-level t      average the 5 matched runs inside each split -> 6 split-level estimates;
                       interval = mean +- t_{0.975,5} * sd / sqrt(6)
    leave-one-split-out  drop each split in turn and repeat on the remaining 5 (t_{0.975,4})

Resolved under the frozen rule = sign stability >= 0.75 AND |mean| >= 0.01 AND the 95% interval
excludes 0. Here the interval changes and nothing else: the magnitude floor is kept, and sign
stability is recomputed across splits (share of split-level estimates with the sign of the mean).

Reads the raw store (untouched) and results/1_main_package_vs_intervention.csv for the frozen
classification. Writes results/appendix_robustness/.

    python harness/experiments/robustness_six_split.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "results", "appendix_robustness")
sys.path.insert(0, HERE)
import build_results as B          # noqa: E402  (the frozen loader)

COORDS = {"dAUC": "int_dauc", "negDP": "int_ndp", "negEO": "int_neo"}
MAG, SIGN = 0.010, 0.75                  # the frozen rule's magnitude floor and sign stability


def store_rows(d, cell):
    """The store writes the configuration into the method name (BIND-1pct, FairSIN-GCN, ...)."""
    cfg = str(cell.configuration)
    for key in ([f"{cell.method}-{cfg}"] if cfg != "default" else []) + [cell.method]:
        g = d[(d.method == key) & (d.dataset == cell.dataset)]
        if not g.empty:
            return key, g
    raise SystemExit(f"no store rows for {cell.method} ({cfg}) / {cell.dataset}")


def interval(split_means: np.ndarray):
    """mean and a t interval over the split-level estimates (the only independent level)."""
    n = len(split_means)
    m = float(np.mean(split_means))
    sd = float(np.std(split_means, ddof=1))
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    stab = float(np.mean(np.sign(split_means) == np.sign(m))) if m != 0 else 0.0
    return m, m - half, m + half, stab


def resolved(m, lo, hi, stab, use_sign=True):
    return bool(abs(m) >= MAG and lo * hi > 0 and (stab >= SIGN or not use_sign))


def main():
    os.makedirs(OUT, exist_ok=True)
    store = B.load_store()
    d = store[(store.protocol == "controlled") & (store.selector == "common_bce")]
    main_csv = pd.read_csv(os.path.join(ROOT, "results", "1_main_package_vs_intervention.csv"))
    as_true = lambda s: s.astype(str).str.strip().str.lower().eq("true")

    rows = []
    for _, cell in main_csv.iterrows():
        key, g = store_rows(d, cell)
        if g.groupby(["split_id", "run_id"]).ngroups != len(g):
            raise SystemExit(f"duplicate units for {key} / {cell.dataset}")
        for coord, col in COORDS.items():
            per_split = g.groupby("split_id")[col].mean().sort_index()
            sm = per_split.to_numpy()
            m, lo, hi, stab = interval(sm)
            froz_m = float(cell[f"tau_I_{coord}_mean"])
            froz_r = bool(as_true(pd.Series([cell[f"tau_I_{coord}_resolved"]]))[0])
            # leave-one-split-out: drop each split, re-evaluate on the remaining five
            loso = []
            for k in range(len(sm)):
                mk, lok, hik, stabk = interval(np.delete(sm, k))
                loso.append(resolved(mk, lok, hik, stabk))
            rows.append(dict(
                method=cell.method, dataset=cell.dataset, configuration=cell.configuration,
                coordinate=coord, n_splits=len(sm), n_units=len(g),
                frozen_mean=froz_m, frozen_resolved=froz_r,
                split_mean=m, split_lo=lo, split_hi=hi, split_sign_stability=stab,
                t_resolved=resolved(m, lo, hi, stab),
                t_resolved_ci_only=resolved(m, lo, hi, stab, use_sign=False),
                loso_resolved_count=int(sum(loso)), loso_all_resolved=bool(all(loso)),
                loso_none_resolved=bool(not any(loso)),
                mean_shift=m - froz_m))
    r = pd.DataFrame(rows)
    assert (r.n_splits == 6).all() and (r.n_units == 30).all()
    assert r.mean_shift.abs().max() < 1e-9, r.mean_shift.abs().max()   # same point estimate
    r.to_csv(os.path.join(OUT, "six_split_sensitivity_cells.csv"), index=False)

    summ = []
    for coord in COORDS:
        s = r[r.coordinate == coord]
        agree = int((s.frozen_resolved == s.t_resolved).sum())
        summ.append(dict(
            coordinate=coord, cells=len(s),
            frozen_resolved=int(s.frozen_resolved.sum()),
            split_t_resolved=int(s.t_resolved.sum()),
            split_t_resolved_ci_only=int(s.t_resolved_ci_only.sum()),
            kept=int((s.frozen_resolved & s.t_resolved).sum()),
            lost=int((s.frozen_resolved & ~s.t_resolved).sum()),
            gained=int((~s.frozen_resolved & s.t_resolved).sum()),
            agreement=agree,
            loso_stable_of_frozen_resolved=int((s.frozen_resolved & (s.loso_resolved_count == 6)).sum()),
            loso_never_of_frozen_unresolved=int((~s.frozen_resolved & (s.loso_resolved_count == 0)).sum())))
    su = pd.DataFrame(summ)
    su.to_csv(os.path.join(OUT, "six_split_sensitivity_summary.csv"), index=False)
    print(su.to_string(index=False))
    print()
    for coord in COORDS:
        s = r[r.coordinate == coord]
        ch = s[s.frozen_resolved != s.t_resolved]
        if len(ch):
            print(f"{coord}: classification changes ->")
            for _, x in ch.iterrows():
                print(f"   {x.method}/{x.dataset} ({x.configuration}) frozen={x.frozen_resolved} "
                      f"t={x.t_resolved}  mean {x.split_mean:+.4f}  t-CI [{x.split_lo:+.4f}, {x.split_hi:+.4f}] "
                      f"sign {x.split_sign_stability:.2f}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
