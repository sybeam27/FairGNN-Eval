"""Paired hierarchical bootstrap for Arm A, per the plan frozen in X8.

One replicate, for one (method, dataset):

    1. resample the splits with replacement;
    2. within each drawn split, resample its runs with replacement;
    3. carry each drawn cell whole -- its B, M0, M1 and its BCE and AUC
       readings stay paired exactly as recorded.

The cell is the resampling unit. No arm is resampled on its own, so a
difference formed inside a cell is never broken across replicates.

10,000 replicates, 95% percentile intervals, for tau_base^audit, tau_int,
tau_pkg^audit and D_selector on both coordinates.

Zero-effect cells enter at their measured value. Nothing is dropped.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))
from analyze_armA import build, fmt                       # noqa: E402

B = 10_000
SEED = 20260914


def cell_table(d, method, dataset):
    """One row per (split, run): the paired quantities of that cell."""
    g = d[(d.method == method) & (d.dataset == dataset)]
    w = g.pivot_table(index=["split_id", "run_id"], columns="selector",
                      values=["abase_auc", "abase_ndp", "int_auc", "int_ndp",
                              "apkg_auc", "apkg_ndp"])
    out = pd.DataFrame(index=w.index)
    for q in ("abase", "int", "apkg"):
        for c in ("auc", "ndp"):
            out[f"{q}_{c}"] = w[(f"{q}_{c}", "common_bce")]
    out["D_auc"] = w[("int_auc", "common_bce")] - w[("int_auc", "common_auc")]
    out["D_ndp"] = w[("int_ndp", "common_bce")] - w[("int_ndp", "common_auc")]
    return out.reset_index()


def boot(cells, cols, rng, reps=B):
    splits = cells.split_id.unique()
    by = {s: cells[cells.split_id == s][cols].to_numpy() for s in splits}
    ns = len(splits)
    out = np.empty((reps, len(cols)))
    for b in range(reps):
        drawn = rng.integers(0, ns, ns)
        acc = []
        for j in drawn:
            m = by[splits[j]]
            acc.append(m[rng.integers(0, len(m), len(m))])
        out[b] = np.concatenate(acc, axis=0).mean(axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    a = ap.parse_args()
    d = build(a.csv)
    cols = ["abase_auc", "abase_ndp", "int_auc", "int_ndp",
            "apkg_auc", "apkg_ndp", "D_auc", "D_ndp"]
    rng = np.random.default_rng(SEED)

    print(f"Paired hierarchical bootstrap, {B:,} replicates, seed {SEED}.")
    print("Resampling unit is the (split, run) cell; B/M0/M1 and BCE/AUC stay "
          "paired inside it.")
    print("95% percentile intervals. Three datasets is not a sample of "
          "datasets, so no interval is\nattached to any cross-dataset mean.\n")

    for ds_ in sorted(d.dataset.unique()):
        print("=" * 104)
        print(ds_)
        print("=" * 104)
        print(f"{'method':<10}{'quantity':<16}{'coord':<7}{'mean':>10}"
              f"{'95% interval':>26}{'excludes 0':>12}")
        for m in sorted(d.method.unique()):
            cells = cell_table(d, m, ds_)
            if cells.empty:
                continue
            reps = boot(cells, cols, rng)
            for q, lab in (("abase", "tau_base^audit"), ("int", "tau_int"),
                           ("apkg", "tau_pkg^audit"), ("D", "D_selector")):
                for c, cl in (("auc", "dAUC"), ("ndp", "-dDP")):
                    k = cols.index(f"{q}_{c}")
                    lo, hi = np.percentile(reps[:, k], [2.5, 97.5])
                    mu = cells[f"{q}_{c}"].mean()
                    print(f"{m:<10}{lab:<16}{cl:<7}{mu:>+10.4f}"
                          f"{f'[{lo:+.4f}, {hi:+.4f}]':>26}"
                          f"{'yes' if lo * hi > 0 else 'no':>12}")
            print(f"{'':<10}n cells = {len(cells)}")


if __name__ == "__main__":
    main()
