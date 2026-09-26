"""The B stage of the FMP component chain, with the FMP analysis's own machinery.

Reads the frozen FMP arms (harness/results/x27/x27_<ds>.csv) and the X31 baseline
(harness/results/x31/x31_fmp_B_<ds>.csv), pairs them by (split, run, selector),
and summarises, per selector:

    base.<c>               F00 - B          the baseline -> base step
    pkg.l1<a>.l2<b>.<c>    F11(a, b) - B    the whole chain, B -> +propagation+fairness

with the identity pkg = base + total checked per unit and in every replicate
(total = F11 - F00 is the frozen quantity, recomputed here only for the check).
Outcome orientation, `stat()`, the bootstrap `boot()` and the 10,000 replicates
are imported from analyze_x27_fmp unchanged; the generator is seeded with that
analysis's seed. Existing prop / fair / total summaries are not touched.

    python harness/experiments/analyze_x31_fmp_baseline.py --dataset pokec_z
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))

import analyze_x27_fmp as A                     # noqa: E402

RES = os.path.join(ROOT, "harness", "results")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("pokec_z", "pokec_n"))
    a = ap.parse_args()
    fmp = A.load(a.dataset, os.path.join(RES, "x27", f"x27_{a.dataset}.csv"))
    b = pd.read_csv(os.path.join(RES, "x31", f"x31_fmp_B_{a.dataset}.csv"))
    if b.groupby(["split_id", "run_id"]).ngroups != 30 or len(b) != 30 * len(A.SELECTORS):
        raise SystemExit(f"[x31] baseline store for {a.dataset} is incomplete (refusing)")
    if b.duplicated(["split_id", "run_id", "selector"]).any():
        raise SystemExit("[x31] duplicate baseline keys")
    if not np.isfinite(b[["auc", "dp", "eo"]].to_numpy(float)).all():
        raise SystemExit("[x31] non-finite baseline outcome")

    rng = np.random.default_rng(A.SEED)
    rows = []
    for sel in A.SELECTORS:
        g = fmp[fmp.selector == sel]
        bb = b[b.selector == sel].sort_values(["split_id", "run_id"])
        ybase = A.outcome_cols(bb)
        wide = {}
        for cfg, gg in g.groupby("config"):
            gg = gg.sort_values(["split_id", "run_id"])
            if not (np.array_equal(gg.split_id.to_numpy(), bb.split_id.to_numpy())
                    and np.array_equal(gg.run_id.to_numpy(), bb.run_id.to_numpy())):
                raise SystemExit(f"[x31] units of {cfg} do not match the baseline units")
            y = A.outcome_cols(gg)
            for c in A.COORDS:
                wide[(cfg, c)] = y[c].to_numpy()
        t = dict(split_id=bb.split_id.to_numpy(), run_id=bb.run_id.to_numpy())
        for c in A.COORDS:
            t[f"base.{c}"] = wide[("F00", c)] - ybase[c].to_numpy()
        for l2 in A.LAM2:
            for l1 in A.LAM1:
                f11 = A.cfg_name(l1, l2)
                for c in A.COORDS:
                    t[f"pkg.l1{l1:g}.l2{l2:g}.{c}"] = wide[(f11, c)] - ybase[c].to_numpy()
                    t[f"_total.l1{l1:g}.l2{l2:g}.{c}"] = wide[(f11, c)] - wide[("F00", c)]
        t = pd.DataFrame(t)
        cols = [c for c in t.columns if c not in ("split_id", "run_id")]
        reps = A.boot(t, cols, rng, reps=A.B_REPS)
        ci = {c: i for i, c in enumerate(cols)}
        for l2 in A.LAM2:
            for l1 in A.LAM1:
                for c in A.COORDS:
                    k = f"l1{l1:g}.l2{l2:g}.{c}"
                    for get in (lambda z: t[z].to_numpy(), lambda z: reps[:, ci[z]]):
                        r = np.max(np.abs(get(f"pkg.{k}") - (get(f"base.{c}") + get(f"_total.{k}"))))
                        if not r <= A.TOL:
                            raise SystemExit(f"[x31] identity pkg = base + total failed on {k}: {r}")
        for c in cols:
            if c.startswith("_"):
                continue
            rows.append(dict(selector=sel, quantity=c, **A.stat(t[c], reps[:, ci[c]])))
        print(f"[x31] {a.dataset} {sel}: pkg = base + total holds per unit and in all "
              f"{A.B_REPS:,} replicates")
    out = os.path.join(RES, "x31", f"x31_fmp_B_{a.dataset}_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
