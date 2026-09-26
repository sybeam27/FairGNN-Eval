"""X31 section 3b analysis: FnRGNN on node regression.

Per matched unit, oriented so that larger is better:
    negMSE = -MSE, negMeanGap = -mean_gap, negWD = -Wasserstein
    tau_nonint = Y(M^-I) - Y(B), tau_I = Y(M^+I) - Y(M^-I), tau_pkg = Y(M^+I) - Y(B)

Summarised with the unchanged paired hierarchical bootstrap (bootstrap_armA.boot,
10,000 replicates, splits then runs, seed bootstrap_armA.SEED) and the unchanged
resolution rule (analyze_armA.SIGN_MIN, NEAR_ZERO, 95% CI excluding 0). The
identity tau_pkg = tau_nonint + tau_I is checked per unit and in every replicate.

    python harness/experiments/analyze_x31_fnrgnn_regression.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))

from analyze_armA import NEAR_ZERO, SIGN_MIN, sign_stability     # noqa: E402
from bootstrap_armA import SEED, boot                             # noqa: E402

RES = os.path.join(ROOT, "harness", "results", "x31")
DATASETS = ("german", "pokec_z", "pokec_n")
COORDS = (("negMSE", "mse"), ("negMeanGap", "mean_gap"), ("negWD", "wd"))
TOL = 1e-12


def main() -> int:
    rows = []
    rng = np.random.default_rng(SEED)
    for ds in DATASETS:
        d = pd.read_csv(os.path.join(RES, f"x31_FnRGNN-regression_{ds}.csv"))
        if d.groupby(["split_id", "run_id"]).ngroups != 30 or len(d) != 30:
            raise SystemExit(f"[x31] {ds}: store incomplete (refusing)")
        t = d[["split_id", "run_id"]].copy()
        cols = []
        for c, k in COORDS:
            yb, y0, y1 = -d[f"b_{k}"], -d[f"m0_{k}"], -d[f"m1_{k}"]
            t[f"tau_nonint_{c}"], t[f"tau_I_{c}"], t[f"tau_pkg_{c}"] = y0 - yb, y1 - y0, y1 - yb
            cols += [f"tau_nonint_{c}", f"tau_I_{c}", f"tau_pkg_{c}"]
        if not np.isfinite(t[cols].to_numpy()).all():
            raise SystemExit(f"[x31] {ds}: non-finite contrast")
        reps = boot(t, cols, rng)
        ci = {c: i for i, c in enumerate(cols)}
        out = dict(method="FnRGNN", dataset=ds, configuration="regression (released task)",
                   selector="validation MSE", n_units=len(t))
        for c, _ in COORDS:
            for get in (lambda z: t[z].to_numpy(), lambda z: reps[:, ci[z]]):
                r = np.max(np.abs(get(f"tau_pkg_{c}") - get(f"tau_nonint_{c}") - get(f"tau_I_{c}")))
                if not r <= TOL:
                    raise SystemExit(f"[x31] {ds}: identity failed on {c} ({r})")
            for q in ("tau_pkg", "tau_nonint", "tau_I"):
                col = f"{q}_{c}"
                mu = float(t[col].mean())
                lo, hi = np.percentile(reps[:, ci[col]], [2.5, 97.5])
                out[f"{col}_mean"], out[f"{col}_lo"], out[f"{col}_hi"] = mu, float(lo), float(hi)
                if q == "tau_I":
                    sg = float(sign_stability(t[col]))
                    out[f"{col}_sign_stability"] = sg
                    out[f"{col}_resolved"] = bool(sg >= SIGN_MIN and abs(mu) >= NEAR_ZERO
                                                  and lo * hi > 0)
        rows.append(out)
        print(f"[x31] {ds}: identity holds per unit and in all {len(reps):,} replicates")
    p = os.path.join(RES, "x31_FnRGNN-regression_summary.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"[written] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
