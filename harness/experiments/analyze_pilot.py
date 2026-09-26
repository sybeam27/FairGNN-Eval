"""Read the balanced pilot. Two questions, in order.

    Q1  Do package gain and intervention gain differ systematically?
    Q2  Does the intervention attribution survive split, run, and a change of
        checkpoint selector?

Three vectors per method, oriented so larger is better in both coordinates:

    tau_pkg        published configuration and selector, outcome by G_c
    tau_int^BCE    M1 - M0 under the primary shared selector
    tau_int^AUC    the same under the pre-registered robustness selector

**Distributions before means.** Per-split and per-run spread is printed first,
and the overall mean last, because in this study the mean has repeatedly been
smaller than the spread it came from. A method whose per-split effects change
sign has no overall mean worth reading.

Nothing here computes a confidence interval. 3 splits x 5 runs gives direction,
magnitude and stability of sign; it does not give an interval and none is
reported.

Usage
-----
    python harness/experiments/analyze_pilot.py --csv harness/results/pilot_tau_3x5.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
PRIM = ("dauc", "ndp")          # [ΔAUC, −ΔDP]
SEC = ("dauc", "neo")           # [ΔAUC, −ΔEO]


def vecs(g: pd.DataFrame, pre: str, coords=PRIM):
    return g[f"{pre}_{coords[0]}"].to_numpy(), g[f"{pre}_{coords[1]}"].to_numpy()


def fmt(u, v) -> str:
    return f"[{np.mean(u):+.4f}, {np.mean(v):+.4f}]"


def sign_stability(x: np.ndarray) -> str:
    """How often the sign agrees with the modal sign. 1.00 means never flips."""
    s = np.sign(x[np.isfinite(x)])
    s = s[s != 0]
    if s.size == 0:
        return "n/a"
    frac = max((s > 0).mean(), (s < 0).mean())
    return f"{frac:.2f}"


def per_method(df: pd.DataFrame, meth: str, coords=PRIM) -> None:
    bce = df[(df.method == meth) & (df.selector == "common_bce")]
    auc = df[(df.method == meth) & (df.selector == "common_auc")]
    if bce.empty:
        return
    print(f"\n{'=' * 86}\n{meth}\n{'=' * 86}")

    print("  per split  (mean over runs within the split)")
    print(f"    {'split':>6s} {'n':>3s} {'tau_pkg':>21s} {'tau_int BCE':>21s} {'tau_int AUC':>21s}")
    for sp in sorted(bce.split_id.unique()):
        b, a = bce[bce.split_id == sp], auc[auc.split_id == sp]
        print(f"    {sp:6d} {len(b):3d} {fmt(*vecs(b, 'pkg', coords)):>21s} "
              f"{fmt(*vecs(b, 'int', coords)):>21s} "
              f"{fmt(*vecs(a, 'int', coords)) if len(a) else 'n/a':>21s}")

    print("\n  per run  (mean over splits within the run index)")
    print(f"    {'run':>6s} {'n':>3s} {'tau_pkg':>21s} {'tau_int BCE':>21s}")
    for r in sorted(bce.run_id.unique()):
        b = bce[bce.run_id == r]
        print(f"    {r:6d} {len(b):3d} {fmt(*vecs(b, 'pkg', coords)):>21s} "
              f"{fmt(*vecs(b, 'int', coords)):>21s}")

    print("\n  spread and sign stability  (1.00 = the sign never flips)")
    for name, g, pre in (("tau_pkg", bce, "pkg"), ("tau_int BCE", bce, "int"),
                         ("tau_int AUC", auc, "int")):
        if g.empty:
            continue
        u, v = vecs(g, pre, coords)
        # between-split spread against within-split spread, per coordinate
        bs = g.groupby("split_id")[f"{pre}_{coords[1]}"].mean().std(ddof=1)
        ws = g.groupby("split_id")[f"{pre}_{coords[1]}"].std(ddof=1).mean()
        print(f"    {name:12s} mean {fmt(u, v)}   sd [{np.std(u, ddof=1):.4f}, "
              f"{np.std(v, ddof=1):.4f}]   sign [{sign_stability(u)}, "
              f"{sign_stability(v)}]   split_sd/run_sd = "
              f"{bs / ws if ws > 0 else float('nan'):.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(RESULTS, "pilot_tau_3x5.csv"))
    ap.add_argument("--secondary", action="store_true",
                    help="report [dAUC, -dEO] instead of [dAUC, -dDP]")
    a = ap.parse_args()
    if not os.path.exists(a.csv):
        sys.exit(f"not run yet: {a.csv}")
    df = pd.read_csv(a.csv)
    coords = SEC if a.secondary else PRIM

    cells = df.groupby(["method", "split_id", "run_id"]).ngroups
    print(f"\n{len(df)} rows, {cells} (method, split, run) cells, "
          f"{df.split_id.nunique()} splits x {df.run_id.nunique()} runs")
    print(f"coordinates: [dAUC, -{'dEO' if a.secondary else 'dDP'}], "
          "larger is better in both")
    nud = int((~df.eo_defined).sum()) if "eo_defined" in df else 0
    if nud:
        print(f"EO undefined in {nud} rows; reported, never dropped from a mean")

    for meth in sorted(df.method.unique()):
        per_method(df, meth, coords)

    print(f"\n{'=' * 86}")
    print("Overall means, read last and only after the distributions above")
    print("=" * 86)
    print(f"  {'method':10s} {'tau_pkg':>21s} {'tau_int BCE':>21s} {'tau_int AUC':>21s}")
    for meth in sorted(df.method.unique()):
        b = df[(df.method == meth) & (df.selector == "common_bce")]
        au = df[(df.method == meth) & (df.selector == "common_auc")]
        print(f"  {meth:10s} {fmt(*vecs(b, 'pkg', coords)):>21s} "
              f"{fmt(*vecs(b, 'int', coords)):>21s} "
              f"{fmt(*vecs(au, 'int', coords)) if len(au) else 'n/a':>21s}")
    print("\nNo confidence interval is computed. Three splits and five runs give "
          "direction,\nmagnitude and whether the sign holds -- not an interval.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
