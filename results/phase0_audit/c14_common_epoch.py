"""C-14 (the old T10): what changes if the two arms are read at one shared epoch.

Today each arm is checkpointed at its own validation-BCE minimum. The rule below was fixed in
STEP_C_PLAN before any result was seen:

    common epoch = argmin_e [ (val_bce_{M+I}(e) + val_bce_{M-I}(e)) / 2 ],  ties to the smaller e,
    restricted to the stored grid where a cell's test outcome exists only on a grid.

Both arms are then read at that one epoch and tau_{-I->+I} is formed on all three coordinates.

**No primary cell can answer this.** The controlled H = 200 runs stored no per-epoch validation
loss (T10, re-checked 2026-09-24), so for all 36 the comparison is [확인 불가] from stored
artifacts. What exists is three case-study cells, all at their published horizon:

    NIFTY/German D0, D1   x25/R10,R11_trajectories   per-epoch test scores, 1001 epochs
    FairGB/Bail           x26/bail_trajectories      test scores on an 11-epoch grid

The independent-selection reference is recomputed from the same files by the same evaluator, so
the two columns differ in the selection rule and nothing else.

Writes results/phase0_audit/c14_common_epoch.csv.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
STUDY = os.path.join(ROOT, "harness", "results")
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))
from core.evaluator import evaluate      # noqa: E402
from bootstrap_armA import SEED, boot    # noqa: E402

REPS = 10_000
COORDS = [("dAUC", "auc", 1), ("negDP", "dp", -1), ("negEO", "eo", -1)]


def outcome(y, a, score):
    m = evaluate(np.asarray(y).astype(int), np.asarray(a).astype(int),
                 raw_score=np.asarray(score), decision="score>0")
    return {"auc": m["auc"], "dp": m["dp"], "eo": m["eo"]}


def units_x25(sub):
    """NIFTY/German: val_bce per epoch and test_raw at every epoch."""
    out = []
    for p in sorted(glob.glob(os.path.join(STUDY, "x25", f"{sub}_trajectories", "*_M1.npz"))):
        q = p.replace("_M1.npz", "_M0.npz")
        z1, z0 = np.load(p, allow_pickle=True), np.load(q, allow_pickle=True)
        out.append(dict(split=int(z1["split"]), seed=int(z1["seed"]),
                        vb1=z1["val_bce"], vb0=z0["val_bce"],
                        ep=z1["epochs"], grid=None,
                        s1=z1["test_raw"], s0=z0["test_raw"],
                        y=z1["test_y"], a=z1["test_a"],
                        sel1=int(z1["slot_bce"]), sel0=int(z0["slot_bce"])))
    return out


def units_x26():
    """FairGB/Bail: val_bce per epoch, but test scores only on an 11-epoch grid."""
    out = []
    for p in sorted(glob.glob(os.path.join(STUDY, "x26", "bail_trajectories", "*_M1.npz"))):
        q = p.replace("_M1.npz", "_M0.npz")
        z1, z0 = np.load(p, allow_pickle=True), np.load(q, allow_pickle=True)
        out.append(dict(split=int(z1["split"]), seed=int(z1["seed"]),
                        vb1=z1["val_bce"], vb0=z0["val_bce"],
                        ep=z1["epochs"], grid=np.asarray(z1["grid_epochs"]),
                        s1=z1["grid_scores"], s0=z0["grid_scores"],
                        y=z1["test_y"], a=z1["test_a"],
                        sel1=int(z1["slot_epoch_full_bce"]), sel0=int(z0["slot_epoch_full_bce"])))
    return out


def tau_rows(units, label, grid_restricted):
    rows = []
    for u in units:
        ep = np.asarray(u["ep"])
        mean_bce = (np.asarray(u["vb1"]) + np.asarray(u["vb0"])) / 2.0
        if u["grid"] is None:
            k = int(np.argmin(mean_bce))              # ties -> smallest epoch, argmin's behaviour
            ce = int(ep[k])
            g1, g0 = u["s1"][k], u["s0"][k]
            i1 = int(np.where(ep == u["sel1"])[0][0])
            i0 = int(np.where(ep == u["sel0"])[0][0])
            a1, a0 = u["s1"][i1], u["s0"][i0]
        else:
            gi = np.asarray([int(np.where(ep == e)[0][0]) for e in u["grid"]])
            k = int(np.argmin(mean_bce[gi]))
            ce = int(u["grid"][k])
            g1, g0 = u["s1"][k], u["s0"][k]
            # independent selection is the stored full-support choice; its scores are not on the
            # grid, so the reference for this cell is read at the nearest stored grid epoch
            j1 = int(np.argmin(np.abs(u["grid"] - u["sel1"])))
            j0 = int(np.argmin(np.abs(u["grid"] - u["sel0"])))
            a1, a0 = u["s1"][j1], u["s0"][j0]
        com = {c: sgn * (outcome(u["y"], u["a"], g1)[k_] - outcome(u["y"], u["a"], g0)[k_])
               for c, k_, sgn in COORDS}
        ind = {c: sgn * (outcome(u["y"], u["a"], a1)[k_] - outcome(u["y"], u["a"], a0)[k_])
               for c, k_, sgn in COORDS}
        rows.append(dict(cell=label, split_id=u["split"], run_id=u["seed"] - 27,
                         common_epoch=ce, sel_M1=u["sel1"], sel_M0=u["sel0"],
                         grid_restricted=grid_restricted,
                         **{f"common_{c}": com[c] for c, _, _ in COORDS},
                         **{f"indep_{c}": ind[c] for c, _, _ in COORDS}))
    return rows


def summarise(d, label, rng):
    out = []
    for kind in ("indep", "common"):
        cells = d[["split_id", "run_id"]].copy()
        for c, _, _ in COORDS:
            cells[c] = d[f"{kind}_{c}"].to_numpy()
        est = boot(cells, [c for c, _, _ in COORDS], rng, reps=REPS)
        for j, (c, _, _) in enumerate(COORDS):
            lo, hi = (float(v) for v in np.percentile(est[:, j], [2.5, 97.5]))
            v = d[f"{kind}_{c}"]
            sign = float(max((v > 0).mean(), (v < 0).mean()))
            out.append(dict(cell=label, selection=kind, coordinate=c, n_units=len(d),
                            tau_I=float(v.mean()), ci_low=lo, ci_high=hi,
                            sign_stability=sign,
                            resolved=bool(sign >= 0.75 and abs(v.mean()) >= 0.010
                                          and (lo > 0 or hi < 0))))
    return out


def main():
    sets = [("NIFTY/German D0", units_x25("R10"), False),
            ("NIFTY/German D1", units_x25("R11"), False),
            ("FairGB/Bail", units_x26(), True)]
    per_unit, summ = [], []
    rng = np.random.default_rng(SEED)
    for label, units, gr in sets:
        if not units:
            print(f"[c14] no trajectories for {label}")
            continue
        rows = tau_rows(units, label, gr)
        d = pd.DataFrame(rows)
        per_unit += rows
        summ += summarise(d, label, rng)
    pu = pd.DataFrame(per_unit)
    sm = pd.DataFrame(summ)
    pu.to_csv(os.path.join(HERE, "c14_common_epoch_units.csv"), index=False)
    sm.to_csv(os.path.join(HERE, "c14_common_epoch.csv"), index=False)

    print("Cells with a stored trajectory: 3 of 3 case-study cells. "
          "Primary cells answerable: 0 of 36 (no per-epoch record under the controlled protocol).")
    print(f"\nunits: {pu.groupby('cell').size().to_dict()}")
    print("\nselected epochs vs the common epoch, per cell (median over units):")
    for cell, g in pu.groupby("cell"):
        print(f"  {cell:18} M+I {g.sel_M1.median():>7.1f}   M-I {g.sel_M0.median():>7.1f}   "
              f"common {g.common_epoch.median():>7.1f}"
              + ("   [grid-restricted]" if g.grid_restricted.iloc[0] else ""))

    print("\ntau_I under the two selection rules:\n")
    for cell in sm.cell.unique():
        print(f"  {cell}")
        for c, _, _ in COORDS:
            a = sm[(sm.cell == cell) & (sm.coordinate == c) & (sm.selection == "indep")].iloc[0]
            b = sm[(sm.cell == cell) & (sm.coordinate == c) & (sm.selection == "common")].iloc[0]
            flip = "SIGN FLIP" if (a.tau_I > 0) != (b.tau_I > 0) else ""
            res = "RESOLVED CHANGE" if a.resolved != b.resolved else ""
            print(f"    {c:6} independent {a.tau_I:+.4f} [{a.ci_low:+.4f},{a.ci_high:+.4f}]"
                  f"{'*' if a.resolved else ' '}   common {b.tau_I:+.4f} "
                  f"[{b.ci_low:+.4f},{b.ci_high:+.4f}]{'*' if b.resolved else ' '}   "
                  f"diff {b.tau_I - a.tau_I:+.4f}  {flip} {res}")
        print()

    diffs = []
    for cell in sm.cell.unique():
        for c, _, _ in COORDS:
            a = sm[(sm.cell == cell) & (sm.coordinate == c) & (sm.selection == "indep")].iloc[0]
            b = sm[(sm.cell == cell) & (sm.coordinate == c) & (sm.selection == "common")].iloc[0]
            diffs.append(dict(cell=cell, coordinate=c, diff=b.tau_I - a.tau_I,
                              sign_flip=(a.tau_I > 0) != (b.tau_I > 0),
                              resolved_change=a.resolved != b.resolved))
    dd = pd.DataFrame(diffs)
    print(f"sign flips: {int(dd.sign_flip.sum())} of {len(dd)} cell-coordinates")
    print(f"resolved changes: {int(dd.resolved_change.sum())} of {len(dd)}")
    print(f"|difference|: median {dd['diff'].abs().median():.4f}, "
          f"max {dd['diff'].abs().max():.4f}")


if __name__ == "__main__":
    main()
