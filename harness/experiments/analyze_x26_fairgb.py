"""X26 analysis: FairGB selection-support replication on Bail and Credit (prereg 630c82a).

Per dataset, inputs are the native-length rerun (pilot CSV + per-arm `.npz`),
the frozen controlled H=200 cells (bridge), and the frozen native cells
(reproduction gate only). Every decomposition quantity is computed within the
rerun trajectories; Bail and Credit are never averaged together.

Selection uses validation records only; test scores are read at the selected
epoch and at the pre-registered grid epochs. One 30-row table per dataset goes
through the frozen `bootstrap_armA.boot()` once, and the frozen resolved rule is
applied unchanged.

    python harness/experiments/analyze_x26_fairgb.py --dataset bail \
        --rerun_csv ... --traj ... --noise_csv ... --outdir harness/results/x26
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, ROOT)
from analyze_armA import sign_stability                               # noqa: E402
from bootstrap_armA import boot                                       # noqa: E402
from core.evaluator import evaluate                                   # noqa: E402
from x26_fairgb_run import CAP_LAST, GRID, SIGMAS                     # noqa: E402

SEED = 20260918
B_REPS = 10_000
SIGN_MIN, NEAR_ZERO = 0.75, 0.010                  # frozen rule
TOL = 1e-12
MAX_FLIPS = 4                                       # amended-G3 search depth (326f04f)
COORDS = ("ndp", "neo", "auc")
LAB = {"ndp": "-dDP", "neo": "-dEO", "auc": "dAUC"}
INT_COL = {"ndp": "int_ndp", "neo": "int_neo", "auc": "int_dauc"}
SEL = {"bce": "common_bce", "auc": "common_auc"}
NATIVE_H = {"bail": 1500, "credit": 2000}
FROZEN = {
    "bail": dict(controlled=["harness/results/armA_bail.csv", "harness/results/armA_bail_s23_25.csv"],
                 native=["harness/results/armB_native_FairGB_bail.csv"]),
    "credit": dict(controlled=["harness/results/armA_credit.csv", "harness/results/armA_credit_s23_25.csv"],
                   native=["harness/results/armB_native_FairGB_credit.csv"]),
}
BOUNDARY_FLIPS = []


def noise_bound(noise_csv):
    r = pd.read_csv(noise_csv)
    spread = float(r.replay_spread.max())
    return max(1e-5, 4 * spread), spread, len(r)


def explain_flips(stored, y, a, dp_ref, eo_ref, bound):
    cand = np.nonzero(np.abs(stored) <= bound)[0]
    cand = cand[np.argsort(np.abs(stored[cand]))]
    for k in range(1, min(MAX_FLIPS, len(cand)) + 1):
        for sub in itertools.combinations(cand.tolist(), k):
            f = stored.copy()
            f[list(sub)] = -f[list(sub)]
            m = evaluate(y, a, raw_score=f, decision="score>0")
            if abs(m["dp"] - dp_ref) <= TOL and abs(m["eo"] - eo_ref) <= TOL:
                return list(sub)
    return None


def outcomes(score, y, a):
    m = evaluate(y, a, raw_score=np.asarray(score, dtype=float), decision="score>0")
    return dict(ndp=-m["dp"], neo=-m["eo"], auc=m["auc"], dp=m["dp"], eo=m["eo"])


def frozen_rows(paths, dataset):
    d = pd.concat([pd.read_csv(os.path.join(ROOT, p)) for p in paths])
    d = d[(d.method == "FairGB") & (d.dataset == dataset)]
    return {(int(r.split_id), int(r.run_id), r.selector): r for r in d.itertuples()}, d


def load_rerun(dataset, csv_path, traj_dir, bound):
    H, grid = NATIVE_H[dataset], GRID[dataset]
    d = pd.read_csv(csv_path)
    d = d[(d.method == "FairGB") & (d.dataset == dataset)]
    errs = []
    if len(d) != 60:
        errs.append(f"{len(d)} rows")
    if d.duplicated(["split_id", "run_id", "selector"]).any():
        errs.append("duplicate keys")
    if set(d.split_id) != set(range(20, 26)) or set(d.run_id) != set(range(5)):
        errs.append("design")
    if not (d.seed == 27 + d.run_id).all():
        errs.append("seed")
    if set(d.method_epochs) != {H} or set(d.b_epochs) != {200}:
        errs.append("horizon/b_epochs")
    if not np.isfinite(d[list(INT_COL.values())].to_numpy()).all():
        errs.append("non-finite")
    if errs:
        raise SystemExit(f"[contract] {dataset} rerun: {errs}")

    arms, ids = {}, {}
    for (sp, rn), g in d.groupby(["split_id", "run_id"]):
        sp, rn, seed = int(sp), int(rn), 27 + int(rn)
        pair = {}
        for arm, col in (("M1", "m1"), ("M0", "m0")):
            p = os.path.join(traj_dir, f"{dataset}_s{sp}_seed{seed}_{arm}.npz")
            if not os.path.exists(p):
                raise SystemExit(f"[contract] {dataset}: missing trajectory {p}")
            z = dict(np.load(p, allow_pickle=False))
            bad = []
            if not np.array_equal(np.asarray(z["epochs"]).astype(int), np.arange(H)):
                bad.append("epochs not 0..H-1 (incomplete trajectory)")
            if not np.array_equal(np.asarray(z["grid_epochs"]).astype(int), np.array(grid)):
                bad.append("grid epochs")
            if int(z["horizon"]) != H or int(z["split"]) != sp or int(z["seed"]) != seed:
                bad.append("identity")
            for k in ("val_bce", "grid_scores", "slot_score_full_bce"):
                if not np.isfinite(np.asarray(z[k], dtype=float)).all():
                    bad.append(f"non-finite {k}")
            if int(z["slot_epoch_cap_bce"]) > CAP_LAST or int(z["slot_epoch_cap_auc"]) > CAP_LAST:
                bad.append("cap slot beyond 199")
            if bad:
                raise SystemExit(f"[contract] {dataset} {p}: {bad}")
            pair[arm] = z
        for k in ("test_node_id", "test_y", "test_a", "val_node_id"):
            if not np.array_equal(pair["M1"][k], pair["M0"][k]):
                raise SystemExit(f"[contract] {dataset} s{sp} r{rn}: {k} differs between arms")
            if (sp, k) in ids and not np.array_equal(ids[(sp, k)], pair["M1"][k]):
                raise SystemExit(f"[contract] {dataset} s{sp}: {k} differs across runs")
            ids[(sp, k)] = pair["M1"][k]
        if set(pair["M1"]["test_node_id"].tolist()) & set(pair["M1"]["val_node_id"].tolist()):
            raise SystemExit(f"[contract] {dataset} s{sp}: validation and test overlap")
        # G3: the full-support slot must equal the pipeline's own selection and outcome
        for arm, col in (("M1", "m1"), ("M0", "m0")):
            z = pair[arm]
            y, a = np.asarray(z["test_y"]).astype(int), np.asarray(z["test_a"]).astype(int)
            for sig in SIGMAS:
                row = g[g.selector == SEL[sig]].iloc[0]
                ep = int(z[f"slot_epoch_full_{sig}"])
                if ep != int(row[f"{col}_epoch"]):
                    raise SystemExit(f"[G3] {dataset} s{sp} r{rn} {arm} {sig}: slot epoch {ep} != "
                                     f"CSV {int(row[f'{col}_epoch'])}")
                stored = np.asarray(z[f"slot_score_full_{sig}"], dtype=float)
                v = outcomes(stored, y, a)
                npos, nneg = int((y == 1).sum()), int((y == 0).sum())
                ddp, deo = abs(v["dp"] - row[f"{col}_dp"]), abs(v["eo"] - row[f"{col}_eo"])
                # AUC: the project's validated noise-derived criterion (X20), the
                # same form the X26 gate uses -- only pairs that replay noise can
                # reorder may swap. The fixed 2-swap form was the bail-scale variant.
                swaps = abs(v["auc"] - row[f"{col}_auc"]) * npos * nneg
                eps = 2 * bound
                pos, neg = np.sort(stored[y == 1]), np.sort(stored[y == 0])
                reorderable = int((np.searchsorted(neg, pos + eps, "right")
                                   - np.searchsorted(neg, pos - eps, "left")).sum())
                if swaps > reorderable + 0.5:
                    raise SystemExit(f"[G3] {dataset} s{sp} r{rn} {arm} {sig}: AUC moved "
                                     f"{swaps:.1f} swaps, beyond the {reorderable} pairs replay "
                                     f"noise can reorder (hard stop)")
                if ddp > TOL or deo > TOL:
                    sub = explain_flips(stored, y, a, row[f"{col}_dp"], row[f"{col}_eo"], bound)
                    if sub is None:
                        raise SystemExit(f"[G3] {dataset} s{sp} r{rn} {arm} {sig}: DP/EO mismatch "
                                         f"(dDP {ddp:.2e}, dEO {deo:.2e}) not explained by <= "
                                         f"{MAX_FLIPS} flips within {bound:.3e} (hard stop)")
                    BOUNDARY_FLIPS.append(dict(dataset=dataset, split=sp, run=rn, arm=arm,
                                               selector=sig, epoch=ep, n_flips=len(sub), dDP=ddp,
                                               dEO=deo))
        arms[(sp, rn)] = pair
    return d, arms


def cell_row(pair, frozen_h200, grid):
    """All cell-level quantities for one (split, run) of one dataset."""
    row, diag = {}, {}
    vals = {}
    for arm in ("M1", "M0"):
        z = pair[arm]
        y, a = np.asarray(z["test_y"]).astype(int), np.asarray(z["test_a"]).astype(int)
        for sig in SIGMAS:
            for supp in ("cap", "full"):
                vals[(arm, supp, sig)] = outcomes(z[f"slot_score_{supp}_{sig}"], y, a)
                diag[(arm, supp, sig)] = int(z[f"slot_epoch_{supp}_{sig}"])
        vals[arm, "grid"] = [outcomes(s, y, a) for s in np.asarray(z["grid_scores"], dtype=float)]
    for sig in SIGMAS:
        for c in COORDS:
            cap = vals[("M1", "cap", sig)][c] - vals[("M0", "cap", sig)][c]
            full = vals[("M1", "full", sig)][c] - vals[("M0", "full", sig)][c]
            h200 = frozen_h200[sig][c]
            row[f"{sig}.cap.{c}"] = cap
            row[f"{sig}.full.{c}"] = full
            row[f"{sig}.h200.{c}"] = h200
            row[f"{sig}.S.{c}"] = full - cap
            row[f"{sig}.T.{c}"] = cap - h200
            row[f"{sig}.Hshift.{c}"] = full - h200
            row[f"{sig}.E1.{c}"] = vals[("M1", "full", sig)][c] - vals[("M1", "cap", sig)][c]
            row[f"{sig}.E0.{c}"] = vals[("M0", "full", sig)][c] - vals[("M0", "cap", sig)][c]
    for i, t in enumerate(grid):
        for c in COORDS:
            row[f"fx{t}.{c}"] = vals["M1", "grid"][i][c] - vals["M0", "grid"][i][c]
    return row, diag


def identity_residuals(get, grid):
    res = {}
    for sig in SIGMAS:
        for c in COORDS:
            g = lambda q: np.asarray(get(f"{sig}.{q}.{c}"), dtype=float)   # noqa: E731
            res[f"{sig}.{c}: Hshift = S + T"] = float(np.max(np.abs(g("Hshift") - (g("S") + g("T")))))
            res[f"{sig}.{c}: S = full - cap"] = float(np.max(np.abs(g("S") - (g("full") - g("cap")))))
            res[f"{sig}.{c}: S = E1 - E0"] = float(np.max(np.abs(g("S") - (g("E1") - g("E0")))))
    return res


def stat(values, rep_col):
    x = np.asarray(values, dtype=float)
    mu = float(x.mean())
    lo, hi = np.percentile(rep_col, [2.5, 97.5])
    sg = sign_stability(pd.Series(x))
    sg = float(sg) if sg != "n/a" else float("nan")
    return dict(mean=mu, lo=float(lo), hi=float(hi), sign=sg,
                resolved=bool(np.isfinite(sg) and sg >= SIGN_MIN and abs(mu) >= NEAR_ZERO
                              and lo * hi > 0))


def case(S_, T_, H_):
    if not H_["resolved"] and not (S_["resolved"] or T_["resolved"]):
        return "E none"
    if S_["resolved"] and T_["resolved"]:
        return "D mixed"
    if S_["resolved"] and np.sign(S_["mean"]) == np.sign(H_["mean"]) and not T_["resolved"]:
        return "A selection-support"
    if T_["resolved"]:
        return "B prefix/run"
    if H_["resolved"] and not S_["resolved"]:
        return "C trajectory-only"
    return "E none"


def fmt(s):
    return (f"{s['mean']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}] s{s['sign']:.2f} "
            f"{'R' if s['resolved'] else 'u'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("bail", "credit"))
    ap.add_argument("--rerun_csv", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--noise_csv", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    ds, H, grid = a.dataset, NATIVE_H[a.dataset], GRID[a.dataset]
    os.makedirs(a.outdir, exist_ok=True)
    outs = {k: os.path.join(a.outdir, f"x26_{ds}_{k}") for k in
            ("cell_table.csv", "summary.csv", "selected_epochs.csv", "gate.txt")}
    for p in outs.values():
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")

    bound, spread, n_slots = noise_bound(a.noise_csv)
    print(f"[{ds}] replay-noise envelope: max(1e-5, 4 x {spread:.3e}) = {bound:.3e} "
          f"from {n_slots} independent slot evaluations (X25 amendment 326f04f form)")
    d, arms = load_rerun(ds, a.rerun_csv, a.traj, bound)
    ctrl, ctrl_d = frozen_rows(FROZEN[ds]["controlled"], ds)
    nat, nat_d = frozen_rows(FROZEN[ds]["native"], ds)
    print(f"[{ds}] contracts + G3 on 120 comparisons: pass "
          f"({len(BOUNDARY_FLIPS)} boundary-flip, rest exact)")
    for col in ("protocol", "method_epochs", "b_epochs", "feature_normalize"):
        if col in nat_d and set(d[col]) != set(nat_d[col]):
            raise SystemExit(f"[config] {ds} {col}: {set(d[col])} != frozen {set(nat_d[col])}")

    rows, sel_rows = [], []
    for (sp, rn), pair in sorted(arms.items()):
        h200 = {sig: {c: getattr(ctrl[(sp, rn, SEL[sig])], INT_COL[c]) for c in COORDS}
                for sig in SIGMAS}
        row, diag = cell_row(pair, h200, grid)
        rows.append(dict(split_id=sp, run_id=rn, **row))
        for (arm, supp, sig), ep in diag.items():
            sel_rows.append(dict(split_id=sp, run_id=rn, arm=arm, support=supp, selector=sig,
                                 epoch=ep))
    t = pd.DataFrame(rows)
    cols = [c for c in t.columns if c not in ("split_id", "run_id")]
    reps = boot(t, cols, np.random.default_rng(SEED), reps=B_REPS)
    ci = {c: i for i, c in enumerate(cols)}
    res = {**identity_residuals(lambda c: t[c], grid),
           **{f"rep {k}": v for k, v in identity_residuals(lambda c: reps[:, ci[c]], grid).items()}}
    bad = {k: v for k, v in res.items() if not v <= TOL}
    if bad:
        raise SystemExit(f"[contract] identity check failed: {bad}")
    print(f"[{ds}] identities (Hshift = S + T, S = full - cap, S = E1 - E0) hold <= {TOL:g} "
          f"per cell and in all {B_REPS:,} replicates")
    S = {c: stat(t[c], reps[:, ci[c]]) for c in cols}

    # ---- reproduction gate ----
    gate_lines, ok = [], True
    for sig in SIGMAS:
        g = nat_d[nat_d.selector == SEL[sig]]
        for c in COORDS:
            x = np.asarray([getattr(r, INT_COL[c]) for r in g.itertuples()], dtype=float)
            frozen_mean = float(x.mean())
            reps_f = boot(pd.DataFrame(dict(split_id=g.split_id.to_numpy(), v=x)), ["v"],
                          np.random.default_rng(SEED + 1), reps=2000)
            lo, hi = np.percentile(reps_f[:, 0], [2.5, 97.5])
            m = S[f"{sig}.full.{c}"]["mean"]
            inside = bool(lo <= m <= hi)
            ok &= inside
            gate_lines.append(f"  {sig} {LAB[c]}: rerun full {m:+.4f} in frozen native "
                              f"[{lo:+.4f},{hi:+.4f}] (mean {frozen_mean:+.4f}): {inside}")
    head = S["bce.full.ndp"]
    if ds == "bail":
        head_ok = not head["resolved"]
        gate_lines.append(f"  headline: BCE -dDP {fmt(head)} must be unresolved (frozen): {head_ok}")
    else:
        head_ok = head["resolved"] and head["mean"] > 0
        gate_lines.append(f"  headline: BCE -dDP {fmt(head)} must be resolved positive (frozen): {head_ok}")
    ok &= head_ok
    gate = "\n".join([f"Reproduction gate ({ds}, X26 section 10):"] + gate_lines +
                     [f"  GATE {'PASS' if ok else 'FAIL'}"])
    print("\n" + gate)
    open(outs["gate.txt"], "w").write(gate + "\n")
    if not ok:
        print(f"\nSTOP ({ds}): reproduction gate failed; no X26 result reported for this dataset.")
        return 2

    t.to_csv(outs["cell_table.csv"], index=False)
    pd.DataFrame(sel_rows).to_csv(outs["selected_epochs.csv"], index=False)
    pd.DataFrame([dict(quantity=c, **S[c]) for c in cols]).to_csv(outs["summary.csv"], index=False)

    print("\n" + "=" * 118)
    print(f"Table — FairGB / {ds}: selection-support decomposition "
          f"(cap {{0..{CAP_LAST}}} vs full {{0..{H - 1}}}; 10,000 paired hierarchical bootstrap)")
    print("=" * 118)
    for c in COORDS:
        print(f"\n  {LAB[c]}")
        for sig in SIGMAS:
            print(f"    {sig.upper():<4} tau_H200 {S[f'{sig}.h200.{c}']['mean']:+.4f}  "
                  f"tau_cap {fmt(S[f'{sig}.cap.{c}'])}  tau_full {fmt(S[f'{sig}.full.{c}'])}")
            print(f"         H_shift {fmt(S[f'{sig}.Hshift.{c}'])}   S {fmt(S[f'{sig}.S.{c}'])}   "
                  f"T {fmt(S[f'{sig}.T.{c}'])}   case: {case(S[f'{sig}.S.{c}'], S[f'{sig}.T.{c}'], S[f'{sig}.Hshift.{c}'])}")
            print(f"         telescoping  E1 {fmt(S[f'{sig}.E1.{c}'])}   E0 {fmt(S[f'{sig}.E0.{c}'])}")

    print("\n" + "=" * 118)
    print(f"Fixed-epoch trajectory — FairGB / {ds} (descriptive; no per-epoch labels)")
    print("=" * 118)
    print(f"  {'epoch':>6}  {'tau_int -dDP':>34}  {'tau_int -dEO':>34}  {'tau_int dAUC':>34}")
    for tt in grid:
        q = [S[f"fx{tt}.{c}"] for c in ("ndp", "neo", "auc")]
        print(f"  {tt:>6}  " + "  ".join(
            f"{s['mean']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}] s{s['sign']:.2f}" for s in q))

    se = pd.DataFrame(sel_rows)
    print("\n" + "=" * 118)
    print(f"Selected epochs — FairGB / {ds} (descriptive)")
    print("=" * 118)
    for sig in SIGMAS:
        for supp in ("cap", "full"):
            for arm in ("M1", "M0"):
                g = se[(se.selector == sig) & (se.support == supp) & (se.arm == arm)]
                capend = CAP_LAST if supp == "cap" else H - 1
                print(f"  {sig.upper():<4} {supp:<4} {arm}: median {g.epoch.median():7.1f} "
                      f"[{g.epoch.min()},{g.epoch.max()}] boundary {np.mean(g.epoch == capend):.2f}")
    print("\n[written] " + ", ".join(outs.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
