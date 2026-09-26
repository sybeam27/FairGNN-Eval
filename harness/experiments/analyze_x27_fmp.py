"""X27 analysis: FMP mechanistic decomposition on Pokec-z and Pokec-n (prereg 00b6b26).

Per dataset, inputs are the runner's CSV (one row per configuration x selector x
cell) and its trajectories. Every quantity is computed within the same
(split, run) cells, which share one initialization, so the contrasts are paired.

    tau_prop(l2)     = mean over cells of [ Y(F01(l2))    - Y(F00) ]
    tau_fair(l1,l2)  = mean over cells of [ Y(F11(l1,l2)) - Y(F01(l2)) ]
    tau_total(l1,l2) = mean over cells of [ Y(F11(l1,l2)) - Y(F00) ]
    identity: tau_total = tau_prop + tau_fair, checked <= 1e-12 per cell,
              on the means, and in every bootstrap replicate

Y = [AUC, -dDP] primary, -dEO secondary, all from the unified outcome the runner
recorded (AUC from the raw continuous margin). Primary selector sigma_last;
sigma_BCE and sigma_AUC are robustness. Datasets are never averaged together.

    python harness/experiments/analyze_x27_fmp.py --dataset pokec_z \
        --csv /tmp/.../pokec_z.csv --outdir harness/results/x27
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
from analyze_armA import sign_stability                               # noqa: E402
from bootstrap_armA import boot                                       # noqa: E402

SEED = 20260919
B_REPS = 10_000
SIGN_MIN, NEAR_ZERO = 0.75, 0.010          # frozen rule, unchanged
TOL = 1e-12
LAM1 = (5.0, 30.0)                          # preregistered endpoints (X27 section 3)
LAM2 = (0.01, 20.0)
SELECTORS = ("last", "bce", "auc")          # last is primary
COORDS = ("auc", "ndp", "neo")
LAB = {"auc": "dAUC", "ndp": "-dDP", "neo": "-dEO"}


def cfg_name(l1, l2):
    if l1 == 0 and l2 == 0:
        return "F00"
    if l1 == 0:
        return f"F01_l2{l2:g}"
    return f"F11_l1{l1:g}_l2{l2:g}"


def outcome_cols(d):
    """Outcome vector per row: AUC, -dDP, -dEO from the recorded absolute gaps."""
    return pd.DataFrame(dict(auc=d.auc.to_numpy(float),
                             ndp=-d.dp.to_numpy(float),
                             neo=-d.eo.to_numpy(float)))


def load(dataset, csv):
    d = pd.read_csv(csv)
    d = d[d.dataset == dataset].copy()
    errs = []
    want_cfgs = {cfg_name(0, 0)} | {cfg_name(0, l2) for l2 in LAM2} | \
                {cfg_name(l1, l2) for l1 in LAM1 for l2 in LAM2}
    if set(d.config) != want_cfgs:
        errs.append(f"configs {sorted(set(d.config))} != {sorted(want_cfgs)}")
    if set(d.selector) != set(SELECTORS):
        errs.append(f"selectors {sorted(set(d.selector))}")
    cells = d.groupby(["split_id", "run_id"]).ngroups
    if cells != 30:
        errs.append(f"{cells} cells, expected 30")
    n_expected = 30 * len(want_cfgs) * len(SELECTORS)
    if len(d) != n_expected:
        errs.append(f"{len(d)} rows, expected {n_expected}")
    if d.duplicated(["split_id", "run_id", "config", "selector"]).any():
        errs.append("duplicate keys")
    if not np.isfinite(d[["auc", "dp", "eo"]].to_numpy(float)).all():
        errs.append("non-finite outcome")
    n_undef = int((d.eo_defined == 0).sum())
    if errs:
        raise SystemExit(f"[contract] {dataset}: {errs}")
    print(f"[{dataset}] contract: 30 cells x {len(want_cfgs)} configurations x "
          f"{len(SELECTORS)} selectors = {len(d)} rows, no duplicates, finite; "
          f"EO undefined in {n_undef} rows (reported, not dropped)")
    return d


def cell_table(d, selector):
    """One row per (split, run) with every decomposition column, for one selector."""
    g = d[d.selector == selector]
    wide = {}
    for cfg, gg in g.groupby("config"):
        gg = gg.sort_values(["split_id", "run_id"])
        y = outcome_cols(gg)
        for c in COORDS:
            wide[(cfg, c)] = y[c].to_numpy()
    idx = g.sort_values(["split_id", "run_id"])[["split_id", "run_id"]].drop_duplicates()
    row = dict(split_id=idx.split_id.to_numpy(), run_id=idx.run_id.to_numpy())
    for l2 in LAM2:
        for c in COORDS:
            row[f"prop.l2{l2:g}.{c}"] = wide[(cfg_name(0, l2), c)] - wide[("F00", c)]
        for l1 in LAM1:
            f11 = cfg_name(l1, l2)
            for c in COORDS:
                row[f"fair.l1{l1:g}.l2{l2:g}.{c}"] = wide[(f11, c)] - wide[(cfg_name(0, l2), c)]
                row[f"total.l1{l1:g}.l2{l2:g}.{c}"] = wide[(f11, c)] - wide[("F00", c)]
    return pd.DataFrame(row)


def identity_residuals(get):
    res = {}
    for l2 in LAM2:
        for l1 in LAM1:
            for c in COORDS:
                tot = np.asarray(get(f"total.l1{l1:g}.l2{l2:g}.{c}"), dtype=float)
                pro = np.asarray(get(f"prop.l2{l2:g}.{c}"), dtype=float)
                fai = np.asarray(get(f"fair.l1{l1:g}.l2{l2:g}.{c}"), dtype=float)
                res[f"l1={l1:g} l2={l2:g} {c}"] = float(np.max(np.abs(tot - (pro + fai))))
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


def fmt(s):
    return (f"{s['mean']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}] s{s['sign']:.2f} "
            f"{'R' if s['resolved'] else 'u'}")


def grid_class(stats, coord="ndp"):
    """Descriptive classification over the preregistered grid endpoints only."""
    cells = [stats[f"fair.l1{l1:g}.l2{l2:g}.{coord}"] for l2 in LAM2 for l1 in LAM1]
    res = [c for c in cells if c["resolved"]]
    if len(res) == len(cells) and all(c["mean"] > 0 for c in res):
        return "grid-consistent improvement", len(res), 0, 0
    if len(res) == len(cells) and all(c["mean"] < 0 for c in res):
        return "grid-consistent harm", 0, len(res), 0
    n_imp = sum(1 for c in res if c["mean"] > 0)
    n_harm = sum(1 for c in res if c["mean"] < 0)
    n_unres = len(cells) - len(res)
    if n_imp and n_harm:
        return "mixed", n_imp, n_harm, n_unres
    return "unresolved", n_imp, n_harm, n_unres


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    outs = {k: os.path.join(a.outdir, f"x27_{a.dataset}_{k}") for k in
            ("cell_table.csv", "summary.csv", "analysis.txt")}
    for p in outs.values():
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")

    d = load(a.dataset, a.csv)
    rng = np.random.default_rng(SEED)
    stats, tables = {}, {}
    for sel in SELECTORS:
        t = cell_table(d, sel)
        cols = [c for c in t.columns if c not in ("split_id", "run_id")]
        reps = boot(t, cols, rng, reps=B_REPS)
        ci = {c: i for i, c in enumerate(cols)}
        res = {**identity_residuals(lambda c: t[c]),
               **{f"rep {k}": v for k, v in
                  identity_residuals(lambda c: reps[:, ci[c]]).items()}}
        bad = {k: v for k, v in res.items() if not v <= TOL}
        if bad:
            raise SystemExit(f"[contract] {a.dataset} {sel}: identity failed {bad}")
        stats[sel] = {c: stat(t[c], reps[:, ci[c]]) for c in cols}
        tables[sel] = t
        print(f"[{a.dataset}] {sel}: tau_total = tau_prop + tau_fair holds <= {TOL:g} "
              f"per cell and in all {B_REPS:,} replicates")

    S = stats["last"]
    tables["last"].to_csv(outs["cell_table.csv"], index=False)
    pd.DataFrame([dict(selector=sel, quantity=c, **stats[sel][c])
                  for sel in SELECTORS for c in stats[sel]]).to_csv(outs["summary.csv"], index=False)

    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("\n" + "=" * 118)
    emit(f"Table 2 - mechanistic decomposition, FMP / {a.dataset}, selector sigma_last "
         f"(10,000 paired hierarchical bootstrap; s = sign stability; R/u = resolved)")
    emit("=" * 118)
    for c in COORDS:
        emit(f"\n  {LAB[c]}")
        for l2 in LAM2:
            emit(f"    tau_prop(l2={l2:g}){'':<6}{fmt(S[f'prop.l2{l2:g}.{c}']):>44}")
            for l1 in LAM1:
                emit(f"      l1={l1:<4g} tau_fair {fmt(S[f'fair.l1{l1:g}.l2{l2:g}.{c}']):>44}   "
                     f"tau_total {fmt(S[f'total.l1{l1:g}.l2{l2:g}.{c}']):>44}")

    emit("\n" + "=" * 118)
    emit(f"Table 3 - fairness-correction attribution, tau_fair(-dDP), FMP / {a.dataset}")
    emit("=" * 118)
    emit(f"  {'l1':>5}{'l2':>8}{'tau_fair(-dDP)':>42}{'sign':>7}  resolved")
    for l2 in LAM2:
        for l1 in LAM1:
            s = S[f"fair.l1{l1:g}.l2{l2:g}.ndp"]
            emit(f"  {l1:>5g}{l2:>8g}{fmt(s):>42}{s['sign']:>7.2f}  {s['resolved']}")

    emit("\n" + "=" * 118)
    emit(f"Table 4 - selector robustness, tau_fair(-dDP), FMP / {a.dataset} "
         f"(sigma_last primary; BCE/AUC robustness)")
    emit("=" * 118)
    emit(f"  {'l1':>5}{'l2':>8}{'last':>34}{'BCE':>34}{'AUC':>34}   agreement")
    for l2 in LAM2:
        for l1 in LAM1:
            k = f"fair.l1{l1:g}.l2{l2:g}.ndp"
            a_, b_, c_ = (stats[s][k] for s in SELECTORS)
            agree = (a_["resolved"] == b_["resolved"] == c_["resolved"]
                     and np.sign(a_["mean"]) == np.sign(b_["mean"]) == np.sign(c_["mean"]))
            emit(f"  {l1:>5g}{l2:>8g}" + "".join(f"{fmt(x):>34}" for x in (a_, b_, c_)) +
                 f"   {'same' if agree else 'DIFFERS'}")

    emit("\n" + "=" * 118)
    emit(f"Table 5 - grid summary over the preregistered official-grid endpoints, FMP / {a.dataset}")
    emit("=" * 118)
    for sel in SELECTORS:
        cls, n_imp, n_harm, n_unres = grid_class(stats[sel])
        emit(f"  selector {sel:<5}: {cls:<32} resolved improve {n_imp}, resolved harm {n_harm}, "
             f"unresolved {n_unres} (of {len(LAM1) * len(LAM2)} cells)")
    emit("\n  This classifies the preregistered endpoints of the official search space, "
         "not all possible lambda.")

    open(outs["analysis.txt"], "w").write("\n".join(lines) + "\n")
    print("\n[written] " + ", ".join(outs.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
