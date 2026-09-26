"""
Tests H1-H3 of `harness/PREREGISTRATION.md` by joining the two matrices:

    E1  discriminability   D(g, G)      harness/results/e1_discriminability.csv
    E2  paired benefit     delta(g, G)  harness/results/e2_signal_swap_R30.csv

Nothing here is chosen after the fact. The primary statistic, the tests, the
Holm family, the exclusion of German from headlines and the admissibility
condition on signals are all fixed in the pre-registration and its two recorded
deviations; this file implements them and nothing else.

Usage
-----
    python harness/experiments/analyze_hypotheses.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze_e2 import paired_table          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E1 = os.path.join(ROOT, "results", "e1_discriminability.csv")
E2 = os.path.join(ROOT, "results", "e2_signal_swap_R30.csv")

BASE = "uniform"
NOISE_EXCLUDED = ["german"]      # PREREGISTRATION: fixed from E0, before any benefit


# ----------------------------------------------------------------- admissibility
def inadmissible_signals(settings: list[str], signals: list[str],
                         warmup: int, device: str) -> set[str]:
    """Signals that are constant within every (y, s) cell on `I`.

    Deviation D2a: such a signal is a *group* reweighting, not a node-level one,
    and cannot rank nodes however large its MI with `s` happens to be. The test
    is the condition itself rather than the numerical proxy D2 first proposed,
    which missed `fairgb_group` on Pokec-z where its four subgroup weights
    collide across `s` (MI 0.0011 against H(s) 0.6443) while still taking one
    value per cell.
    """
    from core import datasets as D                       # local: torch-free path
    from core import signals as S
    from core.trainer import Config, warmup_state

    bad = set()
    for name in settings:
        g = D.load(name)
        st = None
        i = np.asarray(g.idx_fair)
        sv, yv = g.sens[i], g.y[i].astype(int)
        for sig in signals:
            if sig in bad:
                continue
            spec = S.get(sig)
            if spec.needs_model and st is None:
                st = warmup_state(g, Config(seed=27, warmup=warmup, device=device))
            v = spec(g, st)[i]
            cells = [v[(sv == a) & (yv == b)] for a in (0, 1) for b in (0, 1)]
            varies = any(len(c) > 1 and np.ptp(c) > 1e-12 for c in cells)
            if not varies and sig != BASE:
                bad.add(sig)
    return bad


def holm(p: np.ndarray) -> np.ndarray:
    n, order, adj, run = len(p), np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


# ----------------------------------------------------------------------- H1
def h1(m: pd.DataFrame, label: str, n_perm: int = 10000, seed: int = 0) -> dict:
    """Spearman(D, delta) with a null that shuffles D *within* each setting.

    Settings differ enormously in baseline disparity and in noise; a test that
    pooled them could report an association driven entirely by between-setting
    spread. Permuting inside each setting removes that route.
    """
    d, y = m["D"].to_numpy(), m["mean_d"].to_numpy()
    obs = stats.spearmanr(d, y).statistic

    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(m["setting"].to_numpy() == s) for s in m["setting"].unique()]
    null = np.empty(n_perm)
    dp = d.copy()
    for i in range(n_perm):
        for g in groups:
            dp[g] = rng.permutation(d[g])
        null[i] = stats.spearmanr(dp, y).statistic
    p = float((np.abs(null) >= abs(obs)).mean())

    # CI by bootstrapping settings, since settings are the unit of independence
    sets = m["setting"].unique()
    boot = []
    for _ in range(2000):
        pick = rng.choice(sets, size=len(sets), replace=True)
        sub = pd.concat([m[m.setting == s] for s in pick])
        if sub["D"].nunique() > 1:
            boot.append(stats.spearmanr(sub["D"], sub["mean_d"]).statistic)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"which": label, "n_cells": len(m), "spearman": obs,
            "p_perm": p, "ci_lo": lo, "ci_hi": hi}


# ----------------------------------------------------------------------- H2
def h2(e2: pd.DataFrame, dmap: pd.DataFrame, pmap: pd.DataFrame, adm: set[str],
       metric: str = "dp") -> pd.DataFrame:
    """Leave-one-setting-out. Every strategy is applied to a setting it did not
    inform, and all four are evaluated on the same paired seeds."""
    col, vcol = f"test_{metric}", f"val_{metric}"
    per = e2.groupby(["setting", "signal"])[col].mean().unstack("signal")
    pervl = e2.groupby(["setting", "signal"])[vcol].mean().unstack("signal")
    cands = [c for c in per.columns if c in adm and c != BASE]

    rows = []
    for held in per.index:
        train = per.drop(index=held)
        d_held = dmap.loc[held, [c for c in cands if c in dmap.columns]]
        top = d_held.idxmax()
        # D3 (amended twice): abstain unless the best D beats its own permutation
        # null after Bonferroni over the k candidates it was selected from --
        # argmax over k statistics is not a test of one statistic.
        abstain = len(cands) * pmap.loc[held, top] >= 0.05
        picks = {
            "rule":         top,
            "rule_abstain": BASE if abstain else top,
            "best_fixed":  train[cands].mean().idxmin(),
            "val_select":  pervl.loc[held, cands].idxmin(),
            "oracle":      per.loc[held, cands].idxmin(),
            BASE:          BASE,
        }
        row = {"setting": held}
        row["abstained"] = bool(abstain)
        for k, sig in picks.items():
            row[f"{k}_signal"] = sig
            row[k] = per.loc[held, sig]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default=E1)
    ap.add_argument("--e2", default=E2)
    ap.add_argument("--metric", default="dp")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    e1 = pd.read_csv(args.e1)
    e2 = pd.read_csv(args.e2)
    pd.set_option("display.width", 220)

    common = sorted(set(e1.setting) & set(e2.setting))
    e1, e2 = e1[e1.setting.isin(common)], e2[e2.setting.isin(common)]
    print(f"settings joined: {len(common)}  {common}")

    sigs = sorted(e1.signal.unique())
    bad = inadmissible_signals(common, sigs, warmup=int(e1["warmup"].iloc[0]),
                               device=args.device)
    adm = set(sigs) - bad
    print(f"inadmissible (constant within every (y,s) cell, deviations D2/D2a): "
          f"{sorted(bad) or 'none'}")

    dmap = e1.pivot_table(index="setting", columns="signal", values="mi_with_sens",
                          aggfunc="mean")
    pmap = e1.pivot_table(index="setting", columns="signal", values="mi_p_perm",
                          aggfunc="mean")
    ben = paired_table(e2, args.metric)
    m = ben.merge(dmap.stack().rename("D").reset_index(), on=["setting", "signal"])

    # ------------------------------------------------------------------ H1
    print("\n" + "=" * 100)
    print("H1  does discriminability predict benefit?   "
          "(negative Spearman = more discriminable -> lower disparity)")
    print("=" * 100)
    res = [h1(m[m.signal.isin(adm)], "primary (admissible signals)"),
           h1(m, "inclusive (all signals)")]
    for tag, sub in (("no german", m[(m.signal.isin(adm)) &
                                     (~m.setting.isin(NOISE_EXCLUDED))]),):
        if sub.setting.nunique() > 2:
            res.append(h1(sub, f"primary, {tag}"))
    print(pd.DataFrame(res).round(4).to_string(index=False))
    print("\nH1 is falsified if the bootstrap CI of the primary row contains 0.")

    # ------------------------------------------------------------------ H2
    print("\n" + "=" * 100)
    print(f"H2  leave-one-setting-out.  mean test {args.metric} on the held-out setting")
    print("=" * 100)
    lo = h2(e2, dmap, pmap, adm, args.metric)
    print(lo.round(4).to_string(index=False))

    for tag, sub in (("all settings", lo),
                     ("excluding german", lo[~lo.setting.isin(NOISE_EXCLUDED)])):
        if len(sub) < 3:
            continue
        print(f"\n  -- {tag} (n={len(sub)} folds) --")
        means = {k: sub[k].mean() for k in ("rule", "rule_abstain", "best_fixed",
                                            "val_select", "oracle", BASE) if k in sub}
        print("   " + "  ".join(f"{k}={v:.4f}" for k, v in means.items()))
        for lhs, other in [("rule", "best_fixed"), ("rule", BASE),
                           ("rule", "val_select"),
                           ("rule_abstain", "best_fixed"), ("rule_abstain", BASE)]:
            d = (sub[lhs] - sub[other]).to_numpy()
            n = len(d)
            tc = stats.t.ppf(0.975, n - 1)
            se = d.std(ddof=1) / np.sqrt(n)
            verdict = ("better" if d.mean() + tc * se < 0
                       else "worse" if d.mean() - tc * se > 0 else "undecided")
            print(f"   {lhs:<12s} - {other:<11s} = {d.mean():+.4f} "
                  f"[{d.mean()-tc*se:+.4f}, {d.mean()+tc*se:+.4f}]  {verdict}")
    print("\nH2 is falsified if 'rule - best_fixed' does not exclude zero.")

    # ------------------------------------------------------------------ H3
    print("\n" + "=" * 100)
    print("H3  where nothing is discriminable, does nothing help?")
    print("=" * 100)
    cand = [c for c in dmap.columns if c in adm and c != BASE]
    h3 = pd.DataFrame({
        "max_D": dmap[cand].max(axis=1),
        "argmax_D": dmap[cand].idxmax(axis=1),
        "n_helps": ben[ben.verdict == "helps"].groupby("setting").size(),
        "p_of_argmax": [pmap.loc[i, dmap.loc[i, cand].idxmax()] for i in dmap.index],
        "n_admissible_helps": ben[(ben.verdict == "helps") &
                                  (ben.signal.isin(adm))].groupby("setting").size(),
    }).fillna(0).sort_values("max_D")
    h3["regime"] = ben.groupby("setting")["regime"].first()
    print(h3.round(4).to_string())
    if h3["max_D"].nunique() > 2:
        r = stats.spearmanr(h3["max_D"], h3["n_admissible_helps"])
        print(f"\nSpearman(max_D, n_helps) = {r.statistic:.3f}  (p = {r.pvalue:.3f}, "
              f"n = {len(h3)} settings -- read qualitatively, nine points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
