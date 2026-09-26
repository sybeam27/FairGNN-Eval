"""
Analysis for E2 -- paired comparisons against uniform weighting.

E0 established three things that dictate how this has to be read:

  * repeating one seed already moves dp by up to 0.035 (german), and the effect
    the draft claims is 0.011, so a difference is only evidence if its interval
    excludes zero -- a sign is not a result;
  * pairing on seed is what makes the experiment affordable, because the seed
    main effect cancels;
  * the required replicate count differs by two orders of magnitude across
    settings, so R=30 is adequate for some settings and not for others, and
    which is which must be reported rather than averaged over.

Accordingly this script never reports a bare mean difference. For each
(setting, signal) it pairs on seed, gives a t interval and a Wilcoxon signed-rank
p-value, applies a Holm correction across the whole family of comparisons, and
recomputes from the observed paired sd how many replicates that setting would
actually have needed. Settings whose interval contains zero are labelled
undecided and are not folded into any headline.

Usage
-----
    python harness/experiments/analyze_e2.py harness/results/e2_signal_swap_R30.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "uniform"
TARGET_EFFECT = 0.011      # the draft's uniform 0.039 -> adaptive 0.028


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def paired_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    col = f"test_{metric}"
    rows = []
    for (ds, sig), sub in df.groupby(["setting", "signal"]):
        if sig == BASE:
            continue
        base = df[(df.setting == ds) & (df.signal == BASE)][["seed", col]]
        m = sub[["seed", col]].merge(base, on="seed", suffixes=("", "_base"))
        if len(m) < 3:
            continue
        d = (m[col] - m[f"{col}_base"]).to_numpy()      # negative = allocation helped
        n = len(d)
        sd = d.std(ddof=1)
        se = sd / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, n - 1)
        lo, hi = d.mean() - tcrit * se, d.mean() + tcrit * se
        try:
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        rows.append({
            "setting": ds, "regime": sub["regime"].iloc[0], "signal": sig, "n": n,
            "mean_d": d.mean(), "sd_d": sd, "ci_lo": lo, "ci_hi": hi,
            "p_raw": p, "wins": int((d < 0).sum()),
            # replicates this setting would need for a CI half-width of
            # TARGET_EFFECT/2, from the sd we actually observed
            "R_needed": int(np.ceil((2 * tcrit * sd / TARGET_EFFECT) ** 2)) if sd > 0 else 0,
        })
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["p_holm"] = holm(t["p_raw"].to_numpy())
    t["verdict"] = np.where(t.ci_hi < 0, "helps",
                    np.where(t.ci_lo > 0, "hurts", "undecided"))
    # a verdict that does not survive the family-wise correction is not a verdict
    t.loc[(t.verdict != "undecided") & (t.p_holm > 0.05), "verdict"] = "undecided*"
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--metric", default="dp", choices=["dp", "eo", "acc", "auc"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    pd.set_option("display.width", 240)
    print(f"{len(df)} runs, {df.setting.nunique()} settings, "
          f"{df.signal.nunique()} signals, {df.seed.nunique()} seeds\n")

    t = paired_table(df, args.metric)
    if t.empty:
        print("nothing to compare")
        return 1

    show = ["setting", "regime", "signal", "n", "mean_d", "ci_lo", "ci_hi",
            "sd_d", "wins", "p_holm", "R_needed", "verdict"]
    print("=" * 118)
    print(f"PAIRED vs {BASE}   metric={args.metric}   negative mean_d = allocation helped")
    print(f"'undecided*' = interval excluded zero but did not survive Holm over "
          f"{len(t)} comparisons")
    print("=" * 118)
    for ds, sub in t.groupby("setting"):
        print(f"\n{ds}  [{sub.regime.iloc[0]}]")
        print(sub[show].drop(columns=["setting", "regime"])
                .round({"mean_d": 4, "ci_lo": 4, "ci_hi": 4, "sd_d": 4, "p_holm": 4})
                .to_string(index=False))

    print("\n" + "=" * 118)
    print("WAS R=30 ENOUGH?   R_needed from the observed paired sd")
    print("=" * 118)
    need = (t.groupby(["setting", "regime"])["R_needed"].max()
              .reset_index().sort_values("R_needed", ascending=False))
    need["adequate_at_n"] = need["R_needed"] <= t["n"].max()
    print(need.to_string(index=False))

    print("\n" + "=" * 118)
    print("THE E2 PREDICTION: does each signal win only in the regime that suits it?")
    print("=" * 118)
    dec = t[t.verdict == "helps"]
    if dec.empty:
        print("No signal beat uniform with an interval excluding zero, anywhere.\n"
              "Read literally: allocation does not help, and the premise of the paper fails.")
    else:
        piv = (dec.pivot_table(index="signal", columns="regime", values="setting",
                               aggfunc=lambda s: ", ".join(sorted(s)))
                  .fillna("-"))
        print(piv.to_string())
        print("\nA signal that helps in one regime and nowhere else supports the "
              "regime-adaptive\nclaim. A signal that helps everywhere refutes it: "
              "the fixed rule would suffice.")

    print("\n" + "=" * 118)
    print("HEADLINE QUANTITY: mean over settings, per seed (what the draft averages)")
    print("=" * 118)
    per = (df.groupby(["signal", "seed"])[f"test_{args.metric}"].mean().unstack("signal"))
    base = per[BASE]
    hl = []
    for sig in per.columns:
        if sig == BASE:
            continue
        d = (per[sig] - base).dropna()
        n = len(d)
        tcrit = stats.t.ppf(0.975, n - 1)
        se = d.std(ddof=1) / np.sqrt(n)
        hl.append({"signal": sig, "mean": per[sig].mean(), "mean_d": d.mean(),
                   "ci_lo": d.mean() - tcrit * se, "ci_hi": d.mean() + tcrit * se, "n": n})
    h = pd.DataFrame(hl).sort_values("mean_d")
    print(f"{BASE} mean = {base.mean():.4f}\n")
    print(h.round(4).to_string(index=False))
    print("\nThis is the only row that is comparable to the draft's headline. "
          "Settings whose\nper-setting verdict is undecided are still inside this "
          "average, which is exactly\nwhy the draft's version of it was not "
          "interpretable.")

    if args.out:
        t.to_csv(args.out, index=False)
        print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
