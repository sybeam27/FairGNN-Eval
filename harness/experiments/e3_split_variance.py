"""
E3 -- is the train/val/test split a larger source of variation than the seed,
and do the four `helps` verdicts of E2 survive it?

Everything reported so far is conditional on `split_seed = 20`. The 30 seeds of
E2 vary initialisation, dropout and edge-drop; they do not vary the split. That
is not a minor omission on this benchmark: German trains on 50 nodes per class,
Recidivism the same, NBA on a label budget of 100 with 50 sensitive labels, and
re-splitting leaves only 4-29% of the training set in common. The draft's own
Limitations section reports German moving between first and last across five
split seeds -- so the literature already knows this axis flips conclusions, and
we have not been measuring it.

Two things are needed before the next full sweep can be budgeted.

  1. How large is split variance next to initialisation variance? If it is
     comparable, replicates can simply be drawn as (split, init) pairs at no
     extra cost and every conclusion generalises from "at split 20" to "on this
     dataset". If it is much larger, the paired sd grows, R_needed grows with
     its square, and the budget or the number of settings has to give.

  2. Do E2's four surviving `helps` -- `w_deg` on Credit and Recidivism,
     `bind_influence` on Credit and Pokec-z -- hold when the split moves? A
     verdict that exists only at one split is not a verdict about the dataset.

Design: a 6 x 5 factorial of split seed by init seed, so 30 replicates as in E2,
paired across signals within each (split, init) cell. Same replicate count, a
wider population to generalise to. Signals are cut to `uniform` plus the two
that produced decided verdicts, because the question here is variance and
robustness, not a new search.

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python harness/experiments/e3_split_variance.py --device cuda
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import datasets as D          # noqa: E402
from core.trainer import Config, train  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

BASE = "uniform"
SIGNALS = ["uniform", "w_deg", "bind_influence"]
ALL_SIGNALS = ["uniform", "random", "w_bdry", "w_deg", "w_lhd",
               "loss", "fairgb_group", "bind_influence"]
SPLITS = [20, 21, 22, 23, 24, 25]
INITS = [27, 28, 29, 30, 31]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--signals", nargs="*", default=SIGNALS)
    ap.add_argument("--all-signals", action="store_true",
                    help="run the full eight-signal library rather than "
                         "the three-signal variance probe")
    ap.add_argument("--splits", nargs="*", type=int, default=SPLITS)
    ap.add_argument("--inits", nargs="*", type=int, default=INITS)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lambda-fair", type=float, default=0.2)
    ap.add_argument("--q-gate", type=float, default=0.7)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.all_signals:
        args.signals = ALL_SIGNALS
    datasets = args.datasets or D.ALL_SETTINGS
    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(RESULTS, "e3_split_variance.csv")

    n_runs = len(datasets) * len(args.signals) * len(args.splits) * len(args.inits)
    print(f"datasets={datasets}\nsignals={args.signals}")
    print(f"splits={args.splits}  inits={args.inits}\nruns = {n_runs}\n")

    rows, t0 = [], time.time()
    for ds in datasets:
        for sp in args.splits:
            g = D.load(ds, split_seed=sp)          # provenance-checked per split
            for sig, init in itertools.product(args.signals, args.inits):
                t = time.time()
                cfg = Config(signal=sig, seed=init, epochs=args.epochs,
                             warmup=args.warmup, lambda_fair=args.lambda_fair,
                             q_gate=args.q_gate, device=args.device)
                try:
                    r = train(g, cfg)
                except Exception as e:                                # noqa: BLE001
                    print(f"  {ds} sp={sp} {sig} init={init} "
                          f"FAILED {type(e).__name__}: {e}")
                    continue
                rows.append({"setting": ds, "regime": g.regime_paper, "signal": sig,
                             "split_seed": sp, "init_seed": init,
                             # the replicate is the (split, init) cell; naming it
                             # `seed` lets analyze_e2 / analyze_hypotheses pair on
                             # it unchanged, which is the point of the design
                             "seed": f"{sp}_{init}",
                             **{f"test_{k}": v for k, v in r.test.items()},
                             "secs": round(time.time() - t, 1)})
            pd.DataFrame(rows).to_csv(out, index=False)
        sub = pd.DataFrame([x for x in rows if x["setting"] == ds])
        u = sub[sub.signal == BASE]
        bs = u.groupby("split_seed")["test_dp"].mean().std(ddof=1)     # between splits
        ws = u.groupby("split_seed")["test_dp"].std(ddof=1).mean()     # within a split
        print(f"{ds:<12} uniform dp {u.test_dp.mean():.4f}   "
              f"between-split sd {bs:.4f}   within-split (init) sd {ws:.4f}   "
              f"ratio {bs / max(ws, 1e-9):.1f}x")

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}   ({len(df)} runs, {time.time() - t0:.0f}s)\n")
    report(df)
    return 0


def report(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)

    print("=" * 104)
    print("VARIANCE COMPONENTS on the uniform arm.  If between-split >> within-split,")
    print("a conclusion drawn at one split is not a conclusion about the dataset.")
    print("=" * 104)
    rows = []
    for ds, sub in df[df.signal == BASE].groupby("setting"):
        bs = sub.groupby("split_seed")["test_dp"].mean().std(ddof=1)
        ws = sub.groupby("split_seed")["test_dp"].std(ddof=1).mean()
        rows.append({"setting": ds, "regime": sub.regime.iloc[0],
                     "dp_mean": sub.test_dp.mean(),
                     "between_split_sd": bs, "within_split_sd": ws,
                     "ratio": bs / max(ws, 1e-9),
                     "total_sd": sub.test_dp.std(ddof=1)})
    v = pd.DataFrame(rows).sort_values("ratio", ascending=False)
    print(v.round(4).to_string(index=False))

    print("\n" + "=" * 104)
    print("PAIRED vs uniform, pairing on (split, init).  Compare R_needed with the")
    print("init-only figures from E2: that difference is the price of generalising.")
    print("=" * 104)
    out = []
    for (ds, sig), sub in df.groupby(["setting", "signal"]):
        if sig == BASE:
            continue
        b = df[(df.setting == ds) & (df.signal == BASE)]
        m = sub.merge(b, on=["split_seed", "init_seed"], suffixes=("", "_b"))
        d = (m["test_dp"] - m["test_dp_b"]).to_numpy()
        n = len(d)
        sd = d.std(ddof=1)
        tc = stats.t.ppf(0.975, n - 1)
        se = sd / np.sqrt(n)
        lo, hi = d.mean() - tc * se, d.mean() + tc * se
        # how often the sign flips when the split moves
        per_split = m.assign(d=d).groupby("split_seed")["d"].mean()
        out.append({"setting": ds, "signal": sig, "n": n,
                    "mean_d": d.mean(), "ci_lo": lo, "ci_hi": hi, "sd_d": sd,
                    "splits_helping": int((per_split < 0).sum()),
                    "n_splits": len(per_split),
                    "R_needed": int(np.ceil((2 * tc * sd / 0.011) ** 2)),
                    "verdict": "helps" if hi < 0 else "hurts" if lo > 0 else "undecided"})
    t = pd.DataFrame(out).sort_values(["signal", "setting"])
    print(t.round(4).to_string(index=False))
    print("\n`splits_helping` counts how many of the splits the difference is "
          "negative on.\nA verdict that holds at one split and not the others is "
          "not a property of the dataset.")


if __name__ == "__main__":
    raise SystemExit(main())
