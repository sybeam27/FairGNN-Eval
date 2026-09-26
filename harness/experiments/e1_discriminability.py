"""
E1 -- the right half of the discriminability hypothesis.

E2 measures, for each (setting, signal), how much allocating by that signal
helps. This measures, for the same pairs, how *discriminable* the signal is on
that graph at the point where the choice would actually be made. H1 in
`harness/PREREGISTRATION.md` is the claim that the second predicts the first.

The decision point is after warm-up and before fairness training, because
`loss`, `fairgb_group` and `bind_influence` are functions of a trained model.
That boundary already exists in the method -- `T_warm` epochs run at
`lambda_fair = 0` either way -- so reaching it costs one warm-up pass, against
eight full trainings for the validation-selection control. For the purely
structural signals nothing is paid at all: their statistics do not depend on the
model and are identical across seeds, which the output makes checkable rather
than assumed.

The primary statistic is fixed by the pre-registration and is not chosen here:

    D(g, G) = MI(g(v); s_v) over v in I, 10 equal-width bins

Everything else in the output is secondary and exists to be reported alongside,
not to be selected from after the fact.

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python harness/experiments/e1_discriminability.py \
        --device cuda --seeds 27 28 29 30 31
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import datasets as D            # noqa: E402
from core import signals as S             # noqa: E402
from core.trainer import Config, warmup_state   # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Closed at eight by the pre-registration; do not extend after seeing E2.
SIGNALS = ["uniform", "random", "w_bdry", "w_deg", "w_lhd",
           "loss", "fairgb_group", "bind_influence"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--signals", nargs="*", default=SIGNALS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[27, 28, 29, 30, 31])
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--q-gate", type=float, default=0.7)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    datasets = args.datasets or D.ALL_SETTINGS
    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(RESULTS, "e1_discriminability.csv")

    rows, t0 = [], time.time()
    for ds in datasets:
        g = D.load(ds)
        idx = np.asarray(g.idx_fair)
        print(f"\n{g}")
        for seed in args.seeds:
            cfg = Config(seed=seed, warmup=args.warmup, q_gate=args.q_gate,
                         device=args.device)
            st = warmup_state(g, cfg)
            for sig in args.signals:
                spec = S.get(sig)
                try:
                    d = S.describe(sig, g, st, q_gate=args.q_gate, idx=idx)
                except Exception as e:                                # noqa: BLE001
                    print(f"  {sig:<15s} seed={seed} FAILED {type(e).__name__}: {e}")
                    continue
                d |= {"seed": seed, "needs_model": spec.needs_model,
                      "warmup": args.warmup}
                rows.append(d)
        sub = pd.DataFrame([r for r in rows if r["setting"] == g.name])
        agg = sub.groupby("signal")["mi_with_sens"].agg(["mean", "std"])
        for sig in args.signals:
            if sig in agg.index:
                m, sd = agg.loc[sig, "mean"], agg.loc[sig, "std"]
                print(f"  D({sig:<15s}) = {m:.4f} +- {0.0 if pd.isna(sd) else sd:.4f}")
        pd.DataFrame(rows).to_csv(out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}   ({len(df)} rows, {time.time() - t0:.0f}s)\n")
    report(df)
    return 0


def report(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    print("=" * 100)
    print("D(g, G) = MI(g; s) on I   -- the primary statistic, mean over seeds")
    print("=" * 100)
    piv = df.pivot_table(index="setting", columns="signal",
                         values="mi_with_sens", aggfunc="mean")
    order = [c for c in SIGNALS if c in piv.columns]
    print(piv[order].round(4).to_string())

    print("\nmost discriminable signal per setting (the argmax rule of H2):")
    best = piv[[c for c in order if c not in ("uniform", "random")]].idxmax(axis=1)
    reg = df.groupby("setting")["regime"].first()
    print(pd.DataFrame({"regime": reg, "argmax_D": best,
                        "max_D": piv[[c for c in order
                                      if c not in ("uniform", "random")]].max(axis=1).round(4)}
                       ).to_string())

    print("\nsanity: within a setting, a structural signal must not vary with "
          "the seed (max sd over seeds, per signal, should be 0 for those)")
    sd = (df.groupby(["setting", "signal"])["mi_with_sens"].std(ddof=1)
            .groupby("signal").max().reset_index()
            .rename(columns={"mi_with_sens": "max_sd_over_seeds"}))
    sd["needs_model"] = sd["signal"].map(
        df.drop_duplicates("signal").set_index("signal")["needs_model"])
    print(sd.round(6).to_string(index=False))

    print("\nprecision of D: mi_null_sd, the spread of the permutation null. "
          "A D\nsmaller than this is not distinguishable from a signal "
          "unrelated to s.")
    prec = df.pivot_table(index="setting", columns="signal",
                          values="mi_null_sd", aggfunc="mean")
    print(prec[[c for c in SIGNALS if c in prec.columns]].round(4).to_string())


if __name__ == "__main__":
    raise SystemExit(main())
