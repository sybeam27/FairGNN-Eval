"""
E2 -- the central experiment: hold the pipeline fixed, swap only the ranking signal.

E1 asks whether allocating non-uniformly helps at all (uniform vs the rest).
E2 asks whether *which* signal you allocate by matters, or whether any
non-uniform allocation would do. Those are different questions, and only the
second one is what the paper claims.

The prediction being tested: each fixed criterion wins in the regime it suits
and loses elsewhere. If instead one criterion wins everywhere, the
regime-adaptive story is wrong and there is no paper. If nothing beats
`uniform`, allocation itself does not help and there is also no paper. Either
outcome is worth four weeks.

Usage
-----
    # pilot: small graphs, few arms, quick
    python harness/experiments/e2_signal_swap.py --preset pilot

    # full sweep
    python harness/experiments/e2_signal_swap.py --preset full

    python harness/experiments/e2_signal_swap.py \
        --datasets german nba --signals uniform w_bdry w_deg --seeds 27 28
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import datasets as D          # noqa: E402
from core.trainer import Config, train  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Small graphs first: these run in seconds on CPU and settle go/no-go.
PILOT_DATASETS = ["german", "nba", "income", "recidivism"]
PILOT_SIGNALS = ["uniform", "random", "w_bdry", "w_deg"]

FULL_SIGNALS = ["uniform", "random", "w_bdry", "w_deg", "w_lhd",
                "loss", "fairgb_group", "bind_influence"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["pilot", "full"], default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--signals", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[27, 28, 29])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lambda-fair", type=float, default=0.2)
    ap.add_argument("--q-gate", type=float, default=0.7)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=0,
                    help="cap torch CPU threads (0 = leave alone). "
                         "This box runs at load ~28; uncapped BLAS "
                         "grabs every core and contends.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    if args.preset == "pilot":
        datasets = args.datasets or PILOT_DATASETS
        sigs = args.signals or PILOT_SIGNALS
    elif args.preset == "full":
        datasets = args.datasets or D.ALL_SETTINGS
        sigs = args.signals or FULL_SIGNALS
    else:
        datasets = args.datasets or PILOT_DATASETS
        sigs = args.signals or PILOT_SIGNALS

    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(RESULTS, "e2_signal_swap.csv")

    print(f"datasets={datasets}\nsignals={sigs}\nseeds={args.seeds}")
    print(f"runs = {len(datasets) * len(sigs) * len(args.seeds)}\n")

    rows, t0 = [], time.time()
    for ds in datasets:
        g = D.load(ds)                          # provenance-checked
        print(f"{g}")
        for sig, seed in itertools.product(sigs, args.seeds):
            t = time.time()
            cfg = Config(signal=sig, seed=seed, epochs=args.epochs,
                         warmup=args.warmup, lambda_fair=args.lambda_fair,
                         q_gate=args.q_gate, device=args.device)
            try:
                r = train(g, cfg, verbose=args.verbose)
                row = r.row() | {"regime": g.regime_paper,
                                 "device": args.device,
                                 "secs": round(time.time() - t, 1)}
                rows.append(row)
                print(f"  {sig:<15s} seed={seed}  acc={r.test['acc']:.3f} "
                      f"auc={r.test['auc']:.3f} dp={r.test['dp']:.4f} "
                      f"eo={r.test['eo']:.4f}  ({row['secs']}s)")
            except Exception as e:                                    # noqa: BLE001
                print(f"  {sig:<15s} seed={seed}  FAILED {type(e).__name__}: {e}")
        pd.DataFrame(rows).to_csv(out, index=False)     # checkpoint per dataset

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}   ({len(df)} runs, {time.time() - t0:.0f}s total)\n")
    report(df)
    return 0


def report(df: pd.DataFrame) -> None:
    if df.empty:
        print("no runs completed")
        return
    pd.set_option("display.width", 200)

    agg = (df.groupby(["setting", "signal"])[["test_acc", "test_auc", "test_dp", "test_eo"]]
             .mean().round(4).reset_index())
    print("=" * 78)
    print("PER SETTING (mean over seeds)")
    print("=" * 78)
    for ds, sub in agg.groupby("setting"):
        best = sub.loc[sub["test_dp"].idxmin(), "signal"]
        print(f"\n{ds}   (lowest dp: {best})")
        print(sub.drop(columns="setting").to_string(index=False))

    print("\n" + "=" * 78)
    print("BY REGIME (mean over settings and seeds)  -- the E2 prediction")
    print("=" * 78)
    piv = (df.groupby(["regime", "signal"])[["test_acc", "test_auc", "test_dp", "test_eo"]]
             .mean().round(4))
    print(piv.to_string())

    print("\n" + "=" * 78)
    print("vs UNIFORM  (negative dp_delta = allocation helped)")
    print("=" * 78)
    base = (df[df.signal == "uniform"].groupby("setting")[["test_dp", "test_eo"]]
              .mean().rename(columns=lambda c: c + "_uniform"))
    m = agg.merge(base, on="setting", how="left")
    m["dp_delta"] = (m["test_dp"] - m["test_dp_uniform"]).round(4)
    m["eo_delta"] = (m["test_eo"] - m["test_eo_uniform"]).round(4)
    print(m[m.signal != "uniform"][["setting", "signal", "dp_delta", "eo_delta"]]
          .to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
