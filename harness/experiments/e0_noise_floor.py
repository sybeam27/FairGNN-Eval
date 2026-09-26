"""
E0 -- how large is the effect we are trying to measure, relative to the noise?

Motivation. On this box the same seed does not give the same answer:

    german / uniform / seed 27      CPU 1 thread   dp = 0.0164
                                    CPU 8 threads  dp = 0.0792
                                    CUDA           dp = 0.0996 .. 0.1442

CPU is deterministic *within* a thread count; CUDA is not deterministic at all,
because cuSPARSE spmm reorders its reduction. The perturbation is ~1e-7
relative on a single forward pass, so this is not a bug to fix -- it is
evidence that 300 epochs of training amplify float noise into an O(0.05) swing
in the reported disparity. `torch.use_deterministic_algorithms(True)` does not
help: neither the COO nor the CSR spmm has a deterministic CUDA kernel.

That matters because the draft's headline is uniform 0.039 -> adaptive 0.028,
a difference of 0.011. If a run repeated at the *same* seed already moves more
than that, then a seed is not a replicate and five of them prove nothing.

So before running E2 for real, measure two things per setting:

    within-seed sd   repeat one seed R times          (float-noise only)
    across-seed sd   run R different seeds once each  (seed + float noise)

If they are comparable, the seed controls nothing that matters, "replicate"
is the unit of randomness, and the required R follows from the effect size:
to resolve a paired difference d with a 95% CI of half-width d/2, one needs
roughly R >= (4 * sd_paired / d)^2.

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python harness/experiments/e0_noise_floor.py \
        --device cuda --reps 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import datasets as D          # noqa: E402
from core.trainer import Config, train  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

METRICS = ["acc", "auc", "dp", "eo"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--signals", nargs="*", default=["uniform"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--fixed-seed", type=int, default=27)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lambda-fair", type=float, default=0.2)
    ap.add_argument("--q-gate", type=float, default=0.7)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    datasets = args.datasets or D.ALL_SETTINGS
    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(RESULTS, "e0_noise_floor.csv")

    # arm='within': one seed, repeated.   arm='across': one run each, R seeds.
    plan = ([("within", args.fixed_seed, r) for r in range(args.reps)] +
            [("across", args.fixed_seed + r, 0) for r in range(args.reps)])

    rows, t0 = [], time.time()
    for ds in datasets:
        g = D.load(ds)
        print(f"\n{g}")
        for sig in args.signals:
            for arm, seed, rep in plan:
                t = time.time()
                cfg = Config(signal=sig, seed=seed, epochs=args.epochs,
                             warmup=args.warmup, lambda_fair=args.lambda_fair,
                             q_gate=args.q_gate, device=args.device)
                try:
                    r = train(g, cfg)
                except Exception as e:                                # noqa: BLE001
                    print(f"  {arm:<7s} {sig:<12s} seed={seed} rep={rep} "
                          f"FAILED {type(e).__name__}: {e}")
                    continue
                rows.append({"setting": ds, "regime": g.regime_paper, "signal": sig,
                             "arm": arm, "seed": seed, "rep": rep,
                             "device": args.device,
                             **{f"test_{k}": r.test[k] for k in METRICS},
                             "secs": round(time.time() - t, 1)})
            sub = pd.DataFrame([x for x in rows if x["setting"] == ds and x["signal"] == sig])
            w = sub[sub.arm == "within"]["test_dp"]
            a = sub[sub.arm == "across"]["test_dp"]
            print(f"  {sig:<12s} within-seed dp {w.mean():.4f} +- {w.std(ddof=1):.4f}   "
                  f"across-seed dp {a.mean():.4f} +- {a.std(ddof=1):.4f}")
        pd.DataFrame(rows).to_csv(out, index=False)        # checkpoint per dataset

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}   ({len(df)} runs, {time.time() - t0:.0f}s)\n")
    report(df)
    return 0


def report(df: pd.DataFrame) -> None:
    if df.empty:
        print("no runs completed")
        return
    pd.set_option("display.width", 220)

    print("=" * 100)
    print("NOISE FLOOR   sd over replicates.  'draft effect' = 0.011 (uniform 0.039 -> adaptive 0.028)")
    print("=" * 100)
    out = []
    for (ds, sig), sub in df.groupby(["setting", "signal"]):
        row = {"setting": ds, "regime": sub["regime"].iloc[0], "signal": sig}
        for arm in ("within", "across"):
            s = sub[sub.arm == arm]
            for m in ("dp", "eo"):
                row[f"{arm}_{m}_sd"] = round(s[f"test_{m}"].std(ddof=1), 4)
            row[f"{arm}_dp_mean"] = round(s["test_dp"].mean(), 4)
        # Replicates needed for a 95% CI half-width of 0.0055 on a 0.011 effect.
        # Pairing (both arms at the same seeds) cancels the seed main effect, so
        # sd(difference) ~ sqrt(2) * sd_within if there is no signal x seed
        # interaction; without pairing the full across-seed sd is carried.
        # R_paired is therefore a lower bound and must be re-checked against the
        # empirical paired sd once two arms have actually been run.
        for tag, sd in (("paired", row["within_dp_sd"]), ("unpaired", row["across_dp_sd"])):
            row[f"R_{tag}"] = int(np.ceil((4 * np.sqrt(2) * sd / 0.011) ** 2)) if sd > 0 else 0
        out.append(row)
    t = pd.DataFrame(out).sort_values("R_paired", ascending=False)
    print(t.to_string(index=False))

    print("\nRead: R_paired / R_unpaired are the replicates needed for a 95% CI "
          "\nhalf-width of 0.0055 on a 0.011 effect at that setting. Pairing is not "
          "\noptional -- it is the difference between a feasible experiment and an "
          "\ninfeasible one. Where R_paired is still large, the setting cannot settle "
          "\nthe question at any budget and must be reported as uninformative rather "
          "\nthan as evidence.")


if __name__ == "__main__":
    raise SystemExit(main())
