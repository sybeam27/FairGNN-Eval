"""X25 diagnostic: CUDA replay-noise envelope for NIFTY/German on the test path.

Measures numerical replay noise only. It computes no tau, no decomposition and
no scientific quantity, and it runs *before* the G3 amendment is written, so the
envelope cannot be chosen from any effect estimate.

For each requested cell and D level it trains NIFTY exactly as the pipeline
does (same config, same seeds), recording per-epoch in-training test scores
through the X25 wrapper. At each shared-selector slot epoch it then restores
that checkpoint and replays the pipeline's own test scoring `--replays` times,
and reports:

    replay spread        max |replay_i - replay_j| over the repeats (pure device noise)
    stored vs replayed   max |stored - replay_0| at the same epoch
    sign flips           test nodes whose hard prediction differs between the
                         stored and replayed evaluation, with both scores, their
                         absolute difference, and the distance to the decision
                         threshold in each evaluation

The pre-existing project criterion (X14, native gate, reaffirmed in X20) is
`bound = max(1e-5, 4 x replay spread)`, measured there on the validation path.
This script measures the same quantities on the test path at the X25 horizon.

    CUDA_VISIBLE_DEVICES=2 python harness/experiments/x25_replay_noise.py \
        --cells 20:3 20:0 21:0 --levels D0 D1 --epochs 1000 --out /tmp/.../x25_replay_noise.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "harness"))

import experiments.pilot_tau as P                                     # noqa: E402
import experiments.x25_trajectory_run as X                            # noqa: E402
from core.published_config import native_config, published            # noqa: E402

LEVELS = {"D0": lambda: published("NIFTY", "german")["config"],
          "D1": lambda: native_config("NIFTY", "german")["config"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["20:3", "20:0", "21:0"],
                    help="split:run pairs")
    ap.add_argument("--levels", nargs="+", default=["D0", "D1"])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--replays", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=27)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")

    X.install(wrap_train=False)
    rows, flips = [], []
    for D in a.levels:
        cfg = LEVELS[D]()
        for cell in a.cells:
            sp, rn = (int(v) for v in cell.split(":"))
            seed = a.seed0 + rn
            P.DS[0], P.SPLIT[0] = "german", sp
            data = P.load("german", sp, a.device,
                          feature_normalize=bool(cfg.get("feature_normalize", False)))
            ite = np.asarray(data[5].cpu() if torch.is_tensor(data[5]) else data[5]).astype(int)
            for arm, off in (("M1", None), ("M0", P.METHODS["NIFTY"]["off"])):
                torch.manual_seed(seed * 1000 + sp)
                fn, hist, _e, traj = X.record_train("NIFTY", data, seed, a.epochs, a.device,
                                                    off=off, cfg=cfg)
                ep_index = {int(e): i for i, e in enumerate(traj["epochs"])}
                for sel, (ep, st) in hist.slots.items():
                    stored = np.asarray(traj["test_raw"][ep_index[int(ep)]], dtype=np.float64)
                    reps = [np.asarray(fn(st, ite), dtype=np.float64) for _ in range(a.replays)]
                    spread = max(float(np.max(np.abs(reps[i] - reps[j])))
                                 for i in range(len(reps)) for j in range(i + 1, len(reps)))
                    dev = float(np.max(np.abs(reps[0] - stored)))
                    flip = (reps[0] > 0) != (stored > 0)
                    rows.append(dict(D=D, split=sp, run=rn, arm=arm, selector=sel, epoch=int(ep),
                                     replay_spread=spread, stored_vs_replayed=dev,
                                     n_sign_flips=int(flip.sum()),
                                     min_abs_stored=float(np.min(np.abs(stored))),
                                     n_test=int(len(ite))))
                    for k in np.nonzero(flip)[0]:
                        flips.append(dict(D=D, split=sp, run=rn, arm=arm, selector=sel, epoch=int(ep),
                                          node=int(ite[k]), stored=float(stored[k]),
                                          replayed=float(reps[0][k]),
                                          abs_diff=float(abs(stored[k] - reps[0][k])),
                                          dist_thr_stored=float(abs(stored[k])),
                                          dist_thr_replayed=float(abs(reps[0][k]))))
                    print(f"  {D} s{sp} r{rn} {arm} {sel} ep {ep}: replay spread {spread:.3e}, "
                          f"stored-vs-replayed {dev:.3e}, sign flips {int(flip.sum())}, "
                          f"min|stored| {float(np.min(np.abs(stored))):.3e}", flush=True)
    X.uninstall()
    r = pd.DataFrame(rows)
    r.to_csv(a.out, index=False)
    fp = a.out.replace(".csv", "_flips.csv")
    pd.DataFrame(flips).to_csv(fp, index=False)
    print(f"\nslot evaluations: {len(r)}")
    print(f"replay spread:      max {r.replay_spread.max():.3e}  median {r.replay_spread.median():.3e}")
    print(f"stored vs replayed: max {r.stored_vs_replayed.max():.3e}  median {r.stored_vs_replayed.median():.3e}")
    print(f"pre-existing criterion form max(1e-5, 4 x spread): max over slots "
          f"{max(1e-5, 4 * r.replay_spread.max()):.3e}")
    print(f"sign flips between stored and replayed: {int(r.n_sign_flips.sum())} "
          f"over {int(r.n_test.iloc[0])} test nodes x {len(r)} slot evaluations")
    print(f"[written] {a.out} and {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
