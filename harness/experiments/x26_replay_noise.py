"""X26 diagnostic: CUDA replay-noise envelope for FairGB on the test path.

Measures numerical replay noise only. It computes no tau, no decomposition and
no scientific quantity, and it runs *before* any X26 decomposition is computed,
so the envelope cannot be chosen from an effect estimate. Same procedure and
same criterion form as X25 (X14 native gate, X25 amendment 326f04f):

    bound = max(1e-5, 4 x max replay spread)

For each requested cell it trains FairGB exactly as the pipeline does, then at
each of the four selector slots (cap/full x BCE/AUC) restores that checkpoint
and replays the pipeline's test scoring `--replays` times, reporting:

    replay spread        max |replay_i - replay_j| over the repeats
    stored vs replayed   max |recorded slot score - replay_0|
    sign flips           test nodes whose hard prediction differs, with scores,
                         difference and distance to the decision threshold

    CUDA_VISIBLE_DEVICES=2 python harness/experiments/x26_replay_noise.py \
        --dataset bail --epochs 1500 --cells 20:0 21:0 --out /tmp/.../x26_noise_bail.csv
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
import experiments.x26_fairgb_run as X                                # noqa: E402
from core.published_config import native_config                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("bail", "credit"))
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--cells", nargs="+", default=["20:0", "21:0"])
    ap.add_argument("--replays", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=27)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")

    cfg = native_config("FairGB", a.dataset)["config"]
    X.install(wrap_train=False)
    rows, flips = [], []
    for cell in a.cells:
        sp, rn = (int(v) for v in cell.split(":"))
        seed = a.seed0 + rn
        P.DS[0], P.SPLIT[0] = a.dataset, sp
        data = P.load(a.dataset, sp, a.device,
                      feature_normalize=bool(cfg.get("feature_normalize", False)))
        for arm, off in (("M1", None), ("M0", P.METHODS["FairGB"]["off"])):
            torch.manual_seed(seed * 1000 + sp)
            fn, hist, _e, traj = X.record_train("FairGB", data, seed, a.epochs, a.device,
                                                off=off, cfg=cfg, grid=X.GRID[a.dataset],
                                                dataset=a.dataset)
            test_idx = np.asarray(traj["test_node_id"])
            cap = {"bce": X.SLOT["bce"], "auc": X.SLOT["auc"]}
            for sig in X.SIGMAS:
                for supp in ("cap", "full"):
                    ep = int(traj[f"slot_epoch_{supp}_{sig}"])
                    stored = np.asarray(traj[f"slot_score_{supp}_{sig}"], dtype=np.float64)
                    # restore that checkpoint again and replay the same scoring
                    src = hist.slots[cap[sig]] if supp == "full" else None
                    if src is None:
                        # cap slots are not on the history any more; replay the
                        # full-support slot only when the epochs coincide
                        if ep != int(hist.slots[cap[sig]][0]):
                            continue
                        src = hist.slots[cap[sig]]
                    reps = [np.asarray(fn(src[1], test_idx), dtype=np.float64)
                            for _ in range(a.replays)]
                    spread = max(float(np.max(np.abs(reps[i] - reps[j])))
                                 for i in range(len(reps)) for j in range(i + 1, len(reps)))
                    dev = float(np.max(np.abs(reps[0] - stored)))
                    flip = (reps[0] > 0) != (stored > 0)
                    rows.append(dict(dataset=a.dataset, split=sp, run=rn, arm=arm,
                                     support=supp, selector=sig, epoch=ep,
                                     replay_spread=spread, stored_vs_replayed=dev,
                                     n_sign_flips=int(flip.sum()),
                                     min_abs_stored=float(np.min(np.abs(stored))),
                                     n_test=int(len(test_idx))))
                    for k in np.nonzero(flip)[0]:
                        flips.append(dict(dataset=a.dataset, split=sp, run=rn, arm=arm,
                                          support=supp, selector=sig, epoch=ep,
                                          node=int(test_idx[k]), stored=float(stored[k]),
                                          replayed=float(reps[0][k]),
                                          abs_diff=float(abs(stored[k] - reps[0][k])),
                                          dist_thr_stored=float(abs(stored[k]))))
                    print(f"  {a.dataset} s{sp} r{rn} {arm} {supp}/{sig} ep {ep}: "
                          f"spread {spread:.3e}, stored-vs-replayed {dev:.3e}, "
                          f"flips {int(flip.sum())}, min|stored| {float(np.min(np.abs(stored))):.3e}",
                          flush=True)
    X.uninstall()
    r = pd.DataFrame(rows)
    r.to_csv(a.out, index=False)
    pd.DataFrame(flips).to_csv(a.out.replace(".csv", "_flips.csv"), index=False)
    print(f"\nslot evaluations: {len(r)}")
    print(f"replay spread:      max {r.replay_spread.max():.3e}  median {r.replay_spread.median():.3e}")
    print(f"stored vs replayed: max {r.stored_vs_replayed.max():.3e}  median {r.stored_vs_replayed.median():.3e}")
    print(f"bound = max(1e-5, 4 x max spread) = {max(1e-5, 4 * r.replay_spread.max()):.3e}")
    print(f"sign flips: {int(r.n_sign_flips.sum())}")
    print(f"[written] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
