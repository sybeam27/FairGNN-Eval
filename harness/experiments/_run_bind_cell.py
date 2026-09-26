"""One (setting, split) cell of BIND for the audit, in the same CSV shape as
`utils/train_baselines.py` so `e7_baseline_audit.py` can read it unchanged.

Run as a subprocess like every other method, so a failure or an out-of-memory
in one cell cannot take the sweep down with it.

**Configuration.** BIND runs under *its own* published settings -- a one-layer
GCN with 16 hidden units, `weight_decay` 1e-4, 1000 epochs, and the per-dataset
influence `scale` its README specifies -- not under the shared protocol the
other rows use (two layers, 128 hidden, `weight_decay` 0, 500 epochs). Those
shared settings are what our own paper ran its baselines at; BIND was never in
that comparison, so there is no "as submitted" configuration for it here and
the only defensible choice is the authors'. The consequence is that BIND's row
differs from the others in backbone as well as in method, and that has to be
said wherever the table appears.

Usage
-----
    python -m study.experiments._run_bind_cell --dataset german \
        --split_seed 20 --seed 27 --runs 5 --output_file out.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from study.adapters.bind import SCALE, SCALE_FROM_PAPER, run   # noqa: E402
from utils.data import get_dataset                             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split_seed", type=int, default=20)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=1000)     # 1_training.py:27
    ap.add_argument("--hidden", type=int, default=16)       # 1_training.py:33
    ap.add_argument("--wd", type=float, default=1e-4)       # 1_training.py:31
    ap.add_argument("--lr", type=float, default=1e-3)       # 1_training.py:29
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()

    import torch
    from torch_geometric.utils import to_scipy_sparse_matrix

    data, _, _, _ = get_dataset(args.dataset, split_seed=args.split_seed)
    n = data.x.shape[0]
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=n).tocsr()
    A = ((A + A.T) > 0).astype(float)
    itr = data.train_mask.nonzero(as_tuple=False).view(-1)
    iva = data.val_mask.nonzero(as_tuple=False).view(-1)
    ite = data.test_mask.nonzero(as_tuple=False).view(-1)

    rows, t0 = [], time.time()
    for r in range(args.runs):
        seed = args.seed + r
        out = run(data.x.float(), data.edge_index, A, data.y.long(),
                  data.sens.long(), itr, iva, ite,
                  scale=SCALE[args.dataset], nhid=args.hidden, lr=args.lr,
                  wd=args.wd, epochs=args.epochs, device=args.device,
                  seed=seed)
        s = out["selected"]
        rows.append({"seed": seed, "k": s["k"], "max_num": out["max_num"],
                     "n_candidates": out["n_candidates"],
                     # a max_num of 0 read off a diverged estimate is not the
                     # same measurement as one read off a converged estimate
                     "scale_used": out["scale_used"],
                     "scale_steps": out["scale_steps"],
                     "acc": s["test_acc"], "roc_auc": s["test_auc"],
                     "f1": s["test_f1"], "dp": s["test_dp"], "eo": s["test_eo"]})
        print(f"  seed={seed} k={s['k']}/{out['max_num']} "
              f"scale={out['scale_used']} "
              f"acc={s['test_acc']:.4f} dp={s['test_dp']:.4f}", flush=True)

    d = pd.DataFrame(rows)
    summary = {"dataset": args.dataset, "task": "classification", "model": "BIND",
               "split_seed": args.split_seed, "seed": args.seed,
               "runs": args.runs, "lr": args.lr, "weight_decay": args.wd,
               "epochs": args.epochs, "hidden_dim": args.hidden,
               "bind_scale": SCALE[args.dataset],
               "bind_scale_from_paper": args.dataset in SCALE_FROM_PAPER,
               "bind_k_mean": d.k.mean(), "bind_max_num_mean": d.max_num.mean(),
               "bind_scale_used_mean": d.scale_used.mean(),
               "bind_scale_steps_mean": d.scale_steps.mean(),
               "bind_n_candidates_mean": d.n_candidates.mean(),
               "time_sec_mean": round(time.time() - t0, 1) / max(args.runs, 1)}
    for m in ("acc", "roc_auc", "f1", "dp", "eo"):
        summary[f"{m}_mean"] = d[m].mean()
        summary[f"{m}_std"] = d[m].std(ddof=1) if len(d) > 1 else 0.0
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    pd.DataFrame([summary]).to_csv(args.output_file, index=False)
    print(f"[saved] {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
