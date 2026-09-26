"""X24 diagnostic: realized NIFTY validation views for every (split, run), per D level.

No training. For each D level, split 20-25 and run 0-4, this rebuilds the
NIFTY object on the experiment device with the pipeline's own seed sequence:

    torch.manual_seed(seed * 1000 + split)     # pilot main(), before train()
    torch.manual_seed(seed); np.random.seed(seed)   # pilot train()
    NIFTY(...)                                 # encoder init, then the views

It records the kept-edge counts of the two fixed validation views, the number
of perturbed feature columns, and the training drop rates fit() will use. The
output describes the configuration only; it is not an outcome and does not enter
the τ_int analysis.

    CUDA_VISIBLE_DEVICES=2 python harness/experiments/x24_view_probe.py --out /tmp/.../x24_view_probe.csv
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
import core.paths  # noqa: F401  (puts the repository root on sys.path)
from models.algorithms.NIFTY import NIFTY                                    # noqa: E402
from core.published_config import native_config, published           # noqa: E402

DROP_KEYS = ("drop_edge_rate_1", "drop_edge_rate_2",
             "drop_feature_rate_1", "drop_feature_rate_2")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", type=int, default=[20, 21, 22, 23, 24, 25])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=27)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")

    levels = {"D0": published("NIFTY", "german")["config"],
              "D1": native_config("NIFTY", "german")["config"]}
    rows = []
    for D, cfg in levels.items():
        drops = {k: cfg[k] for k in DROP_KEYS if k in cfg}
        for split in a.splits:
            P.DS[0], P.SPLIT[0] = "german", split
            data = P.load("german", split, a.device,
                          feature_normalize=bool(cfg.get("feature_normalize", False)))
            adj, f, y, itr, iva, ite, s, si = data
            E = int(adj.coalesce().indices().shape[1])
            for run in range(a.runs):
                seed = a.seed0 + run
                torch.manual_seed(seed * 1000 + split)
                torch.manual_seed(seed); np.random.seed(seed)
                m = NIFTY(adj, f, y, itr, iva, ite, s, si,
                          num_hidden=cfg.get("num_hidden", 128),
                          num_proj_hidden=cfg.get("num_proj_hidden", 128),
                          lr=cfg.get("lr", 1e-3),
                          weight_decay=cfg.get("weight_decay", 1e-5), device=a.device,
                          sim_coeff=cfg.get("sim_coeff", 0.5), **drops)
                if cfg.get("restore_train_drop_rates"):
                    for k, v in drops.items():
                        setattr(m, k, v)
                k1 = int(m.val_edge_index_1.shape[1]); k2 = int(m.val_edge_index_2.shape[1])
                rows.append(dict(
                    D=D, split_id=split, run_id=run, seed=seed, n_edges=E,
                    view1_edges_kept=k1, view2_edges_kept=k2,
                    view1_edge_drop=1 - k1 / E, view2_edge_drop=1 - k2 / E,
                    view1_feat_cols_perturbed=int((m.val_x_1.detach().cpu() != f.cpu()).any(0).sum()),
                    view2_feat_cols_perturbed=int((m.val_x_2.detach().cpu() != f.cpu()).any(0).sum()),
                    n_feat_cols=int(f.shape[1]),
                    train_edge_drop_1=float(m.drop_edge_rate_1),
                    train_edge_drop_2=float(m.drop_edge_rate_2),
                    train_feat_drop_1=float(m.drop_feature_rate_1),
                    train_feat_drop_2=float(m.drop_feature_rate_2)))
                del m
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(out.groupby("D")[["view1_edge_drop", "view2_edge_drop",
                            "view1_feat_cols_perturbed", "train_edge_drop_1",
                            "train_feat_drop_1"]].agg(["mean", "min", "max"]).to_string())
    print(f"\n[written] {a.out}: {len(out)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
