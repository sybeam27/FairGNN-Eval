"""T1(d) control experiment -- does the common baseline B recover above-chance
AUC on German under a different optimiser setting?

Read-only with respect to every frozen artifact. It writes exactly one file,
results/phase0_audit/t1_baseline_probe.csv.

It reuses the repository's own loader (utils.dataloading.load_data), the
repository's own baseline model (models/algorithms/GNN.py, the class
harness/experiments/pilot_tau.py:457 instantiates as B), the repository's own
checkpoint selector (harness/core/trajectory.ValidationHistory, slot
"common_bce") and the repository's own evaluator (harness/core/evaluator.evaluate,
decision "score>0").

Three arms per (split, run):
  repro  published("GNN","german")      : hidden 16, lr 1e-3, wd 0.0, H=200,
                                          feature_normalize=False
  repro_H1000  the same, at the published horizon the NIFTY README states
                                        : hidden 16, lr 1e-3, wd 0.0, H=1000
  probe  the T1(d) setting              : hidden 64, lr 1e-2, wd 0.0, H=1000,
                                          feature_normalize=False
  probe_norm                            : probe + feature_normalize=True

Per-epoch validation BCE and validation AUC are recorded, because no stored
artifact holds them for B.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "harness"))

import core.paths  # noqa: F401,E402
from core.evaluator import evaluate  # noqa: E402
from core.trajectory import SplitRef, ValidationHistory  # noqa: E402
from models.algorithms.GNN import GNN  # noqa: E402
from utils.dataloading import load_data  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPLITS = [20, 21, 22, 23, 24, 25]
RUNS = 5
SEED0 = 27

ARMS = {
    "repro":      dict(num_hidden=16, num_proj_hidden=16, lr=1e-3,
                       weight_decay=0.0, epochs=200, normalize=False),
    "repro_H1000": dict(num_hidden=16, num_proj_hidden=16, lr=1e-3,
                        weight_decay=0.0, epochs=1000, normalize=False),
    "probe":      dict(num_hidden=64, num_proj_hidden=64, lr=1e-2,
                       weight_decay=0.0, epochs=1000, normalize=False),
    "probe_norm": dict(num_hidden=64, num_proj_hidden=64, lr=1e-2,
                       weight_decay=0.0, epochs=1000, normalize=True),
}


def split_ref(y, idx, sens):
    idx = np.asarray(idx.cpu() if torch.is_tensor(idx) else idx)
    yv = np.asarray(y.detach().cpu() if torch.is_tensor(y) else y)
    sv = np.asarray(sens.detach().cpu() if torch.is_tensor(sens) else sens)
    return SplitRef(node_id=idx, y=yv[idx].astype(int), a=sv[idx].astype(int))


def main() -> int:
    rows = []
    cache: dict[bool, tuple] = {}
    for split in SPLITS:
        cache.clear()
        for run in range(RUNS):
            seed = SEED0 + run
            for arm, a in ARMS.items():
                if a["normalize"] not in cache:
                    cache[a["normalize"]] = load_data(
                        "german", feature_normalize=a["normalize"],
                        split_seed=split)
                adj, f, y, itr, iva, ite, s, si = cache[a["normalize"]]
                ite_np = np.asarray(ite.cpu())

                torch.manual_seed(seed)
                np.random.seed(seed)
                m = GNN(adj, f, y, itr, iva, ite, s, si, device=DEV,
                        num_hidden=a["num_hidden"],
                        num_proj_hidden=a["num_proj_hidden"],
                        lr=a["lr"], weight_decay=a["weight_decay"])
                h = ValidationHistory(
                    split_ref(y, iva, s), a["epochs"] + 1,
                    state_fn=lambda _m=m: {k: v.detach().clone()
                                           for k, v in _m.state_dict().items()})
                m.fit(epochs=a["epochs"], trajectory=h)

                recs = h.classifier_records()
                val_auc = np.array([r.metrics["auc"] for r in recs], dtype=float)
                val_bce = np.array([r.predictive_loss for r in recs], dtype=float)

                def test_score(state, _m=m):
                    _m.load_state_dict(state)
                    _m.eval()
                    with torch.no_grad():
                        emb = _m.forward(_m.features.to(DEV),
                                         _m.edge_index.to(DEV))
                        out = _m.forwarding_predict(emb)
                    return out.squeeze().detach().cpu().numpy()[ite_np]

                ep_bce, st_bce = h.slots["common_bce"]
                q = test_score(st_bce)
                ref = split_ref(y, ite_np, s)
                g = evaluate(ref.y, ref.a, raw_score=q, decision="score>0")
                g_neg = evaluate(ref.y, ref.a, raw_score=-q, decision="score>0")

                rows.append(dict(
                    arm=arm, split_id=split, run_id=run, seed=seed,
                    hidden=a["num_hidden"], lr=a["lr"], epochs=a["epochs"],
                    feature_normalize=a["normalize"],
                    bc_epoch=ep_bce,
                    test_auc=g["auc"], test_auc_neg_score=g_neg["auc"],
                    test_dp=g["dp"], test_eo=g["eo"],
                    val_auc_at_bc=float(val_auc[ep_bce]),
                    val_auc_min=float(np.nanmin(val_auc)),
                    val_auc_max=float(np.nanmax(val_auc)),
                    val_auc_final=float(val_auc[-1]),
                    val_bce_at_bc=float(val_bce[ep_bce]),
                    val_bce_final=float(val_bce[-1]),
                    n_val_epochs_auc_below_half=int((val_auc < 0.5).sum()),
                    n_val_epochs=len(val_auc),
                ))
                print(f"{arm:11s} s{split} r{run}: bc_ep={ep_bce:4d} "
                      f"test_auc={g['auc']:.4f} val_auc@bc={val_auc[ep_bce]:.4f} "
                      f"val_auc_max={np.nanmax(val_auc):.4f}", flush=True)
                del m, h

    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "t1_baseline_probe.csv")
    df.to_csv(out, index=False)
    print("\nwrote", out)
    print(df.groupby("arm")[["test_auc", "val_auc_at_bc", "val_auc_max",
                             "bc_epoch"]].mean().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
