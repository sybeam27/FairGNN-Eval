"""Re-train the common baseline B under the rules fixed in results/phase0_audit/B_rebuild_decision.md.

One baseline per (dataset, split_id, run_id), shared by every method that references it. The method
arms are never re-trained: the frozen per-unit values stand, and only the B-referenced contrasts are
recomputed downstream. This script trains and stores B; it computes no contrast.

It mirrors the baseline block of `pilot_tau.py:450-485` exactly — same loader, same `published("GNN",
dataset)` configuration, same `GNN` class, same `ValidationHistory` slots, same evaluator — and
changes one thing: the horizon comes from the variant, not from the CLI default.

    B_rep1     the resolved published horizon where one exists (german: 1000), else 200   [reported]
    B_H1000    1000 everywhere                                              [baseline-strength check]
    B_rep2     identical to B_rep1, independently re-trained             [nondeterminism check]

    CUDA_VISIBLE_DEVICES=2 python harness/experiments/rebuild_baseline.py --variant B_rep1
    CUDA_VISIBLE_DEVICES=2 python harness/experiments/rebuild_baseline.py --variant B_rep1 --datasets german

Writes results_v2/baselines/<variant>/B_<dataset>.csv, one row per (split, run, selector):
    split_id, run_id, seed, selector, epoch, auc, dp, eo, eo_defined, horizon, config, variant
Nothing under results/ or harness/results/ is read for writing or modified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "harness"))
import core.paths  # noqa: F401,E402  (puts the repository root on sys.path)
from core.evaluator import evaluate  # noqa: E402
from core.published_config import published  # noqa: E402
from core.trajectory import ValidationHistory  # noqa: E402

DATASETS = ["german", "bail", "credit", "income", "pokec_z", "pokec_n", "pokec_z_g", "pokec_n_g"]
SPLITS, RUNS, SEED0 = [20, 21, 22, 23, 24, 25], 5, 27
CONTROLLED_H = 200          # the horizon the frozen bundle used for every baseline


def horizon_for(variant: str, dataset: str, b_pub) -> tuple[int, str]:
    """(horizon, why) under the rule fixed in the decision note."""
    if variant == "B_H1000":
        return 1000, "variant: 1000 on every dataset"
    h = b_pub.get("horizon")
    if h:
        return int(h), f"resolved b_pub horizon ({b_pub.get('source')})"
    return CONTROLLED_H, "no published horizon resolves; the controlled default stands"


def outcome(y, s, idx, score):
    """The unified measurement operator, called exactly as the frozen runs call it."""
    yv = np.asarray(y.cpu() if torch.is_tensor(y) else y)
    sv = np.asarray(s.cpu() if torch.is_tensor(s) else s)
    return evaluate(yv[idx].astype(int), sv[idx].astype(int), raw_score=score, decision="score>0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["B_rep1", "B_H1000", "B_rep2"])
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--splits", nargs="*", type=int, default=SPLITS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_v2", "baselines"))
    a = ap.parse_args()

    if a.device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise SystemExit("this rebuild runs on GPU 2 only: set CUDA_VISIBLE_DEVICES=2")

    import pilot_tau as P                      # the frozen loader, split reference and outcome path
    _split_ref = P._split_ref
    from models.algorithms.GNN import GNN

    out_dir = os.path.join(a.out, a.variant)
    os.makedirs(out_dir, exist_ok=True)
    dev = a.device

    for ds in a.datasets:
        b_pub = published("GNN", ds)
        b_cfg = dict(b_pub["config"])
        H, why = horizon_for(a.variant, ds, b_pub)
        norm = bool(b_cfg.pop("feature_normalize", False))
        rows, t0 = [], time.time()
        print(f"[{a.variant}] {ds}: H={H} ({why}); config={b_cfg}", flush=True)

        for split in a.splits:
            data = P.load(ds, split, feature_normalize=norm, device=dev)
            adj, f, y, itr, iva, ite, s, si = data
            ite_np = np.asarray(ite.cpu() if torch.is_tensor(ite) else ite)
            for run in range(a.runs):
                seed = a.seed0 + run
                torch.manual_seed(seed); np.random.seed(seed)          # pilot_tau.py:458
                b = GNN(adj, f, y, itr, iva, ite, s, si, device=dev, **b_cfg)
                hb = ValidationHistory(_split_ref(y, iva, s), H + 1,
                                       state_fn=lambda: {k: v.detach().clone()
                                                         for k, v in b.state_dict().items()})
                b.fit(epochs=H, trajectory=hb)

                def score(state, idx):
                    if state is not None:
                        b.load_state_dict(state)
                    b.eval()
                    with torch.no_grad():
                        emb = b.forward(b.features.to(dev), b.edge_index.to(dev))
                        out = b.forwarding_predict(emb)
                    return out.squeeze().detach().cpu().numpy()[np.asarray(idx)]

                for sel in ("common_bce", "common_auc"):
                    if sel not in hb.slots:
                        print(f"  s{split} r{run}: slot {sel} missing", flush=True)
                        continue
                    ep, st = hb.slots[sel]
                    m = outcome(y, s, ite_np, score(st, ite_np))
                    rows.append(dict(split_id=split, run_id=run, seed=seed, selector=sel,
                                     epoch=int(ep), auc=m["auc"], dp=m["dp"], eo=m["eo"],
                                     eo_defined=bool(m["eo_defined"]), horizon=H,
                                     config=json.dumps(b_cfg, sort_keys=True), variant=a.variant))
                print(f"  s{split} r{run}: done ({time.time() - t0:.0f}s elapsed)", flush=True)

        import pandas as pd
        d = pd.DataFrame(rows)
        n_units = d.groupby(["split_id", "run_id"]).ngroups
        if n_units != len(a.splits) * a.runs:                          # stop condition
            raise SystemExit(f"[{a.variant}] {ds}: {n_units} units, expected {len(a.splits) * a.runs}")
        path = os.path.join(out_dir, f"B_{ds}.csv")
        d.to_csv(path, index=False)
        print(f"[{a.variant}] {ds}: wrote {path} ({len(d)} rows, {time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
