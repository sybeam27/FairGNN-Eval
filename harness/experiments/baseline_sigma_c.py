"""The common baseline B evaluated under the common protocol σ_c.

`pilot_tau.py` records the baseline only at its own code-native selector
(`b.best_state`), which is the published-protocol view. That is the right
reference for τ_pkg^pub, but it cannot serve the audit decomposition

    τ_pkg^audit = τ_base^audit + τ_int

because every term there must be read at the same σ_c. This pass reproduces
the baseline exactly -- the pilot seeds it with `torch.manual_seed(seed)`
immediately before construction, independently of the method loop, so the same
(split, run, device) reproduces it bit for bit -- and writes B under both
common selectors.

`bpub_*` is written alongside as the reproduction check: it must equal the
`b_*` columns already in the pilot CSV.

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python harness/experiments/baseline_sigma_c.py \
        --splits 20 21 22 23 24 25 --runs 5 --epochs 200 \
        --out harness/results/baseline_sigma_c.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "harness"))

from experiments.pilot_tau import load, outcome, _split_ref   # noqa: E402
from core.trajectory import (ValidationHistory,               # noqa: E402
                             select_max_auc, select_min_bce)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="german")
    ap.add_argument("--splits", nargs="*", type=int,
                    default=[20, 21, 22, 23, 24, 25])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=27)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "harness", "results", "baseline_sigma_c.csv"))
    a = ap.parse_args()

    dev = a.device
    import core.paths  # noqa: F401  (puts the repository root on sys.path)
    from models.algorithms.GNN import GNN

    rows = []
    for split in a.splits:
        for run in range(a.runs):
            seed = a.seed0 + run
            data = load(a.dataset, split, dev)
            adj, f, y, itr, iva, ite, s, si = data
            ite_np = np.asarray(ite.cpu() if torch.is_tensor(ite) else ite)

            # identical to pilot_tau.py: seeded here, nothing before it
            torch.manual_seed(seed); np.random.seed(seed)
            b = GNN(adj, f, y, itr, iva, ite, s, si, num_hidden=128,
                    num_proj_hidden=128, lr=1e-3, weight_decay=0.0, device=dev)
            hb = ValidationHistory(
                _split_ref(y, iva, s), a.epochs + 1,
                state_fn=lambda: {k: v.detach().clone()
                                  for k, v in b.state_dict().items()})
            b.fit(epochs=a.epochs, trajectory=hb)

            def b_score(state, idx):
                if state is not None:
                    b.load_state_dict(state)
                b.eval()
                with torch.no_grad():
                    emb = b.forward(b.features.to(dev), b.edge_index.to(dev))
                    out = b.forwarding_predict(emb)
                return out.squeeze().detach().cpu().numpy()[np.asarray(idx)]

            pub = outcome(y, s, ite_np, b_score(b.best_state, ite_np))
            for sel in ("common_bce", "common_auc"):
                if sel not in hb.slots:
                    print(f"  s{split} r{run}: baseline slot {sel} missing")
                    continue
                ep, st = hb.slots[sel]
                o = outcome(y, s, ite_np, b_score(st, ite_np))
                rows.append(dict(
                    dataset=a.dataset, backbone="GCN", split_id=split,
                    run_id=run, seed=seed, selector=sel, bc_epoch=ep,
                    bc_auc=o["auc"], bc_dp=o["dp"], bc_eo=o["eo"],
                    bc_eo_defined=int(bool(o["eo_defined"])),
                    bpub_auc=pub["auc"], bpub_dp=pub["dp"], bpub_eo=pub["eo"]))
            print(f"  s{split} r{run} B: pub auc={pub['auc']:.4f} dp={pub['dp']:.4f}"
                  f"  bce ep={rows[-2]['bc_epoch']} auc={rows[-2]['bc_auc']:.4f}"
                  f"  auc-sel ep={rows[-1]['bc_epoch']} auc={rows[-1]['bc_auc']:.4f}",
                  flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
