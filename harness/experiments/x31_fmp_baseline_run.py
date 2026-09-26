"""Stage 0 of the FMP component case study: the common baseline B on FMP's split.

The FMP component study (x27_fmp_run.py) runs on FMP's own split and horizon,
so the common baseline trained on the common split cannot be paired with it (its
test set is a different set of nodes). This runner trains the same baseline B
-- `algorithms/GNN.py` at `published("GNN", dataset)` -- under exactly the FMP
arms' experimental setting, so the component chain gains a first step:

    B  ->  base (F00)  ->  +propagation (F01)  ->  +fairness (F11)

What is shared with the FMP arms, by construction (no new choice is made):

* data and split: `x27_fmp_run.load_split` (FMP's own loader, `seed=split`),
  including its label and sensitive-attribute binarisation;
* seeding: `seed_all(seed * 1000 + split)` with `seed = seed0 + run`, seed0 27;
* horizon: `EPOCHS` = 300 training epochs;
* selectors: `last`, validation BCE (strict <, earliest tie) and validation AUC
  (strict >, earliest tie), exactly as `x27_fmp_run.run_cell` selects;
* evaluator: `x27_fmp_run.metrics` on the score `> 0` decision.

B's own configuration is the published baseline configuration used for every
other cell. Rows follow the x27 schema with `config = "B"`, so B pairs with
F00 / F01 / F11 by (dataset, split, run, selector).

    CUDA_VISIBLE_DEVICES=2 /home/sypark/x27_dgl_cuda/bin/python \
        harness/experiments/x31_fmp_baseline_run.py --dataset pokec_z --out <csv>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))
import core.paths  # noqa: F401,E402  (workspace on sys.path for models.*)


def run_unit(ds, split, run, seed0, device, data, epochs):
    import x27_fmp_run as X
    from core.published_config import published
    from core.trajectory import SplitRef, ValidationHistory
    from models.algorithms.GNN import GNN

    seed = seed0 + run
    X.seed_all(seed * 1000 + split)
    cfg = dict(published("GNN", ds)["config"]); cfg.pop("feature_normalize", None)
    src, dst = data["g"].edges()
    ei = torch.stack([src, dst]).cpu()
    n = data["x"].shape[0]
    adj = torch.sparse_coo_tensor(ei, torch.ones(ei.shape[1]), (n, n)).coalesce()
    y, s = data["y"].cpu(), data["sens"].cpu()
    itr, iva, ite = data["idx_train"].cpu(), data["idx_val"].cpu(), data["idx_test"].cpu()
    b = GNN(adj, data["x"].cpu(), y, itr, iva, ite, s, 0, device=device, **cfg)
    ref = SplitRef(node_id=iva.numpy(), y=y.numpy()[iva.numpy()].astype(int),
                   a=s.numpy()[iva.numpy()].astype(int))
    hist = ValidationHistory(ref, epochs, state_fn=lambda: {k: v.detach().clone()
                                                            for k, v in b.state_dict().items()})
    b.fit(epochs=epochs - 1, trajectory=hist)          # range(epochs) training steps
    if len(hist.records) != epochs:
        raise SystemExit(f"[x31] {len(hist.records)} records for {epochs} epochs (hard stop)")
    last_state = {k: v.detach().clone() for k, v in b.state_dict().items()}

    def test_scores(state):
        b.load_state_dict(state)
        b.eval()
        with torch.no_grad():
            out = b.forwarding_predict(b.forward(b.features.to(device), b.edge_index.to(device)))
        return out.squeeze().detach().cpu().numpy()[ite.numpy()]

    yt, at = y.numpy()[ite.numpy()], s.numpy()[ite.numpy()]
    picks = {"last": (epochs - 1, last_state),
             "bce": hist.slots["common_bce"], "auc": hist.slots["common_auc"]}
    rows, nonfinite = [], 0
    for sel, (ep, st) in picks.items():
        sc = test_scores(st)
        nonfinite += int((~np.isfinite(sc)).sum())
        m = X.metrics(yt, at, sc)
        rows.append(dict(dataset=ds, split_id=split, run_id=run, seed=seed, config="B",
                         lambda1=np.nan, lambda2=np.nan, selector=sel, epoch=int(ep),
                         auc=m["auc"], dp=m["dp"], eo=m["eo"], dp_signed=m["dp_signed"],
                         eo_signed=m["eo_signed"], eo_defined=int(m["eo_defined"]),
                         epochs=epochs, K=np.nan, num_gnn_layer=np.nan,
                         hidden=cfg.get("num_hidden")))
    return rows, dict(nonfinite=nonfinite, test_nodes=ite.numpy(), n_records=len(hist.records))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("pokec_z", "pokec_n"))
    ap.add_argument("--splits", nargs="+", type=int, default=[20, 21, 22, 23, 24, 25])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=27)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="outcome-blind gates only: split identity with the FMP arms, "
                         "finite scores, trajectory length; writes no rows")
    a = ap.parse_args()
    if a.device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise SystemExit("GPU 2 only: set CUDA_VISIBLE_DEVICES=2")
    import pandas as pd
    import x27_fmp_run as X
    done = set()
    if a.out and os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        done = {(int(r.split_id), int(r.run_id)) for r in pd.read_csv(a.out).itertuples()}
    header = bool(a.out and os.path.exists(a.out) and os.path.getsize(a.out) > 0)
    for split in a.splits:
        for run in range(a.runs):
            if (split, run) in done:
                continue
            data = X.load_split(a.dataset, split, a.device)
            rows, g = run_unit(a.dataset, split, run, a.seed0, a.device, data, X.EPOCHS)
            traj = os.path.join(ROOT, "harness", "results", "x27", f"{a.dataset}_trajectories",
                                f"{a.dataset}_s{split}_r{run}_F00.npz")
            z = np.load(traj, allow_pickle=True)
            same = np.array_equal(np.asarray(z["test_node_id"]), g["test_nodes"])
            ok = same and g["nonfinite"] == 0 and g["n_records"] == X.EPOCHS
            print(f"  {a.dataset} s{split} r{run}: test nodes identical to the FMP arms {same}; "
                  f"non-finite {g['nonfinite']}; records {g['n_records']}/{X.EPOCHS}", flush=True)
            if not ok:
                raise SystemExit("[x31] gate failed (hard stop; nothing persisted)")
            if a.smoke:
                continue
            pd.DataFrame(rows).to_csv(a.out, mode="a", header=not header, index=False)
            header = True
    print("SMOKE PASS" if a.smoke else f"[x31] done -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
