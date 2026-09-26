"""X31 FairGNN configuration-variation cells: the upstream GCN rows on pokec.

The primary FairGNN pokec cells run the repository's `utils/param.json`
configuration (frozen, kept primary). FairGNN's own repository publishes a GCN
row per pokec dataset (harness/provenance/fairgnn_upstream/, commit 13cdca7);
X31 runs that row as a configuration-variation (robustness) cell:

    pokec_z  alpha 100, beta 1        pokec_n  alpha 50, beta 1

Everything except the configuration is the primary cells' path, imported from
`pilot_tau` unchanged: loader, B at `published("GNN", ds)` trained first in the
same process, `train("FairGNN", ...)`, off-state alpha = beta = 0, H = 200,
sigma_c^BCE / sigma_c^AUC, common evaluator, CellStore.

Configuration from the row, with `train_fairGNN.py`'s parser defaults for what
the row leaves out: alpha, beta, lr (1e-3), weight_decay (1e-5). Recorded but
not applied, because the shared path fixes them for primary and variation
cells alike: `--acc`/`--roc` (thresholds of the script's own test-reading
selection, replaced by sigma_c), `--epochs` (the native horizon), `--num-hidden`
(the wrapper builds its GNN at 64 in `__init__`), `--sens_number` (the
controlled protocol gives every method the training nodes' attribute).

    CUDA_VISIBLE_DEVICES=2 python harness/experiments/x31_fairgnn_config_run.py \
        --dataset pokec_z --out <csv> [--smoke]

GAT rows (`--model GAT`, DGL env /home/sypark/x27_dgl_cuda): pokec_z alpha 10,
beta 0.01; pokec_n alpha 4, beta 0.01. The wrapper's `get_model` is swapped, for
this process only, for the upstream `GAT_body` (models/GAT.py, copied to
provenance as models_GAT.py) with the row's architecture: num_layers 1, heads
[1, 1], feat_drop = dropout 0.5, attn_drop 0.0, negative_slope 0.2, residual
False, num_hidden 64 -- equal to the width the wrapper builds its classifier and
adversary at, so no other module changes. The graph is the upstream one:
the common edges plus self loops (utils.py: adj + sp.eye), as a DGL graph.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))

import pilot_tau as P                                              # noqa: E402
from core.published_config import published                        # noqa: E402

UP = os.path.join(ROOT, "harness", "provenance", "fairgnn_upstream")
GAT_CALLS = {"forward": 0, "built": 0}
APPLIED = ("alpha", "beta", "lr", "weight_decay")


def _parser_defaults():
    d = {}
    for ln in open(os.path.join(UP, "train_fairGNN.py")):
        m = re.search(r"add_argument\(\s*['\"]--([\w-]+)['\"].*?default\s*=\s*([^,)]+)", ln)
        if m:
            d[m.group(1).replace("-", "_")] = m.group(2).strip().strip("'\"")
    return d


def gat_module(nfeat, args, dropout):
    """The upstream GAT_body on the upstream graph, behind the wrapper's
    `GNN(edge_index, x)` call signature."""
    import importlib.util
    import dgl
    spec = importlib.util.spec_from_file_location("_x31_fairgnn_gat", os.path.join(UP, "models_GAT.py"))
    G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
    heads = [int(args["num_heads"])] * int(args["num_layers"]) + [int(args["num_out_heads"])]

    class UpstreamGAT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            GAT_CALLS["built"] += 1
            self.body = G.GAT_body(int(args["num_layers"]), nfeat, int(args["num_hidden"]), heads,
                                   dropout, float(args["attn_drop"]), float(args["negative_slope"]),
                                   args["residual"])
            self._g, self._key = None, None

        def forward(self, edge_index, x):
            key = (edge_index.data_ptr(), edge_index.shape[1])
            if self._key != key:
                n = x.shape[0]
                loops = torch.arange(n, device=edge_index.device)
                ei = torch.cat([edge_index, torch.stack([loops, loops])], 1)
                ei = torch.unique(ei, dim=1)                  # adj + I, one edge per pair
                self._g, self._key = dgl.graph((ei[0], ei[1]), num_nodes=n).to(x.device), key
            GAT_CALLS["forward"] += 1
            return self.body(self._g, x)
    return UpstreamGAT()


def _row(dataset, model="GCN"):
    txt = open(os.path.join(UP, f"{dataset}_train_fair{model}.sh")).read()
    return {k.replace("-", "_"): v for k, v in re.findall(r"--([\w-]+)=(\S+)", txt)}


def config(dataset, model="GCN"):
    row, defs = _row(dataset, model), _parser_defaults()
    if row.get("model") != model or row.get("dataset") != dataset:
        raise SystemExit(f"[x31] {dataset}: upstream row is not the {model} row for this dataset")
    get = lambda k: float(row.get(k, defs[k]))                     # noqa: E731
    cfg = {k: get(k) for k in APPLIED}
    cfg["feature_normalize"] = published("FairGNN", dataset)["config"].get("feature_normalize", False)
    cfg["num_hidden"] = 128           # value the primary cells pass; the wrapper builds at 64
    recorded = {k: row.get(k, defs.get(k)) for k in ("acc", "roc", "epochs", "num_hidden",
                                                     "sens_number", "seed")}
    if model == "GAT":
        arch = {k: row.get(k, defs.get(k)) for k in ("num_hidden", "num_heads", "num_out_heads",
                                                    "num_layers", "attn_drop", "negative_slope",
                                                    "dropout")}
        arch["residual"] = "residual" in row                     # store_true, default False
        recorded = dict(recorded, gat=arch)
    return cfg, recorded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("pokec_z", "pokec_n"))
    ap.add_argument("--splits", nargs="+", type=int, default=[20, 21, 22, 23, 24, 25])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=27)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="GCN", choices=("GCN", "GAT"))
    a = ap.parse_args()
    if a.device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise SystemExit("GPU 2 only: set CUDA_VISIBLE_DEVICES=2")
    cfg, recorded = config(a.dataset, a.model)
    prim = published("FairGNN", a.dataset)["config"]
    if cfg["num_hidden"] != prim.get("num_hidden") or cfg["feature_normalize"] != prim.get("feature_normalize", False):
        raise SystemExit("[x31] shared-path settings differ from the primary cell (hard stop)")
    tag = f"FairGNN-upstream{a.model}"
    if a.model == "GAT":
        import core.paths  # noqa: F401  (puts the repository root on sys.path)
        import models.algorithms.FairGNN as FG
        arch = recorded["gat"]
        if int(arch["num_hidden"]) != 64:       # output width = num_hidden (mean over out heads)
            raise SystemExit("[x31] upstream GAT width differs from the wrapper's 64 (hard stop)")
        FG.get_model = lambda nfeat, args_: gat_module(nfeat, arch, float(arch["dropout"]))
    off = P.METHODS["FairGNN"]["off"]
    P.DS[0] = a.dataset
    store = None if a.smoke else P.CellStore(a.out)
    for split in a.splits:
        P.SPLIT[0] = split
        for run in range(a.runs):
            seed = a.seed0 + run
            if store and store.is_done("x31", tag, a.dataset, split, run):
                continue
            b_cfg = dict(published("GNN", a.dataset)["config"])
            bflag = bool(b_cfg.pop("feature_normalize", False))
            if bflag != bool(cfg["feature_normalize"]):
                raise SystemExit("[x31] B and FairGNN feature policies differ on pokec (unexpected)")
            data = P.load(a.dataset, split, a.device, feature_normalize=bflag)
            adj, f, y, itr, iva, ite, s, si = data
            ite_np = np.asarray(ite.cpu() if torch.is_tensor(ite) else ite)

            from models.algorithms.GNN import GNN
            torch.manual_seed(seed); np.random.seed(seed)
            b = GNN(adj, f, y, itr, iva, ite, s, si, device=a.device, **b_cfg)
            hb = P.ValidationHistory(P._split_ref(y, iva, s), a.epochs + 1,
                                     state_fn=lambda: {k: v.detach().clone()
                                                       for k, v in b.state_dict().items()})
            b.fit(epochs=a.epochs, trajectory=hb)

            def b_score(state, idx):
                if state is not None:
                    b.load_state_dict(state)
                b.eval()
                with torch.no_grad():
                    out = b.forwarding_predict(b.forward(b.features.to(a.device),
                                                         b.edge_index.to(a.device)))
                return out.squeeze().detach().cpu().numpy()[np.asarray(idx)]
            base_pub = P.outcome(y, s, ite_np, b_score(b.best_state, ite_np))
            base_c = {sel: (hb.slots[sel][0], P.outcome(y, s, ite_np, b_score(hb.slots[sel][1], ite_np)))
                      for sel in P.STORE_SELECTORS if sel in hb.slots}

            GAT_CALLS.update(forward=0, built=0)           # per-unit gate
            torch.manual_seed(seed * 1000 + split)
            f1_, h1, e1 = P.train("FairGNN", data, seed, a.epochs, a.device, off=None, cfg=cfg)
            torch.manual_seed(seed * 1000 + split)
            f0_, h0, e0 = P.train("FairGNN", data, seed, a.epochs, a.device, off=off, cfg=cfg)
            m1_pub = P.outcome(y, s, ite_np, f1_(None, ite_np))
            ok = all(sel in h.slots for h in (h1, h0) for sel in P.STORE_SELECTORS) \
                and set(base_c) == set(P.STORE_SELECTORS)
            if a.model == "GAT":
                gat_ok = GAT_CALLS["built"] >= 2 and GAT_CALLS["forward"] >= 2 * a.epochs
                print(f"  upstream GAT built {GAT_CALLS['built']}x, forward {GAT_CALLS['forward']}x "
                      f"(needs both arms, >= {2 * a.epochs}) {gat_ok}", flush=True)
                ok = ok and gat_ok
            sc = {sel: (np.asarray(f1_(h1.slots[sel][1], ite_np)),
                        np.asarray(f0_(h0.slots[sel][1], ite_np)))
                  for sel in P.STORE_SELECTORS} if ok else {}
            fin = ok and all(np.isfinite(v).all() for pr in sc.values() for v in pr)
            print(f"  {tag}/{a.dataset} s{split} r{run}: B and both arms have both selector "
                  f"slots {ok}; finite test scores {fin}; horizon {a.epochs}", flush=True)
            if not (ok and fin):
                raise SystemExit("[x31] gate failed (hard stop; nothing persisted)")
            if a.smoke:
                continue
            _a = np.asarray(s.detach().cpu() if torch.is_tensor(s) else s)[ite_np].astype(int)
            n_a1, n_a0 = int((_a == 1).sum()), int((_a == 0).sum())
            cell = []
            for sel in P.STORE_SELECTORS:
                sc1, sc0 = sc[sel]
                m1 = P.outcome(y, s, ite_np, sc1)
                m0 = P.outcome(y, s, ite_np, sc0)
                flip = (sc1 > 0).astype(int) != (sc0 > 0).astype(int)
                cell.append(dict(
                    method=tag, dataset=a.dataset, backbone=a.model, split_id=split,
                    run_id=run, seed=seed, selector=sel, provenance="official-repo",
                    n_flip=int(flip.sum()), n_flip_a1=int((flip & (_a == 1)).sum()),
                    n_flip_a0=int((flip & (_a == 0)).sum()), n_test_a1=n_a1, n_test_a0=n_a0,
                    dp_min_step=min(1.0 / max(n_a1, 1), 1.0 / max(n_a0, 1)),
                    protocol="x31", method_epochs=a.epochs, b_epochs=a.epochs,
                    eval_rng_seed=-1, rng_contract="deterministic-inference",
                    feature_normalize=int(bool(cfg["feature_normalize"])),
                    m1_epoch=h1.slots[sel][0], m0_epoch=h0.slots[sel][0], code_epoch=e1,
                    m1_auc=m1["auc"], m1_dp=m1["dp"], m1_eo=m1["eo"],
                    m0_auc=m0["auc"], m0_dp=m0["dp"], m0_eo=m0["eo"],
                    b_auc=base_pub["auc"], b_dp=base_pub["dp"], b_eo=base_pub["eo"],
                    bc_epoch=base_c[sel][0], bc_auc=base_c[sel][1]["auc"],
                    bc_dp=base_c[sel][1]["dp"], bc_eo=base_c[sel][1]["eo"],
                    m1pub_auc=m1_pub["auc"], m1pub_dp=m1_pub["dp"], m1pub_eo=m1_pub["eo"],
                    int_dauc=m1["auc"] - m0["auc"], int_ndp=-(m1["dp"] - m0["dp"]),
                    int_neo=-(m1["eo"] - m0["eo"]),
                    pkg_dauc=m1_pub["auc"] - base_pub["auc"],
                    pkg_ndp=-(m1_pub["dp"] - base_pub["dp"]),
                    pkg_neo=-(m1_pub["eo"] - base_pub["eo"]),
                    eo_defined=bool(m1["eo_defined"] and m0["eo_defined"]),
                    config=repr(dict(cfg, recorded_not_applied=recorded))))
            store.append_cell(cell)
            print(f"  {tag}/{a.dataset} s{split} r{run}: persisted", flush=True)
    print("SMOKE PASS" if a.smoke else f"[x31] done -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
