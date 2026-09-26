"""X31 section 3b: FnRGNN on its own task (node regression, MSE), B = the common
GCN trained with MSE. Separate analysis; regression metrics only.

Per matched unit (common split, seed = 27 + run):

    B      algorithms/GNN.py at published("GNN", ds), built and seeded as the
           common baseline, trained by GNN.fit's steps with BCE -> MSE
    M^-I   FnRGNN, use_mmd = use_gwn = use_edge_weight = False   } released class,
    M^+I   FnRGNN, all three True                                } MSE kept

Features: the common features with the target column removed (asserted equal
to the raw target). Target standardised with the training nodes' statistics.
Selection: smallest validation MSE (strict <, earliest tie) for every arm.
Test metrics: MSE, mean gap, Wasserstein distance between the groups.

    CUDA_VISIBLE_DEVICES=2 /home/sypark/x30_edits_env/bin/python \
        harness/experiments/x31_fnrgnn_regression_run.py --dataset german --out <csv> [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "harness", "adapters"))
import core.paths  # noqa: F401,E402

TARGET = {"german": ("data/german/german.csv", "LoanAmount"),
          "pokec_z": ("data/pokec/region_job.csv", "completion_percentage"),
          "pokec_n": ("data/pokec/region_job_2.csv", "completion_percentage")}


def seed_all(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load(dataset, split):
    """Common graph, split and sensitive attribute; regression target from the
    same CSV in node order; the target column removed from the features."""
    from utils.dataloading import load_data
    adj, feats, _, itr, iva, ite, sens, _ = load_data(dataset, feature_normalize=False,
                                                      split_seed=split)
    path, col = TARGET[dataset]
    y = pd.read_csv(os.path.join(ROOT, path))[col].to_numpy(dtype=np.float64)
    if len(y) != feats.shape[0]:
        raise SystemExit(f"[x31] {dataset}: {len(y)} target rows for {feats.shape[0]} nodes")
    X = feats.cpu().numpy().astype(np.float64)
    hit = [j for j in range(X.shape[1]) if np.array_equal(X[:, j], y)]
    if len(hit) != 1:
        raise SystemExit(f"[x31] {dataset}: target column found {len(hit)} times in the features")
    keep = [j for j in range(X.shape[1]) if j != hit[0]]
    feats = feats[:, keep].contiguous()
    tr = itr.cpu().numpy()
    mu, sd = y[tr].mean(), y[tr].std()
    ystd = torch.tensor((y - mu) / sd, dtype=torch.float32)
    return dict(adj=adj, feats=feats, y=ystd, itr=itr, iva=iva, ite=ite,
                sens=(sens.cpu() > 0).long(), target_col=int(hit[0]), n_feat=len(keep))


def train_B(D, seed, epochs, device):
    from core.published_config import published
    from models.algorithms.GNN import GNN
    cfg = dict(published("GNN", D["dataset"])["config"]); cfg.pop("feature_normalize", None)
    torch.manual_seed(seed); np.random.seed(seed)
    b = GNN(D["adj"], D["feats"], D["y"], D["itr"], D["iva"], D["ite"], D["sens"], 0,
            device=device, **cfg)
    y = b.labels.float().view(-1, 1)
    preds, rows = [], 0
    for _ in range(epochs + 1):                       # GNN.fit: range(epochs + 1)
        b.train()
        b.optimizer_2.zero_grad()
        c1 = b.classifier(b.forward(b.features, b.edge_index))
        loss = F.mse_loss(c1[b.idx_train], y[b.idx_train])   # BCE -> MSE
        rows = int(c1[b.idx_train].shape[0])
        loss.backward()
        b.optimizer_2.step()
        b.eval()
        with torch.no_grad():
            c = b.classifier(b.forward(b.features, b.edge_index))
        preds.append(c.squeeze().detach().float().cpu().numpy())
    return preds, dict(loss_rows=rows, cfg=cfg)


def train_arm(D, on, epochs, device):
    import x31_fnrgnn as FA
    from sklearn.preprocessing import StandardScaler
    from torch_geometric.data import Data
    M = FA._load()
    cfg = FA.config(D["dataset"])
    x = torch.tensor(StandardScaler().fit_transform(D["feats"].numpy()), dtype=torch.float32)
    data = Data(x=x, edge_index=D["adj"].coalesce().indices(), y=D["y"],
                sensitive_attr=D["sens"]).to(device)
    idx_train = D["itr"].to(device)
    model = M.FnRGNN(nfeat=x.shape[1], hidden_dim=int(cfg["hidden_dim"]), dropout=cfg["dropout"],
                     lm=cfg["lambda2"], gm=cfg["gamma"], ld=cfg["lambda_dist"],
                     mmd_sample=FA.MMD_SAMPLE, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                     use_mmd=on, use_gwn=on, use_edge_weight=on).to(device)
    h = hashlib.sha1()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    st = dict(steps=0, loss_rows=0, mmd_nonzero=0, gwn_nonzero=0, init=h.hexdigest()[:16])
    model.eval()
    with torch.no_grad():
        ew = (model.compute_edge(data.x, data.edge_index, data.sensitive_attr)
              if model.use_edge_weight else torch.ones(data.edge_index.size(1), device=device))
    st["edge_all_one"] = bool(torch.all(ew == 1).item())
    crit = model.criterion                              # the released nn.MSELoss
    if not isinstance(crit, nn.MSELoss):
        raise SystemExit("[x31] FnRGNN criterion is not the released MSELoss")
    preds = []
    for _ in range(epochs):
        FA.train_step(model, data, idx_train, crit, st)  # section-3 step, criterion MSE
        model.eval()
        with torch.no_grad():
            yhat, _ = model(data)
        preds.append(yhat.squeeze().detach().float().cpu().numpy())
    return preds, st


def select(preds, y, idx):
    best, ep = np.inf, -1
    for e, p in enumerate(preds):
        v = float(np.mean((p[idx] - y[idx]) ** 2))
        if v < best:
            best, ep = v, e
    return ep


def metrics(p, y, s):
    from scipy.stats import wasserstein_distance
    p0, p1 = p[s == 0], p[s == 1]
    return dict(mse=float(np.mean((p - y) ** 2)), mean_gap=float(abs(p0.mean() - p1.mean())),
                wd=float(wasserstein_distance(p0, p1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=tuple(TARGET))
    ap.add_argument("--splits", nargs="+", type=int, default=[20, 21, 22, 23, 24, 25])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=27)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise SystemExit("GPU 2 only: set CUDA_VISIBLE_DEVICES=2")
    done = set()
    if a.out and os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        done = {(int(r.split_id), int(r.run_id)) for r in pd.read_csv(a.out).itertuples()}
    header = bool(done)
    for split in a.splits:
        for run in range(a.runs):
            if (split, run) in done:
                continue
            seed = a.seed0 + run
            D = load(a.dataset, split); D["dataset"] = a.dataset
            y, s = D["y"].numpy(), D["sens"].numpy()
            iv, it_, n_tr = D["iva"].numpy(), D["ite"].numpy(), int(D["itr"].numel())
            pb, gb = train_B(D, seed, a.epochs, a.device)
            seed_all(seed * 1000 + split)
            p1, g1 = train_arm(D, True, a.epochs, a.device)
            seed_all(seed * 1000 + split)
            p0, g0 = train_arm(D, False, a.epochs, a.device)
            E = a.epochs
            checks = {
                "target column removed from the features": (True, f"(column {D['target_col']})"),
                "finite predictions, every arm": (all(np.isfinite(np.stack(p)).all()
                                                      for p in (pb, p1, p0)), ""),
                "trajectory lengths (B 201, arms 200)": (len(pb) == E + 1 and len(p1) == E
                                                         and len(p0) == E, ""),
                "loss on exactly the training nodes, every arm": (
                    gb["loss_rows"] == n_tr and g1["loss_rows"] == n_tr and g0["loss_rows"] == n_tr,
                    f"({gb['loss_rows']}/{g1['loss_rows']}/{g0['loss_rows']} of {n_tr})"),
                "M+I MMD and GWN active every step, edge reweighting on": (
                    g1["mmd_nonzero"] == E and g1["gwn_nonzero"] == E and not g1["edge_all_one"], ""),
                "M-I MMD and GWN zero, unit edge weights": (
                    g0["mmd_nonzero"] == 0 and g0["gwn_nonzero"] == 0 and g0["edge_all_one"], ""),
                "paired arms share initial parameters": (g1["init"] == g0["init"], ""),
            }
            ok = all(v[0] for v in checks.values())
            print(f"  FnRGNN-regression/{a.dataset} s{split} r{run}: gates {'PASS' if ok else 'FAIL'}",
                  flush=True)
            for k, v in checks.items():
                print(f"    [{'PASS' if v[0] else 'FAIL'}] {k} {v[1]}")
            if not ok:
                raise SystemExit("[x31] gate failed (hard stop; nothing persisted)")
            if a.smoke:
                continue
            eb, e1, e0 = select(pb, y, iv), select(p1, y, iv), select(p0, y, iv)
            row = dict(method="FnRGNN-regression", dataset=a.dataset, split_id=split, run_id=run,
                       seed=seed, selector="val_mse", b_epoch=eb, m1_epoch=e1, m0_epoch=e0,
                       method_epochs=E, b_epochs=E, n_test=len(it_))
            for tag, p, e in (("b", pb, eb), ("m1", p1, e1), ("m0", p0, e0)):
                for k, v in metrics(p[e][it_], y[it_], s[it_]).items():
                    row[f"{tag}_{k}"] = v
            pd.DataFrame([row]).to_csv(a.out, mode="a", header=not header, index=False)
            header = True
    print("SMOKE PASS" if a.smoke else f"[x31] done -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
