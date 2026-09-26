"""X27 FMP mechanistic runner: official FMP modules, run isolation, paired cells.

Pre-registered in X27 (00b6b26). Runs in the isolated DGL environment
(/home/sypark/x27_dgl_cuda: torch 2.2.2+cu121, dgl 1.1.3+cu121). The official
sources in `FMP-main/` are imported unmodified; everything below is harness
code around them.

What this runner adds, and nothing else:

* **Run isolation (X27 section 8).** `get_sen` aliases the sensitive tensor and
  mutates it in place, so a second model in the same process would debias
  against a corrupted group vector. Every model is handed a *fresh clone*; an
  immutable original is kept, and all fairness metrics are computed from the
  original. This is a state-mutation correction, not an intervention change.
* **RNG contract (X27 section 7).** Python, NumPy, torch CPU and torch CUDA are
  seeded explicitly per (dataset, split, run). The official path never seeds the
  CPU generator that initialises `nn.Linear`.
* **Paired cells.** Within one (split, run) every configuration (F00, F01, F11)
  starts from identical MLP initialisation; different runs differ.
* **Trajectory.** Per epoch: validation BCE and AUC, unified DP/EO/AUC, plus the
  test raw scores needed to replay sigma_last / sigma_BCE / sigma_AUC.

The seven preregistered configurations per dataset:
F00; F01(0.01); F01(20); F11(5,0.01); F11(30,0.01); F11(5,20); F11(30,20).

    CUDA_VISIBLE_DEVICES=2 /home/sypark/x27_dgl_cuda/bin/python \
        harness/experiments/x27_fmp_run.py --dataset pokec_z --splits 20 --runs 1 \
        --out /tmp/.../x27_pilot.csv --traj_dir /tmp/.../x27_pilot_traj
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "harness"))
from core.paths import repo as _repo  # noqa: E402  (repositories live under models/)

import argparse
import copy
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, _repo("FMP-main"))

# Preregistered grid (X27 section 3): deterministic endpoints of the official
# executable candidate set. Never chosen from results.
LAM1 = (5.0, 30.0)
LAM2 = (0.01, 20.0)
CONFIGS = ([("F00", 0.0, 0.0)]
           + [(f"F01_l2{l2:g}", 0.0, l2) for l2 in LAM2]
           + [(f"F11_l1{l1:g}_l2{l2:g}", l1, l2) for l2 in LAM2 for l1 in LAM1])
EPOCHS = 300               # active sweep line
HIDDEN = 64
NUM_GNN_LAYER = 2          # consistent with paper and script (X27 section 2)
K_PROP = 5                 # --num-layers default
LR = 1e-3
DATA = {"pokec_z": ("region_job", "region"), "pokec_n": ("region_job_2", "region")}
PREDICT_ATTR = "I_am_working_in_field"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(dataset: str, split_seed: int, device):
    """FMP's own loader at its own parameters; only `seed` varies (X27 section 7)."""
    from utils import load_pokec
    name, sens_attr = DATA[dataset]
    adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(
        name, sens_attr, PREDICT_ATTR, path=os.path.join(ROOT, "data", "pokec") + os.sep,
        seed=split_seed, test_idx=False)
    import dgl
    g = dgl.from_scipy(adj)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    labels = labels.clone()
    labels[labels > 1] = 1
    sens = sens.clone()
    sens[sens > 0] = 1
    return dict(g=g, x=features.to(device), y=labels.to(device),
                idx_train=idx_train.to(device), idx_val=idx_val.to(device),
                idx_test=idx_test.to(device), sens=sens.to(device),
                idx_sens_train=idx_sens_train.to(device))


def build_model(args_like, num_features, device):
    from fairgnn import FairGNN
    from fmp import FMP
    prop = FMP(in_feats=num_features, out_feats=num_features, K=K_PROP,
               lambda1=args_like["lambda1"], lambda2=args_like["lambda2"],
               L2=True, cached=True)
    model = FairGNN(input_size=num_features, size=HIDDEN, num_classes=2,
                    num_layer=NUM_GNN_LAYER, prop=prop).to(device)
    return model


def metrics(y_true, sens, scores):
    """Unified outcome: AUC from raw continuous margin, signed and absolute DP/EO."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y_true).astype(int)
    a = np.asarray(sens).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    pred = (s > 0).astype(int)
    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    m1, m0 = a == 1, a == 0
    dp = (float(pred[m1].mean()) - float(pred[m0].mean())) if m1.any() and m0.any() else float("nan")
    e1, e0 = m1 & (y == 1), m0 & (y == 1)
    eo_defined = bool(e1.any() and e0.any())
    eo = (float(pred[e1].mean()) - float(pred[e0].mean())) if eo_defined else float("nan")
    bce = float(np.mean(np.logaddexp(0.0, -np.abs(s)) + np.maximum(s, 0.0) - s * y))
    return dict(auc=auc, dp_signed=dp, dp=abs(dp) if dp == dp else float("nan"),
                eo_signed=eo, eo=abs(eo) if eo == eo else float("nan"),
                eo_defined=eo_defined, bce=bce)


def run_cell(dataset, split, run, seed0, device, data, init_state, cfg_name, l1, l2):
    """One configuration of one (split, run). Returns a row and its trajectory."""
    seed = seed0 + run
    seed_all(seed * 1000 + split)
    model = build_model(dict(lambda1=l1, lambda2=l2), data["x"].shape[1], device)
    if init_state is None:
        init_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                      if not k.startswith("prop.")}
    else:
        missing = model.load_state_dict(init_state, strict=False)
        del missing
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = torch.nn.CrossEntropyLoss()
    yv = data["y"][data["idx_val"]].detach().cpu().numpy()
    av = data["sens"][data["idx_val"]].detach().cpu().numpy()
    yt = data["y"][data["idx_test"]].detach().cpu().numpy()
    at = data["sens"][data["idx_test"]].detach().cpu().numpy()
    traj = dict(epoch=[], val_bce=[], val_auc=[], val_dp=[], val_eo=[],
                test_auc=[], test_dp=[], test_eo=[], test_dp_signed=[], test_eo_signed=[],
                nonfinite=[])
    test_scores = []
    for epoch in range(EPOCHS):
        model.train()
        # run isolation: the official get_sen mutates its argument in place
        sens_train = data["sens"].clone()
        logits = model(data["x"], data["g"], sens_train, data["idx_sens_train"])
        loss = crit(logits[data["idx_train"]], data["y"][data["idx_train"]].long())
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data["x"], data["g"], data["sens"].clone(), data["idx_sens_train"])
        margin = (out[:, 1] - out[:, 0]).detach().cpu().numpy()
        mv = metrics(yv, av, margin[data["idx_val"].detach().cpu().numpy()])
        mt = metrics(yt, at, margin[data["idx_test"].detach().cpu().numpy()])
        traj["epoch"].append(epoch)
        traj["val_bce"].append(mv["bce"]); traj["val_auc"].append(mv["auc"])
        traj["val_dp"].append(mv["dp"]); traj["val_eo"].append(mv["eo"])
        traj["test_auc"].append(mt["auc"]); traj["test_dp"].append(mt["dp"])
        traj["test_eo"].append(mt["eo"]); traj["test_dp_signed"].append(mt["dp_signed"])
        traj["test_eo_signed"].append(mt["eo_signed"])
        traj["nonfinite"].append(int(not np.isfinite(margin).all()))
        test_scores.append(margin[data["idx_test"].detach().cpu().numpy()].astype(np.float32))
    T = {k: np.asarray(v) for k, v in traj.items()}
    sel = dict(last=EPOCHS - 1,
               bce=int(np.argmin(T["val_bce"])),
               auc=int(np.argmax(np.where(np.isfinite(T["val_auc"]), T["val_auc"], -np.inf))))
    rows = []
    for sname, ep in sel.items():
        m = metrics(yt, at, test_scores[ep])
        rows.append(dict(dataset=dataset, split_id=split, run_id=run, seed=seed,
                         config=cfg_name, lambda1=l1, lambda2=l2, selector=sname,
                         epoch=ep, auc=m["auc"], dp=m["dp"], eo=m["eo"],
                         dp_signed=m["dp_signed"], eo_signed=m["eo_signed"],
                         eo_defined=int(m["eo_defined"]), epochs=EPOCHS, K=K_PROP,
                         num_gnn_layer=NUM_GNN_LAYER, hidden=HIDDEN))
    T["test_scores_sel"] = np.stack([test_scores[sel[s]] for s in ("last", "bce", "auc")])
    T["sel_epochs"] = np.array([sel["last"], sel["bce"], sel["auc"]])
    T["test_y"] = yt.astype(int); T["test_a"] = at.astype(int)
    T["val_node_id"] = data["idx_val"].detach().cpu().numpy().astype(int)
    T["test_node_id"] = data["idx_test"].detach().cpu().numpy().astype(int)
    return rows, T, init_state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=tuple(DATA))
    ap.add_argument("--splits", nargs="+", type=int, default=[20, 21, 22, 23, 24, 25])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=27)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.traj_dir, exist_ok=True)
    import pandas as pd
    done = set()
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        prev = pd.read_csv(a.out)
        done = {(int(r.split_id), int(r.run_id), r.config) for r in prev.itertuples()}
        print(f"[x27] resume: {len(done)} configuration cell(s) already persisted")
    header_written = os.path.exists(a.out) and os.path.getsize(a.out) > 0
    for split in a.splits:
        for run in range(a.runs):
            data = load_split(a.dataset, split, a.device)
            sens_hash = hash(data["sens"].detach().cpu().numpy().tobytes())
            init_state = None
            for cfg_name, l1, l2 in CONFIGS:
                if (split, run, cfg_name) in done:
                    continue
                rows, T, init_state = run_cell(a.dataset, split, run, a.seed0, a.device,
                                               data, init_state, cfg_name, l1, l2)
                if hash(data["sens"].detach().cpu().numpy().tobytes()) != sens_hash:
                    raise SystemExit("[x27] sensitive vector mutated during a run (hard stop)")
                p = os.path.join(a.traj_dir,
                                 f"{a.dataset}_s{split}_r{run}_{cfg_name}.npz")
                tmp = p[:-4] + ".tmp.npz"
                np.savez_compressed(tmp, **T, dataset=a.dataset, split=split, run=run,
                                    config=cfg_name, lambda1=l1, lambda2=l2)
                os.replace(tmp, p)
                df = pd.DataFrame(rows)
                df.to_csv(a.out, mode="a", header=not header_written, index=False)
                header_written = True
                r0 = rows[0]
                print(f"  {a.dataset} s{split} r{run} {cfg_name}: last-epoch "
                      f"AUC {r0['auc']:.4f} DP {r0['dp']:.4f} EO {r0['eo']:.4f}", flush=True)
    print(f"[x27] done -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
