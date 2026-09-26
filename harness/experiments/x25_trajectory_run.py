"""X25 trajectory rerun: the unchanged pilot plus per-epoch NIFTY test-score recording.

Pre-registered in X25 (2547bf3). `pilot_tau.main()` runs exactly as in X24, with
the same arguments, B, CellStore persistence and CSV rows. Three
process-local patches add recording and change nothing the method computes:

* `NIFTY.fit` registers the model and its ValidationHistory when recording is on.
* `ValidationHistory.add` calls the original `add`, then, for a registered
  history, scores the test split on the clean graph in eval mode under
  `no_grad`. This is the pipeline's own `score_fn` test path. NIFTY has
  already called `self.eval()`: spectral_norm does no power iteration,
  BatchNorm does not update, and no RNG is drawn (X25 §2.5; verified by G1).
  The scores go into a module-level store, never onto the history object, so
  no selector can reach them.
* `pilot_tau.train` wraps NIFTY training and writes one `.npz` per arm
  (atomic rename). Other methods and B pass through untouched.

Resume: a cell already in the CSV counts only if both arm files exist and are
complete; otherwise the wrapper refuses to start.

    CUDA_VISIBLE_DEVICES=2 python harness/experiments/x25_trajectory_run.py \
        --traj_dir /tmp/.../x25_R10 -- --protocol armA --epochs 200 --method_epochs 1000 \
        --dataset german --methods NIFTY --splits 20 21 22 23 24 25 --runs 5 --out /tmp/.../x25_R10.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "harness"))

import core.paths  # noqa: F401  (puts the repository root on sys.path)
import models.algorithms.NIFTY as NM                                         # noqa: E402
import core.trajectory as TR                                          # noqa: E402
import experiments.pilot_tau as P                                     # noqa: E402

_ACTIVE: dict[int, dict] = {}          # id(ValidationHistory) -> recording slot
_CTX = {"record": False, "test_idx": None, "traj_dir": None}
_orig_add = TR.ValidationHistory.add
_orig_fit = NM.NIFTY.fit
_orig_train = P.train


def _np(x):
    return np.asarray(x.detach().cpu() if torch.is_tensor(x) else x)


def _add(self, epoch, raw_score, **kw):
    r = _orig_add(self, epoch, raw_score, **kw)
    slot = _ACTIVE.get(id(self))
    if slot is not None and kw.get("stage", "classifier") == "classifier":
        m = slot["model"]
        if m.training:
            raise RuntimeError("[x25] NIFTY is not in eval mode at the recording hook")
        with torch.no_grad():
            emb = m.forward(m.features, m.edge_index)
            out = m.forwarding_predict(emb)
        slot["epochs"].append(int(epoch))
        slot["scores"].append(out.squeeze().detach().cpu().numpy()[slot["test_idx"]]
                              .astype(np.float64))
    return r


def _fit(self, *args, **kw):
    traj = kw.get("trajectory")
    if _CTX["record"] and traj is not None:
        _ACTIVE[id(traj)] = dict(model=self, test_idx=_CTX["test_idx"], epochs=[], scores=[])
    return _orig_fit(self, *args, **kw)


def record_train(method, data, seed, epochs, device, off=None, cfg=None):
    """The pilot's original train() with recording on; returns (fn, hist, e, traj)."""
    ite_np = _np(data[5]).astype(int)
    _CTX.update(record=True, test_idx=ite_np)
    try:
        fn, hist, e = _orig_train(method, data, seed, epochs, device, off=off, cfg=cfg)
    finally:
        _CTX.update(record=False, test_idx=None)
    slot = _ACTIVE.pop(id(hist), None)
    if slot is None:
        raise RuntimeError("[x25] no recording was registered for this NIFTY fit")
    recs = hist.classifier_records()
    y, s = _np(data[2]), _np(data[6])
    traj = dict(
        epochs=np.array([r.epoch for r in recs], dtype=int),
        val_raw=np.stack([r.raw_score for r in recs]),
        val_bce=np.array([r.predictive_loss for r in recs], dtype=np.float64),
        val_auc=np.array([r.metrics["auc"] for r in recs], dtype=np.float64),
        val_node_id=np.asarray(hist.ref.node_id, dtype=int),
        val_y=np.asarray(hist.ref.y, dtype=int), val_a=np.asarray(hist.ref.a, dtype=int),
        test_epochs=np.array(slot["epochs"], dtype=int),
        test_raw=np.stack(slot["scores"]) if slot["scores"] else np.empty((0, len(ite_np))),
        test_node_id=ite_np, test_y=y[ite_np].astype(int), test_a=s[ite_np].astype(int),
        slot_bce=int(hist.slots["common_bce"][0]), slot_auc=int(hist.slots["common_auc"][0]),
        code_epoch=int(e), seed=int(seed), split=int(P.SPLIT[0]), horizon=int(epochs),
        arm="M1" if off is None else "M0",
        cfg_json=json.dumps(cfg or {}, sort_keys=True, default=str),
        off_json=json.dumps(off or {}, sort_keys=True, default=str))
    return fn, hist, e, traj


def traj_path(traj_dir, dataset, split, seed, arm):
    return os.path.join(traj_dir, f"{dataset}_s{int(split)}_seed{int(seed)}_{arm}.npz")


def save_traj(traj_dir, dataset, traj):
    p = traj_path(traj_dir, dataset, traj["split"], traj["seed"], traj["arm"])
    tmp = p[:-4] + ".tmp.npz"
    np.savez(tmp, **{k: np.asarray(v) for k, v in traj.items()})
    os.replace(tmp, p)
    return p


def valid_traj(p, horizon=None) -> bool:
    try:
        z = np.load(p, allow_pickle=False)
        ep = z["epochs"]
        H = int(z["horizon"]) if horizon is None else int(horizon)
        return (np.array_equal(ep, np.arange(H + 1)) and np.array_equal(z["test_epochs"], ep)
                and z["test_raw"].shape[0] == H + 1 and z["val_raw"].shape[0] == H + 1
                and bool(np.isfinite(z["test_raw"]).all()) and bool(np.isfinite(z["val_raw"]).all()))
    except Exception:                                                  # noqa: BLE001
        return False


def _train(method, data, seed, epochs, device, off=None, cfg=None):
    if method != "NIFTY":
        return _orig_train(method, data, seed, epochs, device, off=off, cfg=cfg)
    fn, hist, e, traj = record_train(method, data, seed, epochs, device, off=off, cfg=cfg)
    save_traj(_CTX["traj_dir"], P.DS[0], traj)
    return fn, hist, e


def install(traj_dir=None, wrap_train=True):
    TR.ValidationHistory.add = _add
    NM.NIFTY.fit = _fit
    if wrap_train:
        _CTX["traj_dir"] = traj_dir
        P.train = _train


def uninstall():
    TR.ValidationHistory.add = _orig_add
    NM.NIFTY.fit = _orig_fit
    P.train = _orig_train


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def check_resume(out, traj_dir, dataset, seed0, horizon):
    if not (os.path.exists(out) and os.path.getsize(out) > 0):
        return 0
    store = P.CellStore(out)
    for (_proto, _meth, _ds, split, run) in sorted(store.done):
        for arm in ("M1", "M0"):
            p = traj_path(traj_dir, dataset, split, seed0 + int(run), arm)
            if not valid_traj(p, horizon):
                raise SystemExit(f"[x25] persisted cell (split {split}, run {run}) lacks a valid "
                                 f"trajectory {p}; refusing to resume")
    return len(store.done)


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: x25_trajectory_run.py --traj_dir DIR -- <pilot_tau args>")
    i = argv.index("--")
    own, pilot = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", required=True)
    a = ap.parse_args(own)
    methods = pilot[pilot.index("--methods") + 1:] if "--methods" in pilot else []
    methods = [m for m in methods[:1]]
    if methods != ["NIFTY"] or (pilot.index("--methods") + 2 < len(pilot)
                                and not pilot[pilot.index("--methods") + 2].startswith("--")):
        raise SystemExit("[x25] --methods must be exactly NIFTY")
    out = _arg(pilot, "--out")
    if out is None:
        raise SystemExit("[x25] --out is required")
    os.makedirs(a.traj_dir, exist_ok=True)
    dataset = _arg(pilot, "--dataset", "german")
    seed0 = int(_arg(pilot, "--seed0", 27))
    H = _arg(pilot, "--method_epochs")
    n = check_resume(out, a.traj_dir, dataset, seed0, int(H) if H is not None else None)
    if n:
        print(f"[x25] resume: {n} persisted cell(s) have valid trajectories")
    install(a.traj_dir)
    sys.argv = ["pilot_tau.py"] + pilot
    return P.main()


if __name__ == "__main__":
    raise SystemExit(main())
