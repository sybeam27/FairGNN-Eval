"""X26 FairGB trajectory rerun: the unchanged pilot plus minimal-storage recording.

Pre-registered in X26 (630c82a). `pilot_tau.main()` runs exactly as in X23/X24,
with the same arguments, B, CellStore persistence and CSV rows. Three
process-local patches add recording and change nothing the method computes:

* `FairGB.fit` registers the model and its ValidationHistory when recording is
  on, so the harness can find the inference modules afterwards.
* `ValidationHistory.add` calls the original `add`, then, for a registered
  history: at epoch 199 it freezes a copy of the running-best slots (the **cap**
  checkpoints, support {0..199}), and at each pre-registered grid epoch it
  scores the test split through the method's own evaluation path. FairGB's
  inference is deterministic, and the model is in the state the epoch left it
  in; G1 checks empirically that this changes no training behaviour.
* `pilot_tau.train` wraps FairGB training and writes one `.npz` per arm
  (atomic rename) holding per-epoch validation BCE/AUC, the grid-epoch test
  scores, and the test outcomes of the four cap/full x BCE/AUC slots evaluated
  through the pipeline's own `score_fn`. Per-epoch test arrays for all epochs
  are deliberately not stored (X26 section 6).

Test scores live in a module-level store, never on the history object, so no
selector can reach them.

    CUDA_VISIBLE_DEVICES=2 python harness/experiments/x26_fairgb_run.py \
        --traj_dir /tmp/.../x26_bail -- --protocol native --epochs 200 \
        --dataset bail --methods FairGB --splits 20 21 22 23 24 25 --runs 5 --out /tmp/.../x26_bail.csv
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "harness"))

import core.paths  # noqa: F401  (puts the repository root on sys.path)
import models.algorithms.FairGB_alg as GB                                    # noqa: E402
import core.trajectory as TR                                          # noqa: E402
import experiments.pilot_tau as P                                     # noqa: E402

# Fixed epoch grids (X26 section 5), 0-indexed, last point H-1.
GRID = {"bail": (25, 50, 100, 150, 200, 300, 500, 750, 1000, 1250, 1499),
        "credit": (25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 1999)}
CAP_LAST = 199          # cap support is {0..199}
SIGMAS = ("bce", "auc")
SLOT = {"bce": "common_bce", "auc": "common_auc"}

_ACTIVE: dict[int, dict] = {}
_CTX = {"record": False, "grid": (), "traj_dir": None}
_orig_add = TR.ValidationHistory.add
_orig_fit = GB.FairGB.fit
_orig_train = P.train


def _np(x):
    return np.asarray(x.detach().cpu() if torch.is_tensor(x) else x)


def _add(self, epoch, raw_score, **kw):
    r = _orig_add(self, epoch, raw_score, **kw)
    slot = _ACTIVE.get(id(self))
    if slot is None or kw.get("stage", "classifier") != "classifier":
        return r
    e = int(epoch)
    if e == CAP_LAST:
        # the cap checkpoints: running-best over {0..199}, frozen before the
        # trajectory continues past the controlled horizon
        slot["cap_slots"] = {k: (int(ep), copy.deepcopy(st))
                             for k, (ep, st) in self.slots.items()}
    if e in slot["grid"]:
        m = slot["model"]
        enc, cls = m._enc_for_inference, m._cls_for_inference
        dev = next(enc.parameters()).device
        if slot["data"].x.device != dev:
            slot["data"] = slot["data"].to(dev)
        was_training = enc.training or cls.training
        enc.eval(); cls.eval()
        with torch.no_grad():
            *_, out = GB.evaluate_ged3(cls, enc, slot["data"], return_output=True)
        if was_training:
            enc.train(); cls.train()
        v = out.squeeze().detach().cpu().numpy()[slot["test_idx"]].astype(np.float64)
        slot["grid_epochs"].append(e)
        slot["grid_scores"].append(v)
    return r


def _fit(self, *args, **kw):
    traj = kw.get("trajectory")
    if _CTX["record"] and traj is not None:
        _ACTIVE[id(traj)] = dict(model=self, grid=set(_CTX["grid"]), data=_CTX["data"],
                                 test_idx=_CTX["test_idx"], grid_epochs=[], grid_scores=[],
                                 cap_slots=None)
    return _orig_fit(self, *args, **kw)


def record_train(method, data, seed, epochs, device, off=None, cfg=None, grid=(),
                 dataset="bail"):
    """The pilot's original train() with recording on; returns (fn, hist, e, traj)."""
    from utils.data import get_dataset
    d, sens_idx, _, _ = get_dataset(dataset, split_seed=P.SPLIT[0],
                                    feature_normalize=cfg.get("fairgb_get_dataset_normalize", True))
    d.sens_idx = sens_idx
    tmask = d.test_mask.cpu().numpy()
    test_idx = np.nonzero(tmask)[0]
    # FairGB.fit moves its own copy with `data = data.to(args.device)`; this
    # harness copy is built the same way from the same loader call the pilot
    # makes, and is moved once so the grid hook scores on the modules' device.
    d = d.to(device)
    _CTX.update(record=True, grid=tuple(grid), data=d, test_idx=test_idx)
    try:
        fn, hist, e = _orig_train(method, data, seed, epochs, device, off=off, cfg=cfg)
    finally:
        _CTX.update(record=False, grid=(), data=None, test_idx=None)
    slot = _ACTIVE.pop(id(hist), None)
    if slot is None:
        raise RuntimeError("[x26] no recording was registered for this FairGB fit")
    if slot["cap_slots"] is None:
        raise RuntimeError(f"[x26] cap slots were never frozen (horizon {epochs} <= {CAP_LAST}?)")

    recs = hist.classifier_records()
    y = _np(d.y)[test_idx].astype(int)
    a = _np(d.sens)[test_idx].astype(int)
    # the four selector slots, evaluated through the pipeline's own score_fn
    slots = {}
    for sig in SIGMAS:
        for supp, src in (("cap", slot["cap_slots"]), ("full", hist.slots)):
            ep, st = src[SLOT[sig]]
            sc = np.asarray(fn(st, test_idx), dtype=np.float64)
            slots[f"{supp}_{sig}"] = dict(epoch=int(ep), score=sc)
    traj = dict(
        epochs=np.array([r.epoch for r in recs], dtype=int),
        val_bce=np.array([r.predictive_loss for r in recs], dtype=np.float64),
        val_auc=np.array([r.metrics["auc"] for r in recs], dtype=np.float64),
        val_node_id=np.asarray(hist.ref.node_id, dtype=int),
        grid_epochs=np.array(slot["grid_epochs"], dtype=int),
        grid_scores=(np.stack(slot["grid_scores"]) if slot["grid_scores"]
                     else np.empty((0, len(test_idx)))),
        test_node_id=test_idx.astype(int), test_y=y, test_a=a,
        slot_epoch_cap_bce=slots["cap_bce"]["epoch"], slot_epoch_cap_auc=slots["cap_auc"]["epoch"],
        slot_epoch_full_bce=slots["full_bce"]["epoch"], slot_epoch_full_auc=slots["full_auc"]["epoch"],
        slot_score_cap_bce=slots["cap_bce"]["score"], slot_score_cap_auc=slots["cap_auc"]["score"],
        slot_score_full_bce=slots["full_bce"]["score"], slot_score_full_auc=slots["full_auc"]["score"],
        code_epoch=int(e), seed=int(seed), split=int(P.SPLIT[0]), horizon=int(epochs),
        dataset=dataset, arm="M1" if off is None else "M0",
        cfg_json=json.dumps(cfg or {}, sort_keys=True, default=str),
        off_json=json.dumps(off or {}, sort_keys=True, default=str))
    return fn, hist, e, traj


def traj_path(traj_dir, dataset, split, seed, arm):
    return os.path.join(traj_dir, f"{dataset}_s{int(split)}_seed{int(seed)}_{arm}.npz")


def save_traj(traj_dir, traj):
    p = traj_path(traj_dir, traj["dataset"], traj["split"], traj["seed"], traj["arm"])
    tmp = p[:-4] + ".tmp.npz"
    np.savez(tmp, **{k: np.asarray(v) for k, v in traj.items()})
    os.replace(tmp, p)
    return p


def valid_traj(p, horizon=None, grid=None):
    try:
        z = np.load(p, allow_pickle=False)
        H = int(z["horizon"]) if horizon is None else int(horizon)
        g = tuple(grid) if grid is not None else tuple(GRID[str(z["dataset"])])
        ok = (np.array_equal(z["epochs"], np.arange(H))                 # FairGB logs 0..H-1
              and np.array_equal(z["grid_epochs"], np.array(g))
              and z["grid_scores"].shape[0] == len(g)
              and bool(np.isfinite(z["val_bce"]).all())
              and bool(np.isfinite(z["grid_scores"]).all()))
        for k in ("cap_bce", "cap_auc", "full_bce", "full_auc"):
            ok &= bool(np.isfinite(z[f"slot_score_{k}"]).all())
            ok &= 0 <= int(z[f"slot_epoch_{k}"]) < H
            if k.startswith("cap"):
                ok &= int(z[f"slot_epoch_{k}"]) <= CAP_LAST
        return bool(ok)
    except Exception:                                                  # noqa: BLE001
        return False


def _train(method, data, seed, epochs, device, off=None, cfg=None):
    if method != "FairGB":
        return _orig_train(method, data, seed, epochs, device, off=off, cfg=cfg)
    ds = P.DS[0]
    fn, hist, e, traj = record_train(method, data, seed, epochs, device, off=off, cfg=cfg,
                                     grid=GRID[ds], dataset=ds)
    save_traj(_CTX["traj_dir"], traj)
    return fn, hist, e


def install(traj_dir=None, wrap_train=True):
    TR.ValidationHistory.add = _add
    GB.FairGB.fit = _fit
    if wrap_train:
        _CTX["traj_dir"] = traj_dir
        P.train = _train


def uninstall():
    TR.ValidationHistory.add = _orig_add
    GB.FairGB.fit = _orig_fit
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
            if not valid_traj(p, horizon, GRID[dataset]):
                raise SystemExit(f"[x26] persisted cell (split {split}, run {run}) lacks a valid "
                                 f"trajectory {p}; refusing to resume")
    return len(store.done)


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: x26_fairgb_run.py --traj_dir DIR -- <pilot_tau args>")
    i = argv.index("--")
    own, pilot = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", required=True)
    a = ap.parse_args(own)
    if "--methods" not in pilot or pilot[pilot.index("--methods") + 1] != "FairGB":
        raise SystemExit("[x26] --methods must be exactly FairGB")
    out = _arg(pilot, "--out")
    dataset = _arg(pilot, "--dataset")
    if out is None or dataset not in GRID:
        raise SystemExit("[x26] --out is required and --dataset must be bail or credit")
    os.makedirs(a.traj_dir, exist_ok=True)
    seed0 = int(_arg(pilot, "--seed0", 27))
    me = _arg(pilot, "--method_epochs")
    n = check_resume(out, a.traj_dir, dataset, seed0, int(me) if me else None)
    if n:
        print(f"[x26] resume: {n} persisted cell(s) have valid trajectories")
    install(a.traj_dir)
    sys.argv = ["pilot_tau.py"] + pilot
    return P.main()


if __name__ == "__main__":
    raise SystemExit(main())
