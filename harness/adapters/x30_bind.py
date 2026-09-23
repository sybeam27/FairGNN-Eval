"""X30 BIND adapter: the official `BIND-main/implementations` scripts, unmodified.

Runs in the isolated DGL environment (/home/sypark/x27_dgl_cuda). The three
top-level scripts are executed from their own source files (compiled with their
real file names and line numbers, so tracebacks cite BIND-main lines); nothing
in BIND-main is written.

Pre-registered rules (coordinator, X30 admission of BIND for bail/income):

  Data   the common split `utils.dataloading.load_data(d, feature_normalize=False,
         split_seed=split)` is injected by wrapping BIND's `load_bail` /
         `load_income` (W1) and the scripts' own `get_adj` / `del_adj` (W2, W3).
         Each script keeps its own feature preprocessing (1_training.py:55/60,
         2_influence:142/148, 3_removing:216-223).
  Stage A  (shared by both arms, cached per (dataset, split, seed))
         1_training.py lines 1-161 at parser defaults with --seed <seed>
         (training + checkpoint save; lines 163-166, the post-hoc test print,
         are not executed: they read test labels and print outcomes).
         2_influence_computation_and_save.py whole, --seed <seed>, python
         `random` seeded with <seed> (approximator.py:151 samples with it),
         with H1 scale=25 (README.md:55), L1 fairness cost on validation nodes,
         S5 device placement. Influence must be finite (hard stop).
  Stage B  (the arm) 3_removing_and_testing.py --helpfulness_collection 1
         --seed <seed>: lines 1-362 (ranking) then the loop body 365-409 for a
         single k (plus: round(p*|train|), p=0.01 "1pct" / 0.10 "10pct";
         minus: 0), then the loop 411-416 over H epochs calling the script's
         own train(epoch), recording eval-mode full-graph logits each epoch.
         code_epoch is the official strict-< validation-loss selection.

Shims / wrappers (recorded in bind_recovery/RECOVERY_LOG.md):
  S1  BIND-main on sys.path; private cwd per stage under the cache dir
  S2  ctypes.cdll.LoadLibrary('*.dll') -> no-op (1_training:17, 2_influence:18, 3_removing:26)
  S3  `ipdb` stub (GNNs/gcn.py:1, import only); sys.dont_write_bytecode for BIND imports
  S5  approximator.grad_z_graph_faircost: sens moved to CUDA when gpu>=0
  H1  approximator.s_test_graph_cost(scale=25); damp 0.03 / recursion_depth 5000 at defaults
  L1  s_test_graph_cost idx_test argument replaced by the script's idx_val_vanilla
  W1  load_bail/load_income -> common data in BIND's return format: adjacency =
      common binary symmetric adjacency with its diagonal removed (BIND's edge
      files carry no self-loops) + sp.eye, raw features, LongTensor labels/idx,
      FloatTensor sens
  W2  get_adj (2_influence:77, 3_removing:95) -> same common adjacency without eye
      (BIND's get_adj reads its CSV/edge files)
  W3  del_adj (3_removing:142) -> lines 175-183 verbatim on adj_ori; the
      skipped lines 169-173 only read BIND's CSV header, which is unused
  I1  read-only instrumentation: HVP step counter, isfinite counts of
      s_test_graph_cost's return, parameter/checkpoint hashes
  (S4 torch.load weights_only and S6 CPU-indexed sens are not needed: torch
   2.2.2, and the tst() functions that index sens by CUDA ids are not run.)
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "harness"))
from core.paths import repo as _repo  # noqa: E402  (repositories live under models/)

import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import random
import sys
import textwrap
import time
import types

import numpy as np
import scipy.sparse as sp
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIND = _repo("BIND-main")
IMPL = os.path.join(BIND, "implementations")
CACHE = os.path.join(ROOT, "harness", "results", "x30", "cache", "bind")
DATASETS = ("bail", "income")
BUDGET = {"1pct": 0.01, "10pct": 0.10}
SCALE = 25
PROVENANCE = ("official-repo (README scale 25; fairness cost on validation nodes, "
              "leakage removed; device shims)")


# ---------------------------------------------------------------- utilities
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def _state_hash(model):
    h = hashlib.sha1()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _lines(script):
    with open(os.path.join(IMPL, script)) as f:        # universal newlines (files are CRLF)
        return f.read().split("\n")


def _exec(script, g, a, b, dedent=False, anchors=()):
    """Execute lines a..b (1-based, inclusive) of an official script in g,
    padded so code objects carry the file's real line numbers."""
    L = _lines(script)
    for ln, prefix in anchors:
        if not L[ln - 1].startswith(prefix):
            raise SystemExit(f"[x30_bind] {script}:{ln} does not start with {prefix!r} (hard stop)")
    body = "\n".join(L[a - 1:b]) + "\n"
    if dedent:
        body = textwrap.dedent(body)
    code = compile("\n" * (a - 1) + body, os.path.join(IMPL, script), "exec")
    exec(code, g)


@contextlib.contextmanager
def _env(cwd, argv, log):
    """S1/S2/S3 plus cwd, argv and output redirection, all restored on exit."""
    saved = (os.getcwd(), list(sys.argv), list(sys.path), ctypes.cdll.LoadLibrary)
    if BIND not in sys.path:
        sys.path.insert(0, BIND)
    sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))
    orig_ll = ctypes.cdll.LoadLibrary

    def ll(name, *a, **k):
        if isinstance(name, str) and name.lower().endswith(".dll"):
            return None
        return orig_ll(name, *a, **k)
    os.makedirs(cwd, exist_ok=True)
    os.chdir(cwd)
    sys.argv = list(argv)
    ctypes.cdll.LoadLibrary = ll
    with open(log, "a") as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
        try:
            yield
        finally:
            os.chdir(saved[0]); sys.argv = saved[1]; sys.path[:] = saved[2]
            ctypes.cdll.LoadLibrary = saved[3]
    if BIND not in sys.path:
        sys.path.insert(0, BIND)


@contextlib.contextmanager
def _patched(obj, **attrs):
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def _bind_modules():
    sys.dont_write_bytecode = True        # keep __pycache__ out of BIND-main
    if BIND not in sys.path:
        sys.path.insert(0, BIND)
    sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))
    import implementations.utils as U
    import implementations.approximator as A
    import implementations.GNNs  # noqa: F401  (imported here, bytecode-free, before torch.load)
    return U, A


# ---------------------------------------------------------------- common data
def _common(dataset, split):
    from utils.dataloading import load_data
    adj, feats, labels, itr, iva, ite, sens, sens_idx = load_data(
        dataset, feature_normalize=False, split_seed=split)
    c = adj.coalesce()
    i, v = c.indices().numpy(), c.values().numpy()
    n = feats.shape[0]
    A = sp.coo_matrix((v, (i[0], i[1])), shape=(n, n), dtype=np.float32).tocsr()
    A.setdiag(0); A.eliminate_zeros(); A.data[:] = 1.0
    A = A.tocoo()
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)      # BIND's symmetrisation (no-op here)
    return dict(A=sp.coo_matrix(A), feats=feats.float(), labels=labels.long(), itr=itr.long(),
                iva=iva.long(), ite=ite.long(), sens=sens, sens_idx=int(sens_idx), n=n)


def _loader_wrappers(dataset, C):
    """W1: BIND loader signature -> common data (fresh copies every call)."""
    def make(name):
        def load(ds, *a, **k):
            if ds != dataset:
                raise SystemExit(f"[x30_bind] {name}({ds!r}) called for dataset {dataset!r}")
            adj = C["A"] + sp.eye(C["n"])                     # utils.py:64 / :158
            return (adj, C["feats"].clone(), C["labels"].clone(), C["itr"].clone(),
                    C["iva"].clone(), C["ite"].clone(), C["sens"].float().clone())
        return load
    return dict(load_bail=make("load_bail"), load_income=make("load_income"))


# ---------------------------------------------------------------- Stage A
def _stage_a(dataset, split, seed, C):
    key = f"{dataset}_s{split}_seed{seed}"
    d = os.path.join(CACHE, key)
    os.makedirs(d, exist_ok=True)
    meta_p = os.path.join(d, "stageA.json")
    with open(os.path.join(d, ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(meta_p):
            meta = json.load(open(meta_p))
            ok = (_sha(os.path.join(d, meta["ckpt"])) == meta["ckpt_sha"]
                  and _sha(os.path.join(d, meta["infl"])) == meta["infl_sha"])
            if not ok:
                raise SystemExit(f"[x30_bind] cache {d} hash mismatch (hard stop)")
            meta["cached"] = True
            return d, meta
        meta = _run_stage_a(dataset, split, seed, C, d)
        json.dump(meta, open(meta_p, "w"), indent=1)
        meta["cached"] = False
        return d, meta


def _run_stage_a(dataset, split, seed, C, d):
    U, A = _bind_modules()
    log = os.path.join(d, "stageA.log")
    meta = dict(ckpt=f"gcn_{dataset}.pth", infl=f"final_influence_{dataset}.npy", scale=SCALE,
                fairness_cost_nodes="validation", seed=seed, split=split)
    wr = _loader_wrappers(dataset, C)

    # ---- 1_training.py lines 1-161 at parser defaults
    t0 = time.time()
    with _env(d, ["1_training.py", "--dataset", dataset, "--seed", str(seed)], log), \
            _patched(U, **wr):
        g = {"__name__": "__main__", "__file__": os.path.join(IMPL, "1_training.py")}
        _exec("1_training.py", g, 1, 161,
              anchors=((161, "torch.save(model, 'gcn_'"), (165, "model = torch.load")))
        meta["stage1_epochs"] = int(g["args"].epochs)
        meta["stage1_args"] = {k: getattr(g["args"], k) for k in ("lr", "weight_decay", "hidden", "dropout", "epochs")}
    meta["stage1_sec"] = round(time.time() - t0, 1)
    m = torch.load(os.path.join(d, meta["ckpt"]))
    meta["ckpt_params_nonfinite"] = int(sum((~torch.isfinite(p)).sum().item() for p in m.parameters()))

    # ---- 2_influence_computation_and_save.py, whole script
    st = dict(hvp=0, returned=None, l1=None)
    g2 = {"__name__": "__main__", "__file__": os.path.join(IMPL, "2_influence_computation_and_save.py")}
    orig_gzf, orig_stc, orig_hvp = A.grad_z_graph_faircost, A.s_test_graph_cost, A.hvp

    def gzf(adj, features, idx, labels, sens, model, gpu=-1):             # S5
        if gpu >= 0 and torch.is_tensor(sens) and not sens.is_cuda:
            sens = sens.cuda()
        return orig_gzf(adj, features, idx, labels, sens, model, gpu)

    def hvp(y, w, v):                                                     # I1
        st["hvp"] += 1
        return orig_hvp(y, w, v)

    def stc(*a, **k):
        a = list(a)
        if not torch.equal(a[3], g2["idx_test_vanilla"]):
            raise SystemExit("[x30_bind] L1: s_test_graph_cost idx argument is not idx_test_vanilla")
        a[3] = g2["idx_val_vanilla"]                                      # L1
        k["scale"] = SCALE                                                # H1
        out = orig_stc(*a, **k)
        st["returned"] = [[list(h.shape), int((~torch.isfinite(h)).sum().item())] for h in out]
        st["l1"] = dict(n_val=len(a[3]), overlap_test=len(set(a[3].tolist()) & set(g2["idx_test_vanilla"].tolist())))
        return out

    random.seed(seed)
    t0 = time.time()
    with _env(d, ["2_influence_computation_and_save.py", "--dataset", dataset, "--seed", str(seed)], log), \
            _patched(U, **wr), _patched(A, grad_z_graph_faircost=gzf, s_test_graph_cost=stc, hvp=hvp):
        _exec("2_influence_computation_and_save.py", g2, 1, 136,
              anchors=((77, "def get_adj"), (126, "def del_adj"), (137, "if dataset_name == 'bail'")))
        g2["get_adj"] = lambda name: sp.coo_matrix(C["A"])                # W2
        _exec("2_influence_computation_and_save.py", g2, 137, len(_lines("2_influence_computation_and_save.py")))
    meta["stage2_sec"] = round(time.time() - t0, 1)
    meta["hvp_steps"] = st["hvp"]
    meta["s_test_returned"] = st["returned"]
    meta["l1"] = st["l1"]
    infl = np.load(os.path.join(d, meta["infl"]), allow_pickle=True).astype(np.float64)
    meta["infl_len"] = int(infl.shape[0])
    meta["infl_nan"] = int(np.isnan(infl).sum()); meta["infl_inf"] = int(np.isinf(infl).sum())
    bad = meta["infl_nan"] + meta["infl_inf"] + sum(r[1] for r in st["returned"])
    if bad or meta["ckpt_params_nonfinite"] or infl.shape[0] != len(C["itr"]):
        raise SystemExit(f"[x30_bind] Stage A not finite / wrong length for {d}: {meta} (hard stop)")
    meta["ckpt_sha"] = _sha(os.path.join(d, meta["ckpt"]))
    meta["infl_sha"] = _sha(os.path.join(d, meta["infl"]))
    return meta


# ---------------------------------------------------------------- Stage B
def config():
    return dict(stageB_lr=0.01, stageB_weight_decay=0.0, hidden=16, dropout=0.5,
                stageA_scale=SCALE, stageA_damp=0.03, stageA_recursion_depth=5000,
                fairness_cost_nodes="validation", ranking="helpfulness_collection=1")


def train_arm(dataset, encoder, arm, split, seed, epochs, device="cuda"):
    if dataset not in DATASETS:
        raise SystemExit(f"[x30_bind] dataset {dataset!r} not admitted")
    if encoder not in BUDGET:
        raise SystemExit(f"[x30_bind] encoder must be one of {tuple(BUDGET)}")
    U, A = _bind_modules()
    C = _common(dataset, split)
    n, itr_np = C["n"], C["itr"].numpy()
    iva_np, ite_np = C["iva"].numpy(), C["ite"].numpy()
    cdir, meta = _stage_a(dataset, split, seed, C)

    p = BUDGET[encoder]
    k = int(round(p * len(itr_np))) if arm == "plus" else 0
    wdir = os.path.join(cdir, f"stageB_{encoder}_{arm}_{os.getpid()}")
    os.makedirs(wdir, exist_ok=True)
    for fn in (meta["ckpt"], meta["infl"]):
        dst = os.path.join(wdir, fn)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(cdir, fn), dst)
    ckpt_sha = _sha(os.path.join(wdir, meta["ckpt"]))
    wr = _loader_wrappers(dataset, C)
    S3 = "3_removing_and_testing.py"
    g = {"__name__": "__main__", "__file__": os.path.join(IMPL, S3)}
    st = dict(k=k, p=p, stageA=meta, ckpt_sha=ckpt_sha)
    scores = []
    t0 = time.time()
    with _env(wdir, [S3, "--dataset", dataset, "--seed", str(seed), "--helpfulness_collection", "1"],
              os.path.join(wdir, "stageB.log")), _patched(U, **wr):
        _exec(S3, g, 1, 213, anchors=((95, "def get_adj"), (142, "def del_adj"),
                                       (209, "def feature_norm"), (214, "if dataset_name == 'bail'")))
        g["get_adj"] = lambda name: sp.coo_matrix(C["A"])                 # W2

        def del_adj(harmful, dataset_name):                               # W3 = lines 175-183
            adj = g["adj_ori"]
            mask = np.ones(adj.shape[0], dtype=bool)
            mask[harmful] = False
            adj = sp.coo_matrix(adj.tocsr()[mask, :][:, mask])
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = adj + sp.eye(adj.shape[0])
            return adj
        g["del_adj"] = del_adj
        _exec(S3, g, 214, 362, anchors=((363, "for num_of_deleting"), (365, "    adj, features, labels"),
                                         (411, "    for epoch in range(args.epochs)")))
        if g["open_factor"] != 1:
            raise SystemExit("[x30_bind] helpfulness_collection is not 1")
        st["ranking_len"] = len(g["harmful"])
        if k > len(g["harmful"]):
            raise SystemExit(f"[x30_bind] k={k} exceeds the non-overlapping ranking ({len(g['harmful'])})")
        g["num_of_deleting"] = k                                          # loop variable for this k
        _exec(S3, g, 365, 409, dedent=True)                               # official loop body
        deleted = np.asarray(g["harmful_flags"], dtype=np.int64)
        model = g["model"]
        st["init_hash"] = _state_hash(model)
        # official loop 411-416 over H epochs, plus eval-mode logits per epoch
        keep = np.ones(n, dtype=bool); keep[deleted] = False
        kept_ids = np.where(keep)[0]
        for epoch in range(epochs):
            loss_mid = g["train"](epoch)
            if loss_mid < g["loss_val_global"]:
                g["loss_val_global"] = loss_mid
                torch.save(g["model"], "mid_best" + str(g["bin"]) + ".pth")
                g["final_epochs"] = epoch
            model.eval()
            with torch.no_grad():
                out = model(g["features"], g["edge_index"]).squeeze(-1).float().cpu().numpy()
            full = np.zeros(n, dtype=np.float32)
            full[kept_ids] = out
            scores.append(full)
        code_epoch = int(g["final_epochs"])
        mb = "mid_best" + str(g["bin"]) + ".pth"
        if os.path.exists(mb):
            os.remove(mb)                                                 # as 3_removing:422
    st["stageB_sec"] = round(time.time() - t0, 1)
    st["n_deleted"] = int(len(deleted))
    st["deleted"] = deleted.tolist()
    sc = np.stack(scores) if scores else np.zeros((0, n), np.float32)
    vt = np.concatenate([iva_np, ite_np])
    st["valtest_nonfinite"] = int((~np.isfinite(sc[:, vt])).sum())
    st["deleted_in_valtest"] = int(np.isin(deleted, vt).sum())
    st["reindexed_ok"] = bool(len(kept_ids) == g["features"].shape[0])

    tag = f"BIND-{encoder} " + ("M+I" if arm == "plus" else "M-I")
    ch = {f"{tag} Stage A influence finite and full length":
          (meta["infl_nan"] + meta["infl_inf"] == 0 and meta["infl_len"] == len(itr_np)
           and meta["hvp_steps"] == 5000 and all(r[1] == 0 for r in meta["s_test_returned"]),
           f"(hvp {meta['hvp_steps']}/5000, s_test non-finite {[r[1] for r in meta['s_test_returned']]}, "
           f"vector nan {meta['infl_nan']} inf {meta['infl_inf']} len {meta['infl_len']}, "
           f"cached {meta['cached']}, stageA {meta.get('stage1_sec')}+{meta.get('stage2_sec')}s)"),
          f"{tag} L1 fairness cost on validation nodes, disjoint from test":
          (meta["l1"]["overlap_test"] == 0 and meta["l1"]["n_val"] == len(iva_np),
           f"(n_val {meta['l1']['n_val']}, overlap {meta['l1']['overlap_test']})"),
          f"{tag} validation/test logits finite":
          (st["valtest_nonfinite"] == 0, f"(non-finite {st['valtest_nonfinite']})"),
          f"{tag} graph reindexed to the kept nodes":
          (st["reindexed_ok"], f"({len(kept_ids)} kept of {n})"),
          f"{tag} code epoch selected (strict < val loss)":
          (0 <= code_epoch < epochs, f"(epoch index in range; stageB {st['stageB_sec']}s)")}
    if arm == "plus":
        ch[f"{tag} exactly k>0 training nodes deleted"] = (
            k > 0 and st["n_deleted"] == k and bool(np.isin(deleted, itr_np).all())
            and len(set(deleted.tolist())) == k,
            f"(k={k} = round({p}*{len(itr_np)}), deleted {st['n_deleted']}, ranking {st['ranking_len']})")
    else:
        ch[f"{tag} no node deleted"] = (st["n_deleted"] == 0, f"(deleted {st['n_deleted']})")

    def pair_checks(other):
        o = other["gate"]
        return {"BIND arms loaded the byte-identical Stage-A checkpoint":
                (st["ckpt_sha"] == o["ckpt_sha"] == meta["ckpt_sha"], f"({st['ckpt_sha']})"),
                "BIND arms start from identical parameters":
                (st["init_hash"] == o["init_hash"], f"({st['init_hash']})"),
                "BIND validation and test node ids never deleted (both arms)":
                (st["deleted_in_valtest"] == 0 and o["deleted_in_valtest"] == 0, "")}

    masks = tuple(np.isin(np.arange(n), ix) for ix in (itr_np, iva_np, ite_np))
    return dict(scores=scores, code_epoch=code_epoch, gate=st, gate_checks=ch, pair_checks=pair_checks,
                provenance=PROVENANCE, masks=masks,
                config=dict(config(), arm=arm, budget=encoder, p=p, k=k, epochs=epochs,
                            n_train=len(itr_np), stage1=meta.get("stage1_args")))
