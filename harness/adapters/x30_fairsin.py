"""X30 FairSIN adapter: the official `run()` executed unmodified, with wrappers.

The X2 adapter (`adapters/fairsin.py`) transcribed the driver. This one does
not: `FairSIN-main/in-train.py` and `FairSIN-main/train_mlp.py` are loaded as
modules and their own `run(data, args)` is called. Everything added here is a
wrapper around a module global or a class attribute, restored after the call:

* seed/split injection -- `sklearn.model_selection.train_test_split` receives
  `random_state=seed` (the official call is unseeded, so the neutraliser's
  inner split would otherwise differ between the paired arms);
* output extraction -- `evaluate` is wrapped: the classifier logits over all
  nodes are recomputed exactly as `evaluation.evaluate` computes them (eval
  mode, no grad, raw `data.x`), then the original is called;
* cache namespace -- the heterophilous-neighbour matrix `<dataset>_hadj.pt` is
  computed by the official code path once per dataset, inside
  `harness/results/x30/cache/fairsin/`, never in FairSIN-main (whose
  `german_hadj.pt` was built from the authors' loader, not this split's data);
  every arm then takes the official cache-hit branch. The cache-miss branch
  subtracts the identity from `data.adj` and the hit branch does not, which
  `train_mlp.py`'s degree term reads, so mixing the two would make arms differ
  by more than the intervention;
* gate instrumentation -- call counters and norms on the neutraliser MLP and
  the discriminator, and a hash of every module right after its reset.

Configuration rule (X30 protocol section on FairSIN, fixed before any outcome):
the authors' ablation block `experiment.sh` lines 50-116 is parsed. For each
(dataset, encoder) M+I is the first row with delta > 0 and d == 'yes', else the
first row with delta > 0; M-I is that row with delta = 0 and d = 'no'. Nothing
else changes. The script named on that row is the one executed.
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "harness"))
from core.paths import repo as _repo  # noqa: E402  (repositories live under models/)

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import os
import re
import shlex
import sys
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FS = _repo("FairSIN-main")
ABLATION_LINES = (50, 116)
DATASETS = ("german", "bail", "credit", "pokec_z", "pokec_n")
ENCODERS = ("GCN", "GIN", "SAGE")
CACHE = os.path.join(ROOT, "harness", "results", "x30", "cache", "fairsin")


# --------------------------------------------------------------------------
# configuration, parsed from the repository (no transcription)
# --------------------------------------------------------------------------
def _parser_defaults(script: str) -> dict:
    """Defaults of the script's own argparse, read with `ast` from its source."""
    tree = ast.parse(open(os.path.join(FS, script)).read())
    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)):
            name = node.args[0].value.lstrip("-")
            kw = {k.arg: k.value for k in node.keywords}
            if "default" in kw:
                typ = {"int": int, "float": float, "str": str}.get(
                    getattr(kw.get("type"), "id", "str"), str)
                out[name] = typ(ast.literal_eval(kw["default"]))
    return out


def _ablation_rows() -> list[dict]:
    rows = []
    lines = open(os.path.join(FS, "experiment.sh")).read().splitlines()
    lo, hi = ABLATION_LINES
    for no in range(lo, hi + 1):
        s = lines[no - 1].strip()
        if not s.startswith("python "):
            continue
        tok = shlex.split(s)
        flags = {}
        for t in tok[2:]:
            m = re.match(r"--([A-Za-z_]+)=(.*)$", t)
            if m:
                flags[m.group(1)] = m.group(2)
        rows.append(dict(line=no, script=tok[1], flags=flags))
    return rows


def _norm_flags(defaults: dict, flags: dict) -> dict:
    """argparse semantics: `--epoch` is a unique prefix of `--epochs`."""
    out = {}
    for k, v in flags.items():
        full = k if k in defaults else [d for d in defaults if d.startswith(k)]
        if isinstance(full, list):
            if len(full) != 1:
                raise SystemExit(f"[fairsin] ambiguous flag --{k}")
            full = full[0]
        out[full] = type(defaults[full])(v)     # the declared argparse type
    return out


def config(dataset: str, encoder: str) -> dict:
    rows = [r for r in _ablation_rows()
            if r["flags"].get("dataset") == dataset and r["flags"].get("encoder") == encoder]
    if not rows:
        raise KeyError(f"no ablation rows for FairSIN {dataset}/{encoder}")
    pos = [r for r in rows if float(r["flags"].get("delta", "nan")) > 0]
    full = [r for r in pos if r["flags"].get("d") == "yes"]
    plus_row = (full or pos)[0]
    script = plus_row["script"]
    defaults = _parser_defaults(script)
    plus = dict(defaults); plus.update(_norm_flags(defaults, plus_row["flags"]))
    minus = dict(plus, delta=0.0, d="no")
    off_rows = [r for r in rows if float(r["flags"].get("delta", "nan")) == 0
                and r["flags"].get("d") != "yes"]
    return dict(script=script, plus=plus, minus=minus, plus_line=plus_row["line"],
                off_line=(off_rows[0]["line"] if off_rows else None),
                rows=[r["line"] for r in rows])


def native_horizon(dataset: str, encoder: str) -> int:
    """`--epoch(s)` on the M+I ablation row (else the parser default)."""
    return int(config(dataset, encoder)["plus"]["epochs"])


# --------------------------------------------------------------------------
# module loading, isolated from this repository's `utils` package
# --------------------------------------------------------------------------
_MODS: dict = {}


def _load(script: str):
    if script in _MODS:
        return _MODS[script]
    if torch.cuda.is_available():
        torch.cuda.init()          # binds the visible device before the script's
        #                            os.environ['CUDA_VISIBLE_DEVICES'] = '6' runs
    names = ("utils", "dataset", "model", "evaluation")
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k in names or k.startswith("utils.")}
    saved_path = list(sys.path)
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    stub = sys.modules.get("memory_profiler")
    sys.modules["memory_profiler"] = types.SimpleNamespace(memory_usage=lambda *a, **k: [0.0])
    try:
        sys.path.insert(0, FS)
        spec = importlib.util.spec_from_file_location(
            "_x30_fairsin_" + script.replace("-", "_").replace(".py", ""),
            os.path.join(FS, script))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved_path
        if env is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = env
        for k in names:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
        if stub is None:
            sys.modules.pop("memory_profiler", None)
        else:
            sys.modules["memory_profiler"] = stub
    _MODS[script] = mod
    return mod


# --------------------------------------------------------------------------
# data: the common split, in the Data form FairSIN's run() reads
# --------------------------------------------------------------------------
def fairsin_data(dataset: str, split: int):
    """utils.data.get_dataset at FairSIN's own feature policy (dataset.py:462),
    plus `adj`, the scipy adjacency the repository's loader returns and ours
    does not. Rebuilt from edge_index, self-loops included as in dataset.py."""
    import scipy.sparse as sp
    from utils.data import get_dataset
    d, sens_idx, x_min, x_max = get_dataset(dataset, feature_normalize=(dataset != "german"),
                                            split_seed=split)
    n = d.x.shape[0]
    ei = d.edge_index.cpu().numpy()
    adj = sp.coo_matrix((np.ones(ei.shape[1], dtype=np.float32), (ei[0], ei[1])),
                        shape=(n, n)).tocsr()
    adj.data[:] = 1.0
    d.adj = adj
    return d, sens_idx, x_min, x_max


def _hash_state(m) -> str:
    h = hashlib.sha1()
    for k, v in m.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


_HADJ: dict = {}


def _ensure_hadj(mod, dataset, split, device):
    """Compute `<dataset>_hadj.pt` with the official cache-miss branch, once.

    It depends on the graph and the sensitive attribute only, neither of which
    varies with the split, so one file per dataset. `run()` is entered on a
    fresh Data with epochs=0; it saves the file and then fails on the empty
    loop (`test_acc` unbound), which is expected and checked for.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{dataset}_hadj.pt")
    if not os.path.exists(path):
        d, sens_idx, x_min, x_max = fairsin_data(dataset, split)
        a = argparse.Namespace(**dict(_parser_defaults("in-train.py"), dataset=dataset,
                                      runs=1, epochs=0, device=device, sens_idx=sens_idx,
                                      num_features=d.x.shape[1], num_classes=1,
                                      encoder="GCN"))
        cwd = os.getcwd()
        os.chdir(CACHE)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mod.run(d, a)
        except UnboundLocalError:
            pass
        finally:
            os.chdir(cwd)
        if not os.path.exists(path):
            raise SystemExit(f"[fairsin] cache build did not write {path}")
    if dataset not in _HADJ:
        t = torch.load(path)
        _HADJ[dataset] = (t, hashlib.sha1(t.numpy().tobytes()).hexdigest()[:16])
    return path


class _TorchProxy(types.ModuleType):
    """`torch` for the FairSIN module, with `load` of the hadj file memoised.

    The official hit branch is `torch.load(args.dataset + '_hadj.pt')`; on
    pokec that is an 18 GB dense tensor, read once per process instead of once
    per arm. The tensor is only read by run(); its hash is checked after every
    arm."""

    def __init__(self, real, memo):
        super().__init__("torch")
        self._real, self._memo = real, memo

    def __getattr__(self, k):
        return getattr(self._real, k)

    def load(self, f, *a, **k):
        if isinstance(f, str) and f in self._memo:
            return self._memo[f]
        return self._real.load(f, *a, **k)


# --------------------------------------------------------------------------
# one arm
# --------------------------------------------------------------------------
def train_arm(dataset: str, encoder: str, arm: str, split: int, seed: int, epochs: int,
              device: str = "cuda") -> dict:
    cfg = config(dataset, encoder)
    mod = _load(cfg["script"])
    dev = torch.device(device)
    _ensure_hadj(mod, dataset, split, dev)
    hadj, hadj_hash = _HADJ[dataset]

    d, sens_idx, x_min, x_max = fairsin_data(dataset, split)
    flags = dict(cfg["plus"] if arm == "plus" else cfg["minus"])
    args = argparse.Namespace(**flags)
    args.dataset, args.encoder = dataset, encoder
    args.runs, args.seed, args.epochs = 1, int(seed), int(epochs)
    args.device = dev
    args.sens_idx, args.x_min, args.x_max = sens_idx, x_min, x_max
    args.num_features, args.num_classes = d.x.shape[1], 1

    scores, code = [], dict(best=0.0, epoch=-1, n=0)
    gate = dict(mlp_calls=0, mlp_term_abs=0.0, disc_calls=0, init_hash={}, nonfinite=0)

    import sklearn.model_selection as skms
    orig_tts = skms.train_test_split
    orig_eval = mod.evaluate
    orig_torch = mod.torch
    orig_mlp_fwd = mod.MLP.forward
    disc_cls = mod.MLP_discriminator
    orig_disc_fwd = disc_cls.forward
    enc_classes = [mod.GCN_encoder_scatter, mod.GCN_encoder_spmm, mod.GIN_encoder,
                   mod.SAGE_encoder, mod.MLP_classifier, disc_cls, mod.MLP]
    orig_resets = {c: c.reset_parameters for c in enc_classes}
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
             torch.backends.cudnn.allow_tf32)

    def tts(*a, **k):
        k.setdefault("random_state", int(seed))
        return orig_tts(*a, **k)

    def evaluate(x, classifier, hp, encoder_, data, a):
        classifier.eval(); encoder_.eval()
        with torch.no_grad():
            out = classifier(encoder_(data.x, data.edge_index, data.adj_norm_sp))
        v = out.squeeze().detach().float().cpu().numpy()
        gate["nonfinite"] += int((~np.isfinite(v)).sum())
        scores.append(v)
        res = orig_eval(x, classifier, hp, encoder_, data, a)
        accs, aucs, f1s, par, eq = res
        t = aucs["val"] + f1s["val"] + accs["val"] - a.alpha * (par["val"] + eq["val"])
        if t > code["best"]:                     # in-train.py:227, strict
            code["best"], code["epoch"] = t, code["n"]
        code["n"] += 1
        return res

    def mlp_fwd(self, x):
        y = orig_mlp_fwd(self, x)
        if x.shape[0] == d.x.shape[0]:           # the term added to data.x
            gate["mlp_calls"] += 1
            gate["mlp_term_abs"] += float(args.delta) * float(y.detach().abs().mean())
        return y

    def disc_fwd(self, *a, **k):
        gate["disc_calls"] += 1
        return orig_disc_fwd(self, *a, **k)

    def make_reset(c):
        def reset(self):
            orig_resets[c](self)
            gate["init_hash"].setdefault(c.__name__, _hash_state(self))
        return reset

    memo = {f"{dataset}_hadj.pt": hadj}
    cwd = os.getcwd()
    try:
        skms.train_test_split = tts
        mod.evaluate = evaluate
        mod.torch = _TorchProxy(torch, memo)
        mod.MLP.forward = mlp_fwd
        disc_cls.forward = disc_fwd
        for c in enc_classes:
            c.reset_parameters = make_reset(c)
        os.chdir(CACHE)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                mod.run(d, args)
            except UnboundLocalError:
                # in-train.py:240 reads test_acc, bound only when the native
                # tradeoff ever exceeded its floor of 0. Training has finished
                # by then; the native selector simply chose nothing.
                if len(scores) != epochs:
                    raise
                code["epoch"] = -1
    finally:
        os.chdir(cwd)
        skms.train_test_split = orig_tts
        mod.evaluate = orig_eval
        mod.torch = orig_torch
        mod.MLP.forward = orig_mlp_fwd
        disc_cls.forward = orig_disc_fwd
        for c, r in orig_resets.items():
            c.reset_parameters = r
        (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
         torch.backends.cudnn.allow_tf32) = cudnn
    if hashlib.sha1(hadj.numpy().tobytes()).hexdigest()[:16] != hadj_hash:
        raise SystemExit("[fairsin] hadj cache mutated during an arm (hard stop)")
    if len(scores) != epochs:
        raise SystemExit(f"[fairsin] {len(scores)} evaluations for {epochs} epochs")
    tag = "M+I" if arm == "plus" else "M-I"
    if arm == "plus":
        checks = {
            f"{tag} neutralisation term nonzero (delta={args.delta})":
                (gate["mlp_calls"] > 0 and gate["mlp_term_abs"] > 0,
                 f"(calls {gate['mlp_calls']}, mean|delta*mlp(x)| summed {gate['mlp_term_abs']:.3g})"),
            f"{tag} discriminator branch runs iff d=='yes' (d={args.d})":
                ((gate["disc_calls"] > 0) == (args.d == "yes"), f"(calls {gate['disc_calls']})"),
        }
    else:
        checks = {
            f"{tag} neutralisation term exactly zero":
                (gate["mlp_term_abs"] == 0.0, f"(summed {gate['mlp_term_abs']:.3g})"),
            f"{tag} discriminator branch never runs":
                (gate["disc_calls"] == 0, f"(calls {gate['disc_calls']})"),
        }

    def pair_checks(other):
        a_, b_ = gate["init_hash"], other["gate"]["init_hash"]
        common = sorted(set(a_) & set(b_))
        return {"paired arms share initial parameters of every module":
                (bool(common) and all(a_[k] == b_[k] for k in common),
                 "(" + ", ".join(f"{k}:{'=' if a_[k] == b_[k] else '!='}" for k in common) + ")")}

    return dict(scores=scores, code_epoch=code["epoch"], gate=gate,
                gate_checks=checks, pair_checks=pair_checks, provenance="official-repo",
                masks=(d.train_mask.cpu().numpy(), d.val_mask.cpu().numpy(), d.test_mask.cpu().numpy()),
                config=dict(script=cfg["script"], plus_line=cfg["plus_line"],
                            off_line=cfg["off_line"], delta=float(args.delta), d=args.d,
                            hidden=args.hidden, c_lr=args.c_lr, e_lr=args.e_lr,
                            m_lr=getattr(args, "m_lr", None), m_epoch=args.m_epoch,
                            dropout=args.dropout, prop=args.prop, alpha=args.alpha))
