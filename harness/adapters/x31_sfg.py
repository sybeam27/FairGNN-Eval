"""X31 SFG adapter: the official `run()` of `sfg.py` / `sfg_credit.py`, unmodified.

SFG ("Stable Fair Graph Representation Learning with Lipschitz Constraint")
builds on FairVGNN. Its `run.sh` publishes, per dataset, the authors' own
three-row ablation ladder:

    1. FairVGNN (no --with_constraint)                        -> M^-I
    2. + --with_constraint --rho=2
    3. + --with_constraint --rho=2 --loss_alpha=0.5 (SFG)     -> M^+I

I = the Lipschitz constraint on the encoder weights (rho = 2) together with the
stronger chi-square DRO generator loss (loss_alpha 1 -> 0.5). The generator loss
goes through `loss_chisq` in every row whenever `f_mask == 'yes'`; the
unconstrained row does so at its default loss_alpha = 1. Rows 1 and 3 are asserted to agree on every
other flag. The script named on the row is the one executed.

Wrappers only (restored after each call):
* data: the common split from `utils.data.get_dataset` at SFG's own feature
  policy (german raw; others normalised with the sensitive column kept), with
  SFG's own `sens_correlation` for `corr_idx` and `x_min` / `x_max` taken from
  the raw features, as SFG's loader computes them;
* `args.train_ratio` / `args.val_ratio`: the few lines of `sfg.py`'s
  `__main__` that build them, reproduced with their line reference;
* output extraction: during `evaluate_ged3` the classifier's outputs are
  captured and averaged exactly as the function averages them (K draws when
  `f_mask == 'yes'`), so no extra random number is drawn;
* gate instrumentation: calls of `loss_chisq` and of the constraint projection
  with the constraint active; parameter hashes after reset;
* module import isolated from this repository's `utils` package; private cwd
  (the script appends to `logs/`).
"""
from __future__ import annotations

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "harness"))
from core.paths import repo as _repo  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFG = _repo("SFG-main")
DATASETS = ("german", "bail", "credit")
CACHE = os.path.join(ROOT, "harness", "results", "x31", "cache", "sfg")


def _parser_defaults(script):
    out = {}
    for node in ast.walk(ast.parse(open(os.path.join(SFG, script)).read())):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)):
            kw = {k.arg: k.value for k in node.keywords}
            name = node.args[0].value.lstrip("-")
            if "action" in kw and ast.literal_eval(kw["action"]) == "store_true":
                out[name] = False
                continue
            if "default" in kw:
                typ = {"int": int, "float": float, "str": str}.get(
                    getattr(kw.get("type"), "id", "str"), str)
                out[name] = typ(ast.literal_eval(kw["default"]))
    return out


def _ladder():
    """The run.sh command strings, grouped by dataset, in file order."""
    rows = []
    for line in open(os.path.join(SFG, "run.sh")).read().splitlines():
        s = line.strip()
        if not s.startswith('"python '):
            continue
        cmd = s[1:s.index('"', 1)]
        tok = shlex.split(cmd)
        flags, switches = {}, set()
        for t in tok[2:]:
            m = re.match(r"--([A-Za-z_]+)=(.*)$", t)
            if m:
                flags[m.group(1)] = m.group(2)
            elif t.startswith("--"):
                switches.add(t[2:])
        rows.append(dict(script=tok[1], flags=flags, switches=switches))
    return rows


def config(dataset):
    rows = [r for r in _ladder() if r["flags"].get("dataset") == dataset]
    if len(rows) != 3:
        raise SystemExit(f"[sfg] expected the three-row ladder for {dataset}, found {len(rows)}")
    minus_row = [r for r in rows if "with_constraint" not in r["switches"]]
    plus_row = [r for r in rows if "with_constraint" in r["switches"] and "loss_alpha" in r["flags"]]
    if len(minus_row) != 1 or len(plus_row) != 1:
        raise SystemExit(f"[sfg] ladder for {dataset} is not FairVGNN / constraint / SFG")
    minus_row, plus_row = minus_row[0], plus_row[0]
    if minus_row["script"] != plus_row["script"]:
        raise SystemExit(f"[sfg] {dataset}: the two rows name different scripts")
    base = {k: v for k, v in plus_row["flags"].items() if k not in ("rho", "loss_alpha")}
    if base != minus_row["flags"]:
        raise SystemExit(f"[sfg] {dataset}: the SFG row changes more than the intervention "
                         f"({base} vs {minus_row['flags']})")
    defaults = _parser_defaults(plus_row["script"])

    def build(row):
        d = dict(defaults)
        for k, v in row["flags"].items():
            d[k] = type(defaults[k])(v) if defaults.get(k) is not None else v
        d["with_constraint"] = "with_constraint" in row["switches"]
        return d
    return dict(script=plus_row["script"], plus=build(plus_row), minus=build(minus_row))


def native_horizon(dataset, encoder=None):
    return int(config(dataset)["plus"]["epochs"])


_MODS: dict = {}


def _load(script):
    if script in _MODS:
        return _MODS[script]
    names = ("utils", "dataset", "model", "learn", "constraint")
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k in names or k.startswith("utils.")}
    saved_path = list(sys.path)
    try:
        sys.path.insert(0, SFG)
        spec = importlib.util.spec_from_file_location(
            "_x31_sfg_" + script.replace(".py", ""), os.path.join(SFG, script))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved_path
        for k in names:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    _MODS[script] = mod
    return mod


def _hash(m):
    h = hashlib.sha1()
    for k, v in m.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def train_arm(dataset, encoder, arm, split, seed, epochs, device="cuda"):
    from utils.data import get_dataset
    cfg = config(dataset)
    mod = _load(cfg["script"])
    flags = dict(cfg["plus"] if arm == "plus" else cfg["minus"])

    norm = dataset != "german"
    d, sens_idx, _, _ = get_dataset(dataset, feature_normalize=norm, split_seed=split)
    raw, _, _, _ = get_dataset(dataset, feature_normalize=False, split_seed=split)
    x_max, x_min = torch.max(raw.x, dim=0)[0], torch.min(raw.x, dim=0)[0]   # dataset.py:442
    corr_matrix = mod.sens_correlation(d.x, sens_idx)                     # dataset.py:449-453
    corr_idx = np.argsort(-np.abs(corr_matrix))
    if flags["top_k"] > 0:
        corr_idx = corr_idx[:flags["top_k"]]

    args = argparse.Namespace(**flags)
    args.dataset, args.runs, args.seed, args.epochs = dataset, 1, int(seed), int(epochs)
    args.device = torch.device(device)
    args.sens_idx, args.corr_sens, args.corr_idx = sens_idx, corr_matrix, corr_idx
    args.x_min, args.x_max = x_min, x_max
    args.num_features, args.num_classes = d.x.shape[1], 1
    # sfg.py:447-453, verbatim in effect
    args.train_ratio, args.val_ratio = torch.tensor([
        (d.y[d.train_mask] == 0).sum(), (d.y[d.train_mask] == 1).sum()]), torch.tensor([
            (d.y[d.val_mask] == 0).sum(), (d.y[d.val_mask] == 1).sum()])
    args.train_ratio, args.val_ratio = (torch.max(args.train_ratio) / args.train_ratio,
                                        torch.max(args.val_ratio) / args.val_ratio)
    args.train_ratio, args.val_ratio = (args.train_ratio[d.y[d.train_mask].long()],
                                        args.val_ratio[d.y[d.val_mask].long()])

    scores, code = [], dict(best=0.0, epoch=-1, n=0)
    gate = dict(chisq=0, chisq_alphas=set(), constrained_projections=0, init_hash={}, nonfinite=0)
    cap = dict(on=False, outs=[])
    orig_eval, orig_chisq = mod.evaluate_ged3, mod.loss_chisq
    cls_cls, cnst_cls = mod.MLP_classifier, mod.Constraint_SAGE
    orig_cls_fwd, orig_obe = cls_cls.forward, cnst_cls.on_batch_end
    reset_classes = [cls_cls, mod.MLP_discriminator, mod.channel_masker, mod.SAGE_encoder]
    orig_resets = {c: c.reset_parameters for c in reset_classes}

    def cls_fwd(self, *a, **k):
        y = orig_cls_fwd(self, *a, **k)
        if cap["on"]:
            cap["outs"].append(y.detach())
        return y

    def evaluate(x, classifier, discriminator, generator, encoder_, data, a):
        cap["on"], cap["outs"] = True, []
        try:
            res = orig_eval(x, classifier, discriminator, generator, encoder_, data, a)
        finally:
            cap["on"] = False
        outs = cap["outs"]
        if a.f_mask == "yes":
            if len(outs) != a.K:
                raise SystemExit(f"[sfg] captured {len(outs)} classifier calls, expected K={a.K}")
            out = torch.stack(outs).mean(dim=0)                    # learn.py:326
        else:
            out = outs[0]
        v = out.squeeze().float().cpu().numpy()
        gate["nonfinite"] += int((~np.isfinite(v)).sum())
        scores.append(v)
        accs, aucs, f1s, par, eq = res
        t = aucs["val"] + f1s["val"] + accs["val"] - a.alpha * (par["val"] + eq["val"])
        if t > code["best"]:                                        # sfg.py:334-344
            code["best"], code["epoch"] = t, code["n"]
        code["n"] += 1
        return res

    def chisq(loss_vector, alpha):
        gate["chisq"] += 1
        gate["chisq_alphas"].add(float(alpha))
        return orig_chisq(loss_vector, alpha)

    def obe(self, *a, **k):
        if self.with_constraint:
            gate["constrained_projections"] += 1
        return orig_obe(self, *a, **k)

    def make_reset(c):
        def reset(self):
            orig_resets[c](self)
            gate["init_hash"].setdefault(c.__name__, _hash(self))
        return reset

    run_dir = os.path.join(CACHE, "cwd", str(os.getpid()))
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    cwd = os.getcwd()
    try:
        mod.evaluate_ged3, mod.loss_chisq = evaluate, chisq
        cls_cls.forward, cnst_cls.on_batch_end = cls_fwd, obe
        for c in reset_classes:
            c.reset_parameters = make_reset(c)
        os.chdir(run_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            mod.run(d, args)
    finally:
        os.chdir(cwd)
        mod.evaluate_ged3, mod.loss_chisq = orig_eval, orig_chisq
        cls_cls.forward, cnst_cls.on_batch_end = orig_cls_fwd, orig_obe
        for c, r in orig_resets.items():
            c.reset_parameters = r
    if len(scores) != epochs:
        raise SystemExit(f"[sfg] {len(scores)} evaluations for {epochs} epochs")

    tag = "M+I" if arm == "plus" else "M-I"
    # sfg.py routes the generator loss through loss_chisq whenever f_mask == 'yes',
    # in every row; the authors' unconstrained FairVGNN row does so at its default
    # loss_alpha = 1. What the ladder switches is the constraint and loss_alpha.
    want_alpha = {float(flags["loss_alpha"])}
    if arm == "plus":
        checks = {f"{tag} Lipschitz constraint applied": (gate["constrained_projections"] > 0,
                                                          f"({gate['constrained_projections']} projections)"),
                  f"{tag} DRO generator loss at loss_alpha={flags['loss_alpha']}":
                      (gate["chisq"] > 0 and gate["chisq_alphas"] == want_alpha,
                       f"({gate['chisq']} calls, alphas {sorted(gate['chisq_alphas'])})")}
    else:
        checks = {f"{tag} no constraint applied": (gate["constrained_projections"] == 0,
                                                   f"({gate['constrained_projections']})"),
                  f"{tag} DRO generator loss at the unconstrained row's loss_alpha={flags['loss_alpha']}":
                      (gate["chisq_alphas"] <= want_alpha,
                       f"({gate['chisq']} calls, alphas {sorted(gate['chisq_alphas'])})")}

    def pair_checks(other):
        a_, b_ = gate["init_hash"], other["gate"]["init_hash"]
        common = sorted(set(a_) & set(b_))
        return {"paired arms share initial parameters of every module":
                (bool(common) and all(a_[k] == b_[k] for k in common),
                 "(" + ", ".join(f"{k}:{'=' if a_[k] == b_[k] else '!='}" for k in common) + ")")}

    return dict(scores=scores, code_epoch=code["epoch"], gate=gate, gate_checks=checks,
                pair_checks=pair_checks, provenance="official-repo (run.sh ablation ladder)",
                masks=(d.train_mask.cpu().numpy(), d.val_mask.cpu().numpy(), d.test_mask.cpu().numpy()),
                config=dict(script=cfg["script"], with_constraint=flags["with_constraint"],
                            rho=flags["rho"], loss_alpha=flags["loss_alpha"], encoder=flags["encoder"],
                            c_lr=flags["c_lr"], e_lr=flags["e_lr"], ratio=flags["ratio"],
                            native_epochs=cfg["plus"]["epochs"]))
