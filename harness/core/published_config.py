"""The published protocol of each (method, dataset): configuration, horizon,
published selector, and the provenance of each.

Nothing is hand-transcribed. Every value is parsed at call time from an
artifact on disk:

    utils/param.json                         the harness's own record
    FairGB-main/run.sh                       FairGB upstream, vendored
    harness/provenance/fairvgnn_run_*.sh       FairVGNN upstream, downloaded
    harness/provenance/fairvgnn_main.py        FairVGNN argparse defaults
    harness/provenance/nifty_README.md         NIFTY upstream commands

Provenance grades, from strongest to weakest:

    official-repo          the method's own repository sets it explicitly for
                           this dataset
    official-repo-default  the authors' own script runs this dataset, but leaves
                           the value at their parser default
    paper-specified        the paper states it
    third-party-benchmark  another paper's official artifact configures this
                           method on this dataset (named in `source`)
    local-mirror-verified  the local value matches a fetched official value
    local-unverified       only this harness records it; no artifact agrees

A `local-unverified` setting is never called a published protocol.

What the artifacts say, and do not say:

* FairGB publishes all three datasets (`FairGB-main/run.sh`).
* FairVGNN publishes all three (`run_german.sh`, `run_bail.sh`,
  `run_credit.sh`), GCN encoder, default `scatter` propagation.
* NIFTY's repository states german only, in its README. Bail and credit are
  named in the paper as evaluated, but no artifact gives their command.
* FairGNN's repository has scripts for **nba, pokec_n, pokec_z only**
  (`api.github.com/repos/EnyanDai/FairGNN/contents/src/scripts`). It never
  configured german, bail or credit. The only artifact that configures FairGNN
  on german is NIFTY's README, as a baseline -- third party, hence
  `paper-specified`, and nothing at all covers bail or credit.

The official FairVGNN ablation does not hold configuration fixed. On german the
full line is `clip_e=0.1 e_lr=0.001 ratio=0` and the both-off line is
`clip_e=1 e_lr=0.01 ratio=1`; bail changes `c_lr`, `g_epochs` and the horizon
as well. That ablation therefore measures a retuned model, not the removal of a
component. M0 here keeps M1's configuration and flips only `f_mask` and
`weight_clip`, which is what an isolation-preserving control requires; the
official ablation's own values are returned as `official_ablation` for the
record and are not run.
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "harness"))
from core.paths import repo as _repo  # noqa: E402  (repositories live under models/)

import json
import os
import re
import shlex

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARAM_PATH = os.path.join(ROOT, "utils", "param.json")
FAIRGB_RUN = os.path.join(_repo("FairGB-main"), "run.sh")
PROV = os.path.join(ROOT, "harness", "provenance")

_PKEY = {"bail": "recidivism"}          # utils/train_baselines.py:814
_CLI = dict(lr=1e-3, weight_decay=1e-5, hidden_dim=128, proj_hidden_dim=128)

# utils/train_baselines.py:511-521; the floor is the table value minus 0.3
_FAIRGNN_ACC = {"german": 0.66, "recidivism": 0.84, "credit": 0.60}

# `feature_normalize=` of each branch in utils/train_baselines.py. FairGB
# passes none and gets get_dataset's default True (:639) -- the opposite of its
# own upstream rule, which skips german (FairGB-main/data_utils.py:289). A
# recorded implementation difference, not corrected here.
_NORMALIZE = {"FairGNN": lambda d: d in ("nba", "german"),   # :516
              "FairVGNN": lambda d: d == "german",           # :467
              "NIFTY": lambda d: False,                      # :592
              "GNN": lambda d: False,                        # :570
              "FairGB": lambda d: True}                      # :639


def _num(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v.strip("'\"")


def _flags(line):
    """--k=v and --k v pairs of one shell command line."""
    out, toks = {}, shlex.split(line)
    i = 0
    while i < len(toks):
        t = toks[i]
        m = re.fullmatch(r"--([A-Za-z_0-9-]+)=(.*)", t)
        if m:
            out[m.group(1).replace("-", "_")] = _num(m.group(2))
        elif t.startswith("--") and i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            out[t[2:].replace("-", "_")] = _num(toks[i + 1])
            i += 1
        i += 1
    return out


def _read(path):
    return open(path).read().splitlines() if os.path.exists(path) else []


def _fairvgnn_defaults():
    d = {}
    for ln in _read(os.path.join(PROV, "fairvgnn_main.py")):
        m = re.search(r"add_argument\(\s*['\"]--([A-Za-z_0-9]+)['\"].*?default\s*=\s*"
                      r"('[^']*'|\"[^\"]*\"|[-\w.]+)", ln)
        if m:
            d[m.group(1)] = _num(m.group(2).strip("'\""))
    return d


def _fairvgnn_lines(dataset):
    """(full-method line, official both-off line) for GCN at default prop."""
    ds = "bail" if _PKEY.get(dataset, dataset) == "recidivism" else dataset
    path = os.path.join(PROV, f"fairvgnn_run_{ds}.sh")
    full = off = None
    for ln in _read(path):
        if "encoder='GCN'" not in ln or "--prop='spmm'" in ln:
            continue
        f = _flags(ln)
        if f.get("f_mask") == "no" and f.get("weight_clip") == "no":
            off = off or f
        elif "f_mask" not in f and "weight_clip" not in f:
            full = full or f
    return full, off


def _fairgb_runsh(dataset):
    ds = "bail" if _PKEY.get(dataset, dataset) == "recidivism" else dataset
    for ln in _read(FAIRGB_RUN):
        ln = ln.strip()
        if ln.startswith("python") and f"--dataset='{ds}'" in ln:
            f = _flags(ln)
            f.pop("dataset", None)
            return f
    return {}


def _nifty_cmd(script, dataset, must=()):
    """A command line from NIFTY's README naming this script and dataset."""
    for ln in _read(os.path.join(PROV, "nifty_README.md")):
        if script not in ln or f"--dataset {dataset}" not in ln:
            continue
        f = _flags(ln.strip("` "))
        if all(f.get(k) == v for k, v in must):
            return f
    return {}


def _params(method, dataset):
    if not os.path.exists(PARAM_PATH):
        return {}, []
    p = json.load(open(PARAM_PATH))
    raw = p.get(method, {}).get(_PKEY.get(dataset, dataset), {})
    if not isinstance(raw, dict):
        return {}, []
    mapped = {"num_hidden": "hidden_dim", "num_proj_hidden": "proj_hidden_dim"}
    named, extra = {}, []
    for k, v in raw.items():
        (named.__setitem__(mapped[k], v) if k in mapped else extra.append(v))
    return named, extra


def published(method: str, dataset: str) -> dict:
    """{config, horizon, selector, provenance, source, notes, official_ablation}"""
    named, extra = _params(method, dataset)
    prov, source, notes, abl = "local-unverified", "utils/param.json", [], None
    H = None

    if method == "FairVGNN":
        ds_tag = "bail" if _PKEY.get(dataset, dataset) == "recidivism" else dataset
        full, off = _fairvgnn_lines(dataset)
        defs = _fairvgnn_defaults()
        if full is not None:
            prov = "official-repo"
            source = f"harness/provenance/fairvgnn_run_{ds_tag}.sh"
            keys = ("clip_e", "d_epochs", "g_epochs", "c_epochs", "c_lr", "e_lr",
                    "c_wd", "e_wd", "ratio", "top_k", "alpha", "hidden",
                    "dropout", "prop", "K")
            cfg = {k: full.get(k, defs.get(k)) for k in keys
                   if full.get(k, defs.get(k)) is not None}
            H = full.get("epochs", defs.get("epochs"))
            abl = off
            if off:
                diff = {k: (full.get(k, defs.get(k)), off.get(k, defs.get(k)))
                        for k in set(keys) | {"epochs"}
                        if full.get(k, defs.get(k)) != off.get(k, defs.get(k))}
                if diff:
                    notes.append("official ablation retunes " + ", ".join(
                        f"{k}:{a}->{b}" for k, (a, b) in sorted(diff.items())))
            defaulted = sorted(k for k in cfg
                               if k not in full and k in defs)
            if "epochs" not in full:
                defaulted.append("epochs")
            if defaulted:
                notes.append("official-repo-default (the authors' script runs "
                             "this dataset but leaves these at fairvgnn.py's "
                             "parser default): " + ", ".join(defaulted))
            if named or extra:
                notes.append("param.json disagrees "
                             f"(top_k/alpha {extra} vs official "
                             f"{cfg.get('top_k')}/{cfg.get('alpha')})")
        else:
            cfg = dict(hidden=16, top_k=10, alpha=1)

    elif method == "FairGB":
        sh = _fairgb_runsh(dataset)
        if sh:
            prov, source = "official-repo", "FairGB-main/run.sh"
            H = sh.pop("epochs", None)
            sh.pop("runs", None)
            cfg = dict(sh); cfg.setdefault("hidden", 16)
            notes.append("official-repo-default: hidden=16 (run.sh does not "
                         "set it)")
        else:
            cfg = dict(c_lr=0.01, e_lr=0.01, alpha=1, eta=0.5, hidden=16)

    elif method == "NIFTY":
        f = _nifty_cmd("nifty_sota_gnn.py", dataset, must=(("model", "ssf"),))
        if f:
            prov, source = "official-repo", "harness/provenance/nifty_README.md"
            H = f.get("epochs")
            cfg = dict(num_hidden=f.get("hidden", 16),
                       num_proj_hidden=f.get("hidden", 16),
                       lr=f.get("lr", 1e-3), weight_decay=_CLI["weight_decay"],
                       sim_coeff=f.get("sim_coeff", 0.6))
        else:
            cfg = dict(num_hidden=named.get("hidden_dim", 128),
                       num_proj_hidden=named.get("proj_hidden_dim", 128),
                       lr=_CLI["lr"], weight_decay=_CLI["weight_decay"])
            notes.append("NIFTY's repository states german only; this dataset "
                         "has no command in any artifact")

    elif method == "FairGNN":
        f = _nifty_cmd("baseline_fairGNN.py", dataset)
        vals = list(extra)
        cfg = dict(alpha=vals[0] if vals else 1,
                   beta=vals[1] if len(vals) > 1 else 1,
                   acc=_FAIRGNN_ACC.get(_PKEY.get(dataset, dataset), 0.6) - 0.3,
                   num_hidden=named.get("hidden_dim", _CLI["hidden_dim"]),
                   lr=_CLI["lr"], weight_decay=0.0)
        notes.append("FairGNN's own repository configures nba, pokec_n and "
                     "pokec_z only -- never german, bail or credit")
        if f:
            prov = "third-party-benchmark"
            source = "harness/provenance/nifty_README.md (NIFTY, as a baseline)"
            H = f.get("epochs")
            cfg.update(num_hidden=f.get("hidden", 16), lr=f.get("lr", 1e-3))
            notes.append("alpha/beta come from param.json and remain "
                         "local-unverified")

    elif method == "GNN":
        f = _nifty_cmd("nifty_sota_gnn.py", dataset, must=(("model", "gcn"),))
        cfg = dict(num_hidden=named.get("hidden_dim", 128),
                   num_proj_hidden=named.get("proj_hidden_dim", 128),
                   lr=_CLI["lr"], weight_decay=0.0)
        if f:
            prov = "third-party-benchmark"
            source = "harness/provenance/nifty_README.md (NIFTY GCN baseline)"
            H = f.get("epochs")
            cfg.update(num_hidden=f.get("hidden", 16),
                       num_proj_hidden=f.get("hidden", 16),
                       lr=f.get("lr", 1e-3))
    else:
        raise KeyError(method)

    cfg["feature_normalize"] = _NORMALIZE[method](_PKEY.get(dataset, dataset))
    return dict(config=cfg, horizon=H, provenance=prov, source=source,
                notes=notes, official_ablation=abl)


# ---------------------------------------------------------------------------
# Native protocol, Arm B (X13, X14)
#
# `published()` is left exactly as corrected Arm A ran it, so Arm A stays
# reproducible. `native_config()` starts from it and applies only the
# differences X13 established field by field against the authors' artifacts.
# Each value is parsed from the artifact, never typed in.
# ---------------------------------------------------------------------------
FAIRGB_DATA = os.path.join(_repo("FairGB-main"), "data_utils.py")


def _fairgb_normalized_datasets():
    """`if dataname in ['bail', 'credit']:` above feature_norm, data_utils.py:289."""
    src = "\n".join(_read(FAIRGB_DATA))
    m = re.search(r"if\s+dataname\s+in\s+\[([^\]]*)\]\s*:\s*\n\s*norm_features\s*=\s*feature_norm",
                  src)
    if not m:
        raise RuntimeError("FairGB normalization rule not found in data_utils.py")
    return {t.strip().strip("'\"") for t in m.group(1).split(",") if t.strip()}


def _fairvgnn_unnormalized_dataset():
    """`if(dataname != 'german'):` above feature_norm, FairVGNN dataset.py:346."""
    src = "\n".join(_read(os.path.join(PROV, "vg_dataset.py")))
    m = re.search(r"if\s*\(\s*dataname\s*!=\s*'([a-z_]+)'\s*\)\s*:\s*\n\s*norm_features\s*=\s*feature_norm",
                  src)
    if not m:
        raise RuntimeError("FairVGNN normalization rule not found in dataset.py")
    return m.group(1)


def _fairvgnn_credit_clip_c():
    """`--clip_c` default in the official fairvgnn_credit.py argparse (X19).

    run_credit.sh never sets it, so it is official-repo-default.
    """
    src = "\n".join(_read(os.path.join(PROV, "fairvgnn_credit.py")))
    m = re.search(r"add_argument\(\s*['\"]--clip_c['\"].*?default\s*=\s*([-\d.eE]+)", src)
    if not m:
        raise RuntimeError("--clip_c not found in fairvgnn_credit.py")
    return float(m.group(1))


def native_config(method: str, dataset: str) -> dict:
    """Official/native training configuration and horizon for one Arm B cell.

    Returns {config, horizon, provenance, source, native_changes}. Raises for a
    cell whose native protocol the harness cannot instantiate faithfully.
    """
    ds = "bail" if _PKEY.get(dataset, dataset) == "recidivism" else dataset
    base = published(method, dataset)
    cfg = dict(base["config"])
    H = base["horizon"]
    changes = []
    if H is None:
        raise ValueError(f"{method}/{dataset}: no native horizon in any artifact")
    changes.append(f"horizon {H}")

    if method == "NIFTY":
        f = _nifty_cmd("nifty_sota_gnn.py", ds, must=(("model", "ssf"),))
        if not f:
            raise ValueError(f"NIFTY/{dataset}: no official command")
        for k in ("drop_edge_rate_1", "drop_edge_rate_2",
                  "drop_feature_rate_1", "drop_feature_rate_2"):
            if k not in f:
                raise ValueError(f"NIFTY/{dataset}: official command lacks {k}")
            cfg[k] = f[k]
        # the local class zeroes its training rates in __init__ (X13); the
        # harness restores them after construction
        cfg["restore_train_drop_rates"] = True
        changes.append("training and validation-view drop rates from the "
                       "official command")

    elif method == "FairGB":
        normed = ds in _fairgb_normalized_datasets()
        if cfg.get("feature_normalize") != normed:
            changes.append(f"feature normalization {'on' if normed else 'off'} "
                           "(upstream data_utils.py rule)")
        cfg["feature_normalize"] = normed
        cfg["fairgb_get_dataset_normalize"] = normed

    elif method == "FairVGNN":
        if ds == "credit":
            # The authors train credit with fairvgnn_credit.py. Its loop runs in
            # the separate native adapter (X19), never in the shared wrapper.
            cfg["fairvgnn_credit_adapter"] = True
            cfg["clip_c"] = _fairvgnn_credit_clip_c()
            changes.append("training loop from fairvgnn_credit.py via "
                           "harness/adapters/fairvgnn_credit_native.py "
                           f"(clip_c={cfg['clip_c']}, official-repo-default)")
        wrapper_norm = ds != _fairvgnn_unnormalized_dataset()
        cfg["feature_normalize"] = False        # no harness pre-normalization
        cfg["vg_wrapper_normalize"] = wrapper_norm
        if not wrapper_norm:
            changes.append("feature normalization off (official dataset.py "
                           "rule); harness pre-norm and wrapper norm disabled")
    else:
        raise KeyError(method)

    return dict(config=cfg, horizon=int(H), provenance=base["provenance"],
                source=base["source"], native_changes=changes)
