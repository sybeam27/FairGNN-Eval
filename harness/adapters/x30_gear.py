"""X30 GEAR adapter: official `GEAR-main/src` driver functions, unmodified.

`main.py` keeps its whole driver under `__main__` and its functions read module
globals (`args`, `n`, `device`), so the module is loaded with
`argv = ['main.py', '--dataset', <d>]` (parser defaults otherwise), the handful
of setup statements of main.py:500-596 (train branch) are reproduced here with
their line references, and the official `train()` is called one step at a time:
`train(0, ...)` runs exactly one iteration of the loop body (main.py:282-352),
python's `random` stream continuing across calls exactly as in one loop.

Intervention I: the counterfactual-consistency (similarity) objective,
`sim_coeff` (main.py:53, default 0.6).

    M+I   sim_coeff = 0.6     M-I   sim_coeff = 0.0     (nothing else changes)

Recorded properties of this switch (not repaired): at sim_coeff = 0 the
classification loss is `1.0 * l3` instead of `0.4 * l3` (main.py:320), and
`optimizer_1` still steps on a zero similarity gradient, so its Adam update is
driven by weight decay alone and moves the shared encoder by roughly `lr` per
step (main.py:311-312, 496-499). Both are what the repository's switch does.

Assets: the released `graphFair_subgraph/aug/bail_cf_aug_{0,1,2}.pkl`, which
main.py loads by default (`subgraph_load = True`, :541, :561). Verified before
use and re-verified every arm (hard stop otherwise): identical edge set to the
common bail graph, non-sensitive columns identical to GEAR-normalised features,
sensitive column = 1 - s. All three files are identical, although the code
labels aug_1 / aug_2 as sens_rate 0.0 / 1.0; the counterfactual actually used is
a sensitive-attribute flip on the unchanged graph. The CFDA/CFGT generator that
would build other assets has no released checkpoint and is not run.
credit is not admitted: its released assets are built on a different graph
(3% of the common credit edges).

Wrappers: `evaluate` inside `train` replaced by a stub (it only prints and
feeds the native checkpoint rule every 100 steps; native selection is not part
of the controlled cell); per-step full-node logits through the official
`get_all_node_emb` + `model.predict`; a private cwd holding `models_save/` and
the PPR/subgraph cache; generator state saved/restored around the module
import, which seeds numpy/torch at module level.
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "harness"))
from core.paths import repo as _repo  # noqa: E402  (repositories live under models/)

import contextlib
import hashlib
import importlib.util
import io
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GS = os.path.join(_repo("GEAR-main"), "src")
DATASETS = ("bail",)
CACHE = os.path.join(ROOT, "harness", "results", "x30", "cache", "gear")
SIM_PLUS, SIM_MINUS = 0.6, 0.0
_MODS: dict = {}
_SUB: dict = {}


def _load(dataset):
    if dataset in _MODS:
        return _MODS[dataset]
    import random as _r
    rng = (_r.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    tf32 = torch.backends.cudnn.allow_tf32
    names = ("utils", "models", "Preprocessing", "utils_mp", "CFDA", "CFGT", "subgcon", "subgraphs")
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k in names or k.startswith("utils.")}
    saved_path, saved_argv = list(sys.path), list(sys.argv)
    try:
        sys.path.insert(0, GS)
        sys.argv = ["main.py", "--dataset", dataset]
        spec = importlib.util.spec_from_file_location(f"_x30_gear_main_{dataset}",
                                                      os.path.join(GS, "main.py"))
        mod = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved_path
        sys.argv = saved_argv
        # only `utils` collides with this repository; GEAR's other modules stay
        # registered, since multiprocessing pickles utils_mp.PPR by module name
        gear_utils = sys.modules.pop("utils", None)
        mod.__dict__.setdefault("_gear_utils", gear_utils)
        for k, v in saved.items():
            sys.modules[k] = v
        _r.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        torch.backends.cudnn.allow_tf32 = tf32
    _MODS[dataset] = mod
    return mod


class _Holder:
    def __setstate__(self, st):
        self.__dict__.update(st)


class _PyG1(pickle.Unpickler):
    """The released pickles hold torch_geometric 1.x Data objects; their raw
    fields are read into a plain holder (I/O adaptation only)."""

    def find_class(self, module, name):
        if module.startswith("torch_geometric"):
            return _Holder
        return super().find_class(module, name)


def gear_features(x, sens_idx):
    """GEAR-main/src/Preprocessing.py:158-160 with utils.feature_norm (:319)."""
    mn, mx = x.min(axis=0)[0], x.max(axis=0)[0]
    xn = 2 * (x - mn).div(mx - mn) - 1
    xn[:, sens_idx] = x[:, sens_idx]
    return xn


def _assets(dataset, x, sens_idx, edge_index):
    from torch_geometric.data import Data
    E0 = {(int(a), int(b)) for a, b in edge_index.t().tolist()}
    out = []
    for k in (0, 1, 2):
        p = os.path.join(GS, "graphFair_subgraph", "aug", f"{dataset}_cf_aug_{k}.pkl")
        with open(p, "rb") as f:
            raw = _PyG1(f).load()["data_cf"].__dict__
        xc = torch.as_tensor(raw["x"]).float()
        ei = torch.as_tensor(raw["edge_index"]).long()
        other = [c for c in range(x.shape[1]) if c != sens_idx]
        ok = (xc.shape == x.shape
              and torch.equal(xc[:, other], x[:, other].float())
              and torch.equal(xc[:, sens_idx], 1 - x[:, sens_idx].float())
              and {(int(a), int(b)) for a, b in ei.t().tolist()} == E0)
        if not ok:
            raise SystemExit(f"[gear] released asset {p} is not the verified sensitive-attribute "
                             "flip of the common data (hard stop)")
        out.append(Data(x=xc, edge_index=ei))
    return out


@contextlib.contextmanager
def _torch_load_full():
    """torch>=2.6 defaults torch.load to weights_only=True; GEAR's own cache
    files (numpy arrays, PyG Data) predate it. Scoped to GEAR's cache I/O."""
    orig = torch.load

    def load(*a, **k):
        k.setdefault("weights_only", False)
        return orig(*a, **k)
    torch.load = load
    try:
        yield
    finally:
        torch.load = orig


def _subgraphs(mod, dataset, data, cf_list):
    """main.py:533-583: Subgraph(...).build() for the factual graph and each
    counterfactual, cached under a private path (graph- and asset-dependent
    only, so shared by every split and arm)."""
    if dataset in _SUB:
        return _SUB[dataset]
    os.makedirs(CACHE, exist_ok=True)
    ppr_path = os.path.join(CACHE, dataset)
    a = mod.args
    with contextlib.redirect_stdout(io.StringIO()), _torch_load_full():
        sg = mod.Subgraph(data.x, data.edge_index, ppr_path, a.subgraph_size, a.n_order)
        sg.build()
        cfs = []
        for i, dcf in enumerate(cf_list):
            c = mod.Subgraph(dcf.x, dcf.edge_index, ppr_path, a.subgraph_size, a.n_order)
            c.build(postfix="_cf" + str(i))
            cfs.append(c)
    _SUB[dataset] = (sg, cfs)
    return sg, cfs


def train_arm(dataset, encoder, arm, split, seed, epochs, device="cuda"):
    from torch_geometric.data import Data
    from utils.dataloading import load_data
    mod = _load(dataset)
    adj, feats, labels, itr, iva, ite, sens, sens_idx = load_data(
        dataset, feature_normalize=False, split_seed=split)
    x = gear_features(feats, sens_idx)
    coo = adj.coalesce()
    edge_index = coo.indices().long()                         # self-loops included, as GEAR's adj
    n = x.shape[0]
    data = Data(x=x, edge_index=edge_index)                   # main.py:517-518
    data.y = labels
    cf_list = _assets(dataset, x, sens_idx, edge_index)
    subgraph, cf_subgraph_list = _subgraphs(mod, dataset, data, cf_list)

    a = mod.args
    sim = SIM_PLUS if arm == "plus" else SIM_MINUS
    idx_train, _ = torch.sort(itr); idx_val, _ = torch.sort(iva); idx_test, _ = torch.sort(ite)
    num_class = labels.unique().shape[0] - 1                  # main.py:512
    dev = torch.device(device)

    scores, st = [], dict(init_hash=None, sim_nonzero=0)
    old = dict(sim=a.sim_coeff, n=getattr(mod, "n", None), dev=mod.device, evaluate=mod.evaluate,
               sens=getattr(mod, "sens", None))

    def stub_evaluate(*_a, **_k):
        z = 0.0
        return dict(acc=z, auc=z, f1=z, parity=z, equality=z, cf=z, loss=z, loss_c=z, loss_s=z)

    orig_D = mod.models.GraphCF.D

    def D(self, x1, x2):
        v = orig_D(self, x1, x2)
        if a.sim_coeff != 0:
            st["sim_nonzero"] += 1
        return v

    run_dir = os.path.join(CACHE, "cwd", str(os.getpid()))
    os.makedirs(os.path.join(run_dir, "models_save"), exist_ok=True)
    cwd = os.getcwd()
    tf32 = torch.backends.cudnn.allow_tf32
    try:
        a.sim_coeff = sim
        mod.n, mod.device, mod.sens = n, dev, sens      # __main__ globals train() reads
        mod.evaluate = stub_evaluate
        mod.models.GraphCF.D = D
        torch.backends.cudnn.allow_tf32 = False               # main.py:78
        os.chdir(run_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            model = mod.models.GraphCF(                         # main.py:586-591
                encoder=mod.models.Encoder(data.num_features, a.hidden_size, base_model=a.encoder),
                args=a, num_class=num_class).to(dev)
            par_1 = (list(model.encoder.parameters()) + list(model.fc1.parameters())
                     + list(model.fc2.parameters()) + list(model.fc3.parameters())
                     + list(model.fc4.parameters()))
            par_2 = list(model.c1.parameters()) + list(model.encoder.parameters())
            opt1 = torch.optim.Adam(par_1, lr=a.lr, weight_decay=a.weight_decay)
            opt2 = torch.optim.Adam(par_2, lr=a.lr, weight_decay=a.weight_decay)
            h = hashlib.sha1()
            for k_, v in model.state_dict().items():
                h.update(k_.encode()); h.update(v.detach().cpu().numpy().tobytes())
            st["init_hash"] = h.hexdigest()[:16]
            all_mask = torch.ones(n, dtype=torch.bool)
            for step in range(epochs):
                mod.train(0, model, opt1, opt2, data, subgraph, cf_subgraph_list,
                          idx_train, idx_val, idx_test, step)
                model.eval()
                with torch.no_grad():
                    emb = mod.get_all_node_emb(model, all_mask.numpy(), subgraph, n)
                    out = model.predict(emb)
                scores.append(out.squeeze().detach().float().cpu().numpy())
    finally:
        os.chdir(cwd)
        a.sim_coeff = old["sim"]
        mod.n, mod.device, mod.evaluate = old["n"], old["dev"], old["evaluate"]
        mod.sens = old["sens"]
        mod.models.GraphCF.D = orig_D
        torch.backends.cudnn.allow_tf32 = tf32

    tag = "M+I" if arm == "plus" else "M-I"
    checks = ({f"{tag} similarity objective active (sim_coeff={sim})": (st["sim_nonzero"] > 0, f"(D calls with sim_coeff>0: {st['sim_nonzero']})")}
              if arm == "plus" else
              {f"{tag} similarity objective switched off (sim_coeff=0)": (st["sim_nonzero"] == 0, "")})

    def pair_checks(other):
        return {"paired arms share initial parameters":
                (st["init_hash"] == other["gate"]["init_hash"], "")}

    return dict(scores=scores, code_epoch=-1, gate=st, gate_checks=checks, pair_checks=pair_checks,
                provenance="official-repo (parser defaults; released aug assets, verified sens flips)",
                masks=tuple(np.isin(np.arange(n), i.cpu().numpy()) for i in (itr, iva, ite)),
                config=dict(sim_coeff=sim, encoder=a.encoder, hidden=a.hidden_size, lr=a.lr,
                            weight_decay=a.weight_decay, batch_size=a.batch_size,
                            subgraph_size=a.subgraph_size, n_order=a.n_order,
                            proj_hidden=a.proj_hidden, native_epochs=a.epochs))
