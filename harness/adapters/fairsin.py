"""
FairSIN, wrapped — the external repository is never modified.

`FairSIN-main/in-train.py` has no importable entry point (the hyphen alone
prevents it) and its driver contains the loop the trajectory logger has to hook.
So the driver is reimplemented here, as `adapters/bind.py` does for BIND, with
every transcribed block annotated by its source line. Models, dataset code and
the metric functions are imported from the repository unchanged.

What this adapter preserves deliberately
----------------------------------------
**σ_inner stays inside the training algorithm.** The neutralisation MLP is
selected by lowest validation MSE (in-train.py:159-168) and its choice changes
what the classifier is then trained on. That is part of the mechanism, not part
of how the method is reported, so it is not exposed to the external selection
analysis and is not unified away. When the neutralisation component is switched
off for M₀, σ_inner disappears with it — correctly, since it is downstream
machinery of the intervention rather than an operator that must be held fixed.

**M₁ is the published configuration per (dataset, encoder).** `experiment.sh`
runs `--d='yes'` on German with GCN and GIN, omits it on Bail, and uses a
different script on Credit. The parser default for `--d` is `'no'`, so defaults
are not the published method. `PUBLISHED` below records what the official script
actually ran; nothing infers it from the parser.

One discrepancy recorded, not repaired
--------------------------------------
Training uses `encoder(x + delta * model(x), ...)` (in-train.py:179) while
`evaluation.evaluate` uses `encoder(x, ...)` — the neutralisation term is absent
at evaluation. Both M₀ and M₁ are evaluated the same way, so the intervention
contrast is still well defined; what it changes is the description of what the
deployed model is. Left exactly as published.
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "harness"))
from core.paths import repo as _repo  # noqa: E402  (repositories live under models/)

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAIRSIN = _repo("FairSIN-main")

# Published configurations, transcribed from FairSIN-main/experiment.sh.
# `d` is the discriminator flag; the parser default is 'no' and is NOT used.
PUBLISHED = {
    ("german", "GCN"): dict(delta=0.5, d="yes", epochs=150, hidden=18,
                            c_lr=0.1, e_lr=0.1, c_wd=0.001, e_wd=0.001,
                            c_epochs=10, script="in-train.py"),
    ("german", "GIN"): dict(delta=1.0, d="yes", epochs=80, hidden=54,
                            c_lr=0.01, e_lr=0.01, c_wd=0.0, e_wd=0.0,
                            c_epochs=10, script="in-train.py"),
    ("bail", "GCN"):   dict(delta=1.0, d="no", epochs=100, hidden=18,
                            c_lr=0.1, e_lr=0.01, c_epochs=10,
                            script="in-train.py"),
    ("credit", "GCN"): dict(delta=0.25, d="no", epochs=100, hidden=13,
                            c_epochs=10, script="train_mlp.py"),
}
# Datasets FairSIN's own scripts cover. Nothing outside this is a published
# configuration, and the adapter refuses to invent one.
SUPPORTED = sorted({d for d, _ in PUBLISHED})


_REPO_CACHE: dict = {}


def _import_repo():
    """Import FairSIN's modules without copying, editing, or shadowing ours.

    FairSIN-main ships `utils.py`, and this repository has a `utils/` package.
    Putting FairSIN's directory on `sys.path` ahead of ours makes
    `from utils import ...` resolve to the wrong one for the rest of the
    process — which is not a hypothetical: it broke the first adapter run.

    So the path and the `utils` binding are borrowed for the duration of the
    import and then put back, and the result is cached so the swap happens
    once.
    """
    if _REPO_CACHE:
        return _REPO_CACHE["model"], _REPO_CACHE["evaluation"]

    import importlib
    saved_path = list(sys.path)
    saved_utils = sys.modules.get("utils")
    try:
        sys.path.insert(0, FAIRSIN)
        sys.modules.pop("utils", None)
        fs_utils = importlib.import_module("utils")       # FairSIN's utils.py
        sys.modules["utils"] = fs_utils
        fs_model = importlib.import_module("model")
        fs_eval = importlib.import_module("evaluation")
    finally:
        sys.path[:] = saved_path
        if saved_utils is not None:
            sys.modules["utils"] = saved_utils
        else:
            sys.modules.pop("utils", None)
    _REPO_CACHE.update(model=fs_model, evaluation=fs_eval, utils=fs_utils)
    return fs_model, fs_eval


class MLP(torch.nn.Module):
    """The neutralisation network, transcribed from in-train.py:16-32.

    It lives in `in-train.py` rather than `model.py`, and that file cannot be
    imported by name (the hyphen) without executing its module body. Three
    layers are copied here verbatim instead, which is the smaller and more
    inspectable of the two options.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()


class _Args:
    """The attribute bag FairSIN's models read. Mirrors its argparse namespace."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def published_config(dataset: str, encoder: str = "GCN") -> dict:
    key = (dataset, encoder)
    if key not in PUBLISHED:
        raise KeyError(
            f"no published FairSIN configuration for {key}. experiment.sh covers "
            f"{sorted(PUBLISHED)}; inventing one would mean auditing a method "
            "nobody reported.")
    return dict(PUBLISHED[key])


def hetero_neighbour_features(data, device, cache_dir: str | None = None):
    """h_X: mean feature of each node's opposite-group neighbours.

    Transcribed from in-train.py:84-110. The repository caches this to
    `<dataset>_hadj.pt` in its working directory; the cache path is a parameter
    here so the adapter never writes into FairSIN-main.
    """
    import scipy.sparse as sp
    if getattr(data, "adj", None) is None:
        # This study's loader returns edge_index and adj_norm_sp but not the
        # raw scipy adjacency FairSIN's neighbour scan reads. Building it here
        # keeps the audit on one set of splits instead of switching to
        # FairSIN's own loader, which would change the data as well as the
        # method.
        from torch_geometric.utils import to_scipy_sparse_matrix
        n = data.x.shape[0]
        a = to_scipy_sparse_matrix(data.edge_index.cpu(), num_nodes=n).tocsr()
        data.adj = ((a + a.T) > 0).astype(float)
    adj = data.adj - sp.eye(data.adj.shape[0])
    n = adj.shape[0]
    new_adj = torch.zeros((n, n), dtype=torch.int32)
    sens = data.sens
    for i in range(n):                                   # in-train.py:86-92
        nz = torch.tensor(adj[i].nonzero())
        if nz.numel() == 0:
            continue
        cols = nz[1]
        mask = (sens[cols] != sens[i])
        new_adj[i, cols[mask]] = 1

    deg = torch.from_numpy(np.sum(new_adj.numpy(), axis=1)).cpu()
    idx = torch.nonzero(new_adj)
    vals = new_adj[idx[:, 0], idx[:, 1]]
    mat = torch.sparse_coo_tensor(idx.t(), vals, new_adj.shape).float().cpu()
    h_X = torch.spmm(mat, data.x.cpu()) / deg.unsqueeze(-1)   # in-train.py:105
    keep = ~torch.any(torch.isnan(h_X), dim=1)                # :106-109
    return h_X[keep].to(device), data.x.cpu()[keep].to(device), keep


def run(data, *, dataset: str, encoder_name: str = "GCN", device: str = "cpu",
        seed: int = 0, trajectory=None, components=("N", "D"),
        alpha_select: float = 1.0, m_epoch: int = 20, d_epochs: int = 5,
        overrides: dict | None = None):
    """One FairSIN run, transcribed from in-train.py:34-247.

    `components` names which of the method's claimed parts are active:

        "N"  neutralisation — `x + delta * model(x)`. Removing it sets delta = 0,
             which deletes the term exactly (in-train.py:179).
        "D"  discriminator — the adversarial step. Removing it takes the
             `args.d == 'yes'` branch (in-train.py:189) out.

    M₀ is `components=()`. M₁ is the published configuration for this
    (dataset, encoder), which for Bail already has D off — so M₁ there is
    `("N",)`, not `("N", "D")`. Enabling D where the published run did not would
    audit a model nobody reported.
    """
    fs_model, fs_eval = _import_repo()
    cfg = published_config(dataset, encoder_name)
    cfg.update(overrides or {})
    if cfg["script"] != "in-train.py":
        raise NotImplementedError(
            f"{dataset} is published through {cfg['script']}, a different entry "
            "point; that path is a separate transcription and is not this one.")

    dev = torch.device(device)
    data = data.to(dev)
    use_N = "N" in components
    use_D = "D" in components and cfg["d"] == "yes"

    args = _Args(device=dev, hidden=cfg["hidden"], dropout=0.5,
                 num_features=data.x.shape[1], num_classes=1,
                 sens_idx=int(getattr(data, "sens_idx", 0)),
                 encoder=encoder_name, prop="scatter",
                 delta=(cfg["delta"] if use_N else 0.0),
                 alpha=alpha_select, d="yes" if use_D else "no",
                 epochs=cfg["epochs"], c_epochs=cfg.get("c_epochs", 10),
                 d_epochs=d_epochs, m_epoch=m_epoch,
                 c_lr=cfg.get("c_lr", 0.01), e_lr=cfg.get("e_lr", 0.01),
                 d_lr=cfg.get("d_lr", 0.01), m_lr=cfg.get("m_lr", 0.01),
                 c_wd=cfg.get("c_wd", 0.0), e_wd=cfg.get("e_wd", 0.0),
                 d_wd=cfg.get("d_wd", 0.0), K=10)

    torch.manual_seed(seed); np.random.seed(seed)

    discriminator = fs_model.MLP_discriminator(args).to(dev)        # :43
    optimizer_d = torch.optim.Adam(
        [dict(params=discriminator.lin.parameters(), weight_decay=args.d_wd)],
        lr=args.d_lr)
    classifier = fs_model.MLP_classifier(args).to(dev)              # :47
    optimizer_c = torch.optim.Adam(
        [dict(params=classifier.lin.parameters(), weight_decay=args.c_wd)],
        lr=args.c_lr)
    if encoder_name == "GCN":                                       # :55-62
        enc = fs_model.GCN_encoder_scatter(args).to(dev)
        optimizer_e = torch.optim.Adam(
            [dict(params=enc.lin.parameters(), weight_decay=args.e_wd),
             dict(params=enc.bias, weight_decay=args.e_wd)], lr=args.e_lr)
    elif encoder_name == "GIN":                                     # :63-66
        enc = fs_model.GIN_encoder(args).to(dev)
        optimizer_e = torch.optim.Adam(
            [dict(params=enc.conv.parameters(), weight_decay=args.e_wd)],
            lr=args.e_lr)
    else:
        raise NotImplementedError(f"encoder {encoder_name!r}")

    h_X, c_X, _ = hetero_neighbour_features(data, dev)
    mlp = MLP(data.x.shape[1], args.hidden, data.x.shape[1]).to(dev)
    optimizer_m = torch.optim.Adam(
        [dict(params=mlp.parameters(), weight_decay=0.001)], lr=args.m_lr)

    from sklearn.model_selection import train_test_split          # :121-125
    idx = np.arange(c_X.shape[0])
    i_tr, i_te, _, _ = train_test_split(idx, idx, test_size=0.1)
    X_train, X_test = c_X[i_tr], c_X[i_te]
    y_train, y_test = h_X[i_tr], h_X[i_te]

    criterion = torch.nn.BCELoss()                                  # :36
    best_val_tradeoff = 0.0          # the floor; see select_composite
    best_val_loss = float("inf")
    best_epoch = -1
    best_mlp_state = None

    for epoch in range(0, args.epochs):                             # :138
        if use_N:
            for _ in range(0, args.m_epoch):                        # :140-167
                mlp.train(); optimizer_m.zero_grad()
                loss_m = F.mse_loss(mlp(X_train), y_train)
                loss_m.backward(); optimizer_m.step()
                mlp.eval()
                with torch.no_grad():
                    v = F.mse_loss(mlp(X_test), y_test)
                # sigma_inner: inside the training algorithm, not reported
                if v < best_val_loss:
                    best_val_loss = v
                    best_mlp_state = {k: t.detach().clone()
                                      for k, t in mlp.state_dict().items()}
            if best_mlp_state is not None:
                mlp.load_state_dict(best_mlp_state)                 # :168
            mlp.eval()

        def _feat():
            """x + delta * mlp(x); delta is 0 when N is off, so the term drops."""
            return data.x + args.delta * mlp(data.x) if use_N else data.x

        classifier.train(); enc.train()                             # :171-186
        for _ in range(0, args.c_epochs):
            optimizer_c.zero_grad(); optimizer_e.zero_grad()
            out = classifier(enc(_feat(), data.edge_index, data.adj_norm_sp))
            loss_c = F.binary_cross_entropy_with_logits(
                out[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(dev))
            loss_c.backward(); optimizer_e.step(); optimizer_c.step()

        if use_D:                                                   # :189-206
            discriminator.train(); enc.train()
            for _ in range(0, args.d_epochs):
                optimizer_d.zero_grad(); optimizer_e.zero_grad()
                optimizer_m.zero_grad()
                out = discriminator(enc(_feat(), data.edge_index, data.adj_norm_sp))
                loss_d = criterion(out.view(-1), data.x[:, args.sens_idx])
                loss_d.backward()
                optimizer_d.step(); optimizer_e.step(); optimizer_m.step()

        accs, auc_rocs, F1s, par, eq = fs_eval.evaluate(              # :208
            data.x, classifier, discriminator, enc, data, args)

        if trajectory is not None:
            # `evaluate` is deterministic -- torch.no_grad, eval mode, no
            # stochastic op -- so recomputing its logits here consumes no RNG
            # and cannot move training. It also uses data.x without the
            # neutralisation term, exactly as the repository does.
            classifier.eval(); enc.eval()
            with torch.no_grad():
                logit = classifier(enc(data.x, data.edge_index, data.adj_norm_sp))
            trajectory.add(epoch,
                           logit[data.val_mask].squeeze().detach().cpu().numpy(),
                           stage="classifier",
                           extra={"code_auc": auc_rocs["val"],
                                  "code_f1": F1s["val"],
                                  "code_acc": accs["val"],
                                  "code_dp": par["val"],
                                  "code_eo": eq["val"],
                                  "alpha_select": float(args.alpha)})

        score = (auc_rocs["val"] + F1s["val"] + accs["val"]          # :227
                 - args.alpha * (par["val"] + eq["val"]))
        if score > best_val_tradeoff:
            best_val_tradeoff = score
            best_epoch = epoch

    return {"best_epoch": best_epoch, "components": tuple(components),
            "delta": args.delta, "d": args.d, "epochs": args.epochs,
            "published": cfg}
