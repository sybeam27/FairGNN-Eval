"""
One training loop, shared by every arm of the experiment.

This is the point of the rebuild: "uniform", "w_bdry", "bind_influence" and the
rest differ in exactly one line -- which signal is passed to
signals.get(...) -- and in nothing else. Any other difference between arms
would confound the comparison the study is built on.

Phases follow the draft:
  1. warm-up: only the task loss (plus the interval loss if the sigma signal is
     in use). phi comes from structural signals alone, because a model-dependent
     signal is a function of an untrained predictor and the loss-scale
     calibration would be fitted against a task loss that is still falling fast.
  2. fair training: all objectives active, each scaled against the task loss,
     phi refreshed periodically for model-dependent signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from . import allocate as A
from . import metrics as M
from . import objectives as O
from . import signals as S
from .model import GCN, normalized_adj, perturb_cross_group, to_tensors


@dataclass
class Config:
    signal: str = "uniform"

    # allocation
    q_gate: float = 0.7
    phi_min: float = 0.5
    phi_max: float = 2.0
    use_sigma_modulation: bool = False

    # objectives
    lambda_fair: float = 0.2
    lambda_unc: float = 1.0
    p_drop: float = 0.5
    edge_mode: str = "scale"          # 'scale' | 'drop'
    mmd_alpha: float = 0.3
    rho0: float = 0.3
    use_str: bool = True
    use_rep: bool = True
    use_pred: bool = True

    # optimisation
    epochs: int = 500
    warmup: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden: int = 128
    dropout: float = 0.5
    patience: int = 100
    device: str = "cpu"               # 'cpu' | 'cuda'
    # H5 (pre-registered): hold the fairness term back while validation AUC is
    # more than `auc_tol` below a reference run's. `auc_ref` is None for every
    # experiment before E9, so this branch is inert and E2-E5 are unchanged.
    #
    # The pull-back is on/off rather than a smooth scaling on purpose. Any
    # smooth version needs a slope, and a slope is exactly the kind of free
    # parameter the nine settings would end up choosing -- which is what every
    # other guard in PREREGISTRATION.md exists to prevent. `auc_tol` itself is
    # inherited, not tuned: 0.005 is `adaptive_auc_tol`'s value in FairGate's
    # own published configuration (utils/model_fairgate.py:1154).
    auc_ref: float | None = None
    auc_tol: float = 0.005

    refresh_every: int = 20           # recompute phi (model-dependent signals)
    recal_every: int = 20             # recompute the loss-scale factors
    seed: int = 27


@dataclass
class Result:
    setting: str
    signal: str
    seed: int
    test: dict = field(default_factory=dict)
    val: dict = field(default_factory=dict)
    phi: dict = field(default_factory=dict)
    best_epoch: int = -1
    epochs_run: int = 0
    n_held: int = 0                   # epochs the H5 constraint suppressed
    config: dict = field(default_factory=dict)

    def row(self) -> dict:
        r = {"setting": self.setting, "signal": self.signal, "seed": self.seed,
             "best_epoch": self.best_epoch, "epochs_run": self.epochs_run}
        r |= {f"test_{k}": v for k, v in self.test.items()}
        r |= {f"val_{k}": v for k, v in self.val.items()}
        r |= {f"phi_{k}": v for k, v in self.phi.items()}
        return r


def _phi_for(graph, cfg: Config, state, idx_fair_np) -> np.ndarray:
    """signal -> phi, restricted and mean-normalised on the fairness node set."""
    spec = S.get(cfg.signal)
    if spec.needs_model and state is None:
        score = S.get("w_bdry")(graph)          # warm-up fallback
    else:
        score = spec(graph, state)

    mod = None
    if cfg.use_sigma_modulation and state is not None and state.sigma is not None:
        mod = S.minmax(state.sigma)[idx_fair_np]

    phi_sub = A.allocate(score[idx_fair_np], q_gate=cfg.q_gate,
                         phi_min=cfg.phi_min, phi_max=cfg.phi_max,
                         modulation=mod, normalize_mean=True)
    return phi_sub


def train(graph, cfg: Config, verbose: bool = False) -> Result:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dev = torch.device(cfg.device)
    T = to_tensors(graph, device=str(dev))
    n = graph.n_nodes
    idx_fair_np = np.asarray(graph.idx_fair)
    need_sigma = (cfg.signal == "sigma") or cfg.use_sigma_modulation

    model = GCN(graph.x.shape[1], cfg.hidden, cfg.dropout,
                uncertainty_head=need_sigma).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    phi = torch.tensor(_phi_for(graph, cfg, None, idx_fair_np),
                       dtype=torch.float32, device=dev)
    gamma = {"str": 1.0, "rep": 1.0, "pred": 1.0}

    best_score, best_state, best_epoch, bad = -np.inf, None, -1, 0
    last_val_auc, n_held = None, 0
    gen = torch.Generator().manual_seed(cfg.seed)

    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()

        logit, h, sigma = model(T["x"], T["adj"])
        prob = torch.sigmoid(logit)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit[T["idx_train"]], T["y"][T["idx_train"]])
        task_val = float(loss.detach())

        if need_sigma:
            loss = loss + cfg.lambda_unc * O.interval_loss(
                logit, sigma, T["y"], T["idx_train"])

        in_fair_phase = epoch >= cfg.warmup
        # The reference is compared against the *previous* epoch's validation
        # AUC: this epoch's is not computed until after the step, and peeking at
        # it would let the constraint use information the run does not have.
        held_back = (in_fair_phase and cfg.auc_ref is not None
                     and last_val_auc is not None
                     and last_val_auc < cfg.auc_ref - cfg.auc_tol)
        if held_back:
            n_held += 1
        if in_fair_phase and not held_back:
            if epoch % cfg.refresh_every == 0 and S.get(cfg.signal).needs_model:
                st = S.ModelState(
                    probs=prob.detach().cpu().numpy(),
                    sigma=(sigma.detach().cpu().numpy()
                           if sigma is not None else None),
                    epoch=epoch)
                phi = torch.tensor(_phi_for(graph, cfg, st, idx_fair_np),
                                   dtype=torch.float32, device=dev)

            parts = {}
            if cfg.use_str:
                w = perturb_cross_group(T["edge_index"], T["sens"],
                                        cfg.p_drop, cfg.edge_mode, generator=gen)
                adj_p = normalized_adj(T["edge_index"], n, edge_weight=w)
                _, h_p, _ = model(T["x"], adj_p)
                parts["str"] = O.structural_consistency(h, h_p, T["idx_fair"], phi)
            if cfg.use_rep:
                parts["rep"] = O.representation_alignment(
                    h, T["sens"], T["idx_fair"], phi, cfg.mmd_alpha)
            if cfg.use_pred:
                parts["pred"], _, _ = O.prediction_alignment(
                    prob, T["y"], T["sens"], T["idx_fair"], phi, cfg.rho0)

            if epoch % cfg.recal_every == 0:
                for k, v in parts.items():
                    gamma[k] = min(task_val / (float(v.detach()) + 1e-8), 100.0)

            for k, v in parts.items():
                loss = loss + cfg.lambda_fair * gamma[k] * v

        loss.backward()
        opt.step()

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            lg, _, _ = model(T["x"], T["adj"])
            p = torch.sigmoid(lg).cpu().numpy()
        vm = M.evaluate(p, graph.y, graph.sens, graph.idx_val)
        last_val_auc = vm.get("auc", vm.get("roc_auc"))
        score = M.selection_score(vm, cfg.rho0)

        if in_fair_phase:
            if score > best_score:
                best_score, best_epoch, bad = score, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break

        if verbose and epoch % 50 == 0:
            print(f"    ep{epoch:4d} loss={float(loss):.4f} "
                  f"val_acc={vm['acc']:.3f} val_dp={vm['dp']:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        lg, _, _ = model(T["x"], T["adj"])
        p = torch.sigmoid(lg).cpu().numpy()

    return Result(
        setting=graph.name, signal=cfg.signal, seed=cfg.seed,
        test=M.evaluate(p, graph.y, graph.sens, graph.idx_test),
        val=M.evaluate(p, graph.y, graph.sens, graph.idx_val),
        phi=A.summarize(phi.cpu().numpy()),
        best_epoch=best_epoch, epochs_run=epoch + 1,
        n_held=n_held,
        config=asdict(cfg),
    )


def warmup_state(graph, cfg: Config) -> S.ModelState:
    """Run the warm-up phase only and return the model state the signals see.

    `harness/PREREGISTRATION.md` fixes the decision point at "after warm-up,
    before fairness training", because three of the eight signals are functions
    of a trained model and cannot be read off the raw graph. That boundary is
    not invented for this measurement -- the method already runs `cfg.warmup`
    epochs at `lambda_fair = 0` regardless, so nothing extra is paid to reach it.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dev = torch.device(cfg.device)
    T = to_tensors(graph, device=str(dev))
    need_sigma = (cfg.signal == "sigma") or cfg.use_sigma_modulation

    model = GCN(graph.x.shape[1], cfg.hidden, cfg.dropout,
                uncertainty_head=need_sigma).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    for _ in range(cfg.warmup):
        model.train()
        opt.zero_grad()
        logit, _, sigma = model(T["x"], T["adj"])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit[T["idx_train"]], T["y"][T["idx_train"]])
        if need_sigma:
            loss = loss + cfg.lambda_unc * O.interval_loss(
                logit, sigma, T["y"], T["idx_train"])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logit, _, sigma = model(T["x"], T["adj"])
        prob = torch.sigmoid(logit)
    return S.ModelState(
        probs=prob.cpu().numpy(),
        sigma=sigma.cpu().numpy() if sigma is not None else None,
        epoch=cfg.warmup)
