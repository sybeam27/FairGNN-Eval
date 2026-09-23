"""
Phase 1D — validation-trajectory instrumentation and post-hoc selection.

Training happens once, producing one trajectory θ₁ … θ_H. Several checkpoint
selection policies are then applied to **that same trajectory**, so a difference
between them is attributable to the policy and not to a different run.

    Training → validation trajectory → selector(s) → selected checkpoint(s) → test

Two structural guarantees, enforced by types rather than by discipline:

  * A selector receives a `ValidationHistory` and nothing else. The object has
    no test field, so a selector cannot read test outcomes — not "we did not",
    but "the API cannot".
  * Test evaluation takes a checkpoint, never a history, so it cannot feed back
    into selection.

Tie-breaking is fixed in advance and matches the audited code. Every original
selector uses a strict comparison (`>` or `<`, never `>=`), so on a tie the
**earliest** epoch is kept. Getting this wrong is the most likely reason a
replay test fails for a reason that is not a bug.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .evaluator import evaluate

# Stages a record can come from. Only `classifier` is visible to the final
# selector: FairSIN and FairVGNN run nested loops for a neutraliser and a
# discriminator, and mislabelling which loop an epoch belongs to is an easy bug
# that would silently change every replay.
STAGES = ("classifier", "neutralizer", "discriminator")


@dataclass(frozen=True)
class SplitRef:
    """Node ids, labels and sensitive attributes for one split. Stored once.

    Repeating these per epoch would multiply the trajectory by H for data that
    never changes.
    """
    node_id: np.ndarray
    y: np.ndarray
    a: np.ndarray

    def __post_init__(self):
        n = len(self.node_id)
        if not (len(self.y) == len(self.a) == n):
            raise ValueError("node_id, y and a must have equal length")


def inference_state(_rng: bool = False, **modules) -> dict:
    """Snapshot every module that takes part in prediction.

    `state_dict()` carries buffers as well as parameters, which matters here:
    FairGB's encoder and classifier contain BatchNorm1d, whose running_mean and
    running_var change during training and are read at inference. Saving only
    the parameters would restore a model that predicts differently from the
    epoch it claims to be.

    Which modules belong in the prediction path is known to the method and not
    to this file, so the method supplies them. The restore-fidelity test is what
    checks it got the list right.
    """
    st = {name: {k: v.detach().clone() for k, v in m.state_dict().items()}
          for name, m in modules.items()}
    if _rng:
        # FairVGNN's evaluation calls F.gumbel_softmax(..., hard=True), so its
        # inference is stochastic and a restored model does not reproduce an
        # epoch's scores unless the generator state is restored with it. Where
        # inference is deterministic this is omitted rather than carried.
        st["_rng"] = torch.get_rng_state().clone()
    return st


def restore_inference_state(state: dict, **modules) -> None:
    state = dict(state)
    rng = state.pop("_rng", None)
    bundle = state.pop("_rng_bundle", None)
    if bundle is not None:
        # the full bundle supersedes the CPU-only `_rng`: on CUDA a stochastic
        # evaluation draws from the CUDA generator, which `_rng` never held (X10)
        restore_rng_bundle(bundle)
    elif rng is not None:
        torch.set_rng_state(rng)
    missing = set(state) - set(modules)
    if missing:
        raise KeyError(f"no module supplied for saved state {sorted(missing)}")
    for name, m in modules.items():
        if name in state:
            m.load_state_dict(state[name])


# --------------------------------------------------------------------------
# RNG contract for methods with stochastic inference (FairVGNN's Gumbel mask).
# Two purposes, kept apart (X10):
#
#   replay       the complete generator state immediately before an epoch's
#                validation call, so a restored checkpoint reproduces that
#                epoch's recorded validation scores exactly.
#   final eval   one evaluation state per (dataset, split, run), fixed in advance
#                and independent of any outcome, applied identically to every
#                arm and selector slot of the cell -- so M1 and M0, BCE and AUC,
#                differ by their parameters and not by where the generator
#                happened to be when the checkpoint was taken.
# --------------------------------------------------------------------------
def capture_rng_bundle() -> dict:
    """Python, NumPy, Torch CPU and every CUDA generator. Reading consumes none."""
    import random as _random
    return {"py": _random.getstate(),
            "np": np.random.get_state(),
            "cpu": torch.get_rng_state().clone(),
            "cuda": ([t.clone() for t in torch.cuda.get_rng_state_all()]
                     if torch.cuda.is_available() else None)}


def restore_rng_bundle(bundle: dict) -> None:
    import random as _random
    _random.setstate(bundle["py"])
    np.random.set_state(bundle["np"])
    torch.set_rng_state(bundle["cpu"])
    if bundle.get("cuda") is not None:
        torch.cuda.set_rng_state_all(bundle["cuda"])


def eval_rng_seed(dataset: str, split: int, run: int) -> int:
    """The final-evaluation seed of a cell: a fixed function of its identity.

    Chosen before any result and never after; it depends on nothing a test
    outcome could influence.
    """
    import zlib as _zlib
    return _zlib.crc32(f"{dataset}|{int(split)}|{int(run)}|eval".encode())


def seed_all(seed: int) -> None:
    import random as _random
    _random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BundleHistory:
    """Mixin marker; see `bundle_history`."""


def bundle_history(ref, horizon: int, holder: dict):
    """A ValidationHistory whose checkpoints carry the full RNG bundle.

    The instrumented method assigns its own `state_fn`, which snapshots only the
    CPU generator. Any function assigned here is wrapped so the snapshot's
    `_rng` is replaced by `holder["bundle"]` -- the bundle captured immediately
    before the validation call of the epoch being recorded. Done at the harness,
    so the method's source is not edited.
    """
    class _H(ValidationHistory, BundleHistory):
        @property
        def state_fn(self):
            return self.__dict__.get("_inner_state_fn")

        @state_fn.setter
        def state_fn(self, fn):
            if fn is None:
                self.__dict__["_inner_state_fn"] = None
                return

            def wrapped(_fn=fn):
                st = _fn()
                st.pop("_rng", None)
                if holder.get("bundle") is None:
                    raise RuntimeError("no RNG bundle captured before validation")
                st["_rng_bundle"] = holder["bundle"]
                return st
            self.__dict__["_inner_state_fn"] = wrapped
    return _H(ref, horizon)


@dataclass
class EpochRecord:
    epoch: int
    stage: str
    raw_score: np.ndarray
    predictive_loss: float
    metrics: dict
    nonfinite: int = 0
    # Scalars a method's own selector needs that are not derivable from the
    # validation score -- NIFTY's invariance loss is the case that forced this.
    # Without it, replay could not be tested for that method, and "the selector
    # uses something we did not store" would be indistinguishable from "the
    # trajectory is incomplete".
    extra: dict = field(default_factory=dict)


class ValidationHistory:
    """Everything a selector may see. Deliberately holds no test data.

    `add()` computes metrics through the unified operator rather than storing
    whatever the method's own metric code produced, so that a selector replayed
    here is replayed under one metric definition.
    """

    def __init__(self, ref: SplitRef, horizon: int, state_fn=None):
        """`state_fn` returns a snapshot of whatever must be restored later.

        Set by the caller, not by the instrumented method, so no training loop
        needs to know that checkpoints are being kept. Each shared selector gets
        one running-best slot, updated inside `add()`; without this the only
        checkpoint available would be the one the method's *own* selector chose,
        and σ_c could name an epoch whose weights no longer exist.
        """
        self.ref = ref
        self.horizon = int(horizon)
        self.records: list[EpochRecord] = []
        self.state_fn = state_fn
        self.slots: dict[str, tuple[int, object]] = {}
        self._best = {"common_bce": math.inf, "common_auc": -math.inf}

    def add(self, epoch: int, raw_score, *, stage: str = "classifier",
            decision: str = "score>0", extra: dict | None = None) -> EpochRecord:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; have {STAGES}")
        q = np.asarray(raw_score, dtype=np.float64)
        if q.shape != self.ref.y.shape:
            raise ValueError(f"raw_score length {q.shape} != split {self.ref.y.shape}")
        nonfinite = int((~np.isfinite(q)).sum())
        safe = np.where(np.isfinite(q), q, 0.0)
        m = evaluate(self.ref.y, self.ref.a, raw_score=safe, decision=decision)
        yy = self.ref.y.astype(float)
        if decision == "score>0":
            # stable BCE-with-logits: log(1+exp(-|z|)) + max(z,0) - z*y.
            # The naive 1/(1+exp(-z)) overflows on large negative logits, and a
            # silently wrong loss would move the selected epoch.
            z = safe
            bce = float((np.logaddexp(0.0, -np.abs(z)) + np.maximum(z, 0.0)
                         - z * yy).mean())
        else:
            p = np.clip(safe, 1e-12, 1 - 1e-12)
            bce = float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())
        r = EpochRecord(epoch=int(epoch), stage=stage, raw_score=q,
                        predictive_loss=bce, metrics=m, nonfinite=nonfinite,
                        extra=dict(extra or {}))
        self.records.append(r)

        if self.state_fn is not None and stage == "classifier":
            # strict comparisons, so ties keep the earliest epoch, as every
            # audited selector does
            if r.predictive_loss < self._best["common_bce"]:
                self._best["common_bce"] = r.predictive_loss
                self.slots["common_bce"] = (r.epoch, self.state_fn())
            au = r.metrics["auc"]
            if math.isfinite(au) and au > self._best["common_auc"]:
                self._best["common_auc"] = au
                self.slots["common_auc"] = (r.epoch, self.state_fn())
        return r

    def classifier_records(self) -> list[EpochRecord]:
        return [r for r in self.records if r.stage == "classifier"]

    def completeness(self) -> dict:
        """Regression check 4: is the trajectory actually complete?"""
        cls = self.classifier_records()
        eps = [r.epoch for r in cls]
        # An empty trajectory must fail first and loudly. Every other check
        # below passes vacuously on zero records -- "score length constant" and
        # "no non-finite epochs" are both trivially true of nothing -- and a
        # logger that was never wired in once looked like a pass because of it.
        return {
            "n_logged": len(cls),
            "horizon": self.horizon,
            "nonempty": len(cls) > 0,
            "complete": len(cls) > 0 and len(cls) == self.horizon,
            "epochs_unique": len(set(eps)) == len(eps),
            "epochs_contiguous": bool(eps and sorted(eps) == list(range(min(eps),
                                                                       max(eps) + 1))),
            "n_nonfinite_epochs": sum(1 for r in cls if r.nonfinite),
            "score_len_constant": len({len(r.raw_score) for r in cls}) <= 1,
        }


# --------------------------------------------------------------------------
# Selectors. Each takes a ValidationHistory and returns an epoch index.
# Ties go to the earliest epoch, matching every audited implementation.
# --------------------------------------------------------------------------
def _argbest(recs, key, maximise: bool, floor: float | None = None) -> int | None:
    best_i, best_v = None, (-math.inf if maximise else math.inf)
    if floor is not None:
        best_v = floor
    for r in recs:
        v = key(r)
        if v is None or not math.isfinite(v):
            continue
        if (v > best_v) if maximise else (v < best_v):   # strict: earliest wins
            best_i, best_v = r.epoch, v
    return best_i


def select_min_bce(h: ValidationHistory) -> int | None:
    """σ_c primary — the shared fairness-unaware selector.

    Not called "neutral": it is one predictive criterion among several, chosen
    because it does not read the sensitive attribute.
    """
    return _argbest(h.classifier_records(), lambda r: r.predictive_loss, maximise=False)


def select_max_auc(h: ValidationHistory) -> int | None:
    """σ_c secondary — fixed now, before any result.

    BCE is calibration-sensitive: an intervention that barely changes the
    ranking but shifts confidence can move the selected epoch. AUC is
    threshold-free, so agreement between the two is evidence the attribution
    does not hinge on the selector. Registered in advance precisely so that
    adding it later could not look like choosing the selector that gave the
    nicer answer.
    """
    return _argbest(h.classifier_records(), lambda r: r.metrics["auc"], maximise=True)


def select_min_val_loss(h: ValidationHistory) -> int | None:
    """σ_code for GNN and NIFTY."""
    return select_min_bce(h)


def select_max_acc(h: ValidationHistory) -> int | None:
    """σ_code for FairGNN and BeMap."""
    return _argbest(h.classifier_records(), lambda r: r.metrics["acc"], maximise=True)


def select_composite(h: ValidationHistory, alpha: float = 1.0,
                     f1: str = "binary") -> int | None:
    """σ_code for FairVGNN, FairGB and FairSIN: auc + f1 + acc − α(ΔDP + ΔEO).

    The `floor=0.0` reproduces their initialisation `best_val_tradeoff = 0`.
    `f1` selects which F1 the formula means. FairGB and FairSIN call
    `f1_score(y, pred)` with no `average`, so theirs is **binary**; passing
    "micro" here would silently turn `auc + f1 + acc` into `auc + 2·acc`.

    when α(ΔDP + ΔEO) exceeds auc + f1 + acc — the original code selects
    **nothing** and returns whatever state it started from. Replaying without
    the floor would silently disagree with the code on exactly those runs.
    """
    def k(r):
        m = r.metrics
        f1v = m["f1_binary"] if f1 == "binary" else m["f1_micro"]
        eo = m["eo"] if m["eo_defined"] else 0.0
        return m["auc"] + f1v + m["acc"] - alpha * (m["dp"] + eo)
    return _argbest(h.classifier_records(), k, maximise=True, floor=0.0)


def select_nifty(h: ValidationHistory) -> int | None:
    """σ_code for NIFTY: argmin over `val_c_loss + val_s_loss`.

    `val_s_loss` is `sim_coeff * (l1 + l2)`, the counterfactual-invariance
    objective on validation, so this selector is fairness-weighted — and
    `sim_coeff` therefore sits in the training objective *and* here. Both
    scalars are read from `extra`, because neither is recoverable from the
    classifier's validation score alone.
    """
    return _argbest(h.classifier_records(),
                    lambda r: (r.extra.get("val_c_loss", math.nan)
                               + r.extra.get("val_s_loss", math.nan)),
                    maximise=False)


def select_final(h: ValidationHistory) -> int | None:
    """σ_code for FMP: no selection, the last epoch is reported."""
    cls = h.classifier_records()
    return cls[-1].epoch if cls else None


@dataclass
class CheckpointSlots:
    """One retained state per selector, instead of H checkpoints.

    Storing every epoch is unnecessary: the selectors are fixed in advance, so
    only their running bests need to survive. Copying a state must never touch
    the training graph.
    """
    # Four selectors are fixed in advance, so four slots. An earlier draft said
    # three and was wrong: sigma_c has a primary (BCE) and a secondary (AUC)
    # form, both registered before any result, and they can choose different
    # epochs. Methods where sigma_code == sigma_published may share one slot.
    #
    # This is also why the slots must cover every selector that will ever be
    # reported: only running bests are kept, so a selector added after the runs
    # could not be applied retrospectively -- its checkpoint would be gone.
    names: tuple[str, ...] = ("code", "published", "common_bce", "common_auc")
    best_epoch: dict = field(default_factory=dict)
    best_state: dict = field(default_factory=dict)

    def offer(self, name: str, epoch: int, state, is_better: bool) -> None:
        if name not in self.names:
            raise ValueError(f"unknown slot {name!r}")
        if is_better:
            self.best_epoch[name] = int(epoch)
            self.best_state[name] = state
