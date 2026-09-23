"""
The unified evaluation operator — one measurement function for every method.

Written 2026-09-13, after finding that three systems in this study report
`roc_auc_score(y, (logit > 0))`. On binary hard input that is not an AUC; it is
(TPR + TNR)/2, i.e. balanced accuracy. Measured on one trained model the gap
against a real AUC is +0.1373 on German and +0.056 on average — the same order
as effects this study was attributing to architecture.

The rule this module exists to enforce:

    **A method's own code does not own the definition of its outcome metrics.**

The evaluation pipeline has four operators, and only the first is the object of
study:

    training intervention  z   ->  what the method claims as its contribution
    checkpoint selection   σ   ->  which epoch is returned
    decision rule          δ   ->  score -> hard prediction
    measurement            G   ->  (y, a, score, yhat) -> outcomes

The intervention estimand holds σ, δ and G fixed and varies only z. This module
is G. Nothing here reads a model, a loss, or an epoch.

Inputs, and why they are what they are
--------------------------------------
`score` is a **non-thresholded decision score for the positive class** — a logit
is fine, a probability is fine, any strictly increasing transform of either is
fine. ROC AUC is rank-based, so sigmoid(logit) and logit give identical values;
requiring probabilities would be a needless constraint. What is *not* fine is a
thresholded prediction, and `auc()` refuses it rather than silently returning
balanced accuracy.

`yhat` comes from an explicit decision rule that is recorded in the output, so
that "which threshold" never becomes an unlogged degree of freedom the way
"which AUC" did.

Orientation, fixed here and not per caller
------------------------------------------
    positive outcome   yhat == 1
    groups             a == 0 and a == 1
    ΔDP   |P(yhat=1 | a=0) − P(yhat=1 | a=1)|
    ΔEO   |P(yhat=1 | a=0, y=1) − P(yhat=1 | a=1, y=1)|

Both are absolute differences, so smaller is better and the sign convention
cannot drift between callers. ΔEO is undefined when a group has no positive
label in the evaluated set; it returns NaN and sets `eo_defined=False` rather
than silently contributing a zero.
"""
from __future__ import annotations

import numpy as np

POSITIVE_LABEL = 1
DECISION_RULES = {
    "score>0":   lambda s: (s > 0.0).astype(int),        # logits
    "prob>0.5":  lambda s: (s > 0.5).astype(int),        # probabilities
}


class ScoreContractError(ValueError):
    """Raised when the declared input contract is violated."""


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U), ties averaged.

    Computes what it is given. It does **not** inspect the number of distinct
    values to guess whether the caller meant a ranking score: AUC on a binary
    score is perfectly well defined -- it equals balanced accuracy, because a
    two-valued score gives the ROC curve one interior operating point. The
    defect this module exists for was never the arithmetic. It was comparing
    that quantity against a continuous-score AUC under the same column name.

    Guessing from value counts would put the semantics in the wrong place, so
    the contract is enforced in `evaluate()` instead, where the caller has to
    say which field it is passing.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    if y.shape != score.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs score {score.shape}")
    pos, neg = int((y == POSITIVE_LABEL).sum()), int((y != POSITIVE_LABEL).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == POSITIVE_LABEL].sum() - pos * (pos + 1) / 2.0)
                 / (pos * neg))


def balanced_accuracy(y: np.ndarray, yhat: np.ndarray) -> float:
    """(TPR + TNR) / 2 — what `roc_auc_score` returns on binary hard input."""
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    pos, neg = (y == POSITIVE_LABEL), (y != POSITIVE_LABEL)
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float(((yhat[pos] == POSITIVE_LABEL).mean()
                  + (yhat[neg] != POSITIVE_LABEL).mean()) / 2.0)


def _micro_f1(y: np.ndarray, yhat: np.ndarray) -> float:
    """`f1_score(..., average='micro')` for single-label input — equals accuracy."""
    return float((np.asarray(y).astype(int) == np.asarray(yhat).astype(int)).mean())


def _binary_f1(y: np.ndarray, yhat: np.ndarray) -> float:
    """`f1_score(y, yhat)` with sklearn's default, i.e. F1 of the positive class.

    Both variants are reported because the audited implementations disagree on
    which one "F1" means: FairGB (FairGB/eval.py:33) and FairSIN
    (evaluation.py:24) call `f1_score(y, pred)` with no `average`, which is
    binary F1; NIFTY and FairGNN pass `average='micro'`, which is accuracy.

    The difference is not cosmetic. Three selectors score on
    `auc + f1 + acc − α(ΔDP + ΔEO)`. With binary F1 that is three distinct
    terms; with micro F1 it is `auc + 2·acc`, and the accuracy term is silently
    doubled. Measured on German at epoch 0: binary 0.7015, micro 0.6120.
    """
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    tp = float(((yhat == 1) & (y == 1)).sum())
    fp = float(((yhat == 1) & (y == 0)).sum())
    fn = float(((yhat == 0) & (y == 1)).sum())
    return 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)


def _rate(mask: np.ndarray, yhat: np.ndarray) -> float:
    return float((yhat[mask] == POSITIVE_LABEL).mean()) if mask.any() else float("nan")


def demographic_parity(yhat: np.ndarray, a: np.ndarray) -> tuple[float, float]:
    """Returns (|gap|, signed gap) with signed = rate(a=0) − rate(a=1).

    The signed gap is kept because the absolute one hides a reversal: an
    intervention taking +0.10 to −0.10 leaves |ΔDP| at 0.10 and looks like it
    did nothing, while the advantaged group has swapped. Primary analysis uses
    the absolute value; the sign travels with it as diagnostic metadata.
    """
    yhat, a = np.asarray(yhat).astype(int), np.asarray(a).astype(int)
    g = _rate(a == 0, yhat) - _rate(a == 1, yhat)
    return abs(g), float(g)


def equal_opportunity(yhat: np.ndarray, y: np.ndarray, a: np.ndarray
                      ) -> tuple[float, float, bool]:
    """Returns (|ΔEO|, signed ΔEO, defined).

    Undefined when either group has no positive label in the evaluated set. It
    returns NaN and False rather than a silent zero -- and downstream code must
    **report the fraction of undefined cells rather than dropping them from a
    mean**. A method that collapses until one group has no positives would
    otherwise look good precisely where it failed.
    """
    yhat, y, a = (np.asarray(v).astype(int) for v in (yhat, y, a))
    m0, m1 = (a == 0) & (y == POSITIVE_LABEL), (a == 1) & (y == POSITIVE_LABEL)
    if not m0.any() or not m1.any():
        return float("nan"), float("nan"), False
    g = _rate(m0, yhat) - _rate(m1, yhat)
    return abs(g), float(g), True


def evaluate(y, a, *, raw_score=None, hard_prediction=None,
             positive_class: int = 1, decision: str = "score>0",
             mode: str = "primary") -> dict:
    """The measurement operator G, with an explicit input contract.

    The caller states what it is handing over. Nothing is inferred.

        raw_score         a non-thresholded decision score for the positive
                          class -- a logit, a probability, or any strictly
                          increasing transform of either. AUC is rank-based, so
                          all of these give the same value.
        hard_prediction   an already-thresholded 0/1 label.
        positive_class    which class index the score refers to. Declared by
                          the adapter, never inferred from the labels: choosing
                          an orientation by looking at which one scores better
                          would be a decision made with the test outcome.

    mode
        "primary"       AUC must come from `raw_score`. Passing only
                        `hard_prediction` is refused, because the resulting
                        number is balanced accuracy and would sit in an AUC
                        column beside real AUCs from other methods -- the exact
                        confusion this module exists to prevent.
        "reproduction"  AUC may be computed from `hard_prediction`, to reproduce
                        what a published implementation reported. The result is
                        labelled `auc_from="hard_prediction"` so it can never be
                        mistaken for the primary quantity.
    """
    if mode not in ("primary", "reproduction"):
        raise ValueError(f"mode must be 'primary' or 'reproduction', got {mode!r}")
    if raw_score is None and hard_prediction is None:
        raise ScoreContractError("pass raw_score, hard_prediction, or both")
    if positive_class != 1:
        raise ScoreContractError(
            f"positive_class={positive_class}: re-orient the score in the "
            "adapter and declare 1 here, so the orientation is recorded at the "
            "point where it is known rather than fixed up during measurement")

    y, a = np.asarray(y), np.asarray(a)

    # NBA, Pokec-z and Pokec-n encode "unlabelled" as -1. The splits exclude
    # those nodes, verified 2026-09-13, but nothing enforces it: a future loader
    # change or a method evaluating on all nodes would silently push -1 into the
    # negative class and corrupt AUC and EO at once.
    bad = np.setdiff1d(np.unique(y.astype(int)), np.array([0, 1]))
    if bad.size:
        raise ScoreContractError(
            f"labels outside {{0,1}} in the evaluated set: {bad.tolist()}. "
            "-1 marks an unlabelled node in NBA and Pokec; it must be excluded "
            "before measurement, not treated as a negative.")

    if hard_prediction is not None:
        yhat = np.asarray(hard_prediction).astype(int)
        rule = "supplied"
    else:
        if decision not in DECISION_RULES:
            raise ValueError(f"unknown decision rule {decision!r}; "
                             f"have {sorted(DECISION_RULES)}")
        yhat = DECISION_RULES[decision](np.asarray(raw_score, dtype=np.float64))
        rule = decision

    if raw_score is not None:
        auc_val, auc_from = auc(y, np.asarray(raw_score, dtype=np.float64)), "raw_score"
    elif mode == "reproduction":
        auc_val, auc_from = auc(y, yhat.astype(np.float64)), "hard_prediction"
    else:
        raise ScoreContractError(
            "primary mode needs raw_score for AUC. AUC from a thresholded "
            "prediction is well defined but equals balanced accuracy, and "
            "belongs in reproduction mode where it is labelled as such.")

    dp_abs, dp_signed = demographic_parity(yhat, a)
    eo_abs, eo_signed, eo_ok = equal_opportunity(yhat, y, a)
    src = raw_score if raw_score is not None else yhat
    return {
        "auc": auc_val,
        "auc_from": auc_from,
        "dp": dp_abs, "dp_signed": dp_signed,
        "eo": eo_abs, "eo_signed": eo_signed, "eo_defined": eo_ok,
        "acc": float((yhat == np.asarray(y).astype(int)).mean()),
        # micro-F1, which on single-label binary classification equals
        # accuracy. Reported separately because three selectors add `f1` and
        # `acc` as if they were different terms, and a replay that substituted
        # one for the other would look faithful while doubling one quantity.
        "f1_micro": _micro_f1(np.asarray(y).astype(int), yhat),
        "f1_binary": _binary_f1(np.asarray(y).astype(int), yhat),
        "balanced_acc": balanced_accuracy(y, yhat),
        "decision_rule": rule,
        "mode": mode,
        "n": int(len(y)),
        "n_pos": int((np.asarray(y).astype(int) == POSITIVE_LABEL).sum()),
        "n_a1": int((np.asarray(a).astype(int) == 1).sum()),
        "score_distinct": int(len(np.unique(np.asarray(src, dtype=np.float64)))),
    }


def assert_score_orientation(y, raw_score, name: str, tol: float = 0.0) -> None:
    """Assert a declared orientation is not obviously backwards.

    **Diagnostic only.** It must never be used to flip a score: choosing an
    orientation because the flipped one scores better is a decision made with
    the evaluation labels. The adapter declares the orientation; this catches
    the case where the declaration is wrong.
    """
    v = auc(y, np.asarray(raw_score, dtype=np.float64))
    if v < 0.5 - tol:
        raise ScoreContractError(
            f"{name}: AUC {v:.4f} < 0.5 with the declared orientation. The "
            "adapter is probably returning the negative-class score. Fix the "
            "adapter -- do not negate here.")


def check_alignment(reference_ids: np.ndarray, ids: np.ndarray, name: str) -> None:
    """Assert two systems evaluated the same nodes in the same order.

    A metric fixed across methods is worth nothing if the methods are scored on
    different node sets, and nothing in the codebases guarantees they are not.
    """
    r, i = np.asarray(reference_ids), np.asarray(ids)
    if r.shape != i.shape or not np.array_equal(r, i):
        raise ValueError(
            f"{name}: evaluated node ids differ from the reference "
            f"({len(i)} vs {len(r)}; equal={np.array_equal(r, i)})"
        )
