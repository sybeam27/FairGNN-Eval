# Phase 1a (continued) — Outcome-measurement audit, and what it invalidates

Found 2026-09-13 while auditing model selection. The paired-evaluation condition
requires `M0` and `M1` to share the outcome measurement. That had not been
checked either, and it is worse than the selection problem.

## The defect

Three systems compute "AUC" from **hard 0/1 predictions**:

    GNN (baseline)  output_preds = (output.squeeze() > 0)   GNN.py:373
    NIFTY           same                                     NIFTY.py:530
    FairGNN         output_np = (output > 0).long()          FairGNN.py:206

then call `roc_auc_score(labels, output_preds)`. On binary hard labels that is
not an AUC at all — it reduces to balanced accuracy, `(TPR + TNR) / 2`.

Two compute it from probabilities, so it is a real AUC:

    FairGT          prob = softmax(logits)[:, 1]             FairGT_alg.py:309
    FairGate (E10)  sigmoid probabilities                    harness/core/metrics.py

**The audit's `roc_auc_mean` column therefore holds two different quantities
depending on the row.**

## Size, measured

Same trained model, both quantities, split 20:

    setting        hard 0/1     probabilities     gap
    german          0.5819         0.7192       +0.1373
    recidivism      0.8865         0.9052       +0.0187
    credit          0.6876         0.7227       +0.0351
    pokec_z         0.6895         0.7409       +0.0514
                                        mean    +0.056

Finding 16's headline — FairGate's backbone beating the audit's GNN by
**dAUC +0.0800** — compared a real AUC against a balanced accuracy. The metric
gap alone averages +0.056 on these settings. Most of that +0.08 may be the
definition rather than the backbone.

## What this invalidates, and what survives

**Invalidated — any comparison that crosses the metric boundary:**

    Finding 16   backbone vs audit GNN (+31, dAUC +0.0800)
                 all_alloc vs audit GNN (+43, dAUC +0.0728)
    Finding 14   the audit table, wherever FairGate or FairGT (probabilities)
                 is compared against GNN, NIFTY or FairGNN (hard labels)
    H7           the `total` and `backbone` rows of every method whose metric
                 differs from the baseline's

**Correction, same day.** An earlier version of this section said comparisons
inside one convention "survive". That is too strong. Two arms sharing the *same
wrong* metric are internally consistent, but the quantity they are consistent
about is a balanced-accuracy-like utility, not AUC. Every Pareto result mixes
utility with fairness, so **every net/win count in this study must be recomputed
on true AUC before it is quoted again**, including the preprocessing contrast.

What actually survives is narrower: comparisons where **both sides are verified
to have used the same true-AUC computation.**

**Verified same true-AUC computation — still valid:**

    Finding 16   all_alloc vs backbone: net +0, 9 win, 9 lose, 36 incomparable.
                 Both arms are E10, both real AUC. **The core claim stands** —
                 the fairness stack Pareto-dominates its own backbone on net
                 zero cells.
**Internally consistent but measuring the wrong utility — must be recomputed,
not quoted:**

    Finding 14   NIFTY vs GNN and the preprocessing-unified contrast. Both sides
                 use hard labels, so the comparison is self-consistent, but its
                 Pareto counts are over a balanced-accuracy axis. The
                 significance-to-null result may or may not hold on true AUC;
                 it is not established until recomputed.
    H7           NIFTY vs NIFTY_off, FairGNN vs FairGNN_off. Same codebase and
                 therefore the same metric, so the contrast is well defined --
                 but again on the wrong utility axis.

## Why the framework is partly protected by construction

The primary estimand `τ_int = E[Y(M1) − Y(M0)]` is a **within-method**
comparison: `M0` and `M1` are the same codebase and therefore share whatever
outcome computation that codebase uses. A metric defined inconsistently across
repositories cannot corrupt it.

It is the *package* effect `E(M1, B)` — the cross-codebase comparison the
literature reports — that the defect lands on. Which is, uncomfortably, the
thing this paper says is not interpretable. The defect is an instance of the
paper's own thesis, found in our own audit.

## Required before the confirmatory phase

**All outcomes are computed externally, by one function, from stored per-node
probabilities.** No method's own metric code is trusted for the primary
analysis. Each method returns validation and test probabilities; AUC, ΔDP and
ΔEO are computed outside it.

Each method's own reported numbers are kept alongside, as the published-package
reading — the difference between the two is reportable in the same way the
selection contrast is.

This must be built before any confirmatory cell runs, because every one of them
depends on it.

---

## Addendum 2026-09-13 — F1 is inconsistent too, and it reaches a selector

Found while a FairGB selector replay disagreed with the code by three epochs.

    FairGB    f1_score(y, pred)                binary F1   FairGB/eval.py:33
    FairSIN   f1_score(y, pred)                binary F1   evaluation.py:24
    NIFTY     f1_score(..., average='micro')   = accuracy
    FairGNN   f1_score(..., average='micro')   = accuracy

Measured on German at epoch 0: binary 0.7015, micro 0.6120.

This is not only a reporting inconsistency. Three methods select checkpoints on
`auc + f1 + acc − α(ΔDP + ΔEO)`. With binary F1 those are three distinct terms;
with micro F1 the expression is `auc + 2·acc` and the accuracy term is silently
doubled. Replaying FairGB's selector with micro F1 chose epoch 42; with binary
F1 it chose epoch 39, which is what the code chose.

The unified operator now reports both, and a replayed selector must state which
it means.

The diagnosis is also the argument for storing each method's own metric values
beside the unified ones. Without `code_f1` in the trajectory, a three-epoch
disagreement would have been indistinguishable from an incomplete trajectory or
a tie-breaking error — the two other explanations that had to be ruled out
before calling anything a defect.

### Running count of G-operator inconsistencies

    AUC   hard predictions vs raw scores    GNN, NIFTY, FairGNN vs FairGT, FairGB, FairGate
    F1    micro vs binary                   NIFTY, FairGNN vs FairGB, FairSIN

Both were found by instrumenting rather than reading, and both reach a
checkpoint selector in at least one method.
