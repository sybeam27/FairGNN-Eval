# Phase 1a — Model-selection audit

Which epoch does each system return? Audited by reading every training loop,
2026-09-13, before any unification is built.

## Why this is not a detail

Training produces one model per epoch. Something must choose which to keep.
When that choice uses disparity, the method is **searching over epochs for a
fair one** — a fairness intervention by any reading, and one that appears in no
paper's ablation list and in none of X1's component switches.

## The audit

| system | rule | class | source |
|---|---|---|---|
| GNN (baseline) | lowest validation loss | loss | GNN.py:356 |
| NIFTY | lowest `BCE_val + sim_coeff · invariance_val` | **fairness-weighted** | NIFTY.py:514, 389, 399 |
| FairGNN | highest validation **accuracy**, above a threshold that itself adapts to the best accuracy seen in the first 10 epochs | accuracy | FairGNN.py:199-203, 245 |
| BeMap | highest validation accuracy; disparity is reported *at* that epoch | accuracy | train_bemap.py:141-145 |
| **FMP** | **none — the final epoch is reported** | none | main.py:202, 263 |
| **FairGB** | highest `auc + f1 + acc − α·(ΔDP + ΔEO)` | fairness-weighted | FairGB_alg.py:161-163 |
| **FairGT** | highest `acc − ΔDP`, **and only when `val_dp > 0`** | fairness-weighted | FairGT_alg.py:319-328 |
| FairGate | highest `acc − ρ·dp − (1−ρ)·eo` | fairness-weighted | metrics.py:57 |
| BIND | lowest `val_dp + val_eo` | fairness-only | adapters/bind.py:394 |

**Corrected 2026-09-13 during 1D instrumentation.** NIFTY was listed above as
selecting on validation loss with no fairness term. That was read from
`NIFTY.py:435`, which is `fit_GNN` — a *different* method on the same class. The
main `fit` (line 444) selects at line 514 on `val_c_loss + val_s_loss`, and
`ssf_validation` returns `sim_loss = sim_coeff * (l1 + l2)` as the first of
those (lines 389, 399). `sim_loss` is the counterfactual-invariance objective
evaluated on validation, so **NIFTY's selector is fairness-weighted.**

This matters beyond the table. `sim_coeff` appears in the training objective
*and* in the selection criterion, so setting it to 0 to build `M₀` removes the
objective and silently switches the selector to pure BCE at the same time —
the same double duty already found in FairGB's `α`. E11's `NIFTY_off` arm
therefore changed two things at once, which was not known when it ran, and its
result is superseded along with everything else pending unified re-evaluation.

**Five distinct conventions across nine systems**: validation loss, validation
accuracy, none at all, a fairness-weighted trade-off score, and disparity alone.

### Correction to an earlier statement

This study said "four of six methods select their checkpoint using disparity
while the plain baseline does not". That was reached before FairGNN's and
BeMap's loops had been read, and it is wrong. Of the four core-audit methods,
**two** use fairness in selection (FairGB, FairGT) and two do not (FairGNN,
NIFTY). The claim is smaller than stated and the correct version is sharper:
there is no shared convention at all, and the range runs from disparity-only to
no selection whatsoever.

### Two specific oddities worth reporting rather than smoothing over

**FairGT refuses a perfectly fair epoch.** Its update condition is
`score > best_score and val_dp > 0`. An epoch whose validation disparity is
exactly zero can never be selected, however good its accuracy.

**FMP's result is whatever the last epoch gives.** With no selection and no
early stopping, its reported numbers depend entirely on the epoch budget, and
comparing it against a method that selects over 1000 epochs compares two
different things.

## What this blocks

1. **FairGB's α cannot be ablated as it stands.** The same α weights the
   alignment loss *and* the disparity term in the selection score
   (FairGB_alg.py:162). Setting it to zero changes the objective and the
   returned model together.
2. **Every package comparison carries an unmatched component.** FairGB and
   FairGT get to pick a fair epoch; the baseline they are compared against does
   not.

## The unified-selection arm

Two arms, and the difference between them is the quantity:

    arm A — as published   each system keeps its own rule
    arm B — unified        B, M0 and M1 all keep the epoch chosen by one
                           shared rule

Arm B does not "fix" arm A. Arm A is what the literature reports and stays the
primary reading. The contrast measures **how much of a reported fairness gain
is produced by checkpoint selection rather than by training**.

The shared rule must not itself be fairness-weighted, or arm B would install a
fairness intervention in every system including the baseline. It is therefore
**lowest validation loss** — the convention the plain baseline and NIFTY already
use, and the only one in the table that is neutral with respect to disparity.

---

# Phase 1a (part 3) — Decision-rule audit

The evaluation pipeline has four operators and the estimand fixes three of them:

    z  training intervention   ← the object of study
    σ  checkpoint selection    ← audited above: five conventions, not shared
    δ  decision rule           ← audited here
    G  measurement             ← audited in X2_OUTCOME_AUDIT.md: not shared

## δ is consistent, and that is the result

| method | rule | source | probability threshold |
|---|---|---|---|
| GNN | `logit > 0` | GNN.py:373 | 0.5 |
| NIFTY | `logit > 0` | NIFTY.py:530 | 0.5 |
| FairGNN | `logit > 0` | FairGNN.py:206 | 0.5 |
| FairGB | `logit > 0` | FairGB_alg.py:177 | 0.5 |
| BeMap | `logit > 0` | train_bemap.py:184 | 0.5 |
| FairGT | `argmax` over two classes | FairGT_alg.py:310 | 0.5 |
| FMP | `softmax_prob > 0.5` | utils.py:277, 450 | 0.5 |

All seven are the same operating point. FMP's `> 0.5` looked like an outlier
until the input was traced: `all_y = F.softmax(all_logit, dim=1)` (main.py:211),
so it thresholds a probability, not a logit. NIFTY's `argmax(dim=1)` at line 369
is a separate sensitive-leakage probe, not the prediction path.

**No method tunes its threshold on validation fairness.** Had one done so, that
would itself be a fairness intervention sitting outside every component switch,
as checkpoint selection does.

## ΔDP and ΔEO are also consistent

Both families compute

    ΔDP  |P(ŷ=1 | s=0) − P(ŷ=1 | s=1)|
    ΔEO  |P(ŷ=1 | s=0, y=1) − P(ŷ=1 | s=1, y=1)|

with positive outcome ŷ = 1, groups s ∈ {0,1}, absolute difference. Orientation
matches across repositories.

Two small differences, recorded rather than assumed away:

  * FMP defines group 1 as `sens > 0`; the others use `sens == 1`. Identical for
    binary sensitive attributes, divergent if one is ever multi-valued.
  * No implementation guards an empty group. `sum(pred[idx_s0]) / sum(idx_s0)`
    divides by zero when a group or a group's positive stratum is absent from
    the evaluated set, producing nan or inf silently. `core/evaluator.py`
    returns `(nan, False)` and flags it.

## Where Phase 1a leaves the four operators

    z   the object of study, catalogued per method in X1
    σ   NOT shared — five conventions across nine systems
    δ   shared — one operating point, verified
    G   NOT shared — two different quantities under the name "AUC"

So two of the three operators that must be held fixed are not fixed in the
literature, and one is. That is a more precise claim than "the protocol is
inconsistent", and the one that is fixed is worth reporting because it shows the
audit is not simply finding fault everywhere it looks.
