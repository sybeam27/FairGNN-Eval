# Arm A analysis plan, frozen before splits 23-25 of bail and credit exist

Written and committed before the added splits were run. Nothing below was
chosen after seeing their results.

## Naming, corrected

Arm A runs at a fixed audit horizon H = 200, and some bail/credit
configurations are `local-unverified`. Nothing in Arm A is therefore a
published protocol, and the term is reserved for Arm B cells backed by an
official artifact. Arm A reports two views under their own names:

* **Arm-A/native-selector package view** -- M1 at the method's own code-native
  selector against B at its own, with the outcome recomputed by G_c. This
  replaces every earlier use of `tau_pkg^pub` in Arm A.
* **Arm-A/common-selector audit decomposition** -- `tau_base^audit`,
  `tau_int`, `tau_pkg^audit`, all at one sigma_c.

Every `tau_int` in Arm A is an effect **conditional on H = 200**.

## Frozen contract

Unchanged from the completed cells: H = 200; the current M0/M1 definitions;
the per-method preprocessing policy; sigma_c^BCE primary and sigma_c^AUC
robustness; the unified evaluator G_c; B, M0 and M1 recorded in one process per
cell; the restore-fidelity contract. No method, metric, selector,
hyperparameter or threshold rule is added. Splits 20-22 of bail and credit are
not re-run.

## Zero-effect cells are observations, not missing data

Both kinds stay in the primary analysis at their measured value:

1. `tau_int = 0` because M1 and M0 selected the same checkpoint.
2. `dDP = 0` because the scores moved but no hard prediction flipped.

Neither is dropped, imputed or reweighted. Diagnostics are added instead:
`m0_epoch`, `m1_epoch`, `n_flip`, `n_flip_a0`, `n_flip_a1`, `n_test_a0`,
`n_test_a1`, and `dp_min_step` -- the smallest nonzero |dDP| one flipped
prediction can produce on that split, `min(1/n_a0, 1/n_a1)`. `dp_min_step` is a
diagnostic of DP's finite-sample resolution and is never used as a fairness
metric or as a threshold.

The score-level diagnostics require the raw scores, so they exist only for
cells run from this commit onward: bail and credit splits 23-25. `dp_min_step`
depends only on the split and is computed for every cell, old and new.

## What `epoch 0` means

Read from the source: in all four methods the validation record is written
**after** that epoch's parameter update -- `GNN.py:352-367` (step then
validate), `FairGNN.py:195-204` (`self.optimize(...)` then add),
`NIFTY.py:500-516`, `FairGB_alg.py:175-191`, `FairVGNN.py:1097-1241`. So
`epoch 0` is the checkpoint after the first training update, not the untrained
initialization. A selector landing on epoch 0 means no later epoch scored
better on that selector's criterion, which is a fact about the criterion.
Selector behaviour is not changed.

## Primary selector-sensitivity analysis

For every finite cell, including exact zeros at their value 0:

    D_{s,r} = tau_int^{BCE}_{s,r} - tau_int^{AUC}_{s,r}

reported as a distribution with its sign stability, and as the share of cells
with |D| > |tau_int^BCE|. The earlier analysis that excluded cells below DP's
resolution is **reclassified as a secondary diagnostic** and reported as such.

## Uncertainty: hierarchical bootstrap

Per (method, dataset), 10,000 replicates. One replicate:

1. resample the 6 splits with replacement;
2. within each drawn split, resample its 5 runs with replacement;
3. keep each drawn cell's B/M0/M1 and BCE/AUC results paired exactly as
   recorded -- the cell is the resampling unit, never a single arm.

Report 95% bootstrap intervals for `tau_base^audit`, `tau_int`,
`tau_pkg^audit` and `D_selector`, on both coordinates.

The existing descriptive rules -- sign stability 0.75, magnitude 0.010 -- stay
a descriptive taxonomy. They are not a statistical statement and are not
presented as equivalent to a bootstrap interval.

## Cross-dataset reporting

Three datasets is not a sample of datasets. No population-level confidence
interval is attached to any cross-dataset mean. Per-dataset results are the
primary reporting unit.

The observed `dataset mean SD / method mean SD = 1.52` stays a descriptive
observation. No claim that dataset dominates method is made.

## Order of analysis

contract -> raw paired cells -> per split -> per run -> per-dataset
decomposition -> selector sensitivity -> uncertainty -> cross-dataset
descriptive summary. Overall means are computed last.

Primary outcome `[dAUC, -dDP]`, secondary `[dAUC, -dEO]`, for
`tau_base^audit`, `tau_int` and `tau_pkg^audit` in every (method, dataset).

## Questions this plan answers

* **Q1** does `|tau_base| > |tau_int|` hold in bail and credit at 6x5?
* **Q2** how stable is the direction of the intervention effect across splits
  and runs?
* **Q3** including every zero cell, is the selector-induced change comparable
  to or larger than the intervention effect?
* **Q4** is selector sensitivity dataset-specific? No generalization that the
  selector beats the intervention everywhere is to be drawn.

## After Arm A

Arm B starts only when Arm A is complete at 6x5 on all three datasets and this
analysis is done. Its single purpose is to check whether Arm A's attribution
pattern survives the official horizon and configuration. High-cost native cells
such as FairGB on credit at 2000 epochs get their own budget decision after
Arm A's final result.
