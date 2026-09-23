# X24 — NIFTY/German 2×2 protocol-factor decomposition (pre-registration)

**Status: pre-registered before any P10 or P01 result exists.** The commit
hash of this document is recorded in the results report, and no factorial
result is analysed before that commit.

**Scope.** NIFTY on German only. X23 (the frozen native validation, 7 cells),
the Arm A and native CSVs, the X22 transfer classification and all existing
bootstrap outputs are **not modified or redefined**. The factorial results go in
new files with their own analysis.

## 1. Scientific question

Under −ΔDP with σ_c^BCE, NIFTY/German's intervention attribution moved from
τ_int ≈ −0.006 (unresolved; controlled, Arm A) to τ_int ≈ −0.151 (resolved,
harmful; native), as frozen in X23. Two protocol factors changed together, so
the question is:

> Is the controlled → native shift in τ_int associated with the training
> horizon, with the NIFTY augmentation / validation-view configuration, or with
> their interaction?

This experiment decomposes an attribution shift that has already been
observed. It is not another demonstration of protocol sensitivity. No question,
metric, selector, dataset or method is added after results are seen.

## 2. Factors

**H, the training horizon:** H0 = 200, H1 = 1000.

**D, the NIFTY augmentation / validation-view configuration** (short name:
*NIFTY protocol configuration*). It is **not** called "drop rate" or
"dropout".

| | training edge / feature drop | fixed validation-view edge / feature drop (σ_c input) |
|---|---|---|
| **D0**, Arm-A NIFTY configuration (`published("NIFTY","german")`, no drop keys) | 0 / 0 (zeroed by `NIFTY.__init__`) | 0.1 / 0.1 (constructor defaults) |
| **D1**, official/native NIFTY augmentation/validation-view configuration (`native_config`) | 0.001 / 0.1 (`restore_train_drop_rates`) | 0.001 / 0.1 (official command) |

The D1 − D0 contrast is a **protocol-level** contrast bundling three changes:

1. training edge drop: 0 → 0.001;
2. training feature drop: 0 → 0.1;
3. edge drop of the fixed validation view that σ_c reads: 0.1 → 0.001.

Validation-view feature drop is 0.1 in both. Test scores are computed on the
clean graph in both.

These facts are from the reuse audit, checked on CPU for split 20 / seed 27
with no training:

* **D0:** the validation view drops 10.1% of edges (4,502 of 44,484); training
  drop is 0.
* **D1:** the validation view drops 0.085% (38 edges); training drop is
  0.001 / 0.1.
* **Both:** 1 of 27 feature columns is perturbed.

The validation score in every cell is
`forward(val_x_1, val_edge_index_1)[idx_val]` (`NIFTY.py:505-515`), as in X14.

## 3. Protocol cells

| cell | H | D | source |
|---|---|---|---|
| P00 | 200 | D0 | **reused:** NIFTY rows of `harness/results/armA_german.csv` (commit `b2b8ed9`) |
| P10 | 1000 | D0 | **new:** `pilot_tau.py --protocol armA --epochs 200 --method_epochs 1000 --dataset german --methods NIFTY` |
| P01 | 200 | D1 | **new:** `pilot_tau.py --protocol native --epochs 200 --method_epochs 200 --dataset german --methods NIFTY` |
| P11 | 1000 | D1 | **reused:** `harness/results/armB_native_NIFTY_german.csv` (commit `b2678a5`) |

Both new cells use splits 20–25, runs 0–4, seed0 27, GPU 2, and cell-level
persistence to `/tmp` with a SHA-256-verified copy to
`harness/results/x24_nifty_german_P10.csv` and `…_P01.csv`. Each run is detached
from the session.

## 4. Why P00 and P11 may be reused (audit, 2026-09-15)

Between `b2b8ed9` (P00), `b2678a5` (P11) and the HEAD that runs P10/P01, the
following were verified identical:

* dataset, split IDs 20–25 and runs 0–4;
* the run seed (`seed = 27 + run`) and M0/M1 pairing
  (`torch.manual_seed(seed*1000 + split)`);
* the M0 definition (`sim_coeff = 0.0`);
* the data loader, `algorithms/NIFTY.py` and the evaluator G_c (unchanged
  files);
* `select_min_bce`, `select_max_auc`, `ValidationHistory`, `completeness` and
  `add` (identical hashes);
* score orientation (`score>0`), and the outcome and paired-contrast formulas;
* `published("NIFTY","german")`: re-executing the old file gives an identical
  config;
* the pilot's NIFTY `train()` branch (b2678a5 = HEAD). Its additions since
  b2b8ed9 have no effect under the published config (no drop keys, no
  restore flag);
* test scoring on the clean graph, and deterministic inference (no RNG
  bundle).

The only scientific differences are H and D. Arm A's CSV lacks later diagnostic
columns (`protocol`, `method_epochs`, `rng_contract`, `n_flip`, `n_test_*`),
which is schema only. P10 and P01 run on HEAD code whose NIFTY path equals the
P11 code.

Baseline B is trained by the pipeline in P10 and P01 (at H = 200), because no
existing option skips it and no new execution path is written to save time.
**B is not used anywhere in this analysis.**

## 5. Metrics

* **Outcome coordinates.** Primary: −ΔDP. Secondary: −ΔEO. Reported alongside:
  ΔAUC.
* **Selectors.** Primary: σ_c^BCE. Robustness: σ_c^AUC. Every quantity below
  is computed under each selector separately.
* **Primary scalar analysis:** −ΔDP under σ_c^BCE.

## 6. Estimand

For each protocol cell P and each (split, run):

    τ_int(P; split, run) = Y(M1; P) − Y(M0; P)

These are the pipeline's `int_auc`, `int_ndp` and `int_neo` columns: M1 and M0
paired within the same split and run, with test outcomes from G_c at the slot
selected by the given σ_c. Then

    v_hd = τ_int(P_hd) = mean over the 30 (split, run) cells

## 7. Decomposition

    Δ_total = v11 − v00
    Δ_H     = ½[(v10 − v00) + (v11 − v01)]
    Δ_D     = ½[(v01 − v00) + (v11 − v10)]
    I_HD    = v11 − v10 − v01 + v00

The **simple effects** are also reported: `v10 − v00` and `v11 − v01` for H;
`v01 − v00` and `v11 − v10` for D.

Each contrast is formed **per (split, run) cell** from the four protocols'
cell-level τ_int, then averaged. The same (split, run) across protocols shares
data, seed and initialization pairing, which is what makes the cell the pairing
unit.

**Numerical contract:**
* `|Δ_total − (Δ_H + Δ_D)| ≤ 1e-12`, checked on the means and within every
  bootstrap replicate;
* the mean of the cell-level contrasts must equal the contrast of the means.

**Δ_H and Δ_D are protocol-level factorial contrasts, not effects of primitive
mechanisms.** Δ_D is read only as *the effect associated with switching from
the Arm-A NIFTY augmentation/validation-view configuration to the
official/native configuration*. It is never read as a causal effect of dropout
or a pure effect of training augmentation. No percentage attribution is
computed, because signs may differ and cancel.

## 8. Uncertainty

* **Table.** One cell table with one row per (split, run), 30 rows. Its columns
  are the four protocols' τ_int on each coordinate and selector, plus the
  derived cell-level contrasts.
* **Bootstrap.** The frozen `bootstrap_armA.boot()`: 10,000 replicates;
  resample the 6 splits with replacement, then the runs within each drawn split
  with replacement. Every column of a row travels together, so P00/P10/P01/P11
  and M0/M1 share the resampling indices within each replicate, preserving
  paired covariance.
* **Order of computation.** Within each replicate, v00/v10/v01/v11 are the means
  of the resampled rows, and Δ_total/Δ_H/Δ_D/I_HD come from them, which is
  identical to the mean of the resampled cell-level contrasts.
* **Generator.** A new one, `np.random.default_rng(20260916)`, is used once in
  this fixed column order, in a new analysis script
  (`harness/experiments/analyze_x24_factorial.py`). The frozen analyzers and their
  seeds are not touched.
* **Intervals.** 95% percentile.

## 9. Resolved state: frozen X22/X23 rule, unchanged

For any v_hd or contrast, over the full 30-cell design:

* sign stability ≥ **0.75**, the modal-sign fraction of the nonzero
  cell-level values (the analyzer's `sign_stability`);
* **and** |mean| ≥ **0.010**;
* **and** the 95% bootstrap interval excludes 0.

Thresholds are not changed after results. A condition below 30 complete cells
gets no resolved state.

## 10. Interpretation rules, fixed now

Applied to −ΔDP under σ_c^BCE (primary), then reported the same way for −ΔEO,
ΔAUC and σ_c^AUC.

| label | condition |
|---|---|
| **H4** weak / no reproducible factor effect | Δ_total unresolved, **or** none of Δ_H, Δ_D, I_HD resolved |
| **H3** interaction-dominant | I_HD resolved **and** \|I_HD\| ≥ max(\|Δ_H\|, \|Δ_D\|) |
| **H1** horizon-dominant | Δ_H resolved, Δ_D unresolved, I_HD unresolved, and the two H simple effects share a sign |
| **H2** configuration-dominant | Δ_D resolved, Δ_H unresolved, I_HD unresolved, and the two D simple effects share a sign |
| **both factors** | Δ_H and Δ_D resolved, and not H3 (reported as is, with the interaction value) |
| **mixed / other** | anything else, described with the values and nothing more |

* The order is H4 → H3 → H1 → H2 → both → mixed. The first match is the
  primary label.
* **H5, selector-associated explanation.** It is a diagnostic flag, never a
  primary label. It is flagged when the primary label or the resolved state of
  Δ_total differs between σ_c^BCE and σ_c^AUC, **and** the boundary-selection
  rate or median selected_epoch/H differs across the four cells. It is never
  called mediation.

Wording is limited to the NIFTY/German setting, for example: *"Within
NIFTY/German, the controlled-to-native shift in intervention attribution can be
decomposed into the effects associated with training horizon, the NIFTY
augmentation/validation-view configuration, and their interaction."*

**Prohibited readings:**
* "NIFTY's true effect is the native value";
* "all fair GNNs are protocol-dependent";
* "dropout determines fairness";
* "horizon does not matter";
* "preprocessing matters more than horizon";
* any generalization beyond this one setting.

## 11. Selector diagnostic, recorded before results

**Audit observation.** In P00, the σ_c^BCE selected epoch has median 200, the
horizon boundary, for both M1 (range 112–200) and M0 (range 106–200). σ_c^AUC
has median about 50. So a change associated with H = 200 → 1000 may combine:

* the effect of additional training trajectory, and
* the release of boundary-constrained selection at H = 200.

These are not separated. The following is reported for all four cells, per
selector, for M0 and M1:

* selected-epoch median and range;
* selected_epoch / H (median);
* **boundary-selection rate:** the fraction of cells with selected epoch == H.
  NIFTY logs epochs 0..H, so the boundary is epoch H.

Also reported: τ_int under σ_c^BCE vs σ_c^AUC per cell, and whether the
qualitative conclusion (resolved state or direction) differs. None of this is
called causal mediation.

**Per-cell configuration record.** Configured training and validation-view
drop rates come from the config of each protocol. The realized validation-view
edge-drop fraction for every (split, run) of each D level is measured by a
separate no-training probe. It rebuilds the NIFTY object on the experiment
device with the pipeline's seed sequence (`manual_seed(seed*1000+split)`, then
`manual_seed(seed)`, then construction) and records the kept-edge counts. The
probe is a diagnostic only.

## 12. Contract and stopping rules

**Before any full run:**
1. `harness/tests/test_native_phase1.py` must pass on the current HEAD. It covers
   NIFTY/German under D1 and the Arm A path.
2. A 1-cell smoke run per new cell on `/tmp` must pass, verifying:
   * `protocol` and `method_epochs` as specified, and `b_epochs` = 200;
   * NIFTY training/validation-view drop attributes on the constructed object
     equal the §2 table;
   * the recorded trajectory has H + 1 epochs;
   * rows are finite.

   Smoke rows never enter the analysis.

**After each full run:**
* CellStore reload: 30 cells, 60 keys, no duplicates, both selectors;
* no non-finite outcome; EO defined;
* runs distinct within splits;
* selected epochs within [0, H];
* SHA-256-verified copy.

**Stop and report, without re-running for a different result, if:**
* the gate or smoke fails;
* any post-run contract check fails;
* the numerical decomposition contract fails.

A run killed by infrastructure (for example, a session end) is resumed with
CellStore, as in X22, and recorded.

**No additional splits, runs, factor levels or conditions are added in any
outcome.**

## 13. Not done in this experiment

* no FMP, no FairGB/German 2×2, no new method, dataset, metric or selector;
* no separation of training edge drop, training feature drop or
  validation-view edge drop;
* no hyperparameter sweep, no extension of the native validation;
* no causal mediation claim about the selector.

This experiment does not answer:
* the independent effect of training edge drop;
* the independent effect of training feature drop;
* the independent effect of validation-view edge drop;
* any mediated effect of selector changes.

Only if Δ_D is large is decomposing D into components considered, as a separate
future decision.
