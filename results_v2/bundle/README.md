# final_results

Paper-facing result bundle. Regenerate everything with one command:

    python harness/experiments/build_results.py

## What a cell is

One cell is one method x dataset x configuration x protocol, evaluated on
6 splits x 5 runs = 30 matched units. Within a unit, three arms share the
split, the seed and the initialisation:

    B       a common GNN baseline
    M^-I    the method with its claimed fairness intervention switched off
    M^+I    the method at its repository configuration

    tau_nonint = Y(M^-I) - Y(B)     everything the package does apart from the
                                    claimed intervention
    tau_I      = Y(M^+I) - Y(M^-I)  the claimed intervention itself
    tau_pkg    = Y(M^+I) - Y(B)     the package as a whole (= tau_nonint + tau_I)

Coordinates are oriented so that larger is better: `dAUC`, `negDP`, `negEO`.
Uncertainty is a paired hierarchical bootstrap (10,000 replicates, splits
resampled first and then runs within a drawn split, each matched unit carried
whole). An estimate is **resolved** only if sign stability >= 0.75,
|mean| >= 0.01, and the 95% percentile interval excludes zero.

## The evaluation sets, and why they are never pooled

| set | cells | what it is |
|---|---|---|
| primary controlled | 36 | one configuration per method x dataset, `count_in_primary_summary = true` |
| configuration robustness, controlled | 26 | further published rows, alternative backbones or alternative intervention budgets of a method already in the primary set |
| task-adapted, controlled | 3 | a method released for another task (FnRGNN, node regression) whose missing training loop the harness completed; reported on its own, `count_in_primary_summary = false` |
| systematic native comparisons | 16 | every configuration of a method whose published procedure is leakage-free, re-run under that published procedure |
| targeted native validations | 7 | a validation set fixed in advance to probe protocol sensitivity on already-frozen controlled cells |

The two native families answer different questions and were sampled on
different rationales -- one sweeps a method exhaustively, the other targets
specific frozen cells -- so **they are never combined into a single success
rate**, and neither is added to a primary count. Robustness configurations share
data, splits and baseline with the primary configuration of the same dataset, so
they are not independent cells either.

Only a published procedure that selects without touching the test split is
admissible as a native protocol; methods whose published selection reads test
data have no native row at all.

## Schema

* `configuration_role` — `primary` or `robustness`.
* `protocol` — `controlled` (one common horizon and selector for every arm) or
  `native` (the method at its own published horizon; the checkpoint selector stays the controlled
  validation-BCE rule on both sides, and the method's own published rule is recorded but unused).
* `native_evaluation_role` — empty for controlled rows, else `systematic` or
  `targeted`.
* `count_in_primary_summary` — true only for primary controlled cells of methods
  run as released (never for a task-adapted method).
* `task_adaptation` — empty, or what the harness had to change so a method
  released for another task could be evaluated here.
* `baseline_reference_protocol` — `same_protocol` when B ran under the same
  protocol as the method arms, `controlled_fixed_reference` when the row reuses
  the fixed controlled baseline trained at H = 200.
* `backbone` — the message-passing encoder the cell trained (`GCN`, `GIN`,
  `SAGE` or `GAT`), never a configuration label; what distinguishes two
  configurations of one method (another published row, an encoder, a budget)
  lives in `configuration`.

**What a native row's baseline means.** Every native row reuses the fixed
controlled reference B trained at H = 200; the native horizon applies to the
method arms. The scientific target of a controlled-vs-native comparison is
`tau_I = Y(M^+I) - Y(M^-I)`, which is a contrast between two arms that ran under
the same protocol and is therefore unaffected. `tau_nonint` and `tau_pkg` on a
native row are measured against that fixed reference and must not be read as a
same-protocol B -> M contrast. The algebraic identity
`tau_pkg = tau_nonint + tau_I` still holds, because all three contrasts use the
same B.

## How the result files are organised

`experiment_index.csv` lists every section file with what it holds fixed and
what it varies. The families never mix:

| file | section | held fixed | varied |
|---|---|---|---|
| `1_main_package_vs_intervention.csv` | 1. Main | controlled protocol, primary configuration | method vs common baseline, split into `tau_nonint` and `tau_I` |
| `1b_main_task_adapted.csv` | 1b. Main, task-adapted | controlled protocol | as section 1 for FnRGNN, with `task_adaptation` stating what the harness completed |
| `1c_main_released_task_regression.csv` | 1c. Main, released task | FnRGNN on node regression (its own task, MSE), same split and sensitive attribute; B = the common GCN trained with MSE | `tau_nonint`, `tau_I`, `tau_pkg` on regression metrics: `negMSE`, `negMeanGap`, `negWD` (standardised target units); compare only within this file |
| `2_configuration_variation.csv` | 2. Configuration variation | controlled protocol | the configuration: FairSIN GCN -> GIN, SAGE; FairVGNN's further published rows (GCN-spmm, GIN, SAGE); BeMap GCN -> GAT; FairGNN's upstream GCN and GAT rows; BIND 1% -> 10%. `varied_factor` names it |
| `3a_protocol_selector_bce_vs_auc.csv` | 3a. Protocol variation: selector | configuration, horizon, data, units | only the checkpoint selector (validation BCE vs validation AUC), for every controlled cell; `configuration_role` separates primary from robustness rows |
| `3b_protocol_native_horizon_selector.csv` | 3b. Protocol variation: published horizon | configuration, data and the checkpoint selector | the training horizon, set to the method's own published value, and nothing else; validation-BCE selection is retained on both sides, so this is a horizon-only contrast |
| `3c_protocol_native_published_procedure.csv` | 3c. Protocol variation: published procedure | the intervention's configuration and the checkpoint selector | the training horizon plus the preprocessing or training-loop details the published procedure prescribes; validation-BCE selection is retained, so the selector is not among the changed factors; `factors_changed` lists what is |
| `4_mechanistic_case_study.csv` | 4. Mechanistic case study | the frozen cell | selection-support decompositions (NIFTY, FairGB) |
| `5_component_case_study_FMP.csv` | 5. Component-level case study | FMP on its own split and horizon, the baseline trained under the same setting | four stages, one component at a time: B -> base -> +propagation -> +fairness |
| `model_dataset_feasibility.csv` | coverage | -- | every method x dataset the released code configures, by role |

3a and 3b change only the protocol and can be read as protocol effects. 3c
changes the protocol together with what the published procedure bundles with it
(for example FairVGNN on german and FairGB on german switch feature
normalisation off, NIFTY restores its published drop rates, FairVGNN on credit
uses a separate credit training loop), so it is reported as the effect of the
published procedure as a whole. Which of the two a native row belongs to is
decided from the configuration interpreters alone, before any result is read.

## FnRGNN, two views

FnRGNN was released for node regression and ships no training loop. It is
reported twice, never pooled with any other set:

* `1b` evaluates it on the common classification task, with the loss switched
  to BCE;
* `1c` evaluates it on its own regression task, with the released MSE loss.

Both use the same split, the same released configuration and the same
harness-completed loop (training-node mask, mmd_sample = 500, H = 200). `1c`
measures regression quality and fairness in standardised target units, so its
numbers are comparable only with each other.

## Files

| file | what it holds |
|---|---|
| `cell_results.csv` | one row per completed cell: all three estimands on all three coordinates, with intervals, sign stability, resolution, the AUC-selector robustness estimate, the role metadata above, and any caveat |
| `method_configurations.csv` | what each cell actually ran: intervention, both arms, horizon, selector, hyperparameters, preprocessing, source and provenance. Every result cell has exactly one row here, plus the common baseline |
| `coverage.csv` | every cell with units done / required and status; pending and partial cells appear here and nowhere else |

## Reading the caveats

Some cells carry an implementation caveat that belongs beside the number: a
released asset that does not match its own specification, an intervention whose
repository default is very small, a budget expressed as a fraction of the
training set, a wrapper whose parameter routing had to be repaired from its own
stored values, or a native row's fixed-reference baseline. `cell_results.csv`
and `method_configurations.csv` both carry these in a `caveat` column; none of
them is a post-hoc adjustment to a result.

## Status

36 primary controlled, 3 task-adapted controlled,
26 configuration robustness, 16 systematic native, 7 targeted native,
0 pending or partial.
Pending and partial cells are never summarised scientifically.
