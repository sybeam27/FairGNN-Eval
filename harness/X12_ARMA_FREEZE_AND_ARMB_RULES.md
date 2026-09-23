# Arm A frozen; Arm B staged, with its budget rule fixed before any result

Recorded before any Arm B cell has run.

## Arm A is the primary audit result

Corrected Arm A (commit `66966fb`) is frozen: 720 rows, FairGNN / NIFTY / FairGB /
FairVGNN x german / bail / credit x 6 splits x 5 runs, H_audit = 200, the
FairVGNN rows replaced under the X11 RNG contract.

Its scientific claims do not change because of Arm B.

* Primary statement: *Under the controlled H = 200 audit, the magnitude of the
  rest-of-package effect exceeds that of the claimed intervention in 10/12
  method-dataset settings on DP, with the same pattern on EO.*
* In the paper this is phrased as **"the rest-of-package component often
  dominates the intervention component"**, not "the package effect is larger
  than the intervention".
* The fairness-axis intervention effect had a bootstrap interval excluding 0
  in only **2/12** settings, and **11/12** descriptive attribution
  classifications were unresolved. This stays.
* Selector sensitivity stays a method- and dataset-conditional result and is
  not generalized.

## Arm B: purpose

Arm B does not produce a method ranking or a new attribution taxonomy. Its one
question is whether the pattern observed at H_audit = 200 -- rest-of-package
effect > intervention effect, and the intervention-attribution pattern --
survives each method's official/native training configuration and horizon.

## Arm B: what stays fixed

* **Attribution control.** M1 = native configuration + intervention ON; M0 =
  the *same* native configuration + intervention OFF. An official ablation that
  retunes hyperparameters while removing the intervention (FairVGNN's does) is
  never used as M0.
* **Two views, never mixed.**
  * *Native-selector package view*: the full system chosen by its official
    selector, outcome recomputed by G_c.
  * *Native-config attribution view*: native training configuration and native
    horizon, read at sigma_c^BCE and sigma_c^AUC, giving tau_base^native,
    tau_int^native and tau_pkg^native.
* **RNG, seed and evaluation contract**, identical to corrected Arm A: run seed
  passed; for FairVGNN the full Python/NumPy/Torch CPU/CUDA replay bundle, a
  cell-fixed xi_eval, and CUDA restore fidelity. Applying an official training
  configuration does not revert the evaluation contract.
* No new method, metric, selector or dataset.

## Arm B: staging

Phase 1 runs in this order, skipping any cell classified `native-reusable`:

1. NIFTY / german
2. FairGB / bail
3. FairGB / german
4. FairVGNN / bail -- only if its Arm A configuration actually differs from
   native

FairGB / credit does not run in Phase 1.

A candidate is `native-reusable` only if every training field and the horizon
already match official/native in the corrected Arm A run. That is decided from
a field-by-field diff table, never assumed.

## Phase 1 comparison

For each Phase 1 cell, on `[dAUC, -dDP]` primary and `[dAUC, -dEO]` secondary,
tau_base^ArmA is compared with tau_base^native and tau_int^ArmA with
tau_int^native. The question is not whether the numbers agree. It is whether
the qualitative structure holds:

* whether `|tau_base| > |tau_int|`;
* the direction of the intervention's fairness effect;
* whether the intervention effect is resolved or unresolved;
* the rough position of selector sensitivity.

## Sequential budget rule for FairGB / credit, fixed now

FairGB / credit (native horizon 2000) is the costliest cell. It runs **if and
only if** at least one of the following is observed in FairGB / german or
FairGB / bail at Phase 1:

1. the `|tau_base| > |tau_int|` relation reverses relative to Arm A;
2. the fairness-axis `tau_int` direction reverses **stably**;
3. the intervention effect's state changes materially between unresolved and
   resolved.

If both cheaper FairGB cells keep Arm A's qualitative attribution pattern,
FairGB / credit is held as a high-cost confirmatory cell and not run.

The criteria are applied with the instruments already frozen for Arm A, not
invented after Phase 1:

* "resolved" means the descriptive rule (sign stability >= 0.75 and |mean|
  >= 0.010) together with the hierarchical bootstrap's 95% interval for
  tau_int on that coordinate;
* a direction change is "stable" only if the native tau_int is itself resolved
  in the opposite direction;
* "materially" means the resolved/unresolved state flips on the primary
  fairness coordinate, `-dDP`.
