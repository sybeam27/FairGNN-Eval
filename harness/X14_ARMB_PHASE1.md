# Arm B, Phase 1: native-configuration validation

Recorded before any Phase 1 cell runs.

Arm A found a pattern; Phase 1 asks whether it survives the official
configuration. It is an external-validity check. The provenance audit (X13)
found three substantial implementation differences:

* FairVGNN normalizes german, which upstream does not;
* FairVGNN credit has its own training loop upstream;
* NIFTY's local class zeroes its drop rates.

They are not read as "the existing code is broken". They show why this phase
is needed: **attribution depends on which implementation and training protocol
is instantiated, so native-configuration validation is necessary.**

## Question

Does the rest-of-package vs intervention attribution structure hold when the
**method-side** training configuration moves from Arm A's to the
official/native one?

## Cells, in run order

FairGB/bail is last because it is the most expensive.

| # | cell | official source | native horizon | preprocessing | training difference vs Arm A | published selector | M0 definition | expected cost |
|---|---|---|---|---|---|---|---|---|
| 1 | NIFTY / german | chirag126/nifty @ `6c270c5`, README | 1000 (Arm A 200) | none (same as Arm A) | training drop rates restored: edge 0.001/0.001, feature 0.1/0.1 (Arm A trained at 0); validation-view edge rate 0.001 (Arm A 0.1) | argmin(val_c_loss + val_s_loss) | same native config, sim_coeff = 0 | about 1.1 h |
| 2 | FairGB / german | vendored FairGB-main, `run.sh` | 1500 (Arm A 200) | **none** (Arm A min-max) | horizon and normalization only | auc + F1 + acc - 4(parity + equality) | same native config, CAL off + CNM off | about 1.8 h |
| 3 | FairVGNN / german | YuWVandy/FairVGNN @ `938f2e8`, `run_german.sh` | 200 (same) | **none** (Arm A normalized) | normalization only | auc + F1 + acc - 1(parity + equality), floor 0 | same native config, f_mask = no, weight_clip = no | about 1-2.3 h |
| 4 | FairVGNN / bail | YuWVandy/FairVGNN @ `938f2e8`, `run_bail.sh` | 300 (Arm A 200) | min-max, sensitive column kept (same) | horizon only | auc + F1 + acc - 1(parity + equality), floor 0 | same native config, f_mask = no, weight_clip = no | about 2.3 h |
| 5 | FairGB / bail | vendored FairGB-main, `run.sh` | 1500 (Arm A 200) | min-max, sensitive column kept (same) | horizon only | auc + F1 + acc - 2(parity + equality) | same native config, CAL off + CNM off | about 12.9 h |

Total upper estimate is about 20 h. Past estimates of this kind ran 1.1-2.4x
pessimistic.

Design: 6 splits (20-25) x 5 runs (0-4), the same cells as Arm A.

## Not run

* **FairVGNN / credit.** The authors train it with `fairvgnn_credit.py`, a
  different loop. The shared wrapper is not modified to approximate it. If it
  is ever needed, it becomes a separate native adapter that mirrors the
  official script faithfully, with provenance and regression tests.
  `algorithms/FairVGNN.py` is not edited. Deciding on it waits until Phase 1
  shows whether FairVGNN is highly sensitive to its native configuration.
* **FairGB / credit.** Held under the budget rule fixed in X12: it runs only
  if FairGB/german or FairGB/bail reverses `|tau_base| > |tau_int|`, stably
  reverses the fairness-axis `tau_int`, or materially flips resolved vs
  unresolved.

## Baseline reference and notation

Phase 1 reuses Arm A's baseline reference, **B_200**. B has no native protocol,
so these names are **not** used:

* `tau_base^native`
* `tau_pkg^native`

The names used are:

* `tau_int^native = Y(M1^native) - Y(M0^native)`
* `tau_base^fixed-ref = Y(M0^native) - Y(B_200)`
* `tau_pkg^fixed-ref = Y(M1^native) - Y(B_200)`

The reference is identical in both arms. So
`tau_base^fixed-ref,ArmB - tau_base^ArmA` comes from the M0 configuration
changing, never from the baseline.

## What stays fixed

* **M0/M1 isolation.**
  * M1 = native configuration + intervention ON.
  * M0 = the same native configuration + intervention OFF.
  * Nothing is retuned when the intervention is switched off.
  * An official ablation that retunes, as FairVGNN's does, is never used as M0.
* **Evaluation contract of corrected Arm A.**
  * sigma_c^BCE primary, sigma_c^AUC robustness
  * G_c
  * paired split/run
  * same-process recording
  * for FairVGNN, the full RNG replay bundle and a fixed xi_eval
  * restore fidelity, selector replay, test isolation

## Comparison per cell

For each cell, on `[dAUC, -dDP]` primary and `[dAUC, -dEO]` secondary, the
comparisons are `tau_int^ArmA` vs `tau_int^native` and `tau_base^ArmA` vs
`tau_base^fixed-ref`. What is read is the qualitative structure, not whether
the numbers agree:

* whether `|tau_base| > |tau_int|` still holds;
* whether the fairness-axis intervention direction still holds;
* unresolved vs resolved;
* the qualitative position of selector sensitivity.

## Harness implementation, no external source edited

* `core/published_config.native_config()` starts from `published()` (Arm A,
  untouched) and applies only what X13 established. Every value is parsed from
  an artifact:
  * the horizon;
  * NIFTY's drop rates, from the official command;
  * FairGB's normalization, from the `data_utils.py` rule;
  * FairVGNN's german exception, from the `dataset.py` rule.
* `pilot_tau.py --protocol native`: B stays at `--epochs 200`, and methods run
  at their native horizon. Every row records `protocol`, `method_epochs` and
  `b_epochs`.
* **NIFTY:** official rates are passed at construction (validation views), and
  the training attributes the class zeroes are set back after construction.
* **FairGB german:** `get_dataset(..., feature_normalize=False)`.
* **FairVGNN german:** no harness pre-normalization, and the wrapper's
  `feature_norm` is the identity for the duration of `fit`, restored in
  `finally` alongside the RNG-capture wrapper.

## Arm A naming, stated once

Arm A's NIFTY results are an **audited local implementation under the
controlled Arm A protocol**, never "official NIFTY". The zeroed drop rates are
a provenance limitation of Arm A. Arm A is not re-run. NIFTY/german here tests
whether that difference changes the attribution conclusion.

## Pre-run gate, CUDA (`harness/tests/test_native_phase1.py`), and what it exposed

Final run: **127/127 pass**, on split 20, a short horizon of 12, all five cells,
both arms, both common slots. Getting there surfaced three facts. None is a
restore failure, and all are recorded rather than smoothed over.

**1. NIFTY's validation scores come from its augmented view.** `NIFTY.fit`
records validation output as
`forwarding_predict(forward(val_x_1, val_edge_index_1))`: a fixed view with
dropped edges and noised features. NIFTY's own code does the same. Test scoring
uses the clean graph. So a restored NIFTY checkpoint can only be checked against
that view: scored there it matches the record, while the clean graph differs by
50-300. `score_fn(..., replay=True)` now scores the view.

This is also a protocol property. **For NIFTY, sigma_c reads validation scores
from the augmented view while the test outcome uses the clean graph.** It held
in Arm A too. M0 and M1 share it, so the contrast is paired. The native drop
rates change that view (edge rate 0.001 vs 0.1), so moving to native NIFTY
changes the selector's input as well.

**2. CUDA float noise sets the replay floor.** An absolute 1e-5 bound is below
the noise the device produces. Replaying the *same* restored checkpoint
three times, with no random number drawn, gives:

* FairVGNN/german M0, unnormalized features: spread up to 1.1e-4
* NIFTY/german, logits near 4000: spread up to 9.8e-4 (relative about 1e-7)

The replay is therefore required to be within `max(1e-5, 4 x replay spread)` of
the record. The quantities a checkpoint is used for are required to match
exactly: hard predictions, DP and EO.

**3. AUC has a resolution.** Validation scores carry 50-68 near-ties below 1e-6.
On FairGB/bail a 1.2e-7 wobble moves AUC by half a pair to one pair (one pair =
1.91e-7), flaky across reruns, while hard predictions, DP and EO stay exactly
equal. AUC is required to match within `2 / (n_pos * n_neg)`. Final run: at most
2.87e-7 against a bound of 3.83e-7 on bail; 0 on german.

**Recorded, not hidden.** The criterion moved twice, from absolute 1e-5 to the
noise floor, and from exact AUC to AUC resolution. Each move followed a
diagnostic showing the residual was device noise and not restoration:

* no-RNG M0 replays differ from each other by as much as from the record;
* hard predictions never change;
* only AUC moves, and only by pair swaps.

The native changes are all confirmed in effect, and the Arm A path is
unchanged:

* official NIFTY drop rates at fit;
* FairGB/german and FairVGNN/german train on unnormalized features;
* FairVGNN/bail and FairGB/bail stay normalized;
* the wrapper's `feature_norm` is restored after fit;
* `published()` carries no native keys, Arm A FairVGNN/german still normalizes,
  and Arm A NIFTY still trains at the class's zeroed rates.

## Phase 1 launched

After the gate, in the fixed order, from the code this document is committed
with: `harness/results/armB_native_{method}_{dataset}.csv`.
