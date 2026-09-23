# X24 results: NIFTY/German 2×2 protocol-factor decomposition

**Pre-registration:** `c219c58` (X24_NIFTY_GERMAN_FACTORIAL.md). **Tooling:**
`8246efb`. Every analysis below was fixed in those commits; none was added,
dropped or re-thresholded after results. X23, the Arm A and native CSVs, and
X22 are unchanged.

## What was run

* **P00** (H = 200, D0) and **P11** (H = 1000, D1) are the frozen results,
  reused per the audit.
* **P10** (H = 1000, D0) and **P01** (H = 200, D1) are new. Each ran as
  6 splits × 5 runs through the existing pipeline, detached, with cell-level
  persistence to `/tmp`: P10 from 01:09 to 01:50, P01 to 01:59, both rc 0.
* Copies were verified by SHA-256:

  | file | SHA-256 (prefix) |
  |---|---|
  | `harness/results/x24_nifty_german_P10.csv` | `17914864…` |
  | `harness/results/x24_nifty_german_P01.csv` | `4e8e6465…` |

* B was trained by the pipeline (no existing option skips it) and is unused.

**D** is the NIFTY augmentation / validation-view configuration: a bundled
protocol-level contrast. D1 − D0 changes training edge drop 0 → 0.001,
training feature drop 0 → 0.1, and the edge drop of the fixed validation view
that σ_c reads, 0.1 → 0.001. Validation-view feature drop is 0.1 in both.

## Contracts

| check | P10 | P01 |
|---|---|---|
| CellStore reload | 30 cells, 60 keys, 0 duplicates, both selectors | same |
| protocol / method_epochs / b_epochs | armA / 1000 / 200 | native / 200 / 200 |
| seed = 27 + run; splits 20–25 × runs 0–4 | yes | yes |
| non-finite / EO undefined | 0 / 0 | 0 / 0 |
| selected epochs and code_epoch in [0, H] | yes | yes |
| runs distinct | yes | yes |
| exact-zero contrasts | 0 | 4, all under σ_c^AUC, M1 and M0 at the same epoch (P00 has 6 of the same kind) |

**On "runs distinct".** One M0 σ_c^BCE AUC value repeats in P10 split 24 and
in P01 split 22. The tied runs have different seeds, selected epochs, DP and
EO, and every (epoch, AUC, DP, EO) tuple is distinct in all four cells. The
ties are coincidental AUC equalities, not collapsed runs.

**Smoke and gate** (before the full runs):
* the native gate reruns at 127/127 on HEAD;
* the X24 construction smoke test passes;
* the no-training view probe covers all 60 cases (`x24_view_probe.csv`):
  * D0: validation-view edge drop 10.08% (9.93–10.25%), training drop 0/0;
  * D1: 0.10% (0.08–0.11%), training drop 0.001/0.1;
  * perturbed feature columns: 2.8 of 27 at both levels.

**Numerical decomposition contract:** every residual is ≤ 1e-12, for both
selectors and all coordinates, in the means and in every bootstrap replicate.
Reused P00 and P11 reproduce X23 exactly (−0.0065, −0.1510).

## Table A: four protocol cells (σ_c^BCE)

Means over 30 cells; 95% paired hierarchical bootstrap with 10,000 replicates;
s = sign stability; R/u = resolved/unresolved under the frozen rule.

| cell | H | D | τ_int ΔAUC | τ_int −ΔDP | τ_int −ΔEO | −ΔDP resolved | BCE selected epoch median, M1 / M0 |
|---|---|---|---|---|---|---|---|
| P00 | 200 | D0 | −0.031 [−0.056, −0.007] s0.67 u | −0.007 [−0.043, +0.032] s0.60 | +0.004 [−0.035, +0.045] s0.57 u | u | 200 / 200 |
| P10 | 1000 | D0 | +0.016 [−0.004, +0.033] s0.60 u | −0.061 [−0.129, +0.005] s0.70 | −0.045 [−0.121, +0.026] s0.60 u | u | 904 / 450 |
| P01 | 200 | D1 | −0.023 [−0.045, −0.005] s0.63 u | +0.029 [−0.006, +0.074] s0.63 | +0.030 [−0.001, +0.065] s0.66 u | u | 197.5 / 196 |
| P11 | 1000 | D1 | +0.051 [+0.028, +0.072] s0.87 R | **−0.151 [−0.224, −0.082] s0.93** | −0.122 [−0.187, −0.065] s0.93 R | **R** | 947 / 692.5 |

Only P11 has a resolved τ_int on −ΔDP. Neither single-factor switch from P00
produces one: horizon alone (P10) gives −0.061, unresolved, and configuration
alone (P01) gives +0.029, unresolved, in the opposite direction.

## Table B: attribution-shift decomposition (σ_c^BCE)

| metric | Δ_total | Δ_H | Δ_D | I_HD | pre-registered label |
|---|---|---|---|---|---|
| **−ΔDP** | −0.145 [−0.227, −0.061] s0.83 **R** | −0.117 [−0.184, −0.055] s0.90 **R** | −0.028 [−0.085, +0.018] s0.57 u | −0.126 [−0.217, −0.045] s0.73 u | **H1 horizon-dominant** |
| −ΔEO | −0.126 [−0.193, −0.060] s0.90 R | −0.101 [−0.165, −0.042] s0.87 R | −0.026 [−0.071, +0.014] s0.57 u | −0.102 [−0.199, −0.013] s0.67 u | H1 horizon-dominant |
| ΔAUC | +0.081 [+0.048, +0.115] s0.83 R | +0.060 [+0.034, +0.088] s0.83 R | +0.021 [+0.006, +0.038] s0.77 R | +0.028 [−0.000, +0.057] s0.73 u | both factors |

Simple effects on −ΔDP (σ_c^BCE):

| contrast | value |
|---|---|
| H given D0 (v10 − v00) | −0.054 [−0.140, +0.026] s0.67 u |
| H given D1 (v11 − v01) | **−0.180 [−0.254, −0.109] s0.90 R** |
| D given H = 200 (v01 − v00) | **+0.035 [+0.006, +0.072] s0.77 R** |
| D given H = 1000 (v11 − v10) | −0.090 [−0.190, −0.013] s0.60 u |

### Reading, strictly by the pre-registered rules

* **Primary label: H1 (horizon-dominant).** Δ_total is resolved. Δ_H (−0.117)
  is resolved. Δ_D (−0.028) is not. I_HD is unresolved. Both horizon simple
  effects are negative.
* **The rule's outcome on I_HD rests on one criterion.** I_HD = −0.126: its
  95% interval excludes 0, |I_HD| is about the size of Δ_H, and the rule
  records it unresolved **only** because sign stability is 0.73, below the
  frozen 0.75. The threshold is not changed. The label stays H1, and this
  proximity is reported with it. −ΔEO shows the same pattern (I_HD interval
  excludes 0, s0.67).
* **The simple effects are not homogeneous.**
  * The horizon change is resolved and large under D1 (−0.180), and small and
    unresolved under D0 (−0.054).
  * The configuration change is resolved and *fairness-improving* at H = 200
    (+0.035), and negative but unresolved at H = 1000 (−0.090).

  These are the descriptive values. They are not re-labelled as an interaction
  finding.
* **Within NIFTY/German, under σ_c^BCE:** the controlled-to-native shift in
  intervention attribution on −ΔDP decomposes into a resolved average
  association with the training horizon (Δ_H = −0.117), an unresolved average
  association with the NIFTY augmentation/validation-view configuration
  (Δ_D = −0.028), and an interaction term (I_HD = −0.126) that is large but
  falls just short of the frozen resolved rule. The harmful native attribution
  appears only where both factors are at their native levels.

## Table C: selector robustness (τ_int, −ΔDP)

| H | D | under σ_c^BCE | under σ_c^AUC | qualitative conclusion changed? |
|---|---|---|---|---|
| 200 | D0 (P00) | −0.007 u | **+0.024 [+0.004, +0.056] s0.86 R** | **yes** |
| 1000 | D0 (P10) | −0.061 u | −0.020 u | no |
| 200 | D1 (P01) | +0.029 u | +0.010 u | no |
| 1000 | D1 (P11) | **−0.151 R** | −0.020 [−0.092, +0.050] s0.57 u | **yes** |
| Δ_total | | −0.145 R | −0.044 u | yes |
| Δ_H | | −0.117 R | −0.037 u | yes |
| Δ_D | | −0.028 u | −0.008 u | no |
| I_HD | | −0.126 u | +0.014 u | yes (direction) |

**Under σ_c^AUC every coordinate is labelled H4** (weak / no reproducible
factor effect). The resolved harmful native attribution (P11) is not reproduced
under σ_c^AUC. P00, unresolved under σ_c^BCE, is resolved and slightly
fairness-improving under σ_c^AUC. Only one simple effect is resolved under
σ_c^AUC: D given H = 200, −0.015 on −ΔDP and −0.017 on −ΔEO.

## Selector / boundary diagnostic (X24 §11)

| cell | selector | M1 epoch median [range] · epoch/H · boundary rate | M0 epoch median [range] · epoch/H · boundary rate |
|---|---|---|---|
| P00 | BCE | 200 [112, 200] · 1.00 · **0.63** | 200 [106, 200] · 1.00 · **0.63** |
| P10 | BCE | 904 [354, 1000] · 0.90 · 0.07 | 450 [238, 1000] · 0.45 · 0.40 |
| P01 | BCE | 197.5 [183, 200] · 0.99 · 0.37 | 196 [102, 200] · 0.98 · 0.07 |
| P11 | BCE | 947 [368, 1000] · 0.95 · 0.03 | 692.5 [238, 999] · 0.69 · 0.00 |
| P00 | AUC | 50.5 [0, 200] · 0.25 · 0.07 | 52.5 [0, 200] · 0.26 · 0.17 |
| P10 | AUC | 882.5 [244, 1000] · 0.88 · 0.03 | 857.5 [218, 1000] · 0.86 · 0.07 |
| P01 | AUC | 41.5 [0, 197] · 0.21 · 0.00 | 37.5 [0, 199] · 0.19 · 0.00 |
| P11 | AUC | 969.5 [226, 997] · 0.97 · 0.00 | 706.5 [203, 996] · 0.71 · 0.00 |

**H5 (selector-associated diagnostic): flagged on −ΔDP, −ΔEO and ΔAUC.** Labels
or resolved states differ between selectors, and boundary rates and median
epoch/H differ across cells. Descriptive observations only:

* At H = 200 under σ_c^BCE, M1 and M0 are selected at or near the horizon
  boundary: boundary rate 0.63 in P00, and 0.37 / 0.07 for M1 / M0 in P01.
* At H = 1000 under σ_c^BCE, M1 is selected late (median epoch/H 0.90–0.95)
  while M0 is selected markedly earlier (0.45 in P10, 0.69 in P11).
  σ_c^AUC does not show that M1/M0 separation in P10 (0.88 / 0.86).

**This is not causal mediation**, and it does not separate "additional training
trajectory" from "release of boundary-constrained selection". X24 §11
registered the two as confounded within H.

## Figures

* **Main:** `harness/results/figures/x24_nifty_german_interaction_dp_bce.{png,pdf}`.
  τ_int(−ΔDP) under σ_c^BCE against H, one line per D, with 95% bootstrap bars;
  filled markers = resolved.
* **Supplementary:**
  `harness/results/figures/x24_nifty_german_interaction_supplementary.{png,pdf}`.
  Rows are σ_c^BCE / σ_c^AUC; columns are −ΔDP, −ΔEO, ΔAUC.

## Statement of record, and limits

* **Scope.** One setting: NIFTY on German, 6 splits × 5 runs.
* **Primary finding, σ_c^BCE.** The controlled-to-native attribution shift is
  associated mainly with the training horizon, on average over D (H1).
  * The harmful native attribution is resolved only when H and D are both at
    their native levels.
  * The interaction term is of comparable size to Δ_H, and is unresolved under
    the frozen rule only through sign stability.
* **Selector dependence.** Under σ_c^AUC none of the factor contrasts is
  resolved, and the native harmful attribution is not reproduced.
* **Δ_D** is the effect associated with switching from the Arm-A NIFTY
  augmentation/validation-view configuration to the official/native one. It is
  not an effect of dropout, of training augmentation alone, or of the
  validation view alone.
* **Not answered here:** the independent effects of training edge drop,
  training feature drop or validation-view edge drop; any mediated effect of
  selection; anything about other methods or datasets.
* **Not claimed:**
  * that NIFTY's true effect is the native value;
  * that horizon matters more than configuration in general;
  * that fair GNNs are protocol-dependent;
  * any generalization beyond this setting.

The pre-registered follow-up condition for decomposing D ("only if Δ_D is
large") is not met on −ΔDP under either selector: Δ_D is unresolved in both.
