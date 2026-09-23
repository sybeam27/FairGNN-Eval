# X25 results: NIFTY/German trajectory–selection decomposition

**Pre-registration:** `2547bf3`. **Tooling:** `4684cf1`. **G3 amendment:**
`326f04f` (narrow numerical-replay amendment, frozen before any decomposition
was computed). **Original G3: FAILED** and recorded in `X25_STOP_REPORT.md`
(`8f57818`, `bd2edad`); it is not rewritten as a pass.

X22, X23, X24 and every frozen CSV are untouched. No estimand, epoch grid,
selector, bootstrap or resolved rule was changed.

## 1. What ran

| step | outcome |
|---|---|
| read-only audit | no STOP condition; per-epoch test outcomes are not recoverable, so a minimal rerun was required |
| R10 (H=1000, D0) | 02:52–03:39, rc 0, 30/30 cells, 60 valid trajectory files |
| R11 (H=1000, D1) | 03:39–04:28, rc 0, 30/30 cells, 60 valid trajectory files |
| preservation | metadata, seeds, protocol, `method_epochs`, `b_epochs` verified; copies checksum-identical to `/tmp` |
| replay-noise diagnostic | 24 independent slot evaluations, measured **before** the amendment |
| amended G3 | 240 comparisons: **239 exact, 1 boundary-flip, 0 stops** |
| reproduction gate | **PASS**, 12/12 in-interval plus the headline |

**Gates:** G1, G2, G3(smoke), G5 passed before the rerun; G4 passed on HEAD
(native 127/127, X24 smoke 26/26).

## 2. The amended gate, and why

The original G3 required exact DP/EO agreement between the stored in-training
test scores and the pipeline's restored checkpoint. It failed in **1 of 240**
comparisons (R10 s20 r3 M1 σ_c^BCE, epoch 988: |ΔDP| 6.41e-3, |ΔEO| 8.40e-3)
while its selected epoch matched exactly. One test node sits **4.52e-5** from
the `score > 0` threshold, and flipping it alone reproduces the restored DP and
EO to 5.6e-17.

The envelope was measured independently first: 24 slot evaluations, replay
spread max **4.272e-4** (median 1.450e-4), stored-vs-replayed max 8.240e-4,
and **zero** sign flips over 6,000 node comparisons. The project's existing form
(X14, X20) gives **bound = max(1e-5, 4 × 4.272e-4) = 1.709e-3**. The failing
node is 37× inside that bound.

> The original restore-fidelity gate required exact agreement in thresholded
> DP/EO and failed in 1 of 240 comparisons because a single near-zero test score
> changed sign under CUDA replay. Before inspecting the scientific
> decomposition, we amended the gate to permit only decision-boundary flips
> demonstrably bounded by independently measured replay noise and fully
> accounting for the metric discrepancy.

**Sensitivity (amendment §7).** Re-running the entire analysis with the replayed
value substituted at that one comparison changes nothing material: **max
|difference| 4.7e-4** across all reported quantities, **no resolved-state
change**, and **identical** pre-registered labels.

## 3. Reproduction gate against frozen P10/P11

All 12 checks pass: the rerun full-support τ_int mean lies inside the frozen
X24 95% interval for every (D, selector, coordinate). The headline reproduces:
R10 unresolved (−0.0429 vs frozen −0.0608), R11 resolved and negative (−0.1480
vs frozen −0.1510). **0/30 cells are bitwise identical** in either run, exactly
as the nondeterminism audit predicted.

## 4. Table 1 — selector-support decomposition (same H=1000 trajectory)

τ = mean over 30 cells; 95% paired hierarchical bootstrap, 10,000 replicates;
s = sign stability; R/u = resolved/unresolved under the frozen rule.

**−ΔDP (primary)**

| D | selector | τ cap ≤ 200 | τ full ≤ 1000 | S = full − cap |
|---|---|---|---|---|
| D0 | BCE | −0.0113 [−0.0472, +0.0287] s0.60 u | −0.0429 [−0.1172, +0.0317] s0.67 u | −0.0316 [−0.1255, +0.0572] s0.53 u |
| D0 | AUC | +0.0244 [+0.0041, +0.0584] s0.86 R | −0.0286 [−0.1101, +0.0550] s0.57 u | −0.0530 [−0.1462, +0.0390] s0.60 u |
| **D1** | **BCE** | **+0.0288 [−0.0067, +0.0726] s0.60 u** | **−0.1480 [−0.2207, −0.0748] s0.90 R** | **−0.1768 [−0.2471, −0.1015] s0.93 R** |
| D1 | AUC | +0.0097 [−0.0013, +0.0268] s0.71 u | −0.0290 [−0.0961, +0.0352] s0.50 u | −0.0386 [−0.1072, +0.0284] s0.53 u |

* **−ΔEO** mirrors it: D1 BCE S = −0.1520 [−0.2155, −0.0881] R; everything else
  unresolved.
* **ΔAUC:** D1 BCE S = +0.0722 [+0.0435, +0.0981] R; D0 and both AUC-selector
  rows unresolved.

## 5. Table 2 — bridge to the frozen H = 200 runs

T = cap200 − H200, H = full − H200 = S + T (identity holds ≤ 1e-12 per cell and
in every replicate).

**−ΔDP**

| D | selector | H200 | cap200 | T | H | X24 frozen simple effect |
|---|---|---|---|---|---|---|
| D0 | BCE | −0.0065 | −0.0113 | −0.0049 [−0.0204, +0.0066] u | −0.0365 [−0.1288, +0.0500] u | −0.0544 |
| D0 | AUC | +0.0242 | +0.0244 | +0.0002 [−0.0010, +0.0019] u | −0.0528 [−0.1462, +0.0403] u | −0.0439 |
| **D1** | **BCE** | +0.0288 | +0.0288 | **−0.0000 [−0.0071, +0.0064] u** | **−0.1768 [−0.2488, −0.1027] R** | −0.1799 |
| D1 | AUC | +0.0095 | +0.0097 | +0.0002 [−0.0000, +0.0009] u | −0.0384 [−0.1068, +0.0285] u | −0.0294 |

**Prefix check (X25 §6): not an exact match.** Per cell, D1 BCE: selected epochs
equal in 24/30, epochs + DP/EO equal in 17/30, fully exact (≤ 1e-9 on every
outcome) in 1/30, median max|difference| 2.29e-4. D0 BCE: 24/30, 18/30, 3/30,
4.76e-4. Under σ_c^AUC the agreement is closer (D1: 25/30, 24/30, 18/30, median
0).

Consequently, and as pre-registered:

* **S is not called a "pure selector effect"**, and the bridge is not called
  "clean";
* **T is a prefix/run difference including cross-process GPU nondeterminism.**
  Its mean is ≈ 0 and unresolved in every case, so the H = 1000 prefix
  reproduces the H = 200 run *in aggregate* even though individual cells differ
  by device noise.

## 6. Table 3 — fixed-epoch trajectory (no selection at all)

τ_int(−ΔDP) at the pre-registered grid, descriptive, no per-epoch labels:

| epoch | 100 | 200 | 300 | 400 | 600 | 800 | 1000 |
|---|---|---|---|---|---|---|---|
| D0 | +0.016 | −0.006 | −0.000 | −0.066 | −0.075 | −0.049 | −0.084 |
| D1 | +0.015 | +0.001 | +0.004 | +0.006 | −0.046 | −0.012 | −0.061 |

The pre-registered late-minus-early contrast:

* **D0:** L = −0.0737 [−0.1501, +0.0036] s0.67 **unresolved**
* **D1:** L = −0.0431 [−0.0879, +0.0018] s0.70 **unresolved**

So at fixed epochs the intervention effect drifts negative late in training in
both configurations, but under the frozen rule that drift is **not resolved**.

## 7. Table 4 — telescoping decomposition (σ_c^BCE, common cap 200)

Algebraic identity, not mediation: full = common-cap + M1 extension − M0
extension (holds ≤ 1e-12).

**−ΔDP**

| D | common-cap | M1 selection extension | M0 selection extension | full |
|---|---|---|---|---|
| D0 | −0.0113 u | −0.0803 [−0.1546, −0.0047] s0.70 u | −0.0487 [−0.1450, +0.0316] u | −0.0429 u |
| **D1** | +0.0288 u | **−0.0817 [−0.1437, −0.0307] s0.77 R** | **+0.0951 [+0.0472, +0.1527] s0.97 R** | **−0.1480 R** |

For D1 both extensions are resolved and pull in opposite directions: moving M1
from its cap-200 checkpoint to its full-support checkpoint **lowers** −ΔDP by
0.082, while the same move for M0 **raises** it by 0.095. The intervention
contrast subtracts the second, so both add to the harmful full-support value.

## 8. Selector diagnostics (descriptive, correlational only)

| D | selector | cap | M1 median epoch (boundary rate) | M0 median epoch (boundary rate) | median M1−M0 gap |
|---|---|---|---|---|---|
| D0 | BCE | 200 | 200.0 (0.63) | 200.0 (0.63) | +0 |
| D0 | BCE | 1000 | 961.0 (0.17) | 448.5 (0.40) | +36.5 |
| D1 | BCE | 200 | 196.0 (0.37) | 196.0 (0.07) | +2 |
| **D1** | **BCE** | **1000** | **947.0 (0.03)** | **692.5 (0.00)** | **+173.5** |
| D1 | AUC | 1000 | 976.5 (0.03) | 706.5 (0.00) | +94.0 |

Capped at 200 the two arms are selected at essentially the same epoch; released
to 1000 they separate, M1 later than M0. This moves in the same direction as
the τ shift. **This is a correlation, not mediation**, and X25 §11 registered
"more training" and "release of boundary-constrained selection" as confounded
within H.

## 9. Pre-registered cases (X25 §9)

| coordinate | σ_c^BCE | σ_c^AUC |
|---|---|---|
| −ΔDP | D0: none · **D1: A, D** | D0: none · D1: none |
| −ΔEO | D0: none · **D1: A, D** | D0: none · D1: none |
| ΔAUC | D0: none · **D1: A, D** | D0: none · D1: none |

* **A** (S resolved with the sign of H, T unresolved) and **D** (L unresolved
  while S resolved) both hold for D1 under σ_c^BCE, on all three coordinates.
* Nothing is resolved for D0, or under σ_c^AUC anywhere.

## 10. What this supports

**Within NIFTY/German, under the native configuration (D1) and σ_c^BCE:** the
horizon-associated shift in intervention attribution is largely expressed
through access to later checkpoints. On the same H = 1000 trajectories, capping
the selector at epoch 200 gives τ_int = +0.029 (unresolved) while the full
support gives −0.148 (resolved); the selection-support contrast S = −0.177 is
resolved and accounts for essentially all of H = −0.177, with T ≈ 0.

At the same time, **the fixed-epoch trajectory alone does not resolve the
effect** (L unresolved for both D levels), so this is Case D as well as Case A:
model selection is central to the observed attribution, with a late-training
drift that is visible but not resolved at this sample size.

**Under σ_c^AUC none of this reproduces:** no capped, full-support or
selection-support quantity is resolved for either D level. The phenomenon is
specific to the BCE selector in this setting, and that is part of the main
finding.

**Under D0 (the Arm-A configuration) the same expansion produces nothing
resolved**, consistent with X24, where the harmful attribution appeared only
when both factors were at their native levels.

### Not claimed

* not "checkpoint selection causes unfairness";
* not "the BCE selector is wrong";
* not "NIFTY is unfair";
* not "horizon alone explains the effect";
* not "all fair GNN attribution is selector-dependent";
* no generalization beyond NIFTY/German; no mediation claim; no post-hoc epochs
  or thresholds.

### Limits

* One setting, 6 splits × 5 runs.
* The prefix match is not exact, so S is a within-run selection-support
  contrast, not a pure selector effect, and T carries run-to-run divergence.
* "More training" and "a wider checkpoint support" are confounded within H by
  construction; X25 separates support expansion *given the same trajectory*, not
  the causal role of training length.
* The fixed-epoch drift is unresolved, so no claim is made that late training
  alone changes the effect.

## 11. Figures

* **Main:** `harness/results/figures/x25_nifty_german_trajectory_dp.{png,pdf}` —
  τ_int(−ΔDP) at fixed epochs for D0 and D1, with a strip panel showing where
  σ_c^BCE selects over the full support.
* **Second:** `x25_nifty_german_cap_vs_full_dp.{png,pdf}` — cap ≤ 200 vs full
  ≤ 1000 per D and selector, with the frozen H = 200 value as a gray tick.
* **Supplementary:** `x25_nifty_german_supplementary.{png,pdf}` — both views for
  −ΔDP, −ΔEO and ΔAUC.

## 12. Outputs

`harness/results/x25/`: `x25_R10.csv`, `x25_R11.csv`, `R10_trajectories/`,
`R11_trajectories/` (60 `.npz` each, 465 MB total, checksum-manifested in
`x25_R10_trajectories.sha256` and `x25_R11_trajectories.sha256`, not tracked in
git), `x25_replay_noise.csv`, `x25_replay_noise_flips.csv`,
`x25_amended_g3.csv`, `x25_analysis.txt`, `x25_summary.csv`,
`x25_cell_table.csv`, `x25_prefix_cells.csv`, `x25_selected_epochs.csv`,
`x25_gate.txt`.
