# X26 results: FairGB selection-support replication (Bail; Credit stopped)

**Pre-registration:** `630c82a`. **Tooling:** X26's runner, analyzer and figure
script are committed together with this document (`d9a73b6` and `85a1048` are
X27's). X22–X25 and all frozen CSVs are untouched.

## 1. What ran, and what stopped

| dataset | status |
|---|---|
| **Bail** | gates passed, native-length rerun complete, analysed |
| **Credit** | **stopped at gate G1** before any run; reported, not analysed |

**Credit's stop (X26 §12).** The instrumentation-invariance gate compares the
hooked validation trajectory against the spread between no-hook repeats. On
credit, with a three-repeat baseline, M1 gave hook 3.11 against a band of 2.96 —
outside. A follow-up isolation diagnostic showed the *cause is not the hook*: an
"empty" hook that records nothing at grid epochs behaves exactly like the full
hook, and with a four-repeat baseline both fall inside the band (2.64–3.32 vs
2.68–3.46). FairGB/credit's run-to-run instability (~3 in validation-score
units) is simply large enough that the verdict flips with baseline sample size.
Rather than raise the repeat count until the gate passed — the pattern this
project refused in X25 — credit is recorded as **stopped**, and bail proceeds
alone, which §12 explicitly permits.

**Bail's gates:** 26/26 checks, including BatchNorm-buffer restore fidelity with
exact DP/EO, cap slots frozen within {0..199}, grid recording at the
pre-registered epochs, loader node alignment and validation/test disjointness.

**One criterion was aligned, not loosened.** The gate's fixed 2-pair-swap AUC
bound (a bail-scale form) was replaced by the project's **already validated X20
noise-derived criterion**, and the same alignment was then propagated into the
analyzer, where one comparison (s25 r4 M0/BCE) had hit the stale bound with
exact DP/EO (6.3e-17, 6.1e-18) and 3.00 swaps against **2,891** noise-reorderable
pairs. This is reuse of an existing criterion already in force in the gate, and
it was applied before any decomposition was computed or seen.

## 2. Run and contracts (Bail)

* 6 splits × 5 runs, native H = 1500, M0/M1 paired, through the existing
  pipeline; B trained but unused.
* 11:53 → 13:24, rc 0, 30/30 cells, 60 rows, **60/60 valid trajectory files**.
* CellStore reload clean, 0 duplicate keys, protocol `native`,
  `method_epochs` 1500, `b_epochs` 200, seeds 27+run, no non-finite outcome, no
  undefined EO, all selected epochs in [0, H−1], max cap-slot epoch exactly 199.
* Copies into `harness/results/x26/` verified by SHA-256 (CSV `896dd0e1…`, all 60
  trajectory checksums matching `/tmp`).
* **Replay-noise envelope, measured before any decomposition:** max spread
  3.231e-05 → **bound 1.292e-04**, 0 sign flips across 10 slot evaluations.
* **G3 on all 120 comparisons: pass, 0 boundary flips.**
* Identities `H_shift = S + T`, `S = full − cap`, `S = E1 − E0` hold ≤ 1e-12 per
  cell and in all 10,000 replicates.

## 3. Reproduction gate: PASS

All six checks lie inside the frozen native intervals, and the headline
reproduces:

| selector | coordinate | rerun full | frozen native interval (mean) |
|---|---|---|---|
| BCE | −ΔDP | −0.0189 | [−0.0354, +0.0137] (−0.0103) |
| BCE | −ΔEO | −0.0131 | [−0.0191, +0.0211] (+0.0020) |
| BCE | ΔAUC | +0.0643 | [+0.0552, +0.0691] (+0.0624) |
| AUC | −ΔDP | −0.0121 | [−0.0425, +0.0187] (−0.0100) |
| AUC | −ΔEO | −0.0247 | [−0.0447, −0.0089] (−0.0244) |
| AUC | ΔAUC | +0.0432 | [+0.0364, +0.0483] (+0.0423) |

Headline: BCE −ΔDP −0.0189 [−0.0520, +0.0102] s0.60, **unresolved**, as frozen.

## 4. Selection-support decomposition (Bail)

cap = {0..199} (what the frozen H = 200 runs selected over), full = {0..1499},
both on the **same** native-length trajectories. τ_H200 is the frozen controlled
cell, paired by (split, run).

**−ΔDP, σ_c^BCE (primary)**

| quantity | value |
|---|---|
| τ_H200 (frozen) | −0.0597 |
| τ_cap | −0.0659 [−0.1217, −0.0143] s0.73 u |
| τ_full | −0.0189 [−0.0520, +0.0102] s0.60 u |
| **H_shift** | **+0.0407 [+0.0198, +0.0632] s0.80 R** |
| **S** (selection support) | **+0.0470 [+0.0187, +0.0764] s0.83 R** |
| **T** (prefix/run) | −0.0062 [−0.0245, +0.0093] s0.53 u |
| **case** | **A — selection-support** |

Telescoping: **E1 +0.0464 [+0.0176, +0.0763] R**, **E0 −0.0005 u**. The whole
contrast comes from M1's checkpoint moving; M0's selection barely changes.

**All coordinates and selectors**

| coordinate | selector | H_shift | S | T | case |
|---|---|---|---|---|---|
| −ΔDP | BCE | +0.0407 R | +0.0470 R | −0.0062 u | **A selection-support** |
| −ΔDP | AUC | +0.0692 R | +0.0733 u (s0.70) | −0.0041 u | C trajectory-only |
| −ΔEO | BCE | +0.0523 R | +0.0585 R | −0.0062 u | **A selection-support** |
| −ΔEO | AUC | +0.0464 R | +0.0433 u (s0.73) | +0.0031 u | C trajectory-only |
| ΔAUC | BCE | +0.0448 R | +0.0439 R | +0.0009 u | A selection-support |
| ΔAUC | AUC | +0.0300 R | +0.0288 R | +0.0011 u | A selection-support |

Under σ_c^AUC on the fairness coordinates, S has the same size and sign as under
BCE but misses the 0.75 sign-stability threshold (0.70, 0.73), so the frozen
rule records it unresolved and the case falls to C. That is a
resolution difference, not a sign difference.

## 5. Fixed-epoch trajectory (no selection)

τ_int(−ΔDP) deepens to −0.154 at epoch 200 and then **returns toward zero**:
−0.103 (300), −0.057 (500), −0.059 (750), −0.031 (1000), −0.003 (1250), −0.018
(1499). −ΔEO follows the same shape; ΔAUC rises from ≈0 to +0.035–0.046 and
stays there.

So at fixed epochs the intervention effect genuinely weakens late in training —
and the selector, given access to those later checkpoints, lands there.

## 6. Selected-epoch diagnostics (descriptive)

| selector | support | M1 median [range] | M0 median [range] |
|---|---|---|---|
| BCE | cap | 180.5 [107, 199] | 61.0 [33, 194] |
| BCE | full | **999.0 [500, 1437]** | **63.0 [35, 267]** |
| AUC | cap | 169.5 [112, 197] | 140.5 [26, 195] |
| AUC | full | 1254.5 [770, 1489] | 1153.5 [46, 1496] |

Under BCE the asymmetry is stark: releasing the cap moves **M1** from ~180 to
~999 while **M0 stays at ~63**. That is exactly what the telescoping shows
numerically (E1 large, E0 ≈ 0). Correlational only; no mediation is claimed.

## 7. What this replicates, and what it does not

* **The X25 mechanism replicates in FairGB/bail under σ_c^BCE**: the
  controlled→native shift is expressed through the selector gaining access to
  later checkpoints (S resolved, T ≈ 0, Case A) — on all three coordinates.
* **The direction differs from NIFTY/German.** There, the native protocol made a
  near-zero effect *harmful*; here, it makes a harmful effect (−0.060) *weaker*
  (−0.019). The mechanism is the same; the sign of the shift is not.
* **It is one-armed.** In bail the whole effect is M1's checkpoint moving, while
  in NIFTY/German both arms' extensions contributed in opposite directions.
* **Credit is unknown.** Its gate stopped before any run, so X26 replicates on
  one of the two intended datasets. The cross-dataset comparison
  (§8 of the prereg) cannot be completed.

**Not claimed:** that checkpoint selection causes unfairness; that σ_c^BCE is
wrong; that FairGB is protocol-dependent as a method property; anything about
credit; or any dataset-level law from one dataset.

## 8. Outputs

`harness/results/x26/`: `x26_bail.csv`, `bail_trajectories/` (60 `.npz`, 44 MB,
checksum-manifested, excluded from git by the repository's `*.npz` rule),
`x26_bail_analysis.txt`, `x26_bail_summary.csv`, `x26_bail_cell_table.csv`,
`x26_bail_selected_epochs.csv`, `x26_bail_gate.txt`, `x26_noise_bail.csv`,
`x26_noise_bail_flips.csv`, `x26_bail_trajectories.sha256`.
Figures under `harness/results/figures/` with the `x26_bail_` prefix.
