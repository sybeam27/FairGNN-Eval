# X26: FairGB selection-support replication (pre-registration)

**Status: pre-registered before any X26 result exists.** The commit hash of this
file is recorded in the results report; no X26 quantity is computed before that
commit.

**Frozen and untouched:** X22, X23, X24, X25 (including its labels and the G3
amendment), all Arm A / native / X24 / X25 CSVs, the bootstrap design, the
resolved rule (sign stability ≥ 0.75, |mean| ≥ 0.010, 95% interval excluding 0)
and every existing selector, metric and method definition.

**Scope: FairGB only, on Bail and Credit.** No new method, dataset, metric,
selector or run count. No NIFTY, FairVGNN, FMP or FairGB/German work.

**This is a targeted replication, not a confirmation.** A result in which the
X25 selection-support explanation does *not* transfer to FairGB is as
informative as one in which it does, and is reported the same way.

## 1. Why these two cells

Both were frozen in X23/X24 with **horizon as the only native change**, yet they
transferred differently:

| cell | controlled H | native H | τ_int(−ΔDP, BCE): controlled → native | native state |
|---|---|---|---|---|
| FairGB / Bail | 200 | 1500 | −0.060 → −0.010 | unresolved (effect weakened) |
| FairGB / Credit | 200 | 2000 | +0.014 → +0.081 | resolved improvement (effect strengthened) |

Frozen values quoted as the bridge targets (means over 30 cells):

| cell | selector | τ_H200: −ΔDP / −ΔEO / ΔAUC | native: −ΔDP / −ΔEO / ΔAUC |
|---|---|---|---|
| Bail | BCE | −0.0597 / −0.0654 / +0.0195 | −0.0103 / +0.0020 / +0.0624 |
| Bail | AUC | −0.0814 / −0.0711 / +0.0133 | −0.0100 / −0.0244 / +0.0423 |
| Credit | BCE | +0.0139 / +0.0079 / −0.0076 | +0.0810 / +0.0683 / +0.0032 |
| Credit | AUC | −0.0157 / −0.0246 / −0.0056 | +0.0772 / +0.0681 / −0.0036 |

## 2. Questions

* **RQ1.** Is FairGB's controlled → native attribution shift associated with the
  selector gaining access to later checkpoints of the native-length trajectory?
* **RQ2.** Do Bail and Credit, which transferred differently, show different
  structure in the same decomposition?
* **RQ3.** Does any of it hold under σ_c^AUC as well as σ_c^BCE?

Not claimed in any outcome: that the selector causally creates unfairness, that
σ_c^BCE is wrong, that FairGB is protocol-dependent as a method property, or
that two datasets establish a dataset-level law.

## 3. Audit findings (read-only, 2026-09-16)

1. **Total H does not reach the first 200 epochs.**
   * `get_enc_cls_opt` builds plain Adam optimizers; there is **no scheduler**
     anywhere in the FairGB path (no `lr_scheduler`, `StepLR`,
     `CosineAnnealing`, `LambdaLR`, no `param_group['lr']` writes).
   * `args.epochs` is read exactly twice: assignment, and
     `for epoch in range(args.epochs)`.
   * `warmup` is an **absolute** epoch index (default 5) gating CNM
     (`if use_cnm and epoch >= args.warmup`). It is **absent from both frozen
     configs**, so both arms and both horizons use the same value.
   * `mixup.py`, `models.py`, `data_utils.py` never reference total epochs.
   * Per-epoch RNG draws (`sampling_idx_individual_dst` with `eta`,
     `Beta(2,2).sample`) depend on group sizes and `eta`, not on H.

   So the selection-support contrast is meaningful here. Had any schedule
   depended on total H, §17's stop would have applied.
2. **Controlled and native differ only in H.** For both datasets every config
   value matches; `fairgb_get_dataset_normalize` appears explicitly in the
   native config and defaults to the same value in the controlled call.
3. **Epoch convention: FairGB logs 0 … H−1** (`range(args.epochs)`,
   `ValidationHistory(ref, epochs)`), unlike NIFTY's 0 … H. The frozen native
   runs top out at 1499 (Bail) and 1993 (Credit).
4. **No per-epoch or checkpoint artifacts exist** for the frozen FairGB runs, so
   a native-length rerun is required (§7).
5. **Restore path already carries BN buffers.** `inference_state` snapshots full
   `state_dict()`s of encoder and classifier (both contain `BatchNorm1d`), and
   `FairGB.inference_modules()` names them. This is checked again in G2.
6. **Sizes:** Bail N=18,876 (val/test 4,719 each); Credit N=30,000 (val/test
   7,500 each).

## 4. Design

Native-length trajectories: **Bail H = 1500**, **Credit H = 2000**, 6 splits
(20–25) × 5 runs (0–4), M0/M1 paired per cell, unchanged configs.

**Selector supports**, applied independently to M0 and M1, ties to the earliest
epoch via the frozen `_argbest`:

| support | epochs |
|---|---|
| cap | **{0 … 199}** — matching what the frozen H = 200 runs actually selected over |
| full | **{0 … H_native − 1}** |

σ_BCE = argmin validation BCE; σ_AUC = argmax validation AUC, both from the
unified `ValidationHistory` records, validation only.

## 5. Estimands

Per cell, outcome Y from G_c on the test split (`score > 0`), for
Y ∈ {−ΔDP (primary), −ΔEO, ΔAUC} and σ ∈ {BCE (primary), AUC (robustness)}:

    tau_cap  = mean over cells of [ Y(M1 @ sigma_cap)  - Y(M0 @ sigma_cap) ]
    tau_full = mean over cells of [ Y(M1 @ sigma_full) - Y(M0 @ sigma_full) ]
    S        = tau_full - tau_cap                      selection-support expansion
    T        = tau_cap  - tau_H200                     prefix/run bridge
    H_shift  = tau_full - tau_H200                     observed controlled -> native change
    identity:  H_shift = S + T     (checked <= 1e-12 per cell and in every replicate)

τ_H200 is the frozen controlled cell, paired by (split, run).

**Telescoping (algebraic only, never mediation):**

    E1 = Y(M1 @ sigma_full) - Y(M1 @ sigma_cap)
    E0 = Y(M0 @ sigma_full) - Y(M0 @ sigma_cap)
    S  = E1 - E0                                       (checked <= 1e-12)

**Fixed-epoch trajectory**, selector removed, grids fixed now and never
extended or trimmed after seeing results (0-indexed; 1500/2000 are outside the
0…H−1 range, so the last grid point is H−1):

* **Bail:** {25, 50, 100, 150, 200, 300, 500, 750, 1000, 1250, 1499}
* **Credit:** {25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 1999}

  τ_int(t) = mean over cells of [Y(M1 @ t) − Y(M0 @ t)], for DP, EO and AUC.
  Reported as a descriptive curve with intervals; **no per-epoch significance
  claim** and no per-epoch labels.

## 6. Instrumentation (§10: minimal storage)

A wrapper around the unchanged pilot, in the X25 style, patching only in-process:

* **stored per arm:** per-epoch validation BCE and AUC (~0.03 MB), the test
  outcomes and raw scores at the fixed-grid epochs (~0.4–0.7 MB), and the four
  checkpoint slots (cap/full × BCE/AUC) evaluated through the pipeline's own
  `score_fn`;
* **not stored:** per-epoch test score arrays for all epochs (57–120 MB per arm,
  unnecessary);
* cap slots are obtained by freezing a copy of the running-best slots at epoch
  199; full slots are the history's own slots;
* test scores are kept outside `ValidationHistory`, so no selector can see them;
* instrumentation must not change training behaviour — checked in G1.

## 7. Reruns

Required, since no trajectory artifacts exist. Bail (H = 1500) and Credit
(H = 2000), 6 × 5 each, through the existing pipeline: cell-level incremental
persistence to `/tmp`, `setsid` detached, completion marker, resume on
interruption, SHA-256-verified copy into `harness/results/x26/`. B is trained by
the pipeline because no verified M0/M1-only path exists; **B is not used in any
X26 quantity**. No existing CSV is overwritten.

## 8. Gates (all before any X26 estimate is reported)

* **G1 instrumentation invariance** (CUDA, short horizon, same process): with
  and without the hook, the validation trajectory and slots must agree within
  the no-hook repeat difference.
* **G2 restore fidelity including BN buffers:** after restoring a slot, encoder
  and classifier `state_dict()`s — parameters **and** `running_mean` /
  `running_var` — must equal the snapshot exactly; node alignment, raw scores,
  hard predictions, DP, EO and AUC are checked with the project's existing
  noise-aware criterion.
* **G3 replay/restore per comparison**, reusing the frozen criterion: epoch
  exact; alignment exact; finite; AUC within 2 pair swaps; DP/EO exact **or**
  explained solely by decision-boundary sign flips inside an independently
  measured replay-noise envelope, `bound = max(1e-5, 4 × max replay spread)`
  (X14 form, X25 amendment `326f04f`). The envelope is measured on the FairGB
  test path **before** any decomposition is computed, exactly as in X25; no
  tolerance is set or adjusted after seeing results.
* **G4 trajectory completeness** (H records, 0 … H−1, finite) and **test
  isolation** (validation and test disjoint; selectors receive validation only).
* **G5 analyzer mechanics** on synthetic trajectories with known answers
  (S = 0 when cap and full select the same epoch; identities ≤ 1e-12; shared
  bootstrap indices; leakage guard).

## 9. Statistics

6 splits × 5 runs; the frozen `bootstrap_armA.boot()`, 10,000 replicates,
splits resampled then runs within each drawn split. One table per dataset, one
row per (split, run), every column travelling together so M0/M1, cap/full,
BCE/AUC, the grid epochs and the paired frozen H=200 cells share resampling
indices. Generator `np.random.default_rng(20260918)`, fixed column order, used
once per dataset. Frozen resolved rule and 95% percentile intervals; **no new
threshold**.

## 10. Reproduction gate against the frozen native headline

Cell-level bitwise reproduction is not expected (X25 §2.3 demonstrated
cross-process GPU nondeterminism). Before any decomposition is reported, for
each dataset:

1. configuration identity: protocol, `method_epochs`, `b_epochs`,
   `feature_normalize`, seeds (27 + run), splits 20–25, runs 0–4;
2. for each selector and coordinate, the rerun **full-support** τ_int mean lies
   inside the frozen X23/X24 95% interval of that native cell (12 checks per
   dataset);
3. the frozen native headline state on −ΔDP under σ_c^BCE is reproduced: **Bail
   unresolved**, **Credit resolved and positive**.

Failure of 1, 2 or 3 for a dataset stops **that dataset** and is reported; the
other dataset may proceed (§12).

## 11. Interpretation cases, fixed now

Per dataset and selector, using the frozen resolved rule, with the primary read
on −ΔDP under σ_c^BCE:

| case | condition |
|---|---|
| **A** selection-support | S resolved with the sign of H_shift, and T not resolved |
| **B** prefix/run | T resolved |
| **C** trajectory-only | S not resolved while H_shift is resolved |
| **D** mixed | S and T both resolved |
| **E** none | H_shift not resolved, or nothing resolved |

Cross-dataset reading is **descriptive only**, in the frame given: large S in
one and not the other (dataset-conditional); large S in both with different
signs; S ≈ 0 and large T in both (the X25 mechanism does not transfer); both
large (mixed); or both unclear (no replication evidence). Bail and Credit are
never averaged into one number.

Permitted wording: *"In FairGB/Credit, the controlled-to-native attribution
shift is / is not primarily expressed through access to later checkpoints under
BCE selection."* Prohibited: selector-causes-fairness claims, "BCE is a bad
selector", assuming the NIFTY result transfers, method-level or dataset-level
universals, extra runs to obtain significance, or any post-hoc change to the
epoch grids.

## 12. Stop rules

Stop the affected dataset and report on: a controlled/native configuration
mismatch; a discovered total-H-dependent schedule invalidating the
decomposition; instrumentation that changes training behaviour; node-alignment
mismatch; test leakage; checkpoint or BN restore failure; non-finite values;
trajectory incompleteness; or failure of §10 that replay noise does not explain.
One dataset stopping does not stop the other.

## 13. Not done

No FMP, FairVGNN, NIFTY or FairGB/German work; no new dataset, metric, selector
or run count; no change to frozen results; no per-epoch significance claims; no
mediation analysis.
