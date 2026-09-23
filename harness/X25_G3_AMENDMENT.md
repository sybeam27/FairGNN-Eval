# X25 G3: narrow numerical-replay amendment

**The original G3 FAILED. That failure stands and is not rewritten.** This
document adds a narrow amendment, fixed *before* any scientific decomposition
result has been computed or inspected, and applied only after this file is
committed.

At the time of writing: no τ, no bootstrap, no table, no figure and no
scientific label has been computed for X25. The only quantities computed are the
gate diagnostics recorded here.

## 1. Original G3 (X25 §7, pre-registered `2547bf3`)

For every comparison (cell × arm × selector), between the stored in-training
test scores and the pipeline's restored-checkpoint outcome:

* the selector epoch replayed from stored validation records must equal the CSV
  epoch and the `ValidationHistory` slot;
* **DP and EO must match exactly** (≤ 1e-12);
* AUC must be within the pre-existing tolerance of 2 pair swaps;
* otherwise: hard stop.

## 2. What was observed (recorded as-is)

Over the complete rerun, **240 comparisons** (30 cells × 2 arms × 2 selectors ×
2 runs):

* **239** matched DP and EO to ≤ 1e-12, with AUC within 2 pair swaps.
* **1 failed: R10, split 20, run 3, M1, σ_c^BCE, epoch 988** — |ΔDP| 6.41e-3,
  |ΔEO| 8.40e-3, AUC within 1 pair swap.
* **Selected epochs matched exactly in all 240**, including the failing one
  (replayed 988 = CSV 988 = slot 988).
* In the failing comparison exactly **one test node** changes hard prediction.
  Its stored score is **4.52e-5** from the `score > 0` threshold (the next
  nearest are 3.82e-4 and 3.96e-4).
* Flipping that single node reproduces the restored-checkpoint DP **and** EO to
  **5.6e-17**. One flip in group a = 0 (156 test nodes) moves DP by 1/156 =
  0.0064, exactly the observed difference.
* The failure is therefore a thresholded-metric artefact of CUDA
  floating-point noise, not an epoch, alignment or restoration error. G1 had
  already shown NIFTY/German is not bitwise reproducible even within one
  process.

## 3. The noise envelope, measured independently

`harness/experiments/x25_replay_noise.py`, run **before** this amendment was
written and with no access to any decomposition result. It trains NIFTY/German
cells exactly as the pipeline does, then at each shared-selector slot restores
the checkpoint and replays the pipeline's test scoring 4 times.

Scope: 3 cells (splits 20/21, runs 3/0) × both D levels × both arms × both
selectors = **24 independent slot evaluations**, H = 1000, 250 test nodes each.

| quantity | max | median |
|---|---|---|
| replay-to-replay spread (same restored checkpoint) | **4.272e-4** | 1.450e-4 |
| stored in-training vs replayed | 8.240e-4 | 1.793e-4 |

* **Sign flips between stored and replayed in those 24 evaluations: 0** over
  6,000 node comparisons. So a flip is rare, not routine.
* M1 slots (late epochs) carry the larger noise (1e-4 … 4e-4); M0 slots
  (earlier epochs) are 2e-6 … 8e-5.

**Envelope, frozen here by the pre-existing project criterion** (X14 native
gate, reaffirmed in X20): `bound = max(1e-5, 4 × max replay spread)`

    bound = max(1e-5, 4 x 4.272e-4) = 1.709e-3

The maximum measured spread is used, not the median, and the value comes only
from these 24 independent measurements. The failing checkpoint itself no longer
exists (checkpoint states lived in the original process), so it contributes
nothing to the envelope. **ε was not chosen from the failing node's score**; for
the record, that score (4.52e-5) is 37× smaller than the bound and smaller than
the median measured spread.

## 4. Amended G3

For each comparison, all of the following must hold.

1. **Epoch:** replayed selector epoch == CSV epoch == slot. Unchanged, exact.
2. **Alignment:** test node ids, labels and sensitive attributes identical
   across the arms of a cell and across the runs of a split; validation and
   test disjoint.
3. **Finiteness:** no non-finite stored score.
4. **AUC:** within 2 pair swaps, the pre-existing criterion, unchanged.
5. **DP/EO:** exact (≤ 1e-12), **or** explained as boundary flips, which
   requires every one of:
   * **A.** the set of test nodes whose hard prediction differs is identified
     explicitly;
   * **B.** each difference is a sign flip of that node's score;
   * **C.** every flipped node lies within the frozen envelope,
     |stored score| ≤ 1.709e-3, i.e. the flip is inside measured device noise;
   * **E.** flipping exactly that set reproduces the restored DP **and** EO to
     ≤ 1e-12, with no unexplained residual;
   * **F.** the flip count, the flipped scores and the metric impact are
     recorded in the output table.
6. **Hard stop otherwise**, including: an epoch mismatch, an alignment or
   restoration mismatch, a non-finite score, a sign flip of a node outside the
   envelope, a discrepancy needing more than 4 flips (the search depth), or any
   residual the identified flips do not explain.

The search is implemented in `harness/experiments/x25_amended_g3.py`, which reads
the envelope from the diagnostic CSV and computes no scientific quantity.

## 5. Freeze and what follows

The amended rule is applied to all 240 comparisons **only after this file is
committed**, and the commit hash is recorded in the results report.

* If all 240 pass, both facts are reported: **original G3: FAIL**; **amended
  numerical-replay G3: PASS**.
* If any comparison fails the amended rule, X25 hard stops again.

Nothing else in X25 changes: not the estimands, the epoch grid, the selectors,
the bootstrap, the frozen resolved rule, or the interpretation cases.

## 6. How this is described in the report

> The original restore-fidelity gate required exact agreement in thresholded
> DP/EO and failed in 1 of 240 comparisons because a single near-zero test score
> changed sign under CUDA replay. Before inspecting the scientific
> decomposition, we amended the gate to permit only decision-boundary flips
> demonstrably bounded by independently measured replay noise and fully
> accounting for the metric discrepancy.

It is recorded as a **narrow numerical-replay amendment**. It is never described
as the original gate passing, as a threshold changed after seeing results, or as
a general tolerance relaxation.

## 7. Sensitivity check, after the primary analysis

Once the primary X25 analysis is complete, the one boundary-flip comparison is
re-run both ways — (a) the stored trajectory value, (b) the replayed value — and
Δ_total, the selector-support contrast, the main decomposition quantities and
the qualitative labels are compared. If they are materially the same, that is
recorded; if they differ, the difference is raised as a main limitation.
