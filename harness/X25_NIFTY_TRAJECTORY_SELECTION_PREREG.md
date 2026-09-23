# X25: NIFTY/German trajectory–selection decomposition (pre-registration)

**Status: pre-registered before any trajectory rerun or derived result exists.**
The commit hash of this file is recorded in the results report. No X25 quantity
is computed before that commit.

**Frozen, not modified:** X22, X23, X24 (including its H1 label), the
P00/P10/P01/P11 CSVs, the bootstrap and resolved rule (sign stability 0.75,
|mean| ≥ 0.010, 95% interval excluding 0), Arm A and native results, and every
method definition.

**Role in the paper.** X24 labelled the controlled-to-native shift on −ΔDP under
σ_c^BCE H1 (Δ_H −0.117, resolved; I_HD −0.126, sign stability 0.73). X25 asks
*how* the horizon-associated shift arises. It does not revisit whether the shift
exists.

## 1. Questions

* **Q1.** Does the H = 200 → 1000 change in intervention attribution reflect
  later training dynamics that move M0 and M1 differently?
* **Q2.** Or is much of it explained by σ_c^BCE gaining access to checkpoints
  after epoch 200?
* **Q3.** If both, how large is each?

The goal is to separate, experimentally and algebraically, (A) the extension of
the available training trajectory from (B) the extension of the checkpoint
support visible to the selector. It is **not** to show that selection is a
causal mediator. No mediation terms are used.

## 2. Audit findings (read-only, 2026-09-16)

1. **Nothing per-epoch is persisted.** The X24 runs wrote only per-selector
   final CSV rows. `ValidationHistory` holds per-epoch validation raw scores,
   validation BCE and validation AUC in memory. Checkpoint states exist only in
   each selector's running-best slot, and test scores exist only at the selected
   slots. Capped-selector and fixed-epoch **test** outcomes are therefore not
   recoverable, so the minimal-rerun rule (§8) applies. This is not a STOP.
2. **H does not enter epochs 0–200.**
   * NIFTY uses two Adam optimizers with a constant lr and no scheduler;
     `fit()` loops `range(epochs + 1)`.
   * The per-epoch RNG draws (`drop_feature` on both views; `dropout_adj` at
     p = 0 draws nothing) do not depend on H, and nothing pre-draws future
     epochs.
   * The validation views are built once at construction.
   * Horizon enters only `ValidationHistory`'s completeness count.

   So epochs 0..200 of an H = 1000 run follow the same computation as an
   H = 200 run; any difference is execution-level.
3. **Cross-process GPU nondeterminism is present.** The X24 timing cell and the
   full-run cell for P10 (split 20, run 0) share the configuration and run in
   different processes:
   * M0: identical.
   * M1 diverged: σ_c^BCE epoch 627 vs 899, int_ndp −0.121 vs −0.212.
   * B: differs slightly.

   The P01 pair (H = 200) is identical in all 17 compared columns. Consequences:
   * a rerun is not expected to reproduce P10/P11 cell by cell (§8 defines an
     aggregate gate);
   * every X25 decomposition quantity is computed **within the rerun
     trajectories**;
   * the bridge to the frozen H = 200 cells carries run-to-run divergence (§6).
4. **Selector support includes epoch 0.** NIFTY logs epochs 0..H. The frozen
   H = 200 selectors ranged over {0, …, 200}; P00's σ_c^AUC chose epoch 0 in
   some cells. The capped support is therefore **{0, …, 200}**, not
   1 ≤ t ≤ 200, so that it matches what P00/P01 actually selected over. The
   full support is {0, …, 1000}. Ties go to the earliest epoch, as in the frozen
   selectors.
5. **The instrumentation can be side-effect free.**
   * Before its per-epoch validation, NIFTY calls `self.eval()`. In eval mode
     the hook-based `spectral_norm` skips power iteration, BatchNorm does not
     update, and the GCN encoder has no dropout.
   * The clean-graph test forward under `no_grad` is therefore state- and
     RNG-free by construction.
   * The pilot calls `train()` through its module globals, once per arm, after
     `manual_seed(seed·1000 + split)`.
   * So a harness wrapper can record per-epoch test scores without editing
     `pilot_tau.py`, `core/`, or `algorithms/NIFTY.py`. This is verified
     empirically in G1 (§7).
6. German splits have 250 validation and 250 test nodes (N = 1000).

## 3. Data

Two new trajectory reruns, same configuration as X24:

| run | H | D | pilot arguments (unchanged from X24) |
|---|---|---|---|
| **R10** (rerun of P10) | 1000 | D0 | `--protocol armA --epochs 200 --method_epochs 1000 --dataset german --methods NIFTY --splits 20 21 22 23 24 25 --runs 5` |
| **R11** (rerun of P11) | 1000 | D1 | `--protocol native --epochs 200 --dataset german --methods NIFTY --splits 20 21 22 23 24 25 --runs 5` (native H = 1000) |

The frozen H = 200 cells P00 (`armA_german.csv`, NIFTY rows) and P01
(`x24_nifty_german_P01.csv`) are used only in the bridge (§6) and the prefix
check. The frozen P10/P11 are used only in the reproduction gate (§8).

### Recorded per arm (M0, M1), per (split, run), per epoch t = 0..1000

Validation raw score (the NIFTY σ_c input view, exactly what the pipeline logs),
validation BCE and AUC from the unified evaluator, and the **test raw score on
the clean graph**, the same test path as the pipeline's `score_fn`. Also stored:
test node ids, labels and sensitive attributes, for the alignment check.

Test scores are kept in a separate store, **never** on the `ValidationHistory`
object, so selectors cannot see them. Selection reads validation records only;
test scores are indexed only by the selected epoch t.

## 4. Estimands

For outcome coordinate Y ∈ {−ΔDP (primary), −ΔEO, ΔAUC} computed by G_c on the
test split (decision score > 0), a cell (split, run), and arm-specific epochs:

    c(e1, e0) = Y(M1 at epoch e1) − Y(M0 at epoch e0)

Every quantity below is the mean of a cell-level contrast over the 30 cells.

### 4a. Selector-support decomposition (Primary Analysis A)

For σ ∈ {BCE (primary), AUC (robustness)} and cap c ∈ {200, 1000}, arm a's
selected epoch is `e_a^{σ,c} = argbest_{t ∈ {0..c}} σ(val record of arm a at t)`,
with earliest-epoch ties.

    τ_D^{σ,c} = mean over cells of c(e_1^{σ,c}, e_0^{σ,c})     (on R10 for D0, R11 for D1)
    S_D^σ     = τ_D^{σ,1000} − τ_D^{σ,200}                      (selection-support expansion)

**Replay contract.** `e_a^{σ,1000}` must equal the pipeline's own slot epoch for
the rerun (the CSV `m1_epoch` / `m0_epoch`). The outcome from stored test scores
at that epoch must equal the rerun CSV outcome (§7, G3).

### 4b. Fixed-epoch intervention trajectory (Primary Analysis B)

The grid, fixed now and never extended or trimmed:
**t ∈ {25, 50, 100, 150, 200, 300, 400, 600, 800, 1000}.**

    τ_int(t, D) = mean over cells of c(t, t)      (R10 for D0, R11 for D1)

It is reported as a curve with intervals, and **no per-epoch labels are
assigned**. One summary contrast per D is pre-registered to decide Case C/D/E:

    L_D = mean_{t ∈ {600, 800, 1000}} τ_int(t, D) − mean_{t ∈ {100, 150, 200}} τ_int(t, D)

L_D is formed per cell and then averaged.

### 4c. Bridge (§6)

With τ_D^{H200} = the frozen H = 200 cell's τ_int under the same σ (P00 for D0,
P01 for D1), paired by (split, run):

    H_D = τ_D^{σ,1000} − τ_D^{H200}
    S_D = τ_D^{σ,1000} − τ_D^{σ,200}
    T_D = τ_D^{σ,200}  − τ_D^{H200}      (prefix/run difference)
    identity: H_D = S_D + T_D            (checked ≤ 1e-12 per cell, in the means and in every replicate)

H_D here uses the **rerun** full-support value. The X24 simple effects (frozen
P10/P11 minus frozen P00/P01) are reported beside it. Their difference is the
reproduction residual from §8, not a component.

### 4d. Telescoping selection decomposition (optional, σ = BCE)

The common cap is 200, applied to the H = 1000 trajectory:

    c(e_1^{1000}, e_0^{1000})  =  c(e_1^{200}, e_0^{200})                         common-cap intervention contrast
                                + [Y(M1, e_1^{1000}) − Y(M1, e_1^{200})]           M1 selection extension
                                − [Y(M0, e_0^{1000}) − Y(M0, e_0^{200})]           M0 selection extension

The identity is checked ≤ 1e-12. It is called an **algebraic telescoping
decomposition** only: never a direct, indirect or mediated effect.

## 5. Inference

* **Table.** One table with one row per (split, run), 30 rows. It holds every
  cell-level quantity for both D levels, both selectors, both caps, all grid
  epochs, the bridge columns and the telescoping terms.
* **Bootstrap.** It goes through the frozen `bootstrap_armA.boot()` once:
  10,000 replicates; the 6 splits resampled with replacement, then the runs
  within each drawn split. Every column of a row moves together, so M0/M1,
  caps, selectors, D levels, epochs and the paired frozen cells share
  resampling indices.
* **Generator.** `np.random.default_rng(20260917)`, fixed column order, in the
  new analyzer `harness/experiments/analyze_x25_trajectory.py`.
* **Resolved state.** The frozen rule, unchanged: sign stability ≥ 0.75,
  |mean| ≥ 0.010, and the 95% percentile interval excludes 0. The rule is used
  for S_D, T_D, H_D, L_D, τ_D^{σ,c} and the telescoping terms.
* **Fixed-epoch curve points** get mean, interval and sign stability only. No
  resolved labels are declared per epoch.

## 6. Prefix check and bridge wording

Cell level, D0: R10 capped at {0..200} vs P00; D1: R11 capped at {0..200} vs
P01, for each σ. Compared: selected epochs e1 and e0, M1 and M0 test AUC/DP/EO,
and int_*.

* **Exact prefix match:** all 30 cells have identical capped selected epochs
  and |Δ| ≤ 1e-9 on every compared outcome, for both σ. Then it may be written
  that *the H = 1000 prefix reproduces the H = 200 run, so the
  selector-support decomposition provides a clean bridge.*
* **Otherwise:**
  * report the number of exactly matching cells, the per-cell differences, and
    T_D with its interval;
  * call T_D a *prefix/run difference (including cross-process GPU
    nondeterminism)*;
  * **never** call S_D a "pure selector effect". S_D remains a within-run
    selection-support contrast, which is valid inside the rerun trajectory.

## 7. Gates before the full rerun

* **G1, hook side-effect test (CUDA, short horizon, same process).**
  * NIFTY D1 M1 and M0 at H = 30, run three times: without the hook, with the
    hook, and without the hook again.
  * The with-hook validation trajectory and selected slots must equal the
    no-hook trajectories. If the two no-hook runs are themselves not bitwise
    equal, the with-hook run must differ from them by no more than they differ
    from each other.
  * `ValidationHistory` must expose no test field.
  * Model is in eval mode at every hook call.
* **G2, wrapper smoke run.** One cell per D at H = 30 through the wrapper:
  * complete H + 1 records per arm;
  * finite validation and test scores;
  * stored test node ids equal the loader's `idx_test`; stored y and sensitive
    attribute equal the loader's at those ids;
  * validation raw score equals the history records.
* **G3, replay and restore, on the smoke run and later on every rerun cell.**
  * Full-support selectors recomputed from stored validation records must
    equal the pipeline slot epochs.
  * The G_c outcome from stored test scores at the slot epoch must equal the
    CSV outcome from the restored checkpoint: DP and EO exactly, AUC within 2
    pair swaps, the frozen restore criterion for NIFTY (X14).
  * Score orientation must match (`score > 0` gives the same predictions).
* **G4, existing gates.** `test_native_phase1.py` and `test_x24_smoke.py` must
  still pass on HEAD.
* **G5, analyzer mechanics.** On synthetic trajectories with known answers:
  * identical caps give S = 0;
  * a constant trajectory gives L = 0;
  * telescoping and bridge identities hold ≤ 1e-12;
  * leakage guard: the selector functions receive no test arrays.

## 8. Minimal rerun and the reproduction gate

**Execution.**
* Run R10 then R11, each 6 × 5, through
  `harness/experiments/x25_trajectory_run.py`. It wraps the unchanged pilot
  `main()`.
* No existing CSV is overwritten. Output goes to `/tmp` with cell-level CSV
  persistence and per-arm `.npz` trajectory files (atomic rename), using
  `setsid nohup`, a completion marker, and a SHA-256-verified copy to
  `harness/results/x25/`.
* **Resume.** A CSV cell counts only if both arm `.npz` files exist and pass
  integrity. Otherwise the cell is dropped and recomputed with the same
  command.

**Reproduction gate (vs frozen P10/P11)**, defined now because §2.3 shows
cell-level bitwise reproduction is not expected:
1. **Configuration identity:** protocol, `method_epochs`, `b_epochs`,
   `feature_normalize`, seeds (27 + run), splits 20–25, runs 0–4.
2. **Aggregate reproduction:** for each D, σ ∈ {BCE, AUC} and coordinate
   Y ∈ {−ΔDP, −ΔEO, ΔAUC}, the rerun full-support τ_int mean lies inside the
   frozen X24 95% interval of the corresponding frozen cell. That is 12 checks.
3. **Headline reproduction:** under σ_c^BCE on −ΔDP, the rerun resolved state
   and direction equal the frozen ones: R10 unresolved, R11 resolved and
   negative.

The number of bitwise-identical cells and the per-cell differences are reported
either way. **If 1, 2 or 3 fails: STOP.** No X25 analysis is reported; the cause
is investigated and reported.

## 9. Interpretation rules, fixed now

Applied per D (D1 carries the X24 harm, so it is the primary level), on −ΔDP
under σ_c^BCE, then reported for −ΔEO, ΔAUC and σ_c^AUC. "R" means resolved
under the frozen rule.

| case | condition | permitted wording |
|---|---|---|
| **A** | S_D R with the sign of H_D, **and** T_D not R | *the horizon-associated attribution shift is largely expressed through access to later checkpoints under BCE selection* (the words "clean bridge" only with an exact prefix match) |
| **B** | T_D R | *selector expansion alone does not explain the shift; the H = 1000 trajectory already differs from the H = 200 run by epoch 200* (includes run-to-run divergence) |
| **C** | L_D R and negative, and S_D not R | *later training dynamics materially change the intervention effect* |
| **D** | L_D not R, and S_D R | *model selection is central to the observed attribution* |
| **E** | L_D R and S_D R | *the observed attribution arises from interaction between training trajectory and checkpoint selection* |
| **none** | none of the above | reported with values only |

* **A and B** concern the bridge; **C, D and E** concern trajectory vs
  selection. Both families are reported, and both can hold at once.
* **If S_D^AUC is not R where S_D^BCE is R**, or σ_c^AUC shows no harmful
  full-support effect, that fact goes into the main interpretation.
* **Selector diagnostics**, descriptive and correlational only, per D × cap ×
  σ:
  * M0/M1 selected-epoch distributions;
  * selected epoch / cap;
  * boundary rate (selected == cap);
  * BCE minus AUC epoch difference;
  * the M1 − M0 epoch gap;
  * whether these move in the direction of τ.

**Prohibited:**
* "checkpoint selection causes unfairness"; "BCE selector is wrong"; "NIFTY is
  unfair"; "horizon alone explains the effect"; "all fair GNN attribution is
  selector-dependent";
* any generalization beyond NIFTY/German;
* post-hoc epochs or thresholds.

**Permitted template:** *In NIFTY/German, the apparent intervention effect under
the native-length BCE protocol can be decomposed into changes associated with
the extended training trajectory and with access to later checkpoints.*

## 10. Stopping rules

**Stop immediately and report** on any of:
* a configuration mismatch;
* the §8 reproduction gate fails;
* G1–G5 fail;
* test labels reachable by a selector;
* node alignment mismatch;
* score orientation mismatch;
* non-finite scores;
* trajectory incompleteness;
* restore/replay mismatch (G3);
* any need to modify frozen results or method code.

A run killed by infrastructure is resumed per §8 and recorded.

## 11. Not done

* No FMP.
* No FairGB 2×2.
* No new method, dataset, metric or selector.
* No change to X24.
* No further decomposition of D.
* No per-epoch hypothesis labels.
* No mediation analysis.

## 12. Outputs

Tables (planned):
1. selector-support decomposition;
2. the H200 vs H1000-prefix bridge;
3. the fixed-epoch trajectory;
4. telescoping.

Figures (planned): main, τ_int(−ΔDP) against epoch per D with the BCE-selected
epoch distributions marked; second, cap-200 vs full-1000 per D; supplementary,
the σ_c^AUC, −ΔEO and ΔAUC versions.

Every output goes to new paths under `harness/results/x25/` and
`harness/results/figures/`.
