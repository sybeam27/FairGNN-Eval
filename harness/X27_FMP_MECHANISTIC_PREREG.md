# X27: FMP mechanistic decomposition case study (pre-registration)

**Status: pre-registered before any FMP scientific result exists.** The commit
hash of this file is recorded in the results report; no τ is computed before it.

**Frozen and untouched:** X22, X23, X24, X25, the X26 FairGB job, and every
existing CSV, selector, metric, bootstrap design and resolved rule. X27 writes
only to its own paths (`harness/results/x27/`), so it cannot collide with the
FairGB selection-support job.

## 1. Purpose and role

**Question.** When a fair-GNN package contains several algorithmic components,
how much of the observed fairness–utility change can be attributed to *graph
propagation* and how much to the *claimed fairness correction*?

FMP is used as a **mechanistic case study**, not as an additional benchmark
method. It is **excluded** from every benchmark aggregate: the 10/12 counts, the
native-transfer counts, and any method win/loss frequency. Its implementation
admits a valid propagation-only control (F01) and a full cell (F11), which is
what makes the incremental fairness correction measurable *conditional on the
same propagation configuration*.

**Datasets: Pokec-z and Pokec-n only.** NBA is excluded because the official
code sets `test_idx=True`, making validation = test, which is unusable for a
common checkpoint-selector analysis. NBA is not run.

## 2. Official artifact (re-audited 2026-09-16)

| item | value |
|---|---|
| paper | Jiang et al., *Chasing Fairness in Graphs: A GNN Architecture Perspective*, AAAI 2024 (arXiv 2312.12369) |
| repository | `zhimengj0326/FMP`, pinned commit `7c528413cf110c68becc9cd4f3b7a85cf496447b` (2023-08-17) |
| local copy | `FMP-main/` is **byte-identical** to the pinned copies in `harness/provenance/fmp/` for all seven source files |
| supported datasets | `pokec_z`, `pokec_n`, `nba` (`main.py` choices) |
| architecture | MLP (`num_gnn_layer` Linear layers, hidden 64, output 2) followed by the FMP propagation/debiasing layer applied to the logits |
| K (propagation steps) | `--num-layers` default **5** (`run_fgnn.sh` never sets it) |
| epochs | **300** (active sweep line) |
| optimizer | Adam, lr 1e-3, **no weight decay passed** (`--weight_decay` parsed, never used) |
| selector | **none**: the last epoch's test metrics are reported |
| normalization | pokec: none; nba: `feature_norm` (not used here) |
| projection | `--L2 True` default → `L2_projection` (scaling by 2λ/(2λ+β)) |
| split | `load_pokec(..., train_ratio=0.8, seed=20, test_idx=False)` → 80/10/10 of labelled nodes |

**Paper vs executable code discrepancies** (recorded, not reconciled; the
**executable behaviour is the implementation target**):

| item | paper (Appendix I) | official code | taken |
|---|---|---|---|
| split | 50/25/25 | 80/10/10 (pokec) | code |
| propagation/debiasing stacks | "stack 2 layers" | K = 5 | code |
| projection | ℓ∞ ball (clamp) | L2 scaling (`--L2 True`) | code |
| weight decay | 1e-5 | parsed, not applied → 0 | code |
| MLP depth | "2 layers" | `num_gnn_layer` ∈ {2, 5} in the script | **2**, the value consistent with both (§4) |
| runs | "55 times" (apparent typo) | `running_times` 5 | code |
| λ values | grids only (§3) | grids only (§3) | **no native pair exists** |
| selection | not stated | last epoch | code |

## 3. λ candidates and the deterministic grid rule

**Extraction (executable active sweep line of `run_fgnn.sh`):**

* λ1 ∈ {5, 15, 20, 30}
* λ2 ∈ {0., 0.01, 0.1, 0.5, 1.0, 2.0, 3, 5, 10, 15, 20}
* commented-out alternatives (recorded, unused): λ1 ∈ {0, 10, 100, 1000, 5000},
  λ2 ∈ {10, 15, 30}; header comments list λ1 ∈ {1000, 5000, 8000, 10000}
* paper grids: λ_f ∈ {0, 5, 10, 15, 20, 30, 100}, λ_s ∈ {0, 0.01, 0.1, 0.5, 1,
  2, 3, 5, 10, 15, 20}

**There is no dataset-specific selected native (λ1, λ2).** The parser defaults
(3, 3) are **not** native and are not used as such. No pair is called
"published", none is chosen by looking at any result, and λ is never selected
using test outcomes. The analysis is therefore **configuration-conditional**:
results are reported as a surface over the preregistered grid.

**Deterministic rule (fixed here, applied to the executable candidate set).**
Sort the positive candidates of each dimension: one → use it; two → use both;
three or more → use the smallest and the largest positive value. This gives

* **λ1 ∈ {5, 30}** (from {5, 15, 20, 30})
* **λ2 ∈ {0.01, 20}** (from {0.01, 0.1, 0.5, 1.0, 2.0, 3, 5, 10, 15, 20})

low/high are **preregistered endpoints of the official search space**, not
best/worst. The grid is not changed after seeing results.

**Seven configuration types per dataset:** F00; F01(0.01); F01(20);
F11(5, 0.01); F11(30, 0.01); F11(5, 20); F11(30, 20).

`num_gnn_layer` = 2 and K = 5 are held fixed across all cells.

## 4. Structural cells and contrasts

Verified in the X17 code trace of `FMP.emp_forward` (γ = 1/(1+λ2), β = 1/(2γ)):

| cell | λ1 | λ2 | code path |
|---|---|---|---|
| **F00** | 0 | 0 | γ = 1 → `y = hh`, and `x = y` each step: pure MLP |
| **F01(λ2)** | 0 | > 0 | K-step propagation, debiasing branch skipped |
| **F11(λ1,λ2)** | > 0 | > 0 | propagation and debiasing interleaved, same γ as F01 |

**F10 (λ1 > 0, λ2 = 0) is not run.** λ2 is not a pure propagation switch: γ also
scales the debiasing step, so F11 − F10 would not be a clean propagation
contrast. F10 is never called "fairness-only".

    tau_prop(λ2)      = E[ Y(F01(λ2))      - Y(F00) ]
    tau_fair(λ1,λ2)   = E[ Y(F11(λ1,λ2))   - Y(F01(λ2)) ]
    tau_total(λ1,λ2)  = E[ Y(F11(λ1,λ2))   - Y(F00) ]
    identity:  tau_total = tau_prop + tau_fair
               checked <= 1e-12 on the means and in every bootstrap replicate

τ_fair is a **configuration-conditional intervention contrast**, never a
"native-method effect". τ_prop and τ_fair are not claimed to be independent
causal mechanisms.

## 5. Outcomes

Unified evaluator G_c on the test split, decision `score > 0`:

* **primary vector** Y = [AUC, −ΔDP]; **secondary** −ΔEO;
* AUC from the **raw continuous score** (never hard predictions);
* positive-class orientation is fixed in the adapter from the model's own logit
  convention; no orientation flip using test labels;
* DP/EO stored **signed** and as absolute gaps; the main analysis uses the
  project's existing convention;
* undefined EO is recorded explicitly and reported, never silently dropped.

## 6. Selectors

* **Primary: σ_last** (last epoch) — what the official code effectively uses.
* **Robustness: σ_c^BCE** (argmin validation BCE) and **σ_c^AUC** (argmax
  validation AUC), on the same trajectories, identical definitions for F00, F01
  and F11.
* NBA is excluded from any selector analysis (validation = test).
* σ_last is primary; BCE/AUC results are not emphasised to match the NIFTY X25
  finding, and a negative replication is reported as such.

## 7. Splits, runs, RNG

* **6 splits × 5 runs per dataset**, all F cells paired within a (split, run).
* **Split policy:** the loader's own `seed` parameter, extended to
  `seed ∈ {20, …, 25}` (official `main.py` uses 20), `train_ratio=0.8`,
  `test_idx=False`. Verified: pokec_z 8,209/1,026/1,027 and pokec_n
  7,037/880/880 labelled nodes, validation ∩ test = ∅, train ∩ test = ∅,
  deterministic per seed, and different seeds give genuinely different splits
  (test overlap 73–109). **Test data never reaches training or any selector.**
* **RNG contract**, set explicitly per (dataset, split, run): Python, NumPy,
  Torch CPU and Torch CUDA. The official path does not seed the CPU generator
  that initialises `nn.Linear`; the adapter does.
* **Pairing:** within one (split, run), F00/F01/F11 start from **identical MLP
  initialisation**; different runs differ. Gates check both.

## 8. Official-code bug: sensitive-tensor mutation

`get_sen` aliases the sensitive tensor and mutates it in place, so a second
model in the same process debias against a corrupted group vector (X17
diagnostic). **The official source is not edited.** The adapter instead:

* passes each model a **fresh copy** of the sensitive vector;
* keeps an immutable original;
* computes validation/test fairness metrics from the original.

This is recorded as **run isolation / state-mutation correction**, not an
intervention modification. The buggy behaviour is demonstrated once as a
diagnostic and never used for scientific results.

**Gate:** the original sensitive tensor's checksum is identical before and after
run 1, and identical at the start of each of 5 consecutive runs.

## 9. Environment

**Option A (chosen).** Official DGL `GraphConv` runs in an isolated environment
(`/home/sypark/x27_dgl_cuda`: `torch 2.2.2+cu121`, `dgl 1.1.3+cu121`, CUDA
visible), never disturbing the main `dev` environment.

**Equivalence check, recorded with tolerance fixed in advance (1e-5 relative):**
official DGL `GraphConv(weight=False, bias=False, norm='both')` versus a
torch-only `D^-1/2 (A+I) D^-1/2 X`, on the same graphs, features and self-loop
convention (`from_scipy` → `remove_self_loop` → `add_self_loop`):

| case | worst relative difference |
|---|---|
| synthetic (n=64) | 1.11e-07 |
| pokec_z subset 5,000 | 1.56e-07 |
| pokec_n subset 5,000 | 1.08e-07 |
| **pokec_z full (67,796 nodes, 1.30M nnz)** | **2.70e-07** |
| **pokec_n full (66,569 nodes, 1.10M nnz)** | **2.30e-07** |

All within tolerance, so the two implementations are numerically equivalent on
the graphs actually used. Scientific runs use **official DGL**; the equivalence
is corroboration, not a substitute.

## 10. Trajectory and restore contract

Per epoch, per cell: validation BCE, validation AUC, unified DP, EO and AUC,
the epoch index, λ1, λ2, cell identity, split, run, and a non-finite flag, plus
the test raw scores needed to replay σ_last / σ_BCE / σ_AUC.

Gates: restoring a selected checkpoint must reproduce the recorded metrics under
the project's existing noise-aware criterion (hard predictions, DP and EO exact;
AUC within the measured noise-reorderable pair count, X20 form); and inference
determinism is **verified by gate** rather than assumed — if deterministic, no
ξ_eval bundle is needed.

## 11. Instrumentation invariance

Short runs with and without the trajectory hook must agree on training loss,
final raw scores and selected metrics, within a criterion fixed before any
result, and the hook must not change RNG consumption. Failure is a **HARD
STOP**.

## 12. Runtime policy

A pilot runs Pokec-z, 1 split × 1 run, all 7 configurations, purely for timing
(never interpreted). The projection covers 2 datasets × 6 × 5 × 7
configurations. **≤ 24 h projected → proceed with both datasets without further
approval.** **> 24 h → run Pokec-z 6×5 in full, do not start Pokec-n, and report
the projection.** The runtime criterion is independent of any scientific result.

## 13. Statistics

Paired hierarchical bootstrap, **10,000 replicates**, splits resampled then runs
within each drawn split, with F00/F01/F11 sharing the resampling indices inside
each replicate so paired covariance is preserved. The existing 95% percentile
interval, sign stability and resolved rule are reused unchanged; **no new
threshold is created**, and none is adjusted after seeing results.

## 14. Configuration-stability classification (descriptive)

Over the preregistered grid, on τ_fair(−ΔDP):

* **grid-consistent improvement** — every cell resolved in the fairness-improving direction;
* **grid-consistent harm** — every cell resolved harmful;
* **mixed** — resolved effects in both directions;
* **unresolved** — otherwise.

Stated explicitly as a claim about the **preregistered official-grid endpoints**,
never about "all possible λ".

## 15. Interpretation cases (fixed now)

**A** large τ_prop, τ_fair mostly small/unresolved → propagation explains more of
the observed change (never "the fairness mechanism is useless"). **B** τ_fair
consistently improving → mechanistic evidence of incremental benefit at the same
propagation state. **C** τ_fair direction changes with λ → attribution is
configuration-conditional. **D** τ_prop and τ_fair oppose → package-level results
can hide component-level trade-offs. **E** selector changes the τ_fair
conclusion → attribution conditional on selection protocol. **F** stable across
selectors → useful negative replication of the NIFTY selector phenomenon.

**Prohibited:** calling any λ pair native or published; calling the parser
default native; presenting a grid-best result as FMP's performance; calling F10
fairness-only; claiming τ_prop and τ_fair are independent causal mechanisms;
generalising from two Pokec datasets to FMP as a whole; folding FMP into the
benchmark counts; quietly fixing paper/code discrepancies; changing the λ grid
after seeing results.

## 16. Hard stop conditions

Official λ candidates not reliably recoverable; F00/F01/F11 structure differing
from the audit; sensitive-tensor mutation not eliminable by run isolation;
initialization pairing failure; GraphConv equivalence failure; node-alignment
mismatch; test leakage; non-finite values; restore-fidelity failure;
instrumentation changing training; or an unexplained difference between the
adapter and official executable behaviour. **Disliking a scientific result is
not a stop condition**; positive, negative and mixed outcomes are all reported.

## 17. Outputs

`harness/results/x27/` for all data and tables; figures under
`harness/results/figures/` with the `x27_` prefix. Tables 1–5 and Figures 1–2 as
specified in the task, plus the EO and selector-robustness supplements.
