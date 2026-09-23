# X30 — Method-coverage extension protocol

**Scientific status.** X22–X27 (core), X28 (paper figures and tables) and X29
(dataset-coverage extension) are FROZEN and are not modified, re-run or
re-analysed here. X30 is a **post-hoc, non-core method-coverage extension**. The
accurate statement about its timing is:

> the method-coverage protocol, the admitted cell list, the adapters and the
> analyzer were committed after a complete recovery pass and before any X30
> outcome (AUC, DP, EO or any τ) was computed or viewed.

Scope: every fairness-aware method present in the repository **other than** the
four frozen core methods (FairGNN, NIFTY, FairVGNN, FairGB). FairGate is excluded
from this study by user decision and appears in no roster, count or headline.

Written 2026-09-17. GPU policy: every X30 job runs with `CUDA_VISIBLE_DEVICES=2`
(enforced by the runner, which refuses any other setting).

---

## 1. What "admitted" means

A (method, dataset) cell is admitted only if **all** hold, each checked without
viewing any outcome:

1. **I is identified from repository evidence**: a code path the repository
   itself switches (a flag, a coefficient, a loop bound, a stage the method's
   own pipeline consumes), with M^{+I} and M^{-I} differing in nothing else.
2. **M^{+I} is a repository configuration**, not one we chose. Where the
   repository offers several, a mechanical rule fixed here picks one.
3. **The dataset is repository-common** (`utils/data.py:get_dataset`) and the
   method's repository supports it; no dataset is ported.
4. **No test leakage** in training, in the intervention, or in selection.
5. **Outcome-blind smoke gates pass** (section 8): imports, dimensions, finite
   logits over the whole trajectory, trajectory completeness, the intervention
   code path toggles (active in M^{+I}, provably absent in M^{-I}), paired arms
   start from identical parameters (or identical generator state where the
   intervention changes a layer's shape), split identity with the common split.

Allowed recovery (all recorded): isolated environments, dependency installs,
path/device repair, checkpoint and cache namespace separation, wrappers, seed and
split injection, output extraction, I/O adapters. Not allowed: rewriting an
algorithm, inventing a hyperparameter, replacing a scientific component, tuning
on test, suppressing NaNs, clamping or rescaling an estimator, porting a dataset.

**Dataset-level exclusion (all methods).** nba: under the common split the
validation set is contained in the test set (`val ∩ test = 213 =` all of
validation), so any selection on validation reads test nodes. No nba cell is
admitted for any method.

## 2. Recovery pass — verdicts per method

Evidence files: `harness/results/x30/smoke/*.log` (gate outputs),
`harness/results/x30_env_edits_fairedit.log` (environment recovery),
`harness/results/x30/bind_recovery/RECOVERY_LOG.md`,
`harness/results/x30/X30_SOURCES.sha256` (every external file an adapter executes
or reads, hashed at freeze).

### 2.1 FairSIN — ADMITTED (controlled + native). X2 exclusion not inherited.

* The X2 rationale (unnormalised sum aggregation) is **wrong on the code**:
  `propagate2` applies symmetric degree normalisation
  (`FairSIN-main/utils.py:29-41`, `deg_inv_sqrt[row] * deg_inv_sqrt[col]`).
  German's features are unnormalised by the authors' own loader
  (`dataset.py:462`). The X2 German/GCN collapse is therefore a property of that
  published configuration, not a defect that bars the method.
* **I** = the authors' neutralisation term `x + delta * mlp(x)` (in-train.py:179;
  train_mlp.py multiplies by a degree term `lam`) together with the
  discriminator step `if args.d == 'yes'` (in-train.py:189).
* **Configuration rule.** The authors' own ablation block
  (`experiment.sh:50-116`) is parsed programmatically
  (`adapters/x30_fairsin.py:config`). For each (dataset, encoder), **M^{+I}** is
  the first row with `delta > 0` and `d == 'yes'`, else the first row with
  `delta > 0`; **M^{-I}** is that row with `delta = 0`, `d = 'no'`. Parser
  defaults are read from each script's argparse with `ast`, with declared types.
  pokec rows have no `d = 'yes'` full row, so there I is the neutralisation only.
* **Execution.** The official `run(data, args)` of `in-train.py` or
  `train_mlp.py` (whichever the row names) is executed unmodified. Wrappers:
  `train_test_split(random_state=seed)` (the official call is unseeded);
  `evaluate` wrapped to record full-node logits exactly as it computes them; the
  heterophilous-neighbour cache computed once per dataset by the official
  cache-miss branch in `harness/results/x30/cache/fairsin/`, after which every arm
  takes the official cache-hit branch (the miss branch mutates `data.adj`, which
  `train_mlp.py` reads); module import isolated from this repository's `utils`
  package; `CUDA_VISIBLE_DEVICES='6'` in the script body neutralised by binding
  CUDA before import; `memory_profiler` stubbed.
* **Gates (split 20, run 0, H = 200), all 15 (dataset, encoder) pairs:** see
  `smoke/fairsin_smoke.log`. Recorded diagnostic, not a gate: German/GCN M^{+I}
  gives a constant validation decision in 81 % of epochs.
* **Native**: validation-only tradeoff selector (in-train.py:227-233, strict `>`
  with floor 0), native horizon = the M^{+I} row's `--epoch(s)`; leakage-free.
  VALID-NATIVE for all 15 pairs.
* Datasets: german, bail, credit, pokec_z, pokec_n. Not pokec_*_g (sensitive
  index hard-coded to region), not nba (§1), not income (no loader).

### 2.2 EDITS — ADMITTED (controlled only)

* Source: `algorithms/EDITS.py` (repository wrapper; the upstream EDITS
  repository is not vendored), driven as `utils/train_baselines.py:567-587`.
* **I** = the debiasing stage `EDITS.fit`: attribute debiasing (then the
  `truncation = 4` lowest-weight columns zeroed) and structural debiasing
  (`Adj_renew`, binarised at `threshold_proportion` inside `predict`).
* **M^{-I}** = the stage bypassed: `X_debiased` = the same preprocessed features,
  `adj1` = the original adjacency; `predict` then edits nothing
  (`the_con1 ≡ 0`, checked: 0 edited entries). EDITS is a pre-processing method;
  the downstream GCN in `predict` consumes whatever the stage produced.
* **Routing repair (recorded).** train_baselines maps param.json `dropout` onto
  `args.dropout`, which the EDITS branch never reads; `threshold_proportion`
  then becomes `param1` and is passed as the debiaser's dropout, and `predict`
  receives the placeholder `threshold_proportion = 1`, under which no edge can
  ever be edited. The wrapper's positional intent (param1 = dropout,
  param2 = threshold_proportion, param.json key order) is restored. Both values
  come from param.json.
* Seed injection before `predict` in both arms: `EDITS.optimize` reseeds every
  generator to 10 each epoch, in M^{+I} only.
* Configuration provenance: **local-unverified** (param.json). Native:
  D (no official configuration in the repository).
* Datasets: german, bail, credit (the param.json entries).

### 2.3 FairEdit — ADMITTED (controlled only)

* Source: `algorithms/FairEdit.py`, driven as `train_baselines.py:550-565`.
* **I** = gradient-guided graph editing `fair_graph_edit`, called while
  `epoch < edit_num` (FairEdit.py:780). **M^{+I}** `edit_num = 10` (the fit()
  default the wrapper uses); **M^{-I}** `edit_num = 0`.
* Controlled horizon passed to `trainer.train(epochs=H)` (predict() hard-codes
  100). Private working directory per process.
* Provenance local-unverified (param.json weight_decay, hidden); native D.
* Datasets: german, bail, credit.

### 2.4 BeMap — ADMITTED (controlled only)

* Source: `BeMap-main` (models.py, train_bemap.py) in `/home/sypark/x27_dgl_cuda`.
* **I** = balance-aware per-epoch subgraph sampling. **M^{+I}**
  `train_bemap.train(epoch, ...)`; **M^{-I}** `train_bemap.train(-1, ...)`, the
  model's own full-graph branch (`BeMap_GCN.forward`, `epoch == -1`), which is
  also what inference always uses. Configuration: README command at parser
  defaults.
* Native: **B. NATIVE-INVALID** — `train_bemap.py:136-142` selects on the test
  split. The official `test()` is not called.
* The module-level `np.random.seed(53)` / `torch.manual_seed(53)` at import is
  isolated (generator state saved and restored) so it cannot break pairing.
* Datasets: bail, credit, pokec_z (nba excluded, §1).

### 2.5 GEAR — ADMITTED on bail only (controlled only), with an asset caveat

* Source: `GEAR-main/src/main.py` functions, parser defaults, `--dataset bail`.
* **I** = the counterfactual-consistency objective, `sim_coeff` 0.6 → 0.
  Recorded properties of this switch, not repaired: the classification loss is
  scaled by `(1 - sim_coeff)`, and at 0 `optimizer_1` still takes Adam steps
  driven by weight decay alone on the shared encoder.
* **Assets.** main.py loads the released `graphFair_subgraph/aug/*.pkl` by
  default. Verified (and re-verified every arm, hard stop otherwise): for bail
  the three files are **identical**, have the common bail edge set exactly, have
  non-sensitive columns identical to GEAR-normalised common features, and have
  the sensitive column equal to `1 - s`. The code labels aug_1 / aug_2 as
  sens_rate 0.0 / 1.0, so the released assets do not match the code's own
  specification; the CFDA/CFGT generator that would build genuine
  counterfactuals has no released checkpoint. **The GEAR cell therefore
  attributes the similarity objective given a sensitive-attribute-flip
  counterfactual on an unchanged graph**, and is reported with that caveat.
* credit: **E** — the released credit assets are on a different graph (3.0 % of
  the common credit edges). german: **E** — no released assets.
* Native: **E** — the published GEAR cannot be reproduced without genuine
  counterfactual assets.
* Horizon: GEAR's `epochs` are mini-batch steps (batch 100); H = 200 steps under
  the controlled protocol, native 1000.

### 2.6 FairGT — EXCLUDED (C. structurally invalid)

`FairGT-main/train_fairgt.py:84-103` applies the same-sensitive complete graph
and the eigenvector encoding unconditionally for `--model fairgt`; the authors'
code has no off-state. The `eig` / `sgr` names in `train_baselines.py`'s
`--components_off` help text were added by this study (commit `b396f76`, X2a) and
are not implemented in `algorithms/FairGT_alg.py`. Any M^{-I} would be a
constructed substitution (A_s → A) that also changes the feature scale.

### 2.7 BIND — ADMITTED on bail and income (controlled only)

* Source: `BIND-main/implementations` (official), in `/home/sypark/x27_dgl_cuda`;
  adapter `harness/adapters/x30_bind.py` executes the official scripts' own lines
  (compiled with their real file names and line numbers). Full recovery record:
  `harness/results/x30/bind_recovery/RECOVERY_LOG.md`.
* **The earlier claim that BIND's LiSSA estimator diverges was a property of the
  X2 transcription, not of the official code.** With device-placement shims only
  (S5: `sens` onto the device the function already uses for adj/features/labels;
  S6 not needed in the adapter), the official influence step completes
  5000/5000 HVP steps with 0 non-finite values on bail and income, at the code
  default scale 60 **and** at the README-published scale 25
  (`BIND-main/README.md:55`, "for Income and Recidivism, set the scale ... as
  25"). Scale 25 is used because it is the authors' instruction, not a repair.
* **Test leakage in the published estimator, and its removal.** The fairness
  cost is computed on `idx_test` and reads test labels
  (`2_influence_computation_and_save.py:173` → `approximator.py:149, 131-133`).
  L1 passes the script's own validation indices in that argument instead
  (asserted to replace exactly `idx_test_vanilla`; validation ∩ test = 0). The
  estimator, damp 0.03, recursion depth 5000 and `cal_influence_graph` are
  unchanged; with L1 the estimate is finite on both datasets. Native: **B.
  NATIVE-INVALID** (the published estimator reads test labels).
* **I** = influence-guided deletion of training nodes, official
  `3_removing_and_testing.py --helpfulness_collection 1` ordering.
  **Stage A** (shared by both arms, cached per dataset/split/seed, hash-checked):
  `1_training.py` at parser defaults on the common split, then the influence
  step (H1 scale 25, L1). **Stage B** (the arm): the official loop body for one
  k, then the official `train(epoch)` for H = 200 epochs (native 30).
  **M^{-I}** k = 0; **M^{+I}** k = round(p · |train|).
* **Budget rule.** The authors report "BIND 1%" and "BIND 10%" as k = 10 and 100
  of their 1000 training nodes (`debiasing_gnns.py`). On the common split
  |train| differs (bail 100), so the budget is the published **fraction**:
  p = 0.01 (**primary**, BIND-1pct, the first reported) and p = 0.10 (variant,
  BIND-10pct, reported apart and never pooled).
* Wrappers W1–W3 inject the common data into BIND's loaders, `get_adj` and
  `del_adj` (lines 175-183 verbatim on the common adjacency); each script keeps
  its own feature preprocessing. Deleted (training-only) nodes carry logit 0 in
  the stored trajectory; selection and evaluation read validation and test only.
* pokec1 / pokec2: not admitted (their zip data are not the repository-common
  pokec loaders; not ported).

### 2.8 FMP — no new X30 cell; X27 stands

`FMP-main/run_fgnn.sh` publishes a grid (λ1 ∈ {5, 15, 20, 30}, λ2 ∈ 0…20,
num_gnn_layer ∈ {2, 5}), not a configuration. Any package-level M^{+I} would be
our choice. X27 already covers FMP as a component case study at the grid's
endpoints on pokec_z / pokec_n. The bundled baselines `adv_gnn.py`, `reg_gnn.py`,
`Fmix_gnn.py` have `--hyper_reg default=0.0` and no published non-zero value (D).
nba excluded (§1).

### 2.9 Not candidates

* FairWalk, CrossWalk — C: sensitive groups are random in the wrapper and the
  `run()` entry is dead code.
* GNN_cf, NIFTY_cf — D: counterfactual-evaluation variants of the core NIFTY
  mechanism, not separate fairness interventions.

## 3. Admitted cells

### 3.1 METHOD EXTENSION — primary (one backbone per method × dataset)

| # | method | dataset | backbone | env |
|---|---|---|---|---|
| 1–5 | FairSIN-GCN | german, bail, credit, pokec_z, pokec_n | GCN (scatter) | dev |
| 6–8 | EDITS | german, bail, credit | GCN (EDITS.py) | x30_edits_env |
| 9–11 | FairEdit | german, bail, credit | GCN (FairEdit.py) | x30_edits_env |
| 12–14 | BeMap | bail, credit, pokec_z | BeMap_GCN (DGL) | x27_dgl_cuda |
| 15 | GEAR | bail | SAGE (GEAR Encoder) | x30_edits_env |
| 16–17 | BIND-1pct | bail, income | GCN (BIND GNNs/gcn.py) | x27_dgl_cuda |

FairSIN's primary backbone is **GCN**, fixed now: it is the backbone of B and of
the core roster. BIND's primary budget is 1 % (section 2.7).

### 3.2 METHOD EXTENSION: VARIANTS (reported apart, never pooled)

FairSIN-GIN and FairSIN-SAGE on german, bail, credit, pokec_z, pokec_n
(10 cells), and BIND-10pct on bail and income (2 cells). They share data,
splits and B (and for BIND the Stage-A model and M^{-I}) with the primary cell of
the same dataset and are not independent evidence.

### 3.3 METHOD EXTENSION: NATIVE

FairSIN × {GCN, GIN, SAGE} × {german, bail, credit, pokec_z, pokec_n}
(15 cells), native horizon, code-native selector replayed as `m1pub`, B at 200.

## 4. Controlled protocol (identical to the core wherever the core fixes a choice)

| item | value |
|---|---|
| arms per unit | B (algorithms/GNN.py at `published("GNN", dataset)`), M^{-I}, M^{+I} |
| horizon | H = 200 for all three (method epochs = the code's own epoch unit) |
| splits × runs | splits 20–25 × runs 0–4 = 30 units per cell; seed = 27 + run |
| pairing | both method arms start from `seed_all(seed * 1000 + split)`; the gate requires identical initial parameters (EDITS: identical generator state, since M^{+I} removes input columns) |
| trajectory | full-node logits after every training epoch, stored in memory; the validation slice feeds a `ValidationHistory` (no test field); σ picks the epoch; only then is that epoch's test slice read |
| selectors | σ_c^BCE primary, σ_c^AUC robustness (the frozen `ValidationHistory` slots, strict comparisons, earliest tie) |
| evaluator | the common `G_c`, decision `score > 0` |
| coordinates | ΔAUC and −ΔDP primary; −ΔEO secondary |
| persistence | the X15 `CellStore` contract, protocol key `x30` |

## 5. Native protocol (FairSIN only)

Protocol key `x30native`. Method arms at the native horizon (§2.1), everything
else as §4. `m1pub_*` = M^{+I} at the code-native selector on the same stored
trajectory; `pub_*` = that against B at B's own selector. A unit where the native
selector never fires (the authors' code then raises `UnboundLocalError` after
training) is recorded with `code_epoch = -1` and NaN `m1pub_*`, and counted.

## 6. Estimands

    tau_nonint = Y(M^{-I}) - Y(B)
    tau_I      = Y(M^{+I}) - Y(M^{-I})
    tau_pkg    = Y(M^{+I}) - Y(B)

The identity `tau_pkg = tau_nonint + tau_I` is checked per row.

## 7. Uncertainty and the resolved rule — unchanged

`bootstrap_armA.boot`, 10,000 paired hierarchical replicates, seed 20260914
(splits, then runs within split, each unit carried whole). **Resolved** = sign
stability ≥ 0.75 **and** |mean| ≥ 0.010 **and** the 95 % interval excludes 0
(`analyze_armA.SIGN_MIN`, `NEAR_ZERO`). Not restated as new, not adjusted.

## 8. Gates

**Admission (smoke, split 20 / run 0 / H = 200), outcome-blind.** Per arm:
finite logits at every epoch; trajectory length H; validation decisions not
constant at every epoch. Per method: the intervention toggles (FairSIN:
neutralisation term non-zero / exactly zero, discriminator calls iff `d = yes`;
EDITS: stage ran and changed features or structure / stage bypassed and
adjacency and features are the originals; FairEdit: `fair_graph_edit` called 10
times and edges changed / never called and unchanged; BeMap: a subgraph drawn
every epoch and smaller than the full graph / never drawn; GEAR: similarity
objective active / `sim_coeff = 0`; BIND: Stage-A influence finite, fairness
cost on validation nodes disjoint from test, exactly k > 0 training nodes deleted
/ none deleted, validation and test never deleted, identical Stage-A checkpoint
and parameters); paired initialisation; split identity.

**Per unit in the full run (hard stop of that cell, nothing persisted):** split
identity; finite logits; trajectory length; intervention toggles; paired
initialisation. The constant-decision fraction is **recorded** per unit
(`constsign_plus`, `constsign_minus`) and never stops a cell.

## 9. Execution (GPU 2 only, autonomous after freeze)

Three streams, each sequential, resumable:

* **S1 (dev):** FairSIN-GCN german → bail → credit → pokec_z → pokec_n; then
  FairSIN-GIN, FairSIN-SAGE in the same dataset order; then native, same order.
* **S2 (x30_edits_env):** EDITS german → bail → credit; FairEdit german → bail →
  credit; GEAR bail. FairEdit/credit peaks near 39 GB on GPU 2 and runs only
  while S1 and S3 hold small allocations; if it cannot allocate, S2 waits and
  retries rather than moving GPUs.
* **S3 (x27_dgl_cuda):** BeMap bail → credit → pokec_z; then BIND-1pct bail;
  then BIND-10pct bail.
* **S4 (x27_dgl_cuda), BIND income.** On the common split income has 4605
  training nodes, so the official per-node influence loop takes about 2.5 h per
  Stage A; the admission smoke hit its 90-minute timeout (rc 124) for cost, not
  for any gate. It is therefore run **first in S4 without a timeout**; its Stage-A
  cache (split 20, seed 27) is reused by the first real unit. Only if every gate
  passes do the income cells start: six processes, one per split, each writing
  its own `x30_BIND-<budget>_income_s<split>.csv` (1pct, then 10pct, which reuses
  the Stage-A caches). If the smoke fails, both income cells are stopped and
  reported as unknown.

Queue order is fixed here and never changed by any result. The first completed
unit of each cell may be timed for cost only.

## 10. STOP rules — per cell, hard; a stopped cell does not block others

Pairing impossible; split or node misalignment; test leakage discovered; any
non-finite logit; RNG pairing failure; an asset verification failure (GEAR);
instrumentation shown to change training; configuration unclear; a BIND
Stage-A cache hash mismatch or non-finite influence. A stopped cell
is reported as unknown, never as a negative finding. Thresholds, repeat counts
and gates are not changed after a gate result.

Global STOP (report to the user, do not proceed): a needed change to scientific
assumptions, an invalid intervention definition, a fundamentally unavailable
asset for an admitted cell, unremovable test leakage, a protocol amendment, or
any risk to a frozen result.

## 11. Analysis, strata and paper classification

`harness/experiments/analyze_x30.py` (committed with this protocol), which imports
the frozen construction, bootstrap and resolved rule and the X29 views
unchanged, and refuses an incomplete store. Strata, never silently merged:

    CORE                        frozen 12 (X22)
    DATASET EXTENSION           X29, 4 cells
    METHOD EXTENSION            X30 primary cells (section 3.1)
    METHOD EXTENSION: VARIANTS  section 3.2
    METHOD EXTENSION: NATIVE    section 3.3
    COMPONENT CASE STUDIES      X24–X27, referenced only
    COMBINED DESCRIPTIVE        CORE + DATASET EXTENSION + METHOD EXTENSION

Reported per stratum: cell count, τ_I and τ_nonint on −ΔDP / ΔAUC / −ΔEO, count
of |τ_nonint| > |τ_I|, count of resolved τ_I, intervention quadrant, selector
agreement. Counts are counts of these cells, not rates over fair-GNN research.

**Paper classification, fixed now:** CORE stays MAIN and is unchanged by X30.
METHOD EXTENSION primary → APPENDIX (a single coverage sentence may appear in
the main text, pointing to the appendix). NATIVE → APPENDIX. VARIANTS →
REPRODUCIBILITY ONLY. GEAR/bail → APPENDIX with the asset caveat in the same
table row. Excluded methods → listed with reasons in the appendix inventory
(EXCLUDED). No X30 number may be substituted for a core claim.

Outputs: `harness/results/x30/` (`x30_<method>_<dataset>.csv`,
`x30native_FairSIN-<enc>_<dataset>.csv`, analysis text, summary and cell tables,
checksums).

## 12. Do not

* modify X22–X29 or overwrite a frozen CSV
* select, reorder or drop cells by result
* port a method to a dataset it does not support
* invent a configuration, or choose among repository configurations by result
* clamp, rescale or regularise an estimator to obtain finite values
* select on test, or report a leaking native protocol as valid
* inflate a tolerance, or change a gate after seeing it
* include FairGate anywhere in X30

---

## Amendment 1 — execution order only (2026-09-18, before any X30 outcome)

By user instruction, cells on the core datasets (german, bail, credit) run
before cells on pokec_z, pokec_n and income. No X30 outcome had been analysed or
viewed; the instruction does not depend on any result. Nothing scientific
changes: the admitted cells, configurations, gates, estimands, analysis and
strata are exactly as frozen at `32aba55`. Units already persisted are kept
(CellStore resume). Running cells were not interrupted; only the queue order was
replaced (`harness/experiments/x30_streams_v2.sh`):

* **S1:** FairSIN {GCN, GIN, SAGE} × {german, bail, credit}; then native on the
  same; then {pokec_z, pokec_n} controlled; then their native cells.
* **S2:** unchanged (all its cells are on core datasets).
* **S3:** BeMap bail, credit; BIND-1pct bail; BIND-10pct bail; then BeMap pokec_z.
* **S4:** the BIND income admission smoke completes as before; the income cells
  start only after the core-dataset cells of S1 and S3 are finished.

## Amendment 2 — pokec cells paused (2026-09-18, before any X30 outcome)

By user instruction, the pokec_z / pokec_n cells are **paused** so that the
german / bail / credit cells finish first. No X30 outcome had been analysed or
viewed, and the instruction does not depend on any result. The pokec cells
remain admitted exactly as frozen and are run afterwards; nothing is dropped and
no cell is selected by its result.

* The running `x30native FairSIN-GCN pokec_z` cell was stopped at 14/30 units;
  its persisted units are kept and will be resumed (CellStore).
* `x30_FairSIN-GIN_pokec_z` had failed at its first unit with a CUDA
  out-of-memory error while FairEdit/credit held 38 GB of GPU 2 (protocol 9:
  wait and retry on GPU 2, never move GPUs). It persisted no unit and is
  re-queued with the other pokec cells.
* `harness/experiments/x30_sweep.sh` re-runs any incomplete admitted cell after
  the streams end, checking free GPU memory first; `DS_SWEEP` restricts what it
  may start (german bail credit while this amendment holds).
* Logging fix: `echo "... rc=$?"` after a `$( )` substitution reported the
  substitution's status, so the OOM above was logged as `rc=0`. Fixed in the
  stream and sweep scripts; no result is affected.
* BIND/income (not a pokec cell) continues; it is unaffected by this amendment.

### Amendment 2a — pokec cells resumed (2026-09-19, user instruction)

The pause of amendment 2 is lifted and the seven remaining pokec cells are run:
`x30 FairSIN-GIN pokec_z` (the cell that had failed with a CUDA OOM, 0 units
persisted) and the six FairSIN native pokec cells (`x30native FairSIN-{GCN,GIN,
SAGE} × {pokec_z, pokec_n}`, of which GCN/pokec_z resumes from 21/30). Run
sequentially by `x30_sweep.sh` with `DS_SWEEP="pokec_z pokec_n"`, on GPU 2, with
the free-memory check before each cell; nothing else changes. The core-dataset
cells were complete and their interim analysis was already reported, so no cell
is selected, ordered or dropped by any result.
