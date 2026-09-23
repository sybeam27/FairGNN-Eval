# X31 — Additional cells after the method-coverage extension

Frozen before any X31 outcome is computed or viewed. Nothing in X22–X30 is
modified, re-run or re-analysed; X31 adds cells and one analysis stage, each
admitted by the same outcome-blind rules as X30 (repository evidence for the
intervention and its configuration, no test leakage, smoke gates that read no
outcome). GPU 2 only. Every section below is committed before its runs start.

Decisions taken by the user before any X31 run: no FairGNN native cells; FnRGNN
is included as a task-adapted stratum; order is (1) the FMP baseline stage,
(2) SFG controlled cells, then the rest.

## 1. FMP component case study: a baseline stage (committed with the runner)

The FMP component study runs on FMP's own split (its own loader, `seed=split`)
and horizon (300 epochs). The common baseline B was trained on the common split,
whose test nodes differ (pokec_z 2,566 vs 1,027; pokec_n 2,200 vs 880), so B
could not be paired with the FMP arms. X31 trains B under exactly the FMP arms'
setting, turning the chain into four stages:

    B  ->  base (F00: no propagation, no fairness)  ->  +propagation (F01)
       ->  +propagation+fairness (F11)

* **B's configuration:** `published("GNN", dataset)`, as for every other cell.
* **Shared with the FMP arms:** data and split (`x27_fmp_run.load_split`),
  seeding (`seed_all(seed*1000 + split)`, `seed = 27 + run`), horizon (300
  training epochs), selectors (`last`, validation BCE strict-min, validation AUC
  strict-max, earliest tie) and evaluator (`x27_fmp_run.metrics`).
* **Runner:** `harness/experiments/x31_fmp_baseline_run.py`.
* **Cells:** pokec_z and pokec_n, splits 20–25 × runs 0–4.
* **Gates, per unit:** the test node ids equal the FMP arms' stored ids for the
  same (split, run); finite scores; 300 trajectory records. A failure stops the
  cell and persists nothing. Smoke (split 20, run 0) passed on both datasets.
* **New step:** `B -> base` = F00 − B per matched unit and selector, summarised
  with the unchanged paired hierarchical bootstrap (10,000 replicates, splits then
  runs) and resolution rule, seeded like the FMP analysis it extends
  (`analyze_x27_fmp.SEED` = 20260919). An earlier draft of this line said
  20260914 (the controlled-cell seed); the analysis committed before it ran uses
  20260919, and this line is corrected to match it. The existing `prop`, `fair`
  and `total` steps are not recomputed.
* **Output:** `harness/results/x31/x31_fmp_B_<dataset>.csv` (x27 schema,
  `config = "B"`).
* **Analysis:** `harness/experiments/analyze_x31_fmp_baseline.py` (committed before
  it was run) reports `base = F00 - B` and `pkg = F11 - B` per λ setting, with the
  FMP analysis's own `stat`, `boot`, seed and 10,000 replicates, and checks
  `pkg = base + total` per unit and in every replicate.

## 2. SFG

* **Source:** `models/SFG-main` (official), adapter `harness/adapters/x31_sfg.py`,
  which executes the official `run()` of the script named on the row
  (`sfg.py` for german and bail, `sfg_credit.py` for credit).
* **Intervention and its off-state, from the authors' `run.sh` ladder:**
  M^-I = the row the authors label FairVGNN (no `--with_constraint`),
  M^+I = the SFG row (`--with_constraint --rho=2 --loss_alpha=0.5`). The
  adapter asserts the two rows differ only in `with_constraint`, `rho` and
  `loss_alpha`. The generator loss goes through `loss_chisq` whenever
  `f_mask == 'yes'`, in both rows; the unconstrained row does so at its default
  `loss_alpha = 1`. So I = the Lipschitz constraint (rho = 2) together with the
  stronger DRO weighting (loss_alpha 1 -> 0.5).
* **Data:** the common split at SFG's own feature policy; `corr_idx` from SFG's
  `sens_correlation`; `x_min`/`x_max` from the raw features; `train_ratio` /
  `val_ratio` as `sfg.py:447-453` builds them.
* **Controlled cells:** german, bail, credit; H = 200; selectors and evaluator
  as every controlled cell; SAGE encoder (the rows' own).
* **Native cells (later):** horizon from the row (german 200, bail 160,
  credit 200); selector = the script's validation tradeoff (strict `>`, floor 0).
  The early-stopping counter reads test accuracy only once
  `epoch >= pretrain (200)`, which no published horizon reaches, so the native
  selection is validation-only.
* **Gates (outcome-blind), per arm and per unit:** finite logits; trajectory
  length; M^+I applies the constraint and calls `loss_chisq` at loss_alpha 0.5
  only; M^-I applies no constraint and calls it at loss_alpha 1 only; the paired
  arms start from identical parameters of every module. A first smoke used a
  wrong expectation (no `loss_chisq` call at all in M^-I); the code routes the
  generator loss through it in every row, so the gate was corrected to the
  alpha value before any outcome was read. German smoke: all gates pass.
* **Output:** `harness/results/x31/x31_SFG_<dataset>.csv`, protocol key `x31`.

## 2b. FairVGNN configuration variation (robustness cells)

The primary FairVGNN cell per dataset is the official GCN row at the default
propagation. The authors' run scripts publish further full rows, each with its
own both-off ablation. X31 runs every such row as a robustness cell, never
counted in the primary summary:

    german  GCN-spmm, GIN, SAGE      bail  GCN-spmm, SAGE      credit  GIN, SAGE

* **Runner:** `harness/experiments/x31_fairvgnn_config_run.py`, which imports the
  primary cells' path from `pilot_tau` unchanged (loader, B at
  `published("GNN", ds)` trained first in the same process, `train("FairVGNN")`
  with bundle replay, `eval_rng_seed` test evaluation, paired seeding, CellStore).
* **Configuration:** read from the row with `published()`'s key list and the
  `fairvgnn_main.py` parser defaults for keys the row omits; `encoder` from the
  row; feature normalisation as the primary cell of the same dataset.
* **Off-state:** `f_mask='no', weight_clip='no'` with everything else held at the
  full row (the controlled rule; the authors' ablation rows retune and are not
  used as M^-I).
* **Controlled:** H = 200, sigma_c^BCE / sigma_c^AUC, common evaluator. The credit
  rows run through the same wrapper as the primary credit cell; the credit
  script's `clip_c` belongs to its native loop and is recorded only.
* **Gates (outcome-blind):** normalisation leaves labels/splits unchanged; B and
  both arms hold both selector slots; finite test scores. Failure stops the cell.
* **Output:** `harness/results/x31/x31_FairVGNN-<config>_<dataset>.csv`, method tag
  `FairVGNN-<config>`, protocol key `x31`.
* **Native: withdrawn (user decision, 2026-09-22 01:05 KST).** The user decided
  not to run these cells. The decision was taken before any of their outcomes
  was read; the cells had just started and none was complete. The three
  one-unit partial files were moved unread to
  `harness/results/x31/withdrawn_fairvgnn_native/` and are never analysed.
  FairVGNN's native comparison remains the three frozen targeted cells of its
  primary row. The design these cells would have followed is kept below for the
  record.
* **Native (systematic, withdrawn):** `--native`, protocol key `x31native`, output
  `x31native_FairVGNN-<config>_<dataset>.csv`. Each row runs at its own horizon
  (german GCN-spmm 400, GIN 200, SAGE 200; bail GCN-spmm 300, SAGE 200; credit
  GIN 100, SAGE 200) with the method's own validation selection for M^+I.
  The native changes the primary FairVGNN native cells receive
  (`native_config`) are applied to the row:
  * credit trains with the `fairvgnn_credit.py` loop, using the row's own
    `--clip_c`, or else that script's parser default;
  * feature normalisation follows the official `dataset.py` rule.
  B stays the fixed controlled reference B_200. tau_I is a within-protocol
  contrast (sigma_c at the native horizon, as for every native cell).

## 2c. BeMap with GAT (robustness cells)

* **Source:** the README names GCN and GAT; `train_bemap.py --model gat`
  (lines 115-118) builds `BeMap_GAT` with the same arguments as the GCN branch.
  All other settings stay at the parser defaults of the primary BeMap cells.
* **Adapter:** `harness/adapters/x30_bemap.py` with `--encoder GAT` selecting that
  branch; `--encoder` omitted keeps the primary GCN path byte-for-byte.
* **Intervention and off-state:** unchanged (balanced subgraph per training
  epoch vs the model's own full-graph branch).
* **Inference dropout:** `BeMap_GAT.forward` calls `F.dropout(h, p)` without
  `training=`, so dropout is active at evaluation too. This is the official code
  and is kept. The adapter makes one evaluation forward after every training
  step, as the official loop's `test()` does, so the random stream is consumed
  as in the official script, identically in both arms.
* **Cells:** bail, credit, pokec_z (BeMap's supported datasets in the study),
  controlled only. There is no native cell because the official loop selects on test.
* **Output:** `harness/results/x31/x31_BeMap-GAT_<dataset>.csv`, protocol key `x31`.

## 2d. FairGNN upstream GCN configuration (robustness cells)

The primary FairGNN pokec cells keep the frozen `utils/param.json` configuration
(user decision). The upstream repository (EnyanDai/FairGNN @ 13cdca7, copied to
`harness/provenance/fairgnn_upstream/`) publishes one GCN row per pokec dataset.
X31 runs that row as a robustness cell: pokec_z alpha 100, beta 1; pokec_n
alpha 50, beta 1.

* **Runner:** `harness/experiments/x31_fairgnn_config_run.py`. It uses the same
  `pilot_tau` path as the primary cells, with B trained first in the same process.
* **Applied from the row / parser defaults:** alpha, beta, lr 1e-3,
  weight_decay 1e-5.
* **Recorded, not applied:** these are fixed by the shared path for primary and
  variation cells alike.
  * `--acc`/`--roc`: thresholds of the script's own selection, which is replaced
    by sigma_c.
  * `--epochs`: the native horizon.
  * `--num-hidden 128`: the wrapper builds its GNN and classifier at 64 inside
    `__init__`, and this holds equally for the primary cells.
  * `--sens_number 200`: the controlled protocol gives every method the training
    nodes' sensitive attribute.
* **Off-state:** alpha = beta = 0. Controlled, H = 200. No native cell (user
  decision).
* **Output:** `harness/results/x31/x31_FairGNN-upstreamGCN_<dataset>.csv`.
* **Upstream GAT rows** (`--model GAT`, tag `FairGNN-upstreamGAT`, env
  `/home/sypark/x27_dgl_cuda`): pokec_z alpha 10, beta 0.01; pokec_n alpha 4,
  beta 0.01, with the same applied / recorded split as the GCN rows.
  * **GNN:** for that process only, the wrapper's `get_model` is swapped for the
    upstream `GAT_body` (`models_GAT.py` in provenance). It uses the row's
    architecture: 1 layer, heads [1, 1], feat_drop 0.5, attn_drop 0,
    negative_slope 0.2, no residual, num_hidden 64. That is the width the
    wrapper already builds its classifier and adversary at, so no other module
    changes. The sensitive-attribute estimator stays the wrapper's GCN.
  * **Graph:** the upstream one, i.e. the common edges plus self loops.
  * **Extra gate:** the upstream GAT is built for both arms and called on every
    step.
  * **Output:** `x31_FairGNN-upstreamGAT_<dataset>.csv`.

## 3. FnRGNN, task-adapted stratum (harness-completed loop)

User decision (after the admission check below was reported): FnRGNN is
included. It is reported in its own stratum, never pooled with or counted in
the primary summary, with the caveat stated on every row.

**Admission check (recorded before any run).** `models/FnRGNN-master` (CIKM
2025) is a node-regression method. It releases the model class
(`utils/model.py:FnRGNN`), `logs/best_configs/*.json` and weight-only
checkpoints. It releases no training loop. `FnRGNN.optimize()` fits every row
of `data.y` without a mask, and the released `load_and_prepare_dataset` keeps
the validation and test nodes in `data`. Used as released, the model therefore
trains on test labels. The epochs, the required constructor argument
`mmd_sample`, the train mask and the model-selection rule appear in no released
artifact.

**What the harness completes, fixed here and applied identically to both arms:**
* **Task:** binary node classification on the common labels. The criterion
  `nn.MSELoss` is replaced by `nn.BCEWithLogitsLoss`, the loss every other
  classification cell uses.
* **Mask:** every loss term (supervised, MMD, GWN = Sinkhorn + moment distance)
  is computed on the training nodes of the common split only. Validation and
  test nodes enter only through message passing and through the edge
  reweighting, which reads features and the sensitive attribute and never
  labels, as BeMap's sampler does.
* **mmd_sample = 500:** the sample size the class itself uses for its other
  sampled loss (`compute_sinkhorn_loss(sample_size=500)`).
* **Horizon and selection:** H = 200 with sigma_c^BCE / sigma_c^AUC on the
  stored validation trajectory. After each training step, one evaluation
  forward (`model.eval()`) over all nodes is recorded.
* **Everything else:** as released. The `FnRGNN` class is imported unmodified;
  the training step is `optimize()` with only the mask and the criterion
  changed.

**Configuration:** `logs/best_configs/<config>_best_joint.json` (the authors'
joint accuracy-fairness pick). Sensitive attributes match the study's:
* german: `german_g` (gender);
* pokec_z: `region_job_r` (region);
* pokec_n: `region_job_2_r` (region).

Applied fields: hidden_dim, dropout, lambda2, gamma, lambda_dist, lr and
weight_decay. The configurations were tuned for the regression targets
(LoanAmount; completion_percentage); this is part of the caveat.

**Preprocessing:** FnRGNN's own feature standardisation (`StandardScaler` over
all nodes' features, as `load_and_prepare_dataset` does). Its removal of
degree-0 nodes is not applied, because the common graph and split are kept.

**Intervention and off-state:** the class's own ablation switches.
* M^+I: `use_mmd = use_gwn = use_edge_weight = True`.
* M^-I: all three False. This leaves the same two-layer GCN with unit edge
  weights and the supervised loss only.

The released `model/ablation/` checkpoints switch these components off one at
a time; I is their union.

**Datasets:** german, pokec_z, pokec_n. NBA is excluded study-wide.

**Native:** none, because no training loop or selection rule is released.

**Environment:** `/home/sypark/x30_edits_env` (geomloss 0.3.1, deeprobust,
torch_sparse); tensorized Sinkhorn backend. GPU 2.

**Gates (outcome-blind), per arm and unit:**
* finite logits and trajectory length;
* both arms' criterion is BCE on exactly the training nodes;
* M^+I: MMD and GWN terms non-zero and edge weights not all 1;
* M^-I: MMD and GWN terms identically 0 and edge weights all 1;
* the paired arms start from identical parameters.

A failure stops the cell and persists nothing.

**Adapter and output:** `harness/adapters/x31_fnrgnn.py` through `x30_run.py`,
protocol key `x31`. Output: `harness/results/x31/x31_FnRGNN_<dataset>.csv`.

## 3b. FnRGNN on its own task: node regression

User decision (2026-09-22): FnRGNN is also run on the task it was released for,
node regression with MSE, with B = the common GCN trained with MSE. The
section's metrics are regression metrics, so it is a separate analysis. It is
never compared with or pooled into the classification sections; it is compared
only with itself.

**Targets** are taken from the study's own data files, which are byte-identical
(md5) to FnRGNN's copies. They match the released `best_joint` configurations
(section 3):
* german: `LoanAmount`;
* pokec_z and pokec_n: `completion_percentage`.

The target is read from the CSV in node order (row i = node i in the common
loaders).

**Features:** the common features with the target column removed. This applies
to B and to both arms, because the common loaders keep the column as an input
feature. The removed column is asserted equal to the raw target (gate). The
arms then apply FnRGNN's own StandardScaler over all nodes' features, as in
section 3; B keeps its own feature policy.

**Target scaling:** the target is standardised with the training nodes' mean
and standard deviation. FnRGNN's loader fits the scaler on all nodes, which
would read test targets. All metrics are in these standardised units.

**Split and sensitive attribute:** the common split (train, validation and test
nodes) and the common sensitive attribute. The seeds and 6 splits x 5 runs are
the same as in every cell.

**Arms**
* **M^+I / M^-I:** the released class and the section 3 loop with the released
  criterion `nn.MSELoss` kept (no task change). The loop keeps the training-node
  mask on every loss term, mmd_sample = 500 and H = 200. The configuration is
  the same `best_joint` file, and the off-state is the three switches off.
* **B:** `algorithms/GNN.py` at `published("GNN", ds)`, constructed and seeded
  exactly as the common baseline. It is trained by the steps of `GNN.fit`
  (optimizer_2, encoder + classifier, 201 steps as for every B) with
  `F.binary_cross_entropy_with_logits` replaced by `F.mse_loss` on the
  training nodes.

**Selection:** for every arm, the epoch with the smallest validation MSE
(strict <, earliest tie). This is the regression counterpart of sigma_c^BCE.
One evaluation forward over all nodes is recorded per step.

**Metrics on the test nodes** follow FnRGNN's `utils/metric.py`. Each
coordinate is oriented so that larger is better:
* `negMSE` = -MSE;
* `negMeanGap` = -|mean prediction(s=0) - mean prediction(s=1)|
  (`output_fairness.mean_gap`, the regression analogue of DP);
* `negWD` = -Wasserstein distance between the two groups' predictions
  (`output_fairness.wasserstein`).

**Estimands:** tau_nonint, tau_I and tau_pkg as everywhere, per matched unit.
They use the paired hierarchical bootstrap (`bootstrap_armA.boot`, 10,000
replicates, splits then runs, seed `bootstrap_armA.SEED`) and the unchanged
resolution rule: sign stability >= 0.75, |mean| >= 0.010 and a CI excluding 0.
The 0.010 threshold here applies in standardised target units.

**Gates (outcome-blind), per unit:**
* the removed feature column equals the raw target;
* finite predictions and trajectory lengths;
* the loss is computed on exactly the training nodes;
* the M^+I / M^-I component checks of section 3;
* the paired arms start from identical parameters.

**Runner:** `harness/experiments/x31_fnrgnn_regression_run.py`. **Output:**
`harness/results/x31/x31_FnRGNN-regression_<dataset>.csv`.
**Analysis:** `harness/experiments/analyze_x31_fnrgnn_regression.py`, written
and committed before it is run.

