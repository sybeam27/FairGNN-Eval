# T12 — Nondeterminism / reproducibility fact check

Evidence-gathering only. No interpretation beyond what the cited artefacts state,
no fixes applied. Every claim is labelled [확인됨] (verified) / [추정] (inferred) /
[확인 불가] (cannot be determined).

Repository: `/home/sypark/workspace/FairGNN-Eval`
Python: `~/miniconda3/envs/dev/bin/python`
Written: 2026-09-24

---

## 1. What "16 of 30" in `harness/X3_BASELINE_NONDETERMINISM.md` actually means

### 1.1 The claim, quoted

`harness/X3_BASELINE_NONDETERMINISM.md:21-30`:

```
21  On two cells (split 20, runs 0-1) the reproduction matched to 7.6e-05 AUC and
22  0.0 DP. Over the full 6 x 5 design it did not.
23
24  | residual | mean | median | max |
25  |---|---|---|---|
26  | \|dAUC\| | 0.0128 | 0.0006 | 0.0754 |
27  | \|dDP\|  | 0.0306 | 0.0053 | 0.3362 |
28
29  **16 of 30 (split, run) cells** diverged beyond 1e-3. The worst, split 24 run 2,
30  reproduced B at DP 0.000 where the pilot measured 0.336.
```

### 1.2 Unit of the "30" — [확인됨]

**30 = (split × run) units inside ONE cell, not 30 cells.**

The design is a single dataset, a single backbone, a single arm, a single
method-free configuration: 6 splits × 5 runs = 30 units.

Evidence:

- `harness/experiments/baseline_sigma_c.py:45-49` — the generating script's
  defaults: `--dataset german` (single dataset), `--splits 20 21 22 23 24 25`
  (6 splits), `--runs 5`, `--seed0 27`.
- `harness/experiments/baseline_sigma_c.py:61-62` —
  `for split in a.splits: / for run in range(a.runs):` — 6 × 5 = 30 iterations.
- `harness/results/baseline_sigma_c.csv` contains 60 data rows = 30 units × 2
  selectors (`common_bce`, `common_auc`). Verified:
  `dataset` ∈ {german}; `split_id` ∈ {20,21,22,23,24,25}; `run_id` ∈ {0..4};
  `selector` ∈ {common_bce, common_auc}.
- The doc's own wording at line 29 is `(split, run) cells` — i.e. it uses the
  word "cell" for a (split, run) unit, not for a (dataset × method × backbone)
  benchmark cell. Line 21 uses the same convention ("On two cells (split 20,
  runs 0-1)").

So **the wording "16 of 30 cells" in the paper must be read as 16 of 30
split × run units within a single (german, GCN, common baseline B)
configuration.** [확인됨]

### 1.3 Dataset and arm — [확인됨]

- Dataset: **german only**. `baseline_sigma_c.py:45` default `--dataset german`;
  all 60 rows of `harness/results/baseline_sigma_c.csv` have `dataset=german`.
- Backbone: **GCN** (`baseline_sigma_c.py:96` writes `backbone="GCN"`).
- Arm: **the common baseline B itself** — i.e. `models/algorithms/GNN.GNN`,
  the method-free GCN baseline, not any fairness method and not Arm A / Arm B.
  `baseline_sigma_c.py:58` `from models.algorithms.GNN import GNN`;
  `:70-71` constructs it. No fairness method is trained in this script at all.
- Seeds: `seed = 27 + run` (`baseline_sigma_c.py:63`), i.e. 27..31 per split.
- Horizon: 200 epochs (`baseline_sigma_c.py:50`, `--epochs 200`); device `cuda`
  (`:51`).

### 1.4 What exactly was compared with what — [확인됨]

**A freshly retrained B (in a second process) against the *stored* B from the
pilot's CSV.** Not two runs of the same script, and not a retrained-vs-retrained
pair.

- `harness/experiments/baseline_sigma_c.py:15-16` (docstring):
  "`bpub_*` is written alongside as the reproduction check: it must equal the
  `b_*` columns already in the pilot CSV."
- The retrained side: `baseline_sigma_c.py:69-76` seeds with
  `torch.manual_seed(seed); np.random.seed(seed)`, constructs `GNN(...)`, fits
  200 epochs; `:87` `pub = outcome(..., b_score(b.best_state, ite_np))` reads it
  at B's **code-native selector** `b.best_state`; `:99` writes
  `bpub_auc/bpub_dp/bpub_eo`.
- The stored side: `harness/experiments/pilot_tau.py:458-459` seeds identically
  (`torch.manual_seed(seed); np.random.seed(seed)`) and constructs the same
  `GNN`; `pilot_tau.py:560` writes `b_auc=base_pub["auc"], b_dp=base_pub["dp"],
  b_eo=base_pub["eo"]` — also at the code-native selector.
- So the residual is `|bpub_x − b_x|` per (split, run):
  **same nominal configuration, same seed, same split, same device, two
  different processes.** [확인됨]

Caveat on configuration matching — [확인됨 + 추정]:

- `baseline_sigma_c.py:70-71` **hardcodes** `num_hidden=128, num_proj_hidden=128,
  lr=1e-3, weight_decay=0.0`, whereas `pilot_tau.py:453-459` takes the config
  from `published("GNN", a.dataset)`.
- Today, `published("GNN","german")` resolves to `num_hidden=16,
  num_proj_hidden=16, lr=0.001` (run against
  `harness/core/published_config.py:261-272`; the 16 comes from the
  `harness/provenance/nifty_README.md` override at `:270-272`). [확인됨]
- However 2 of 30 units reproduce **bit-exactly to 16 significant digits**
  (see §1.5), which is impossible under different hidden widths. And
  `harness/provenance/nifty_README.md` has mtime `2026-09-14 11:51:47`, later
  than all three CSVs involved (`pilot_tau_3x5.csv` 00:32, `baseline_sigma_c.csv`
  00:45, `pilot_tau_s23_25.csv` 01:18 on 2026-09-14). [확인됨]
- [추정] At the time these runs were made, `published("GNN","german")` still
  returned the code default `num_hidden=128` (`published_config.py:263-265`),
  so the two processes were genuinely config-matched. Re-running
  `baseline_sigma_c.py` **today** would no longer be config-matched.

### 1.5 Recomputation — [확인됨], reproduces exactly

The underlying CSVs still exist:

- retrained side: `harness/results/baseline_sigma_c.csv` (60 rows)
- stored side: `harness/results/pilot_tau_3x5.csv` (splits 20-22) +
  `harness/results/pilot_tau_s23_25.csv` (splits 23-25) — 120 rows each,
  4 methods × 5 runs × 2 selectors × 3 splits. These are the **pre-fix** pilot
  CSVs: neither has a `bc_auc` column, matching the doc's account that the
  `bc_*` columns were added only by the fix (`X3_...md:50-53`).

Command:

```
~/miniconda3/envs/dev/bin/python  # pandas
bs = read_csv('harness/results/baseline_sigma_c.csv')
pt = concat([read_csv('harness/results/pilot_tau_3x5.csv'),
             read_csv('harness/results/pilot_tau_s23_25.csv')])
B  = bs.drop_duplicates(['split_id','run_id'])[['split_id','run_id','bpub_auc','bpub_dp','bpub_eo']]
P  = pt.drop_duplicates(['split_id','run_id'])[['split_id','run_id','b_auc','b_dp','b_eo']]
m  = B.merge(P, on=['split_id','run_id'])   # 30 rows
```

Output:

```
n= 30
|dauc| mean=0.0128 median=0.0006 max=0.0754
|ddp|  mean=0.0306 median=0.0053 max=0.3362
diverged>1e-3 (auc|dp): 16
  of which |ddp|>1e-3: 16 ;  |dauc|>1e-3: 13
split 24 run 2:  bpub_dp=0.000000  b_dp=0.336188  (|ddp|=0.336188)
                 bpub_auc=0.542324 b_auc=0.483962
split 20 run 0:  dauc=0.0  ddp=0.0
split 20 run 1:  dauc=0.0  ddp=0.0
```

**All four table entries, the "16 of 30", and the split-24-run-2 anecdote
reproduce exactly.** [확인됨]
(Minor discrepancy: the doc says split 20 runs 0-1 "matched to 7.6e-05 AUC";
in the stored CSVs those two units match to exactly 0.0 on AUC, DP and EO at the
code-native selector. The 7.6e-05 is presumably the residual at the σ_c
selectors, not at the published selector. [추정])

Divergence criterion: the doc says "diverged beyond 1e-3". Using
`|ΔDP| > 1e-3 OR |ΔAUC| > 1e-3` gives exactly 16; using `|ΔDP| > 1e-3` alone
also gives 16 (the 13 AUC-divergent units are a subset). [확인됨]

Note: comparing against the **post-fix** rerun `harness/results/pilot_tau_6x5_audit.csv`
instead gives a different answer (mean |ΔDP| 0.0310, max 0.2144, still 16/30) —
that file is a separate rerun of the pilot, so it is a third draw. The doc's
numbers come from the pre-fix pair. [확인됨]

### 1.6 Stated cause, quoted

`X3_BASELINE_NONDETERMINISM.md:32-36`:

```
32  The cause is ordinary GPU nondeterminism -- non-associative reductions and
33  TF32 -- compounding over 200 epochs. It is amplified by checkpoint selection:
34  two trajectories that differ in the sixth decimal can put the validation
35  minimum at a different epoch (e.g. 94 vs 132), and the two checkpoints are then
36  genuinely different models, not two readings of one model.
```

`X3_BASELINE_NONDETERMINISM.md:58-63` states that no attempt was made to make
training bit-deterministic.

---
## 2. Does every evaluated method go through a sparse spmm / scatter path?

Each line names the model class **actually instantiated by the harness adapter /
runner** (not merely present in the vendored repo), with the ctor line and the
line in `forward()` where the primitive is applied. All paths absolute.
All entries below are [확인됨] unless marked otherwise.

### 2.1 One line per method

| method | propagation primitive in the trained forward pass | ctor | applied in forward |
|---|---|---|---|
| **B (common GCN baseline)** | PyG **GCNConv** (1 layer + linear head); instantiated `harness/experiments/x30_run.py:92,94`, encoder default `"gcn"` (`models/algorithms/GNN.py:185`); also `harness/experiments/pilot_tau.py:458-459` | `/home/sypark/workspace/FairGNN-Eval/models/algorithms/GNN.py:36` | `/home/sypark/workspace/FairGNN-Eval/models/algorithms/GNN.py:39` |
| **BIND** | PyG **GCNConv** (`GCN_Body` + fc); driven from `/home/sypark/workspace/FairGNN-Eval/harness/adapters/x30_bind.py:231,344`, model created `models/BIND-main/implementations/1_training.py:67` | `/home/sypark/workspace/FairGNN-Eval/models/BIND-main/implementations/GNNs/gcn.py:32` | `/home/sypark/workspace/FairGNN-Eval/models/BIND-main/implementations/GNNs/gcn.py:35` |
| **BeMap** | **DGL `GraphConv`** (2 layers, on a per-epoch resampled subgraph); instantiated `/home/sypark/workspace/FairGNN-Eval/harness/adapters/x30_bemap.py:148`, `cfg["model"]` default `gcn` (`x30_bemap.py:104`) | `/home/sypark/workspace/FairGNN-Eval/models/BeMap-main/models.py:248-249` | `/home/sypark/workspace/FairGNN-Eval/models/BeMap-main/models.py:260,263` |
| **EDITS** | trained classifier: PyG **GCNConv**; instantiated `/home/sypark/workspace/FairGNN-Eval/harness/adapters/x30_edits.py:121`, classifier built in `EDITS.predict` at `models/algorithms/EDITS.py:434` | `/home/sypark/workspace/FairGNN-Eval/models/algorithms/EDITS.py:56` | `/home/sypark/workspace/FairGNN-Eval/models/algorithms/EDITS.py:59` |
| **FairEdit** | PyG **GCNConv** (2 `GCN_Body` blocks); `/home/sypark/workspace/FairGNN-Eval/harness/adapters/x30_fairedit.py:83` with `model_name="gcn"` (`x30_fairedit.py:45`), built `models/algorithms/FairEdit.py:1105` | `/home/sypark/workspace/FairGNN-Eval/models/algorithms/FairEdit.py:936` | `/home/sypark/workspace/FairGNN-Eval/models/algorithms/FairEdit.py:938` |
| **FairGB** | PyG **SAGEConv** (2 layers, `aggr='mean'`); encoder default `'SAGE'` (`models/algorithms/FairGB_alg.py:47`), built `models/algorithms/FairGB/utils.py:41` | `/home/sypark/workspace/FairGNN-Eval/models/algorithms/FairGB/models.py:107,114` | `/home/sypark/workspace/FairGNN-Eval/models/algorithms/FairGB/models.py:122,124` |
| **FairGNN** | PyG **GCNConv** (backbone + sensitive-attribute estimator); instantiated `/home/sypark/workspace/FairGNN-Eval/utils/train_baselines.py:522`, submodules `models/algorithms/FairGNN.py:65-66` via `get_model` `:29` | `/home/sypark/workspace/FairGNN-Eval/models/algorithms/FairGNN.py:14` | `/home/sypark/workspace/FairGNN-Eval/models/algorithms/FairGNN.py:17` |
| **FairSIN (GCN, default)** | hand-written **`torch_scatter.scatter(..., reduce='add')`** (`--prop scatter` default, `models/FairSIN-main/in-train.py:263`; Arm-A adapter hardcodes it at `/home/sypark/workspace/FairGNN-Eval/harness/adapters/fairsin.py:224,245`) | `/home/sypark/workspace/FairGNN-Eval/models/FairSIN-main/model.py:46-48` | `/home/sypark/workspace/FairGNN-Eval/models/FairSIN-main/model.py:56` -> `/home/sypark/workspace/FairGNN-Eval/models/FairSIN-main/utils.py:43` |
| **FairSIN (GIN / SAGE cells)** | PyG **GINConv** / **SAGEConv**; both cells are run (`/home/sypark/workspace/FairGNN-Eval/harness/experiments/x30_streams_v2.sh:33`) | `/home/sypark/workspace/FairGNN-Eval/models/FairSIN-main/in-train.py:63` / `:67` | inside the PyG conv |
| **FairVGNN** | hand-written **`torch.spmm`** by default; **`torch_scatter.scatter`** on german (`/home/sypark/workspace/FairGNN-Eval/utils/train_baselines.py:485` passes `prop="scatter"`); selection at `models/algorithms/FairVGNN.py:1068-1073`, default `'spmm'` at `:1321` | spmm `/home/sypark/workspace/FairGNN-Eval/models/algorithms/FairVGNN.py:843-845`; scatter `:813-815` | spmm `/home/sypark/workspace/FairGNN-Eval/models/algorithms/FairVGNN.py:862`; scatter `:832` -> `:486` |
| **FnRGNN** | PyG **GCNConv** with per-edge `edge_weight` (2 layers); loaded via `/home/sypark/workspace/FairGNN-Eval/harness/adapters/x31_fnrgnn.py:52-56`, harness-completed loop `:79+` | `/home/sypark/workspace/FairGNN-Eval/models/FnRGNN-master/utils/model.py:802-803` | `/home/sypark/workspace/FairGNN-Eval/models/FnRGNN-master/utils/model.py:894,897` |
| **GEAR** | PyG **SAGEConv** (2 layers, `aggr='mean'`); instantiated `/home/sypark/workspace/FairGNN-Eval/harness/adapters/x30_gear.py:232-233`, `base_model` default `'sage'` (`models/GEAR-main/src/main.py:57`), never overridden | `/home/sypark/workspace/FairGNN-Eval/models/GEAR-main/src/models.py:83,90` | `/home/sypark/workspace/FairGNN-Eval/models/GEAR-main/src/models.py:103-105` |
| **NIFTY** | PyG **GCNConv** (1 layer); encoder default `"gcn"` (`models/algorithms/NIFTY.py:214`), instantiated `/home/sypark/workspace/FairGNN-Eval/utils/train_baselines.py:607` | `/home/sypark/workspace/FairGNN-Eval/models/algorithms/NIFTY.py:54` | `/home/sypark/workspace/FairGNN-Eval/models/algorithms/NIFTY.py:57` |
| **SFG** | PyG **SAGEConv** stack; `encoder='SAGE'` in every `run.sh` row (`models/SFG-main/run.sh:11-23`), read by `/home/sypark/workspace/FairGNN-Eval/harness/adapters/x31_sfg.py:112-118,193`; built `models/SFG-main/sfg.py:104` | `/home/sypark/workspace/FairGNN-Eval/models/SFG-main/model.py:177` | `/home/sypark/workspace/FairGNN-Eval/models/SFG-main/model.py:214` |
| **FMP** | **two-stage**: (a) pure-MLP stage, `nn.Linear` only, **no propagation**; (b) propagation stage = **DGL `GraphConv`**, entered only when `lambda2 > 0`; plus a **dense** `sen @ ...` correction inside the same trained loop. Built `/home/sypark/workspace/FairGNN-Eval/harness/experiments/x27_fmp_run.py:99-103` | MLP `/home/sypark/workspace/FairGNN-Eval/models/FMP-main/fairgnn.py:14,16,17`; GraphConv `/home/sypark/workspace/FairGNN-Eval/models/FMP-main/fmp.py:88` | MLP `/home/sypark/workspace/FairGNN-Eval/models/FMP-main/fairgnn.py:30,33,35`; GraphConv `/home/sypark/workspace/FairGNN-Eval/models/FMP-main/fmp.py:137`; dense `sen @` `/home/sypark/workspace/FairGNN-Eval/models/FMP-main/fmp.py:146,152,161,168` (`sen` densified at `:33`) |

Also, for completeness (not an evaluated fairness method): the **harness-internal
backbone** used by the E-series diagnostics (including E0, §3) is a hand-written
`torch.sparse.mm` GCN — ctor `/home/sypark/workspace/FairGNN-Eval/harness/core/model.py:32,42-43`,
applied `:47` and `:50`, driven from `harness/core/trainer.py:129,254`. It is
**not** used by the x30/x31 method cells.

### 2.2 The FMP qualification — [확인됨]

`models/FMP-main/fmp.py:135-139`: the DGL `GraphConv` call is reached only when
`lambda2 > 0`. With `lambda2 == 0` the step is
`y = gamma * hh + (1-gamma) * x` (`fmp.py:139`), and with `gamma = 1/(1+lambda2) = 1`
this reduces to `y = hh` — i.e. **the graph is never touched: the model is the
bare MLP of `fairgnn.py:8-39`.**

Corroborated by the harness's own documentation:
- `harness/X17_FMP_AUDIT.md:58-59`, `:81-82`, `:94`:
  "λ2 = 0 removes propagation. γ = 1 gives y = hh, so F00 is exactly the MLP."
- `harness/X27_FMP_MECHANISTIC_PREREG.md:97`:
  "**F00** | 0 | 0 | γ = 1 → `y = hh`, and `x = y` each step: pure MLP".

**This is a released cell, not a hypothetical.** `results/5_component_case_study_FMP.csv`
contains 18 rows with `step=base, from_stage=B, to_stage=base,
lambda1_fairness=0.0, lambda2_propagation=0.00` (verified by groupby over the
released CSV). So the published artefact set contains a measured configuration
whose trained forward pass has **no graph propagation of any kind**.

Scope note — [확인됨]: FMP does **not** appear in the main method table.
`results/1_main_package_vs_intervention.csv` covers 11 methods
(BIND, BeMap, EDITS, FairEdit, FairGB, FairGNN, FairSIN, FairVGNN, GEAR, NIFTY, SFG)
and `results/cell_results.csv` / `results/coverage.csv` cover 12 (the same plus
FnRGNN). FMP appears only in `results/5_component_case_study_FMP.csv`
(section 5, component case study), per `harness/METHOD_INVENTORY.csv` (FMP row:
adapter `harness/experiments/x27_fmp_run.py`, isolated virtualenv).

### 2.3 Answer to the wording question

**A paper may NOT say, unqualified, "every method uses a nondeterministic sparse
propagation path."** Three separate qualifications are required.

**(a) FMP's F00 cell has no propagation at all.** [확인됨]
Its trained forward pass is a pure `nn.Linear` MLP (`fmp.py:139` + `fairgnn.py:8-39`),
and it is a released cell in `results/5_component_case_study_FMP.csv`
(`lambda2_propagation = 0.00`, 18 rows). Nothing sparse, nothing graph-based.

**(b) FMP also carries a dense matmul inside the trained forward pass.** [확인됨]
`fmp.py:146,152,161,168` apply `sen @ ...` / `sen.t() @ z` with `sen` built dense
at `fmp.py:33`, inside the K-step `emp_forward` loop. This is *not* preprocessing.
So even in F01/F11, FMP's propagation stage is DGL `GraphConv` **plus** dense ops.

**(c) The primitive is not one primitive.** [확인됨]
It is at least four distinct implementations, with different determinism
characteristics:
- PyG `MessagePassing` conv layers (internal scatter / segment reductions):
  B, BIND, EDITS(classifier), FairEdit, FairGB, FairGNN, FairSIN-GIN/SAGE,
  FnRGNN, GEAR, NIFTY, SFG
- **DGL** `GraphConv`: BeMap, FMP(propagation stage)
- **hand-written** `torch.spmm`: FairVGNN (default)
- **hand-written** `torch_scatter.scatter`: FairSIN-GCN (default), FairVGNN on german

**Dense paths that are preprocessing only** (so they do NOT need qualifying as
part of the trained forward pass) — [확인됨]:
- **EDITS**: `A_norm.mm(X_de)` at `models/algorithms/EDITS.py:187` (called from
  `EDITS.forward` `:208`), adjacency densified at `:622`, `feature_smoothing`
  dense at `:677-679`. This runs only inside `model.fit(...)` (the debias stage,
  `harness/adapters/x30_edits.py:124`); the evaluated classifier trained in
  `predict()` uses GCNConv. In the M-I arm `fit` is skipped entirely
  (`harness/adapters/x30_edits.py:127-128`), so the dense path does not run at all.
- **GEAR**: dense adjacency/feature ops only in counterfactual-asset
  preprocessing, `models/GEAR-main/src/Preprocessing.py:230,233,273,329,356,359`;
  the trained `GraphCF` forward is SAGEConv only.
- **BIND**: [확인됨 — negative result] no dense adjacency matmul found; node
  deletion is scipy-sparse (`models/BIND-main/implementations/3_removing_and_testing.py:393`,
  adapter `harness/adapters/x30_bind.py:327-333`); the influence stage is
  gradient/HVP-based. Both stages train the same GCNConv model.

### 2.4 Suggested precise wording

> Every method in the main table (BIND, BeMap, EDITS, FairEdit, FairGB, FairGNN,
> FairSIN, FairVGNN, FnRGNN, GEAR, NIFTY, SFG) and the common GCN baseline train
> a graph neural network whose forward pass performs a sparse neighbourhood
> reduction — PyG `MessagePassing` convolutions for most, DGL `GraphConv` for
> BeMap, and hand-written `torch.spmm` / `torch_scatter.scatter` for FairVGNN and
> FairSIN. None of these reductions has a deterministic CUDA kernel. The one
> exception is the FMP component case study (Section 5), whose backbone is an MLP
> and whose propagation stage is gated by λ2: its λ2 = 0 configuration performs no
> graph propagation at all, and its λ1 > 0 configurations additionally use a dense
> matrix product inside the trained loop.

Caveat that also belongs in any such statement — [확인됨 from §3.4]: a sparse
CUDA path is a *necessary* but not sufficient condition for observed
nondeterminism. On the E0 diagnostic, 2 of 9 settings (income, NBA) were
**bit-exact** across 10 repeats at a fixed seed while using `torch.sparse.mm` on
CUDA.

---

## 3. `harness/experiments/e0_noise_floor.py` — what it measured and what it found

### 3.1 Recovery status — [확인됨], recovered

The working-tree outputs were deleted earlier in this session. They were
recovered **read-only** from the bare repository
`/home/sypark/workspace/FairGNN-Eval-oldhistory.git`.

Search for the path and revision:

```
$ G=/home/sypark/workspace/FairGNN-Eval-oldhistory.git
$ git --git-dir=$G log --oneline --all -- 'study/results/e0_*'
ea10163 E0: 노이즈 바닥 측정 — 노이즈가 주장하려는 효과와 같은 크기
3e97daf study/README: 발견 3 정정 — 두 번째 Credit 그래프는 존재하나 읽히지 않음
```

Only one revision ever touched the file:
**`ea10163f28d067e782806e22dff48cf24f09a31b`**, authored
`Soyoung Park <wlsl89822@gmail.com>`, `Wed Sep 9 15:18:26 2026 +0900`.
Files it added/changed: `study/README.md`, `study/experiments/e0_noise_floor.py`,
`study/results/e0_noise_floor.csv` (+120 lines).

Recovered copies (written only under `results/phase0_audit/recovered/`, nothing
restored into the working tree):

- `results/phase0_audit/recovered/e0_noise_floor.csv` — 180 data rows + header
  (`git --git-dir=$G show ea10163:study/results/e0_noise_floor.csv`)
- `results/phase0_audit/recovered/study_README_at_ea10163.md`
  (`git --git-dir=$G show ea10163:study/README.md`)

The script itself is byte-identical to the current working-tree copy except for
the usage path (`study/experiments/...` → `harness/experiments/...`):

```
$ diff <old ea10163 copy> harness/experiments/e0_noise_floor.py
33c33
<     CUDA_VISIBLE_DEVICES=2 python study/experiments/e0_noise_floor.py \
---
>     CUDA_VISIBLE_DEVICES=2 python harness/experiments/e0_noise_floor.py \
```

So the recovered CSV is the output of the current script; the project directory
was renamed `study/` → `harness/`. [확인됨]

### 3.2 What the diagnostic measures — [확인됨]

Per `harness/experiments/e0_noise_floor.py:21-29` (docstring):

```
21  So before running E2 for real, measure two things per setting:
22
23      within-seed sd   repeat one seed R times          (float-noise only)
24      across-seed sd   run R different seeds once each  (seed + float noise)
```

Implementation `:79-81`:

```
79  # arm='within': one seed, repeated.   arm='across': one run each, R seeds.
80  plan = ([("within", args.fixed_seed, r) for r in range(args.reps)] +
81          [("across", args.fixed_seed + r, 0) for r in range(args.reps)])
```

Motivation `:4-19`: the same seed does not give the same answer on this box
(german / uniform / seed 27: CPU 1-thread dp=0.0164, CPU 8-thread dp=0.0792,
CUDA dp=0.0996..0.1442), because "cuSPARSE spmm reorders its reduction"
(`:12-13`), and `torch.use_deterministic_algorithms(True)` does not help because
"neither the COO nor the CSR spmm has a deterministic CUDA kernel" (`:14-15`).

Reported metrics: `METRICS = ["acc", "auc", "dp", "eo"]`
(`e0_noise_floor.py:53`), written as `test_acc/test_auc/test_dp/test_eo`
(`:102`).

Replicate-count formula `:141-142`:
`R = ceil((4 * sqrt(2) * sd / 0.011)^2)`, i.e. replicates needed for a 95 % CI
half-width of 0.0055 on a 0.011 effect.

**Scope caveat — [확인됨].** This diagnostic runs the *project's own* trainer
(`harness/core/trainer.py::train`, `harness/core/datasets.py`,
`e0_noise_floor.py:48-49`), not any of the 13 vendored fairness methods. Its
model uses a hand-written sparse propagation:
`harness/core/model.py:47` `h = torch.sparse.mm(adj, self.lin1(x)).relu()` and
`:49` `logit = torch.sparse.mm(adj, self.lin2(h)).squeeze(-1)`, with the
adjacency built at `:32` `torch.sparse_coo_tensor(...)`. So E0's noise floor is
evidence about `torch.sparse.mm` on CUDA, measured on this repo's nine dataset
settings — **not** a per-method measurement of BIND/BeMap/.../FMP.

### 3.3 Cells covered and repeats — [확인됨]

From the recovered CSV (180 rows):

- **9 settings** (`setting`, with `regime`):
  `german` (saturated), `recidivism` (saturated), `nba` (saturated),
  `credit` (degree-skewed), `income` (clustered), `pokec_n` (clustered),
  `pokec_z` (clustered), `pokec_n_g` (mixed), `pokec_z_g` (mixed).
  This is `D.ALL_SETTINGS` (`e0_noise_floor.py:75`).
- **1 signal**: `uniform` (`:59` default `--signals uniform`).
- **device**: `cuda` for all 180 rows.
- **arms**: `within` 90 rows, `across` 90 rows.
- **within arm**: seed **27 only**, **10 repeats** per setting
  (`rep` 0..9) → 9 × 10 = 90.
- **across arm**: seeds **27..36**, 1 run each → 9 × 10 = 90.
- Training: 300 epochs, warmup 100, `lambda_fair=0.2`, `q_gate=0.7`
  (`e0_noise_floor.py:62-65` defaults).
- Total: **180 runs**, ~13 minutes on GPU (commit message of `ea10163`).

### 3.4 Results — spread across repeats at the fixed seed — [확인됨]

Recomputed from `results/phase0_audit/recovered/e0_noise_floor.csv`.
**within arm — seed 27 fixed, 10 repeats** (sd = sample sd, ddof=1;
range = max − min):

| setting | regime | n | AUC mean | AUC sd | AUC range | ΔDP mean | ΔDP sd | ΔDP range | ΔEO mean | ΔEO sd | ΔEO range |
|---|---|---|---|---|---|---|---|---|---|---|---|
| credit     | degree-skewed | 10 | 0.6868 | 0.0067 | 0.0158 | 0.0312 | 0.0046 | 0.0110 | 0.0170 | 0.0082 | 0.0195 |
| german     | saturated     | 10 | 0.6218 | 0.0235 | 0.0809 | 0.0996 | 0.0351 | 0.1185 | 0.1083 | 0.0518 | 0.1597 |
| income     | clustered     | 10 | 0.7509 | 0.0000 | 0.0000 | 0.0281 | 0.0000 | 0.0000 | 0.0031 | 0.0000 | 0.0000 |
| nba        | saturated     | 10 | 0.7784 | 0.0000 | 0.0000 | 0.0546 | 0.0000 | 0.0000 | 0.0701 | 0.0000 | 0.0000 |
| pokec_n    | clustered     | 10 | 0.7380 | 0.0007 | 0.0017 | 0.0678 | 0.0011 | 0.0032 | 0.1047 | 0.0018 | 0.0043 |
| pokec_n_g  | mixed         | 10 | 0.7172 | 0.0170 | 0.0586 | 0.0165 | 0.0087 | 0.0252 | 0.0529 | 0.0096 | 0.0310 |
| pokec_z    | clustered     | 10 | 0.7566 | 0.0065 | 0.0242 | 0.0235 | 0.0097 | 0.0344 | 0.0380 | 0.0118 | 0.0409 |
| pokec_z_g  | mixed         | 10 | 0.7546 | 0.0020 | 0.0064 | 0.0560 | 0.0098 | 0.0302 | 0.0230 | 0.0108 | 0.0298 |
| recidivism | saturated     | 10 | 0.8598 | 0.0087 | 0.0208 | 0.0735 | 0.0003 | 0.0007 | 0.0498 | 0.0084 | 0.0199 |

**across arm — seeds 27..36, 1 run each** (for contrast):

| setting | AUC sd | ΔDP mean | ΔDP sd | ΔEO sd |
|---|---|---|---|---|
| credit     | 0.0325 | 0.0447 | 0.0245 | 0.0188 |
| german     | 0.0242 | 0.0358 | 0.0386 | 0.0464 |
| income     | 0.0152 | 0.0331 | 0.0093 | 0.0169 |
| nba        | 0.0046 | 0.0628 | 0.0420 | 0.0274 |
| pokec_n    | 0.0019 | 0.0719 | 0.0067 | 0.0084 |
| pokec_n_g  | 0.0170 | 0.0296 | 0.0199 | 0.0326 |
| pokec_z    | 0.0066 | 0.0338 | 0.0121 | 0.0098 |
| pokec_z_g  | 0.0072 | 0.0528 | 0.0059 | 0.0084 |
| recidivism | 0.0047 | 0.0676 | 0.0035 | 0.0058 |

The `ΔDP` sd columns reproduce the table in commit `ea10163`'s message exactly
(german 0.0351 / 0.0386, pokec_z_g 0.0098 / 0.0059, pokec_z 0.0097 / 0.0121,
pokec_n_g 0.0087 / 0.0199, credit 0.0046 / 0.0245, pokec_n 0.0011 / 0.0067,
recidivism 0.0003 / 0.0035, nba 0.0000 / 0.0420, income 0.0000 / 0.0093), with
the corresponding `R_paired / R_unpaired` = 326/395, 26/10, 25/39, 21/105,
6/159, 1/12, 1/4, 0/467, 0/23. [확인됨]

### 3.5 Findings as recorded at the time — quoted

From `results/phase0_audit/recovered/study_README_at_ea10163.md`
("Finding 4 — the same seed does not give the same answer, and the spread is the
size of the claim"):

1. "The noise is the size of the claim." Recomputing the 9-setting headline mean
   per seed gives 0.0480 with an across-seed sd of 0.0075; at five seeds a 95 %
   interval on either arm is about ±0.0075 — wider than half the 0.011
   difference claimed.
2. "Pairing is not optional." Same-seed pairing shrinks the required replicate
   count by 1-2 orders of magnitude (credit 159→6, nba 467→0, pokec_n_g 105→21).
   `R_paired` assumes no signal × seed interaction and is a **lower bound**.
3. "German cannot settle anything." `R_paired = 326`; repeating one seed moves
   ΔDP by 0.035, three times the effect. Seed 27 gives 0.0996 while the ten-seed
   mean is 0.0358 — "both an unlucky seed and an unstable one."
4. "Instability is not a property of graph size or regime." Income (n=14,821)
   and NBA (n=403) are bit-exact across repeats while pokec_z (n=67,796) is not;
   among the three saturated settings recidivism has the lowest sd of all nine
   and german the highest. Attributed to cuSPARSE kernel path selection
   depending on the sparsity pattern.

### 3.6 Key point for a reproducibility statement — [확인됨]

Two of nine settings — **income and NBA — are bit-exact across all 10 repeats at
a fixed seed** (sd = 0.0000 and range = 0.0000 on AUC, ΔDP and ΔEO), on CUDA,
using `torch.sparse.mm`. Nondeterminism therefore is **not** uniform across
settings even for one and the same sparse code path: a blanket statement that
"every run is nondeterministic" is contradicted by this diagnostic's own data.

---
