# T2 — Consistency of the "common" baseline B across methods

Scope: `protocol=controlled`, `selector=common_bce`, `state=B`, from the frozen
`results/per_unit_metrics.csv.gz` (16,200 rows; 1,950 B rows in this slice).
Factual audit. No result file was modified; every artifact of this task is under
`results/phase0_audit/`.

Labels: **[확인됨]** verified here, **[추정]** inferred, **[확인 불가]** not determinable.

Environment constraint honoured: no command in this task used a GPU. The loader /
index-hash job was run with `CUDA_VISIBLE_DEVICES=""` (pure CPU data loading).
Exact commands are listed in §6.

---

## 0. Executive finding

**[확인됨] B is NOT trained once per (dataset, split, run) and shared across
methods.** Every runner script trains its own B inside its own process, and the
production runs were launched one method (or one method×dataset, or even one
method×dataset×split) per process. On Credit and Income the resulting B draws
diverge materially; on Bail/German/Pokec they do not, because there the
retraining happens to be numerically reproducible.

The divergence mechanism is already documented in the repository itself
(`harness/X3_BASELINE_NONDETERMINISM.md`) — but that document treats it as a
reason to read B *in-process*, and does not note that the per-method process
split re-introduces exactly the failure it describes across methods.

---

## 1. Verified spread of B across methods (per dataset)

Computed per `(dataset, split_id, run_id)` cell as `max(metric) − min(metric)`
over the methods present in that cell, then the maximum over the 30 cells
(6 splits × 5 runs).

| dataset | # methods | max cell-spread AUC | max cell-spread DP | max cell-spread EO | worst cell (metric / pair) |
|---|---|---|---|---|---|
| **credit**  | 14 | **0.047111** | **0.102214** | **0.134849** | AUC: s23 r3 EDITS↔FairGB; DP: s21 r0 FairGB↔SFG; EO: s21 r0 FairGB↔FairSIN-GIN |
| **income**  | 2 (BIND-1pct / BIND-10pct) | **0.053515** | **0.253850** | **0.213740** | AUC: s20 r1; DP: s24 r4; EO: s25 r0 |
| bail        | 17 | 0.000002 | 0.000000 | 0.000000 | — |
| german      | 14 | 0.000305 | 0.000000 | 0.000000 | — |
| pokec_z     | 9  | 0.000001 | 0.000000 | 0.000000 | — |
| pokec_n     | 7  | 0.000002 | 0.000000 | 0.000000 | — |
| pokec_z_g / pokec_n_g | 1 each | n/a (single method, no cross-method comparison possible) | | | |

**[확인됨] The numbers in the task brief are confirmed**: Credit AUC up to 0.047
(exactly 0.047111) and DP up to 0.102 (0.102214); Income DP up to 0.25
(0.253850, BIND-1pct vs BIND-10pct); Bail/German/Pokec identical to ≤3.05e-4
AUC and exactly 0.0 DP/EO. Additionally the **EO** spread on credit (0.1348) and
income (0.2137) is at least as large as DP and is not mentioned in the brief.

Per-method deviation from an arbitrary in-dataset reference method (max over the
30 cells) — credit:

| method | max\|ΔAUC\| | max\|ΔDP\| | max\|ΔEO\| |
|---|---|---|---|
| BeMap (reference) | 0 | 0 | 0 |
| BeMap-GAT | 0.023949 | 0.067209 | 0.087858 |
| EDITS | 0.032215 | 0.062106 | 0.080335 |
| FairEdit | 0.026042 | 0.056485 | 0.081335 |
| FairGB / FairGNN / NIFTY | 0.028350 | 0.078152 | 0.088137 |
| FairSIN-GCN | 0.017343 | 0.083252 | 0.112485 |
| FairSIN-GIN | 0.027653 | 0.075773 | 0.108291 |
| FairSIN-SAGE | 0.030653 | 0.085866 | 0.125895 |
| FairVGNN | 0.020997 | 0.082020 | 0.094769 |
| FairVGNN-GIN | 0.029463 | 0.078152 | 0.088137 |
| FairVGNN-SAGE | 0.040023 | 0.079678 | 0.107098 |
| SFG | 0.031701 | 0.080385 | 0.101764 |

**[확인됨] Identity structure on credit**: in 24 of 30 cells there are exactly
**12 distinct B values among 14 methods**, and the only always-identical group is
`{FairGB, FairGNN, NIFTY}` — the three methods that `pilot_tau.py` runs together
in one process (`FairGB`, `NIFTY` vs `FairGNN`: bit-identical in **30/30** cells).
Every other method has its own B — `FairVGNN-GIN` and `FairVGNN-SAGE`, whose
per-method max-deviation happens to be close to the FairGNN group's, are
identical to `FairGNN` in **0/30** cells (max |ΔAUC| 0.0271 and 0.0379).
The occasional extra collisions (e.g. `BeMap`≡`FairVGNN` at s20 r2)
are coincidental agreement of two independent draws that happened to select the
same epoch, not sharing.

**[확인됨] B is `bc_*`**: `state=B` rows of `per_unit_metrics.csv.gz` reproduce
the `bc_auc/bc_dp/bc_eo` columns of the corresponding frozen harness CSV to
≤5e-7 (the export is rounded to ~6 decimals). Checked on
`harness/results/x30/x30_EDITS_credit.csv`,
`harness/results/x30/x30_FairSIN-GIN_credit.csv`,
`harness/results/x30/x30_BIND-{1,10}pct_income_s20.csv`.

---

## 2. (a) Is B trained once and shared, or re-trained per method?

### 2.1 Within one process: shared

`harness/experiments/pilot_tau.py`

- `main()` loops `for split ... for run ...` at **pilot_tau.py:416-417**
  (`seed = a.seed0 + run`).
- B is constructed and fitted **once per (split, run), before the method loop**:
  - **pilot_tau.py:458-459** — `from ...GNN import GNN`;
    `torch.manual_seed(seed); np.random.seed(seed)`;
    `b = GNN(adj, f, y, itr, iva, ite, s, si, device=dev, **b_cfg)`
  - **pilot_tau.py:460-463** — `hb = ValidationHistory(...)`, `b.fit(epochs=a.epochs, trajectory=hb)`
- B is read at both common selectors in the same process at
  **pilot_tau.py:480-486** (`base_c[_sel] = (_ep, outcome(...))`).
- The method loop starts at **pilot_tau.py:488** (`for meth in a.methods:`) and
  writes `bc_epoch/bc_auc/bc_dp/bc_eo` from that one `base_c` at
  **pilot_tau.py:561-563**.

So *within a single invocation* every method in `--methods` shares one B.
**[확인됨]**

### 2.2 Across processes: re-trained, and the production runs are per-method processes

`harness/experiments/x30_run.py`

- **x30_run.py:89-107** — `def train_B(data, seed, epochs, dev, dataset)`, docstring
  *"B exactly as pilot_tau.main builds it (same constructor, same seed call)"*;
  **x30_run.py:93-94** `cfg = dict(published("GNN", dataset)["config"]); cfg.pop("feature_normalize", None)` /
  `torch.manual_seed(seed); np.random.seed(seed)`.
- **x30_run.py:226** — `b, hb, b_score = train_B(data, seed, a.epochs, dev, a.dataset)`
  is called **inside the per-(split, run) loop of a single-method run**
  (the CLI takes one `--method` and one `--dataset`,
  **x30_run.py:150-179**).
- Note the ordering: the two method arms are trained **first**
  (**x30_run.py:201-210**), and B is trained **after** them, at
  **x30_run.py:226**. In `pilot_tau.py` B is trained **before** the method arms.
  The CUDA context, allocator state and cuBLAS/cuDNN workspace at the moment B
  starts therefore differ between runners and between methods. **[확인됨]** that
  the order differs; **[추정]** that this contributes to the divergence (it is not
  separable from ordinary run-to-run GPU nondeterminism without a controlled
  experiment, which this audit does not run).

The frozen outputs show one file per method×dataset (and per split for income):
`harness/results/x30/x30_EDITS_credit.csv`, `x30_FairSIN-GIN_credit.csv`,
`x30_BIND-1pct_income_s20.csv` … i.e. **each method's B came from its own
process**. **[확인됨]**

### 2.3 FairVGNN on credit: a third B

`harness/results/armA_credit.csv` + `armA_credit_s23_25.csv` (the `pilot_tau`
output) contain FairGB, FairGNN, FairVGNN, NIFTY with one shared B. But the
FairVGNN rows that reached `per_unit_metrics.csv.gz` come from the **separate
X11 rerun** `harness/results/armA_fairvgnn_credit.csv`:

| per_unit `state=B`, credit | vs `armA_credit*.csv` `bc_auc` | vs `armA_fairvgnn_credit.csv` `bc_auc` |
|---|---|---|
| FairGNN / NIFTY / FairGB | **4.4e-07 – 4.7e-07** (match) | 0.0414 (differ) |
| FairVGNN | 0.0144 / 0.0414 (differ) | **4.8e-07** (match) |

Cause: `harness/X11_RNG_CONTRACT.md` — *"Every FairVGNN row produced before this
contract … is removed from the primary analysis"* — so FairVGNN was re-run alone
(`harness/experiments/x31_fairvgnn_config_run.py:143-148`, which re-trains B with
the same `torch.manual_seed(seed); np.random.seed(seed)` + `GNN(...)` +
`b.fit(...)` recipe). The discarded FairVGNN rows kept the shared B; the retained
ones carry a fresh, different B. **[확인됨]**

### 2.4 Other runners that re-train B

| file:line | how B is seeded |
|---|---|
| `harness/experiments/pilot_tau.py:457` | `torch.manual_seed(seed); np.random.seed(seed)` |
| `harness/experiments/x30_run.py:94` | `torch.manual_seed(seed); np.random.seed(seed)` |
| `harness/experiments/x31_fairvgnn_config_run.py:144` | `torch.manual_seed(seed); np.random.seed(seed)` |
| `harness/experiments/x31_fairgnn_config_run.py:156-160` | same recipe, B data at `published("GNN", ds)` |
| `harness/experiments/x31_fnrgnn_regression_run.py:75` | `torch.manual_seed(seed); np.random.seed(seed)` — **but the loss is `F.mse_loss` (line 82), a deliberately different regression B** |
| `harness/experiments/x31_fmp_baseline_run.py:50` | **`X.seed_all(seed * 1000 + split)`** — a *different seed for B* than every other runner |

**[확인됨]** `x31_fmp_baseline_run.py:50` seeds B with `seed*1000+split` where all
other runners use `seed`. Its rows feed the FMP mechanistic case study
(`results/5_component_case_study_FMP.csv`), which is reported separately, so this
does not enter the controlled `state=B` slice audited here — **[추정]**, based on
the method list of the slice (no `FMP` method appears in
`per_unit_metrics.csv.gz`).

### 2.5 Adapters do not train B

**[확인됨]** None of `harness/adapters/x30_*.py`, `x31_*.py`,
`fairvgnn_credit_native.py` trains or touches B; they only expose
`train_arm(dataset, encoder, arm, split, seed, epochs, device)` returning
per-epoch full-graph scores plus masks. `harness/adapters/fairvgnn_credit_native.py`
is a **native-protocol** artefact only (`X19`/`X20`; module docstring lines 1-28)
and therefore cannot affect any `protocol=controlled` B row.

---

## 3. (b) Do the train/val/test index sets match across methods?

### 3.1 Code argument

**[확인됨]** All index sets originate from one function,
`balanced_split(labels, label_number, seed=split_seed)` at `utils/data.py:83-108`
(and `pokec_split` for the pokec settings). It draws from a **local, explicitly
seeded** generator — `rng = random.Random(seed)` at `utils/data.py:87` — so it is
independent of the global RNG state, of the device, and of everything a method
does.

**[확인됨]** `feature_normalize` cannot change the indices: in every loader the
split is computed *before* `_finalize_as_data(...)`, which is the only place
normalisation is applied (`utils/data.py:289` split → `utils/data.py:292-297`
finalize; `_finalize_as_data` at `utils/data.py:137-174`, normalisation at
`utils/data.py:149-151`).

**[확인됨]** Every runner and every adapter reaches the split through the same two
entry points:

| consumer | file:line | call |
|---|---|---|
| pilot_tau (all core methods + B) | `harness/experiments/pilot_tau.py:60-71` | `utils.dataloading.load_data(ds, feature_normalize=…, split_seed=split_seed)` |
| x30_run (B) | `harness/experiments/x30_run.py:193-195` | `load(...)` (the same `pilot_tau.load`) |
| EDITS | `harness/adapters/x30_edits.py:75-79` | `load_data(dataset, feature_normalize=False, split_seed=split)` |
| FairEdit | `harness/adapters/x30_fairedit.py:50-54` | `load_data(dataset, feature_normalize=cfg["feature_normalize"], split_seed=split)` |
| BeMap | `harness/adapters/x30_bemap.py:115-121` | `load_data(dataset, feature_normalize=False, split_seed=split)` |
| GEAR | `harness/adapters/x30_gear.py:185-188` | `load_data(dataset, feature_normalize=False, split_seed=split)` |
| BIND | `harness/adapters/x30_bind.py:168-180` | `load_data(dataset, feature_normalize=False, split_seed=split)`, injected into BIND's `load_bail`/`load_income` |
| FnRGNN | `harness/adapters/x31_fnrgnn.py:111-117` | `load_data(dataset, feature_normalize=False, split_seed=split)` |
| FairSIN | `harness/adapters/x30_fairsin.py:185-186` | `get_dataset(dataset, feature_normalize=(dataset != "german"), split_seed=split)` |
| SFG | `harness/adapters/x31_sfg.py:164-165` | `get_dataset(dataset, feature_normalize=norm, split_seed=split)` **and** `get_dataset(dataset, feature_normalize=False, split_seed=split)` |

`load_data` derives the indices from the masks of the very same `get_dataset`
object (`utils/dataloading.py:52-62`), so the two entry points are the same split
by construction.

**[확인됨] A runtime hard stop already enforces this.** `x30_run.py:206-210`:

```
for nm, mk, ref in (("train", trm, itr_np), ("val", vam, iv), ("test", tem, it_)):
    if not np.array_equal(np.where(np.asarray(mk))[0], np.sort(ref)):
        raise SystemExit(f"[x30] {tag} {arm} {nm} split differs from the "
                         f"common split on {a.dataset} s{split} (hard stop)")
```

Both `M_plus_I` and `M_minus_I` arms are checked against B's own index arrays in
every cell, for every x30/x31 method. `pilot_tau.py:441-447` and
`x31_fairvgnn_config_run.py:134-138` carry the analogous guard for the
normalisation cache (`raise SystemExit("... B, M0 and M1 would not share a split")`).

### 3.2 Empirical check

Re-ran the repo's loaders on CPU and hashed the *sorted* index arrays
(SHA-256, first 16 hex chars) for each of the 4 loader paths that the adapters
actually use × 8 datasets × 6 splits. Output:
**`results/phase0_audit/T2_index_hashes.csv`**
(columns: `dataset, split_id, loader_path, status, train_hash, n_train,
val_hash, n_val, test_hash, n_test`).

**Underlying loader sweep:** 8 datasets × 6 splits × 4 loader paths = **192 loads,
all `status=ok`, no missing assets.** For **all 48 (dataset, split) cells the four
loader paths produce identical `train_hash`, `val_hash` and `test_hash`** —
i.e. `load_data` ≡ `get_dataset`, and `feature_normalize=True` ≡ `False`.

**Method × split table** (`T2_index_hashes.csv`, 390 rows = 65 method×dataset
pairs × 6 splits; columns `method, dataset, split_id, loader_path,
loader_evidence, status, train_hash, n_train, val_hash, n_val, test_hash,
n_test`). Each method is mapped to the loader entry point its adapter/runner
actually calls (file:line in `loader_evidence`), and the hashes are the sweep's.

**[확인됨] Result: in 0 of 48 (dataset, split) cells do any two methods disagree on
any of the three index hashes.** All methods on a given (dataset, split) share
one train/val/test partition.

Worked example — credit, split 20 (all 14 methods):

| method | loader path | train_hash | val_hash | test_hash |
|---|---|---|---|---|
| BeMap | load_data(fn=False) | 5806c99c61f017fb | ccaa201444ffda35 | 32313bc4e61a8eaf |
| BeMap-GAT | load_data(fn=False) | 5806c99c61f017fb | ccaa201444ffda35 | 32313bc4e61a8eaf |
| EDITS | load_data(fn=False) | 5806c99c61f017fb | ccaa201444ffda35 | 32313bc4e61a8eaf |
| FairEdit | load_data(fn=False) | 5806c99c61f017fb | ccaa201444ffda35 | 32313bc4e61a8eaf |
| FairGB | load_data(fn=True) | 5806c99c61f017fb | ccaa201444ffda35 | 32313bc4e61a8eaf |
| FairGNN | load_data(fn=True) | 5806c99c61f017fb | ccaa201444ffda35 | 32313bc4e61a8eaf |
| FairSIN-GCN/GIN/SAGE | get_dataset(fn=True) | 5806c99c61f017fb | ccaa201444ffda35 | 32313bc4e61a8eaf |
| FairVGNN, -GIN, -SAGE | load_data(fn=True) | 5806c99c61f017fb | ccaa201444ffda35 | 32313bc4e61a8eaf |
| NIFTY | load_data(fn=False) | 5806c99c61f017fb | ccaa201444ffda35 | 32313bc4e61a8eaf |
| SFG | get_dataset(fn=False) | 5806c99c61f017fb | ccaa201444ffda35 | 32313bc4e61a8eaf |

Split sizes (constant across splits and methods):

| dataset | n_train | n_val | n_test |
|---|---|---|---|
| bail | 100 | 4719 | 4719 |
| credit | 6000 | 7500 | 7500 |
| german | 100 | 250 | 250 |
| income | 4605 | 3705 | 3706 |
| pokec_z / pokec_z_g | 500 | 2565 | 2566 |
| pokec_n / pokec_n_g | 500 | 2199 | 2200 |

**[확인 불가] / scope note.** The hashes are computed by re-running the two
repository loader entry points, not by re-executing each third-party
`train_arm` body end to end. A method that silently re-partitioned *inside*
its own code would not show here — but `x30_run.py:206-210` asserts exactly
that at run time for every x30/x31 cell, and BIND, the one adapter that does
reshape the graph, is fed the common indices verbatim
(`x30_bind.py:177-180`, `x30_bind.py:189-192`). `income` and `pokec_*_g` carry a
single method each, so no cross-method index comparison is possible there;
`bail` on BIND-1pct vs BIND-10pct confirms the two BIND budgets share one split
(identical hashes, and identical B to 1e-6).

### 3.3 Consequence for the matched design of M_minus_I / M_plus_I

**[확인됨] No index mismatch was found, so the matched-design claim for
`M_minus_I` vs `M_plus_I` is NOT threatened by differing splits.** Both arms of a
cell are produced by one `train_arm` call pair inside one process against one
loaded split, and are additionally asserted equal to B's split at
`x30_run.py:206-210`.

**The B inconsistency is therefore a *baseline-referenced* problem only.**
Contrasts internal to a method — `tau_int = M_plus_I − M_minus_I` — are unaffected.
Contrasts that reference B — `tau_base = M_minus_I − B`, `tau_pkg = M_plus_I − B`,
and any cross-method comparison of those — are affected on Credit and Income.
Concretely: a cross-method difference in `tau_pkg` on credit of less than
~0.10 DP / ~0.047 AUC cannot be distinguished from the baseline draw.

---

## 4. (c) Per-method data-loading differences

### 4.1 Feature normalisation — differs by method, but **not for B**

`harness/core/published_config.py:74-78`:

```
_NORMALIZE = {"FairGNN": lambda d: d in ("nba", "german"),   # :516
              "FairVGNN": lambda d: d == "german",           # :467
              "NIFTY": lambda d: False,                      # :592
              "GNN": lambda d: False,                        # :570
              "FairGB": lambda d: True}                      # :639
```
applied at `harness/core/published_config.py:276`.

Adapter-side policies (all method arms, never B):

| method | file:line | policy |
|---|---|---|
| EDITS | `x30_edits.py:78-79` | always `feature_normalize=False` |
| FairEdit | `x30_fairedit.py:46, 53-54` | `feature_normalize=(dataset == "german")` |
| FairSIN | `x30_fairsin.py:185-186` | `feature_normalize=(dataset != "german")` |
| BeMap / GEAR / FnRGNN / BIND | `x30_bemap.py:120-121`, `x30_gear.py:187-188`, `x31_fnrgnn.py:116-117`, `x30_bind.py:169-171` | always `False` |
| SFG | `x31_sfg.py:164-165` | `norm` for the model, plus a second **un-normalised** load for its own use |

**[확인됨] B's own load is `published("GNN", dataset)` → `_NORMALIZE["GNN"] = False`
for every dataset**, in every runner:
`pilot_tau.py:449-450` (`b_cfg.pop("feature_normalize", False)`),
`x30_run.py:193-195` (`b_cfg.get("feature_normalize", False)`),
`x31_fairvgnn_config_run.py:127-129`, `x31_fnrgnn_regression_run.py:74`,
`x31_fmp_baseline_run.py:51`.

→ **B's input tensors are byte-identical across all methods.** The Credit/Income
B divergence is **not** attributable to preprocessing. **[확인됨]**

### 4.2 Graph construction

- Canonical adjacency: `utils/data.py:73-81` `make_adj_from_edges` —
  COO from the edge list, symmetrised (`adj + adj.T.multiply(adj.T>adj) − adj.multiply(adj.T>adj)`),
  then **`adj = adj + sp.eye(n)`** (self-loops added, `utils/data.py:80`).
- `utils/dataloading.py:69-77` re-symmetrises and **binarises** (`scipy_adj.data = np.ones_like(...)`)
  and returns a torch sparse COO tensor. Self-loops survive.
- `GNN.__init__` takes `edge_index = adj.coalesce().indices()` at
  `models/algorithms/GNN.py:192` — so B runs `GCNConv` on the self-looped,
  binarised, symmetric edge index. **[확인됨]**
- Adapter reconstructions of a scipy matrix from that same tensor:
  `x30_edits.py:87-89`, `x30_bemap.py:123-125`, `x30_gear.py:190-191`
  ("self-loops included, as GEAR's adj"), `x31_fnrgnn.py:119`.
- **Two adapters deliberately differ**:
  - `x30_bind.py:172-179` — `A.setdiag(0); A.eliminate_zeros()` **removes the
    diagonal**, then `_loader_wrappers` re-adds `sp.eye(n)` at
    `x30_bind.py:189` to match BIND's `utils.py:64/:158`.
  - `x30_fairsin.py:181-192` — rebuilds `d.adj` from `edge_index` because the
    upstream loader returns a scipy adjacency this repo's loader does not; the
    module docstring at `x30_fairsin.py:14-19` records that the upstream
    cache-miss branch subtracts the identity from `data.adj` while the hit branch
    does not, and that `<dataset>_hadj.pt` is namespaced per split for that reason.
  Both are **method-arm only**; B's graph is unchanged. **[확인됨]**

### 4.3 Node ordering

**[확인됨] Node order is the CSV row order throughout and is never permuted.**
`utils/data.py:284-287` (`idx = np.arange(features.shape[0]); idx_map = {j: i ...}`)
is the identity map; features/labels/sens come from the same DataFrame in row
order. Indices are integers into that order. `x30_gear.py:200` sorts the index
tensors (`torch.sort(itr)` …) — sorting an index set, not reordering nodes, and
the hashes in §3.2 are computed on sorted arrays so this is a no-op for the
comparison.

### 4.4 Global numerical flags touched by adapters

**[확인됨]** `harness/adapters/x30_gear.py:229` sets
`torch.backends.cudnn.allow_tf32 = False` (to mirror GEAR's `main.py:78`) and
restores it in a `finally` at `x30_gear.py:260` (saved at `x30_gear.py:223`).
`harness/adapters/x30_fairsin.py:293-294` saves
`(cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32)` and restores at
`x30_fairsin.py:362-363`. Both restore correctly, **but in `x30_run.py` the
adapter runs *before* `train_B` (x30_run.py:201-210 vs :226)**, so any adapter
that failed to restore would silently change B's arithmetic. No such adapter was
found. **[확인됨]** for the two adapters inspected; **[확인 불가]** as an exhaustive
claim — not every third-party `train_arm` body was traced for global-state writes.

---

## 5. (d) Seed propagation and nondeterminism

### 5.1 Seed path

- `seed0` default **27** in both production runners:
  `harness/experiments/pilot_tau.py:388` and `harness/experiments/x30_run.py:161`
  (`ap.add_argument("--seed0", type=int, default=27)`), also
  `x31_fairvgnn_config_run.py:97`, `x31_fairgnn_config_run.py:127`,
  `x31_fnrgnn_regression_run.py:152`, `x31_fmp_baseline_run.py:97`.
- `seed = a.seed0 + run` — `pilot_tau.py:417`, `x30_run.py:188`.
- **B** receives it as `torch.manual_seed(seed); np.random.seed(seed)`
  immediately before construction — `pilot_tau.py:458`, `x30_run.py:94`.
  Note: **B is seeded with `seed`, the method arms with `seed*1000+split`**
  (`x30_run.py:202`, `pilot_tau.py:503, 506`). Python's `random` and the CUDA
  generators are *not* separately reseeded for B (`torch.manual_seed` does cover
  CUDA devices; `random` is left wherever the previous arm left it). **[확인됨]**
- Method arms: `seed_all(seed * 1000 + split)` — `x30_run.py:202`; definition at
  `harness/core/trajectory.py:141-147`:
  ```
  def seed_all(seed: int) -> None:
      import random as _random
      _random.seed(seed)
      np.random.seed(seed % (2 ** 32))
      torch.manual_seed(seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed)
  ```
- Evaluation seed: `eval_rng_seed(dataset, split, run) = crc32("ds|split|run|eval")`
  at `harness/core/trajectory.py:130-138`, used for FairVGNN only
  (`pilot_tau.py:516-517`).
- The split seed is a **separate** axis: `split_seed=split` ∈ {20,…,25}, consumed
  by `random.Random(seed)` at `utils/data.py:87`.

### 5.2 The nondeterminism is documented — and is the cause

`harness/X3_BASELINE_NONDETERMINISM.md`, verbatim:

> `baseline_sigma_c.py` was written to recover the missing term cheaply: retrain
> B from the same seed, split and device, and read it at sigma_c^BCE and
> sigma_c^AUC. … **On two cells (split 20, runs 0-1) the reproduction matched to
> 7.6e-05 AUC and 0.0 DP. Over the full 6 x 5 design it did not.**
>
> | residual | mean | median | max |
> |---|---|---|---|
> | \|dAUC\| | 0.0128 | 0.0006 | 0.0754 |
> | \|dDP\|  | 0.0306 | 0.0053 | 0.3362 |
>
> **16 of 30 (split, run) cells** diverged beyond 1e-3. The worst, split 24 run 2,
> reproduced B at DP 0.000 where the pilot measured 0.336.
>
> The cause is ordinary GPU nondeterminism -- non-associative reductions and
> TF32 -- compounding over 200 epochs. It is amplified by checkpoint selection:
> two trajectories that differ in the sixth decimal can put the validation
> minimum at a different epoch (e.g. 94 vs 132), and the two checkpoints are then
> genuinely different models, not two readings of one model.

and:

> **Recorded, not corrected.** No attempt is made to make training
> bit-deterministic. Determinism is not needed once every term of a comparison is
> read from one process; forcing it would change every number already collected
> for a property the design does not rely on.

`harness/X11_RNG_CONTRACT.md` confirms residual CUDA nondeterminism even for a
deterministic arm:

> The first version of check 7 required exact float equality and failed at about
> 1e-7, for M0 as well as M1. **M0 draws no random number at inference, so that
> residual is CUDA kernel nondeterminism, not RNG.**

and documents the run-seed / evaluation-seed separation:

> xi_eval = crc32("dataset|split|run|eval") … Immediately after a checkpoint is
> restored for test scoring, every generator is reseeded to xi_eval. … xi_eval
> depends only on the cell's identity, never on an outcome.

**[확인됨] The repository sets no global determinism flags.** The only
`torch.backends.cudnn.deterministic = True` in the tree is
`utils/train_baselines.py:125`, which the harness runners do not execute.
No `torch.use_deterministic_algorithms` anywhere.

### 5.3 Direct confirmation of the "amplified by checkpoint selection" mechanism

Comparing the two BIND income streams (same B code, same seed, same data, two
processes), from `harness/results/x30/x30_BIND-{1,10}pct_income_s2*.csv`:

| pattern | cells | \|ΔB AUC\| | \|ΔB DP\| |
|---|---|---|---|
| `bc_epoch` identical in both streams | 18 / 30 | ≤ 4.3e-4 | ≤ 1.3e-2 |
| `bc_epoch` differs (e.g. 199 vs 150, 191 vs 98, 10 vs 105) | 12 / 30 | up to 0.0535 | up to 0.2539 |

Examples: s20 r1 `bc_epoch` 199 vs 150 → AUC 0.6935 vs 0.6400; s24 r4
`bc_epoch` 173 vs 171 → DP 0.0090 vs 0.2628; s21 r0 `bc_epoch` 118 vs 149 →
DP 0.2329 vs 0.0026. **[확인됨]** This is exactly the X3 mechanism, reproduced
here across two *methods* rather than across two reproduction passes.

### 5.4 Why Bail/German/Pokec are unaffected

**[추정]** The B retraining is bit-stable enough on those graphs that the
validation-BCE minimum lands on the same epoch in every process, so the selected
checkpoint is identical (DP/EO agree exactly; AUC differs only at 1e-6–3e-4,
consistent with a tiny final-forward-pass residual rather than a different
checkpoint). Credit is the largest graph in the design (30,000 nodes /
2,873,716 edges, `harness/core/datasets.py:44`), where scatter-add atomics
dominate; income's B trajectory is evidently flat near the minimum. This audit
did not run the controlled experiment that would confirm the causal attribution
([확인 불가] as a strict causal claim).

---

## 6. Commands run (all CPU; no GPU used)

```bash
# 1. spread / identity structure / bc_* correspondence / bc_epoch comparison
#    (pandas only, no torch, no GPU) -- several heredoc scripts of the form:
cd /home/sypark/workspace/FairGNN-Eval && ~/miniconda3/envs/dev/bin/python << 'EOF'
import pandas as pd, numpy as np
d = pd.read_csv('results/per_unit_metrics.csv.gz')
b = d[(d.protocol=='controlled')&(d.selector=='common_bce')&(d.state=='B')]
# per-cell max-min spread per dataset/metric; per-method deviation vs a reference
# method; identity grouping of rounded (auc,dp,eo) tuples; merges against
# harness/results/{armA_credit*.csv, armA_fairvgnn_credit.csv,
# x30/x30_*_credit.csv, x30/x30_BIND-*_income_s*.csv}
EOF

# 2. loader index hashes -- explicitly CPU-only, no GPU visible
cd /home/sypark/workspace/FairGNN-Eval && CUDA_VISIBLE_DEVICES="" \
  ~/miniconda3/envs/dev/bin/python <scratchpad>/hash_idx.py

# 3. method x split projection of those hashes (pandas only) -> T2_index_hashes.csv
```

`hash_idx.py` is kept at
`/tmp/claude-1007/-home-sypark-workspace/4a7ed2e4-.../scratchpad/hash_idx.py`
(session scratch, not in the repo).

The hash script calls only `utils.dataloading.load_data` and `utils.data.get_dataset`
for `feature_normalize ∈ {False, True}` over
`dataset ∈ {bail, credit, german, income, pokec_z, pokec_n, pokec_z_g, pokec_n_g}`
and `split_id ∈ {20..25}`, and SHA-256s the sorted `idx_train/idx_val/idx_test`.
No GPU work of any kind was performed by this task. An earlier launch of the same
script (without `CUDA_VISIBLE_DEVICES=""`) was terminated before completion and
its output discarded.

---

## 7. Cause / blast radius / proposed fix (no fix applied)

**Cause.** `train_B` is a per-process function (`x30_run.py:89-107`;
`pilot_tau.py:458-463`), and the experiment was executed one method per process
(`harness/results/x30/x30_<METHOD>_<DATASET>.csv`). B is therefore re-trained once
per method, and on Credit/Income the retrained draws differ because of the GPU
nondeterminism + checkpoint-selection amplification the repository already
documents in `harness/X3_BASELINE_NONDETERMINISM.md`. `harness/X11_RNG_CONTRACT.md`
adds one more B draw for FairVGNN.

**Blast radius.**
- **Unaffected:** every `tau_int = M_plus_I − M_minus_I` contrast (paired within one
  process, one split, one B-independent comparison); the matched-design claim for
  `M_minus_I`/`M_plus_I` (indices verified identical, §3);
  Bail, German, pokec_z, pokec_n, pokec_*_g.
- **Affected:** every B-referenced quantity on **credit** (14 methods) and
  **income** (2 methods): `tau_base`, `tau_pkg`, `tau_pkg^audit`, and in particular
  any *cross-method* ranking or count derived from them. On credit the baseline
  draw alone spans 0.047 AUC / 0.102 DP / 0.135 EO; on income 0.054 / 0.254 / 0.214.
  Any cross-method claim on those two datasets with an effect smaller than that is
  not separable from the baseline draw.
- `results/1_main_package_vs_intervention.csv`, `results/1b_*`, `results/2_*` and the
  bootstrap intervals built on `pkg_*` columns inherit this on credit/income
  **[추정]** — the exact downstream propagation was not traced in this task.

**Proposed fix (not applied).** In order of cost:
1. **Report-level:** state, next to every credit/income B-referenced number, the
   measured baseline-draw spread (0.047/0.102/0.135 and 0.054/0.254/0.214) as a
   floor on cross-method resolution, and restrict cross-method claims on those two
   datasets to effects above it.
2. **Cheap re-analysis:** designate one canonical B per `(dataset, split, run)` —
   e.g. the `pilot_tau` B present in `armA_credit*.csv` — and recompute all
   B-referenced quantities against it. This requires no retraining, only a join;
   it does, however, break the X3 invariant that B and M come from one process,
   so it must be reported as a second, explicitly non-paired view rather than a
   replacement.
3. **Correct but expensive:** persist B's selected checkpoint (or its per-epoch
   test scores) once per `(dataset, split, run)` to disk under the split seed, and
   have every runner load it instead of calling `train_B`. This restores a single
   shared B by construction and keeps the in-process reading property, at the cost
   of re-running the credit and income cells.
