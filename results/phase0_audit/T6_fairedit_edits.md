# T6 — FairEdit intervention size

All commands below are CPU-only (csv/ast parsing, grep, sed). **No GPU was used and
no training was re-run**, so the GPU-2-only constraint is vacuously satisfied.

All counts are **directed edge-index columns** (`edge_index.shape[1]`), i.e. each
undirected edge counts twice. 20 columns removed = 10 undirected edges.

## What the code does

- `harness/adapters/x30_fairedit.py:39` `EDIT_NUM_PLUS = 10`;
  `:53` `edit_num = EDIT_NUM_PLUS if arm == "plus" else 0`.
- `harness/adapters/x30_fairedit.py:87,90` record `st["e0"]/st["e1"] = tr.edge_index.shape[1]`
  before and after `tr.train(epochs)`. `FairEdit.fit()` (models/algorithms/FairEdit.py:1083-1136)
  only builds the trainer — it never trains — so `e0` is the untouched graph.
- `harness/adapters/x30_fairedit.py:110` gate check asserts `st["edits"] == min(10, epochs)`
  and `:111` asserts `e1 != e0` for M+I; `:113-114` assert 0 edits / `e1 == e0` for M-I.
- `models/algorithms/FairEdit.py:780` `if epoch < self.edit_num: ... fair_graph_edit()`.
- `models/algorithms/FairEdit.py:711` **`add = False`** — the whole add branch
  (`:713-722`) is dead. `add_drop_edge_random` (`:626`) and `perturb_graph` (`:673`)
  still build candidate additions, and `GNNExplainer` still scores them (`:705`),
  but `self.edge_index` is never extended.
- `models/algorithms/FairEdit.py:724-736`: exactly one undirected edge is deleted per
  call — the column for `(u,v)` (`:730`) and the column for `(v,u)` (`:736`).

=> **per run: 10 calls x 2 columns = 20 columns removed, 0 added.**

## What the stored artifacts show

```
$ python -c "...ast.literal_eval(row['config']) over x30_FairEdit_<ds>.csv..."
bail    60 rows  (edit_num=10, edges_before=642616,  edges_after=642596)  x60
credit  60 rows  (edit_num=10, edges_before=2873716, edges_after=2873696) x60
german  60 rows  (edit_num=10, edges_before=44484,   edges_after=44464)   x60
splits 20..25, runs 0..4  (= 30 units x M+I/M-I bookkeeping; 60 rows each)
```
[확인됨] Identical across every split and run: one single (before, after) tuple per
dataset, count 60/60. The delta is exactly -20 in all three datasets.

Totals cross-check against `harness/core/datasets.py:46-49`
(credit 2873716, recidivism/bail 642616, german 44484).

## The "20 edges out of 44k" caveat

`results/method_configurations.csv:57,58,59` — the *identical* string
`"the intervention removes 20 of ~44k edges at the repository default"`
appears on the german **and** the bail **and** the credit row.

[확인됨] The number refers to **german only** (44484 directed columns).
It is **wrong for bail** (642,616) and **for credit** (2,873,716) — off by 14x and 65x.

Relative intervention size:
| dataset | removed / total | fraction |
|---|---|---|
| german | 20 / 44,484 | 4.5e-4 |
| bail | 20 / 642,616 | 3.1e-5 |
| credit | 20 / 2,873,716 | 7.0e-6 |

**Cause.** Two independent hard-coded strings, both dataset-blind:
- `harness/experiments/x30_config_table.py:192`
  `caveat="the intervention removes 20 of ~44k edges at the repository default"`
  — emitted inside the `for ds in (...)` loop, so all three rows get it
  (-> `results/method_configurations.csv:57,58,59`).
- `harness/experiments/build_results.py:248,250,252` — the same "~44k" is typed
  out separately for german, bail AND credit:
  `"the repository default removes 20 of ~44k edges, so a near-zero estimate
  reflects the size of the intervention"` -> the `caveat` column of
  `results/1_main_package_vs_intervention.csv:10,11,12`.

**Blast radius.** Presentation only — no computed number depends on it. Two result
CSVs (6 rows total: method_configurations.csv:57-59 and
1_main_package_vs_intervention.csv:10-12) plus any prose quoting them. It
understates FairEdit's inertness on bail/credit by 14x / 65x, and the
build_results.py wording actively uses the wrong fraction to *explain away* the
near-zero tau_I estimate.
**Proposed fix** (not applied): make the caveat a format string over the
per-dataset `edges_before` already stored in the x30 row's `config` field, e.g.
"the intervention removes 20 of 642,616 directed edge entries (10 undirected
edges) at the repository default", and add "no edges are ever added:
FairEdit.py:711 sets add=False".

## Secondary finding [확인됨]

The method is named FairEdit and its paper claims *edge addition and deletion*;
in this frozen code the addition branch is disabled (`add=False`,
models/algorithms/FairEdit.py:711). M+I is therefore a pure 10-edge deletion.
`results/method_configurations.csv:57-59` describes I as "gradient-guided edge
editing" without stating that adds are off. Proposed fix: state it in the caveat.
