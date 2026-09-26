# T5 — Coverage ledger

Audit date 2026-09-24. Repository `/home/sypark/workspace/FairGNN-Eval`.
Python `~/miniconda3/envs/dev/bin/python`. **CPU-only** — no GPU was used for
any step of this task. No frozen file was modified; artifacts are written under
`results/phase0_audit/`.

Commands run (all read-only over frozen CSVs):

```
wc -l harness/coverage_manifest.csv
~/miniconda3/envs/dev/bin/python -c "<pandas value_counts over harness/coverage_manifest.csv>"
~/miniconda3/envs/dev/bin/python -c "<pandas outer merge manifest x model_dataset_feasibility>"
grep -n ... models/FairGT-main/train_fairgt.py
sed -n '80,110p' models/FairGT-main/train_fairgt.py
cat harness/provenance/fairgnn_upstream/pokec_z_train_fairGCN.sh
cat harness/provenance/fairgnn_upstream/SOURCE.txt
find models -iname "*pokec*"
```

---

## (a) Does `harness/coverage_manifest.csv` exist, and how many rows? — [확인됨]

Yes. **126 data rows** (127 lines including the header):

```
$ wc -l harness/coverage_manifest.csv
127 harness/coverage_manifest.csv
```

Shape `(126, 11)`, columns: `method, dataset, primary_eligible, included,
configuration, backbone, units_done, configuration_source,
matched_construction_status, exclusion_reason, notes`.

126 = **14 methods × 9 datasets**, an exact cross product:

- methods (9 rows each): BIND, BeMap, EDITS, FMP, FairEdit, FairGB, FairGNN,
  FairGT, FairSIN, FairVGNN, FnRGNN, GEAR, NIFTY, SFG
- datasets (14 rows each): bail, credit, german, income, nba, pokec_n,
  pokec_n_g, pokec_z, pokec_z_g

The cross product is enforced in code:
`harness/experiments/build_manifests.py:99-101,126` —

```
 99:  methods, datasets = sorted(feas.method.unique()), sorted(feas.dataset.unique())
...
126:  assert len(d) == len(methods) * len(datasets)
```

So the paper's "126 candidate rows" claim matches the file exactly — **[확인됨]**
for the file. I could not locate the string "126" in any in-repo document
(`grep -rn "126" --include=*.md harness/ README.md PAPER_ARTIFACT_MAP.md`
returns no candidate-count sentence), so the paper's exact wording is
**[확인 불가]** from this repository alone; only the artifact is verifiable.

### Rows per exclusion category — [확인됨]

| `exclusion_reason` | rows | `included` | `primary_eligible` |
|---|---:|---|---|
| *(empty — included in the primary set)* | **36** | yes | yes |
| `included_reported_separately` | **5** | yes | no |
| `unsupported_released_configuration` | **75** | no | no |
| `matched_intervention_not_separable` | **5** | no | no |
| `invalid_evaluation_split` | **4** | no | no |
| `required_artifact_unavailable` | **1** | no | no |
| **total** | **126** | 41 yes / 85 no | 36 yes / 90 no |

Category definitions are fixed in
`harness/experiments/build_manifests.py:33-40` (`CATEGORIES`) and assigned by
note-text matching at `build_manifests.py:42-52` (`RULES`) and
`:79-91` (`category()`), which raises if a row matches zero or more than one
rule (`build_manifests.py:89-90`).

The 36 `primary_eligible == yes` rows are **exactly** the 36 rows of
`results/1_main_package_vs_intervention.csv` (verified as a set equality on
(method, dataset); symmetric difference empty). **[확인됨]**

The 5 `included_reported_separately` rows: FMP/pokec_n, FMP/pokec_z (component
case study, table 5), FnRGNN/german, FnRGNN/pokec_n, FnRGNN/pokec_z
(task-adapted stratum, tables 1b/1c).

---

## (b) Relation to `results/model_dataset_feasibility.csv` (54 rows) — [확인됨]

### The generating relation

`harness/experiments/build_manifests.py:93-127` builds the manifest **from**
the feasibility file:

```
 94:  feas = pd.read_csv(os.path.join(RESULTS, "model_dataset_feasibility.csv"))
...
 99:  methods, datasets = sorted(feas.method.unique()), sorted(feas.dataset.unique())
100:  known = {(r.method, r.dataset): r for r in feas.itertuples()}
...
105:          r = known.get((m, ds))
106:          if r is None:          # the released code configures no such pair at all
107:              cat, elig, src, note, status = ("unsupported_released_configuration", "no", "",
108:                                              "not configured by the released code", "not_established")
```

So the manifest is the **dense 14 × 9 completion** of the sparse feasibility
ledger: the 54 feasibility rows are carried through with their own categories,
and the 72 missing pairs are auto-filled with
`unsupported_released_configuration` / "not configured by the released code".

### Verified correspondence

Outer merge on (method, dataset):

| | count |
|---|---:|
| manifest rows **with** a feasibility row (`both`) | **54** |
| manifest rows **without** a feasibility row (`left_only`) | **72** |
| feasibility rows **without** a manifest row (`right_only`) | **0** |
| total | 126 |

**Every feasibility row maps to exactly one manifest row. There are no orphan
feasibility rows.** 54 + 72 = 126.

Full row-level correspondence table:
`results/phase0_audit/T5_manifest_feasibility_correspondence.csv`
(126 rows; columns `method, dataset, primary_eligible, included, configuration,
backbone, units_done, matched_construction_status, exclusion_reason,
in_feasibility, primary_controlled, robustness_controlled, native`).

### Category breakdown of the 54 matched rows

| `exclusion_reason` | matched rows | cells |
|---|---:|---|
| *(included)* | 36 | the primary set |
| `included_reported_separately` | 5 | FMP/pokec_n, FMP/pokec_z, FnRGNN/german, FnRGNN/pokec_n, FnRGNN/pokec_z |
| `invalid_evaluation_split` | 4 | BeMap/nba, FMP/nba, FairGNN/nba, FnRGNN/nba |
| `matched_intervention_not_separable` | 5 | FairGT/{bail, credit, german, income, nba} |
| `required_artifact_unavailable` | 1 | GEAR/german |
| `unsupported_released_configuration` | 3 | BIND/pokec_n, BIND/pokec_z, GEAR/credit |

Note the asymmetry worth flagging: `unsupported_released_configuration` has
**75** rows total, of which only **3** carry a substantive, hand-written reason
from the feasibility ledger (BIND's pokec1/pokec2 are differently-sized
subgraphs; GEAR's released credit assets are on a 3 %-edge-overlap graph). The
other **72** carry the generic auto-filled note "not configured by the released
code" and were never individually adjudicated — they are the *absence* of a
feasibility row, not a recorded decision. [확인됨] on the mechanism;
[추정] on the implication: a reader taking the 126-row manifest as 126 audited
candidates would be overcounting the evidential work by 72 rows. Proposed fix:
add a column (e.g. `decision_source = ledger | inferred_absent`) so the 72
auto-filled rows are visibly distinct from the 54 adjudicated ones.

### The 72 manifest rows with no feasibility row, by method

| method | datasets (no feasibility row) | n |
|---|---|---:|
| BIND | credit, german, nba, pokec_n_g, pokec_z_g | 5 |
| BeMap | german, income, pokec_n, pokec_n_g, pokec_z_g | 5 |
| EDITS | income, nba, pokec_n, pokec_n_g, pokec_z, pokec_z_g | 6 |
| FMP | bail, credit, german, income, pokec_n_g, pokec_z_g | 6 |
| FairEdit | income, nba, pokec_n, pokec_n_g, pokec_z, pokec_z_g | 6 |
| FairGB | income, nba, pokec_n, pokec_n_g, pokec_z, pokec_z_g | 6 |
| FairGNN | income | 1 |
| FairGT | pokec_n, pokec_n_g, pokec_z, pokec_z_g | 4 |
| FairSIN | income, nba, pokec_n_g, pokec_z_g | 4 |
| FairVGNN | income, nba, pokec_n, pokec_n_g, pokec_z, pokec_z_g | 6 |
| FnRGNN | bail, credit, income, pokec_n_g, pokec_z_g | 5 |
| GEAR | income, nba, pokec_n, pokec_n_g, pokec_z, pokec_z_g | 6 |
| NIFTY | income, nba, pokec_n, pokec_n_g, pokec_z, pokec_z_g | 6 |
| SFG | income, nba, pokec_n, pokec_n_g, pokec_z, pokec_z_g | 6 |
| **total** | | **72** |

Cross-check against the independent ledger
`harness/METHOD_EXTENSION_COMPATIBILITY_MATRIX.csv`: FairGT's four pokec rows
there are labelled `D. UNSUPPORTED / not supported by the method's repository`
(lines 50-53), consistent with their auto-filled manifest category. [확인됨]

---

## (c) FairGT excluded on 5 datasets for "no off-state in the authors' code" — [확인됨]

The 5 rows: `FairGT × {german, bail, credit, income, nba}`,
`exclusion_reason = matched_intervention_not_separable`,
`notes = "no off-state in the authors' code"`
(`harness/coverage_manifest.csv`; ledger source
`results/model_dataset_feasibility.csv` rows 28-32). Those 5 datasets are
exactly the 5 the upstream README configures
(`models/FairGT-main/README.MD`: nba, german, income, bail, credit).

Corroborating ledgers:

- `harness/METHOD_EXTENSION_COMPATIBILITY_MATRIX.csv:47-49,54-55` —
  `FairGT,{german,bail,credit,nba,income},C. STRUCTURALLY-INVALID,C. STRUCTURALLY-INVALID,no off-state in authors' code`
- `harness/METHOD_EXTENSION_INVENTORY.csv:7` —
  `FairGT,new,FairGT-main (official),-,same-sensitive complete graph + eigenvector PE,-,none in authors' code,official-repo README commands,-,C,"switch search in authors' code and wrapper; components_off eig/sgr is study-authored help text (b396f76), unimplemented",EXCLUDED (C),train_fairgt.py:84-103`
- `harness/X1_CONTROL_INVENTORY.md:32` —
  `| **FairGT** | eigenvector PE, same-sens graph, k-hop features | three pieces, all inline: eigsh :35, _get_same_sens_complete_graph :254, _re_features :262 | **C** | A with three flags |`

### The code evidence

`models/FairGT-main/train_fairgt.py:84-103` — the entire fairness intervention
is applied unconditionally inside one `if args.model=='fairgt':` block, with no
flag, no `else` arm inside it and no `--no_*` switch:

```python
 84   if args.model=='fairgt':
 85       lpe=None
 86       filepath = './PE_files/'+args.model+'/'+args.dataset+'_'+str(args.pe_dim)+'_eig.pt'
 87       try:
 88           #
 89           eignvalue, eignvector = torch.load(filepath)
 90           lpe=eignvector
 91       except FileNotFoundError:
 92           print('pe file no exist!')
 93           #
 94           eignvalue, eignvector = adjacency_positional_encoding(adj, args.pe_dim)
 95           torch.save([eignvalue, eignvector], filepath)
 96           lpe=eignvector
 97       # get_same_sens_complete_graph
 98       features = torch.cat((feature, lpe), dim=1)
 99       print('original graph')
100       adj = get_same_sens_complete_graph(adj, sens, args)
101       # adj = torch.from_numpy(adj.todense())
102       processed_features = re_features(adj, features, args.hops)
103       g.ndata['feat'] = processed_features
```

The two fairness components — the eigenvector positional encoding concatenated
into the features (line 98) and the **same-sensitive-attribute complete graph**
substituted for the adjacency (line 100) — are both inside the single guard.
The only way to skip them is `args.model != 'fairgt'`
(`models/FairGT-main/train_fairgt.py:29`:
`parser.add_argument('--model', type=str, default='fairgt', ...)`), which
selects a *different model entirely*, not FairGT with its intervention off —
so `M^{-I}` cannot be constructed from a released switch.

The only two boolean flags in the released parser are unrelated to the
intervention and are both dead in this path
(`models/FairGT-main/train_fairgt.py:42-43`):

```
42   parser.add_argument('--sens_idex', type=bool, default=False, help='FFN layer size')
43   parser.add_argument('--is_lap',    type=bool, default=False, help='FFN layer size')
```

`--sens_idex` is consumed inside `utils.load_dataset` (drops the sensitive
column from the node attributes) and `--is_lap` is never read in
`train_fairgt.py` at all — neither disables the same-sensitive complete graph.

The study's own wrapper mirrors the upstream structure:
`models/algorithms/FairGT_alg.py:261-262`

```
261   print(f"[FairGT] Running hop aggregation (hops={args.fairgt_hops})...")
262   processed_features = _re_features(adj_norm, features, args.fairgt_hops)
```

with no off-switch either; `METHOD_EXTENSION_INVENTORY.csv:7` records that a
`components_off eig/sgr` help string exists in the wrapper at commit `b396f76`
but is **"study-authored help text … unimplemented"** — i.e. an advertised flag
with no implementation behind it. **[확인됨]** The exclusion is correctly
founded: no `M^{-I}` exists that differs from `M^{+I}` by the intervention alone.

---

## (d) FairGNN on pokec_z / pokec_n — native "not run by decision" — [확인됨]

### What the results record says

`results/model_dataset_feasibility.csv` rows 23-24:

> FairGNN, pokec_z, primary_controlled=**done**, robustness_controlled=**done**,
> native=**not run (decision)**,
> configuration_source = "primary: utils/param.json; robustness: upstream
> src/scripts/pokec_z (GCN alpha=100 beta=1 H=2000, GAT alpha=10 beta=0.01 H=800)",
> note = "robustness = the upstream GCN and GAT rows; **native not run by decision**"

(pokec_n identical, with `GCN alpha=50 beta=1 H=1800, GAT alpha=4 beta=0.01 H=1800`.)

The literal `"not run (decision)"` string is emitted at
`harness/experiments/build_results.py:960`:

```
960   TD, ND = "done, task-adapted", "not run (decision)"
```

### The decision record — document, date, wording

**Document:** `harness/X31_ADDITIONAL_CELLS_PROTOCOL.md`
("X31 — Additional cells after the method-coverage extension").
**Status line, `:3`:** "Frozen before any X31 outcome is computed or viewed."

**Wording, `harness/X31_ADDITIONAL_CELLS_PROTOCOL.md:9-11`:**

> Decisions taken by the user before any X31 run: **no FairGNN native cells**;
> FnRGNN is included as a task-adapted stratum; order is (1) the FMP baseline
> stage, (2) SFG controlled cells, then the rest.

Restated for the robustness rows at
`harness/X31_ADDITIONAL_CELLS_PROTOCOL.md:166-167` (section 2d, "FairGNN
upstream GCN configuration"):

> * **Off-state:** alpha = beta = 0. Controlled, H = 200. **No native cell (user
>   decision).**

and at `:146-147`:

> The primary FairGNN pokec cells keep the frozen `utils/param.json`
> configuration **(user decision)**.

**Date — [확인됨] partial / [확인 불가] exact.** The two other user decisions in
the same document are timestamped — `:106` "Native: withdrawn (user decision,
**2026-09-22 01:05 KST**)" and `:259` "User decision (**2026-09-22**): FnRGNN is
also run on the task it was released for" — but the FairGNN line at `:9` carries
**no timestamp of its own**. It is bounded only as "before any X31 run", and
X31 follows X30 which was "Written 2026-09-17"
(`harness/X30_METHOD_COVERAGE_EXTENSION_PROTOCOL.md:16`), with X31's other
decisions dated 2026-09-22. Git cannot narrow it further: the repository history
is squashed to a single commit `bb80dde` dated 2026-09-24 00:26:50 +0900. So the
decision is **recorded and pre-outcome [확인됨]**, but its **exact date is
[확인 불가]**; the defensible statement is "on or before 2026-09-22". Proposed
fix: add the timestamp to line 9 in the same form as lines 106 and 259.

**Independent code-level corroboration** that a native FairGNN cell could not
have been run without new work: `harness/core/published_config.py:324-380`,
`native_config()` handles only NIFTY (`:341`), FairGB (`:355`) and FairVGNN
(`:362`), and falls through to

```
377       else:
378           raise KeyError(method)
```

so `native_config("FairGNN", <any dataset>)` raises. This is stated in the
protocol at `harness/X29_COVERAGE_EXTENSION_PROTOCOL.md:140-141`:

> `native_config()` raises for FairGNN on every dataset, and succeeds only for
> the seven pairs already frozen in X23.

### Does an upstream Pokec script exist in the FairGNN repository under `models/`? — [확인됨: NO]

**There is no vendored FairGNN repository under `models/` at all.** The
directory listing of `models/` is:

```
algorithms  BeMap-main  BIND-main  FairGB-main  FairGT-main  FairSIN-main
FMP-main  FnRGNN-master  GEAR-main  SFG-main
```

— no `FairGNN-main`. The only FairGNN artifact under `models/` is the study's
own wrapper `models/algorithms/FairGNN.py`, which is a self-contained
PyTorch-Geometric re-implementation (`models/algorithms/FairGNN.py:9-18`
defines its own `GCN` on `torch_geometric.nn.GCNConv`) and contains no shell
scripts and no pokec configuration. `find models -iname "*pokec*"` returns only
dataset archives and checkpoints belonging to *other* methods
(`models/BIND-main/data/pokec_dataset.zip`, `models/BeMap-main/dataset/pokec`,
`models/FnRGNN-master/dataset/pokec`, `models/FnRGNN-master/model/fairgnn/Pokec_*_md.pth`
— the last are FnRGNN's own FairGNN-baseline checkpoints, not upstream scripts).

The upstream Pokec scripts **do** exist, but under `harness/provenance/`, not
`models/`:

```
harness/provenance/fairgnn_upstream/pokec_z_train_fairGCN.sh
harness/provenance/fairgnn_upstream/pokec_n_train_fairGCN.sh
harness/provenance/fairgnn_upstream/pokec_z_train_fairGAT.sh
harness/provenance/fairgnn_upstream/pokec_n_train_fairGAT.sh
harness/provenance/fairgnn_upstream/train_fairGNN.py
harness/provenance/fairgnn_upstream/models_GAT.py
harness/provenance/fairgnn_upstream/SOURCE.txt
```

`harness/provenance/fairgnn_upstream/SOURCE.txt`, verbatim:

> Upstream: https://github.com/EnyanDai/FairGNN.git @
> 13cdca7627d7779a2396299cdc8edd3a790db7cc (2023-08-17)
> Files: src/scripts/\<dataset\>/train_fair{GCN,GAT}.sh renamed
> \<dataset\>_train_fair{GCN,GAT}.sh; src/train_fairGNN.py (parser defaults).
> models/GAT.py copied as models_GAT.py

`harness/provenance/fairgnn_upstream/pokec_z_train_fairGCN.sh`, verbatim:

```sh
python train_fairGNN.py \
        --seed=42 \
        --epochs=2000 \
        --model=GCN \
        --sens_number=200 \
        --dataset=pokec_z \
        --num-hidden=128 \
        --acc=0.69 \
        --roc=0.76 \
        --alpha=100 \
        --beta=1
```

### Assessment

- **[확인됨]** A published, runnable upstream pokec procedure exists (the four
  `.sh` scripts, H = 2000/800 for pokec_z, 1800 for pokec_n), so a native
  FairGNN pokec cell was *technically feasible*. It was not run, and the reason
  on record is a user decision, not an infeasibility.
- **[확인됨]** The provenance copies are under `harness/provenance/`, not
  `models/`; the question's premise ("in the FairGNN repository under `models/`")
  is false — no FairGNN upstream repository is vendored under `models/`.
- **[추정]** This is the one coverage gap in the study that is a *choice* rather
  than a constraint, and the ledger already says so plainly
  ("native not run by decision"). Blast radius: table 3b/3c has no
  controlled-vs-native pair for FairGNN, so the protocol-sensitivity claim rests
  on the 7 frozen native pairs (FairSIN, NIFTY, FairVGNN, FairGB, SFG) and
  cannot speak to the one method whose configuration the upstream repository
  publishes *only* for pokec. Proposed fix (documentation, no re-run):
  state in the paper that FairGNN's native pokec cells were feasible and
  declined, with the X31:9 citation, rather than listing them beside the
  infeasible exclusions.

---

## Artifacts written

- `results/phase0_audit/T5_manifest_feasibility_correspondence.csv` — 126 rows,
  the manifest ↔ feasibility row correspondence with `in_feasibility ∈ {yes, no}`.
