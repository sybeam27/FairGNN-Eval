# T1 — the common baseline B has test AUC 0.441 on German

Phase-0 factual audit. Nothing under `results/*.csv`, `results/per_unit_metrics.csv.gz`,
`harness/results/` or any trajectory/checkpoint was modified. Every new artifact is in
`results/phase0_audit/`.

Labels: **[확인됨]** verified, **[추정]** inferred, **[확인 불가]** could not determine.

Environment: `~/miniconda3/envs/dev/bin/python`, torch 2.6.0+cu124, PyG 2.8.0.post1.
Every training command in this report was run as

```
cd /home/sypark/workspace/FairGNN-Eval && \
CUDA_VISIBLE_DEVICES=2 ~/miniconda3/envs/dev/bin/python results/phase0_audit/t1_baseline_probe.py
```

Disclosure: an identical first pass of the 3-arm version of this script was executed on
`CUDA_VISIBLE_DEVICES=0` before the GPU-2 constraint reached me. It was discarded and is
not used anywhere in this report; all numbers below come from the GPU-2 run, whose
`repro` arm reproduces the frozen values to 1.2e-4 (see (d)).

---

## One-line conclusions

| # | conclusion | label |
|---|---|---|
| a | B is trained by `harness/experiments/pilot_tau.py:450-463` with `models/algorithms/GNN.py`, at `published("GNN","german")` = hidden 16 / proj 16 / lr 1e-3 / weight_decay 0.0 / `feature_normalize=False` / BCE-with-logits / 1 scalar logit, but at **H = 200**, not at the published horizon 1000. `harness/core/trainer.py` and `harness/core/model.py` are *not* on this path — nothing imports them. | **[확인됨]** |
| b | Score orientation is identical for B and every other arm (positive class = 1, scalar logit, decision `q > 0`, one shared loader). The below-chance AUC is **not** a sign flip. Raw test scores are not stored for B in any artifact, but I regenerated them: AUC(−score) = 0.5585 mean, i.e. exactly 1 − AUC(score) up to ties — a flip would make B "better", which is what a genuine orientation bug looks like, but no code path produces it. | **[확인됨]** |
| c | Per-epoch train loss / val BCE / val AUC are **not stored** for B anywhere (`harness/results/x25,x26,x27/*_trajectories/*.npz` hold M0/M1 only). The **selected epoch is** recoverable, from the `bc_epoch` column of the frozen per-cell CSVs. Regenerating the trajectory shows validation AUC is *also* below 0.5: 0.4306 mean at the selected epoch, below 0.5 in 27/30 cells and in 85.8 % of all 201 epochs. | **[확인됨]** (stored: **[확인 불가]** for val curves, **[확인됨]** for selected epoch) |
| d | Control experiment ran (≈13 min on GPU 2). lr=1e-2 / hidden=64 / H=1000 → mean test AUC **0.6461** (28/30 above 0.5) vs frozen 0.4415. Changing **only the horizon** 200→1000 at the published hyperparameters → **0.6545** (30/30 above 0.5). | **[확인됨]** |
| e | B's configuration comes from `published("GNN", ds)` (`harness/core/published_config.py:261-273`), i.e. `utils/param.json` overridden by the NIFTY README's GCN command. German is `third-party-benchmark`; bail, credit, income and both pokec are `local-unverified`. B does **not** share architecture, width, learning rate, weight decay, preprocessing or backbone with the method arms — the bail/pokec gap is a configuration difference, not an intervention effect. | **[확인됨]** |

---

## (a) Where B is trained, and with what

**Code path** — `harness/experiments/pilot_tau.py`:

| line | what it does |
|---|---|
| `pilot_tau.py:450` | `b_pub = published("GNN", a.dataset)` |
| `pilot_tau.py:451-452` | `b_cfg = dict(b_pub["config"])`; `data = _data(bool(b_cfg.pop("feature_normalize", False)))` |
| `pilot_tau.py:457` | `from FairGate.models.algorithms.GNN import GNN` |
| `pilot_tau.py:458` | `torch.manual_seed(seed); np.random.seed(seed)` with `seed = 27 + run` (`pilot_tau.py:387`, `:417`) |
| `pilot_tau.py:459` | `b = GNN(adj, f, y, itr, iva, ite, s, si, device=dev, **b_cfg)` |
| `pilot_tau.py:460-462` | `ValidationHistory(...)` with a state snapshot per epoch |
| `pilot_tau.py:463` | `b.fit(epochs=a.epochs, trajectory=hb)` |
| `pilot_tau.py:481-486` | B re-read at the `common_bce` / `common_auc` slots, scored through `outcome()` → `core.evaluator.evaluate` |

**Settings**, resolved by `published("GNN","german")` (`harness/core/published_config.py:261-273`,
verified by running it):

```
{'num_hidden': 16, 'num_proj_hidden': 16, 'lr': 0.001, 'weight_decay': 0.0,
 'feature_normalize': False}   horizon=1000   provenance='third-party-benchmark'
 source='harness/provenance/nifty_README.md (NIFTY GCN baseline)'
```

| setting | value | file:line |
|---|---|---|
| hidden size | 16 (`num_hidden`), 16 (`num_proj_hidden`) | `published_config.py:262,271-272`; from `harness/provenance/nifty_README.md:38` (`--hidden 16`), which overrides `utils/param.json`'s `german: {num_hidden: 32, num_proj_hidden: 32}` |
| learning rate | 1e-3 | `published_config.py:265,273` (`--lr 1e-3`, `nifty_README.md:38`) |
| weight decay | **0.0** | `published_config.py:265` — hard-coded `weight_decay=0.0` for the `GNN` branch, *not* `_CLI["weight_decay"]=1e-5` (`published_config.py:70`) |
| epochs (horizon) | **200 actually used**; 1000 published | `pilot_tau.py:389` (`--epochs` default 200) → `pilot_tau.py:463`. `published("GNN","german")["horizon"]` is 1000 and is discarded: `b_pub["horizon"]` is never read. Under `--protocol armA` the method arms are also pinned to `a.epochs` (`pilot_tau.py:495-496`); only `--protocol native` uses a resolved horizon, and even there B stays at `a.epochs` (`b_epochs == 200` in every frozen row of `harness/results/armB_native_*_german.csv`) |
| feature normalization | `False` | `published_config.py:77` (`"GNN": lambda d: False`) applied at `published_config.py:276` |
| loss | `F.binary_cross_entropy_with_logits` on `idx_train` | `models/algorithms/GNN.py:348-349` |
| output dimension | 1 scalar logit (`nclass=1`) | `models/algorithms/GNN.py:186` (`nclass=1`), `:233` (`Classifier(ft_in=num_hidden, nb_classes=nclass)`), `:26` (`spectral_norm(nn.Linear(...))`) |
| optimiser | Adam over `c1` + encoder (`optimizer_2`) | `models/algorithms/GNN.py:240-242`, stepped at `:338,352` |
| architecture | **one** `GCNConv(nfeat, 16)` with **no activation and no dropout**, then a spectral-normed linear head | `models/algorithms/GNN.py:33-40` (`class GCN`), `:161` (`base_model=='gcn'`), `:196`, `:265` |
| dataset / split | `utils.dataloading.load_data("german", feature_normalize=False, split_seed=split)` → `utils/data.py:337` `load_german`, `label_number=100`, `balanced_split` (`utils/data.py:83-108`, `:381`) | |

**Finding a-1 [확인됨].** The task brief points at `harness/core/trainer.py` / `harness/core/model.py`.
Those files are dead code with respect to every published result:
`grep -rn "core.trainer\|core.model"` over the repository returns only
`harness/core/trainer.py:28` importing `harness/core/model.py`. Their defaults
(`trainer.py:53-58`: epochs 500, lr 1e-3, hidden 128, dropout 0.5; `model.py:35-53`: a
2-layer GCN) are **not** B's. `harness/score_convention_manifest.csv` nevertheless cites
`harness/core/model.py` as the evidence path for "GNN (baseline)" — a documentation error,
not a numerical one.

**Finding a-2 [확인됨].** `pilot_tau.py:457` imports `FairGate.models.algorithms.GNN`.
The workspace parent (`harness/core/paths.py:16,19-20` → `/home/sypark/workspace`) contains
only `FairGNN-Eval`, so this import now fails:

```
$ python -c "import sys; sys.path.insert(0,'harness'); import core.paths; from FairGate.models.algorithms.GNN import GNN"
ModuleNotFoundError: No module named 'FairGate'
```

`pilot_tau.py` cannot be re-run as shipped. (My probe imports `models.algorithms.GNN`
directly; same file, same class.)

---

## (b) Score orientation

**[확인됨] The positive class is the same for B as for every other arm.**

* One loader for all arms (`pilot_tau.py:62-74`, `:428-447`; the harness even raises if
  `feature_normalize` changes labels or indices — `pilot_tau.py:437-447`).
* German labels: `predict_attr = "GoodCustomer"` (`utils/data.py:342`), values `{1,-1}`
  mapped by `labels_np[labels_np == -1] = 0` (`utils/data.py:372`). So `y=1` is
  *GoodCustomer*, for every arm.
* B's score is the raw classifier logit (`pilot_tau.py:465-473` → `GNN.forwarding_predict`,
  `models/algorithms/GNN.py:313-318`), trained with BCE-with-logits against that same `y`
  (`GNN.py:348-349`). Higher logit ⇒ higher P(y=1).
* The evaluator fixes the orientation globally and refuses any other:
  `POSITIVE_LABEL = 1` (`harness/core/evaluator.py:54`), `evaluate(..., positive_class=1)`
  raises otherwise (`evaluator.py:204-208`), decision `score>0` (`evaluator.py:56`),
  `auc()` is rank-based Mann-Whitney (`evaluator.py:65-99`). `outcome()` in
  `pilot_tau.py:83-86` passes `decision="score>0"` for B and for every method arm.
* `harness/score_convention_manifest.csv` records `positive_class = 1`,
  `raw_output = scalar_logit`, `decision_rule = q > 0` for all 14 methods **and** for
  "GNN (baseline)". No arm declares a different orientation.

**Raw test scores are not stored for B** in `results/per_unit_metrics.csv.gz` (aggregate
metrics only) nor anywhere under `harness/results/` (the only stored `raw` arrays,
`harness/results/x25|x26|x27/*_trajectories/*.npz`, carry `arm ∈ {M0, M1}` — no B).
So I regenerated them (see (d)). Per split×run values are in the table below:

* AUC(score) mean **0.4415** (4/30 above 0.5)
* AUC(−score) mean **0.5585** (26/30 above 0.5) — i.e. exactly `1 − AUC(score)` up to
  tied scores.

**[추정]** A sign flip would therefore "fix" the number, which is why it is the natural
first hypothesis — but no code path produces one, and the same loader, same logit
convention and same evaluator are used by the arms that score 0.60–0.67 on the same
split. The cause is elsewhere; see (d).

---

## (c) Training trajectories for B

**[확인 불가] — per-epoch train loss, val BCE and val AUC are not stored for B.**
`harness/core/trajectory.py` does compute all three at every epoch
(`ValidationHistory.add`, `trajectory.py:224-260`: `predictive_loss` = stable
BCE-with-logits at `:239-247`, `metrics` via the unified evaluator at `:236`), and
`pilot_tau.py:460-463` wires it into B's `fit`. But the object is never serialised for B:
the only `.npz` trajectory dumps in the repository are
`harness/results/x25/R1{0,1}_trajectories/german_s*_seed*_M{0,1}.npz`,
`harness/results/x26/bail_trajectories/*`, `harness/results/x27/pokec_*_trajectories/*` —
all `arm ∈ {M0, M1}`, none for B.

**[확인됨] — the selected epoch *is* recoverable**, from `bc_epoch` in the frozen per-cell
CSVs (`harness/results/armA_german.csv` col 22, `harness/results/x30/x30_*_german.csv`
col 33). It is written by the `common_bce` slot of `trajectory.py:254-256` (strict `<`,
ties keep the earliest epoch). All 4+ independent campaigns agree on the same `bc_epoch`
per (split, run) — `bc_epoch_nuniq == 1` in every cell.

Distribution of the selected epoch (30 cells, H = 200 so epochs run 0…200):

| bc_epoch | count |
|---|---|
| 200 (= last epoch) | **20** |
| 72–102 | 10 |

**[확인됨] — validation AUC is also below 0.5.** Regenerated (see (d)):
val AUC at the selected epoch = **0.4306** mean, below 0.5 in **27/30** cells; across the
full trajectory **85.8 %** of the 201 epochs have val AUC < 0.5; the *best* epoch by val
AUC only reaches **0.5651** mean (never below 0.5, never above 0.613). So this is not a
test-set artifact and not purely a selector artifact: under this configuration the model's
ranking is anti-correlated with the label for most of training, and even its best epoch is
barely above chance.

Cross-check **[확인됨]**: the frozen `common_auc` selector row for German B gives test AUC **0.5632** (`results/per_unit_metrics.csv.gz`, `dataset=german, state=B, selector=common_auc`, identical under `protocol=controlled` and `protocol=native`). My regenerated best-val-AUC epoch reaches 0.5651 mean — the same quantity, which independently confirms the regenerated trajectory is the one the frozen pipeline produced.

---

## (d) Control experiment

Script: `results/phase0_audit/t1_baseline_probe.py` (new; reuses `utils.dataloading.load_data`,
`models/algorithms/GNN.GNN`, `harness/core/trajectory.ValidationHistory` slot `common_bce`,
`harness/core/evaluator.evaluate` with `decision="score>0"`). Outputs:
`results/phase0_audit/t1_baseline_probe.csv` (raw, 120 rows),
`results/phase0_audit/T1_german_B_per_split_run.csv`,
`results/phase0_audit/T1_probe_arm_summary.csv`.

Command (GPU 2 only, ≈13 min wall clock):

```
cd /home/sypark/workspace/FairGNN-Eval && \
CUDA_VISIBLE_DEVICES=2 ~/miniconda3/envs/dev/bin/python results/phase0_audit/t1_baseline_probe.py
```

Four arms × 6 splits × 5 runs (seed = 27 + run), all with `weight_decay = 0.0`:

| arm | hidden | lr | H | feature_norm | mean test AUC | sd | min | max | n > 0.5 | mean bc_epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| `repro` (= B as frozen) | 16 | 1e-3 | 200 | False | **0.4415** | 0.0535 | 0.3551 | 0.5535 | 4/30 | 162.5 |
| `repro_H1000` | 16 | 1e-3 | **1000** | False | **0.6545** | 0.0605 | 0.5333 | 0.7596 | **30/30** | 886.6 |
| `probe` (T1(d) setting) | **64** | **1e-2** | **1000** | False | **0.6461** | 0.0734 | 0.4718 | 0.7615 | 28/30 | 295.1 |
| `probe_norm` | 64 | 1e-2 | 1000 | **True** | 0.5631 | 0.0784 | 0.3622 | 0.6572 | 26/30 | 3.5 |

**Reproduction check [확인됨].** `repro` matches the frozen `results/per_unit_metrics.csv.gz`
values cell by cell: max |frozen − repro| = **1.2e-4**, and the selected epoch is identical
in all 30 cells. The frozen 0.441 is therefore a faithful record of what the code does —
it is not a post-processing or export bug.

**Cause [확인됨].** The single change that fixes it is the **horizon**. `repro_H1000` keeps
every published hyperparameter and only trains to the published `H = 1000` instead of the
harness's `--epochs 200`, and test AUC goes 0.4415 → 0.6545 with every one of the 30 cells
above chance. Mean selected epoch under H = 1000 is 886.6 — far beyond 200, and 20/30 of
the frozen runs selected the very last available epoch (200), i.e. the runs were still
descending when the horizon cut them off. lr and width matter much less: at lr = 1e-2 the
same H = 1000 gives 0.6461, essentially the same. Feature normalisation is *not* the fix
(0.5631, and it collapses the selector to epoch ~3).

**[추정]** Mechanism: with `feature_normalize=False`, German's raw columns
(`LoanAmount`, `LoanDuration`, …) are O(10³), so the initial logits and the BCE are huge
(observed train loss at epoch 0: 30–210). At lr = 1e-3 the first ~hundreds of epochs are
spent shrinking the scale rather than fitting the ranking, and H = 200 stops inside that
phase. This is consistent with the observation that the val-BCE selector still improves at
epoch 200 while val AUC is still < 0.5.

---

## (e) Provenance of B, and the comparison with the method arms

**Where the configuration comes from [확인됨].** `harness/core/published_config.py:261-273`,
branch `elif method == "GNN":` — start from `utils/param.json["GNN"][dataset]`
(`_params`, `published_config.py:158-170`), force `lr=_CLI["lr"]=1e-3` and
`weight_decay=0.0`, then, *if* the NIFTY README has a `nifty_sota_gnn.py … --model gcn`
command for that dataset, override hidden/proj/lr from it and grade the row
`third-party-benchmark`. Resolved values and grades:

| dataset | B config | horizon field | provenance | source |
|---|---|---|---|---|
| german | hidden 16 / proj 16 / lr 1e-3 / wd 0.0 | **1000** | `third-party-benchmark` | `harness/provenance/nifty_README.md:38` |
| bail | hidden **8** / proj **4** | None | `local-unverified` | `utils/param.json` |
| credit | hidden 32 / proj 4 | None | `local-unverified` | `utils/param.json` |
| pokec_z | hidden **4** / proj 16 | None | `local-unverified` | `utils/param.json` |
| pokec_n | hidden 32 / proj 64 | None | `local-unverified` | `utils/param.json` |
| income | hidden 128 / proj 128 | None | `local-unverified` | `utils/param.json` |

`published_config.py:27` states the rule explicitly: *"A `local-unverified` setting is
never called a published protocol."* Five of the six B rows are `local-unverified`, and
`harness/intervention_manifest.csv` records exactly that in the `provenance` column for
every `B (common baseline)` row. Every B row also records `H = 200`, which for german
contradicts the published horizon 1000 the same resolver returns.

**Is B the same architecture/hyperparameters as the method arms? No [확인됨].**
From `harness/intervention_manifest.csv` (`shared_backbone`, `shared_optimizer_and_hyperparameters`,
`shared_preprocessing`):

| cell | B | the method arm (M_minus_I and M_plus_I) |
|---|---|---|
| bail | GCN, **hidden 8 / proj 4**, lr 1e-3, wd 0.0, `feature_normalize=False` | FairGNN: **hidden 128**, wd 0.0, raw · NIFTY: hidden 8/4 but **wd 1e-5** · BeMap: **hidden 128**, wd 1e-5, BeMap feature_norm · FairGB: **SAGE**, c_lr/e_lr 1e-2, hidden 16, normalised · GEAR: **SAGE, hidden 1024** · FairSIN: hidden 18/54/36, lr 1e-2 · EDITS: nhid 50, wd 1e-5 |
| pokec_z | GCN, **hidden 4 / proj 16**, lr 1e-3, wd 0.0, raw | FairGNN: **hidden 128** · BeMap: hidden 128, wd 1e-5, feature_norm · FairSIN: hidden 18, **c_lr 0.1** · FnRGNN: **hidden 64**, lr 8.3e-3, StandardScaler |
| german | GCN, hidden 16 / proj 16, lr 1e-3, wd 0.0, raw | FairGNN / FairVGNN normalise German (`published_config.py:74-78`); FairSIN/SFG keep German raw; widths 16–128, lrs 1e-3–1e-1 |

Three structural differences, all of which cut the same way:

1. **Depth / nonlinearity [확인됨].** B's encoder is a *single* `GCNConv` with no activation
   and no dropout (`models/algorithms/GNN.py:33-40`, reached via `:161,196`), followed by
   one spectral-normed linear head (`:26, :233`). Most method arms use a 2-layer or
   deeper backbone with a nonlinearity (e.g. `SAGE`, `GNN.py:92-121`; the method repos'
   own encoders).
2. **Width [확인됨].** B is 8 units wide on bail and **4** on pokec_z; the method arms on
   those cells run 16–1024.
3. **Backbone is never varied for B [확인됨].** In `results/per_unit_metrics.csv.gz` the
   `B` rows for `backbone ∈ {GIN, SAGE, GAT}` are numerically identical to the `GCN` ones
   (German B mean per method: 0.441462–0.441484 across all 14 methods and all 4 backbone
   labels). B is always the GCN baseline. So on a `backbone=GIN/SAGE/GAT` cell the
   `B → M_minus_I` contrast confounds the backbone change with everything else.

**The corroborating internal control [확인됨].** `NIFTY` is the one method whose encoder is
the *same* class as B's (`models/algorithms/NIFTY.py:51-57` and `:224` — identical
`Encoder(base_model='gcn')` / single `GCNConv`), configured on bail with the same
`hidden 8 / proj 4` and differing only in `weight_decay=1e-5`. Its `M_minus_I` AUCs are by
far the closest to B's, and on German/credit it is *below* B:

| dataset | B | NIFTY `M_minus_I` | all other methods' `M_minus_I` |
|---|---|---|---|
| german | 0.441 | **0.502** | 0.545–0.666 |
| credit | 0.720 | **0.596** | 0.698–0.748 |
| bail | 0.773 | **0.802** | 0.878–0.917 |

**[추정]** The B ↔ `M_minus_I` gap on bail (0.773 vs 0.878–0.917) and pokec (0.704/0.711 vs
0.73–0.76) is therefore largely attributable to B's narrower, shallower, differently
optimised and differently preprocessed configuration, not to the removal of an
intervention. `M_minus_I` is a well-specified control for `M_plus_I` (same architecture,
one flag flipped); `B` is not a matched control for either, and the manifest's own
`shared_optimizer_and_hyperparameters` column shows it is not shared.

---

## German B per split × run

Frozen values from `results/per_unit_metrics.csv.gz` (`protocol=controlled`,
`selector=common_bce`, `state=B`, averaged over the 14 methods — spread ≤ 1e-4);
`frozen_bc_epoch` from the `bc_epoch` column of the frozen per-cell CSVs;
everything else regenerated by `t1_baseline_probe.py` on GPU 2.
Machine-readable copy: `results/phase0_audit/T1_german_B_per_split_run.csv`.

| split | run | frozen AUC(B) | sel. epoch | repro AUC | **AUC(−score)** | val AUC @ sel | val AUC best | AUC @ H=1000 | AUC probe (lr 1e-2, h 64, H 1000) |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 0 | 0.3966 | 91 | 0.3966 | 0.6034 | 0.3933 | 0.5584 | 0.7455 | 0.7538 |
| 20 | 1 | 0.5374 | 200 | 0.5374 | 0.4626 | 0.4941 | 0.5551 | 0.7596 | 0.7557 |
| 20 | 2 | 0.4765 | 200 | 0.4765 | 0.5235 | 0.5323 | 0.5551 | 0.7335 | 0.7162 |
| 20 | 3 | 0.4077 | 200 | 0.4077 | 0.5923 | 0.3955 | 0.5524 | 0.7326 | 0.7615 |
| 20 | 4 | 0.4587 | 200 | 0.4587 | 0.5413 | 0.4704 | 0.5535 | 0.7419 | 0.6747 |
| 21 | 0 | 0.3748 | 91 | 0.3747 | 0.6253 | 0.3780 | 0.5558 | 0.6824 | 0.6825 |
| 21 | 1 | 0.4841 | 76 | 0.4841 | 0.5159 | 0.4903 | 0.5497 | 0.6732 | 0.6885 |
| 21 | 2 | 0.5047 | 200 | 0.5048 | 0.4952 | 0.4923 | 0.5534 | 0.6880 | 0.6753 |
| 21 | 3 | 0.3900 | 200 | 0.3899 | 0.6101 | 0.4206 | 0.5531 | 0.6951 | 0.6830 |
| 21 | 4 | 0.4035 | 200 | 0.4036 | 0.5964 | 0.4239 | 0.5492 | 0.6846 | 0.6274 |
| 22 | 0 | 0.3627 | 100 | 0.3627 | 0.6373 | 0.3451 | 0.5693 | 0.5565 | 0.5278 |
| 22 | 1 | 0.4344 | 88 | 0.4345 | 0.5655 | 0.4324 | 0.5625 | 0.5333 | 0.6000 |
| 22 | 2 | 0.4507 | 200 | 0.4506 | 0.5494 | 0.4057 | 0.5618 | 0.5630 | 0.6091 |
| 22 | 3 | 0.4369 | 200 | 0.4369 | 0.5631 | 0.4130 | 0.5590 | 0.5707 | 0.5058 |
| 22 | 4 | 0.3551 | 200 | 0.3551 | 0.6449 | 0.3627 | 0.5586 | 0.5427 | 0.4718 |
| 23 | 0 | 0.4007 | 89 | 0.4007 | 0.5993 | 0.3480 | 0.6123 | 0.6682 | 0.6772 |
| 23 | 1 | 0.4898 | 80 | 0.4898 | 0.5102 | 0.4893 | 0.6085 | 0.6616 | 0.6817 |
| 23 | 2 | 0.4458 | 200 | 0.4457 | 0.5543 | 0.4979 | 0.6105 | 0.6639 | 0.6798 |
| 23 | 3 | 0.3962 | 200 | 0.3962 | 0.6038 | 0.3840 | 0.6082 | 0.6558 | 0.6830 |
| 23 | 4 | 0.4472 | 72 | 0.4472 | 0.5528 | 0.4719 | 0.6053 | 0.6542 | 0.6832 |
| 24 | 0 | 0.3590 | 102 | 0.3590 | 0.6410 | 0.3572 | 0.5669 | 0.6720 | 0.6146 |
| 24 | 1 | 0.4838 | 200 | 0.4838 | 0.5162 | 0.5282 | 0.5621 | 0.6485 | 0.6671 |
| 24 | 2 | 0.5535 | 200 | 0.5535 | 0.4465 | 0.5428 | 0.5650 | 0.6619 | 0.6732 |
| 24 | 3 | 0.4445 | 200 | 0.4446 | 0.5554 | 0.4459 | 0.5624 | 0.6799 | 0.6559 |
| 24 | 4 | 0.4901 | 200 | 0.4901 | 0.5099 | 0.4305 | 0.5579 | 0.6491 | 0.6725 |
| 25 | 0 | 0.3997 | 85 | 0.3996 | 0.6004 | 0.3352 | 0.5546 | 0.6551 | 0.4895 |
| 25 | 1 | 0.4612 | 200 | 0.4613 | 0.5387 | 0.4297 | 0.5477 | 0.5746 | 0.6482 |
| 25 | 2 | 0.5201 | 200 | 0.5201 | 0.4799 | 0.4176 | 0.5504 | 0.6268 | 0.5698 |
| 25 | 3 | 0.4001 | 200 | 0.4001 | 0.5999 | 0.4446 | 0.5502 | 0.6324 | 0.6139 |
| 25 | 4 | 0.4787 | 200 | 0.4786 | 0.5214 | 0.3449 | 0.5431 | 0.6278 | 0.6389 |
| **mean** | | **0.4415** | | **0.4415** | **0.5585** | **0.4306** | **0.5651** | **0.6545** | **0.6461** |

---

## Defects found (not fixed, per the audit rules)

**D1 — B is trained at H = 200 while its own resolved protocol says H = 1000.**
*Cause:* `pilot_tau.py:463` passes `a.epochs` (default 200, `pilot_tau.py:389`) to B's
`fit`, while `published("GNN","german")["horizon"] == 1000` is never read (`pilot_tau.py:495-496` uses a resolved horizon only under `--protocol native`, and then only for the method arms). *Blast radius:* every `B` row of
`results/per_unit_metrics.csv.gz` and therefore every `tau_{B→−I}` and `tau_{B→+I}`
estimand in `results/1_main_package_vs_intervention.csv`, Fig. 1, Table 1 and the six-split
sensitivity tables. Measured effect on German: mean B AUC 0.4415 → 0.6545, i.e.
`tau_{B→−I}` for German shrinks by ≈0.21 AUC across all 14 cells. Bail/credit/pokec/income
have no published horizon at all in `published()` (`horizon=None`), so H = 200 is an
unbacked choice there rather than a contradicted one. *Proposed fix:* run B at
`b_pub["horizon"]` when it is not None and record the horizon per row; where it is None,
state the choice as `local-unverified` in the manifest and report the horizon sensitivity.
Do **not** silently re-baseline the frozen files — publish a corrected arm beside them.

**D2 — B is not a matched control for the method arms.** Different width, depth,
weight decay, preprocessing and, on GIN/SAGE/GAT cells, a different backbone entirely
(evidence in (e)). *Blast radius:* the same estimands as D1. `tau_{−I→+I}` is unaffected —
that contrast *is* matched. *Proposed fix:* report `tau_{B→·}` as a
"configuration + intervention" quantity, or add a B variant per cell that matches the
method arm's backbone, width and preprocessing.

**D3 — documentation errors.** (i) `harness/score_convention_manifest.csv` names
`harness/core/model.py` as the baseline's evidence path; B is `models/algorithms/GNN.py`
and `harness/core/{model,trainer}.py` are unimported. (ii) `pilot_tau.py:457` and the
other `FairGate.…` imports no longer resolve after the repository rename
(`ModuleNotFoundError: No module named 'FairGate'`), so the pipeline is not re-runnable
as shipped. *Proposed fix:* correct the manifest path; make `harness/core/paths.py` alias
`FairGate` to the repository root (or change the imports to `models.algorithms.…`).
