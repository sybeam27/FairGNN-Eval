# Phase 0 numerical integrity audit

Scope: fact-checking, not interpretation. Frozen files (`results/*.csv`, `results/per_unit_metrics.csv.gz`,
`harness/results/**`, trajectories) were read only; every artifact of this audit is in this directory.
Nothing found here has been fixed — each item states the proposed change without applying it.

Labels: **[확인됨]** verified, **[추정]** inferred, **[확인 불가]** could not be determined.

| item | status |
|---|---|
| T1 baseline B on German (AUC 0.441) | done — the most consequential finding |
| T2 "common" B differs across methods on Credit / Income | done |
| T3 sign-stability implementation | done |
| T4 NIFTY/German 2×2 and the published selector | done |
| T5 coverage ledger | done |
| T6 FairEdit intervention size | done |
| T7 FairGB backbone label | done |
| T8 intervention activation, 36 cells | done |
| T9 direct uncertainty of Δ_attr (optional) | done |
| T10 common-epoch comparison (optional) | [확인 불가], see T10 |

---

## T1 — the German baseline is below chance because training is cut at 200 epochs

**Conclusion.** The orientation is correct; B on German is simply undertrained. Re-running the same
baseline with the same code and the published horizon (H = 1000) lifts test AUC from 0.4415 to 0.6545.
**[확인됨]**

Evidence
* Where B is trained: `harness/experiments/pilot_tau.py:450-463`, using `models/algorithms/GNN.py` —
  **not** `harness/core/trainer.py` / `model.py`, which nothing imports. Config from
  `published("GNN", "german")` (`harness/core/published_config.py:261-273`): hidden 16, lr 1e-3,
  weight decay 0, `feature_normalize=False` (`:77`), `BCEWithLogitsLoss` (`GNN.py:348`), one scalar logit
  (`GNN.py:186,233`), Adam (`:240-242`), a single `GCNConv` with no activation or dropout (`:33-40,161,196`).
* Horizon: the run uses **H = 200** (`pilot_tau.py:389 → :463`); the resolved published horizon is 1000 and
  `b_pub["horizon"]` is never read.
* Orientation: German `y = GoodCustomer` with −1 → 0 (`utils/data.py:342,372`), one shared loader,
  `POSITIVE_LABEL = 1` and `positive_class != 1` raises (`harness/core/evaluator.py:54,204-208`), decision
  `score > 0` for B and all 14 implementations. Regenerated scores give AUC(score) = 0.4415 and
  AUC(−score) = 0.5585 — **not** a sign flip.
* Selected epochs (from `bc_epoch` in the frozen CSVs): 20/30 units select epoch 200, the last one;
  the other 10 select 72–102. Regenerated validation AUC at the selected epoch averages 0.4306 and is
  below 0.5 in 27/30 units and in 85.8% of all 201 epochs.
* Control (GPU 2 only, ~13 min, `results/phase0_audit/t1_baseline_probe.py`), 6 splits × 5 runs, reusing the
  repository's loader, model, selector and evaluator:

  | arm | test AUC (mean) | units > 0.5 |
  |---|---|---|
  | reproduction: hidden 16, lr 1e-3, H = 200 | 0.4415 | 4/30 |
  | same, H = 1000 | **0.6545** | 30/30 |
  | hidden 64, lr 1e-2, H = 1000 | 0.6461 | 28/30 |
  | + feature normalization | 0.5631 | — |

  The reproduction matches the frozen values to max |diff| 1.2e-4 with identical selected epochs in 30/30,
  so the probe is measuring the same pipeline. **[추정]** for the mechanism: German features are unnormalised
  and O(1e3), epoch-0 BCE is 30–210, and at lr = 1e-3 the first few hundred epochs are spent on scale, so
  H = 200 cuts inside that phase.
* B is not matched to the method arms: German B is hidden 16, while the method arms use hidden 64–128;
  Bail B is hidden 8 / proj 4 against FairGNN 128, BeMap 128, GEAR SAGE-1024, FairGB SAGE. B is always the
  GCN baseline even for cells whose backbone is GIN, SAGE or GAT (B's AUC is identical to 5 decimals across
  all 14 methods and 4 backbone labels). NIFTY, the only method sharing B's encoder class
  (`NIFTY.py:51-57,224`), has the M^{-I} arm closest to B on every dataset. **[확인됨]**

Effect on the paper. τ_{−I→+I} is untouched. Everything referenced to B is affected: τ_{B→−I} and τ_{B→+I}
in `1_main_package_vs_intervention.csv`, Figure 1, Table 1 and the six-split tables. On German the
baseline-to-disabled contrast would shrink by roughly 0.21 AUC if B were trained to its published horizon.
The paper describes B as a common baseline under one protocol; on the evidence above it is a GCN at a
different capacity from the arms it is contrasted with, trained for a fifth of its published horizon.

Proposed (not applied), in order of cost: (1) state the baseline's horizon and capacity explicitly wherever
τ_{B→·} appears, and note that the German baseline is below chance at H = 200; (2) report a
baseline-sensitivity appendix using the H = 1000 probe already computed here; (3) re-run B at the published
horizon and rebuild the B-referenced columns — this changes published numbers and needs a new freeze.

Third finding, unrelated to the numbers: `pilot_tau.py:113,143,191,221,...` import `FairGate.models...`.
The repository was renamed to `FairGNN-Eval` in this session, so those imports now raise `ModuleNotFoundError`
and the pipeline is **not re-runnable as shipped**. 25 files carry the old package name. Proposed: import
through `harness/core/paths.py` instead of a hard-coded top-level package name. **[확인됨]**

---

## T2 — the "common" baseline B is re-trained per method

**Conclusion.** B is shared within `pilot_tau.py` but re-trained per method by the later runners, and on
Credit and Income the re-trained baselines differ materially. Split indices are identical everywhere, so
the matched design of M^{-I}/M^{+I} is not affected. **[확인됨]**

Evidence
* Shared: `harness/experiments/pilot_tau.py:458-463` trains B once per (split, run) before the method loop
  (`:488`); FairGB, FairGNN and NIFTY on Credit are bit-identical in 30/30 units.
* Re-trained: `harness/experiments/x30_run.py:89-107` (`train_B`), called at `:226`, runs once per
  single-method process; x31 runners likewise. Credit carries 12 distinct B values across 14 methods.
* Spread of B (controlled, `common_bce`), maximum over the 30 units:
  Credit AUC 0.0471, DP 0.1022, EO 0.1348; Income AUC 0.0535, DP 0.2539, EO 0.2137;
  Bail / German / Pokec ≤ 3.05e-4 on AUC and exactly 0 on DP and EO.
* Index sets: 192 loads (8 datasets × 6 splits × 4 loader paths); 0 of 48 (dataset, split) cells disagree
  on the train/val/test hashes (`T2_index_hashes.csv`). `x30_run.py:206-210` asserts both arms use B's split.
* Cause: no determinism flags are set in the harness; `harness/X3_BASELINE_NONDETERMINISM.md` already records
  that re-training B diverged in 16/30 cells with max |ΔDP| 0.3362. Reproduced here on BIND/Income: the 18/30
  units whose `bc_epoch` agrees differ by ≤ 0.013 in DP, the 12/30 whose selected epoch differs by up to 0.254.

Effect on the paper. τ_{−I→+I} is unaffected (it never references B). τ_{B→−I} and τ_{B→+I} on **Credit and
Income cells** carry a baseline-draw component; those terms enter the "surrounding package is larger"
counts (26/36 on −Δ_DP, 29/36 on −Δ_EO, 31/36 on ΔAUC) and every package-level statement.

Proposed (not applied): report the measured baseline-draw floor beside credit/income B-referenced numbers; or
publish a second view in which all B-referenced terms are joined to one canonical B per (dataset, split, run) —
noting that this breaks the same-process invariant X3 relies on.

---

## T3 — sign stability is a per-unit share, not a bootstrap share

**Conclusion.** The stored `sign_stability` is the share of non-zero **units** on the majority side, not the
share of bootstrap replicates. The frozen counts 11/8/9 are reproduced from the published per-unit export.
Under a bootstrap-replicate reading the counts become 13/12/11. **[확인됨]**

Evidence
* Formula: `harness/experiments/analyze_armA.py:40-43`, `max((nz>0).mean(), (nz<0).mean())` over non-zero units;
  thresholds at `:33-34`; fed the raw 30-unit frame at `analyze_x29.py:141,150`. The 10,000 replicates at
  `analyze_x29.py:140` feed **only** the interval (`:148`). Numerical proof: BIND/Bail ΔAUC = 0.6333 = 19/30.
* Recomputing the share as "units whose sign equals the point estimate" leaves all 108 flags unchanged
  (11/8/9). The two definitions differ in 8 of 108 cells, all near 0.5 (`T3_sign_stability_recompute.csv`).
* Bootstrap-replicate share (10,000 replicates, splits then runs, seed 20260914): resolved becomes
  ΔAUC 13, −Δ_DP 12, −Δ_EO 11; nothing is lost, 8 cells are added.
* The 8 cells that pass the interval and magnitude conditions but fail the unit share: NIFTY/Bail 0.633 and
  NIFTY/German 0.667 (ΔAUC); FairGB/Bail 0.733, FairGNN/Pokec-z 0.700, SFG/Bail 0.733, SFG/German 0.700
  (−Δ_DP); FairGNN/Pokec-z 0.633, SFG/German 0.733 (−Δ_EO). Three sit at 22/30, one unit short of the threshold.
* Rule implementations: canonical `analyze_x29.resolved` for sections 1, 1b, 2, 3b, 3c and the BCE side of 3a;
  re-implemented inline for the AUC side of 3a (`build_results.py:234-235`) and upstream of sections 4 and 5
  (`analyze_x24_factorial.py:97`, `analyze_x26_fairgb.py:225`); deliberately different in
  `robustness_six_split.py:37,56,60-61`. Four spellings, none disagreeing today, nothing enforcing agreement.

Effect on the paper. An interval that excludes 0 forces the replicate share above 0.975, so under the
bootstrap reading the sign clause is redundant and the rule collapses to "|mean| ≥ 0.010 and interval excludes
0". A reader who implements it that way gets 13/12/11 and cannot reproduce 11/8/9.

Proposed (not applied): state in the appendix that sign stability is the per-unit share; add a single shared
`resolved()` and an assertion in `build_results.py`.

---

## T4 — NIFTY/German factorial, and the "published selector" wording

**(a) −0.117 is the D-averaged horizon effect, not ΔH|D=1. [확인됨]**
`harness/X24_NIFTY_GERMAN_FACTORIAL.md:129` and `analyze_x24_factorial.py:157` define Δ_H as
½[(θ10−θ00)+(θ11−θ01)] = −0.1171 [−0.1836, −0.0550]. The simple effect at D=1 is θ11−θ01 = **−0.1799**
[−0.2543, −0.1089] (dedicated rerun −0.1768). Every label inside the repository is correct; the manuscript
text is not in the repository, so whether it mislabels this is **[확인 불가]** — but if it says
"ΔH|D=1 ≈ −0.117", that is the wrong number by a factor of 1.54.

**(b) The interaction term changes resolved status between run sets. [확인됨]**
Recomputed intervals for every level and derived quantity (`T4_nifty_factorial_ci.csv`,
`t4_factorial_ci.py`; 10,000 replicates, splits then runs, the same resampled units applied to all four
cells, seed 20260924). The seed of every unit is identical across all six source files (30/30), so mixing
run sets does not mix seeds.

| run set | Δ_H | Δ_D | Γ_HD | Δ_H\|D=1 |
|---|---|---|---|---|
| frozen | −0.1171 [−0.1845, −0.0561] R | −0.0275 [−0.0863, +0.0168] u | −0.1255 [−0.2201, −0.0423] **u** (0.733) | −0.1799 [−0.2553, −0.1073] R |
| rerun | −0.1065 [−0.1760, −0.0430] R | −0.0350 [−0.0980, +0.0130] u | −0.1405 [−0.2396, −0.0552] **R** (0.767) | −0.1768 [−0.2494, −0.1035] R |

The interaction changes resolved status between run sets on a margin of one unit (22/30 vs 23/30).
Under the frozen rule's own branch order this flips the case label from horizon-dominant to
interaction-dominant. Terms, intervals and run sets: `T4_terms.csv`.

**(c) "selector → the method's own published rule" is a hard-coded string. [확인됨]**
`build_results.py:1043` (and `:529-531`) concatenates that phrase unconditionally for every native row.
For NIFTY the reported −0.151 is selected by σ_c^BCE; the selector was **not** changed. NIFTY's real published
rule is `argmin(val_c_loss + val_s_loss)` (`NIFTY.py:513-515,525-532`) and it disagrees with σ_c^BCE in 27/30
units. Affects `factors_changed` in 3b (4 rows) and 3c (4 rows) and the selector field of ~7 native rows in
`method_configurations.csv`; **no number changes**. It also affects the label of Figure 3(b) ("Horizon +
selector") and the corresponding sentences.
D1.bce.frz (−0.15102324761229) and the 3c NIFTY row (−0.15102324761229) are the same run, same 30 units, same
selector; only the intervals differ, because the two bootstrap seeds differ. D1.auc.frz = −0.0200
[−0.0918, +0.0490], unresolved, as expected.

**(d) FairGB/Bail −0.010 vs −0.019 is a run-set difference. [확인됨]**
3b's −0.010277 is the frozen `armB_native_FairGB_bail.csv`; X26's −0.018934 is the native-length retrain in
`harness/results/x26/x26_bail.csv`. Same selector, same horizon (1500), same full support, same (split, run)
sets; the rerun lies inside the frozen interval, so X26's reproduction gate passes. Both are unresolved.
Proposed (not applied): print a run-set provenance column wherever the two appear together.

---

## T5 — coverage ledger

* `harness/coverage_manifest.csv` exists with **126 data rows** = 14 methods × 9 dataset settings, enforced in
  `build_manifests.py:99-101,126`. Categories: included 36, included_reported_separately 5,
  unsupported_released_configuration 75, matched_intervention_not_separable 5, invalid_evaluation_split 4,
  required_artifact_unavailable 1. The 36 `primary_eligible=yes` rows equal the 36 rows of
  `1_main_package_vs_intervention.csv` exactly. **[확인됨]**
* Relation to `model_dataset_feasibility.csv`: the manifest is the dense completion of that 54-row ledger
  (`build_manifests.py:93-108`); outer merge gives 54 matched, 72 manifest-only, 0 feasibility-only
  (`T5_manifest_feasibility_correspondence.csv`). **Only 3 of the 75 "unsupported" rows were adjudicated**
  (BIND/Pokec-n, BIND/Pokec-z, GEAR/Credit); the other 72 are inferred absence. Describing all 126 as audited
  candidates overstates the work by 72 rows. **[추정]** Proposed: add a `decision_source` column.
* FairGT: the intervention is one unconditional block —
  `models/FairGT-main/train_fairgt.py:84-103` concatenates the eigenvector positional encoding (`:98`) and
  substitutes the same-sensitive complete graph (`:100`); the only escape selects a different model. So no
  off-state exists and M^{-I} cannot be built. **[확인됨]**
* FairGNN on Pokec: the native cell was **declined, not infeasible**. Decision recorded at
  `harness/X31_ADDITIONAL_CELLS_PROTOCOL.md:9-11` (restated `:166-167`), with no timestamp on that line, so the
  date is **[확인 불가]** (bounded by other dated lines as on or before 2026-09-22). The upstream script exists:
  `harness/provenance/fairgnn_upstream/pokec_z_train_fairGCN.sh` (α=100, β=1, 2000 epochs).

---

## T6 — FairEdit removes 10 undirected edges and adds none

| dataset | edit_num | removed (directed cols) | added | total edges | identical across runs |
|---|---|---|---|---|---|
| bail | 10 | 20 | 0 | 642,616 | yes (60/60) |
| credit | 10 | 20 | 0 | 2,873,716 | yes (60/60) |
| german | 10 | 20 | 0 | 44,484 | yes (60/60) |

Additions are structurally impossible: `models/algorithms/FairEdit.py:711` sets `add=False` and the add branch
(`:713-722`) is dead. Per-run counts are recorded in the `config` column of
`harness/results/x30/x30_FairEdit_*.csv` (`harness/adapters/x30_fairedit.py:87,90`). **[확인됨]**

The caveat "20 edges out of 44k" is **correct for German only** and wrong for Bail (14×) and Credit (65×); it
is a dataset-blind literal at `harness/experiments/x30_config_table.py:192` and
`harness/experiments/build_results.py:248,250,252`, reaching 3 rows of `method_configurations.csv` and 3 caveat
rows of `1_main_package_vs_intervention.csv`. Real fractions: German 4.5e-4, Bail 3.1e-5, Credit 7.0e-6.
Presentation only; no computed number depends on it. Proposed: format the caveat from the stored
`edges_before`, and state that additions are disabled.

---

## T7 — FairGB is SAGE; the per-unit export says GCN

**Conclusion.** SAGE is correct. `results/per_unit_metrics.csv.gz` (and the frozen run CSVs it copies) label
FairGB as GCN. **[확인됨]**

`pilot_tau.py:547` writes the literal `backbone="GCN"` for every method; `build_manifests.py:158-163` copies it.
FairGB's config never sets an encoder, so `FairGB_alg.py:47` falls back to `encoder='SAGE'`
(`FairGB/models.py:101-107`, `SAGEConv`). All other `results/*.csv` pass through
`build_results.py:132-137` with `CANONICAL_BACKBONE["FairGB"]="SAGE"` and are correct.
Values are unaffected (FairGB/Bail τ_I on −Δ_DP recomputes to −0.059678). The practical damage: joining
`per_unit_metrics.csv.gz` to any `results/*.csv` on (method, dataset, backbone) returns **zero FairGB rows**.
Proposed (not applied): map through `canonical_backbone` in `build_manifests.py` and note it in
`results/README.md`. The frozen run CSVs stay as they are.

---

## T8 — the intervention is active in every cell, but two cells sit at the noise floor

All 36 primary cells matched at 30 units. In 26 cells no unit has identical arms. Exceptions:
FairEdit/German 7/30, **FairEdit/Credit 5/30**, FairGB/German 1/30 (`T8_activation.csv`).
Smallest movers: FairEdit/Credit max |ΔAUC| = 4.0e-6 with max |ΔDP| = max |ΔEO| = 0.0 (effectively inert);
FairEdit/Bail 1.18e-4; FairGNN/Bail 7.73e-4; FairEdit/German 1.37e-3. Largest: BeMap/Bail 0.393.

FairGNN/Bail (α=2, β=0.05) is wired correctly: `models/algorithms/FairGNN.py:125-132` computes the covariance
and adversary terms and backpropagates `G_loss = cls_loss + α·cov − β·adv_loss` every epoch, unconditionally
(`:195`). Full-precision recheck on `harness/results/armA_bail*.csv`: max |ΔAUC| 7.72e-4, 0/30 units identical,
selected epochs equal in 30/30. **[확인됨]** Why it is so small: on Bail only 100 of 18,876 nodes carry a true
sensitive value (`pilot_tau.py:135`, `harness/core/datasets.py:48`), so the covariance term stays around 1e-3;
the same code moves ΔAUC by 2.9e-2 on Pokec-z. **[추정]**
Prediction-vector identity: **[확인 불가]** — FairGNN stores scalars only; checking it needs a GPU re-run.

Effect on the paper: FairGNN/Bail and FairEdit/Credit report τ_I on differences at or below the noise floor.
Proposed: keep them in the table with a footnote that the intervention is numerically inert in that setting.

---

## T9 — direct uncertainty of the attribution shift (optional task)

Δ_attr = τ_alt − τ_primary as its own estimand, matched per unit where both sides exist, with the paper's
paired hierarchical bootstrap (10,000 replicates, splits then runs, seed 20260924). Script: `t9_delta_attr.py`,
output `T9_delta_attr.csv`.

| family | comparisons | intervals excluding 0 (of 3 coordinates each) | median abs Δ |
|---|---|---|---|
| configuration | 26 | 28 / 78 | 0.016 |
| selector | 36 | 27 / 108 | 0.006 |
| native horizon + selector | 21 | 8 / 63 | 0.004 |
| published procedure | 4 | 6 / 12 | 0.064 |

On −Δ_DP alone: configuration 8/26, selector 8/36, native 3/21, published procedure 2/4.
Pairing is recorded per row: selector pairs are the same trained run evaluated twice (`same_run`); the others
are separate executions on the same splits (`same_split_different_run`).

Effect on the paper: the sign-flip counts (e.g. 13/26 on −Δ_DP under configuration change) are much larger than
the number of shifts whose own interval excludes zero (8/26). Both can be reported; only one of them is a
statement about evidence.

---

## T10 — common-epoch comparison (optional task)

**[확인 불가] for all 36 primary cells.** Per-epoch trajectories exist only for
`harness/results/x25/R10,R11_trajectories` (NIFTY/German, H=1000), `harness/results/x26/bail_trajectories`
(FairGB/Bail, native H=1500) and `harness/results/x27/pokec_{z,n}_trajectories` (FMP). All of these are
native-length reruns; the controlled H=200 runs that produce the 36 primary cells stored no per-epoch record.
The artifact needed is a per-unit trajectory (per-epoch validation loss and test outcome for both arms) under
the controlled protocol, which requires a re-run.

---

## Step 1 — import paths made checkout-name independent (verification open)

`harness/core/paths.py` now puts the repository root on `sys.path`, and 36 files / 41 sites were rewritten
from `FairGate.models.*` to `models.*`, so the code no longer depends on what the checkout is called.
All Python files compile and `models.algorithms.{GNN,NIFTY,FairGNN}` import successfully. No logic changed.

Verification (NIFTY/German controlled, 60 rows, GPU 2, `step1_nifty_german.csv`): **not within 1e-4**.
B and M^{-I} match the frozen run (selected epoch identical 60/60; max |diff| 1.5e-4 and 7.6e-5 on AUC),
but M^{+I} differs (max |ΔAUC| 9.5e-3, |ΔDP| 0.027, |ΔEO| 0.040) and its selected epoch differs in 13/60
units. τ_I on −Δ_DP: −0.010022 vs the frozen −0.006452. **[추정]** this is the CUDA nondeterminism already
recorded in `harness/X3_BASELINE_NONDETERMINISM.md` and `X25_RESULTS.md` (NIFTY's M1 arm diverging across
processes), not a consequence of the import change, because the deterministic parts of the pipeline
reproduce and only the intervention arm moves. Confirming it requires re-running the same cell from the
pre-change commit and comparing the spread. **Held here pending that control run.**

## T11 — compute environment (for the paper's reproducibility section)

See `T11_compute_environment.md`. One GPU at a time (`CUDA_VISIBLE_DEVICES=2`, enforced at
`x30_run.py:175-176`); the GPU model is **not** recorded in any log — inferred as RTX 6000 Ada (48 GB) from an
OOM traceback and the current machine. Roughly 44.4 GPU-hours for the 36 primary cells (~1.23 h per cell) and
~81 GPU-hours for everything reported. Splits 20–25 × runs 0–4, seeds {27, 28, 29, 30, 31}, 30 units per cell,
93/93 cells complete. Environments: `dev` (torch 2.6.0+cu124, PyG 2.8.0) and `x27_dgl_cuda`
(torch 2.2.2+cu121, dgl 1.1.3, PyG 2.5.3); no environment.yml or requirements.txt exists in the repository.

---

## Step 2 — artifact fixes (no reported number changes)

* **T7 backbone / configuration.** `results/per_unit_metrics.csv.gz` now carries `method`,
  `configuration`, `backbone` and `store_method` separately: the stored name (`BIND-1pct`,
  `FairSIN-GCN`) is split with `build_results.split_name` and the backbone comes from
  `canonical_backbone`, the same map the section tables use (`build_manifests.py`, `per_unit_metrics`).
  FairGB's 1,080 rows now read SAGE. The file joins **1:1 with `cell_results.csv`** on
  (method, dataset, backbone, configuration, protocol): 90 cells both sides, 0 unmatched either way.
* **Export precision.** The 6-decimal rounding is gone. Point-estimate reproduction improves from
  1.96e-7 to **9.8e-17**, and the 5 manufactured ties in FairEdit/Credit disappear (0/30). File size
  186 KB → 355 KB.
* **T6 caveat.** `build_results.py` and `x30_config_table.py` now emit the per-dataset wording.
  Symmetry checked: `models/algorithms/FairEdit.py:727-731` deletes one direction and `:733-736` the
  reverse, called 10 times (`:780-781`), i.e. **10 undirected edges = 20 directed entries**, identical
  on all three datasets. **[확인됨]**
* **Verification.** `phase0_verify.py` cannot read the new schema (it looks up `BIND-1pct` as a
  method). An adapted copy, `phase0_verify_v2.py` (3 lines different), reproduces sections 1–6:
  resolved counts 11/8/9, bootstrap-rule 13/12/11, headline 31/26/29 and 9/23/24, sign flips 5/13/12
  with 1/2/4 substantive, NIFTY 2×2 −0.1171 / −0.1799 — all unchanged. The only movement is the two
  improvements above (reproduction error, and 9 → 8 rows where the two sign-stability definitions
  differ).

## R2 — the native `factors_changed` string is wrong in all 25 rows

**[확인됨]** The selector clause is false in **25/25** native rows (FairSIN 15, SFG 3, FairGB 3,
FairVGNN 3, NIFTY 1); the horizon clause is correct in all 25, and the preprocessing / training-loop
clauses are correct where present. The reported native τ_I is built from the `common_bce` slot
(`pilot_tau.py:520-525`); over all 25 stores `max|int_ndp + (m1_dp − m0_dp)| ≤ 1.7e-16`, whereas the
published-rule outputs differ by ≥ 3.0e-2. `published_config.native_config` has no `selector` key at
all; the string is the constant at `build_results.py:1046`.

Each method's own published rule **is** implemented and recorded (`code_epoch`, `m1pub_*`) but feeds
only τ_pkg, and the M^{-I} arm has no published-rule counterpart, so a published-rule τ_I was never
constructible. Published-rule and σ_c^BCE epochs agree in only 0–8 of 30 units per cell.

Consequences: section 3b is **horizon only**, not "horizon and selector"; the same error is inherited
by `results/README.md` (`build_results.py:893-894, 1189-1192`) and by the label of Figure 3(b).
Corrected wording: `selector unchanged (σ_c^BCE, same as the controlled arm); the method's own
published rule was replayed and recorded as code_epoch / m1pub_*, but does not enter this estimate`.
Table: `R2_native_selector_facts.csv`.

## T9 — Δ_attr intervals excluding zero, by coordinate

| family | pairs | −Δ_DP | −Δ_EO | ΔAUC |
|---|---|---|---|---|
| configuration | 26 | 8 | 9 | 11 |
| selector | 36 | 8 | 7 | 12 |
| native horizon (+ selector as printed) | 21 | 3 | 2 | 3 |
| published procedure | 4 | 2 | 2 | 2 |

## T12 — nondeterminism, stated precisely

* **"16 of 30" is units inside one cell, not cells. [확인됨]** German only, the common baseline B, a
  retrained B in a second process against the stored B at the same seed, split, device and config
  (`harness/experiments/baseline_sigma_c.py:45-49,61-62,87,99`). Recomputed exactly: |ΔAUC| mean
  0.0128 / max 0.0754; |ΔDP| mean 0.0306 / max **0.3362**; 16/30 units beyond 1e-3; the maximum is
  split 24 run 2 (DP 0.000000 vs 0.336188).
* **"every method uses a sparse path" is not allowed. [확인됨]** Four different implementations are in
  play: PyG message passing (B, BIND, EDITS, FairEdit, FairGNN, NIFTY, FairGB, GEAR, SFG, FnRGNN,
  FairSIN-GIN/SAGE), DGL `GraphConv` (BeMap, FMP's propagation stage), `torch.spmm` (FairVGNN) and
  `torch_scatter` (FairSIN-GCN). FMP's F00 configuration (λ₂ = 0) has **no propagation at all** — it is
  a pure MLP (`models/FMP-main/fmp.py:135-139`) and it is a reported cell (18 rows of
  `5_component_case_study_FMP.csv`). EDITS' and GEAR's dense operations are preprocessing only.
* **A sparse CUDA path is necessary but not sufficient. [확인됨]** `e0_noise_floor.csv` (recovered
  read-only from the old history, commit `ea10163`, in `recovered/`): 9 settings × 10 repeats at a
  fixed seed. Within-seed ΔDP standard deviation — German 0.0351 (range 0.1185), Pokec-z-g 0.0098,
  Credit 0.0046, Pokec-n 0.0011, and **NBA and Income exactly 0.0000, bit-identical across all ten
  repeats** on the same CUDA sparse path.

## R3 — SFG/German and SFG/Credit in section 3b are a pure re-run

**[확인됨]** Nothing differs between the controlled and the native execution of SFG on german (or
credit) except the execution itself. SFG never enters `published_config.py` (both `published` and
`native_config` raise `KeyError`, `:274`, `:378`); its configuration comes from `x31_sfg.config(ds)`,
a pure function of the upstream `run.sh` that takes no protocol argument. The two dicts diff to the
empty set and the stored `config` column is byte-identical on all 60 matched rows. The `--native`
flag reaches only `m_epochs`, and `native_horizon` is 200 for both datasets, so it is inert (it bites
only on bail, 200 → 160). Seeds match on 60/60 units.

The decisive control: the harness-trained baseline is **identical** across the two executions
(`bc_dp`, `bc_epoch` equal 60/60, `b_auc` max diff 1.5e-4), which fixes the data, the split and the
seed and isolates the divergence to SFG's own training loop. There, 30/30 units differ and the
selected epoch moves by up to 184 epochs.

So the 3b row for SFG/German measures no horizon effect — it measures the instability of SFG's
validation-BCE checkpoint selection under nondeterministic CUDA. The paired difference is
−0.0457 with a bootstrap interval of [−0.108, +0.019], which contains 0; on credit it is +0.0005.

**Two corrections to what this audit said earlier.** (i) The cell does not move from resolved to
unresolved: both sides are unresolved (`resolution_changed=False`), because sign stability is 0.700
controlled and 0.600 native — the interval excluding 0 is not sufficient under the frozen rule.
(ii) The shipped 3b CSV still prints the old selector string; the generator now emits the corrected
one, and the CSV will carry it when the bundle is rebuilt. Full evidence:
`R3_sfg_controlled_vs_native.md`.
