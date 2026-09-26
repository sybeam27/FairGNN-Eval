# T4 — NIFTY/German 2×2 factorial and the published selector

Audit date 2026-09-24. All work CPU-only (pandas/numpy over stored per-unit CSVs).
**No GPU was used and no training was re-run.** Frozen files were read only.

Commands run (all from `/home/sypark/workspace/FairGNN-Eval`, python
`~/miniconda3/envs/dev/bin/python`):

```
python /tmp/.../scratchpad/t4b.py       # bootstrap over the 4 protocol CSVs + x25 reruns
python /tmp/.../scratchpad/t4b2.py      # bootstrap over harness/results/x25/x25_cell_table.csv
```

Both scripts are reproduced verbatim in `results/phase0_audit/T4_recompute.py`.
Outputs: `results/phase0_audit/T4_terms.csv`.

---

## (a) What the number −0.117 actually is

**[확인됨] −0.117 is Δ_H, the D-averaged horizon main effect, not ΔH|D=1.**

Definition, pre-registered:

* `harness/X24_NIFTY_GERMAN_FACTORIAL.md:129`
  `Δ_H     = ½[(v10 − v00) + (v11 − v01)]`
* `harness/X24_NIFTY_GERMAN_FACTORIAL.md:133` — the *simple effects* `v10 − v00`
  and `v11 − v01` are a separate, additionally reported quantity.

Implementation, identical:

* `harness/experiments/analyze_x24_factorial.py:157`
  `t[f"H_{c}"] = 0.5 * ((v["P10"] - v["P00"]) + (v["P11"] - v["P01"]))`
* `harness/experiments/analyze_x24_factorial.py:161`
  `t[f"Hs1_{c}"] = v["P11"] - v["P01"]`   (= ΔH|D=1)

Values in the frozen record:

| quantity | value | file:line |
|---|---|---|
| Δ_H (D-averaged) | **−0.117** [−0.184, −0.055] s0.90 R | `harness/X24_RESULTS.md:78`; also `:93`, `:112`, `:127` |
| Δ_H, 4-dp | **−0.1171** [−0.1836, −0.0550] | `harness/results/tables/paper/tableA3_nifty_factorial.tex:25` |
| ΔH given D1 (v11 − v01) | **−0.180** [−0.254, −0.109] s0.90 R | `harness/X24_RESULTS.md:87` |
| same, dedicated rerun (X25 "H" column, D1/BCE) | **−0.1768** [−0.2488, −0.1027] R | `harness/X25_RESULTS.md:91`; `harness/results/tables/paper/tableA4_x25_decomposition.tex:36` |
| X25 prose restating it | "H = −0.177" | `harness/X25_RESULTS.md:178` |

Cell means, all frozen and all reproduced by this audit (see (b)):
θ00 = −0.0065, θ01 = +0.0288, θ10 = −0.0608, θ11 = −0.1510
(`harness/X24_RESULTS.md:65-68`; `tableA3_nifty_factorial.tex:10,13,16,19`).
Arithmetic check: ½[(−0.0608 − (−0.0065)) + (−0.1510 − 0.0288)] = −0.1171 ✓;
(−0.1510 − 0.0288) = −0.1799 ✓.

**Verdict.** The frozen harness documents, the analysis script and the generated
LaTeX table all label −0.117 correctly as the **D-averaged horizon main effect
Δ_H**. If the paper prose attaches the label "ΔH|D=1" to −0.117, **the label does
not match the number**: ΔH|D=1 is −0.180 on the frozen runs and −0.177 on the
dedicated rerun. The two differ by a factor of ~1.54 and the mislabel also
silently drops the information that the H effect is small and unresolved at D0
(−0.054, `harness/X24_RESULTS.md:86`), which is the entire content of the
interaction discussion at `harness/X24_RESULTS.md:96-106`.

**[확인 불가]** The paper manuscript (prose/PDF) is not in this repository — only
`PAPER_ARTIFACT_MAP.md`, `results/*.csv` and `harness/results/tables/paper/*.tex`.
`grep -rn "0.117"` over the repo returns only the four `X24_RESULTS.md` lines
above, one quote in `harness/X25_NIFTY_TRAJECTORY_SELECTION_PREREG.md:13`
("Δ_H −0.117, resolved"), `tableA3…tex:25`, and unrelated CI endpoints. So the
mislabel cannot be located in a repo file; it is a paper-text-only defect.

Proposed fix (not applied): in the paper, write "Δ_H = −0.117 (averaged over D);
the horizon effect at the native configuration is ΔH|D=1 = −0.180
(rerun −0.177), and ΔH|D=0 = −0.054, unresolved."

---

## (b) Recomputed point estimates and 95 % intervals

Method: exactly the frozen paired hierarchical bootstrap
`harness/experiments/bootstrap_armA.py:50-62` (`boot()`) — resample the 6 splits
with replacement, then the runs within each drawn split with replacement, every
column of a row travelling together so all four protocol cells stay matched.
**10,000 replicates. Seed = `np.random.default_rng(20260916)`** (the same
generator seed the frozen analyzer uses, `analyze_x24_factorial.py:38`), drawn
once for the single 12-column table. Coordinate −ΔDP (`int_ndp`), selector
`common_bce`. Resolved rule unchanged (s ≥ 0.75, |mean| ≥ 0.010, CI excludes 0).

### Per-cell source files

| cell | H | D | file used | rows |
|---|---|---|---|---|
| θ00 (P00) | 200 | D0 | `harness/results/armA_german.csv` (NIFTY/german rows) | 30 cells × 2 selectors |
| θ10 (P10) | 1000 | D0 | `harness/results/x24_nifty_german_P10.csv` | 60 |
| θ01 (P01) | 200 | D1 | `harness/results/x24_nifty_german_P01.csv` | 60 |
| θ11 (P11) | 1000 | D1 | `harness/results/armB_native_NIFTY_german.csv` | 60 |
| rerun θ10 | 1000 | D0 | `harness/results/x25/x25_R10.csv` (and the replayed column `D0.bce.c1000.ndp` of `harness/results/x25/x25_cell_table.csv`) | 60 |
| rerun θ11 | 1000 | D1 | `harness/results/x25/x25_R11.csv` (and `D1.bce.c1000.ndp` of `x25_cell_table.csv`) | 60 |
| rerun θ00, θ01 | 200 | — | **[확인 불가] — no H = 200 rerun exists.** X25 ran only R10 and R11 at H = 1000 (`harness/X25_RESULTS.md:16-17`) and itself borrows the frozen H = 200 cells as the bridge (`harness/experiments/analyze_x25_trajectory.py:5-6`, `:133` `h200`). Both rerun rows below therefore reuse the frozen P00/P01 for θ00/θ01. |

### Results (−ΔDP, σ_c^BCE)

**Frozen run set** — reproduces `harness/X24_RESULTS.md:65-68, 78, 86-89` exactly
to 4 dp:

| term | mean | 95 % CI | s | resolved |
|---|---|---|---|---|
| θ00 | −0.0065 | [−0.0426, +0.0323] | 0.60 | u |
| θ01 | +0.0288 | [−0.0064, +0.0736] | 0.63 | u |
| θ10 | −0.0608 | [−0.1285, +0.0048] | 0.70 | u |
| θ11 | −0.1510 | [−0.2243, −0.0824] | 0.93 | **R** |
| **Δ_H (D-averaged)** | **−0.1171** | [−0.1836, −0.0550] | 0.90 | **R** |
| Δ_D | −0.0275 | [−0.0845, +0.0178] | 0.57 | u |
| **Γ (interaction I_HD)** | **−0.1255** | [−0.2174, −0.0446] | 0.73 | u (CI excludes 0; fails only on s) |
| **ΔH \| D=1** | **−0.1799** | [−0.2543, −0.1089] | 0.90 | **R** |
| ΔH \| D=0 | −0.0544 | [−0.1404, +0.0258] | 0.67 | u |
| Δ_D \| H=200 | +0.0353 | [+0.0059, +0.0720] | 0.77 | **R** |
| Δ_D \| H=1000 | −0.0902 | [−0.1901, −0.0133] | 0.60 | u |
| Δ_total | −0.1446 | [−0.2268, −0.0605] | 0.83 | **R** |

**Dedicated rerun (X25 R10/R11 at H = 1000; H = 200 cells frozen)** — pipeline
`int_ndp` column:

| term | mean | 95 % CI | s | resolved |
|---|---|---|---|---|
| θ00 | −0.0065 | [−0.0426, +0.0323] | 0.60 | u *(frozen, not rerun)* |
| θ01 | +0.0288 | [−0.0064, +0.0736] | 0.63 | u *(frozen, not rerun)* |
| θ10 | −0.0427 | [−0.1184, +0.0299] | 0.67 | u |
| θ11 | −0.1480 | [−0.2208, −0.0773] | 0.90 | **R** |
| **Δ_H (D-averaged)** | **−0.1065** | [−0.1762, −0.0425] | 0.83 | **R** |
| Δ_D | −0.0350 | [−0.0955, +0.0127] | 0.57 | u |
| **Γ (interaction)** | **−0.1405** | [−0.2384, −0.0573] | 0.77 | **R** *(flips to resolved on the rerun)* |
| **ΔH \| D=1** | **−0.1768** | [−0.2499, −0.1037] | 0.93 | **R** |
| ΔH \| D=0 | −0.0362 | [−0.1269, +0.0493] | 0.57 | u |
| Δ_D \| H=200 | +0.0353 | [+0.0059, +0.0720] | 0.77 | **R** *(frozen inputs)* |
| Δ_D \| H=1000 | −0.1053 | [−0.2116, −0.0242] | 0.63 | u |
| Δ_total | −0.1415 | [−0.2245, −0.0561] | 0.80 | **R** |

A third row-set in `T4_terms.csv` repeats the rerun using X25's own **replayed**
trajectory values (`x25_cell_table.csv`, `D*.bce.c1000.ndp`) instead of the
pipeline CSV column. It differs only in θ10 (−0.0429 vs −0.0427) and the terms
that contain it; this is the stored-vs-replayed CUDA discrepancy X25 measured
(`harness/X25_RESULTS.md:37`, max 8.24e-4). The replayed θ10 = −0.0429 and
θ11 = −0.1480 match `harness/X25_RESULTS.md:70,72` and
`results/4_mechanistic_case_study.csv:5,92` exactly.

### Notes on agreement with the frozen record

* **[확인됨]** Frozen reproduction is exact at 4 dp for every published quantity,
  including the CI endpoints of Δ_H and ΔH|D=1.
* **[확인됨]** My ΔH|D=1 on the rerun, −0.1768 [−0.2499, −0.1037], matches
  X25's own [−0.2488, −0.1027] to Monte-Carlo noise (different column order in
  the shared RNG stream; X25 uses seed 20260917, I used 20260916).
* **[확인됨]** The interaction Γ is the one term whose *resolved state changes*
  between run sets: unresolved on the frozen runs (s = 0.73, below the 0.75
  threshold, `harness/X24_RESULTS.md:96-101`) and **resolved** on the rerun
  (s = 0.77). The paper's "H1 horizon-dominant" label is therefore not stable
  across the two run sets — under the frozen rule applied to the rerun, Γ is
  resolved and |Γ| = 0.1405 ≥ |Δ_H| = 0.1065, which is the **H3
  interaction-dominant** branch (`analyze_x24_factorial.py:106`). This is
  reported, not fixed; the rerun was pre-registered as a *trajectory-selection*
  experiment (X25), not as a re-analysis of X24, so re-labelling X24 from it
  would itself be a protocol violation.

---

## (c) NIFTY's "published procedure" in `3c_protocol_native_published_procedure.csv`

### What the file says

`results/3c_protocol_native_published_procedure.csv:5`

```
NIFTY,GCN,german,default,targeted,200,1000,
  "horizon 200 -> 1000; selector: sigma_c -> the method's own published rule;
   training and validation-view drop rates from the official command",
  controlled_tau_I_negDP,-0.006451846881340765,...,
  native_tau_I_negDP,-0.15102324761229438,-0.22529634717881347,-0.08137234689544902,True
```

### Code/config evidence

| element | value | evidence |
|---|---|---|
| horizon | 200 → **1000** | `harness/provenance/nifty_README.md:48` (`--epochs 1000`), parsed by `harness/core/published_config.py:339` (`_nifty_cmd`) |
| drop rates | `drop_edge_rate_1 = drop_edge_rate_2 = 0.001`, `drop_feature_rate_1 = drop_feature_rate_2 = 0.1`, plus `restore_train_drop_rates=True` | `harness/provenance/nifty_README.md:48`; `harness/core/published_config.py:343-350`; realized values in `results/method_configurations.csv` row 89 |
| what the drop rates mean | training edge drop 0 → 0.001, training feature drop 0 → 0.1, **validation-view edge drop 0.1 → 0.001** | `harness/X24_NIFTY_GERMAN_FACTORIAL.md:35-47`; measured by the no-training probe, `harness/X24_RESULTS.md:50-52` |
| selector actually used for the reported −0.151 | **σ_c^BCE** (`selector == "common_bce"`) | `harness/results/armB_native_NIFTY_german.csv` contains only `common_bce` / `common_auc` (30 rows each); its `common_bce` `int_ndp` mean is `-0.15102324761229438` |
| NIFTY's *own* published rule | `argmin_epoch (val_c_loss + val_s_loss)` — classifier + similarity validation loss | `models/algorithms/NIFTY.py:525-532`; commented at `models/algorithms/NIFTY.py:513-515` ("this method's selector is argmin(val_c_loss + val_s_loss)") |
| where the own rule is recorded | `code_epoch` / `m1pub_auc,dp,eo` columns only, for **M1 only** | `harness/experiments/pilot_tau.py:140`, `:558` (`code_epoch=e1`), `:564-565` |

### [확인됨] The `factors_changed` string is boilerplate, and it is wrong about the selector

`harness/experiments/build_results.py:1043` hard-codes

```python
sel = "selector: sigma_c -> the method's own published rule"
```

and every branch of `native_factors()` (lines 1044-1072) concatenates that
constant unconditionally, for every method and dataset. It is never derived from
what the run actually did. Likewise `build_results.py:529-531` writes the
`selector` field of `results/method_configurations.csv` as the constant string
"the method's own published selection rule, replayed on the stored trajectory;
sigma_c^BCE / sigma_c^AUC also recorded".

The τ actually reported is σ_c^BCE:

* The four `armB_native_*` CSVs carry no published-rule τ column at all; there is
  no `m0pub_*`, so a published-rule **intervention contrast is not even
  computable** from the frozen artifacts.
* NIFTY's own rule and σ_c^BCE **disagree**: on the native cell,
  `m1_epoch` (σ_c^BCE) median 947 vs `code_epoch` (published rule) median 929,
  and they coincide in only **3 of 30** cells.
* Evaluating M1 at its published-rule epoch against M0 at its σ_c^BCE epoch (the
  only thing the stored columns permit) gives −0.118, not −0.151 — a different
  number, and not a valid paired contrast.
* X23, the freezing document, lists the NIFTY native change as
  "**H 200→1000, drop rates restored**" with no selector change
  (`harness/X23_NATIVE_VALIDATION_FROZEN.md:113`).

**Cause:** a hard-coded descriptive string in `build_results.py:1043` and
`:529-531`, not a computed field.
**Blast radius:** every row of `results/3b_protocol_native_horizon_selector.csv`
and `results/3c_protocol_native_published_procedure.csv` (the `factors_changed`
column) and every `protocol == native` row of
`results/method_configurations.csv` (the `selector` column) — 4 + 3 + ~7 rows.
The **numbers are unaffected**; only the prose describing them is wrong.
**Proposed fix:** derive the string from the selector column actually present in
the run CSV, i.e. "selector unchanged (σ_c^BCE primary, σ_c^AUC robustness); the
method's own rule is recorded per-arm as `code_epoch`/`m1pub_*` but is not used
for any reported contrast."

### Is D1.bce.frz the same run as the 3c value?

**[확인됨] Yes — the same run set, the same rows, the same selector; only the
bootstrap seed differs.**

| source | mean | CI |
|---|---|---|
| `results/4_mechanistic_case_study.csv:99` (`NIFTY,german,negDP,D1.bce.frz`) | `-0.1510232476122943` | [−0.2242361571819782, −0.0805760200588961] |
| `results/3c_protocol_native_published_procedure.csv:5` (`native_tau_I_negDP`) | `-0.15102324761229438` | [−0.22529634717881347, −0.08137234689544902] |
| my recomputation of `mean(int_ndp)` on `armB_native_NIFTY_german.csv`, `common_bce` | `-0.15102324761229438` | — |

The means agree to 16 significant digits; the two CIs differ by ~0.003 because
the X25 analyzer and the X23/native analyzer draw from differently-seeded
generators (`analyze_x25_trajectory.py:36` seed 20260917 vs the note at
`harness/X23_NATIVE_VALIDATION_FROZEN.md:32-38`, seed 20260915, cell-order
dependent). `D1.bce.frz` is literally the *frozen* reference column the X25
analyzer carries through (`analyze_x25_trajectory.py:140` `frz_full`).

### D1.auc.frz

**[확인됨]** `results/4_mechanistic_case_study.csv:132`:

```
selection-support decomposition,NIFTY,german,negDP,D1.auc.frz,
  -0.0199825815291569,-0.0917751700127871,0.0489703643633163,False
```

−0.0200 [−0.0918, +0.0490], **unresolved**. Consistent with
`harness/X24_RESULTS.md:43-45` / `:125` (P11 under σ_c^AUC, −0.0200
[−0.0921, +0.0503], s0.57, u) and `tableA3_nifty_factorial.tex:43`. The
expectation "−0.020, unresolved" is **confirmed**: the resolved harmful native
attribution under σ_c^BCE does not reproduce under σ_c^AUC, and the file records
this correctly. Its 3c row carries no σ_c^AUC counterpart, so a reader of
`3c_protocol_native_published_procedure.csv` alone sees only the σ_c^BCE value
−0.151 labelled as the "published procedure" result.

---

## (d) FairGB/Bail: −0.010 (3b) vs −0.019 (X26 τ_full under BCE)

**[확인됨] Different run sets. Same method, dataset, design, horizon, selector
and selection support; the numbers differ only by GPU nondeterminism between the
frozen native run and X26's dedicated native-length rerun.**

| quantity | value | source |
|---|---|---|
| 3b `native_tau_I_negDP` | **−0.010277096017912745** | `results/3b_protocol_native_horizon_selector.csv:2` |
| my `mean(int_ndp)` on `harness/results/armB_native_FairGB_bail.csv`, `common_bce`, n=30 | **−0.010277096017912752** | recomputation |
| X26 τ_full, BCE, −ΔDP | **−0.0189** [−0.0520, +0.0102] s0.60 u | `harness/X26_RESULTS.md:82`; also `:61`, `:68` |
| `bce.full` term of the mechanistic table | **−0.0189338514995949** | `results/4_mechanistic_case_study.csv:177` |
| my `mean(int_ndp)` on `harness/results/x26/x26_bail.csv`, `common_bce`, n=30 | **−0.01893385149959493** | recomputation |

Ruled out, one by one:

* **Not a different selector.** Both are `selector == "common_bce"` (σ_c^BCE).
  3b's `factors_changed` claims "selector: sigma_c -> the method's own published
  rule", but that is the same hard-coded boilerplate string from
  `harness/experiments/build_results.py:1043` diagnosed in (c); the run CSV has
  only `common_bce`/`common_auc`.
* **Not a different support.** Both run at native `method_epochs == 1500` and
  select over the full trajectory `{0..1499}`
  (`harness/X26_RESULTS.md:45,72`); the frozen native file's `m1_epoch` spans
  502–1464.
* **Not a different unit set.** Both are splits 20–25 × runs 0–4; the
  `(split_id, run_id)` sets are identical (verified).
* **It is a different run set.** X26 retrained the whole cell to record
  trajectories (`harness/X26_RESULTS.md:40-47`, "11:53 → 13:24, rc 0, 30/30
  cells", output `harness/results/x26/x26_bail.csv`, SHA `896dd0e1…`), because
  per-epoch test outcomes were not recoverable from the frozen run. X26 itself
  frames the pair as a reproduction gate and reports it as a **pass**:
  `harness/X26_RESULTS.md:61` — "rerun full −0.0189 · frozen native interval
  [−0.0354, +0.0137] (−0.0103)". The rerun value sits inside the frozen 95 %
  interval, and both are unresolved, so no conclusion changes.

**Residual integrity issue [확인됨].** The two numbers are nevertheless printed
in the paper as if they were the same cell: `results/3b_…csv` carries −0.010 as
"the" native FairGB/Bail attribution while
`results/4_mechanistic_case_study.csv:177` and
`harness/results/tables/paper/table3_mechanistic_combined.tex:11` build the
mechanistic decomposition (τ_H200 −0.0597 → τ_full −0.0189, S = +0.047 R) from
the −0.019 rerun. Neither artifact states that the two come from different
executions; the −0.047 gap between 3b's controlled −0.0597 and native −0.0103
and X26's +0.0407 `H_shift` differ by exactly the same 0.0086 nondeterminism.
**Cause:** no run-set provenance column in `3b`/`4`.
**Blast radius:** presentational — any reader combining Fig. 3 and Fig. 4 for
FairGB/Bail.
**Proposed fix:** add a `source_run` column naming
`armB_native_FairGB_bail.csv` vs `x26/x26_bail.csv`, and state the reproduction
gate result in the caption.

---

## Summary of findings

| # | finding | label |
|---|---|---|
| a | −0.117 is Δ_H, the D-averaged horizon main effect (`X24_NIFTY_GERMAN_FACTORIAL.md:129`, `X24_RESULTS.md:78`). ΔH\|D=1 = −0.180 frozen / −0.177 rerun. A paper label "ΔH\|D=1 ≈ −0.117" does not match the number. | 확인됨 |
| a | The mislabel cannot be located in any repo file; the manuscript prose is not in the repository. | 확인 불가 |
| b | Frozen factorial reproduced exactly (12 terms, seed 20260916, 10,000 reps). | 확인됨 |
| b | No H = 200 rerun exists; the rerun row-set necessarily reuses the frozen θ00/θ01. | 확인 불가 (for those two cells) |
| b | Γ (interaction) is unresolved on the frozen runs (s 0.73) but **resolved** on the rerun (s 0.77, \|Γ\| ≥ \|Δ_H\|) — the H1 label is not run-set-stable. | 확인됨 |
| c | NIFTY's published procedure = H 1000, drop rates 0.001/0.001/0.1/0.1 with `restore_train_drop_rates`, but the **selector was NOT switched** to NIFTY's own rule; the reported −0.151 is σ_c^BCE. The "own published rule" text is a hard-coded constant at `build_results.py:1043`. | 확인됨 |
| c | `D1.bce.frz` (−0.1510232476122943) and the 3c value (−0.15102324761229438) are the **same run**, same rows, same selector; CIs differ only by bootstrap seed. | 확인됨 |
| c | `D1.auc.frz` = −0.0200 [−0.0918, +0.0490], unresolved — as expected; correctly recorded. | 확인됨 |
| d | 3b −0.010 (frozen `armB_native_FairGB_bail.csv`) vs X26 −0.019 (`x26/x26_bail.csv` rerun): same selector, support and units; **different run set**. X26's own reproduction gate passes. Neither published artifact discloses the run-set difference. | 확인됨 |
