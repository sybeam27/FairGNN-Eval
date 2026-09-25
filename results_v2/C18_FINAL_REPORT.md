# Phase 0 remediation — final report

Written 2026-09-25, after Step A (verification), Step B (GPU runs) and Step C (rebuild).
Everything below is computed from `results_v2/`; the frozen bundle under `results/` was never
modified. Labels: **[확인됨]** verified from an artifact or a command's output, **[추정]** inferred,
**[확인 불가]** not determinable from what exists.

---

## 1. What changed in the headline numbers

The intervention contrast **τ_{−I→+I} did not move at all**. It does not reference B, and the
rebuild only replaced B, so this is structural — but it was checked rather than assumed: all 90
cells reproduce every mean, bound and resolved flag at **max |diff| 0.0**.

Everything referenced to B did move. On the 36 primary cells:

| quantity | coordinate | frozen | **B_rep1 (reported)** |
|---|---|---|---|
| package larger | ΔAUC | 31/36 | **27/36** |
| | −Δ_DP | 26/36 | 26/36 |
| | −Δ_EO | 29/36 | 29/36 |
| τ_pkg < 0 | ΔAUC | 9/36 | **18/36** |
| | −Δ_DP | 23/36 | **19/36** |
| | −Δ_EO | 24/36 | **20/36** |
| opposite sign | ΔAUC | 19/36 | **16/36** |
| | −Δ_DP | 18/36 | 18/36 |
| | −Δ_EO | 15/36 | 15/36 |
| median abs τ_{B→−I} | ΔAUC | 0.104 | **0.037** |
| | −Δ_DP | 0.039 | 0.040 |
| | −Δ_EO | 0.028 | 0.028 |
| resolved | all three | 11 / 8 / 9 | 11 / 8 / 9 |

**The cause is German alone.** It is the only dataset whose published horizon the runner resolved
and then dropped (`pilot_tau.py:389 → :463`; `published("GNN","german")` returns 1000, every other
dataset returns `None`). Restoring it takes B's test AUC on German from **0.4415 — below chance —
to 0.6543**, and the median selected epoch is 991, so the horizon was genuinely binding. The
fairness coordinates barely move; ΔAUC moves a lot, because a below-chance baseline made the
surrounding package look large on accuracy.

**Aggregate robustness** (`tab:aggregate_robustness`), under the definitions verified in §6:

| coordinate | | cell-weighted | method-balanced | LOMO |
|---|---|---|---|---|
| ΔAUC | B_rep1 | 75.0% | 78.6% | 72.7–78.8% |
| | B_H1000 | 55.6% | 62.1% | 51.5–60.6% |
| −Δ_DP | B_rep1 | 72.2% | 79.0% | 69.7–79.3% |
| | B_H1000 | 52.8% | 52.7% | 48.5–58.1% |
| −Δ_EO | B_rep1 | 80.6% | 85.2% | 78.8–86.2% |
| | B_H1000 | 55.6% | 55.7% | 51.5–61.3% |

**Baseline sensitivity.** Nondeterminism changes nothing; baseline strength changes a great deal.

| | frozen | B_rep1 | B_rep2 | B_H1000 |
|---|---|---|---|---|
| package larger ΔAUC | 31 | 27 | **27** | 20 |
| package larger −Δ_DP | 26 | 26 | **26** | 19 |
| package larger −Δ_EO | 29 | 29 | **29** | 20 |
| τ_pkg < 0 | 9 / 23 / 24 | 18 / 19 / 20 | 18 / 19 / 20 | 30 / 13 / 16 |
| opposite sign | 19 / 18 / 15 | 16 / 18 / 15 | 16 / 18 / 15 | 14 / 10 / 11 |

B_rep1 and B_rep2 are two independent re-trainings under the same rule. **Not one cell's verdict
differs between them, on any coordinate.** B_rep1 vs B_H1000 differs on 7, 9 and 11 cells, almost
all in one direction — training the baseline to 1000 epochs shrinks |τ_{B→−I}| and overturns
"package larger". The one cell that moves the other way on both fairness coordinates is
**FairGNN/credit**.

---

## 2. Results that need care

### 2.1 The estimator covers re-execution noise — checked, and it does [확인됨]

Five primary cells were re-executed twice with their own frozen commands (only `--out` redirected),
giving three realizations and three pairwise differences each, read with the paper's own paired
hierarchical bootstrap.

**0 of 45 intervals exclude zero.** The two SFG pairs reclassified as re-executions add **0 of 6**.
So a difference produced by re-running an identical command is not something these intervals call
real. This was the one result that could have forced the uncertainty story to be rewritten, and it
came out the reassuring way.

**But re-execution noise is strongly cell-dependent, and a single "noise floor" number misleads.**
On −Δ_DP the cell-level mean τ_I moved 0.0668 between realizations of SFG/German and 0.0013 on
NIFTY/German — a factor of 50. Per unit the extremes are 0.635 (SFG/German) against 0.004
(FairVGNN/German). Between rep 1 and rep 2, all 30 units changed on SFG, FairGB and FairSIN-GCN;
only 1–2 changed on FairVGNN/German.

**Cross-check.** SFG/German appears both as a noise-floor cell and as a reclassified pair. Its
frozen controlled-vs-published difference is 0.0457, which sits inside the range two deliberate
re-executions produce (0.018–0.067). That is independent support for Decision 2.

### 2.2 C-9: three disagreements, all in the same direction [확인됨]

Of 19 native and 4 procedure pairs, the primary criterion (interval excludes zero) and the
cell-matched comparison (|Δ_attr| exceeds that cell's own re-execution maximum) disagree on three:

| pair | coordinate | \|Δ_attr\| | interval excludes 0 | exceeds own-cell max |
|---|---|---|---|---|
| FairSIN/credit/SAGE | −Δ_DP | 0.0240 | no | yes (0.0187) |
| FairSIN/credit/SAGE | −Δ_EO | 0.0182 | no | yes (0.0158) |
| SFG/german | ΔAUC | 0.0032 | no | yes (0.0012) |

**None go the other way.** No pair the paper calls a protocol effect fails its own noise check, so
the text may keep using the primary criterion alone. Counts: interval excludes zero on 3/19, 3/19,
2/19 of the native pairs and 2/4 on each coordinate for procedure.

### 2.3 C-12: two cells excluded as numerically inert [확인됨]

The rule was fixed before the noise floor was measured. Per coordinate the threshold is the
smallest unit-level max |Δ| between two re-executions of one cell, which resolved to
FairVGNN/German on all three: **0.000229 / 0.004364 / 0.018908**. A cell is excluded only when all
three of its coordinates fall at or below it.

Excluded: **FairEdit/credit** (4e-6, 0.0, 0.0) and **FairEdit/bail**. FairGNN/bail passes two of
three (ΔAUC 7.7e-4 > 2.3e-4) and is kept, by the rule as written.

| | ΔAUC | −Δ_DP | −Δ_EO |
|---|---|---|---|
| all 36 | 27 | 26 | 29 |
| inert removed (34) | 25 | 24 | 27 |

**Limitation, to print beside the number:** the noise floor was measured on SFG/German,
NIFTY/German, FairGB/Bail, FairSIN-GCN/Credit and FairVGNN/German. **Neither FairEdit nor FairGNN is
among them**, so the threshold is transferred from other methods rather than measured on the cells
it excludes.

### 2.4 C-14: the common-epoch comparison cannot reach the primary cells [확인 불가]

`ValidationHistory` (`harness/core/trajectory.py:202`) states it "deliberately holds no test data":
the test slice is read only after selection. So the primary runs left no per-epoch test output and
no checkpoint weights on disk.

| stored per cell | primary 36 | x25 NIFTY/German | x26 FairGB/Bail | x27 FMP |
|---|---|---|---|---|
| per-epoch validation curve | in memory only | ✓ 1001 | ✓ 1500 | ✓ 300 |
| per-epoch test output | ✗ | ✓ all epochs | **11-epoch grid only** | ✓ all epochs |
| checkpoint weights | 2 slots, in memory | ✗ | ✗ | ✗ |
| written to disk | selected epochs' AUC/DP/EO only | | | |

**For all 36 primary cells the answer is [확인 불가] without a re-run.** What could be computed:

| cell | ΔAUC | −Δ_DP | −Δ_EO |
|---|---|---|---|
| NIFTY/German D0 | +0.0146 → +0.0020 | −0.0429 → −0.0422 | −0.0263 → −0.0281 |
| NIFTY/German D1 | +0.0488★ → +0.0226 | −0.1480★ → −0.1092★ | −0.1223★ → −0.0912★ |

(independent selection → common epoch; ★ = resolved). **0 sign flips of 9**; magnitudes shrink
20–46%; one resolved status changes, D1 on ΔAUC. The recomputed independent value reproduces the
frozen τ_full exactly on −Δ_DP (diff 0.00000), which validates the method.

**FairGB/Bail is reported separately and is not interpreted.** Its test scores exist only at 11 grid
epochs while the selector chose 500–1437 (M⁺ᴵ) and 35–267 (M⁻ᴵ), so the independent reference must
be read at the nearest grid epoch and lands 0.0157 from the frozen value. Its three resolved-status
changes are confounded with that approximation.

**Cost of answering it properly** (not run; measured on 23 of 36 cells, the rest imputed at the
per-method median):

| scope | wall-clock, single GPU | storage |
|---|---|---|
| all 36 primary cells | **≈ 93 h (3.9 days)** | ≈ 7 GB |
| 11 cells, one per method | **≈ 14 h** | ≈ 2 GB |

The bottleneck is GPU time, not disk. The 11-cell subset would be BIND/bail, BeMap/bail,
EDITS/german, FairEdit/german, FairGB/credit, FairGNN/bail, FairSIN/german, FairVGNN/credit,
GEAR/bail, NIFTY/bail, SFG/german.

### 2.5 Two definitions that had to be recovered, and differ [확인됨]

Neither existed as code anywhere in the repository; both were reconstructed and then verified
against published values.

* **opposite sign** = `sign(τ_pkg) × sign(τ_I) < 0`, so an estimate of exactly zero counts as
  neither sign. The alternative (a disagreement of booleans) adds one cell on −Δ_DP and one on
  −Δ_EO at every baseline, because FairEdit's intervention is numerically inert. The confirmed
  definition reproduces the Table 1 template exactly (19, 18, 15 against 5+8+6, 6+6+6, 5+6+4).
  **The section 4.2.1 values 16 / 18 / 15 are correct.**
* **method-balanced / LOMO** = the per-method share averaged over the 11 methods, and the
  cell-weighted share with each method dropped in turn. These reproduce all nine published values
  of `tab:aggregate_robustness` exactly. Now `build_results.aggregate_robustness`.

### 2.6 Things reported but not fixed

* **Seeding is not uniform, and `pilot_tau` does not seed per split.** `pilot_tau.py:503`/`:506`
  call `manual_seed(seed*1000+split)` before each arm, but `train()` at `:109` reseeds with `seed`
  alone as its first statement, so for FairGNN, NIFTY, FairVGNN and FairGB the effective seeding is
  **split-independent**. The x30/x31 arms do use `seed_all(seed*1000+split)`, and EDITS uses
  `+split+1`. The pairing the comment claims still holds — both arms start from the same state.
  Left as it stands because the frozen results depend on it; now recorded in the `rng_rule` column.
* **Compute figures.** The published "44.4 / 81 GPU-hours" are not measurements: 12 of the 36
  primary cells have no timing artifact at all, and up to ten processes shared one GPU, so a
  per-cell split is not recoverable. **≈ 52 GPU-hours** is the union of every interval the logs
  date-stamp and is what the paper should state, as a floor.
* **Appendix K is unaffected [확인됨].** FMP's baseline is built by its own runner at H = 300 on
  pokec_z and pokec_n, where `published()` returns `horizon = None`, so the ignored-horizon defect
  cannot reach it; and it is already one draw per unit, so T2 does not either. Worth stating in the
  paper: FMP's B is **not** the primary cells' B (AUC 0.7166 vs 0.7038 on pokec_z, 0.7531 vs 0.7112
  on pokec_n), so τ_{B→base} there and τ_{B→−I} in Table 1 are not comparable.
* **The public repository cannot execute any of the 36 cells.** `.gitignore:49` excludes every
  runner, `:9` all of `models/`, `:48` `harness/provenance/` (which `published_config` reads at run
  time), `:6` `data/`. Six work-tree fixes exist that no commit carries; see
  `results/phase0_audit/EXPORT_CHECKLIST.md`.

---

## 3. Figures

Every figure is now generated at the size it is placed at, so `width=\textwidth` scales nothing.
Type sizes were **not** reduced when the heights were cut.

| figure | paper | **final (in)** | target | min glyph | printed |
|---|---|---|---|---|---|
| `fig1_intervention_attribution` | Fig. 1 | **5.48 × 2.35** | 2.35 ✓ | 4.9 pt | **4.90 pt** |
| `fig3_protocol_variation` | Fig. 2 | **5.49 × 2.18** | 2.05 (+0.13) | 5.6 pt | **5.60 pt** |
| `fig3_selection_support_trajectory` | Fig. 3 | **5.35 × 2.20** | 2.20 ✓ | 4.55 pt | **4.55 pt** |
| `figS1_intervention_attribution_negEO` | App. | **5.48 × 2.35** | = Fig. 1 ✓ | 4.9 pt | **4.90 pt** |

**Fig. 1 and Fig. 3 meet their targets; Fig. 2 is 0.13 in over, by request.** Every glyph below 7 pt is a
single-level sub- or superscript such as `$M^{+I}$`; the 3.92 pt double nesting is gone, because
the τ label is now `τ_{−I→+I} (−Δ_DP)` rather than a subscript inside a superscript.

**Fig. 3 now sits at 4.55 pt, 0.05 pt above the 4.5 floor.** Its per-unit median labels and arm
labels were taken from 7 to 6.5 pt on request, and 6.5 × 0.7 = 4.55 is the superscript of `$M^{+I}$`.
Nothing else in the set is affected, but this figure has no margin left: any further reduction of
those two labels crosses the floor.

Height saved in the body: 0.38 + 0.25 + 0.24 = **≈ 0.87 in ≈ 8 lines**. Fig. 2 sits 0.13 in
above its 2.05 target because panels (a) and (b) are square identity plots: their size is set by
the row height, not the column width, so widening them on request could only be done by giving
the axes row more of the figure. The appendix EO figure does
not count toward the body.

*Why this mattered:* before this pass, `fig1` was generated at 7.41 in and shrunk by 0.74 on
inclusion, putting its smallest type on the page at **2.91 pt**; `fig3_protocol_variation` printed
at 3.08 pt.

Other figure changes: Fig. 2 panel (b) is "Published horizon", the axis reads "Published", the two
SFG re-execution pairs are open squares excluded from the systematic set (18 → 16), and
`figS4_fixed_epoch_trajectory` and `figS_selection_support_bars` got the title-only edits
("D1 (published)", "published drop rates"). `fig2_sign_resolution` was excluded by request and is
unchanged at 3.57 in.

---

## 4. Tables

Table generation was moved out of `results/plot.ipynb` into the tracked
`harness/experiments/build_tables.py`. **The Overleaf file is the template**: caption, footnotes,
column spec and row order pass through untouched and only numbers are substituted.

**Verification:** filling each template from the *frozen* bundle reproduces it exactly —
`table1_summary`, `table1_cells_dAUC`, `table1_cells_negDP`, `table1_cells_negEO`. Getting there
required recovering the two definitions in §2.5 and one formatting rule (a value rounding to zero
prints `+0.000`, never `−0.000`).

Regenerating from the notebook would have destroyed the hand edits: `table1_summary` is a different
shape there, and the per-cell footnote still says "sign stability" where the manuscript says
"unit-level sign consistency" — the manuscript's wording is the correct one.

New: `tableS_noise_floor.tex` (`tab:noise_floor`, two labelled blocks, caption clause computed not
typed) and `tableS_baseline_spec.tex` (`tab:baseline_spec`).

Still to fill against their templates: `tableS1_1_fnrgnn_two_tasks` (τ_B column changes),
`tableS2_configuration_pairs` (Δ_attr marker), `tableS3b_native_horizon_selector`
(reclassification + caption).

---

## 5. Artifacts

| path | contents |
|---|---|
| `results_v2/bundle/` | the rebuilt bundle against B_rep1, including `per_unit_metrics.csv.gz` with the new `seed` and `rng_rule` columns |
| `results_v2/bundle_B_rep2/`, `bundle_B_H1000/` | the two sensitivity bundles |
| `results_v2/baselines/B_rep1|B_rep2|B_H1000/` | the rebuilt baselines, 8 datasets × 30 units each |
| `results_v2/tables/` | regenerated tables and the two new ones |
| `results_v2/phase0_verify_v2.txt` | the full verification output |
| `results_v2/paper_numbers_opposite_sign.csv` | the opposite-sign counts pinned to the confirmed definition |
| `results/phase0_audit/noise_floor_delta.csv` | 45 rows: 5 cells × 3 pairings × 3 coordinates |
| `results/phase0_audit/c8_reexecution_delta.csv` | the two reclassified SFG pairs |
| `results/phase0_audit/c9_native_procedure.csv` | 19 native + 4 procedure + 2 re-execution, three criteria |
| `results/phase0_audit/c11_subsets.csv`, `c12_inert_excluded.csv` | subset and inert-cell aggregates |
| `results/phase0_audit/c14_common_epoch.csv`, `_units.csv` | the common-epoch comparison |
| `results/phase0_audit/c2_baseline_sensitivity*.csv` | the four-baseline blocks and per-dataset B |
| `results/phase0_audit/provenance_read_check.csv` | 90 cells resolved with and without provenance |
| `results/phase0_audit/STEP_C_PLAN.md` | the plan, with every rule fixed before its result was seen |
| `results/phase0_audit/EXPORT_CHECKLIST.md` | what the public repository is missing |
| `paper/table_template/` | the Overleaf templates the regenerated tables must match |

**Verification that passed:** τ_I identical across all 90 cells (0.0); the identity
τ_pkg = τ_nonint + τ_I at 1.1e-16; point estimates reproduce from the per-unit export at 9.8e-17;
B's AUC spread across methods is now exactly 0.0 on every dataset, which is the T2 per-method
re-training gone; frozen data reproduces four table templates exactly; the 9 published values of
`tab:aggregate_robustness` reproduce exactly.

---

## 6. Open

* Three table templates still to fill (§4).
* `results_v2/paper_numbers.csv` — the consolidated version; only the opposite-sign slice exists.
* `results_v2/B_rebuild_diff.md` — per `B_rebuild_decision.md` "Deliverables".
* Whether to re-run for C-14 on the primary cells (§2.4) — a separate decision, 14 h or 93 h.
* The public-repository gap (§2.6, last bullet) — a fetch-script and licensing plan exists in
  `EXPORT_CHECKLIST.md` but is not executed.
