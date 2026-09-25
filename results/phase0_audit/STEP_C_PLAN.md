# Step C — the rebuild, run once after every Step B run has finished

Written before any of it runs. Each item names its input and what it must assert, so the
build either produces the item or stops. Nothing under `results/` or `harness/results/` is
written; everything lands in `results_v2/`.

## Ordering

Step B must be complete first: `B_rep1` (done, 8 datasets x 30 units, 961 s) -> noise floor
rep1 and rep2 (5 cells each) -> `B_H1000` -> `B_rep2`. C-1 is the only item that rebuilds the
bundle; everything else reads what C-1 wrote.

## Items

**C-1. `results_v2/` via `build_results.py`**, against `B_rep1`.
* asserts `tau_{-I->+I}` is bit-identical to the frozen value for every cell, and stops otherwise
  (the rebuild only changes B, so this is structural, not a hope)
* reclassifies SFG/german and SFG/credit out of 3b: family becomes `re_execution`, and the
  systematic native set becomes **16 pairs**, not 18 (Decision 2, from A-1: the two rows are a
  pure re-run, the `--native` flag reaches only `m_epochs` and both datasets have
  `native_horizon = 200`)
* per-unit export gains two columns: `seed` (= 27 + run_id, verified exact on all 3,900 rows)
  and `rng_rule`, the seeding actually used on that row's path --
  `manual_seed(seed)` for B and for the pilot_tau method arms,
  `seed_all(seed*1000+split)` for the x30/x31 arms,
  `seed_all(seed*1000+split+1)` for EDITS (`x30_edits.py:131`).
  B rows carry `manual_seed(seed)` (`pilot_tau.py:458`, `rebuild_baseline.py:99`).
* `backbone` and `configuration` are already separate columns, split out of the stored method name
  by `build_manifests.per_unit_metrics()`; keep that and do not re-derive it.

**C-2. Sensitivity CSVs** for `B_rep2` (baseline nondeterminism) and `B_H1000` (baseline
strength), on the headline blocks fixed in `B_rebuild_decision.md` rule 6.

**C-3. Tables** — Table 1, 20–22, 2·3, 24–27, 11, FnRGNN, plus a new B-specification table.

**C-4. Figures** — regenerate Figs 1, 3, 4 and the S1 series.

*Print-size specification.* Every figure is generated at the size it is placed at, so LaTeX
scales nothing. The paper's Fig. 1 = `fig1_intervention_attribution`, Fig. 2 =
`fig3_protocol_variation`, appendix Fig. 4 = the `figS1_*_negEO` series.

* **Width exactly 5.5 in** (\textwidth), current aspect ratio kept, included as
  `width=\textwidth`. The bbox that `bbox_inches="tight"` produces is what counts, not `figsize`,
  so tune `figsize` until the saved PDF measures 5.5 in.
* **Print-size type:** panel titles 8 pt, axis labels 7.5 pt, ticks / legend / inset labels >= 7 pt.
  A legend that no longer fits on one line at that size may wrap to two.
* **Unnested tau notation**, as the new Fig. 3 uses: `tau_{-I->+I} (-Delta_DP)`, never
  `tau^{-Delta_DP}_{-I->+I}`. mathtext shrinks 70% per nesting level, so a subscript inside a
  superscript prints "DP" at 3.9 pt.
* **U+2212** for every minus in a value label, matching the axis ticks.
* Fig. 2 panel (b) title becomes "Published horizon", and the two SFG re-execution pairs are
  handled per Decision 2 -- dropped from 3b and shown separately, not counted in the systematic 16.
* After generating, measure the smallest glyph in each PDF by decompressing its streams and
  reading the `Tf` operands. Report it.

*Why this matters -- measured on the current files (2026-09-24):*

| figure | width | scale at \textwidth | smallest glyph | **as printed** |
|---|---|---|---|---|
| `fig1_intervention_attribution` | 7.41" | 0.742 | 3.92 pt | **2.91 pt** |
| `fig3_protocol_variation` | 6.99" | 0.787 | 3.92 pt | **3.08 pt** |
| `figS1_intervention_attribution_negEO` | 7.41" | 0.742 | 3.92 pt | **2.91 pt** |
| `figS1_tradeoff_direction_negEO` | 5.92" | 0.930 | 5.04 pt | 4.69 pt |
| `figS1_arms_by_family_dataset_negEO` | 6.81" | 0.808 | 5.04 pt | 4.07 pt |
| `fig3_selection_support_trajectory` (new) | 5.35" | 1.029 | 4.90 pt | 5.04 pt |

Every figure but the new one is generated wider than the text block and silently shrunk on
inclusion, so its smallest type prints at 2.9-4.7 pt. Fixing the width is therefore not cosmetic.

**`fig2_sign_resolution` is excluded** (user, 2026-09-24). It is 3.57 in wide -- the one figure
that would be scaled *up* at \textwidth -- and it keeps its current size. It is not regenerated,
so it does not get the 5.5 in width, the print-size type or the unnested tau notation; if it is
later placed at \textwidth its line weights and markers will grow by 1.54x and its tone will no
longer match the rest of the set.

**C-5. `phase0_verify` on `results_v2/`**, full output kept.

**C-6. `results_v2/paper_numbers.csv`.**

**C-7. `results_v2/B_rebuild_diff.md`** — per `B_rebuild_decision.md` "Deliverables".

**C-8. The two `re_execution` pairs.** Delta_attr and its 95% interval per coordinate for
SFG/german and SFG/credit, by the same estimator as A-3.

State per coordinate, explicitly, whether the interval excludes zero. **If any of them does, report
that first, before the numbers themselves.** These two pairs differ by nothing but the process they
ran in -- A-1 established the configuration diff is empty and the `--native` flag is inert at
`native_horizon = 200` -- so a Delta_attr the interval calls non-zero is an interval that does not
cover re-execution noise. That would mean the paired hierarchical bootstrap, which resamples splits
and then runs, understates the variability actually present, and every interval in the paper built
with it inherits the problem. It would not be a finding about SFG; it would be a finding about the
estimator, and it is the one result here that could force the uncertainty story to be rewritten.
Cross-check it against the C-9 / noise-floor measurement of the same quantity on the same estimator.

*Report it in the C-16 format* -- Delta and 95% interval per coordinate, one row per pair.

*Can these two rows join the C-16 appendix table?* **Structurally yes, in a separate labelled
block, not interleaved.** The quantity is identical: a paired Delta of tau_I between two
realizations of one cell, same estimator, same seed, same 30 matched units. But the provenance
differs and the table must not hide it -- the C-16 rows are two re-executions run now, on one
machine, on a single code state, whereas the SFG rows are two runs from the original study
recorded weeks apart under the `controlled` and `native` protocol labels. Merging them without a
column distinguishing "re-executed for this audit" from "frozen pair reclassified as a re-run"
would present a claim about reproducibility as though the two were collected the same way.

**SFG/german appears on both sides**, as a noise-floor cell and as a re-execution pair, and that
overlap is the useful part: it gives a direct check of whether the frozen controlled-vs-native
difference sits inside the range two deliberate re-executions produce. Report that comparison
explicitly. SFG/credit has no noise-floor counterpart, so it only gets the cross-cell maxima.

**C-9. Native and procedure pairs against the measured re-execution noise.**

**The criteria, fixed here before the recount is run:**

> **Primary criterion.** Whether the pair's Delta_attr 95% interval excludes zero, on the
> reclassified sets: **19 native pairs** (21 minus the two SFG re-execution pairs) and
> **4 procedure pairs**. This is the criterion the paper reports.
>
> **Secondary criterion, conservative.** Whether `|Delta_attr|` exceeds the largest cell-level
> mean `|Delta|` measured between two realizations of one cell in the Step B noise floor:
> **dAUC 0.01174, negDP 0.06677, negEO 0.04599** (column `exceeds_rerun_max`). A pair that does
> not clear this did not move more than re-running the identical command moved a cell.
>
> **Cell-matched comparison, where it is available.** The noise floor was measured on
> SFG/german, FairGB/bail, FairVGNN/german and NIFTY/german (and FairSIN-GCN/credit). Where one of
> these cells appears in a native or procedure comparison, add a column comparing that pair's
> `|Delta_attr|` against **that same cell's** re-execution `|Delta|` maximum, rather than against
> the cross-cell maximum. Name the cell and the value used. This is the strongest form of the
> check, because it removes the transfer between methods and datasets.

**Disagreements lead the report.** Before any counts, list every pair where the **primary
criterion and the cell-matched comparison disagree** -- an interval excluding zero on a pair whose
|Delta_attr| does not clear its own cell's re-execution maximum, or the reverse. The paper's text
uses the primary criterion alone, so a short list means the text stands as written and a long one
means it has to be qualified. Name each pair, both verdicts, |Delta_attr|, and the cell-matched
threshold used.

Report all three side by side; do not collapse them into one verdict. Re-execution variability is
strongly cell-dependent -- SFG/german moved 0.0668 on negDP between realizations while
NIFTY/german moved 0.0013 -- so the cross-cell maximum is conservative for some cells and
generous for others, and the cell-matched column is the one to trust where it exists.

**C-10. Systematic-16 tables.** Regenerate Tables 3, 7, 13 and 26 on the 16-pair systematic set,
and emit a separate re-execution table for the two withdrawn pairs, both under
`results_v2/tables/`.

**C-11. G.2 single-operation subset, without FairEdit.** The subset is BIND (2), BeMap (3),
FairEdit (3), GEAR (1), NIFTY (3) = 12 cells; recompute the "surrounding package is larger"
share per coordinate on the **9** cells that remain once FairEdit's three are removed.

*Done ahead of the rebuild, because it required a code change.* "Package larger" was computed
in two places: `build_results.py` wrote the flag for negDP and negEO before the CSV round-trip,
while dAUC was recomputed after reading the CSV by `dominance()` in `results/plot.ipynb` (cell
5), which recorded which path it took in `table1_summary.csv:nonint_larger_source`. Both
already applied the same rule to the same inputs, and agreed on the frozen bundle (negDP 26/26,
negEO 29/29, zero disagreements). They are now a single function, `build_results.nonint_larger`:

    |tau_{B->-I}| > |tau_{-I->+I}|   -- strict, on the unrounded cell means, all three coordinates

`nonint_larger_dAUC` is now emitted alongside the other two. Verified against the frozen bundle
(`--out` to a scratch directory; nothing under `results/` written): **dAUC 31, negDP 26,
negEO 29**, and every one of the 63 shared columns identical except the three text columns
already changed by approved commits (the FairEdit per-dataset caveat, `e1c000d`).

The 9- and 12-cell subsets are reported as count/denominator, never as a bare percentage: one
cell moves a 9-cell share by 11 points. The method-balanced and LOMO aggregates keep percentages.

**C-12. The 36-cell aggregate with the inert cells removed.** Same headline counts, excluding
cells whose two arms are effectively identical.

**The rule, fixed here before any noise-floor result is seen:**

> **Per-coordinate threshold** = among the five noise-floor cells, the *smallest* value of the
> unit-level maximum |Delta| between two re-executions of the same arm. A cell is classified as
> excluded when, **on all three coordinates**, its
> `max_unit |Y(M^{+I}) - Y(M^{-I})|` is less than or equal to that coordinate's threshold.

Comparison is unit-level on both sides: the threshold comes from per-unit differences between
realizations, and the cell statistic is the per-unit arm difference already in
`T8_activation.csv` (`max_abs_dauc`, `max_abs_ddp`, `max_abs_deo`). Taking the *smallest* of the
five cells makes the threshold conservative -- it excludes a cell only if it moves less than the
quietest cell we measured re-execution noise on.

On the frozen numbers the candidates, in order, are FairEdit/credit (4.0e-6, 0.0, 0.0 — the only
cell with identical units, 5/30), FairEdit/bail (1.2e-4 / 8.7e-4 / 1.1e-3) and FairGNN/bail
(7.7e-4 / 1.7e-3 / 3.1e-3). Report the aggregate **both with and without** the excluded cells.

**Stated limitation, to appear beside the result:** the five noise-floor cells are SFG/german,
NIFTY/german, FairGB/bail, FairSIN-GCN/credit and FairVGNN/german. **Neither FairEdit nor
FairGNN is among them**, so the threshold is transferred from other methods and datasets rather
than measured on the cells it excludes. It bounds re-execution noise for this harness on this
machine, not for those two methods specifically.

**C-14. Common-epoch comparison (the old T10).** Today the two arms of a unit are checkpointed
independently, each at its own validation-BCE minimum. This asks what changes if they are forced
to share one epoch.

**The rule, fixed here before any result is seen:**

> For a unit, the **common epoch** is the epoch minimising the *mean of the two arms'* validation
> BCE: `argmin_e [ (val_bce_{M+I}(e) + val_bce_{M-I}(e)) / 2 ]`. Both arms are then read at
> that one epoch and tau_I is formed from the two test outcomes, on all three coordinates. Ties are
> broken by the smallest epoch. Where a cell's test outcome is stored only on a grid of epochs
> rather than at every epoch, the argmin is taken **over that grid**, and the cell is reported as
> grid-restricted with its grid size.

Report against the current independent-selection tau_I: the number of cells whose sign flips, the
number whose resolved verdict changes, and the median and maximum |difference|, per coordinate.
Cells with no stored trajectory are listed, not estimated.

*Feasibility, established in T10 and re-checked 2026-09-24 — read this before scheduling the work.*
**No primary cell has a stored trajectory.** The controlled H = 200 runs that produce the 36 primary
cells recorded no per-epoch validation loss, so for all 36 the answer is **[확인 불가]** without a
re-run. What exists:

| trajectory set | cell | protocol | per-epoch test outcome |
|---|---|---|---|
| `x25/R10_trajectories`, `R11_trajectories` (60 files each) | NIFTY / German | native, H = 1000 | yes -- `test_raw` at all 1001 epochs |
| `x26/bail_trajectories` (60 files) | FairGB / Bail | native, H = 1500 | **grid only** -- `grid_scores` at 11 epochs |
| `x27/pokec_{z,n}_trajectories` (210 files each) | FMP | component study | to be checked |

So C-14 runs on the two case-study cells (grid-restricted for FairGB/Bail) plus FMP if its files
carry the same fields, and reports the 36 primary cells as unanswerable from stored artifacts. If
the comparison is wanted for the primary cells, that is a re-run with trajectory recording enabled
and has to be decided separately -- it is not covered by "no new training".

**C-16. Appendix table of re-execution noise.** `results_v2/tables/tableS_noise_floor.tex`, label
`tab:noise_floor`. Rows: 5 cells x 3 pairings (frozen-rep1, frozen-rep2, rep1-rep2); columns: Delta
and its 95% interval on each of the three coordinates. Built directly from
`results/phase0_audit/noise_floor_delta.csv`, asserting every printed value against that file, and
emitting a `.csv` beside the `.tex`. Minus signs as U+2212 in the text, `$-$` in the LaTeX.

**Two blocks, and the zero-exclusion claim verified separately for each.** The table carries two
groups of rows that were collected differently and must stay visually separate, with a column
saying which is which:

* **Block A -- re-executed for this audit.** The 5 cells x 3 pairings above: two re-executions run
  now, one machine, one code state.
* **Block B -- frozen pairs reclassified as re-runs.** The two C-8 rows, SFG/german and
  SFG/credit: two runs from the original study recorded weeks apart under the `controlled` and
  `native` labels.

The sentence *"No interval excludes zero"* is a claim about every row, so it is **verified
separately for Block A and for Block B**, and the caption states each result rather than asserting
one for both. Block A is already measured: 0 of 45 intervals exclude zero. Block B is not yet
computed. **If any Block-B interval excludes zero, report that before anything else** -- per C-8,
it would mean the estimator does not cover re-execution noise, which is a finding about the
bootstrap rather than about SFG.

**Caption (the user's text, 2026-09-25). Use verbatim.** Only the bracketed clause is filled, from
the per-block verification. If the verification does not support it, **the sentence is not
rewritten silently -- it is reported first**:

```latex
\caption{\textbf{Re-execution differences in the cell-level mean intervention contrast.}
Upper block: five primary cells re-executed twice under the frozen configuration on a single machine and code state, giving three realizations and three pairwise differences per cell.
Lower block: the two SFG cells whose controlled and published-horizon runs differ only by re-execution, recorded separately in the original study.
Entries give the difference and its 95\% paired hierarchical-bootstrap interval; [no interval excludes zero in either block].
Re-execution variation differs by more than an order of magnitude across cells (on \(-\Delta_{\mathrm{DP}}\), from 0.001 for NIFTY/German to 0.067 for SFG/German).}
\label{tab:noise_floor}
```

Filling rules, fixed in advance:

* Block A is settled: 0 of 45 intervals exclude zero. Only Block B decides the bracket.
* If **every** Block-B interval also excludes zero nowhere, the bracket reads
  `no interval excludes zero in either block`.
* If **any** Block-B interval excludes zero, the bracket reads
  `no interval excludes zero in the upper block`, and those rows are reported separately and
  before the table is considered done.
* The range figures 0.001 (NIFTY/German) and 0.067 (SFG/German) are Block A only, and are correct
  against `noise_floor_delta.csv` (0.0013 and 0.0668 on -Delta_DP). **If a Block-B value falls
  outside that range, report it** -- the sentence claims a range for the upper block, so a larger
  lower-block value does not contradict it but the reader should not be left to assume otherwise.

**C-15. One word for the protocol, in the figures and the caption.** "native" becomes "published"
wherever a reader sees it: axis labels, legends and panel titles of Fig. 2
(`fig3_protocol_variation`) and appendix Fig. 5 (`figS4_fixed_epoch_trajectory`), and the caption of
Table 24 (`tableS3b`), where "controlled -> native" becomes "controlled -> published horizon". The
CSV column values and the internal term names stay as they are; this is the reader-facing wording
only, and the two must not be conflated when checking.

**C-13. Table 27 gains an interval marker.** `tableS2_configuration_pairs` gets a column (or a
symbol on the estimate) showing whether that pair's Delta_attr 95% interval excludes zero, from
`configuration_delta_attr_bootstrap.csv` (26 pairs x 3 coordinates, seed 20260914, 10,000
replicates). Counts to reproduce: 11 of 26 for dAUC, 8 for -Delta_DP, 9 for -Delta_EO.

## C-17. Table generation moves out of the notebook

**Requirement (user, 2026-09-25): every figure and table is generated from `build_results.py`
output, with no `plot.ipynb` dependency.**

*Figures already satisfy this.* Every `figures/src/make_fig*.py` reads `results/*.csv`:
`1_main_package_vs_intervention.csv`, `2_configuration_variation.csv`, the three `3a/3b/3c` files,
`4_mechanistic_case_study.csv`, `5_component_case_study_FMP.csv`, plus `experiment_index.csv` and
the frozen `x25`/`x26` selected-epoch files. None of them imports the notebook. The notebook does
contain figure code, but under different names (`fig2_configuration_shift`,
`fig4_decomposition_negDP_bce`), so it is an older parallel set and not what `figures/*.pdf` came
from.

*Tables do not.* **Every table in `results/tables/` is produced only inside `results/plot.ipynb`**,
which is untracked (`.gitignore:65`). Its generators are `table1_summary`, `table2_comparisons`,
`table2_summary`, `table3a_selector_{cells,summary}`, `table3b_native_{pairs,summary}`,
`table3c_published_procedure`, `table4_decomposition`, `table5_fmp_components`,
`tableS1_1_fnrgnn_two_tasks`. So C-3, C-10, C-13 and C-16 all depend on code that the repository
does not carry, and the `dominance()` split that C-11 unified lived there too.

**The work:** port the notebook's table generation into a tracked script (`harness/experiments/`
whitelist, alongside `build_results.py`), reading only `results_v2/*.csv`, and verify it reproduces
the frozen tables before it is used for anything new. Until that is done, "no plot.ipynb
dependency" cannot be claimed for tables.

## C-18. Final report

One document, in this order: the headline numbers that changed; then the results that need care
(the C-9 disagreements, any cell whose verdict changes under C-14, any block-B interval excluding
zero, the cells C-12 excludes); then the list of artifacts. Not a log of what was run.

Must also contain (user, 2026-09-25):

* **C-14 scope.** Why the common-epoch comparison could not be run on the 36 primary cells, the
  per-cell table of what is stored (validation curve / per-epoch test outcome / checkpoint
  weights), and the re-run cost estimate for the full 36 and for an 11-cell one-per-method subset.
* **B_H1000:** the `tau_pkg < 0` and opposite-sign counts on all three coordinates.
* **method-balanced and LOMO** under B_rep1 and under B_H1000.

*Definitions, recorded because no code for them exists anywhere in the repository* -- they were
computed outside it for the inline `tab:aggregate_robustness`, so these are supplied here and must
be checked against whatever produced the published numbers:

    cell-weighted    larger cells / 36
    method-balanced  mean over the 11 methods of (larger cells in that method / its cells)
    LOMO             min-max of the cell-weighted share, dropping each method in turn

## C-19. How the tables are produced (user, 2026-09-25)

**Keys are LaTeX labels and Overleaf filenames, never paper numbers.** Appendix numbering shifts
when a table is inserted, so every reference in this plan, in `paper_numbers.csv` and in the
regeneration scripts uses `tab:<label>` and `table/<file>.tex`.

1. **The Overleaf `table/` files are the template.** Caption, footnotes and table structure are
   kept exactly; only the numbers are recomputed against `results_v2`. Hand edits that must survive:
   the "unit-level sign consistency" footnote on the per-cell tables and the FnRGNN table, the
   captions of `tableS3b` and `tableS3c`, and the whole format of
   `table2_configuration_sensitivity`.
2. **Where a generator disagrees with the template**, the template wins and the difference is
   reported -- never silently reconciled in either direction.
3. **Inline tables get no file.** `tab:evaluation_sets`, `tab:controlled_protocol_contract`,
   `tab:native_horizon_selector` and `tab:aggregate_robustness` are typed into the paper body, so
   their values go into `results_v2/paper_numbers.csv` keyed by label plus row and column position.
4. **New tables:** `table/tableS_noise_floor.tex` (`tab:noise_floor`) and
   `table/tableS_baseline_spec.tex` (`tab:baseline_spec`).
5. **C-17 order:** the ported generator must reproduce the frozen tables first; only then is it
   used to produce anything new.

## C-20. The table register (user, 2026-09-25)

**Keyed by label and filename. The numbers in the first column are the latest PDF's and are
expected to shift** once `tab:noise_floor` and `tab:baseline_spec` are inserted; nothing in the
pipeline may key off them.

| no. (drifts) | label | Overleaf file | content | Step C effect |
|---|---|---|---|---|
| 1 | `tab:exp1-summary` | `table/table1_summary.tex` | family summary | **B-dependent**: package larger, median abs tau_B |
| 2 | `tab:config-sensitivity` | `table/table2_configuration_sensitivity.tex` | configuration sensitivity, 4x3 | tau_I only -- unchanged |
| 3 | `tab:protocol-sensitivity` | `table/table3_protocol_counts_compact.tex` | protocol sensitivity | systematic 18 -> 16 |
| 8 | `tab:evaluation_sets` | **inline** | cells per evaluation set | 18 -> 16, re-execution row added |
| 10 | `tab:controlled_protocol_contract` | **inline** | controlled protocol contract | B wording |
| 12 | `tab:six_split_sensitivity` | **inline** | split-level sensitivity | unchanged |
| 14 | `tab:native_horizon_selector` | **inline** | published-horizon comparison spec | SFG rows, total 21 -> 19 |
| 20 | `tab:aggregate_robustness` | **inline** | robustness summary, subsets and LOMO | **B-dependent**; C-11 and C-12 |
| 17-19 | `tab:exp1-cells-dAUC`, `-negDP`, `-negEO` | `table/table1_cells_{dAUC,negDP,negEO}.tex` | 36 cells | **two tau_B columns change** |
| 21 | `tab:exp1-1-fnrgnn` | `table/tableS1_1_fnrgnn_two_tasks.tex` | FnRGNN, two tasks | **tau_B column changes** |
| 22 | (label in file) | `table/tableS2_configuration_pairs.tex` | 26 configuration pairs | C-13: Delta_attr marker |
| 23 | (label in file) | `table/tableS3a_selector_pairs.tex` | 36 selector pairs | unchanged |
| 24 | `tab:native-horizon-pairs` | `table/tableS3b_native_horizon_selector.tex` | published-horizon pairs | reclassification, C-15 caption |
| 25 | (label in file) | `table/tableS3c_published_procedure.tex` | 4 procedure pairs | unchanged |
| 27 | `tab:nifty_factorial` | `table/tableS4b_nifty_factorial.tex` | NIFTY 2x2 | unchanged |
| 29 | `tab:selection-support-all-coordinates` | `table/tableS4_selection_support_summary.tex` | selection-support | unchanged |
| new | `tab:noise_floor` | `table/tableS_noise_floor.tex` | re-execution noise, two blocks | C-16 |
| new | `tab:baseline_spec` | `table/tableS_baseline_spec.tex` | B specification and absolute metrics | new |

Tables 4-7, 9, 11, 13, 15, 16, 26, 28, 30 carry no numbers affected by this work.
`tableS5_fmp_fair_selector_summary.tex` exists in the folder but is not used by the manuscript.

*Note against the earlier plan:* C-3 and C-10 were written against paper numbers that do not
survive this register. C-10's "Tables 3, 7, 13, 26" resolves to `tab:protocol-sensitivity` and the
inline `tab:evaluation_sets` / `tab:native_horizon_selector`; Tables 7, 13 and 26 are not in the
affected set. C-13's "Table 27" is `tableS2_configuration_pairs`, not `tab:nifty_factorial`.
**Where this register and an earlier item disagree, the register wins.**

## Blocked — the Overleaf `table/` folder has still not arrived

The register above is received and recorded. The templates are not: the message names them as the
thing to send rather than carrying them. Needed in the state that already includes
`tableS4b_nifty_factorial.tex`, the R2 caption patches and the new
`table2_configuration_sensitivity.tex`.

Nothing in this repository substitutes for them. `results/tables/` holds the generated `.tex` only,
and it is missing most of the register entirely -- it has `table1_summary`, `table1_cells_*` and
`tableS1_1_fnrgnn_two_tasks` and none of `tableS2*`, `tableS3a/b/c*`, `tableS4*`, `table2_*`,
`table3_*`. So the hand edits cannot be recovered here, and regenerating from the notebook would
drop them.

**Until the folder arrives, every file-backed table in the register is on hold.** The work that
does not depend on it proceeds: C-1, C-2, C-9, C-11, C-12, C-14, the figures, `phase0_verify`, and
the value computation behind the four inline tables, which lands in `paper_numbers.csv` and needs
no template.

## Standing constraints

* The frozen bundle is never modified. `results/*.csv`, `per_unit_metrics.csv.gz` and every
  trajectory or checkpoint stay as they are.
* Every conclusion carries a file path + line, or the command and its output.
* Conclusions are labelled [확인됨] / [추정] / [확인 불가]. An estimate is never written as a fact.
* A bug found here is reported -- cause, blast radius, proposed fix -- not silently fixed.
