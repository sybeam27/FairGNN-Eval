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

Caption, draft -- to be replaced by the user's own text, and re-checked against the generated
table rather than carried over: *"Re-execution differences in the cell-level mean intervention
contrast under identical settings. Block A re-executes each cell twice for this audit; Block B
reclassifies two frozen pairs that differ only in the process they ran in. [zero-exclusion result,
stated per block.] Re-execution noise is strongly cell-dependent: between realizations the
cell-level mean tau_I moved by 0.0668 on -Delta_DP for SFG/German and by 0.0013 for NIFTY/German,
and the largest single-unit movement was 0.635 against 0.004 for FairVGNN/German."*

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

## Standing constraints

* The frozen bundle is never modified. `results/*.csv`, `per_unit_metrics.csv.gz` and every
  trajectory or checkpoint stay as they are.
* Every conclusion carries a file path + line, or the command and its output.
* Conclusions are labelled [확인됨] / [추정] / [확인 불가]. An estimate is never written as a fact.
* A bug found here is reported -- cause, blast radius, proposed fix -- not silently fixed.
