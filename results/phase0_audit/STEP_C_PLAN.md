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
`fig2_sign_resolution` is 3.57 in wide and is the one figure that would be scaled *up*; it is not
in this list and its intended placement should be confirmed before it is regenerated.

**C-5. `phase0_verify` on `results_v2/`**, full output kept.

**C-6. `results_v2/paper_numbers.csv`.**

**C-7. `results_v2/B_rebuild_diff.md`** — per `B_rebuild_decision.md` "Deliverables".

**C-8. The two `re_execution` pairs.** Delta_attr and its 95% interval per coordinate for
SFG/german and SFG/credit, by the same estimator as A-3.

**C-9. Native recount on 19 pairs.** Re-aggregate the native Delta_attr interval-excludes-zero
counts with the two re-execution pairs removed (21 -> 19). Add a column
`exceeds_rerun_max`: whether that pair's |Delta_attr| exceeds the largest |Delta| observed
between two realizations of the *same* cell in the Step B noise floor. A native pair that does
not clear its own re-execution noise is not evidence of a protocol effect.

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
