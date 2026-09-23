# X28 publication tables — source artifact mapping

Every number in every table is parsed from a frozen artifact. **No training was
run**, no frozen value was altered, and no threshold, classification or
resolved rule was redefined.

Builder: `harness/experiments/x28_tables/make_tables.py`
Shared parsing/formatting/checking: `harness/experiments/x28_tables/tcommon.py`
(the artifact parsers are imported from `harness/experiments/x28_figures/common.py`
rather than rewritten, because those were validated by the 401 checks of the
figure job).

Outputs per table: `harness/results/tables/paper/<name>.tex`,
`.../source_data/<name>_source.csv`, `.../<name>_preview.png`.

> The preview images are **matplotlib renderings of the generated table** — the
> rows the `.tex` actually contains, with LaTeX markup stripped — not LaTeX
> output and not the source CSV. No TeX toolchain is installed in this
> environment, so the `.tex` files were validated structurally instead of
> compiled: environment balance, no line break outside `tabular`, and the column
> count of every row (counting `\multicolumn` spans) against the colspec.
> 309 rows across 14 files pass.

## Frozen artifacts used

| artifact | what is taken from it |
| --- | --- |
| `harness/results/armA_final_bootstrap.txt` | 12 controlled cells: means and 95% intervals for `tau_base^audit`, `tau_int`, `tau_pkg^audit`, `D_selector`, on ΔAUC and −ΔDP |
| `harness/results/armA_final_report.txt` §[4] | 12 controlled cells: sign stability of `tau_int` on −ΔDP, and the magnitude relation between them |
| `harness/results/armA_final_report.txt` §[5] | the J1 count (10 of 12), used only to check the table's own count |
| `harness/results/armB_phase1_seven_cell_analysis.txt` | 7 native cells: controlled and native means, intervals, sign stability, **resolved flags**, and the frozen X22/X23 transfer class, on all three coordinates |
| `harness/results/x24_nifty_german_summary.csv` | P00/P10/P01/P11 and Δ_total, Δ_H, Δ_D, I_HD, both selectors, three coordinates |
| `harness/results/x25/x25_summary.csv` | X25 selection-support terms: h200, cap, full, S, T, H, and the telescoping E_on/E_off |
| `harness/results/x26/x26_bail_summary.csv` | X26 terms: h200, cap, full, S, T, H_shift, E_on/E_off |
| `harness/results/x27/x27_pokec_{z,n}_summary.csv` | FMP τ_prop, τ_fair, τ_total over the λ grid, three selectors, three coordinates |
| `harness/X26_RESULTS.md` | the stopped-cell record behind Table A7 |

## Table → source mapping

| table | file | sources | notes |
| --- | --- | --- | --- |
| **1** controlled attribution | `table1_controlled_attribution` | armA bootstrap + armA §[4] | −ΔDP, σ_c^BCE; resolved computed by the frozen rule (see below) |
| **1b** with ΔAUC | `table1b_controlled_attribution_with_auc` | same | compact variant carrying the utility coordinate |
| **A1** controlled full | `tableA1_controlled_full` | armA bootstrap + §[4] + armB seven-cell | −ΔEO only where frozen (7 cells); selector shown as frozen `D_selector` |
| **2** transfer | `table2_transfer` | armB seven-cell | frozen X22/X23 transfer class, unchanged |
| **A2** transfer full | `tableA2_transfer_full` | armB seven-cell | 7 cells × 3 coordinates = 21 rows |
| **3a** selection support | `table3a_selection_support` | x25_summary (D1), x26_bail_summary | σ_c^BCE, −ΔDP |
| **3b** FMP components | `table3b_fmp_components` | x27 summaries | σ_last, −ΔDP |
| **3** combined | `table3_mechanistic_combined` | 3a + 3b | one-table alternative, for readability comparison |
| **4** design coverage | `table4_design_coverage` | design facts; no results | counts cross-checked against the artifacts |
| **A3** NIFTY factorial | `tableA3_nifty_factorial` | x24 summary | 2 selectors × 8 quantities × 3 coordinates = 48 rows |
| **A4** X25 full | `tableA4_x25_decomposition` | x25_summary | D0/D1 × BCE/AUC × 3 coordinates |
| **A5** X26 full | `tableA5_x26_decomposition` | x26_bail_summary | BCE/AUC × 3 coordinates |
| **A6** FMP full grid | `tableA6_fmp_grid` | x27 summaries | 2 datasets × 3 selectors × 2 λ1 × 2 λ2 × 3 coordinates = 72 rows |
| **A7** excluded | `tableA7_excluded` | X26_RESULTS | FairGB/Credit, stopped at its pre-registered gate |

## The one computed quantity, and how it is validated

No frozen Arm A artifact carries a *resolved* flag for the 12 controlled cells:
the bootstrap table records only whether the interval excludes zero, and §[4]
records only sign stability. The three ingredients of the **frozen** resolved
rule are nevertheless all present in frozen artifacts — |mean| and the interval
from the bootstrap table, sign stability from §[4] — so the tables apply the
registered rule to registered ingredients rather than inventing a new one.

That application is **validated against the seven Arm A resolved flags that the
armB seven-cell analysis states explicitly**. All seven reproduce, and the sign
stabilities agree between the two artifacts (0.73, 0.57, 0.55, 0.67, 0.53, 0.60,
0.60). Had any of the seven disagreed, the run would have stopped without
writing a table.

Where a value genuinely does not exist in any frozen artifact, it is left as
`--` and the absence is stated in the caption — specifically **−ΔEO for the five
controlled cells that are not native-validation cells** (FairGNN on all three
datasets, NIFTY on Bail and Credit), which no frozen Arm A artifact bootstraps.
Likewise, the σ_c^AUC effect is presented through the frozen paired difference
`D_selector` rather than as a reconstructed column, because no frozen artifact
carries an interval for the σ_c^AUC effect itself.

## Automated checks (522, all passed)

Run by `make_tables.py` before anything is written; a failure raises and stops.

* every interval satisfies `lo ≤ mean ≤ hi`;
* no missing cell, no duplicated (method, dataset);
* `tau_pkg = tau_nonint + tau_int` for all 12 controlled cells on both
  coordinates (tolerance 3e-4: the frozen file prints 4 decimals);
* the two Arm A artifacts agree on the means they both carry;
* the resolved reproduction described above;
* the Table 1 count of |τ_nonint| > |τ_I| equals the frozen J1 count (10 of 12);
* every transfer class is one of the five frozen X22/X23 classes;
* `Δ_total = P11 − P00` for X24, both selectors, three coordinates;
* `S = full − cap`, `T = cap − short`, `H = S + T` for X25 and X26, and
  `S = E_on − E_off`;
* `τ_total = τ_prop + τ_fair` for all 72 FMP grid entries;
* no FMP fairness-correction cell is resolved (the bound printed in Table 3b is
  computed, not typed);
* the rendered cells of Tables 1 and 2 all appear verbatim in their `.tex`.

Worst residual across all 522 checks: **1.0e-04**, from the 4-decimal printing
precision of `armA_final_bootstrap.txt` in the `tau_pkg` identity.

## Defects found and fixed

No scientific mismatch was found — the 522 checks passed on the first complete
run, so nothing was stopped. Three defects in the *table production* were found
and fixed:

1. **Table A1's column specification was one column short of its rows** (13 vs
   14): the −ΔDP group needs six specs and I had written five. This is an
   "Extra alignment tab" compile error. Because no TeX toolchain exists here it
   could not have been caught by compiling; the structural validator caught it.
2. **The preview images rendered the source CSV, not the table.** Table 3b's
   `.tex` has 5 columns while its preview showed 13 raw full-precision columns,
   overlapping and unreadable. A preview that shows different content from the
   table it previews is misleading, so previews are now built from the generated
   LaTeX rows; the CSV keeps full precision as the machine-readable source.
3. **The LaTeX emitter put table notes after `\caption{}` as `\\[2pt]
   \footnotesize …`.** A bare `\\` there sits outside any `tabular` and makes
   LaTeX report "There's no line here to end". Notes now live inside the
   caption.

One further bug was mine, in the checking code rather than the output: the first
version of the structural validator extracted the colspec with `\{([^}]*)\}`,
which truncates `p{2.4cm}` at the first brace and crashed on Table 4.
