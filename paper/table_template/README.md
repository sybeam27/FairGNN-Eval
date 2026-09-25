# Overleaf `table/` — the templates the regenerated tables must match

Received 2026-09-25. These are the manuscript's current files, including edits made by hand in
Overleaf that no generator in this repository produces. **They are the template: caption, footnotes
and structure are kept exactly, and only the numbers are recomputed against `results_v2`.** Where a
generator disagrees with a file here, the file wins and the difference is reported.

Keyed by LaTeX label and filename, never by the paper's table number — inserting `tab:noise_floor`
and `tab:baseline_spec` shifts the later numbers.

| file | label | Step C effect |
|---|---|---|
| `table1_summary.tex` | `tab:exp1-summary` | **B-dependent**: package larger, median abs tau_B |
| `table1_cells_dAUC.tex` | `tab:exp1-cells-dAUC` | **two tau_B columns change** |
| `table1_cells_negDP.tex` | `tab:exp1-cells-negDP` | **two tau_B columns change** |
| `table1_cells_negEO.tex` | `tab:exp1-cells-negEO` | **two tau_B columns change** |
| `table2_configuration_sensitivity.tex` | `tab:config-sensitivity` | tau_I only — unchanged |
| `table3_protocol_counts_compact.tex` | `tab:protocol-sensitivity` | systematic 18 -> 16 |
| `tableS1_1_fnrgnn_two_tasks.tex` | `tab:exp1-1-fnrgnn` | **tau_B column changes** |
| `tableS2_configuration_pairs.tex` | `tab:config-pairs` | C-13: Delta_attr marker |
| `tableS3a_selector_pairs.tex` | `tab:selector-primary-pairs` | unchanged |
| `tableS3b_native_horizon_selector.tex` | `tab:native-horizon-pairs` | reclassification, C-15 caption |
| `tableS3c_published_procedure.tex` | `tab:published-procedure-pairs` | unchanged |
| `tableS4_selection_support_summary.tex` | `tab:selection-support-all-coordinates` | unchanged |
| `tableS4b_nifty_factorial.tex` | `tab:nifty_factorial` | unchanged (T4 filled the CIs) |
| `tableS5_fmp_fair_selector_summary.tex` | `tab:fmp-fair-selector-summary` | **not used by the manuscript** |

Two labels differ from the register, which listed them as "label in file"; they are read off the
files themselves: `tab:config-pairs` and `tab:published-procedure-pairs`, plus
`tab:selector-primary-pairs`.

## Hand edits that must survive regeneration

* The footnote on `table1_cells_*` and `tableS1_1_fnrgnn_two_tasks`: *"Bold = resolved (unit-level
  sign consistency >= 0.75, |tau| >= 0.01, and a 95% paired hierarchical-bootstrap interval
  excluding 0). All coordinates are oriented so that higher is better."*
* The captions of `tableS3b` and `tableS3c` (the R2 revisions), including tableS3b's note that the
  systematic and targeted blocks are not pooled into a common success rate.
* `table2_configuration_sensitivity.tex` entirely: it is a new 4x3 body with no `table` environment,
  and the previous version is retained beneath it as a comment block.
* `table3_protocol_counts_compact.tex` likewise has its `table` wrapper commented out — the file is
  a bare `tabular`, and the caption and label live in the manuscript body.
* `tableS4b_nifty_factorial.tex` carries the `\providecommand{\tbd}` guard and the footnote marked
  `\ast` about unit-level sign consistency 0.733 vs the rerun's 0.767.
* `tableS3a_selector_pairs.tex` uses a two-`minipage` split layout, with the single-column version
  retained as a comment block.

## Structural note for the generator

Three of these files are **fragments, not `table` environments**:
`table2_configuration_sensitivity.tex` and `table3_protocol_counts_compact.tex` emit a bare
`tabular`, and their captions and labels are written in the manuscript. A generator that wraps its
output in `\begin{table}` would break them.
