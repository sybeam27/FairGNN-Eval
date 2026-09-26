# Where each figure and table in the paper comes from

The filenames below are the generators' own output names, which do not match the paper's figure
numbers: `fig3_protocol_variation.pdf` is Figure 2, and `fig4_selection_support_bridge.pdf` is
Figure 6. The numbers move whenever an appendix item is added, so the files are keyed to their
LaTeX labels and the generator that writes them, and this table carries the numbering of the
submitted version.

## Figures

| Paper | Where | File | Generator |
|---|---|---|---|
| Figure 1 | §4.2.1 | `figures/fig1_intervention_attribution.pdf` | `figures/src/make_fig1.py` |
| Figure 2 | §4.2.2 | `figures/fig3_protocol_variation.pdf` | `figures/src/make_fig3.py` |
| Figure 3 | §4.2.3 | `figures/fig3_selection_support_trajectory.pdf` | `figures/src/make_fig3_trajectory.py` |
| Figure 4 | App. G.1 | `figures/figS1_intervention_attribution_negEO.pdf` | `figures/src/make_fig1.py` |
| Figure 5 | App. J.4 | `figures/figS4_fixed_epoch_trajectory.pdf` | `figures/src/make_fig4.py` |
| Figure 6 | App. J.5 | `figures/fig4_selection_support_bridge.pdf` | `figures/src/make_fig4.py` |
| Figure 7 | App. K.1 | `figures/fig5_fmp_component_bridge.pdf` | `figures/src/make_fig5.py` |
| Figure 8 | App. K.2 | `figures/figS5_fmp_component_bridge_eo_auc.pdf` | `figures/src/make_fig5.py` |
| Figure 9 | App. K.3 | `figures/figS5_fmp_fair_by_selector.pdf` | `figures/src/make_fig5.py` |

## Tables generated from the result bundle

The rest of the paper's tables are written directly in the manuscript: they describe the design
(Tables 4-10, 14, 17-20, 29, 31, 33) or are aggregates whose values are listed in
`results/paper_numbers.csv` (Tables 12, 16, 26, 28, 30, 32).

| Paper | LaTeX label | File |
|---|---|---|
| Table 1 | `tab:exp1-summary` | `results/tables/table1_summary.tex` |
| Table 11 | `tab:baseline_spec` | `results/tables/tableS_baseline_spec.tex` |
| Table 13 | `tab:noise_floor` | `results/tables/tableS_noise_floor.tex` |
| Table 15 | `tab:threshold_sensitivity` | `results/tables/tableD3_threshold_sensitivity.tex` |
| Table 21 | `tab:exp1-cells-dAUC` | `results/tables/table1_cells_dAUC.tex` |
| Table 22 | `tab:exp1-cells-negDP` | `results/tables/table1_cells_negDP.tex` |
| Table 23 | `tab:exp1-cells-negEO` | `results/tables/table1_cells_negEO.tex` |
| Table 24 | `tab:exp1-1-fnrgnn` | `results/tables/tableS1_1_fnrgnn_two_tasks.tex` |
| Table 25 | `tab:config-pairs` | `results/tables/tableS2_configuration_pairs.tex` |
| Table 27 | `tab:native-horizon-pairs` | `results/tables/tableS3b_native_horizon_selector.tex` |

## Every reported number

`results/paper_numbers.csv` lists each value the paper states with its location, the definition that
produces it, the file it was computed from, and whether the baseline rebuild moved it.
