# final_figures_src

Sources of the PDFs in `../final_figures/` (that folder holds only the PDFs).

Every paper figure, redrawn in one tone (that of Fig. 2). Regenerate all of them with

    python final_figures_src/make_all.py

Inputs come from `final_results/` (and `study/results/x25`, `x26` for Fig. S4's selected epochs); nothing there is
written. Numbers, filters, labels and resolved flags are exactly those of the per-experiment scripts and the
notebook; only the look is unified (`style.py`).

| file | role | source |
|---|---|---|
| fig1_tradeoff_direction | main | Exp. 1 (`make_fig1.py`) |
| figS1_tradeoff_direction_negEO | appendix | Exp. 1 |
| figS1_arms_by_family_dataset(_negEO) | appendix | Exp. 1 |
| figS1_1_fnrgnn_two_tasks | appendix | Exp. 1.1 |
| fig2_sign_resolution | main (single column) | Exp. 2 (`make_fig2.py`) |
| fig3_protocol_variation | main | Exp. 3 (`make_fig3.py`) |
| fig4_selection_support_bridge | main | Exp. 4 (`make_fig4.py`) |
| figS4_fixed_epoch_trajectory | appendix | Exp. 4 |
| fig5_fmp_component_bridge | main | Exp. 5 (`make_fig5.py`) |
| figS5_fmp_component_bridge_eo_auc | appendix | Exp. 5 |
| figS5_fmp_fair_by_selector | appendix | Exp. 5 |

Tone (`style.py`): 6.5 pt text, 6 pt ticks, black labels, grey axes; bars and markers filled with a light tint
of their role colour and outlined in the full colour; legends in a thin light box. Full-width figures are 7.0 in,
Fig. 2 is one column (3.35 in).

One palette for the whole paper (`style.ROLE`): a colour always means the same **role**, never a method.
Method families are told apart by **marker shape** only (square = Modification, triangle = Constraint,
circle = Regularization).

| role | colour | where |
|---|---|---|
| intervention side ($M^{+I}$) | plum `#8c3b6b` | Fig. 1 arms/trade-off intervention arrows, Fig. 4 $E_{+I}$, Fig. 5 +fair |
| surrounding package ($M^{-I}$) | slate `#3b6b8c` | Fig. 1 $M^{-I}$ markers, Fig. 4 $-E_{-I}$, Fig. 5 +prop |
| step away from the baseline (B → $M^{\mathrm{base}}$) | teal `#2a7f7f` | Fig. 5 base |
| levels and totals | ink `#3d3c39` | Fig. 1 package arrows, Fig. 3 resolved-in-both, Figs. 4-5 levels / pkg |
| prefix-run difference | grey `#9a9993` | Fig. 3 controlled, Fig. 4 $R_{\mathrm{traj}}$ |
| sign flip | purple `#7b3fa0` | Figs. 2-3 |
| resolved → unresolved | magenta `#c2185b` | Figs. 2-3 |
| unresolved → resolved | ochre `#c99a1e` | Figs. 2-3 |

`preview/` holds PNG copies for on-screen checking; `../final_figures/` holds only the PDFs.
