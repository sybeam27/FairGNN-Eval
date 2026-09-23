# Paper ↔ artifact map

Where each number in the paper comes from. Every file below is in this repository.

## Main text

| paper | file |
|---|---|
| Fig. 1 — fairness attribution and the intervention trade-off | `results/1_main_package_vs_intervention.csv` |
| Fig. 2 — configuration variation (sign flips, resolution changes) | `results/2_configuration_variation.csv` |
| Fig. 3 — protocol variation (selector; horizon + selector; published procedure) | `results/3a_…csv`, `results/3b_…csv`, `results/3c_…csv` |
| Fig. 4 — selection-support decomposition (NIFTY/German, FairGB/Bail) | `results/4_mechanistic_case_study.csv` |
| Fig. 5 — FMP component decomposition | `results/5_component_case_study_FMP.csv` |
| Table 1 — attribution by method family, and per cell | `results/1_main_package_vs_intervention.csv` |
| The rendered figures | `figures/*.pdf` |

## Appendix

| appendix item | file |
|---|---|
| Which method × dataset cells are in the study, and why the others are not | `harness/coverage_manifest.csv` |
| What the intervention toggles, and what the two arms share | `harness/intervention_manifest.csv` |
| Score semantics and the decision rule of each implementation | `harness/score_convention_manifest.csv` |
| Six-cluster sensitivity (split-level t intervals, leave-one-split-out) | `results/appendix_robustness/six_split_sensitivity_summary.csv` (per cell: `…_cells.csv`) |
| FnRGNN on both tasks (classification, released regression task) | `results/1b_…csv`, `results/1c_…csv` |
| Matched per-unit outcomes behind every contrast and interval | `results/per_unit_metrics.csv.gz` |
| What each section file holds fixed and varies | `results/experiment_index.csv` |
| Column definitions | `results/README.md` |

## Recomputing a reported number

`results/per_unit_metrics.csv.gz` has one row per (cell, split, run, arm) with `auc`, `dp`, `eo`.
Taking the controlled protocol and the `common_bce` selector, the three estimands are

    tau_{B->-I}  = mean over units of  Y(M_minus_I) - Y(B)
    tau_{-I->+I} = mean over units of  Y(M_plus_I)  - Y(M_minus_I)
    tau_{B->+I}  = mean over units of  Y(M_plus_I)  - Y(B)

with `Y` = `auc` for ΔAUC and `-dp`, `-eo` for −Δ_DP and −Δ_EO, so that every coordinate is
higher-is-better. Averaging those over the 30 matched units reproduces the `*_mean` columns of
`results/1_main_package_vs_intervention.csv` exactly (checked to 5e-6 on all 36 cells and all three
coordinates). The intervals are the paired hierarchical bootstrap of
`harness/experiments/bootstrap_armA.py` (10,000 replicates: splits, then runs within a split).

## Provenance of the evaluated implementations

`harness/external_repos.tsv` and `harness/METHOD_INVENTORY.csv` record the source repository and the
pinned commit of every method; the adapters in `harness/adapters/` state which lines they wrap. The
implementations themselves are not redistributed here.
