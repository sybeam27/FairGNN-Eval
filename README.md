# Protocol-Conditional Attribution in Fairness-Aware Graph Neural Networks

Code, frozen protocols and results for the paper.

A fairness-aware GNN is a package: a baseline architecture, a training procedure, an evaluation
protocol, and the fairness-motivated intervention the paper is named after. Reported gains are
usually attributed to the last of these. This study separates them. For every cell (method ×
dataset × configuration) the package effect is split into the part that comes from everything
around the intervention and the part that comes from the intervention itself,

    B --tau_{B->-I}(P)--> M^{-I} --tau_{-I->+I}(P)--> M^{+I},
    tau_{B->+I}(P) = tau_{B->-I}(P) + tau_{-I->+I}(P),

with both arms trained under one protocol P: the same split, horizon, checkpoint selector,
decision rule and evaluator. Every estimate is a mean over 30 matched units (6 splits × 5 runs)
with a paired hierarchical bootstrap interval.

## What is here

| path | contents |
|---|---|
| `harness/core` | the harness: evaluator, metrics, trainer, model, trajectory recording |
| `harness/adapters` | one adapter per evaluated method; the method's own code is never modified |
| `harness/experiments` | `build_results.py`, which turns the raw per-cell store into `results/`, `robustness_six_split.py` (appendix sensitivity check) and the three analysis modules they import |
| `harness/*.md` | pre-registrations, stop reports and results documents, frozen before each run (X1-X31) |
| `harness/*.csv`, `*.tsv` | the method and dataset inventories, the data manifest and the pinned upstream commits |
| `results/` | the paper-facing tables: one CSV per paper section, and `experiment_index.csv`, which says what each one holds fixed and varies |
| `results/per_unit_metrics.csv.gz` | the matched per-unit outcomes (cell x split x run x arm) every contrast and interval is computed from |
| `harness/coverage_manifest.csv` | every method x dataset candidate with its eligibility decision and, when omitted, a fixed exclusion category |
| `harness/intervention_manifest.csv` | what the intervention toggles between M^{-I} and M^{+I}, and what the two arms share |
| `harness/score_convention_manifest.csv` | the loss, raw output, canonical decision margin and decision rule of each implementation |
| `results/appendix_robustness` | the six-cluster sensitivity check: `six_split_sensitivity_summary.csv` compares the frozen hierarchical-bootstrap classification with the split-level t-interval analysis, and `six_split_sensitivity_cells.csv` holds the cell-level values |
| `figures/` | the paper figures (PDF) |

`results/README.md` documents every column; the figures are drawn from `results/` alone.

## Reproducing

`results/` is the bundle the paper reports: the common baseline B rebuilt at the resolved published
horizon, one draw per unit. It is produced by

```bash
python harness/experiments/build_results.py --baseline results_v2/baselines/B_rep1 --out results
python harness/experiments/robustness_six_split.py   # appendix sensitivity check
```

`results/superseded/` holds the pre-rebuild bundle — the "before" side of that fix, and the
reference the table check verifies against. Without a GPU you can still check the whole chain:

```bash
python phase0_verify.py results                      # recompute every contrast from per-unit data
python harness/experiments/build_tables.py --check   # fill the paper's tables and diff them
python harness/experiments/build_paper_numbers.py --out results/paper_numbers.csv
```

`results/paper_numbers.csv` lists every number the paper states with its location, the definition
that produces it, the file it came from, and whether the baseline rebuild moved it.
`results/B_rebuild_diff.md` documents that rebuild across four baselines.

**The runners, the method wrappers and the figure scripts are in this repository.** Re-running the
experiments additionally needs the datasets and the third-party method repositories, neither of
which is redistributed here; `fetch_upstream.sh --list` says where each one comes from.

The training artifacts and per-epoch trajectories are not redistributed — 4 GB, and the runners
regenerate them. For auditability we publish instead a compact per-unit metric export,
`results/per_unit_metrics.csv.gz` (16,200 rows): the matched split × run outcomes of all three
arms, from which every reported contrast and its bootstrap interval can be reconstructed.
`PAPER_ARTIFACT_MAP.md` shows how.

Nothing above needs a GPU. What does need one is re-training, and re-training does not reproduce
the stored numbers: see the note below.

**Re-training is not deterministic, and the paper's numbers come from the stored per-unit results.**
The sparse propagation these methods use has no deterministic CUDA kernel, so two runs of the same
command at the same seed differ. We measured how much: five primary cells were re-executed twice,
and across 45 paired comparisons of the cell-level mean intervention contrast **no 95% interval
excluded zero**, so the paper's intervals do cover re-execution noise. The size of that noise is
strongly cell-dependent — on \(-\Delta_{\mathrm{DP}}\) the cell mean moved 0.067 between
realizations of SFG/German and 0.001 on NIFTY/German. Every row is in
`results/phase0_audit/noise_floor_delta.csv`. That is why the estimates in `results/` are published
alongside the code.

`PAPER_ARTIFACT_MAP.md` maps every figure, table and appendix item to the file it comes from.

## What is deliberately not in this repository

* **Datasets** (German, Credit, Bail, Income, Pokec-z, Pokec-n, and the two Pokec variants with the alternative sensitive attribute). Obtained from their own sources;
  `harness/data_manifest.tsv` records the checksums of the copies used.
* **The evaluated implementations.** Each method is run from its own published repository under
  its own licence, and six of them ship no licence text at all, so we have no right to
  redistribute them. `fetch_upstream.sh` fetches them; `harness/provenance/` records the upstream
  sources the configuration resolver parses, and the adapters state exactly which lines they wrap.
  BeMap and FairGB are included under their MIT licences. Please cite the original papers and
  repositories when you use their results.
* **Trajectory artifacts** (`*.npz`, `*.pt`) and caches: large and regenerated by the runners;
  the runs that produced them are checksum-manifested in `harness/results/`.

## How the results are labelled

An estimate is called **resolved** when its sign is stable in at least 75% of bootstrap
replicates, its magnitude is at least 0.010, and its 95% interval excludes 0. This is a reporting
category, not a hypothesis test, and it is recorded per coordinate: ΔAUC, −Δ_DP and −Δ_EO, all
oriented so that higher is better. `results/README.md` documents every column.

## Licence

* Code: MIT (`LICENSE`).
* Result tables and figures: CC BY 4.0 (`LICENSE-DATA`).
* Datasets and third-party method implementations: their own licences; not redistributed here.

## Citation

```bibtex
@inproceedings{fairgnn-eval,
  title     = {Protocol-Conditional Attribution in Fairness-Aware Graph Neural Networks},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2026}
}
```
