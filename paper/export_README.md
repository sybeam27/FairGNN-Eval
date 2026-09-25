# What makes a fairness-aware GNN fair? Intervention attribution across protocol

Code, matched per-unit results and analysis for the paper. Anonymised for review: no git history,
no author names, no absolute paths.

## Read this first: the numbers do not come from re-running anything

**Re-training is not deterministic on this pipeline, and the paper's numbers are computed from the
frozen per-unit results, not from a fresh run.** The sparse propagation these methods use has no
deterministic CUDA kernel, so two runs of the identical command, same seed, same machine, produce
different models. We measured how different:

* Five primary cells were re-executed twice. Across 45 paired comparisons of the cell-level mean
  intervention contrast, **no 95% interval excluded zero** — so the paper's intervals do cover
  re-execution noise.
* The size of that noise varies by more than an order of magnitude between cells. On
  \(-\Delta_{\mathrm{DP}}\) the cell mean moved 0.067 between realizations of SFG/German and 0.001
  on NIFTY/German; per unit the extremes are 0.635 and 0.004.

`results/phase0_audit/noise_floor_delta.csv` has every row. Anything you recompute from
`per_unit_metrics.csv.gz` will match the paper exactly; anything you retrain will not.

## Reproducing the paper's numbers, without a GPU

```bash
python phase0_verify.py results                 # frozen bundle: recompute every contrast
python phase0_verify.py results_v2/bundle       # the rebuilt bundle the paper reports
python harness/experiments/build_tables.py --check                       # tables vs the templates
python harness/experiments/build_tables.py --results results_v2/bundle --out out/tables
python harness/experiments/build_paper_numbers.py --out out/paper_numbers.csv
```

`phase0_verify` recomputes the three contrasts per cell from `results/per_unit_metrics.csv.gz` and
checks them against the paper-facing CSVs; it reproduces every point estimate to ~1e-16.
`build_tables --check` fills each Overleaf template from the frozen bundle and diffs, so a change in
the pipeline shows up as a table that no longer reproduces.

`results_v2/paper_numbers.csv` lists every number the manuscript states with its location, the
definition that produces it, the file it came from, and whether it changed in the rebuild.

## The baseline was rebuilt

The published bundle trained the common baseline B separately for each method, and on German it
trained for 200 epochs although the configuration resolver returns a published horizon of 1000 —
the runner read the configuration and dropped the horizon. That left German's baseline below chance
(test AUC 0.4415). The rebuild trains one B per (dataset, split, run), shared by every method, and
restores the resolved horizon where one exists, which is German alone (AUC 0.6543).

`results_v2/B_rebuild_diff.md` gives the full comparison across four baselines. In short: the
intervention contrast \(\tau_{-I\rightarrow+I}\) is unchanged — it never references B, and all 90
cells reproduce it at max |diff| 0.0 — while everything referenced to B moves, most of it on
\(\Delta\mathrm{AUC}\).

## Seeding is not uniform, and is recorded rather than fixed

`per_unit_metrics.csv.gz` carries a `seed` column (= 27 + run_id, the same across splits) and an
`rng_rule` column, because the study's three runner paths do not seed the same way:

| rule | which rows |
|---|---|
| `manual_seed(seed)` | the baseline B, and the FairGNN / NIFTY / FairVGNN / FairGB arms |
| `seed_all(seed*1000 + split)` | the FairSIN / BeMap / BIND / GEAR / SFG / FnRGNN / FairEdit arms |
| `seed_all(seed*1000 + split + 1)` | EDITS |

`pilot_tau.py` seeds per (seed, split) before each arm, but `train()` reseeds with `seed` alone as
its first statement, so for the first group the effective seeding is split-independent. The pairing
the design needs still holds — both arms of a cell start from the same state. This is left as it
stands, because the frozen results depend on it.

## What the method wrappers are, and how close to upstream

`models/algorithms/*.py` are our own `fit`/`predict` wrappers, **not** upstream code. Measured
against the official sources (`harness/X6_PROVENANCE.md`):

| wrapper | fidelity to upstream | status |
|---|---|---|
| NIFTY | 86.7% | local-mirror-verified |
| FairVGNN | 83.4% | local-mirror-verified |
| FairGNN | 51.0% | local-unverified |
| GNN (baseline) | 38.0% | local-unverified |

The other methods are run through their own upstream code, loaded by path so the upstream file
executes unmodified (`harness/adapters/`).

## What is not in this checkout

**Included:** all of our own code — runners, wrappers, figure generators, the analysis and table
builders; the frozen and rebuilt result bundles; the matched per-unit outcomes; the audit record;
`harness/provenance/`, which the configuration resolver reads at run time.

**Not included, and why:**

* **Six method repositories** — BIND, FairSIN, FMP, GEAR, SFG, FairGT — ship no licence text, so we
  have no right to redistribute them. `fetch_upstream.sh --list` gives what we know about each.
  GEAR (`https://github.com/jma712/gear`) and SFG (`https://github.com/sh-qiangchen/SFG`) were
  fetched and checked on 2026-09-25: **every file the adapters read is byte-identical to the
  repository's default branch**. Only FairGT still has no URL we could verify.
* **FnRGNN** — its upstream identifies an author, so it is withheld during review.
* **Datasets** — redistributed by their own sources under their own terms. Checksums for the exact
  files used are in `harness/data_manifest.tsv`.

BeMap and FairGB **are** included, under their MIT licences, with their bundled `dataset/` and
`figures/` directories removed (226 MB and 13 MB of data, available from the same sources).

`patches/fairgb_local_edits.patch` carries the three files we changed in the vendored FairGB copy —
`eval.py` gains a `return_output=True` branch, `utils.py` and `main.py` change import paths.

**Honest limitation:** we did not record commit hashes when these repositories were vendored, so
`fetch_upstream.sh` cannot pin them. The two hashes it does print for GEAR and SFG record what
those default branches held on 2026-09-25, when the file-by-file check was made; they are **not**
the commits the experiments ran against, and the entries stay UNPINNED. It fetches the default branch and points you at
`harness/external_repos.tsv` and `harness/METHOD_EXTENSION_INVENTORY.csv`, which record what each
adapter expects. The adapters load upstream files by path and by line number, so a repository that
has moved will fail loudly rather than quietly run something else.

## Layout

```
harness/core/          the evaluator, the trajectory recorder, the configuration resolver
harness/adapters/      one adapter per method, loading upstream code by path
harness/experiments/   the runners, and build_results / build_tables / build_paper_numbers
harness/provenance/    upstream commands and argparse defaults the resolver parses
models/algorithms/     our wrappers
figures/src/           the figure generators; figures/*.pdf are their output
results/               the frozen bundle, per_unit_metrics.csv.gz, the audit record
results_v2/            the rebuilt bundle, the regenerated tables, paper_numbers.csv
paper/table_template/  the manuscript's table files, which the generators must match
```

`PAPER_ARTIFACT_MAP.md` maps each figure and table to the file its numbers come from.
