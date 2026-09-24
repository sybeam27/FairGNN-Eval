# Export checklist — what the public repository must contain

The release currently publishes 170 tracked files, and none of the 36 primary cells can be
executed from them: `.gitignore:49` excludes every runner under `harness/experiments/` bar five
analysis scripts, `:9` excludes all of `models/`, `:48` excludes `harness/provenance/` (which
`published_config.py` reads at run time), `:66` excludes `figures/src/`, and `:6` excludes
`data/`. This file tracks what has to be true before the anonymous repository is published.

## A. Files fixed in the working tree that no commit carries

Each was edited to keep the study runnable or correct, and each lives under an ignored path, so
`git` does not carry it. Every one must appear in the export.

| # | path | what was changed | why it is not committed |
|---|---|---|---|
| 1 | `figures/src/make_fig3.py` | panel (b) title -> "Published horizon" | `.gitignore:66` |
| 2 | `results/plot.ipynb` | 5 strings -> published-horizon wording | `.gitignore:65` |
| 3 | `models/algorithms/FairGB/main.py` (5 sites), `FairGB/utils.py`, `FairGT_alg.py`, `FairGT_alg copy.py`, `FairGT/train_fairgt.py` | `from FairGate.models.` -> `from models.` (10 sites) | `.gitignore:9` |
| 4 | `harness/experiments/noise_floor_rerun.py` | new: the Step B noise-floor runner | `.gitignore:49` |
| 5 | `figures/src/make_fig3_trajectory.py` | new: the main-text selection-support trajectory figure | `.gitignore:66` |
| 6 | `figures/src/preview/fig3_selection_support_trajectory.png` | preview of the above | `.gitignore:28`, `:66` |

Item 3 is the one that breaks execution outright: without it FairGB and FairGT fail at import
with `ModuleNotFoundError: No module named 'FairGate'`.

## B. The difference that must be empty

At the clean-checkout verification step, collect the set of files actually imported and executed
while running the two probe cells, and subtract the tracked set:

    (files imported or executed to produce a paper result)  \  (git ls-files)  ==  {}

Collect the left side by measurement, not by grepping: run each probe cell and dump
`sys.modules` (plus the data and provenance paths each resolver opened) rather than reasoning
about imports statically. Any non-empty difference names a file the public repository is missing.

## C. Clean-checkout probe

In a fresh environment, from the export alone, run the fetch scripts and then one unit each of
**NIFTY/German** and **FairGB/Bail**, and check that the resolved configuration equals the frozen
run's. `published_config._read` now raises on a missing source file, so a broken checkout fails
loudly instead of resolving a different configuration.

## D. Licensing route per file group

| group | route |
|---|---|
| runners, `models/algorithms/*.py`, `figures/src/*.py`, `harness/provenance/` | (a) `.gitignore` exception — our own code, or short upstream excerpts the resolvers read |
| BeMap, FairGB (MIT, licence text present) | (a) exception, licence and copyright notice carried |
| FairGB vendored copy | (a) + patch: `eval.py` and `utils.py` carry local edits over the upstream copy |
| BIND, FairSIN, FMP, GEAR, SFG, FairGT (no licence text) | (b) fetch script pinning upstream URL + commit hash; local edits as patch files |
| FnRGNN | (b) via an anonymous mirror — upstream is `sybeam27/FnRGNN`, which de-anonymises the submission |
| `data/` | (b) fetch script; checksums already in `data_manifest.tsv` |

## E. Anonymisation

Remove from the export: git history, author names and e-mail addresses in file headers and
comments, and user paths (`/home/sypark/...`, the `FairGate` and `FairGNN-Eval` directory names
where they appear in hard-coded strings). Known hard-coded absolute paths to fix:
`harness/experiments/x28_tables/tcommon.py:30`, `harness/experiments/x28_figures/common.py:25`,
`harness/results/x30/bind_recovery/*.py`, and the `cd /home/sypark/workspace/FairGate` lines in
every `harness/experiments/*.sh`. Attach the `grep` output showing no identifying string remains.

## F. README must state

* wrapper fidelity against upstream, per `X6_PROVENANCE.md`: FairGNN 51.0 %, GNN 38.0 %
  (`local-unverified`); FairVGNN 83.4 %, NIFTY 86.7 % (`local-mirror-verified`).
* that re-training is nondeterministic and the paper's numbers are computed from the frozen
  per-unit results, not from a re-run.
* that `reproduction/` was captured after the runs finished, and the contemporaneous record
  covers only the PyTorch, PyG and DGL versions.
* that `pilot_tau.py:503`/`:506` seed per (seed, split) but `train()` at `:109` immediately
  reseeds with `seed` alone, so for FairGNN, NIFTY, FairVGNN and FairGB the effective seeding is
  split-independent. Left as it stands, because the frozen results depend on it.
