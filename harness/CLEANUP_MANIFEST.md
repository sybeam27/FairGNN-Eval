# Cleanup manifest — X29 coverage-extension preparation

Written **before** anything is deleted, per the standing rule. Disk audit taken
2026-09-17 on the working tree at `/home/sypark/workspace/FairGate`.

## Disk state before cleanup

| scope | size |
|---|---|
| repository total | 1.5 G |
| `.git` | 26 M |
| `harness/` | 800 M |
| `harness/results/` | 610 M |
| — `harness/results/x25/` | 465 M |
| — `harness/results/x27/` | 86 M |
| — `harness/results/x26/` | 44 M |
| `/home` free | 341 G of 3.0 T (89 % used) |

Working tree: 2 modified tracked files, 164 untracked, 29 deleted-by-user
(the deleted set is not restored, not staged and not commented on further, per
the standing instruction).

## Approved for deletion

| path | size | tracked? | reason | canonical copy |
|---|---|---|---|---|
| 15 × `__pycache__/` directories | 2.1 M total | **no** (`git ls-files` → 0) | Python bytecode cache, regenerated on next import; already matched by `.gitignore:24` | the `.py` sources they were compiled from |
| 111 × `*.pyc` files | (counted inside the 2.1 M above) | **no** | same | same |

Locations: `harness/experiments`, `algorithms`, `utils`, `harness/core`,
`harness/adapters`, `BIND-main/implementations{,/GNNs}`, `FairSIN-main`,
`FMP-main`, `algorithms/FairGB`, `algorithms/FairGT`, `data`, repository root,
`harness/experiments/x28_tables`, `harness/experiments/x28_figures`.

This is the **entire** approved list. 2.1 M is a negligible reclaim; it is
included only because it is unambiguously safe. No wildcard `rm`, no
`git clean`, in any form.

## Explicitly NOT deleted, and why

| path | size | why it stays |
|---|---|---|
| `harness/results/x25/**/*.npz` (R10/R11 trajectories) | 465 M | trajectory artifacts still re-analysable; X25's decomposition and its G3 amendment are computed from them |
| `harness/results/x27/**/*.npz` (Pokec-z/n trajectories) | 86 M | same, for the FMP case study |
| `harness/results/x26/bail_trajectories/*.npz` | 44 M | same, for the FairGB selection-support replication |
| `harness/results/e7_raw/` (159 untracked CSVs) | 2.1 M | raw per-cell scientific output of the earlier e7 audit across all 9 datasets; the only record that those datasets were ever run |
| `harness/results/e10_raw/`, `e7_raw_normunified/`, `e7_raw_diverged/` | 2.2 M | same class |
| all `harness/results/*.csv` | — | frozen scientific results, bootstrap outputs, figure source data |
| all `*.sha256` manifests | — | provenance for the excluded `.npz` |
| `harness/*.md` (preregistrations, amendments, STOP reports, results) | — | the scientific record |
| `outputs/ours/` | — | FairGate run provenance, kept by `.gitignore:44` |
| `BeMap-main/`, `BIND-main/`, `FairGB-main/`, `FairSIN-main/`, `FMP-main/`, `algorithms/FairGT/` | — | vendored upstream repositories; `harness/external_repos.tsv` marks them **절대 수정 금지** |
| 2 modified tracked CSVs (`e8_h4.csv`, `pilot_tau.csv`) | — | pre-existing modifications, unrelated to this job; not touched |

## Duplicate-checkpoint sweep: none found

No duplicate checkpoint copies were found. The `.npz` trajectory files are one
copy each, each covered by a committed SHA-256 manifest, and the `/tmp` staging
copies from earlier runs are already gone.

## Reproducibility gate

Deleting bytecode caches cannot affect any metric, figure, bootstrap or table:
nothing reads `__pycache__`, and every entry is regenerated from a `.py` file
that remains in the tree. No artifact needed to reproduce X22–X27 or the X28
figures and tables is in the approved list.
