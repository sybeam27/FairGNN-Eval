# Dataset set for the expansion, fixed before any result

## Decision

    german   (kept as run: 6 splits x 5 runs)
    bail     (a.k.a. recidivism -- one alias, one loader, one file)
    credit

Pokec-z is **excluded**. It is not in the officially supported set of FairGB.

## Why the intersection is three

The binding constraint is FairGB, the only method in the roster with a real
upstream repository in this checkout.

* `FairGB-main/data_utils.py:260-279` branches on exactly `credit`, `bail`,
  `german` and has no `else`; any other name raises `UnboundLocalError` on the
  undefined `load`.
* `FairGB-main/dataset/` contains only `bail.zip`, `credit.zip`, `german.zip`.
* `FairGB-main/run.sh` has exactly three lines, one per dataset.
* `FairGB-main/README.md:28-36` states three datasets and tables only those.

FairVGNN's own vendored loader agrees independently: the commented-out upstream
`get_dataset` at `algorithms/FairVGNN.py:1003-1008` lists `credit`, `bail`,
`german` and nothing else, and the harness's tuned branch
(`utils/train_baselines.py:472-493`) covers only those three, with every other
dataset falling through to an untuned `else` at `:495`.

Running FairGB or FairVGNN on Pokec-z would therefore mean inventing a
configuration neither method publishes. The instruction is explicit that no
dataset is force-ported and no approximate configuration is constructed, so
Pokec-z is out.

`bail == recidivism`: `utils/data.py:482` accepts both names for one loader
over `data/bail/bail.csv`; `utils/train_baselines.py:814` maps `bail` to the
`recidivism` key when reading `param.json`.

## What loads but is not officially configured (recorded, not used)

These run in the harness but have no upstream configuration for at least one
roster method, so they are not admitted:

* FairGB on pokec_z, pokec_n, pokec_z_g, pokec_n_g, nba, income -- no upstream
  branch, no zip, no run.sh line, no README row, and no `FairGB` key in
  `utils/param.json` at all.
* FairVGNN on the same six -- official set is credit/bail/german; the
  `param.json` pokec entries are byte-identical to credit's, i.e. copied.
* All five methods on `nba` and `income` -- no `param.json` entry for GNN,
  FairGNN, NIFTY or FairVGNN.

## Provenance of the published configuration, per method

| method | provenance | status |
|---|---|---|
| FairGB | `FairGB-main/run.sh:2,5,8`, README, upstream repo present | **resolved** |
| GNN | `utils/param.json` only (local) | **provenance unresolved** |
| FairGNN | `utils/param.json` + `train_baselines.py:511-514` (local) | **provenance unresolved** |
| NIFTY | `utils/param.json` only (local) | **provenance unresolved** |
| FairVGNN | `utils/param.json` + `train_baselines.py:472-493` (local), corroborated only by the commented-out upstream loader | **provenance unresolved** |

No upstream repository for FairGNN, NIFTY or FairVGNN exists in this checkout
(`harness/external_repos.tsv` vendors FairGT, BIND, FairGB, FairSIN only). For
those three, "the published configuration" cannot be verified against an
upstream artifact from here. Per-dataset values are recorded as the harness's
local record and labelled `provenance unresolved`; nothing is estimated.

This affects the interpretation of `tau_pkg^pub` only. `tau_int` is unaffected:
M0 and M1 share whatever configuration is used, and differ only in the claimed
intervention.

## Contract checks passed before admission

`harness/tests/test_split_contract.py` -- `load_data` and `get_dataset` agree on
train/val/test membership, labels, sensitive attribute and sensitive column
index:

* german, splits 20-25: OK
* bail, splits 20-22: OK
* credit, splits 20-22: OK

## Budget, fixed before any result

All added datasets run at 3 splits x 5 runs. No dataset is expanded to 6x5
because its result looked good; expansion, if it happens, applies to the whole
fixed set on a compute-budget decision recorded before the results are read.
Repeats are run-level stochastic variation, never seed uncertainty.
