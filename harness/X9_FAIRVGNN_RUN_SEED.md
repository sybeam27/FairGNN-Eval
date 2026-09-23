# FairVGNN's five runs per split were not seeded apart

Found while computing run-level variation on the completed Arm A 6x5.

## Evidence

`pilot_tau.py` calls `FairVGNN.fit(...)` without a `seed` argument. The key
`"seed"` sits in the kwarg filter, but the configuration dict never contains
it, so nothing is passed. `fit` falls back to its default `seed=1`
(`algorithms/FairVGNN.py:1321`), and `seed_everything(count + args.seed)` at
`:1088` overwrites the harness's `torch.manual_seed(seed)` inside the loop.
Every run of every split trains FairVGNN from the same seed.

Distinct outcomes across the 5 runs within a split (sigma_c^BCE, M1 AUC):

| dataset | FairVGNN | FairGB | FairGNN | NIFTY | B |
|---|---|---|---|---|---|
| german | 1,1,1,1,1,1 | 5 each | 5 each | 5 each | 5 each |
| bail   | 1,1,1,1,2,1 | 5 each | 5 each | 5 each | 5 each |
| credit | 5,5,5,4,5,5 | 5 each | 5 each | 5 each | 5 each |

On german the runs are bit-identical. On bail and credit the only thing
separating them is GPU nondeterminism, which grows with graph size. None of
it is variation in the seed.

FairGB receives `seed=seed` explicitly. FairGNN, NIFTY and GNN do no internal
reseeding, so the harness seed applies to them.

## What stays valid

* M0/M1 pairing: both arms of a FairVGNN cell used seed 1, so each tau_int is
  a correctly paired contrast. This is not an M0/M1 wiring fault.
* Every other method's run-level variation.
* All split-level results.

## What is affected

* FairVGNN has effectively 6 independent cells per dataset, not 30.
* Its run-level SD (0.0000 on german and bail) is an artifact, and so is any
  split/run ratio built on it.
* "runs same sign 5/5" for FairVGNN says nothing.
* The hierarchical bootstrap for FairVGNN resamples copies within a split, so
  its interval is in effect a split-only bootstrap and understates run-level
  uncertainty.
* The earlier harness-uniform german 6x5 (the configuration-sensitivity arm)
  has the same defect.

## Not fixed here

A fix means re-running FairVGNN with `seed=seed` on all three datasets, which
changes the frozen Arm A protocol. Recorded and reported; waiting for a
decision.
