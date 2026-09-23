# X29 — Coverage-extension protocol

**Scientific status.** X22–X27 are PRIMARY and FROZEN. This document defines a
**post-hoc coverage extension**. It is not a preregistration of the core study,
and it is not called one. The accurate statement is:

> the coverage-extension protocol was frozen before inspecting the newly added
> cells.

Nothing here modifies, reinterprets or re-analyses any frozen result. No frozen
CSV is overwritten. The extension is stored and reported separately.

Written 2026-09-17, before any new cell has been run.

## 1. What the audit found, and why the extension is small

The method and dataset rosters were restored from the repository read-only
(`harness/METHOD_INVENTORY.csv`, `harness/DATASET_INVENTORY.csv`,
`harness/COMPATIBILITY_MATRIX.csv`), not from memory.

* **Datasets: exactly 9**, from `utils/data.py:get_dataset` — german,
  bail (alias recidivism), credit, pokec_z, pokec_z_g, pokec_n, pokec_n_g, nba,
  income. All nine have split artifacts at 20–25 from the earlier e7 audit.
* **Audited systems: exactly 7**, from the roster closed in
  `X2_ADAPTER_CONTRACT.md` — FairGNN, NIFTY, FairVGNN, FairGB, FairSIN (core
  audit), FMP (mechanistic), GNN (baseline, audited but not counted as a
  method). An earlier draft roster in `X1_CONTROL_INVENTORY.md` listed FairGate,
  FairGT and BeMap instead; `X2_ADAPTER_CONTRACT.md` explicitly excludes those
  and is the later closure. The discrepancy is recorded, not smoothed over.

Of the 63 (system, dataset) cells, **four are new and eligible**. The reason so
few survive is that the repository's own frozen rules forbid inventing
configuration, and most cells have none:

* `native_config()` succeeds for exactly seven pairs — **precisely the frozen
  X23 native set**. There is therefore **no new VALID-NATIVE cell at all**, and
  §7 of this protocol has an empty queue. That is a finding, not a gap to fill.
* On the six non-core datasets, FairGB has no upstream branch, zip, `run.sh`
  line or `param.json` key; FairVGNN's pokec entries are byte-identical copies
  of credit's; NIFTY's pokec entries carry hidden dimensions but **not
  `sim_coeff`**, the intervention coefficient itself; and no roster method has
  any entry for nba or income.
* FairSIN is terminated by a frozen scientific decision
  (`X2_FAIRSIN_NOTE.md`, 2026-09-14) and is not revived here.
* FMP has no native λ and is already covered by X27 as a case study excluded
  from all aggregates.

## 2. The extension set — exactly four cells, fixed now

| # | method | dataset | baseline B | sensitive attribute |
|---|---|---|---|---|
| 1 | FairGNN | pokec_z | GNN | region |
| 2 | FairGNN | pokec_n | GNN | region |
| 3 | FairGNN | pokec_z_g | GNN | gender |
| 4 | FairGNN | pokec_n_g | GNN | gender |

These are chosen by eligibility, **not** by any result. Every VALID-CONTROLLED
cell that is not already frozen is in this list; none is omitted, and none may
be dropped later because of what it shows.

**Why these are eligible and the neighbouring cells are not.** FairGNN's
`param.json` entries are genuinely dataset-specific (german α=8 β=0.005, bail
α=2 β=0.05, credit α=1 β=0.005, pokec_z/_z_g α=4 β=0.01, pokec_n/_n_g α=8
β=0.01) and carry the same provenance grade as the entries already used in the
frozen core. The baseline GNN likewise has distinct per-dataset entries. No new
implementation, no new loader and no invented configuration is required.

**Verified before freezing, read-only:**

* Split contract (`harness/tests/test_split_contract.py`) — all four datasets,
  **splits 20–25, 24/24 pass**: `load_data` and `get_dataset` agree on
  train/val/test membership, labels, sensitive attribute and sensitive index.
* Loader contract — pokec_z N=67,796 F=277; pokec_n N=66,569 F=266; train,
  validation and test labels exactly {0,1} (the −1 unlabelled nodes are
  excluded from all three index sets); features finite; train and validation
  disjoint from test; sensitive attribute binary.
* No dense adjacency on the live path — `FairGNN.fit` and `GNN.__init__` both
  take `adj.coalesce().indices()`. pokec_z has 1.3 M edges ≈ 16 MB sparse; a
  dense matrix would be 18.4 GB and is never built.

## 3. Intervention definition

FairGNN's claimed fairness mechanism, from its own loss
(`algorithms/FairGNN.py:131`):

    G_loss = cls_loss + alpha * cov - beta * adv_loss

The two components are crossed (`closure=True`).

    M^{+I}   published configuration for that dataset (alpha, beta from param.json)
    M^{-I}   the same, with off = dict(alpha=0.0, beta=0.0)
    B        the GNN baseline at its own published configuration

M^{-I} differs from M^{+I} in nothing else: same data, same seed, same horizon,
same instrumentation.

**Component marginals are not run here.** X2_FROZEN_DECISIONS §2 licenses a
component study, but that is a different estimand from package-level coverage
and would not be comparable with the frozen 12 cells. It is out of scope for
X29 and recorded as a possible separate study.

## 4. Primary controlled protocol

Identical to the frozen core wherever the core fixes a choice.

| item | value |
|---|---|
| horizon | H = 200 for B, M^{-I} and M^{+I} (the armA protocol; the frozen core also used 200 regardless of each method's published horizon) |
| splits × runs | 6 splits (20–25) × 5 runs = 30 cells per (method, dataset) |
| pairing | paired by (split, run); B, M^{-I} and M^{+I} share the split, the seed and the RNG state |
| evaluator | the common `G_c`, decision `score > 0` |
| selectors | σ_c^BCE primary; σ_c^AUC as robustness |
| coordinates | ΔAUC and −ΔDP primary; −ΔEO secondary |
| feature normalisation | as `published()` dictates per method; FairGNN does not normalise pokec, GNN never normalises, so B and M agree |

No H=200 exception is needed: pokec has no published horizon for any roster
method, and the controlled protocol fixes the horizon by design.

**No method-specific tuning.** Nothing is re-tuned for these datasets.

## 5. Estimands

    tau_nonint = Y(M^{-I}) - Y(B)
    tau_I      = Y(M^{+I}) - Y(M^{-I})
    tau_pkg    = Y(M^{+I}) - Y(B)

with the identity `tau_pkg = tau_nonint + tau_I` checked per cell.

## 6. Uncertainty

Unchanged from the core. The frozen `bootstrap_armA.boot()`, 10,000 paired
hierarchical replicates: resample splits, then runs within each drawn split,
carrying each cell whole so that B, M^{-I}, M^{+I} and both selectors stay
paired inside a replicate.

**The resolved rule is unchanged and is not restated as new:** sign stability
≥ 0.75 **and** |mean| ≥ 0.010 **and** a 95 % interval excluding 0.

## 7. Native coverage — empty by construction

`native_config()` raises for FairGNN on every dataset, and succeeds only for the
seven pairs already frozen in X23. There is therefore nothing to run in this
section. The frozen X23 seven and any future native work are **not** pooled into
one preregistered sample.

## 8. Execution order and autonomy

Priority order, fixed now:

1. FairGNN / pokec_z  (region)
2. FairGNN / pokec_n  (region)
3. FairGNN / pokec_z_g (gender)
4. FairGNN / pokec_n_g (gender)

Run detached, one dataset at a time, on an idle GPU. Cell-level incremental
persistence (the X15 `CellStore` contract): each (protocol, method, dataset,
split, run, selector) row is appended and fsynced the moment both selector rows
of the cell exist; a duplicate key is an error, not an overwrite; an interrupted
job resumes and never re-runs a completed cell.

**Timing pilot, pre-registered.** pokec is ~68× german in node count, so cost is
not extrapolable from the frozen runs. The first (split, run) of pokec_z is
timed and interpreted **for cost only, never scientifically**. If the projection
for all four cells exceeds 24 h, the queue still runs in the order above and
simply reports which cells completed; cells are never reordered or dropped by
result.

## 9. STOP rules — per cell, hard

A cell stops immediately, is reported, and does **not** block the others:

* intervention +/− pairing impossible
* node alignment mismatch between B, M^{-I} and M^{+I}
* any test-set leakage
* non-finite outcome
* RNG pairing failure
* checkpoint restore fidelity failure
* evaluator mismatch
* instrumentation demonstrably changing training
* scientific configuration unclear for that cell

Repeat counts, tolerances and gate thresholds are **not** changed after seeing a
gate result in order to make it pass. A stopped cell is reported as unknown,
never as a negative finding.

## 10. Analysis and storage

Results are stored and reported in three separate views, never silently merged:

    CORE                  the frozen 12 controlled cells (X22) - untouched
    EXTENSION             these 4 new cells
    COMBINED DESCRIPTIVE  16 cells, descriptive only

Reported for the extension: eligible cell count, τ_I and τ_nonint distributions,
the count of cells with |τ_nonint| > |τ_I|, the count of resolved τ_I, the
utility–fairness quadrant, BCE/AUC qualitative agreement, and per-method and
per-dataset heterogeneity.

Descriptive proportions over the new cells may be reported **as counts of these
cells**. They are not generalised to a population probability over fair-GNN
research.

Outputs: `harness/results/x29/` with its own CSVs and checksum manifests. Figure
source CSVs follow the X28 pipeline conventions; no X22–X27 figure or table is
overwritten.

## 11. Do not

* modify X22–X27, or overwrite any frozen CSV
* select cells to run by their results
* force-port a method onto a dataset it does not publish
* invent a configuration, or treat a copied `param.json` entry as published
* develop a new fairness metric
* tune hyperparameters after seeing results
* inflate a tolerance to pass a failed gate
* substitute extension numbers for any core claim
