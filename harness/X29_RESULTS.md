# X29 results — coverage extension

**Scientific status.** X22–X27 remain PRIMARY and FROZEN and are untouched. This
is a **post-hoc coverage extension**. The accurate statement about its timing is:

> the coverage-extension protocol was frozen before inspecting the newly added
> cells.

Protocol: `harness/X29_COVERAGE_EXTENSION_PROTOCOL.md`, frozen at `811b1a3`
(matrix repair `3d84547`). Analyzer committed at `c204b04` / `425a906` and the
figure emitter at `180dc84`, all **before** any extension number existed.

## 1. What ran

Four cells, exactly the set frozen in the protocol — every VALID-CONTROLLED cell
that was not already in the core, and no others:

| cell | dataset | sensitive attribute | units | GPU |
|---|---|---|---|---|
| 1 | pokec_z | region | 30 | 0 |
| 2 | pokec_n | region | 30 | 1 |
| 3 | pokec_z_g | gender | 30 | 2 |
| 4 | pokec_n_g | gender | 30 | 4, then 2 |

FairGNN as M, GNN as baseline B, H = 200, 6 splits (20–25) × 5 runs, σ_c^BCE
primary and σ_c^AUC as robustness. **120 units, 240 rows.**

**Timing.** The pre-registered pilot measured 24 s for the first complete cell,
projecting well under the 24 h budget, so the full queue ran: 6m28s, 9m00s,
9m17s, and 4 units + 7m38s on resume — about 35 minutes in total.

**One interruption, recorded rather than hidden.** Cell 4 began on GPU 4; the
instruction to use GPU 2 only arrived mid-run, so the job was stopped and
relaunched on GPU 2. The CellStore resume contract behaved exactly as designed:
4 units had been written, all 4 had both selector rows, none was half-written,
and the relaunch skipped precisely those 4 and computed the remaining 26. The
earlier three cells had already finished on GPUs 0, 1 and 2 and were not re-run,
since a re-run under CUDA non-determinism would produce slightly different
numbers for no scientific gain.

**Contracts: clean.** 0 duplicate keys, 0 non-finite outcomes, 0 undefined EO,
0 half-written units. The registered identity `tau_pkg = tau_nonint + tau_I`
holds to |residual| ≤ 1.4e-17 on all three coordinates.

## 2. The extension result

**τ_I on −ΔDP, σ_c^BCE — 0 of 4 cells resolved.**

| dataset | τ_I | 95% interval | sign | resolved |
|---|---|---|---|---|
| pokec_z | −0.0180 | [−0.0343, −0.0058] | 0.70 | no |
| pokec_z_g | +0.0012 | [−0.0062, +0.0089] | 0.53 | no |
| pokec_n | −0.0039 | [−0.0157, +0.0070] | 0.53 | no |
| pokec_n_g | −0.0049 | [−0.0109, +0.0002] | 0.63 | no |

**τ_nonint on −ΔDP — 1 of 4 resolved.**

| dataset | τ_nonint | 95% interval | sign | resolved |
|---|---|---|---|---|
| pokec_z | −0.0364 | [−0.0581, −0.0161] | 0.87 | **yes** |
| pokec_z_g | −0.0023 | [−0.0121, +0.0078] | 0.53 | no |
| pokec_n | −0.0038 | [−0.0123, +0.0068] | 0.67 | no |
| pokec_n_g | −0.0044 | [−0.0114, +0.0025] | 0.60 | no |

`|τ_nonint| > |τ_I|` in **2 of 4** cells. In the two pokec_n variants the two
magnitudes are nearly equal (0.0038 vs 0.0039; 0.0044 vs 0.0049), so that count
is decided by differences far below what these intervals can separate, and
should not be read as a finding.

**τ_I on ΔAUC — 0 of 4 resolved, but worth stating precisely.** All four are
negative and close: −0.0070, −0.0070, −0.0059, −0.0059. On pokec_n the interval
[−0.0084, −0.0037] excludes zero with sign stability 0.93, and on pokec_n_g
[−0.0083, −0.0036] with 0.90 — yet both are unresolved because |mean| < 0.010,
the frozen near-zero threshold. The honest reading: **a consistently signed but
small utility cost that the pre-registered rule declines to resolve.** The rule
was not adjusted to change that.

**τ_I on −ΔEO (secondary) — 0 of 4 resolved**: +0.0000, −0.0084, −0.0136,
+0.0091, every interval covering zero.

**Selector robustness.** Sign differs between σ_c^BCE and σ_c^AUC in 2 of 4
cells (pokec_n: −0.0039 vs +0.0027; pokec_z_g: +0.0012 vs −0.0051). Both flips
occur among unresolved values within ±0.005 of zero, and the paired difference
`D_selector` is unresolved in all four cells. So this is not evidence that the
selector changes a conclusion — there is no resolved conclusion here for it to
change.

## 3. CORE is unchanged, and was re-derived as a check

The core view is reported separately and was not modified:
**10 of 12** cells with `|τ_nonint| > |τ_I|`, **1 of 12** with a resolved τ_I
(NIFTY/credit). Both numbers were reproduced by code written after the frozen
record, and a direct comparison against `armA_final_bootstrap.txt` found **no
drift across 288 frozen values** (worst difference 5.0e-05, the frozen file's
printing precision).

**One caveat on CORE −ΔEO.** No frozen Arm A artifact bootstraps −ΔEO for the 12
core cells; EO exists frozen only for the seven native cells, through armB. The
CORE EO figures in `x29_summary.csv` are therefore **computed here for the first
time and are not part of the frozen record**. Where they can be compared they
agree on substance — FairGB/bail reads −0.0654 [−0.1116, −0.0230] s0.77 against
armB's Arm A-side −0.0654 [−0.1104, −0.0225] s0.77, the interval differing in
the third decimal because armB is a separate script with its own RNG stream.

## 4. COMBINED DESCRIPTIVE

16 cells: `|τ_nonint| > |τ_I|` in **12 of 16**, resolved τ_I in **1 of 16**.
Descriptive only, and kept in its own view so no core claim is restated through
it.

## 5. What this does and does not support

* **It does not change any core claim.** No frozen value was modified, and the
  extension is stored and reported separately.
* **0 of 4 resolved is an absence of a resolved effect, not a demonstrated
  zero.** With 30 units per cell these intervals are simply wide relative to the
  effects.
* **These are counts of these cells**, not a rate in any population of fair-GNN
  research.
* **The four cells are not four independent datasets.** pokec_z and pokec_z_g
  are the same graph with region and gender as the sensitive attribute;
  likewise pokec_n and pokec_n_g. The extension covers **two graphs × two
  sensitive attributes**, for **one method**.
* **The native queue was empty by construction**, not by omission:
  `native_config()` succeeds for exactly the seven pairs already frozen in X23.

## 6. What was deliberately not run

* **NIFTY, FairVGNN and FairGB on pokec** — UNSUPPORTED. FairVGNN's pokec
  entries are byte-identical copies of credit's; NIFTY has no per-dataset
  `sim_coeff`, so its intervention strength would have been ours to choose;
  FairGB has no upstream branch, zip, `run.sh` line or `param.json` key at all.
* **Anything on nba or income** — no roster method has a `param.json` entry, so
  `published()` returns only CLI defaults with no method hyperparameter.
* **FairSIN** — terminated by the frozen decision in `X2_FAIRSIN_NOTE.md`
  (2026-09-14) because its published configuration collapses to constant
  prediction natively. Not revived here.
* **FairGNN component marginals** (α-only, β-only) — licensed by
  X2_FROZEN_DECISIONS §2 as a *component study*, which is a different estimand
  from package-level coverage and would not be comparable with the frozen 12.
  Out of scope for X29 and recorded as a possible separate study.

## 7. Outputs

`harness/results/x29/`: `x29_FairGNN_pokec_{z,n,z_g,n_g}.csv` (240 rows),
`x29_analysis.txt`, `x29_summary.csv`, `x29_cell_table.csv`, per-run logs, and
`x29_artifacts.sha256`. Figure source CSVs under
`harness/results/x29/figure_sources/` (five files).

Protocol §12's sixth figure — observed protocol span over native-available
cells — is **not** emitted: the extension adds no native cell, so an empty file
would imply a measurement that does not exist. The frozen cells are already
covered by `harness/results/figures/paper/source_data/figG_protocol_span_source.csv`.
