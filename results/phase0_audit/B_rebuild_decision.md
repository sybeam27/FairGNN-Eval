# Decision note — rebuilding the common baseline B

Written **before** any rebuilt B exists, and committed before the runs start. Nothing below is
changed after a result is seen.

## Why

The audit (T1) established that B on German is trained for 200 epochs while the configuration
resolver returns a published horizon of 1000, which `pilot_tau.py:389 → :463` never reads. The
resulting baseline is below chance (test AUC 0.4415; 0.6545 when the same code is run to H = 1000).
Every quantity referenced to B — τ_{B→−I}, τ_{B→+I}, Figure 1, Table 1, the six-split tables — rests
on that baseline. τ_{−I→+I} does not reference B and is unaffected.

## This is a bug fix, not a new design choice

The existing design already resolves a published configuration for B through `published("GNN", ds)`
(`harness/core/published_config.py:261-277`), horizon included. What the runner does is read the
configuration and **drop the horizon**: `pilot_tau.py:389` takes H from the CLI default (200) and
`:463` trains B with it, while `b_pub["horizon"]` is never read anywhere. Restoring the resolved
horizon therefore applies the design as written; it does not introduce a new rule, and it does not
change what "published" means for any dataset.

## What is resolved per dataset, before the rebuild

`published("GNN", <dataset>)`, probed 2026-09-24 (`harness/core/published_config.py:261-277`). The
horizon is whatever `_nifty_cmd` finds on a matching command line in the vendored upstream README
(`published_config.py:148-156`, `H = f.get("epochs")` at `:268`); if no line matches that dataset, the
function returns `{}` and the horizon stays `None`:

| dataset | b_pub horizon | resolved from | hidden / proj | lr | wd | feat. norm | provenance |
|---|---|---|---|---|---|---|---|
| german | **1000** | `harness/provenance/nifty_README.md:38` (`--epochs 1000 --model gcn --dataset german`), parsed at `published_config.py:148-156`, assigned `:268` | 16 / 16 | 1e-3 | 0.0 | off | third-party-benchmark |
| bail | none | no matching command line in the upstream README → `_nifty_cmd` returns `{}` (`:156`) | 8 / 4 | 1e-3 | 0.0 | off | local-unverified |
| credit | none | same | 32 / 4 | 1e-3 | 0.0 | off | local-unverified |
| income | none | same | CLI defaults | 1e-3 | 0.0 | off | local-unverified |
| pokec_z | none | same | 4 / 16 | 1e-3 | 0.0 | off | local-unverified |
| pokec_n | none | same | CLI defaults | 1e-3 | 0.0 | off | local-unverified |
| pokec_z_g | none | same | CLI defaults | 1e-3 | 0.0 | off | local-unverified |
| pokec_n_g | none | same | CLI defaults | 1e-3 | 0.0 | off | local-unverified |

**German is the only dataset with an ignored published horizon.** Everywhere else `horizon` is
`None`, so H = 200 was not overriding a published value; there was nothing to read. This is recorded
here so the rebuild is not later described as fixing more than it fixes.

## Rules, fixed in advance

1. **Configuration.** B uses exactly what `b_pub` resolves for that dataset: hidden, projection, lr,
   weight decay and feature normalization as in the table above, and **the resolved horizon where one
   exists** (German: 1000). Where `horizon` is `None` the existing H = 200 stands, because no
   published value exists to restore. No hyperparameter is chosen by looking at an outcome.
2. **One baseline per unit.** B is trained once per (dataset, split_id, run_id) and that single draw
   is shared by every method, every configuration and every native row that references it. This
   removes the per-method re-training documented in T2 (12 distinct baselines on Credit).
3. **The method arms are never re-trained.** M^{-I} and M^{+I} keep their frozen per-unit values. The
   rebuilt B is joined to them on (dataset, split_id, run_id), and only τ_{B→−I} and τ_{B→+I} are
   recomputed. τ_{−I→+I} is carried over unchanged — structurally, not by assertion. The rebuild
   pipeline asserts it is bit-identical to the frozen value and stops if it is not.
4. **Selector.** Unchanged: validation BCE is primary, validation AUC is the robustness replay.
5. **Three baselines, their roles fixed now, before any of them exists.**

   | name | configuration | role |
   |---|---|---|
   | **B_rep1** | rule 1 above: resolved b_pub horizon where one exists (German 1000), otherwise the controlled default H = 200 | **the baseline reported in the paper** |
   | **B_rep2** | identical rule, independently re-trained | sensitivity to baseline nondeterminism |
   | **B_H1000** | H = 1000 on **every** dataset setting; all other settings exactly as b_pub resolves them | sensitivity to baseline strength |

   The selection rule is the same for all three (validation BCE primary, validation AUC replay).
   Seeding is the existing one (`torch.manual_seed(seed)`, `seed = 27 + run`); no determinism flags
   are set, because they would not help — the sparse propagation used here has no deterministic CUDA
   kernel (`harness/experiments/e0_noise_floor.py:11-15`), and two of nine settings reproduce
   bit-exactly on that same path anyway (T12).
   **Execution order: `B_rep1` → `B_H1000` → `B_rep2`.**
6. **What the extra draws are for.** For each of the three baselines we recompute τ_{B→−I}, τ_{B→+I},
   the "surrounding package is larger" counts, the number of cells with τ_pkg < 0, the opposite-sign
   count and the family medians of |τ_{B→−I}|, and we list every cell whose classification differs
   from `B_rep1`. `B_rep2` measures how much of the headline is baseline nondeterminism; `B_H1000`
   measures how much of it is baseline strength.
7. **The same-process invariant, and why this combination is still valid.** The method arms keep
   their frozen per-unit values while B is trained in a separate process. Matching is at the level of
   the split and the seed assignment, not of the process. `pilot_tau.py:474-479` warns that a
   separate reproduction pass is not equivalent — but that warning is about **reproducing a frozen B**,
   which nondeterminism makes impossible even inside one process (T12: 16 of 30 units diverge on
   German, worst case DP 0.000 vs 0.336). It is not a validity condition on the contrast itself: the
   contrast needs the two arms to share the split, the seed and the evaluator, which they do. The
   extra variation this combination introduces is precisely what `B_rep2` measures and reports.
7. **Nothing is overwritten.** The frozen bundle stays as it is. The rebuild writes a parallel
   `results_v2/`, produced by the same `build_results.py` with its output path parameterised.
8. **Stop conditions.** The rebuild stops and reports, rather than continuing, if: τ_{−I→+I} changes
   for any cell; any unit fails to join on (dataset, split_id, run_id); or a cell ends with fewer
   than 30 units.

## Scope

* The 8 classification dataset settings used by the 36 primary cells, plus the 3 FnRGNN
  classification cells that reference the same B.
* Checked for the same defect, and rebuilt under the same rules if present: the MSE baseline of
  FnRGNN's released regression task (§1c) and FMP's own baseline stage (X31), which trains B with a
  different seeding call (`x31_fmp_baseline_run.py:50`).
* Optional, reported separately if run: a SAGE-backbone B for the SAGE cells (FairGB, SFG, GEAR), to
  show how much of the "surrounding package is larger" count depends on B being a GCN of a different
  capacity from the arm it is contrasted with (T1).

## Deliverables

* `results_v2/` — the full bundle rebuilt against `B_rep1`.
* `results_v2/B_rebuild_diff.md` — per dataset, B's absolute AUC/DP/EO before → after; the headline
  counts before → after (package larger 31/26/29, τ_pkg < 0 9/23/24, opposite sign 19/18/15, family
  median |τ_{B→−I}|); the list of cells whose "package larger" verdict changes; the list of tables
  and figures that contain a B-referenced number; and the change in the fixed-reference τ_B of the
  native rows.
* `results_v2/B_rebuild_diff.md` also carries, per dataset, B's absolute AUC/DP/EO for
  frozen → `B_rep1` → `B_rep2` → `B_H1000`; the four headline blocks for the same four baselines; the
  cells whose verdict differs from `B_rep1` under `B_rep2` and under `B_H1000`, separately; and the
  T9 table of Δ_attr intervals excluding zero, for reference.
