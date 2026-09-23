# Controlled → native transfer classification, fixed before FairVGNN/credit

Written and committed while the FairVGNN/credit native 6x5 run is in progress
and before its analysis. It defines how report item 6 is produced: whether a
cell's intervention conclusion is maintained from Arm A to the native
configuration. No existing claim or category is redefined. Every input is an
instrument already frozen for Arm A and Phase 1.

## Inputs, all pre-existing

For each (method, dataset, coordinate), under sigma_c^BCE, in each arm:

* **resolved:** sign stability >= 0.75 **and** |mean tau_int| >= 0.010
  **and** the 95% paired hierarchical bootstrap interval (10,000 replicates)
  excludes 0. The rule is unchanged from X12 and X14.
* **direction:** sign of mean tau_int.
* **relation:** |tau_base| > |tau_int|. On the native side tau_base is
  tau_base^fixed-ref.

A cell below the full 6 x 5 design gets no class.

## Classes, applied in this order (first match)

| class | condition |
|---|---|
| stable reversal | resolved in both arms with opposite signs |
| resolution gained | unresolved (Arm A) → resolved (native) |
| resolution lost | resolved (Arm A) → unresolved (native) |
| relation changed | resolved state unchanged, `\|tau_base\| > \|tau_int\|` flipped |
| maintained | resolved state, direction (when resolved), and relation all unchanged |

The classes are descriptive. They are reported separately per coordinate: -dDP
primary, -dEO secondary, and dAUC. They are not a significance test, and they
are not merged into one label per method.

## Where it is computed

`harness/experiments/analyze_armB_phase1.py` computes `transfer_class()` and a
method x dataset summary per coordinate. Each summary row shows:

* tau_base, Arm A -> fixed-ref;
* tau_int, Arm A -> native, with interval and sign stability;
* the relation change;
* the resolved change;
* the selector share;
* the transfer class.

The same code runs on every native cell, including the five already completed
and FairGB/credit, so all cells are classified by one rule.

## Interrupted run, resumed

The FairVGNN/credit run started 17:18 was killed at 17:44 when the Claude Code
session ended. Four complete cells (8 rows) had been persisted, and a fifth was
mid-training. `CellStore` validated the partial file (no incomplete row,
duplicate key, or half cell). The same command was relaunched detached, so the
four persisted cells are skipped and the rest computed. Each (split, run)
reseeds independently, so skipping does not change any later cell. The
pre-interruption log is kept as `armB_fairvgnn_credit.part1.log`.

## Next priority, as decided

After FairVGNN/credit, the native validation is consolidated first. The next
research priority is protocol-factor decomposition, not adding a model. FMP
stays on hold.
