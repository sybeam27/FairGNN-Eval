# Phase 1 decisions recorded before the FairGB/bail rerun result exists

Written while the FairGB/bail native rerun is still running. Nothing here uses
its numbers.

## FairGB/credit: the X12 trigger is satisfied

X12 fixed the rule before any Arm B result: FairGB/credit runs if FairGB/german
**or** FairGB/bail shows at least one of

1. a reversal of `|tau_base| > |tau_int|` relative to Arm A,
2. a stable reversal of the fairness-axis `tau_int` direction,
3. a material resolved/unresolved flip on `-dDP`.

FairGB/german (committed `d408063`) shows condition 1 on `-dDP`:

| | Arm A | native (fixed-ref) |
|---|---|---|
| `tau_base` | -0.1465 [-0.2455, -0.0359] | +0.0127 [-0.0663, +0.0898] |
| `tau_int` | -0.0273 [-0.0952, +0.0340], unresolved | +0.0204 [-0.0219, +0.0560], unresolved |
| `\|tau_base\| > \|tau_int\|` | true | **false** |

Because the rule is "any one cell", the trigger is **recorded as satisfied**
regardless of what FairGB/bail shows.

**Interpretation note, recorded rather than acted on.** The reversal is a
magnitude comparison between two unresolved, near-zero effects (0.013 vs
0.020), and both intervals include 0. Conditions 2 and 3 did not occur. This
does not change the trigger: its definition is not revised after seeing the
result.

**Consequence.** FairGB/credit native validation is planned as the next
confirmatory run, after the Phase 1 final report. Before it runs, the actual
runtime of one cell is measured again and the 6 x 5 cost is reported. That
measurement is a budget estimate only; no choice is made from results.

## FairVGNN/credit: the review condition is met, nothing implemented

The condition fixed earlier: consider an official-credit adapter if Phase 1
shows FairVGNN's attribution is sensitive to its native configuration.

FairVGNN/german (committed `1c19d45`) changed only preprocessing, at the same
horizon (200), and `tau_int` on `-dDP` moved from -0.003 (unresolved) to
+0.190 [+0.086, +0.291] (resolved). **The condition is recorded as met.**

Nothing is implemented now. After the Phase 1 final report and the FairGB/credit
scheduling, the next step is a scope and cost estimate only: an adapter that
faithfully mirrors the official `fairvgnn_credit.py` (encoder frozen during the
generator phase, no encoder step there, classifier clipping with `clip_c`). The
shared wrapper `algorithms/FairVGNN.py` is not bent to approximate it.

## Wording fixed for the Phase 1 report

* **Arm A:** *controlled attribution under a fixed audit protocol*
* **Phase 1:** *native-configuration validation with a fixed baseline
  reference*
* **Question:** *Does the intervention attribution identified under a
  controlled protocol survive when the method is instantiated under its native
  training configuration?*
* The Phase 1 results are not described as showing Arm A was wrong.
* **Descriptive finding, from FairVGNN/german:** *preprocessing alone can be
  sufficient to materially change intervention attribution*. Its horizon was
  unchanged and only normalization differed.
* Not claimed: comparing FairVGNN/german with FairVGNN/bail does not support
  "preprocessing matters more than horizon". The two cells differ in dataset
  as well as in which setting changed.

No method, metric, selector or dataset is added.
