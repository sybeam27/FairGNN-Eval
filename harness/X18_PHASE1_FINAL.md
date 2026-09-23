# Arm B Phase 1: final report

**Arm A** is *controlled attribution under a fixed audit protocol*.
**Phase 1** is *native-configuration validation with a fixed baseline
reference*. The question it answers:

> Does the intervention attribution identified under a controlled protocol
> survive when the method is instantiated under its native training
> configuration?

The Phase 1 results are not read as showing Arm A was wrong. Arm A stays frozen
as the primary audit result.

## Data and contract

Five cells, each 6 splits (20-25) x 5 runs (0-4), 30 cells and 60 rows per
cell:

| cell | commit | native change vs Arm A |
|---|---|---|
| NIFTY/german | `b2678a5` | H 200 -> 1000, official drop rates restored |
| FairGB/german | `d408063` | H 200 -> 1500, normalization off |
| FairVGNN/german | `1c19d45` | normalization off (H unchanged at 200) |
| FairVGNN/bail | `9a71431` | H 200 -> 300 |
| FairGB/bail | this commit | H 200 -> 1500 (rerun after the /home write failure, X15) |

The contract held in every cell:

* alignment residual 0, no non-finite outcome, EO defined in all 300 rows;
* runs distinct within every split;
* protocol = native, method horizon = native H, B at 200;
* for FairVGNN, `bundle-replay/xi-eval` with 30 distinct evaluation seeds.

B_200 drifts from Arm A's B by a mean |dAUC| of at most 0.0001, so every
difference below comes from the method side. For FairGB/bail, `CellStore`
re-validates the persisted file (0 duplicate keys, every cell complete) and all
selected epochs fall within H.

Notation: `tau_int^native`, `tau_base^fixed-ref`, `tau_pkg^fixed-ref`. Never
`tau_base^native` or `tau_pkg^native`, since B has no native protocol.

Resolved means sign stability >= 0.75 and |mean| >= 0.010, **and** a 10,000
replicate paired hierarchical bootstrap 95% interval excluding 0, as frozen for
Arm A. Selection is sigma_c^BCE unless noted.

## Results on the primary coordinate, -dDP

| cell | Arm A tau_base / tau_int | native tau_base^fixed-ref / tau_int^native | `\|base\| > \|int\|` A -> N | tau_int resolved A -> N | selector share A -> N |
|---|---|---|---|---|---|
| NIFTY/german | +0.038 / -0.007 [-0.043, +0.031] | +0.119 / **-0.151 [-0.226, -0.081]** | T -> **F** | F -> **T** | 0.03 -> 0.63 |
| FairGB/german | -0.147 / -0.027 [-0.094, +0.031] | +0.013 / +0.020 [-0.021, +0.055] | T -> **F** | F -> F | 0.50 -> 0.67 |
| FairVGNN/german | -0.155 / -0.003 [-0.106, +0.106] | -0.068 / **+0.190 [+0.084, +0.290]** | T -> **F** | F -> **T** | 0.53 -> 0.40 |
| FairVGNN/bail | -0.030 / +0.007 [-0.003, +0.016] | -0.030 / +0.003 [-0.006, +0.012] | T -> T | F -> F | 0.63 -> 0.57 |
| FairGB/bail | -0.031 / -0.060 [-0.108, -0.015] | -0.032 / -0.010 [-0.036, +0.014] | F -> **T** | F -> F | 0.30 -> 0.43 |

Secondary coordinates:

* **-dEO** follows -dDP in NIFTY/german and FairVGNN/german (tau_int
  resolved under native: -0.122 and +0.149). FairVGNN/bail is unchanged.
* In FairGB/bail, Arm A's resolved tau_int (-0.065) becomes unresolved
  (+0.002).
* In FairGB/german the relation holds on -dEO, both unresolved.
* **dAUC:** `|base| > |int|` holds in all five cells under both arms.
  tau_int^native is resolved in all five, and changes sign in NIFTY/german
  (-0.031 -> +0.051) and FairVGNN/german (+0.013 -> -0.141).

## Reading

1. **The attribution structure survived the native configuration fully in one
   of five cells**, FairVGNN/bail, where only the horizon changed.
2. **The `|tau_base| > |tau_int|` relation changed in four of five cells**, in
   both directions:
   * three german cells lose it, because the native intervention effect grows
     (NIFTY, FairVGNN) or both effects shrink toward zero (FairGB);
   * FairGB/bail gains it, because Arm A's large intervention effect shrinks
     at H = 1500.
3. **The intervention effect became resolved in two cells, with opposite
   signs.**
   * NIFTY/german: native NIFTY **worsens** DP by about 0.15.
   * FairVGNN/german: native FairVGNN **improves** DP by about 0.19, at an AUC
     cost of about 0.14.
   * Neither was distinguishable from zero under Arm A.
4. **`tau_base^fixed-ref` moved only where M0's configuration moved.** In both
   bail cells, M0 is selected early (median epoch 70 for FairGB, 33 for
   FairVGNN), so extending the horizon leaves tau_base unchanged. The change in
   those cells is on M1's side. In the german cells the native preprocessing or
   horizon changes M0 itself.
5. **Descriptive finding:** *preprocessing alone can be sufficient to materially
   change intervention attribution.* In FairVGNN/german the horizon was
   unchanged and only normalization differed, and tau_int on -dDP moved from
   -0.003 (unresolved) to +0.190 (resolved).
6. **Not claimed:** that preprocessing matters more than horizon. FairVGNN/german
   and FairVGNN/bail differ in dataset as well as in which setting changed. Nor
   is any cause separated where two settings changed together (NIFTY/german,
   FairGB/german).

The overall statement: **attribution depends on which implementation and
training protocol is instantiated**, so native-configuration validation is
necessary. Arm A's controlled result and Phase 1's native result answer
different questions and are reported side by side.

## Consequences fixed in advance (X12, X16)

* **FairGB/credit: the trigger is satisfied.** It was recorded satisfied on
  FairGB/german before this result (X16). FairGB/bail independently shows a
  relation reversal too, so the analyzer's mechanical verdict is `RUNS`.
  * Interpretation note, unchanged:
    * both FairGB reversals are between effects that are unresolved on -dDP;
    * neither cell shows a stable direction reversal or a resolved-state flip
      on -dDP;
    * the trigger definition is not revised.
  * FairGB/credit is planned as the next confirmatory run. One real cell is
    timed first and the 6 x 5 cost reported; no choice is made from results.
* **FairVGNN/credit: the review condition is met** (X16). After FairGB/credit is
  scheduled, only a scope and cost estimate for a faithful `fairvgnn_credit.py`
  adapter follows. The shared wrapper is not modified.

## Limitations

* **B_200 is a fixed reference, not a native baseline.** tau_base^fixed-ref
  describes M0's position relative to that reference.
* **For NIFTY, sigma_c reads validation scores on NIFTY's augmented view**
  (X14). The native drop rates change that view, so NIFTY/german's native change
  includes a change of selector input.
* **NIFTY/german and FairGB/german changed two settings at once**, so their
  causes are not separated.
* **Five cells, two datasets.** No population statement is made.
