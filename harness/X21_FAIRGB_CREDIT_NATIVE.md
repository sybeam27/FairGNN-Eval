# FairGB/credit native validation, and FairGB across three datasets

FairGB/credit was triggered under the rule fixed in X12, recorded as satisfied
in X16, and approved (X20).

## Run and contract

FairGB/credit native (13:49–16:37):

* **Design:** 6 splits (20–25) × 5 runs (0–4).
* **Configuration:** H = 2000, upstream normalization on (credit), B at
  H = 200, isolation-preserving M0 (CAL off + CNM off), σ_c^BCE primary and
  σ_c^AUC robustness, G_c.
* **Storage:** cell-level persistence to `/tmp`, then copied to
  `harness/results/armB_native_FairGB_credit.csv`. The SHA-256 matches the
  recorded `fdb5d465…`.
* **The earlier timing cell (X19) is not in this analysis.** Split 20 / run 0
  was recomputed as an ordinary cell into a fresh file.

| check | result |
|---|---|
| CellStore reload | 30 cells, 60 keys, 0 duplicate keys, every cell has both selectors |
| design | 6 × 5, protocol native, `method_epochs` 2000, `b_epochs` 200, `feature_normalize` 1 |
| runs distinct | M1 and M0 AUC take 5 distinct values in every split |
| finite / EO | 0 non-finite outcomes; EO defined in all 60 rows |
| selected epochs within H | yes; M1 median 848 (496–1956), M0 median 1022 (124–1947) |
| exact-zero contrasts | 0 |

## Result on −ΔDP (σ_c^BCE, 10,000-replicate paired hierarchical bootstrap)

| | Arm A (H = 200) | native (H = 2000) |
|---|---|---|
| τ_base (native: fixed-ref) | +0.049 [+0.018, +0.086] | +0.060 [+0.021, +0.098] |
| τ_int | +0.014 [−0.017, +0.049], unresolved | **+0.081 [+0.043, +0.120], resolved** (sign stability 0.87) |
| `\|τ_base\| > \|τ_int\|` | true | **false** |
| selector share `\|D\| > \|τ_int\|` | 0.53 | 0.23 |

* **−ΔEO** follows −ΔDP: τ_int^native +0.068 [+0.029, +0.109], resolved, and
  the relation flips from true to false.
* **ΔAUC:** `|τ_base| > |τ_int|` holds in both arms. τ_int^native is +0.003,
  unresolved.

In words: at the native horizon, FairGB's claimed interventions (CAL + CNM)
have a distinguishable fairness-improving effect on credit (DP about 0.08
lower) with no distinguishable accuracy cost. That effect exceeds the
rest-of-package component on DP. Under Arm A's H = 200 it was not
distinguishable from zero.

### B_200 on credit: a redraw at GPU noise, disclosed

Unlike every German and Bail cell (drift ≈ 0), the in-process B_200 of the
native run differs from Arm A's B_200 on credit: mean |ΔAUC| 0.0095, mean
|ΔDP| 0.0209, 0/30 cells bit-identical, same selected epoch in 11/30.

This is not a new effect.

* **Within Arm A itself**, the "same" B_200 for one (split, run), trained in
  different method processes, already differs in **29/30 credit cells**: mean
  DP range 0.016, max 0.071. This is the X3 phenomenon, GPU nondeterminism
  compounding over 200 epochs, and it is largest on the largest graph.
* The signed drift varies by split with no consistent direction.

What it does and does not affect:

* **τ_int never involves B**, so it is unaffected.
* **The structural conclusion survives either reference.**

  | −ΔDP τ_base | value |
  |---|---|
  | native M0 against its own B_200 | +0.060 |
  | native M0 against Arm A's B | +0.065 |
  | Arm A | +0.049 |

  In both native readings |τ_base| stays below |τ_int^native| = +0.081, so the
  relation reversal holds.
* **On credit, the magnitude of τ_base^fixed-ref − τ_base^ArmA is not a pure M0
  change.** It carries B redraw noise of the same order.

## FairGB across German, Bail, Credit (−ΔDP)

| dataset | native change | τ_base: Arm A → fixed-ref | τ_int: Arm A → native | `\|base\|>\|int\|` A → N | τ_int resolved A → N | selector share A → N |
|---|---|---|---|---|---|---|
| german | H 200→1500, normalization off | −0.147 → +0.013 | −0.027 → +0.020 | T → **F** | F → F | 0.50 → 0.67 |
| bail | H 200→1500 | −0.031 → −0.032 | −0.060 → −0.010 | F → **T** | F → F | 0.30 → 0.43 |
| credit | H 200→2000 | +0.049 → +0.060 | +0.014 → **+0.081** | T → **F** | F → **T** | 0.53 → 0.23 |

How the structure depends on the dataset, descriptively:

* **The relation changed on all three datasets, but not in one direction.**
  Two datasets lose the dominance of the rest-of-package component and one
  gains it. The mechanisms also differ:
  * **German:** τ_base moves toward zero, because the native preprocessing
    changes M0.
  * **Bail:** τ_int shrinks.
  * **Credit:** τ_int grows into a resolved effect.
* **A resolved FairGB intervention effect on DP appears only on credit.** On
  German and Bail it is unresolved in both arms.
* **τ_base^fixed-ref moves substantially only on German,** where native
  preprocessing differs. On Bail and Credit the preprocessing is unchanged, so
  M0's position relative to B stays close; the Credit residual is at B-redraw
  scale.
* **Selector sensitivity does not move in one direction** across datasets.

No cause is separated. German changed horizon and preprocessing together, and
three datasets are not a sample from which a dataset-level law follows.

## Phase 1, now six cells (−ΔDP)

| cell | `\|base\|>\|int\|` A → N | τ_int resolved A → N |
|---|---|---|
| NIFTY/german | T → F | F → T (−0.151) |
| FairGB/german | T → F | F → F |
| FairVGNN/german | T → F | F → T (+0.190) |
| FairVGNN/bail | T → T | F → F |
| FairGB/bail | F → T | F → F |
| FairGB/credit | T → F | F → T (+0.081) |

* The structure survived intact in **1 of 6** cells.
* The relation changed in **5 of 6**.
* The intervention effect became resolved in **3 of 6**: two fairness
  improvements (FairVGNN/german, FairGB/credit) and one fairness degradation
  (NIFTY/german).

Wording, unchanged from the decision of record:

* *Controlled intervention attribution did not consistently transfer to the
  methods' native training configurations*, stated for these six cells on three
  datasets and not generalized to fair GNNs.
* The central message remains: *Package-level evidence does not identify the
  claimed intervention effect, and intervention-level conclusions themselves
  require explicit protocol specification and robustness checks.*

Arm A is unchanged and not re-run.
