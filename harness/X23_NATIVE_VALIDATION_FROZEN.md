# Native validation: frozen (7 cells)

**Status: frozen.** No new experiment starts from this document. The next step,
a 2x2 protocol-factor decomposition, needs its own decision. FMP stays on hold.
Arm A is unchanged and was not re-run.

Framing:

* **Arm A** is *controlled attribution under a fixed audit protocol*.
* **Native validation** is *native-configuration validation with a fixed
  baseline reference*.
* The question: does the intervention attribution identified under a
  controlled protocol survive when the method is instantiated under its native
  training configuration?

## Canonical artefacts

| artefact | sha256 (prefix) |
|---|---|
| `harness/results/armB_native_NIFTY_german.csv` | commit `b2678a5` |
| `harness/results/armB_native_FairGB_german.csv` | commit `d408063` |
| `harness/results/armB_native_FairVGNN_german.csv` | commit `1c19d45` |
| `harness/results/armB_native_FairVGNN_bail.csv` | commit `9a71431` |
| `harness/results/armB_native_FairGB_bail.csv` | `e4a55582…` (X18) |
| `harness/results/armB_native_FairGB_credit.csv` | `fdb5d465…` (X21) |
| `harness/results/armB_native_FairVGNN_credit.csv` | `64605030…` (this commit) |
| `harness/results/armB_phase1_seven_cell_analysis.txt` | `46131226…`, **the frozen analysis output** |

The classification rule is X22 (commit `198bc0d`), fixed before the
FairVGNN/credit analysis.

**Bootstrap note.** The analyzer draws all cells' bootstrap replicates from one
generator seeded once (`default_rng(20260915)`), in cell order. Adding a cell
therefore shifts later cells' draws. Between the six- and seven-cell runs,
intervals moved by up to about 0.004, for example FairVGNN/german −ΔDP
[+0.084, +0.290] → [+0.088, +0.291], and no resolved state or class changed.
The seven-cell file above is the number of record. Earlier documents' intervals
are superseded only at that Monte Carlo scale.

## FairVGNN/credit native (the seventh cell)

### Run and contract

* **Design:** 6 splits × 5 runs.
* **Configuration:** the gated `fairvgnn_credit_native` adapter (X20), native
  H = 200, B at 200, M0 = f_mask no + weight_clip no (so classifier clipping
  off), M1 = official credit configuration, σ_c^BCE / σ_c^AUC, G_c.
* **Storage:** full RNG bundle replay with fixed ξ_eval, cell-level persistence
  to `/tmp`, SHA-256-verified copy.
* **Timing pilot (X20) excluded.** Split 20 / run 0 was recomputed.
* **Interruption.** The run was killed with the Claude Code session at 17:44
  after 4 persisted cells. `CellStore` validated the partial file, and the same
  command resumed detached at 17:52 (X22): 26 cells trained, 4 skipped, rc 0,
  finished 19:16.

| check | result |
|---|---|
| CellStore reload | 30 cells, 60 keys, 0 duplicates, both selectors per cell |
| design | native, `method_epochs` 200, `b_epochs` 200, `feature_normalize` 0, splits 20–25, runs 0–4 |
| RNG contract | `bundle-replay/xi-eval`; 30 distinct ξ_eval, each equal to `eval_rng_seed("credit", split, run)` and identical across a cell's selectors |
| runs distinct | 5 distinct M1 and M0 AUC in every split |
| finite / EO | 0 non-finite outcomes, EO defined in all rows |
| selected epochs | within H; M1 median 143.5 (6–195), M0 median 5 (0–78) |
| resume segments | 4 pre-interruption + 26 resumed cells, identical schema |

### Effects (σ_c^BCE, 10,000-replicate paired hierarchical bootstrap, s = sign stability)

| coordinate | τ_base: Arm A → fixed-ref | τ_int: Arm A → native | `\|b\|>\|i\|` | resolved | X22 class |
|---|---|---|---|---|---|
| −ΔDP | +0.038 [+0.014, +0.062] s0.67 → +0.105 [+0.080, +0.130] s1.00 | −0.000 [−0.040, +0.033] s0.53 → **+0.032 [+0.014, +0.050] s0.80** | T → T | F → **T** | **resolution gained** |
| −ΔEO | +0.046 → +0.102 | −0.014 [−0.062, +0.020] s0.53 → +0.020 [+0.004, +0.038] s0.70 | T → T | F → F | maintained |
| ΔAUC | +0.013 → +0.006 | −0.012 [−0.016, −0.009] s1.00 → −0.018 [−0.026, −0.011] s0.93 | T → **F** | T → T | relation changed |

Selector share `|D| > |τ_int|`: −ΔDP 0.53 → 0.93, −ΔEO 0.70 → 0.93, ΔAUC
0.37 → 0.30.

Reading, descriptive:

* Under its native credit loop, FairVGNN's interventions give a small,
  resolved DP improvement (about 0.03) at a resolved AUC cost (about 0.02).
  Under Arm A neither the DP effect nor its direction was distinguishable.
* **−ΔEO is classified "maintained" by the frozen rule, reported as is.** Its
  interval excludes 0 and |mean| ≥ 0.010, but sign stability is 0.70, below
  0.75, so the rule records it as unresolved.
* **The native σ_c choice is fragile in this cell.** The selector share rises
  to 0.93: in 28 of 30 cells, the σ_c^BCE-vs-σ_c^AUC difference exceeds the
  mean intervention effect.

### τ_base^fixed-ref on credit: an M0 change, not B redraw

B_200 drifts (mean |ΔDP| 0.0165, 0/30 bit-identical, same-cell range within
Arm A mean 0.0155, max 0.071), as on FairGB/credit (X21). Unlike FairGB/credit,
the τ_base shift is not at that scale:

| −ΔDP τ_base | value |
|---|---|
| Arm A | +0.038 |
| native M0 vs its own B_200 | +0.105 |
| native M0 vs Arm A's B_200 | +0.104 |

* **M0 itself changed:** mean DP 0.136 → 0.070 (AUC 0.731 → 0.727), and its
  σ_c^BCE epoch fell from median 118.5 to 5.
* **This is expected from the adapter.** The credit loop's encoder freeze during
  the discriminator phase is not an intervention flag, so it acts on M0 as well
  as M1 (X20).
* **The native change moved both arms.** M1's σ_c^BCE epoch rose from median 11
  to 143.5.

## The frozen transfer table (−ΔDP primary, X22 rule)

| method / dataset | native change | τ_int: controlled → native | resolved | `\|base\|>\|int\|` | transfer class |
|---|---|---|---|---|---|
| NIFTY / German | H 200→1000, drop rates restored | −0.006 → −0.151 | F → T | T → F | resolution gained |
| FairGB / German | H 200→1500, normalization off | −0.027 → +0.020 | F → F | T → F | relation changed |
| FairVGNN / German | normalization off | −0.003 → +0.190 | F → T | T → F | resolution gained |
| FairVGNN / Bail | H 200→300 | +0.007 → +0.003 | F → F | T → T | maintained |
| FairGB / Bail | H 200→1500 | −0.060 → −0.010 | F → F | F → T | relation changed |
| FairGB / Credit | H 200→2000 | +0.014 → +0.081 | F → T | T → F | resolution gained |
| FairVGNN / Credit | official credit training loop | −0.000 → +0.032 | F → T | T → T | resolution gained |

The six previously reported classifications are identical under X22.

Counts on −ΔDP: **maintained 1/7, relation changed 2/7, resolution gained
4/7**, with no resolution lost or stable reversal. Of the four gained
resolutions, three are fairness improvements (FairVGNN/german,
FairGB/credit, FairVGNN/credit) and one is a degradation (NIFTY/german).

Secondary coordinates, same rule:

* **−ΔEO:** maintained 3 (FairGB/german, FairVGNN/bail, FairVGNN/credit),
  resolution gained 3 (NIFTY/german, FairVGNN/german, FairGB/credit),
  resolution lost 1 (FairGB/bail).
* **ΔAUC:** maintained 4 (FairGB bail/credit/german, FairVGNN/bail), resolution
  gained 2 (NIFTY/german, FairVGNN/german), relation changed 1
  (FairVGNN/credit).

## FairVGNN across German, Bail, Credit (−ΔDP)

| dataset | native change | τ_base: A → fixed-ref | τ_int: A → native | `\|b\|>\|i\|` | resolved | class |
|---|---|---|---|---|---|---|
| German | normalization off | −0.155 → −0.068 | −0.003 → +0.190 | T → F | F → T | resolution gained |
| Bail | H 200→300 | −0.030 → −0.030 | +0.007 → +0.003 | T → T | F → F | maintained |
| Credit | official credit loop | +0.038 → +0.105 | −0.000 → +0.032 | T → T | F → T | resolution gained |

Descriptive only:

* Where the native change reaches the training path that M0 also uses
  (German preprocessing, the credit loop), τ_base moves. Where only the horizon
  moves and M0 is selected early (Bail), it does not.
* The native intervention effect on DP is resolved and positive on German and
  Credit, and unresolved on Bail.
* **Not claimed:** that preprocessing or loop changes matter more than horizon.
  The three cells differ in dataset and in which factor changed.

## Statement of record

*Controlled intervention attribution did not consistently transfer to the
methods' native training configurations.* Scope: 7 cells, 3 methods, 3
datasets. It is not generalized to fair GNNs.

The central message stays: *Package-level evidence does not identify the
claimed intervention effect, and intervention-level conclusions themselves
require explicit protocol specification and robustness checks.*
"Protocol-dependent" is not used as a standalone headline.

Why transfer fails is **not** answered here. Several native cells change more
than one factor. Separating the factors is the purpose of the proposed 2x2
protocol-factor experiment, pending decision.

## Repository hygiene

29 files under `bf/`, `etc/` and `utils_bf/` show as unstaged working-tree
deletions. Read-only diagnostics found them tracked since `4ee2ebc` (initial
commit) and unchanged in history; the directories are absent on disk, and no
code in `harness/`, `algorithms/` or `utils/` imports them. They were not created
by this research work and are left untouched: not staged, committed, restored
or deleted. Research commits use explicit paths only.
