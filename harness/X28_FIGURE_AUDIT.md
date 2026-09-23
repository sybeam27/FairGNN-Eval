# X28: paper figure candidates — audit

**No training was run.** Every number comes from a frozen artifact: the Arm A
bootstrap table, the Arm B seven-cell analysis, and the X25/X26/X27 summary
CSVs. No frozen value was edited, no threshold or classification was redefined,
and nothing was re-rounded for display. The only computation is derived
plotting coordinates (sums and differences of frozen quantities), each of which
is checked against the identity it is supposed to satisfy.

Outputs: `harness/results/figures/paper/` (`.pdf` + `.png` per figure) and
`harness/results/figures/paper/source_data/` (`*_source.csv`, exactly the numbers
plotted). Scripts: `harness/experiments/x28_figures/`.

## 1. Automated consistency checks

Every script re-derives what it plots and refuses to render on a mismatch.
**401 checks, all passed.**

| figure | checks | worst \|difference\| | what was verified against what |
|---|---|---|---|
| A / A2 | 303 | 5.00e-05 | all 96 frozen Arm A bootstrap entries re-derived by replaying `bootstrap_armA` with its own RNG stream (same seed, same dataset→method order) |
| B / B2 | 37 | 4.92e-05 | both endpoint means recomputed from the Arm A and `armB_native_*` per-cell CSVs; 30 cells asserted per native cell |
| C / C2 | 9 | 1.11e-16 | identities `S = full − cap`, `T = cap − h200`, `H_shift = S + T` on the plotted means |
| D | 5 | 0.00e+00 | every grid epoch present in the frozen summary; selected-epoch row count |
| E | 20 | 1.04e-16 | identity `τ_total = τ_prop + τ_fair` on the plotted means, both coordinates |
| F | 25 | 4.82e-05 | each controlled point against frozen `tau_int`, and `bce − auc` against the frozen `D_selector` entry |
| G | 2 | 0.00e+00 | seven cells carry a native protocol point |

The 5e-05 residuals are the frozen files' 4-decimal printing precision, not
disagreement. In addition, every figure asserts that all plotted coordinates,
interval ends included, lie inside the drawn axes — the check that would have
caught the X27 autoscale bug.

## 2. Figures

### Figure A — controlled package vs intervention (`figA_effect_space`)

* **Question (RQ1).** Under the controlled protocol, how large is the fairness
  intervention beside the rest of the package it ships in?
* **Source.** `armA_final_bootstrap.txt`; re-derived from the eight `armA_*.csv`.
* **Plotted.** Per cell, two vectors from the origin: `tau_base^audit`
  (= Y(M_off) − Y(B)) and `tau_int` (= Y(M_on) − Y(M_off)), x = ΔAUC,
  y = −ΔDP, with 95% interval crosses at both endpoints. 12 cells, 3 panels.
* **Encoding.** Method = colour **and** marker; vector kind = line style and
  weight (grey dashed vs coloured solid); fill = "95% interval excludes 0".
  Quadrant tags on the leftmost panel only. Zero lines drawn.
* **Main paper: HIGH.** It is the figure that states the paper's premise.
* **Redundancy.** Replaces reading two columns of the Arm A table; the table
  still carries sign stability, which this figure does not show.
* **Misreading risk.** (a) The three panels have **different axis scales** — a
  German arrow is not comparable in length to a credit arrow; the subtitle says
  so. (b) Fill here means interval-excludes-0, **not** the frozen resolved rule,
  because no Arm A artifact carries a combined resolved flag for these 12 cells;
  the legend says "hollow: 95% interval covers 0" and never says "unresolved".
  (c) Two vectors from a shared origin are components of one package, not
  independent causes.
* **Caption message.** In the controlled protocol the non-intervention package
  moves utility and fairness more than the intervention does in most cells.

### Figure A2 — per-cell small multiples (`figA2_effect_space_small_multiples`)

Same numbers, 12 panels, endpoints only. **Appendix.** Removes the shared-origin
clutter and the cross-cell scale confusion; too large for the main text.

### Figure B — controlled → native shift (`figB_native_shift`)

* **Question (RQ2).** How does the intervention contrast move when the protocol
  changes from controlled to native?
* **Source.** `armB_phase1_seven_cell_analysis.txt`; means re-derived from the
  per-cell CSVs.
* **Plotted.** Seven arrows, controlled endpoint → native endpoint, in the same
  effect space.
* **Encoding.** Method = colour + marker; tail = hollow circle, head = method
  marker; head fill = the frozen `tau_int resolved` flag on −ΔDP; labels hand-
  placed; legend outside the axes.
* **Main paper: HIGH.**
* **Redundancy.** Complements the X22 transfer-class table; the class strings
  are carried in the source CSV, not drawn.
* **Misreading risk.** (a) An arrow is **not** a trajectory — it joins two
  separate measurements. (b) Arrow length mixes the two coordinates; only −ΔDP
  carries an interval here. (c) The crossing metadata in the source CSV
  (`crossed_fairness_sign` etc.) is descriptive for caption writing and is not a
  new classification.
* **Caption message.** Changing the protocol moves the contrast substantially,
  and in several cells across the fairness sign.

### Figure B2 — paired slope (`figB2_native_shift_slope`)

−ΔDP only, with intervals at both ends and labels spread by leader lines.
**Appendix, or main text if B is cut** — it is the honest version for the one
coordinate that has intervals.

### Figure C — selection-support replication (`figC_selection_support`)

* **Question.** When the horizon is extended, is the shift carried by the longer
  trajectory or by the selector's access to later checkpoints?
* **Source.** `x25_summary.csv` (D1, the NIFTY native configuration) and
  `x26_bail_summary.csv`, σ_c^BCE.
* **Plotted.** Three points per panel (frozen H=200 → long run capped → long run
  full) with intervals, and T and S as staggered brackets.
* **Encoding.** Case = colour + marker; point role = marker shape; fill = frozen
  `resolved`; **shared y-axis** across the two panels.
* **Main paper: HIGH.** Two methods, opposite directions, one mechanism.
* **Redundancy.** Compresses X25 Table 1–2 and X26 §4 into one panel pair.
* **Misreading risk.** (a) The lines join three *conditions*, not epochs.
  (b) X26/credit is **absent because it stopped at a gate**, not because it was
  negative — the caption must say so. (c) The two panels are different methods
  *and* different datasets; neither alone is the comparison.
* **Caption message.** The same mechanism appears in both, with opposite sign:
  releasing the checkpoint cap makes NIFTY/German more harmful and FairGB/Bail
  less harmful; the prefix term T is ≈ 0 in both.

### Figure C2 — decomposition forest (`figC2_selection_support_forest`)

H_shift, S, T for both cases as six intervals. **Appendix** — or swap with C if
the reviewer wants the estimates rather than the path.

### Figure D — NIFTY fixed-epoch trajectory (`figD_nifty_trajectory`)

* **Source.** `x25_summary.csv` (fixed-epoch grid), `x25_selected_epochs.csv`.
* **Plotted.** D0 and D1 at the ten pre-registered epochs with bootstrap bands;
  strip below showing where σ_c^BCE selects per arm at full support.
* **Encoding.** Configuration = colour + marker + line style; no interpolation.
* **Main paper: MEDIUM → appendix.** It explains *why* C looks as it does, but
  C already carries the claim.
* **Misreading risk.** The joining lines are legibility only — no value between
  grid epochs was measured. The top panel involves **no checkpoint selection**;
  that is why it is identical in the BCE and AUC versions, and the subtitle now
  says so explicitly.
* **Caption message.** The effect drifts with training, and the selector's
  chosen epochs sit where it has drifted.
* **Supplementary.** `figD_supp_nifty_trajectory_auc_selector` — same curves,
  σ_c^AUC strip.

### Figure E — FMP component decomposition (`figE_fmp_components_ndp`)

* **Question.** Within FMP, which component moves fairness: graph propagation or
  the claimed fairness correction?
* **Source.** `x27_pokec_{z,n}_summary.csv`, σ_last.
* **Plotted.** Top row, true scale: F00 → F01(λ2) then F01 → F11(λ1, λ2), with
  F01 drawn as a real marker. Bottom row: the fairness component alone with its
  intervals.
* **Encoding.** Component = colour + marker + line style; fill = frozen
  `resolved`; λ2 by line style and a direct label.
* **Main paper: HIGH.**
* **Misreading risk.** (a) FMP has **no native λ** — every value is
  configuration-conditional, stated in the subtitle. (b) The bottom row is a
  zoom of the top row, not a separate measurement; showing it alone would
  exaggerate a component that is two orders of magnitude shorter. (c) All four
  fairness estimates are unresolved: absence of a resolved effect, not a
  demonstrated zero. (d) FMP is excluded from every benchmark aggregate.
* **Caption message.** Propagation carries the fairness change, and on DP it
  carries it in the harmful direction; the fairness correction adds no resolved
  change at any preregistered grid endpoint.

### Figure F — selector robustness (`figF_selector_scatter`)

* **Question.** Does the choice of checkpoint selector change the conclusion?
* **Source.** Arm A CSVs + `armB_native_*.csv`; controlled points verified
  against the frozen `D_selector`.
* **Plotted.** 19 cells: −ΔDP under σ_c^BCE against σ_c^AUC, with y = x.
* **Encoding.** Method = colour + marker; protocol = fill; cells more than 0.05
  from the diagonal are named. No cell is faded.
* **Main paper: APPENDIX.**
* **Misreading risk.** (a) These are point means **without intervals** — the
  distance from the diagonal is not tested. (b) Proximity to the diagonal is not
  evidence of correctness of either selector. (c) Controlled and native points
  are different protocols sharing one panel.
* **Caption message.** The selector leaves most cells near the diagonal and
  moves a few far from it, NIFTY/German native being the largest.

### Figure G — observed protocol span (`figG_protocol_span`)

* **Source.** Frozen controlled and native τ_int(−ΔDP) per cell.
* **Plotted.** Per cell, the range of the protocol points actually evaluated,
  with every point drawn.
* **Main paper: APPENDIX only.**
* **Misreading risk — the important one.** This is **not** an identified set and
  **not** full protocol uncertainty. The protocol points are sparse and were
  chosen for other reasons; five cells show a zero-width span only because they
  were measured once. The title says "Observed … across evaluated protocols" and
  the subtitle states the one-point caveat.
* **Caption message.** Where we evaluated more than one protocol, the
  intervention effect moved by more than its own typical size.

## 3. Recommended main-paper set (ICLR, 9 pages)

1. **Figure A** — the premise: the intervention beside its package (RQ1).
2. **Figure B** — the protocol shift (RQ2).
3. **Figure C** — the selection-support mechanism, replicated across two methods.
4. **Figure E** — FMP component attribution.

**Appendix:** A2, B2, C2, D (+ its σ_c^AUC supplement), F, G.

If a fifth main-text slot exists, **B2** is the strongest addition, because it
is the only view of the protocol shift that shows intervals on both ends.

## 4. Visualization bugs found and fixed

1. **Figure D claimed a selector for a quantity that has none.** The subtitle
   read "σ_c^BCE" over fixed-epoch curves that involve no checkpoint selection —
   which is precisely why the BCE and AUC versions had identical top panels. A
   wrong claim, not a layout defect. Fixed.
2. **Figure A** quadrant tags collided with data and tick labels on two of three
   panels → drawn on the leftmost panel only, inset and lighter.
3. **Figure B** the four centre labels overprinted each other and the legend sat
   on a quadrant tag → per-cell label offsets, legend moved below the axes.
4. **Figure B2** four labels stacked on one another → vertical spreading with
   leader lines.
5. **Figure C** both brackets anchored to the same height, so "T …" ran into the
   S bracket → staggered.
6. **Figure F** the `y = x` annotation ran under the legend; my first fix moved
   it onto the x tick labels, which was worse → legend moved to the empty
   lower-right quadrant and the annotation returned to the diagonal's end.
7. **A broken edit script** left a dangling `NATIVE = [` in `fig_b`, a syntax
   error; the figure silently did not re-render and the stale PNG stayed on
   disk. Caught by checking the run output rather than trusting the file.

Carried over from X27: propagation endpoints are drawn as real markers, and
`Checks.limits` now asserts on every figure that all plotted coordinates and
interval ends are inside the axes.

## 5. Palette

Categorical slots are fixed and never cycled: FairGB `#2a78d6`, FairGNN
`#eb6834`, FairVGNN `#1baf7a`, NIFTY `#9a6ad6`. Validated with the palette
checker: lightness band, chroma floor, CVD separation (worst adjacent pair
ΔE 9.2 deutan, above the 8 threshold) and normal-vision floor all **pass**;
contrast of the green against white is **2.74, below 3:1 — a WARN**. The relief
that warning requires is present in every figure: a legend, a distinct marker
shape per method, and a `_source.csv` table. Two alternatives were tested
(a darker green, and Okabe–Ito); both traded the outright CVD pass for a CVD
warning, which is the worse direction, so the original was kept.
