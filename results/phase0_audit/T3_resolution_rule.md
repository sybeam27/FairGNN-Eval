# T3 — The resolution rule and sign stability

Audit date 2026-09-24. Repository `/home/sypark/workspace/FairGNN-Eval`.
Python `~/miniconda3/envs/dev/bin/python`. **All work in this task is CPU-only**
(pandas/numpy over frozen CSVs); no GPU was used. No frozen file was modified;
every artifact below is written under `results/phase0_audit/`.

Commands run:

```
~/miniconda3/envs/dev/bin/python results/phase0_audit/T3_recompute.py
```

(the script is a verbatim copy of the scratchpad script that produced
`results/phase0_audit/T3_sign_stability_recompute.csv`).

The frozen rule, as stated in the paper and in the code header
(`harness/experiments/build_results.py:10-12`):

> resolved iff sign stability ≥ 0.75 **and** |mean| ≥ 0.010 **and** the 95 %
> percentile interval excludes 0.

---

## (a) Where sign stability and `resolved` are computed — [확인됨]

### The formula

`harness/experiments/analyze_armA.py:40-43`

```python
def sign_stability(x):
    x = np.asarray(x, float)
    nz = x[x != 0.0]
    return 0.0 if nz.size == 0 else max((nz > 0).mean(), (nz < 0).mean())
```

Thresholds: `harness/experiments/analyze_armA.py:33-34`
(`SIGN_MIN = 0.75`, `NEAR_ZERO = 0.010`).

### What `x` actually is

`harness/experiments/analyze_x29.py:131-157` is the function that build_results
calls. Inside it:

```
141:  g = d[(d.method == m) & (d.dataset == ds_) & (d.selector == SEL)]
...
150:  sgn = (sign_stability(g[f"{q}_{c}"])
151:         if q != "D" else sign_stability(cells[f"{q}_{c}"]))
...
155:  resolved=resolved(mu, lo, hi, sgn),
```

`g` is the **raw per-unit frame** — one row per (split_id, run_id) at
`selector = common_bce` (`analyze_x29.py:59`), i.e. 30 units per cell. `reps`
(the 10,000 bootstrap replicate means, line 140) is used **only** for `lo`/`hi`
(line 148) and never for `sgn`.

The resolved predicate: `harness/experiments/analyze_x29.py:74-76`

```python
def resolved(mean, lo, hi, sign):
    """The frozen rule, unchanged."""
    return bool(sign >= SIGN_MIN and abs(mean) >= NEAR_ZERO and lo * hi > 0)
```

The values reach `results/1_*.csv` via
`harness/experiments/build_results.py:199-217` (`frozen_estimates`, which calls
`analyze_x29.per_cell`) → `build_results.py:214-215`:

```
214:  out[f"{q_name}_{c_name}_sign_stability"] = float(r["sign"])
215:  out[f"{q_name}_{c_name}_resolved"] = bool(r.resolved)
```

`harness/experiments/bootstrap_armA.py` contains **no** sign-stability code at
all; `boot()` (lines 48-61) returns only replicate means.

### Verdict — [확인됨]

The stored `tau_I_*_sign_stability` is a **per-unit share over the 30 matched
(split, run) units**, not a bootstrap-replicate share. More precisely it is the
share of **non-zero units whose sign equals the MAJORITY sign** —
`max(P(+), P(−))` — which is **not** necessarily the sign of the point estimate.
The prior check is correct, with the refinement that the reference sign is the
majority sign, not the estimate's sign. The documentation string at
`analyze_6x5.py:38` calls it "modal-sign share", which is accurate; the paper's
word "sign stability" next to a bootstrap-interval condition invites the
bootstrap-replicate reading, which is wrong.

Numerical confirmation: `results/1_main_package_vs_intervention.csv` row 1
(BIND/GCN/bail/1pct, `n_units = 30`) stores
`tau_I_dAUC_sign_stability = 0.6333333333333333 = 19/30`. A replicate share
over 10,000 draws could not be an exact 30ths fraction.

---

## (b) Recompute as "share of units whose sign equals the point estimate" — [확인됨]

Source: `results/per_unit_metrics.csv.gz`, filtered to
`protocol == "controlled"`, `selector == "common_bce"`; per unit
`tau_I = M_plus_I − M_minus_I` on AUC, and negated on dp/eo. Store method names
were mapped to `results/1_main_package_vs_intervention.csv` by
`store_name = method` if `configuration == "default"` else `f"{method}-{configuration}"`
(the same rule the repository itself uses —
`harness/experiments/robustness_six_split.py:39-46`). All 36 primary cells
matched with exactly 30 units each (asserted in the script).

Definition recomputed: `share of non-zero units with sign == sign(mean)`.

| coordinate | frozen resolved | recomputed (point-estimate sign) | change |
|---|---|---|---|
| dAUC  | 11 | **11** | none |
| negDP | 8  | **8**  | none |
| negEO | 9  | **9**  | none |

**The 36-cell resolved counts do NOT change: 11 / 8 / 9 is unaffected.**
Cell-by-cell the resolved flag is identical in all 3 × 36 = 108 comparisons.

Why the counts survive even though the two definitions differ: the point-estimate
share differs from the majority share only in 8 of 108 cells, all of them cells
where the two signs disagree, and in every such cell the share is near 0.5 —
far below the 0.75 threshold under either definition.

Cells where majority-sign share ≠ point-estimate share (all remain unresolved):

| coord | cell | majority share | point-est. share | mean |
|---|---|---|---|---|
| negDP | FairGNN / bail      | 0.5625 | 0.4375 | −0.00001 |
| negDP | FairGNN / pokec_n   | 0.5333 | 0.4667 | −0.00392 |
| negDP | FairGNN / pokec_z_g | 0.5333 | 0.4667 | +0.00116 |
| negDP | FairVGNN / credit   | 0.5333 | 0.4667 | −0.00013 |
| negEO | FairGB / german     | 0.5517 | 0.4483 | −0.01647 |
| negEO | FairSIN / pokec_n   | 0.5667 | 0.4333 | −0.00050 |
| negEO | FairVGNN / credit   | 0.5333 | 0.4667 | −0.01386 |
| negEO | NIFTY / german      | 0.5667 | 0.4333 | +0.00424 |

One further, unrelated discrepancy — [확인됨], cosmetic:
`FairEdit / credit`, dAUC: the frozen store reports 0.5666667 (= 17/30), the
recomputation from `per_unit_metrics.csv.gz` gives 0.56 (= 14/25). Cause:
`per_unit_metrics.csv.gz` rounds `auc` to 6 decimals, which manufactures 5 exact
ties (τ = 0) that do not exist in the raw store, and `sign_stability` drops
exact zeros. Blast radius: the published per-unit file cannot reproduce the
published sign-stability column bit-for-bit for near-degenerate cells. No
resolved flag changes (0.56 and 0.5667 are both far below 0.75). Proposed fix:
export `per_unit_metrics.csv.gz` at full float precision (e.g. `float_format=
"%.17g"`), or publish the per-unit τ columns directly rather than the three
state readings.

Artifact: `results/phase0_audit/T3_sign_stability_recompute.csv`
(36 rows × all three definitions, means, intervals and per-condition flags).

---

## (c) Recompute as the share of BOOTSTRAP REPLICATES matching the point estimate — [확인됨]

Design reproduced exactly as frozen (`harness/experiments/bootstrap_armA.py:14-27,
48-61`): **B = 10,000 replicates; the 6 splits are resampled with replacement
first, then the 5 runs within each drawn split; each unit's
(B, M_minus_I, M_plus_I) triple is carried whole** (the τ is formed inside the
unit before any resampling, so the pairing can never be broken).

**Seed used: 20260914** (`numpy.random.default_rng(20260914)`), the frozen
`bootstrap_armA.SEED` (`bootstrap_armA.py:32`). One RNG object is advanced
across the 36 cells in the row order of
`results/1_main_package_vs_intervention.csv`. Note that the frozen pipeline
iterates cells in a different order (`analyze_x29.per_cell` loops dataset-then-
method), so the draw streams are not byte-identical; the resulting Monte-Carlo
differences in the interval endpoints are ≤ 4.2e-3 and change no resolved flag
under the unit-level definitions (verified: 108/108 flags agree with frozen).

Sign stability redefined as `mean(sign(replicate_mean) == sign(point_estimate))`:

| coordinate | frozen resolved | resolved under replicate-share | change |
|---|---|---|---|
| dAUC  | 11 | **13** | +2 |
| negDP | 8  | **12** | +4 |
| negEO | 9  | **11** | +2 |

### Resolved cell lists under the replicate-share definition

**dAUC (13):** BeMap/bail, BeMap/credit, BeMap/pokec_z, EDITS/credit,
FairGB/bail, FairGB/german, FairSIN(GCN)/credit, FairSIN(GCN)/german,
FairVGNN/bail, FairVGNN/credit, **NIFTY/bail\***, NIFTY/credit,
**NIFTY/german\***
(*= newly resolved relative to frozen)

**negDP (12):** BIND(1pct)/income, BeMap/bail, EDITS/credit, EDITS/german,
**FairGB/bail\***, **FairGNN/pokec_z\***, FairSIN(GCN)/credit,
FairSIN(GCN)/german, FairSIN(GCN)/pokec_z, NIFTY/credit, **SFG/bail\***,
**SFG/german\***

**negEO (11):** BIND(1pct)/income, BeMap/bail, EDITS/credit, EDITS/german,
FairGB/bail, **FairGNN/pokec_z\***, FairSIN(GCN)/credit, FairSIN(GCN)/german,
FairSIN(GCN)/pokec_z, NIFTY/credit, **SFG/german\***

No cell is *lost*: the replicate-share condition is strictly weaker here,
because a bootstrap interval that already excludes 0 forces the replicate-sign
share above 0.975 by construction. This is the material finding:

> **Under the replicate-share reading of "sign stability", the condition is
> almost entirely redundant with the interval condition and the rule collapses
> to `|mean| ≥ 0.010 AND interval excludes 0`.** The 8 extra resolved cells are
> exactly the cells the unit-level condition is currently filtering out.

Blast radius: the paper's headline counts (11/8/9) depend on the *unit-level*
reading. If a reader implements "sign stability" as a bootstrap quantity — which
the surrounding text invites — they get 13/12/11 and cannot reproduce the paper.
Proposed fix (documentation only, no code change needed): state the definition
explicitly in the paper, e.g. "sign stability = the share of the 30 matched
(split, run) units, excluding exact zeros, that carry the modal sign", and cite
`analyze_armA.py:40`.

---

## (d) Cells passing interval + magnitude but failing unit-level sign stability — [확인됨]

Using the point-estimate-sign definition from (b) with threshold 0.75. These are
precisely the cells the sign condition is doing work on.

**dAUC (2):**

| cell | unit sign stability | mean |
|---|---|---|
| NIFTY / bail   | 0.6333 | see recompute CSV |
| NIFTY / german | 0.6667 | |

**negDP (4):**

| cell | unit sign stability |
|---|---|
| FairGB / bail     | 0.7333 |
| FairGNN / pokec_z | 0.7000 |
| SFG / bail        | 0.7333 |
| SFG / german      | 0.7000 |

**negEO (2):**

| cell | unit sign stability |
|---|---|
| FairGNN / pokec_z | 0.6333 |
| SFG / german      | 0.7333 |

Total 8 cells (2 + 4 + 2) — identical to the set of cells that (c) newly
resolves, confirming that the sign condition is the *only* binding constraint
for them. Three of the eight (FairGB/bail negDP, SFG/bail negDP, SFG/german
negEO) sit at 0.7333, i.e. one unit short of the threshold (22/30 vs the 23/30
needed). [추정] The 11/8/9 headline is therefore sensitive to a single unit in
three cells; this is worth a sentence in the paper rather than a silent
threshold.

---

## (e) Is the same resolution function used for tables 2, 3a, 3b, 3c and appendix_robustness? — [확인됨]

### Who writes each file

All of `results/1_*`, `2_*`, `3a`, `3b`, `3c`, `4_*`, `5_*` are written by a
single function: `harness/experiments/build_results.py:1076` `section_tables()`,
dispatched through the `files` dict at `build_results.py:1164-1175` and written
at `build_results.py:785-786`.

`results/appendix_robustness/*` is written by a **different** script:
`harness/experiments/robustness_six_split.py:99` and `:117`.

### Is the rule shared?

| output | resolved flag comes from | shared or re-implemented |
|---|---|---|
| `1_main_package_vs_intervention.csv` | `build_results.py:199-217` → `analyze_x29.per_cell:155` → `analyze_x29.resolved:74-76` | **shared** (canonical) |
| `1b_main_task_adapted.csv` | same `cells` frame (`build_results.py:1089-1090`) | **shared** |
| `2_configuration_variation.csv` | `build_results.py:1096-1100` reads `tau_I_{c}_resolved` off the same `cells` frame for both the primary and the robustness row; no recomputation | **shared** |
| `3a_protocol_selector_bce_vs_auc.csv` (`s3b`) | `build_results.py:1134`; rows built in `protocol_pairs` at `build_results.py:365-382`. The BCE side (`x_resolved`, line 375) is the canonical flag. The **AUC side** (`y_resolved`, line 379) comes from `auc_selector_estimates`, `build_results.py:219-236` | **x: shared; y: RE-IMPLEMENTED** |
| `3b_protocol_native_horizon_selector.csv` / `3c_..._published_procedure.csv` | `build_results.py:1160-1161`, slices of `s3a` = `protocol_pairs` `controlled_vs_native` rows (`build_results.py:342-364`); both `x_resolved` (line 368) and `y_resolved` (line 371) are the canonical `cells` flags | **shared** |
| `4_mechanistic_case_study.csv` | `build_results.py:443-462` (`mechanistic()`), reading the frozen x25/x26 summary CSVs; those were produced by `analyze_x26_fairgb.py:222-226`, an independent inline implementation | **RE-IMPLEMENTED upstream** |
| `5_component_case_study_FMP.csv` | same `mech` frame, from x27/x31 summaries | **RE-IMPLEMENTED upstream** |
| `appendix_robustness/six_split_sensitivity_*.csv` | `robustness_six_split.py:60-61` | **RE-IMPLEMENTED, deliberately and differently** |

### The re-implementations, verbatim

1. `harness/experiments/build_results.py:234-235` — AUC-selector column of 3a:

```python
out[f"tau_I_{c_name}_auc_selector_resolved"] = bool(
    sgn >= SIGN_MIN and abs(mu) >= NEAR_ZERO and lo * hi > 0)
```

Same three conditions, same constants (imported at `build_results.py:60`), but
it does not call `analyze_x29.resolved`. `sgn` at `build_results.py:230` is
`sign_stability(a[f"int_{c_key}"])` over the `common_auc` per-unit rows — the
same unit-level definition. **Behaviourally equivalent; a duplication risk, not
a current defect.**

2. `harness/experiments/analyze_x24_factorial.py:95-99` and
`harness/experiments/analyze_x26_fairgb.py:222-226` — the x24/x26 mechanistic
stages each re-spell the predicate inline. Both import `sign_stability` from
`analyze_armA` (`:35` and `:31` respectively) and both use `SIGN_MIN`/`NEAR_ZERO`.
**Behaviourally equivalent.**

3. `harness/experiments/robustness_six_split.py` — **intentionally different**,
and documented as such at `robustness_six_split.py:13-15`:

```
37:  MAG, SIGN = 0.010, 0.75    # the frozen rule's magnitude floor and sign stability
...
56:      stab = float(np.mean(np.sign(split_means) == np.sign(m))) if m != 0 else 0.0
...
60:  def resolved(m, lo, hi, stab, use_sign=True):
61:      return bool(abs(m) >= MAG and lo * hi > 0 and (stab >= SIGN or not use_sign))
```

Three differences from the canonical rule, all declared in the module docstring:
the interval is a split-level *t* interval rather than the bootstrap percentile
interval; sign stability is the share of the **6 split-level means** (not the 30
units) matching the sign of the mean; and it uses the **point-estimate sign**,
not the modal sign. This is the appendix's stated purpose (a conservative
six-cluster re-analysis) and is correct as designed — but note that its sign
definition is the (b) definition while the main tables use the (a) definition,
so "the frozen rule" comment on line 37 is slightly overstated.

### Verdict

- The rule is **shared** for tables 1, 1b, 2, 3b, 3c and the BCE side of 3a.
- It is **re-implemented but equivalent** for the AUC side of 3a
  (`build_results.py:234-235`) and for the upstream mechanistic summaries feeding
  tables 4 and 5 (`analyze_x24_factorial.py:97`, `analyze_x26_fairgb.py:225`).
  Four independent spellings of the same three-clause predicate exist in the
  repository; none currently disagrees, but nothing enforces that.
- It is **deliberately different** in `results/appendix_robustness/`
  (`robustness_six_split.py:56,60-61`), by design and documented.

Proposed fix (no behaviour change): move `resolved()` and `sign_stability()` into
a single module (they already live in `analyze_armA` / `analyze_x29`) and have
`build_results.auc_selector_estimates`, `analyze_x24_factorial` and
`analyze_x26_fairgb` import it rather than re-spell it; add a one-line assertion
in `build_results` that every `*_resolved` column agrees with
`analyze_x29.resolved` applied to its own `mean/lo/hi/sign` columns.

---

## Artifacts written

- `results/phase0_audit/T3_sign_stability_recompute.csv` — 36 primary cells ×
  3 coordinates, with `*_sign_majority`, `*_sign_pointest`, `*_sign_boot`,
  `*_interval_excl0`, `*_mag_ok` and `*_resolved_{majority,pointest,boot}`.
- `results/phase0_audit/T3_recompute.py` — the script that produced it.
