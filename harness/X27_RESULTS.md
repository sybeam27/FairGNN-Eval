# X27 results: FMP mechanistic decomposition (Pokec-z, Pokec-n)

**Pre-registration:** `00b6b26`. **Tooling:** `d9a73b6` (runner, gates),
`85a1048` (analyzer, figures, mechanics test). X22–X26 and every frozen CSV are
untouched.

**Role.** FMP is a mechanistic case study separating graph propagation from the
claimed fairness correction. It is **excluded** from the benchmark aggregates:
the 10/12 counts, the native-transfer counts, and any win/loss frequency.

**No native λ exists**, so every quantity here is **configuration-conditional**:
τ_fair(λ1, λ2) is an intervention contrast at a stated configuration, not a
native-method effect. The parser defaults (3, 3) are not native, no pair is
called published, and λ was never chosen from any result.

## 1. What ran

* **Environment (Option A):** official DGL in an isolated env
  (`torch 2.2.2+cu121`, `dgl 1.1.3+cu121`, PyG 2.5.3); the main environment was
  never modified. Official `FMP-main` sources were imported unmodified.
* **GraphConv equivalence**, tolerance fixed at 1e-5 relative beforehand:
  worst **2.70e-07** on the full 67,796-node pokec_z and **2.30e-07** on
  pokec_n, versus a torch-only `D^-1/2 (A+I) D^-1/2 X`.
* **Runs:** 6 splits (loader seeds 20–25) × 5 runs × 7 configurations, both
  datasets. pokec_z 12:12→12:51, pokec_n 12:51→13:29, rc 0 each,
  **630/630 rows and 210/210 trajectories per dataset**.
* **Runtime pilot** (timing only, never interpreted): 103 s per cell of seven
  configurations → 1.72 h projected for both datasets, under the 24 h rule.

## 2. Gates

| gate | result |
|---|---|
| G1 sensitive-vector run isolation | the original tensor is byte-identical at the start and end of five consecutive runs |
| G2 official mutation bug (diagnostic) | confirmed: `get_sen` mutates its argument in place; the harness original stays untouched |
| G3 initialization pairing | all seven configurations of a cell share one MLP initialization; five runs give five different ones |
| G4 RNG contract | the same (split, run) reproduces its initial weights |
| G5 test isolation | train, validation and test mutually disjoint |
| G6 inference reproducibility | see below |
| analyzer mechanics | identity ≤3.5e-18 per cell and ≤4.9e-17 per replicate; null case exactly zero; planted +0.03 offset recovered as +0.0330; grid rules; three contract refusals |

**G6 changed once, before any τ existed.** FMP's forward draws no RNG, so two
evaluations of the same model differed only by CUDA kernel noise (spread
9.5e-07). The bitwise assertion was replaced by the project's existing
noise-aware criterion — hard predictions, DP and EO exact, AUC within
noise-reorderable pairs (X20 form) — which passes with 0 swaps against 0
reorderable pairs. Recorded here, not hidden.

**Contracts on both datasets:** 630 rows = 30 cells × 7 configurations × 3
selectors, no duplicates, all finite, **0 undefined EO**, seeds 27+run, and
epochs/K/`num_gnn_layer` exactly as registered (300/5/2). The identity
`τ_total = τ_prop + τ_fair` holds **≤1e-12 per cell and in all 10,000
replicates**, for all three selectors on both datasets.

## 3. Decomposition, σ_last (primary), −ΔDP

| dataset | λ2 | τ_prop | λ1 | τ_fair | τ_total |
|---|---|---|---|---|---|
| pokec_z | 0.01 | +0.0021 u | 5 | −0.0017 u | +0.0005 u |
| pokec_z | 0.01 | | 30 | −0.0022 u | −0.0001 u |
| **pokec_z** | **20** | **−0.0585 [−0.0847, −0.0412] s1.00 R** | 5 | −0.0005 u | **−0.0590 R** |
| pokec_z | 20 | | 30 | −0.0032 u | −0.0617 R |
| pokec_n | 0.01 | +0.0012 u | 5 | −0.0012 u | +0.0000 u |
| pokec_n | 0.01 | | 30 | −0.0016 u | −0.0004 u |
| **pokec_n** | **20** | **−0.0427 [−0.0549, −0.0295] s0.97 R** | 5 | +0.0007 u | **−0.0420 R** |
| pokec_n | 20 | | 30 | +0.0034 u | −0.0393 R |

* **Propagation is the component that moves fairness**, and it moves it in the
  harmful direction on DP: at λ2 = 20, τ_prop = −0.059 (pokec_z) and −0.043
  (pokec_n), both resolved. At λ2 = 0.01 — nearly no propagation, γ ≈ 0.99 —
  τ_prop is ≈ 0 and unresolved, as the structure predicts.
* **The fairness correction adds no resolved effect at any grid endpoint.**
  Every τ_fair is unresolved, with magnitudes ≤ 0.0034 across both datasets.
* **τ_total tracks τ_prop almost exactly**, because τ_fair is negligible beside
  it.
* **−ΔEO**: pokec_z shows the same pattern (τ_prop(20) = −0.0572 resolved, all
  τ_fair unresolved); on pokec_n τ_prop(20) is unresolved (−0.0047).
* **ΔAUC**: no τ_prop or τ_fair is resolved on either dataset; the utility
  effects are small (|·| ≤ 0.007).

## 4. Grid stability (Table 5)

Over the preregistered endpoints, for **all three selectors on both datasets**:

| dataset | selector | classification | resolved improve | resolved harm | unresolved |
|---|---|---|---|---|---|
| pokec_z | last / bce / auc | **unresolved** | 0 | 0 | 4 of 4 |
| pokec_n | last / bce / auc | **unresolved** | 0 | 0 | 4 of 4 |

This classifies **the preregistered endpoints of the official search space**,
not all possible λ.

## 5. Selector robustness

τ_fair(−ΔDP) under σ_last, σ_c^BCE and σ_c^AUC:

* **pokec_z:** all four grid cells agree across selectors (same resolution
  state, same sign) — every one unresolved.
* **pokec_n:** three of four cells are marked "DIFFERS", but only because the
  sign flips among values that are **all unresolved** and within ±0.005 of zero.
  No selector produces a resolved fairness-correction effect.

**This is a negative replication of the X25 NIFTY selector phenomenon**, and it
is reported as such: in FMP, the fairness-correction attribution does not become
resolved under any of the three selectors, so there is nothing for the choice of
selector to change. Prereg **Case F**.

## 6. What the evidence supports

* **Prereg Case A.** Within FMP on these two datasets, at the preregistered
  official-grid endpoints and under σ_last, the observed fairness–utility change
  is attributable almost entirely to **graph propagation**, with the claimed
  fairness correction contributing no resolved change at the same propagation
  state.
* The decomposition itself is sound: F01 is a valid propagation-only control
  (λ1 = 0 removes the debiasing branch), F11 shares γ with F01, and the identity
  holds exactly.
* **Case F** additionally: the conclusion is stable across selectors.

## 7. What it does **not** support

* Not "FMP's fairness mechanism is useless". τ_fair is unresolved at these four
  endpoints with 30 cells; that is an absence of resolved effect, not a
  demonstration of zero effect, and other λ values were not tested.
* Not a native-method claim: no native λ exists, so these are
  configuration-conditional contrasts.
* Not a generalization from two Pokec datasets to FMP as a whole, nor to other
  fair GNNs.
* τ_prop and τ_fair are **not** independent causal mechanisms; they are ordered
  components of one package, and F10 was deliberately not run because λ2 also
  scales the debiasing step.
* Nothing here enters the benchmark counts.

## 8. Outputs

`harness/results/x27/`: `x27_pokec_z.csv`, `x27_pokec_n.csv`, per-dataset
`_analysis.txt`, `_summary.csv`, `_cell_table.csv`, the trajectory folders
(210 `.npz` each, checksum-manifested, excluded from git by the repository's
`*.npz` rule), and `x27_graphconv_equivalence.txt`.
Figures under `harness/results/figures/` with the `x27_fmp_` prefix: the
propagation/fairness vector decomposition, the τ_fair grid, the −ΔEO versions,
and the selector-robustness comparison.
