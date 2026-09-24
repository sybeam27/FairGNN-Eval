# Changelog

## 2026-09-24 — Phase 0 audit follow-ups

Artifact and documentation fixes. **No reported estimate changes**; `phase0_verify` sections 1–6
reproduce the frozen numbers (resolved 11/8/9, headline 31/26/29 and 9/23/24, NIFTY 2×2
−0.1171 / −0.1799).

### Fixed
* **`results/per_unit_metrics.csv.gz` schema.** The stored method name encodes the configuration
  (`BIND-1pct`, `FairSIN-GCN`) and `pilot_tau.py:547` writes the literal `backbone="GCN"` for every
  method, which is wrong for FairGB (SAGE). The export now splits the name into `method` +
  `configuration`, takes `backbone` from `build_results.canonical_backbone`, and keeps the raw name as
  `store_method`. The file now joins 1:1 with `cell_results.csv` on
  (method, dataset, backbone, configuration, protocol) — 90 cells, 0 unmatched.
* **Export precision.** Values are no longer rounded to 6 decimals. Reproduction of the published
  means improves from 1.96e-7 to 9.8e-17, and 5 spurious exact ties in FairEdit/Credit disappear.
* **FairEdit caveat text.** `build_results.py` and `x30_config_table.py` printed "20 of ~44k edges"
  for every dataset; correct for German, out by 14× on Bail and 65× on Credit. The caveat is now
  per dataset and states that the edge-addition branch is disabled (`FairEdit.py:711`).
* **Import paths.** `FairGate.models.*` → `models.*` in 36 files (41 sites), with the repository root
  put on `sys.path` by `harness/core/paths.py`, so the code no longer depends on the checkout name.

### Known-wrong, not yet fixed
* The `factors_changed` string on all 25 native rows claims the selector changed; it did not
  (see `results/phase0_audit/R2_native_selector_facts.md`).
* The German baseline B is trained for 200 epochs while its resolved published horizon is 1000
  (see `results/phase0_audit/SUMMARY.md`, T1).

### Fixed (native protocol wording)
* The `factors_changed` string claimed "selector: sigma_c -> the method's own published rule" on all
  25 native rows. The reported native estimate uses the controlled validation-BCE selector; the
  method's own rule is replayed and recorded (`code_epoch`, `m1pub_*`) but never enters tau_I, and the
  M^{-I} arm has no published-rule counterpart. Corrected in `build_results.native_factors`, in the
  `selector` field of `method_configurations.csv`, and in the README rows for sections 3b and 3c:
  3b is a **horizon-only** contrast with validation-BCE selection retained on both sides.
  The regenerated CSVs land with `results_v2/`; `results/` is untouched.
* `figures/src/make_fig3.py`: panel (b) title "Horizon + selector" -> "Published horizon". The figure
  itself is regenerated with `results_v2/`.
