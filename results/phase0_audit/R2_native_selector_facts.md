# R2 — native-row fact table: what the `factors_changed` string claims vs what ran

Scope: the 25 native rows of `results/3b_protocol_native_horizon_selector.csv` (21) and
`results/3c_protocol_native_published_procedure.csv` (4). Read-only audit; nothing was changed.

## 1. Headline

**All 25 rows print `selector: sigma_c -> the method's own published rule`. For all 25 the claim is false.**
The reported `native_tau_I_negDP` is selected by sigma_c^BCE — the *same* selector as the controlled arm.
The published rule was implemented and recorded, but never entered the reported quantity. [확인됨]

### Evidence chain

1. The string is a hard-coded constant, concatenated unconditionally:
   `harness/experiments/build_results.py:1046` — `sel = "selector: sigma_c -> the method's own published rule"`,
   spliced in at `:1051`, `:1056`, `:1069`, `:1075` for every method branch; and the twin constant at
   `harness/experiments/build_results.py:529-531` for `results/method_configurations.csv`.
   No branch ever omits or varies it, and no code path checks which selector the stored run used.

2. The stored native runs carry **both** selector arms. Every native store file has a `selector` column with
   exactly `{common_bce, common_auc}`, 30 matched units each (`split_id` x `run_id`):
   `harness/results/armB_native_*.csv`, `harness/results/x30/x30native_*.csv`, `harness/results/x31/x31native_*.csv`.
   There is no third, published-rule arm.

3. `tau_I` is built from the sigma_c slots only. `harness/experiments/pilot_tau.py:520-525`:

   ```
   for sel in ("common_bce", "common_auc"):
       e1s, st1 = h1.slots[sel]
       e0s, st0 = h0.slots[sel]
   ```
   and `:570-572`: `int_dauc=m1["auc"]-m0["auc"]`, `int_ndp=-(m1["dp"]-m0["dp"])`, `int_neo=-(m1["eo"]-m0["eo"])`.
   The published-rule reading `m1_pub = outcome(..., f1_(None, ...))` (`:518`) is stored as `m1pub_*` (`:564-565`)
   and `code_epoch=e1` (`:557`), and is consumed **only** by `pkg_*` (`:571-573`).
   The same split holds for the x30/x31 runners: `harness/experiments/x30_run.py:233,255,261`
   (docstring at `x30_run.py:19`: "`m1pub_*` is M+I at its own code-native ..."; `:168` "code-native selector
   replayed as m1pub; B stays at --epochs"), `harness/experiments/x31_fairvgnn_config_run.py:200,206`,
   `harness/experiments/x31_fairgnn_config_run.py:225,231`.

   Numerically, over all 25 store files (25 x 60 rows):
   `max|int_ndp + (m1_dp - m0_dp)| <= 1.7e-16` and `max|int_ndp + (m1pub_dp - m0_dp)| >= 3.0e-02` (up to 5.5e-01).
   `pkg_ndp` conversely matches `-(m1pub_dp - b_dp)` exactly. [확인됨]

4. The **BCE** arm is the one published. `harness/experiments/build_results.py:198-200` —
   `frozen_estimates(g)`: "The frozen per-cell bootstrap for one cell (BCE selector, as frozen)."
   The AUC arm is emitted separately as `tau_I_*_auc_selector_*` (`:219-239`). Checked row by row:
   for all 25 rows the published `y_mean` equals the mean of `int_ndp` over the `common_bce` rows of
   the corresponding store to `<1e-9`, and differs from the `common_auc` mean. [확인됨]

5. The published rule *was* implemented — it just is not the reported selector:
   `models/algorithms/NIFTY.py:513-515,525-532` (argmin `val_c_loss+val_s_loss`),
   `models/algorithms/FairGB_alg.py:102,208`, `models/algorithms/FairVGNN.py:1096,1271`,
   `harness/adapters/x30_fairsin.py:280,310-312,392` (`code_epoch=code["epoch"]`),
   `harness/adapters/x31_sfg.py:187,220-222,286`. Each writes `best_epoch` / `code_epoch`,
   which is stored and then used only for `tau_pkg`. **Recorded, not used.** [확인됨]

6. `harness/core/published_config.py:324-381` (`native_config`) returns
   `{config, horizon, provenance, source, native_changes}` — **no `selector` key at all**.
   The native protocol as instantiated changes the horizon and (for 3c) preprocessing/training-loop;
   it never changes the selector. [확인됨]

7. The published rule is not merely a relabelling: it selects a different epoch. Per cell, the
   published-rule epoch equals the sigma_c^BCE epoch in only 0-8 of 30 units:

   - FairGB/SAGE/bail: published-rule epoch equals sigma_c^BCE epoch in only 1/30 units
   - FairGB/SAGE/credit: published-rule epoch equals sigma_c^BCE epoch in only 1/30 units
   - FairSIN/GCN/bail: published-rule epoch equals sigma_c^BCE epoch in only 0/30 units
   - FairSIN/GCN/credit: published-rule epoch equals sigma_c^BCE epoch in only 1/30 units
   - FairSIN/GCN/german: published-rule epoch equals sigma_c^BCE epoch in only 5/30 units
   - FairSIN/GCN/pokec_n: published-rule epoch equals sigma_c^BCE epoch in only 3/30 units
   - FairSIN/GCN/pokec_z: published-rule epoch equals sigma_c^BCE epoch in only 1/30 units
   - FairSIN/GIN/bail: published-rule epoch equals sigma_c^BCE epoch in only 4/30 units
   - FairSIN/GIN/credit: published-rule epoch equals sigma_c^BCE epoch in only 2/30 units
   - FairSIN/GIN/german: published-rule epoch equals sigma_c^BCE epoch in only 6/30 units
   - FairSIN/GIN/pokec_n: published-rule epoch equals sigma_c^BCE epoch in only 2/30 units
   - FairSIN/GIN/pokec_z: published-rule epoch equals sigma_c^BCE epoch in only 3/30 units
   - FairSIN/SAGE/bail: published-rule epoch equals sigma_c^BCE epoch in only 0/30 units
   - FairSIN/SAGE/credit: published-rule epoch equals sigma_c^BCE epoch in only 1/30 units
   - FairSIN/SAGE/german: published-rule epoch equals sigma_c^BCE epoch in only 0/30 units
   - FairSIN/SAGE/pokec_n: published-rule epoch equals sigma_c^BCE epoch in only 2/30 units
   - FairSIN/SAGE/pokec_z: published-rule epoch equals sigma_c^BCE epoch in only 5/30 units
   - FairVGNN/GCN/bail: published-rule epoch equals sigma_c^BCE epoch in only 7/30 units
   - SFG/SAGE/bail: published-rule epoch equals sigma_c^BCE epoch in only 1/30 units
   - SFG/SAGE/credit: published-rule epoch equals sigma_c^BCE epoch in only 2/30 units
   - SFG/SAGE/german: published-rule epoch equals sigma_c^BCE epoch in only 3/30 units
   - FairGB/SAGE/german: published-rule epoch equals sigma_c^BCE epoch in only 0/30 units
   - FairVGNN/GCN/credit: published-rule epoch equals sigma_c^BCE epoch in only 8/30 units
   - FairVGNN/GCN/german: published-rule epoch equals sigma_c^BCE epoch in only 0/30 units
   - NIFTY/GCN/german: published-rule epoch equals sigma_c^BCE epoch in only 3/30 units

   Moreover the **M-I (off) arm has no published-rule counterpart at all** — only `h0.slots[sel]`
   exists — so a published-rule `tau_I` could not have been formed from these artifacts even in principle. [확인됨]

## 2. What *is* true in the string

- **Horizon**: correct in all 25 rows. Verified against `method_epochs` (native) and `b_epochs`=200
  (controlled reference) in each store file; all 25 match the printed numbers, including the four
  `horizon unchanged (200)` rows (SFG/credit, SFG/german, FairVGNN/credit, FairVGNN/german). [확인됨]
- **Preprocessing / training-loop clauses (3c and FairGB/german)**: correct. [확인됨]
  - FairGB/german `feature normalization off (upstream data_utils.py rule)`: controlled `feature_normalize=1`
    (`harness/results/armA_german.csv`), native `=0` (`harness/results/armB_native_FairGB_german.csv`);
    rule at `harness/core/published_config.py:353-359`.
  - FairVGNN/german `feature normalization off ...; harness pre-norm and wrapper norm disabled`:
    controlled `=1` (`armA_german.csv`, `armA_fairvgnn_german.csv`), native `=0`; `published_config.py:370-376`,
    applied at `pilot_tau.py:254`.
  - FairVGNN/credit `training loop from fairvgnn_credit.py ... (clip_c=1.0)`: `published_config.py:363-369`,
    applied at `pilot_tau.py:241`; `harness/adapters/fairvgnn_credit_native.py`.
  - NIFTY/german `training and validation-view drop rates from the official command`:
    `published_config.py:341-352`, applied at `pilot_tau.py:154-159`.
- The 21 rows in 3b correctly carry **no** preprocessing clause: FairGB/bail and FairGB/credit keep
  `feature_normalize=1` in both arms, FairVGNN/bail keeps `0` in both, and FairSIN/SFG native and controlled
  rows share the same `preprocessing` text in `results/method_configurations.csv`. [확인됨]

## 3. Fact table

Full machine-readable table: `results/phase0_audit/R2_native_selector_facts.csv`.
Compact view (the two selector columns and the label are constant across all 25 rows — see section 1):

| method | backbone | dataset | family | sel changed | H ctrl | H native | other factors actually applied | string OK | label |
|---|---|---|---|---|---|---|---|---|---|
| FairGB | SAGE | bail | targeted | **no** | 200 | 1500 | none (feature_normalize 1 -> 1) | **no** | [확인됨] |
| FairGB | SAGE | credit | targeted | **no** | 200 | 2000 | none (feature_normalize 1 -> 1) | **no** | [확인됨] |
| FairSIN | GCN | bail | systematic | **no** | 200 | 100 | none (horizon only) | **no** | [확인됨] |
| FairSIN | GCN | credit | systematic | **no** | 200 | 100 | none (horizon only) | **no** | [확인됨] |
| FairSIN | GCN | german | systematic | **no** | 200 | 150 | none (horizon only) | **no** | [확인됨] |
| FairSIN | GCN | pokec_n | systematic | **no** | 200 | 100 | none (horizon only) | **no** | [확인됨] |
| FairSIN | GCN | pokec_z | systematic | **no** | 200 | 100 | none (horizon only) | **no** | [확인됨] |
| FairSIN | GIN | bail | systematic | **no** | 200 | 100 | none (horizon only) | **no** | [확인됨] |
| FairSIN | GIN | credit | systematic | **no** | 200 | 100 | none (horizon only) | **no** | [확인됨] |
| FairSIN | GIN | german | systematic | **no** | 200 | 100 | none (horizon only) | **no** | [확인됨] |
| FairSIN | GIN | pokec_n | systematic | **no** | 200 | 100 | none (horizon only) | **no** | [확인됨] |
| FairSIN | GIN | pokec_z | systematic | **no** | 200 | 100 | none (horizon only) | **no** | [확인됨] |
| FairSIN | SAGE | bail | systematic | **no** | 200 | 150 | none (horizon only) | **no** | [확인됨] |
| FairSIN | SAGE | credit | systematic | **no** | 200 | 150 | none (horizon only) | **no** | [확인됨] |
| FairSIN | SAGE | german | systematic | **no** | 200 | 100 | none (horizon only) | **no** | [확인됨] |
| FairSIN | SAGE | pokec_n | systematic | **no** | 200 | 150 | none (horizon only) | **no** | [확인됨] |
| FairSIN | SAGE | pokec_z | systematic | **no** | 200 | 150 | none (horizon only) | **no** | [확인됨] |
| FairVGNN | GCN | bail | targeted | **no** | 200 | 300 | none (feature_normalize 0 -> 0) | **no** | [확인됨] |
| SFG | SAGE | bail | systematic | **no** | 200 | 160 | none (horizon only) | **no** | [확인됨] |
| SFG | SAGE | credit | systematic | **no** | 200 | 200 | none (horizon only) | **no** | [확인됨] |
| SFG | SAGE | german | systematic | **no** | 200 | 200 | none (horizon only) | **no** | [확인됨] |
| FairGB | SAGE | german | targeted | **no** | 200 | 1500 | feature_normalize 1 -> 0 | **no** | [확인됨] |
| FairVGNN | GCN | credit | targeted | **no** | 200 | 200 | fairvgnn_credit.py loop, clip_c=1.0 | **no** | [확인됨] |
| FairVGNN | GCN | german | targeted | **no** | 200 | 200 | harness pre-norm 1 -> 0, wrapper norm off | **no** | [확인됨] |
| NIFTY | GCN | german | targeted | **no** | 200 | 1000 | NIFTY train/val-view drop rates restored | **no** | [확인됨] |

## 4. Summary

- **25 / 25** native rows print a `factors_changed` string that does not describe what ran.
- The wrong claim is the **selector** clause, and it is wrong in every row and for every method:
  FairSIN (15 rows), SFG (3), FairGB (3), FairVGNN (3), NIFTY (1).
- The **horizon** clause is correct in all 25 rows. The **preprocessing / training-loop** clauses are
  correct in all 4 rows that carry one (plus FairGB/german in 3c), and correctly absent elsewhere.
- So 3b is not "horizon and selector" — it is **horizon only**. `results/README.md` (built at
  `build_results.py:893-894, 1189-1192`) inherits the same error for both 3b and 3c.

### Corrected wording that would be true

Family `systematic` (FairSIN 15, SFG 3) and the horizon-only targeted rows (FairGB/bail, FairGB/credit,
FairVGNN/bail):

> `horizon 200 -> {H}; selector unchanged (sigma_c^BCE); the method's own published rule was replayed
> on the stored trajectory and recorded as code_epoch / m1pub_*, but does not enter this estimate`

(and `horizon unchanged (200); selector unchanged (sigma_c^BCE); ...` for SFG/credit and SFG/german).

Family `targeted` with a published-procedure bundle (3c), e.g. NIFTY/german:

> `horizon 200 -> 1000; selector unchanged (sigma_c^BCE) — NIFTY's own rule argmin(val_c_loss + val_s_loss)
> (models/algorithms/NIFTY.py:513-515,525-532) was recorded but not used; training and validation-view
> drop rates from the official command`

and correspondingly for FairGB/german (`feature normalization off (upstream data_utils.py rule)`),
FairVGNN/credit (`training loop from fairvgnn_credit.py ... (clip_c=1.0)`) and FairVGNN/german
(`feature normalization off (official dataset.py rule); harness pre-norm and wrapper norm disabled`).

A strictly accurate short form for the selector clause in every row:
`selector unchanged (sigma_c^BCE, same as the controlled arm)`.
