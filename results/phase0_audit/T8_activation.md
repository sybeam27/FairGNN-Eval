# T8 — Is the intervention actually active? (36 primary cells)

All commands below are CPU-only (gzip/csv parsing and grep). **No GPU was used and
no training was re-run.** Under the GPU-2-only constraint nothing in this task
needed a GPU, so the constraint is vacuously satisfied.

## Method

Source: `results/per_unit_metrics.csv.gz`, filtered to
`protocol == "controlled" AND selector == "common_bce"`.
Cells: the 36 rows of `results/1_main_package_vs_intervention.csv`.

Name mapping (store encodes the configuration into `method`):
`BIND + configuration=1pct -> BIND-1pct`; `FairSIN + configuration=GCN -> FairSIN-GCN`;
every other method maps by its own name. [확인됨] All 36 cells matched a store key
and each yielded exactly 30 paired units (`M_plus_I`, `M_minus_I`), 30/30 for all 36.

Script: `scratchpad/t8.py` (pure stdlib; reads the frozen store read-only).
"Identical" = `float(M_plus_I) == float(M_minus_I)` on auc AND dp AND eo.

**Precision caveat [확인됨].** The store is rounded to 6 decimals
(e.g. `0.885018`). So "identical" here means identical *at 1e-6*, not
bit-identical. For FairGNN/bail this is re-checked below at full precision.

## Result

`results/phase0_audit/T8_activation.csv`. Headlines:

- 26 of 36 cells: **0 / 30** units identical — the intervention moves every unit.
- Cells with any identical units:
  - **FairEdit/german 7/30**, max |dAUC| 1.371e-3, |dDP| 2.039e-2, |dEO| 2.909e-2
  - **FairEdit/credit 5/30**, max |dAUC| **4.0e-6**, |dDP| **0.0**, |dEO| **0.0**
  - **FairGB/german 1/30** (but max |dDP| 0.40 — one coincidental tie)
- Smallest movers (max |dAUC| over 30 units):
  - FairEdit/credit 4.0e-6 · FairEdit/bail 1.18e-4 · FairEdit/german 1.37e-3
  - **FairGNN/bail 7.73e-4** · FairGNN/pokec_n_g 1.89e-2 · FairGB/credit 1.90e-2
- Largest: BeMap/bail 0.393, FairSIN/german 0.300, FairSIN/credit 0.262.

[확인됨] **FairEdit/credit is effectively inert**: DP and EO are bit-identical in the
store for all 30 units (max |dDP| = max |dEO| = 0.0) and AUC moves at most 4e-6.
This is the direct numeric counterpart of T6: 20 of 2,873,716 directed edges.

## FairGNN / bail (alpha=2, beta=0.05, max |dAUC| ~ 8e-4)

### Is the intervention wired in at all?

[확인됨] Yes, and the plumbing is correct.

- `harness/experiments/pilot_tau.py:55`
  `"FairGNN": dict(closure=True, off=dict(alpha=0.0, beta=0.0))` — M-I is alpha=beta=0.
- `harness/experiments/pilot_tau.py:116-119`
  `FairGNN(nfeat=..., acc=cfg.get("acc",0.39), alpha=off.get("alpha", cfg.get("alpha", 8)),
   beta=off.get("beta", cfg.get("beta", 0.005)))`; bail cfg is alpha=2, beta=0.05
  (`results/method_configurations.csv:3`, from `utils/param.json`).
- `models/algorithms/FairGNN.py:59-60` stores them on `args`.
- `models/algorithms/FairGNN.py:125-132`, unconditional, every epoch:
  ```
  self.cov      = torch.abs(torch.mean((s_score - mean(s_score)) * (y_score - mean(y_score))))
  self.cls_loss = self.criterion(y[idx_train], labels[idx_train]...)
  self.adv_loss = self.criterion(s_g, s_score)
  self.G_loss   = self.cls_loss + self.args.alpha * self.cov - self.args.beta * self.adv_loss
  self.G_loss.backward()
  ```
  No `if alpha > 0` guard, no detach on `y_score`/`s_g` w.r.t. the generator path
  (`s_score` alone is detached, `:122`), and `optimize()` is called every epoch
  (`models/algorithms/FairGNN.py:195`). So both terms **are** backpropagated.
- The adversary itself is separately stepped (`:135-140`), also unconditionally —
  it runs identically in both arms; only its *weight in G_loss* (beta) differs.

So the two arms genuinely differ, and the outputs confirm it:

### Full-precision re-check (not the 6-dp store)

```
$ python -c "csv over harness/results/armA_bail.csv + armA_bail_s23_25.csv,
             method=FairGNN, selector=common_bce"
n = 30 units, splits 20..25
max |dAUC| = 7.7227e-04   max |dDP| = 1.7208e-03   max |dEO| = 3.0832e-03
units with all three deltas exactly 0 : 0 / 30
units with dAUC == 0                  : 0 / 30
units with dDP  == 0                  : 14 / 30
units with dEO  == 0                  : 21 / 30
median |dAUC| = 3.367e-05 ; min |dAUC| = 5.070e-06
m1_epoch == m0_epoch : 30 / 30
```
[확인됨] The arms are **never** identical in AUC — the covariance/adversary terms
are non-zero and do change the weights. The many zero dDP/dEO are an artifact of
the decision rule `score>0`: DP/EO are computed from thresholded 0/1 predictions
(`harness/core/evaluator.py`, `decision="score>0"`), so sub-1e-3 logit shifts flip
no labels on 14/21 of the 30 units. AUC, which uses the raw score, always moves.

### Why is the effect so small? [추정]

Mechanism, from the code (not measured — no loss value is stored anywhere):

1. `s_score = sigmoid(s.detach())` with `s_score[idx_sens_train] = sens[idx_sens_train]`
   (`FairGNN.py:122-123`). The harness passes `idx_sens_train = idx_train`
   (`pilot_tau.py:135`: `m.fit(adj, f, y, itr, iva, ite, s, itr, ...)`).
   bail has `label_number=100` (`harness/DATASET_INVENTORY.csv`, bail row), i.e.
   **100 of 18,876 nodes** (`harness/core/datasets.py:48`) carry a true sensitive
   value; the other 99.5% keep `sigmoid(estimator_logit)`, which at a freshly
   initialised GCN estimator sits tightly around 0.5.
2. `cov` is the mean of `(s_score - mean)(y_score - mean)` over **all** nodes, with
   both factors in [0,1] and `s_score` nearly constant -> `cov` is order 1e-3.
   `alpha * cov = 2 * O(1e-3)` is ~1e-3 against a `cls_loss` of order 0.5.
3. `beta * adv_loss = 0.05 * O(0.7)`, and its gradient w.r.t. the shared encoder is
   scaled by 0.05.

So the fairness terms contribute a ~1e-3-scale perturbation to a ~0.5-scale loss.
Contrast: the same code on pokec_z/pokec_n (far larger sensitive-label sets) gives
max |dAUC| 2.9e-2 / 1.8e-2 — the mechanism is not broken, it is under-weighted on bail.

### Prediction-vector-level identity

[확인 불가] It cannot be checked from stored artifacts. Every FairGNN artifact in
`harness/results/` (`armA_bail*.csv`, `x29_FairGNN_*.csv`, `x31_FairGNN-upstream*.csv`)
stores only scalar metrics per unit; `find harness/results -name "*.npz" -o -name "*.pt"`
returns only x26 (FairGB) / x27 (FMP) trajectory dumps, none for FairGNN, and no log
records `cov`, `adv_loss` or `G_loss` (`grep -rl "cov\b" harness/results/` -> empty).

Artifact that would settle it: the per-node validation/test score vectors that
`ValidationHistory` already holds in memory (`harness/core/trajectory.py`) for both
arms at the selected epoch, persisted as an .npz per unit — plus a per-epoch log of
`(cls_loss, cov, adv_loss)`. Both are additive instrumentation; neither requires
changing training. Producing them **would** require re-running FairGNN/bail on a GPU,
which this task is forbidden to do.

## Bugs found: none

[확인됨] No wiring bug. The FairGNN M+I arm is genuinely active; the near-zero
effect on bail is a **configuration/scale** property (alpha=2 against an
O(1e-3) covariance with 100 sensitive labels), not a disabled intervention.
Reporting note, not a code fix: `results/1_main_package_vs_intervention.csv`
reports FairGNN/bail `tau_I_dAUC_resolved` on deltas whose largest magnitude over
30 units is 7.7e-4 — below any plausible noise floor. The honest statement is
"the intervention is active but its effect is ~1e-4, i.e. indistinguishable from
run-to-run noise", and the same applies a fortiori to FairEdit/credit
(max |dDP| = |dEO| = 0.0).
