# Arm B candidates: field-by-field diff against native, before Phase 1

Every Arm B candidate is compared with its official/native configuration,
field by field. The Arm A side is what corrected Arm A actually ran. The native
side comes from the authors' own artifacts, pinned and stored under
`harness/provenance/`:

* FairVGNN `938f2e8`: `run_{german,bail,credit}.sh`, `fairvgnn.py`,
  `fairvgnn_credit.py`, `dataset.py`, `learn.py`
* NIFTY `6c270c5`: `README.md`, `nifty_sota_gnn.py`, `models/ssf.py`,
  `utils.py`
* FairGB: vendored `FairGB-main/` (`run.sh`, `main.py`, `data_utils.py`,
  `eval.py`)

Code fields were decided by normalized diffs of the relevant functions, with
the harness's own instrumentation removed. Nothing is assumed from line-overlap
ratios.

**Result: 0 of 7 candidates are native-reusable.**

## Summary

| cell | horizon | preproc | arch | hidden | optim | fairness hp | intervention / augmentation | schedule | selector | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| NIFTY / german | **200 vs 1000** | = | = | = | = | = | **drop rates differ** | = | = | run |
| FairGB / bail | **200 vs 1500** | = | = | = | = | = | = | = | = | run |
| FairGB / german | **200 vs 1500** | **norm vs none** | = | = | = | = | = | = | = | run |
| FairGB / credit | **200 vs 2000** | = | = | = | = | = | = | = | = | held (budget rule, X12) |
| FairVGNN / bail | **200 vs 300** | = | = | = | = | = | = | = | = | run |
| FairVGNN / german | = | **norm vs none** | = | = | = | = | = | = | = | **not reusable** |
| FairVGNN / credit | = | = | = | = | = | = | = | **fairvgnn.py vs fairvgnn_credit.py** | = | **not reusable** |

## NIFTY / german

| field | Arm A | native | source | match |
|---|---|---|---|---|
| horizon | 200 (+1 epoch loop) | 1000 (+1 epoch loop) | README `--epochs 1000`; both loops `range(epochs+1)` | no |
| preprocessing | none | none | `nifty_sota_gnn.py:135-145`, no `feature_norm` for german | yes |
| architecture | GCN `Encoder` + SSF projection/prediction, spectral norm | same | `models/ssf.py` | yes |
| hidden / proj | 16 / 16 | 16 / 16 (argparse default) | README `--hidden 16` | yes |
| optimizer, lr, wd | Adam 1e-3, 1e-5 | Adam 1e-3, 1e-5 (default) | `nifty_sota_gnn.py:219-220` | yes |
| sim_coeff | 0.6 | 0.6 | README | yes |
| training drop rates | **edge 0 / 0, feature 0 / 0** | edge 0.001 / 0.001, feature 0.1 / 0.1 | README; `nifty_sota_gnn.py:277-280` | **no** |
| validation-view edge rate | 0.1 / 0.1 (class default) | 0.001 / 0.001 | `nifty_sota_gnn.py:213-214` | **no** |
| validation-view feature rate | 0.1 / 0.1 | 0.1 / 0.1 | `:215-216` | yes |
| schedule | per-epoch SSF loop | same | normalized diff: only `x_1` uses `rate_1` locally vs `rate_2` officially, both 0.1 | yes |
| published selector | argmin(val_c_loss + val_s_loss) | same | `nifty_sota_gnn.py:325` | yes |

**Local-code finding.** `algorithms/NIFTY.py.__init__` sets
`self.drop_edge_rate_1 = self.drop_edge_rate_2 = 0` and
`self.drop_feature_rate_1 = self.drop_feature_rate_2 = 0` after building the
validation views. The training loop reads these attributes, so the local class
trains with **no edge dropping and no feature noise**. Only the
sensitive-attribute flip in `drop_feature(..., sens_flag=True)` remains. The
official loop applies the rates every epoch. Every NIFTY cell of corrected Arm A
trained this way. M0 and M1 share it, so the Arm A contrast is valid for NIFTY
as implemented; it is not official NIFTY. Arm A stays frozen and this is
recorded, not re-opened. The native run restores the rates at the harness,
after construction.

## FairGB / bail, german, credit

| field | Arm A | native | source | match |
|---|---|---|---|---|
| horizon | 200 | bail 1500, german 1500, credit 2000 | `run.sh` | no |
| preprocessing, bail and credit | min-max to [-1, 1], sensitive column preserved | same | `data_utils.py:289`; harness `utils/data.py` | yes* |
| preprocessing, german | **min-max** | **none** | `data_utils.py:289` (`if dataname in ['bail','credit']`) | **no** |
| encoder, hidden, dropout | SAGE, 16, 0.5 | SAGE, 16, 0.5 (argparse defaults) | `main.py:138-141` | yes |
| c_lr, e_lr, c_wd, e_wd | per `run.sh` | per `run.sh` | `run.sh` | yes |
| alpha (selector weight), eta, warmup | per `run.sh`, 5 | per `run.sh`, 5 | `run.sh`, `main.py:142-145` | yes |
| intervention | CNM on, CAL on | same | | yes |
| schedule | `FairGB_alg.fit` epoch loop | `main.py` epoch loop | normalized diff: names (`encoder_m`), line breaks, CAL/CNM flags, instrumentation | yes |
| evaluation | `eval.py` + `return_output` | `eval.py` | `diff` | yes |
| published selector | auc + F1 + acc - alpha(parity + equality) | same | `main.py:107` | yes |

\* The harness `feature_norm` sets a zero range to 1 before dividing; the
upstream version divides by zero. This matters only for a constant column.
Recorded.

## FairVGNN / bail, german, credit

| field | Arm A | native | source | match |
|---|---|---|---|---|
| horizon | 200 | german 200 (default), bail 300, credit 200 | `run_*.sh`, `fairvgnn.py` argparse | bail no |
| preprocessing, bail and credit | `feature_norm` in the wrapper's `get_dataset` | same | `dataset.py:346-349` (`if dataname != 'german'`) | yes |
| preprocessing, german | **normalized** (harness pre-normalizes, and the wrapper normalizes unconditionally) | **none** | `dataset.py:346` | **no** |
| encoder, prop, hidden, dropout, K | GCN, scatter, 16, 0.5, 10 | same | run lines + argparse | yes |
| c/e/g/d lr and wd | per run line, defaults otherwise | same | | yes |
| top_k, alpha, clip_e, ratio, d/g/c epochs | per run line, defaults otherwise | same | | yes |
| intervention | f_mask yes, weight_clip yes | same | | yes |
| schedule, german and bail | `run()` loop | `fairvgnn.py` `run()` | normalized diff: instrumentation, extra per-group diagnostics, formatting only | yes |
| schedule, credit | **`fairvgnn.py` loop** | **`fairvgnn_credit.py`**: encoder in eval mode with no encoder step during the generator phase, plus `classifier.clip_parameters()` (`--clip_c 1`) every epoch | `diff fairvgnn.py fairvgnn_credit.py` | **no** |
| evaluation | `evaluate_ged3` | `learn.py:301` | normalized diff: local adds per-group diagnostics and returns output; selector inputs identical | yes |
| published selector | auc + F1 + acc - alpha(parity + equality), floor 0 | same | `fairvgnn.py:185` | yes |

**Why FairVGNN / german and credit are not reusable.** Their horizons match
(200), but german differs in preprocessing, and credit uses a different
training script whose schedule the local wrapper does not implement. Neither is
in the Phase 1 list.

* **german** can be made native at the harness, the same way the RNG capture
  works. Skip the harness pre-normalization, and swap the wrapper's
  `feature_norm` for identity for the duration of `fit`.
* **credit** cannot. The differences sit inside the training loop, so a native
  run needs a local mirror of `fairvgnn_credit.py` or an edit to
  `algorithms/FairVGNN.py`. Either would need its diff recorded in provenance.

## What a native run changes, and what it does not

* **Changed per cell:** horizon; NIFTY's training and validation-view drop
  rates; FairGB german's normalization off; FairVGNN bail's horizon.
* **Unchanged:** splits 20-25 and runs 0-4; M1 = native + intervention ON; M0 =
  same native + intervention OFF; sigma_c^BCE and sigma_c^AUC; G_c; one-process
  B/M0/M1 recording; the X11 RNG contract.
* **Baseline B:** held at corrected Arm A's B (GNN, H = 200) in every native
  cell. B has no native protocol of its own on bail and credit, and holding it
  fixed means `tau_base^native - tau_base^ArmA` reflects the method's M0
  changing, not the reference moving.
