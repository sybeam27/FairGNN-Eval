# Native FairVGNN/credit adapter: implementation and gate

Decisions of record:

1. FairGB/credit native 6x5 is approved and running.
2. The FairVGNN/credit native adapter is approved for implementation and
   verification.
3. **The FairVGNN/credit 6x5 run is not approved yet.** It waits for this gate
   and a 1-cell timing report.
4. FMP training remains on hold.

## Correction log

X13 said FairVGNN's credit script freezes the encoder "during the generator
phase". That was wrong. The official `fairvgnn_credit.py` freezes it during the
**discriminator** phase: `encoder.eval()`, with no encoder `zero_grad()` or
`step()` inside the discriminator loop. X19 first recorded the correction, and
it is restated here as a log entry. X13's conclusion (the shared wrapper cannot
serve FairVGNN/credit) is unaffected.

## Implementation: nothing shared or upstream is edited

* **`harness/adapters/fairvgnn_credit_native.py`** is generated, not hand-edited,
  by `harness/adapters/_build_fairvgnn_credit_native.py` from the local
  `algorithms/FairVGNN.py` `FairVGNN.run()`. It applies exactly the four
  verified credit edits, each to an anchor asserted to occur exactly once:
  1. discriminator phase: `encoder.train()` becomes `encoder.eval()`;
  2. discriminator phase: `optimizer_e.zero_grad()` removed;
  3. discriminator phase: `optimizer_e.step()` removed;
  4. inside `if args.weight_clip == 'yes'`, after
     `encoder.clip_parameters(weights)`: `classifier.clip_parameters()`.

  It also adds one plumbing line, `args.clip_c = self._clip_c`, because the
  official argparse supplies `--clip_c` and the local `args_class` does not.
  `_build_fairvgnn_credit_native.py --check` confirms the committed adapter
  equals a fresh build.
* **Live globals.** The compiled `run` is rebound to the live
  `algorithms.FairVGNN` module globals, so `evaluate_ged3`, `seed_everything`,
  `tqdm` and the model classes resolve exactly as in the shared wrapper, and the
  harness's RNG-bundle capture applies unchanged. Checked: the adapter's
  `run.__globals__` is `algorithms.FairVGNN.__dict__`, and nothing is added to
  that namespace.
* **Class.** `FairVGNNCreditNative(VG.FairVGNN)` overrides only `run`, and takes
  `clip_c`. `fit()` calls `self.run(...)` (`FairVGNN.py:1373`), so the override
  is used; `fit`, `inference_modules` and restoration are inherited.
* **Routing.**
  * `native_config("FairVGNN", "credit")` no longer raises. It returns the
    `run_credit.sh` configuration with `fairvgnn_credit_adapter=True` and
    `clip_c=1.0`, parsed from the pinned official argparse (the default, since
    `run_credit.sh` does not set it).
  * `published()` is unchanged, so Arm A is unaffected.
  * `pilot_tau.train()` instantiates the adapter only when that flag is set.
* **Controls.**
  * M1 = f_mask yes, weight_clip yes.
  * M0 = f_mask no, weight_clip no. Classifier clipping switches off with
    weight_clip, because the official script places it inside that block.
  * The discriminator-phase encoder freeze is not a flag, so it is on in both
    arms, as in the official script.

## Gate

Gate 1 was checked separately; gates 2-10 ran on CUDA
(`harness/tests/test_fairvgnn_credit_adapter.py`), on credit split 20, run seeds
27-29, 8 epochs, through the pilot's own `train()`.

| # | gate | result |
|---|---|---|
| 1 | normalized control flow vs official `fairvgnn_credit.py` | matches. The residuals are the same non-behavioural set X13 found against `fairvgnn.py`: formatting, `pbar`, harness hooks, per-group diagnostics, return tuple. Against `fairvgnn.py` the same diff shows exactly the four credit lines |
| 2 | encoder frozen across the discriminator phase | hash equal at discriminator start and classifier start, 8/8 epochs, every run, both arms. **Negative control:** the shared loop changes it in 3/3 epochs |
| 3 | classifier clipping follows weight_clip | M1: one call per epoch, max\|w\| <= 0.44, restored checkpoints within [-1, 1]. M0: 0 calls |
| 4 | M0/M1 same-run initialization | identical within a run; distinct across runs |
| 5 | run seed passed | FairVGNN reseeds with 27, 28, 29 in both arms |
| 6 | CUDA replay fidelity | within max(1e-5, 4x replay spread); hard predictions, DP, EO identical; AUC within noise-reorderable pairs |
| 7 | fixed xi_eval | test scores within tolerance; predictions, DP, EO identical |
| 8 | trajectory completeness | complete, contiguous, finite |
| 9 | selector replay | `select_min_bce` and `select_max_auc` equal the stored slots |
| 10 | test isolation | reference is the validation split, disjoint from test; no test field |

Final run: **111/111 pass.**

### The gate criterion changed once; recorded, not hidden

The first run failed 6 of 24 checks in gate 6, which then required AUC within a
**fixed 2 pair swaps**, a bound taken from the smaller bail case (X14). A
diagnostic across all 12 slots showed:

* **0** hard-prediction flips, ΔDP = **0**, ΔEO = **0**;
* replay differed from the record by no more than the replay-to-replay spread
  (<= 9.5e-7), five of the six failures on M0, which draws no random number;
* AUC moved by **0.5-9.5 pair swaps**. Credit's 7,500-node validation set has
  n_pos x n_neg of about 9.7e6 (one pair = 1.03e-7) and many near-tied scores.

So the residual is device noise reordering near-ties, not restoration. The
fixed bound was replaced by what the noise itself allows. |ΔAUC| x n_pos x n_neg
must not exceed the number of positive/negative pairs whose score gap is within
2x the measured replay difference, since only those can be reordered. The
combined check was split, so hard predictions, DP and EO each still must be
exactly equal. Final run: 0-19 swaps observed against 84-96 noise-reorderable
pairs per slot.

## Not yet done

* **1-cell timing.** GPU 2 is running FairGB/credit native 6x5, and a timing
  under contention would overstate the cost to be approved. The single cell is
  timed when FairGB/credit finishes, then 3x5 and 6x5 estimates are reported.
  No scientific result from that cell is read.
* **FairVGNN/credit 6x5** does not launch without a separate approval.

## Phase 1 wording, as decided

* **Arm A** is unchanged and not re-run.
* **Phase 1 result:** *Controlled intervention attribution did not consistently
  transfer to the methods' native training configurations*, stated for 5 cells
  on 2 datasets and not generalized to fair GNNs.
* **Central message**, kept in place of a standalone "protocol-dependent"
  headline: *Package-level evidence does not identify the claimed intervention
  effect, and intervention-level conclusions themselves require explicit
  protocol specification and robustness checks.*

## FMP, as decided

* **No training.** No per-dataset native (λ1, λ2) exists (X17), so no parser
  default such as (3, 3) is called native.
* **After credit native validation**, FMP is reviewed as a separate mechanistic
  case study. The preferred design is a pre-registered small λ grid, analysing
  `tau_fair = Y(F11) - Y(F01)` and `tau_prop = Y(F01) - Y(F00)` over it rather
  than a single λ pair.
* **No ranking or count with the four methods.** No new model, dataset, metric
  or selector.

## 1-cell timing (after FairGB/credit freed GPU 2)

One FairVGNN/credit native cell was timed on an otherwise idle GPU 2:
split 20, run 0, `--protocol native`, the adapter, H = 200, B at 200, the X11
RNG contract, cell-level persistence to `/tmp`, kept out of `harness/results`.

* **Wall clock: 197 s** for the whole cell (B, M1, M0, both selectors), pilot
  rc 0, 2 rows, protocol native, `rng_contract` bundle-replay/xi-eval.
* The background job reported exit 1. That came from the timing wrapper's
  final `grep -c` for the adapter class name, which the pilot never prints
  (count 0, so grep exits 1). It was not the run.
* The adapter being used follows from the same `native_config` →
  `pilot_tau.train()` routing the gate verified
  (`type(model) == FairVGNNCreditNative`). It is not printed in this log.
* **Only the wall-clock time is used.** The cell's effect values are not read.

| design | cells | estimated wall clock |
|---|---|---|
| 3 x 5 | 15 | about 49 min |
| 6 x 5 | 30 | about 1.6 h |

This agrees with the X19 estimate (1.5-1.7 h for 6 x 5). The FairVGNN/credit
6 x 5 run still needs separate approval.
