# RNG contract for stochastic inference, and the FairVGNN rerun gate

Decision after X10: fix the RNG restoration contract on the experiment device
(CUDA), then rerun FairVGNN on german, bail and credit at 6 splits x 5 runs. No
CPU rerun. This is a repair of the replication and restoration contract, not a
new protocol variant: training algorithm, H = 200, preprocessing, M0/M1
definitions, selectors and metrics are unchanged.

## Two purposes, kept apart

**A. Replay state.** For each epoch, the complete generator state immediately
before the stochastic validation call:

* Python `random`
* NumPy
* Torch CPU
* Torch CUDA, every device (`get_rng_state_all`)

Restoring a checkpoint restores this bundle, and a validation forward pass must
then reproduce that epoch's recorded raw validation scores. This is checked on
CUDA; a CPU-only fidelity test does not count.

**B. Final evaluation state.** Test evaluation does not inherit whatever
generator position the checkpoint happened to hold. Each (dataset, split, run)
has one evaluation seed fixed in code before any result:

    xi_eval = crc32("dataset|split|run|eval")

Immediately after a checkpoint is restored for test scoring, every generator is
reseeded to xi_eval. The same xi_eval is used for M0^BCE, M1^BCE, M0^AUC and
M1^AUC, and for the native-selector package view. So tau_int^BCE, tau_int^AUC
and D_selector compare parameters under one mask draw, not two Gumbel draws.
xi_eval depends only on the cell's identity, never on an outcome.

## Implementation, harness level

`algorithms/FairVGNN.py` is not edited. Everything is in `harness/`:

* `core/trajectory.py` adds `capture_rng_bundle`, `restore_rng_bundle`,
  `seed_all`, `eval_rng_seed` and `bundle_history`. `restore_inference_state`
  restores a `_rng_bundle` when present; it supersedes the CPU-only `_rng`.
* `experiments/pilot_tau.py`, FairVGNN branch, while `fit` runs:
  * `algorithms.FairVGNN.evaluate_ged3` is swapped for a wrapper that captures
    the bundle and then calls the original. The training loop has one call
    site (`FairVGNN.py:1216`), looked up by name, so the wrapper sits
    immediately before the stochastic validation call. The original is put
    back in a `finally`.
  * The history is `bundle_history(...)`. When the method assigns its own
    `state_fn`, which snapshots only the CPU generator, the setter wraps it and
    replaces `_rng` with the bundle for that epoch.
  * `score_fn(state, idx, xi=None, replay=False)`:
    * `replay=True` restores the checkpoint's bundle and returns validation
      scores in recorded order.
    * `xi=` restores the parameters, then reseeds to xi_eval.
  * `main()` passes xi only for FairVGNN, and writes `eval_rng_seed` and
    `rng_contract` on every row: `bundle-replay/xi-eval` for FairVGNN,
    `deterministic-inference` for the others.
* The run seed fix from X9 stays: `FairVGNN.fit(seed=run_seed)`.

FairGNN, NIFTY, FairGB and B take no xi and are unaffected. The end-to-end
smoke gives them identical code paths and `deterministic-inference` rows.

## Gate, CUDA, german split 20, runs 27-31, 15 epochs

`harness/tests/test_fairvgnn_seed.py`, all pass:

| check | result |
|---|---|
| internal FairVGNN seed = run_seed, distinct across runs | [27, 28, 29, 30, 31] |
| M0 and M1 of a run reseed identically | 5/5 |
| initial parameters distinct across runs, identical M0/M1 | 5/5 |
| every slot carries the full bundle and no CPU-only `_rng` | 20/20 |
| CUDA replay fidelity, M1 (`f_mask=yes`), all slots | max 2.4e-07 (was 0.38-0.88) |
| CUDA replay fidelity, M0, all slots | max 1.6e-06 |
| negative control, a different epoch's scores | 1.01 |
| completeness, selector replay, test isolation | pass |
| final-epoch validation scores differ across runs | 5/5 |
| xi_eval applied twice, other arms scored in between: test scores | max 7.2e-07 |
| xi_eval applied twice: hard predictions and G_c AUC/DP/EO | identical |
| a different xi moves M1's test scores | 0.52 |

Tolerance is 1e-5, the same one the replay check uses.

**Recorded, not hidden.** The first version of check 7 required exact float
equality and failed at about 1e-7, for M0 as well as M1. M0 draws no random
number at inference, so that residual is CUDA kernel nondeterminism, not RNG.
The criterion was set to the replay gate's existing tolerance, and tightened at
the same time: hard predictions and the unified evaluator's metrics must now
match exactly, and the non-vacuous check must exceed the tolerance rather than
merely exceed 0.

## What happens to the earlier FairVGNN results

Every FairVGNN row produced before this contract -- the seed=1 Arm A cells on
german, bail and credit, and every earlier FairVGNN run on CUDA -- is removed
from the primary analysis and kept as the
**`fixed-seed / incomplete-CUDA-RNG diagnostic arm`**. Those rows are
conditional on seed 1 in each split. They are not 30 independent cells, and
their Y(M1) under sigma_c was scored under an unrestored mask.

## Provisional until the corrected Arm A is recomputed

Cross-method counts that include FairVGNN, such as `10/12`, are provisional.
The only evidence carried forward in the meantime comes from the three
unaffected methods: `|tau_base| > |tau_int|` holds in **7 of 9**
FairGNN/NIFTY/FairGB (method, dataset) combinations. Arm B does not start
until the corrected Arm A analysis is done.
