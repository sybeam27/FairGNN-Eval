# The baseline cannot be reproduced in a second process

## What was attempted

`pilot_tau.py` originally recorded the common baseline B only at its own
code-native selector (`b.best_state`). That serves

    tau_pkg^pub = Y(M1; H_p, sigma_p) - Y(B; H_B, sigma_B)

but not the audit decomposition, where every term must be read at one sigma_c:

    tau_pkg^audit = tau_base^audit + tau_int

`baseline_sigma_c.py` was written to recover the missing term cheaply: retrain
B from the same seed, split and device, and read it at sigma_c^BCE and
sigma_c^AUC. The pilot seeds B with `torch.manual_seed(seed)` immediately
before construction, independently of the method loop, so this looked exact.

## What happened

On two cells (split 20, runs 0-1) the reproduction matched to 7.6e-05 AUC and
0.0 DP. Over the full 6 x 5 design it did not.

| residual | mean | median | max |
|---|---|---|---|
| \|dAUC\| | 0.0128 | 0.0006 | 0.0754 |
| \|dDP\|  | 0.0306 | 0.0053 | 0.3362 |

**16 of 30 (split, run) cells** diverged beyond 1e-3. The worst, split 24 run 2,
reproduced B at DP 0.000 where the pilot measured 0.336.

The cause is ordinary GPU nondeterminism -- non-associative reductions and
TF32 -- compounding over 200 epochs. It is amplified by checkpoint selection:
two trajectories that differ in the sixth decimal can put the validation
minimum at a different epoch (e.g. 94 vs 132), and the two checkpoints are then
genuinely different models, not two readings of one model.

## Consequence

A reproduced baseline is a *different draw*, not the same B. Using it would
have made `tau_base^audit = Y(M0) - Y(B')` and `tau_pkg^pub = Y(M1) - Y(B)`
refer to two different baselines while presenting them as one decomposition.
The identity `tau_pkg^audit = tau_base^audit + tau_int` would still have held
arithmetically -- it holds by construction for any common subtrahend -- so the
alignment check would **not** have caught this. It passed at 1e-16 on the
contaminated join.

## Fix

`pilot_tau.py` now reads B at both common selectors inside the same process
that trains it, from `hb.slots`, and writes `bc_epoch`, `bc_auc`, `bc_dp`,
`bc_eo`. `analyze_6x5.py` refuses to run on a CSV that lacks those columns
rather than falling back to a join.

`baseline_sigma_c.py` is kept for the record. It is the instrument that
established this, and it should not be used to produce results.

## Recorded, not corrected

No attempt is made to make training bit-deterministic. Determinism is not
needed once every term of a comparison is read from one process; forcing it
would change every number already collected for a property the design does not
rely on.
