# FairGB/credit scheduling and FairVGNN/credit adapter scope

Written after the Phase 1 final report (X18). Nothing here was chosen from
results. The single timing cell below was used only for its wall-clock time.

## FairGB/credit native: measured cost

The X12 trigger is satisfied (X16, X18). FairGB/credit native validation is the
next confirmatory run.

**One real cell** was run on GPU 2 to `/tmp`, kept out of `harness/results`:

* split 20, run 0, `--protocol native`: H = 2000, upstream normalization (on
  for credit), B at 200, cell-level persistence;
* verified: `method_epochs` 2000, `b_epochs` 200, `feature_normalize` 1;
  selected epochs M1 773/810 and M0 1575/242 for BCE/AUC;
* wall clock **286 s** for the whole cell (B, M1, M0, both selectors), rc 0,
  2 rows.

The earlier per-epoch estimate (1570 ms/epoch, "about 26 h for 3x5") came from
a 15-epoch probe, where one-time setup (loading, normalization, mixup
neighbor-distribution precompute) dominates. The FairGB/bail rerun is
consistent with the new figure: 30 cells in 89 min, about 178 s/cell at
H = 1500 on the smaller graph.

| design | cells | estimated wall clock |
|---|---|---|
| 3 x 5 | 15 | about 1.2 h |
| 6 x 5 (the frozen design) | 30 | about 2.4 h |

Plan: run 6 x 5, matching every other Phase 1 cell, under
`--protocol native` and cell-level persistence, then add it to the Phase 1
analysis.

## FairVGNN/credit: faithful adapter, scope and cost only

Nothing is implemented; this is the estimate X16 called for.

### What the official credit script changes

A unified diff of `fairvgnn.py` against `fairvgnn_credit.py` (FairVGNN
`938f2e8`, both under `harness/provenance/`) shows exactly four behavioural lines,
plus one argument:

1. Discriminator phase: `encoder.train()` becomes **`encoder.eval()`**.
2. Discriminator phase: `optimizer_e.zero_grad()` is **removed**.
3. Discriminator phase: `optimizer_e.step()` is **removed**. The encoder is not
   updated while the discriminator trains.
4. Inside `if args.weight_clip == 'yes':`, right after
   `encoder.clip_parameters(weights)`: **`classifier.clip_parameters()`**
   is added.
5. Argument: `--clip_c`, default 1. `run_credit.sh` never sets it, so it is
   *official-repo-default*.

Everything else (generator and classifier phases, evaluation, selector) is
identical.

**Correction to X13.** X13 said the encoder is frozen "during the generator
phase". It is the **discriminator** phase. The conclusion is unchanged: the
shared wrapper does not implement this loop, so FairVGNN/credit cannot reuse it.

### State of the local wrapper

`algorithms/FairVGNN.py` already has `MLP_classifier.clip_parameters()`
(line 951, clamping to `±args.clip_c`). But `args_class` is empty and `fit()`
never sets `clip_c`, and `run()` never calls it or freezes the encoder. The
wrapper is not edited.

### Adapter design

* A separate module, `harness/adapters/fairvgnn_credit_native.py`, holds a copy
  of the local `FairVGNN.run()` loop with the four official changes applied and
  `clip_c = 1`.
* It keeps everything the corrected contract requires:
  * run seed passed through;
  * the full RNG replay bundle captured before each stochastic validation
    call;
  * cell-fixed `xi_eval`;
  * trajectory logging;
  * M0/M1 flags.
* Used only for `(FairVGNN, credit)` under `--protocol native`.
  `native_config("FairVGNN", "credit")` stops raising and routes there.
* **Control definition.** M1 = f_mask yes, weight_clip yes. M0 = f_mask no,
  weight_clip no. Because classifier clipping sits inside the weight_clip block
  officially, M0 switches it off together with encoder weight clipping. The
  discriminator-phase encoder freeze is not a flag and stays on in both arms,
  as in the official script. Isolation is preserved.

### Verification before any use

1. **Provenance:** a normalized diff of the adapter's loop against
   `fairvgnn_credit.py`, with instrumentation stripped, must be identical. This
   is the same method X13 used for `fairvgnn.py`.
2. **Behaviour tests on CUDA:**
   * encoder parameters are unchanged across the discriminator loop;
   * with weight_clip yes, every classifier weight is in `[-1, 1]` after each
     epoch;
   * with weight_clip no, no classifier clipping happens.
3. **The existing FairVGNN gate:** seed, initial parameters, replay fidelity
   within the CUDA noise floor, xi_eval reproducibility, completeness, selector
   replay, test isolation.

### Estimated cost

* **Engineering:** about 2-3 h. That covers the adapter (about 150-200 lines),
  tests (about 150 lines), the provenance diff and one gate run.
* **Compute:** corrected Arm A FairVGNN/credit at H = 200 took 101 min for
  6 x 5 with the `fairvgnn.py` loop. The credit loop drops the encoder step in
  the discriminator phase and adds a cheap clamp, so about **1.5-1.7 h for
  6 x 5** and about 50 min for 3 x 5. To be confirmed by one timed cell after
  the adapter passes its gate.

Not started. The adapter is built only if approved.
