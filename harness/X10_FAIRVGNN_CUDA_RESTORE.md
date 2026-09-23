# FairVGNN checkpoint restore is exact on CPU and fails on CUDA

Found by the pre-rerun regression check for the X9 seed fix
(`harness/tests/test_fairvgnn_seed.py`). The full FairVGNN rerun was **not**
started: a failed restore-fidelity check is one of the stop conditions.

## The seed fix itself passes

German, split 20, run seeds 27-31, 15 epochs, GPU 2, through the pilot's own
`train()`:

* the seed FairVGNN reseeds with inside `run()` equals `run_seed` for every run,
  and differs across all five runs;
* M0 and M1 of each run reseed with the same seed;
* initial parameters, hashed right after `reset_parameters()`, differ across
  all five runs and are identical for M0 and M1 of a run;
* trajectory completeness (15/15), selector replay (`select_min_bce`,
  `select_max_auc` equal the stored slots) and test isolation pass everywhere;
* final-epoch validation scores differ across runs, 5/5.

## What fails

Restore fidelity, for **M1 only**, on **CUDA only**:

| arm | device | max restored-vs-recorded, all slots |
|---|---|---|
| M1 (`f_mask=yes`) | CUDA | 3.8e-01 to 8.8e-01, all 10 slots fail |
| M0 (`f_mask=no`) | CUDA | at most 1.2e-06, all pass |
| M1 and M0 | CPU | 0.000e+00, all pass |

The negative control (a different epoch's scores) differs by 1.2-1.6, so the
failure is not a check that passes vacuously.

## Cause

FairVGNN's evaluation draws `F.gumbel_softmax(..., hard=True)`, so a restored
M1 reproduces an epoch's scores only if the random generator is restored with
it. Both the snapshot in `FairVGNN.py:1215` (`_rng_pre`) and
`core.trajectory.inference_state` save `torch.get_rng_state()`, which is the
**CPU** generator. On CUDA the mask is drawn from the **CUDA** generator, which
is never saved. Verified directly on GPU 2:

* CUDA `gumbel_softmax` leaves the CPU generator unchanged and advances the
  CUDA one;
* restoring the CPU state alone does not reproduce the mask;
* restoring the CUDA state does.

M0 disables the mask, so nothing stochastic runs at inference and it restores
exactly. Every earlier FairVGNN fidelity pass (german, bail, credit) came from
`test_restore_fidelity.py`, which hard-codes `device="cpu"`. The contract was
never verified on the device the experiments run on.

## Consequence for results already collected

Every FairVGNN Arm A cell ran on CUDA. For those cells:

* **affected:** Y(M1) under sigma_c^BCE and sigma_c^AUC. The selected epoch is
  right, because it comes from the recorded validation scores, but the test
  outcome is scored under a fresh mask draw, not the one the checkpoint was
  selected with. So `tau_int`, `tau_pkg^audit` and `D_selector` for FairVGNN
  are contaminated.
* **not affected:** Y(M0), Y(B), and therefore `tau_base^audit`. FairGNN,
  NIFTY and FairGB have no stochastic inference and do not use `_rng`.

This applies to the seed=1 FairVGNN results kept as the fixed-seed diagnostic
arm, and to every earlier FairVGNN run on CUDA.

## Seed audit of the other methods (X9, item 7)

Code path. `pilot_tau.train()` calls `torch.manual_seed(seed);
np.random.seed(seed)` (line 107) before any model is built:

* **FairGNN** is constructed at `train()` after that call, and does no internal
  reseeding.
* **NIFTY** is the same.
* **FairGB** stores `fit(seed=...)` in `args.seed` but never uses it:
  `seed_everything` is imported in `FairGB_alg.py:9` and never called on this
  path (only upstream `FairGB/main.py:35` calls it). Its modules are built at
  `FairGB_alg.py:98`, after the harness seed; `mixup` draws from `np.random`,
  which the harness seeds.
* **GNN (B)** is seeded with `torch.manual_seed(seed)` immediately before
  construction in `pilot_tau.main()`.
* The `torch.manual_seed(seed * 1000 + split)` placed before each arm is
  overwritten by line 107, so both arms effectively start from `run_seed`. This
  does not break M0/M1 pairing.

Empirical, `harness/tests/test_seed_wiring.py` (parameters hashed at
construction, before any update, runs 27-31): FairGNN, NIFTY and FairGB each
start runs 0-4 from 5/5 distinct parameters and M0/M1 of a run from identical
ones; B starts from 5/5 distinct parameters. All pass.

## Not changed, pending a decision

The frozen Arm A contract includes the RNG-state checkpoint restoration, so
it is not modified here. A fix would save and restore the CUDA generator
alongside the CPU one. Only the restore path would change; training would not.
