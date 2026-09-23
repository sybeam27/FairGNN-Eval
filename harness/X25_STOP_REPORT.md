# X25 stopped at gate G3: stored-score outcome ≠ restored checkpoint by one near-threshold node

**Status: STOPPED.** This is a pre-registered hard stop (X25 §7, §10: "restore
or replay mismatch (G3)"). No X25 decomposition, bootstrap, table or figure has
been computed. The G3 criterion is **not** changed after the fact. X22, X23, X24
and every frozen CSV are untouched.

## What ran before the stop

| step | result |
|---|---|
| read-only audit | passed; §2 of the pre-registration |
| X25 pre-registration | `2547bf3` |
| tooling + tests | `4684cf1`: G1 (hook side effect), G2 (wrapper smoke), G3 (replay/restore on the H=30 smoke run), G5 (analyzer mechanics) ALL PASS |
| G4 | `test_native_phase1.py` 127/127, `test_x24_smoke.py` 26/26, on HEAD |
| rerun | R10 (H=1000, D0), then R11 (H=1000, D1), 6 × 5 each; started 02:52:02, detached (`setsid nohup`), `/tmp` only |
| early integrity check (02:58) | on the first R10 cells as they persisted: **1 G3 failure** |

## The failure

R10, split 20, run 3, arm M1, σ_c^BCE:

* **The selected epoch matches.** Full-support replay from the stored
  validation records gives 988, equal to the pipeline CSV `m1_epoch` and to the
  ValidationHistory slot.
* **The outcome at that epoch does not.** From the stored in-training test
  scores vs the pipeline's restored-checkpoint scores: **|ΔDP| = 6.41e-3,
  |ΔEO| = 8.40e-3**, AUC within 1 pair swap. G3 requires DP and EO to agree
  exactly (≤ 1e-12).

**Scope at the time of the check:** 4 R10 cells persisted, 16 arm-selector
checks, 1 failure. The other 15 agree on DP/EO to ≤ 1e-16 with ≤ 1 AUC pair
swap. R11 had not started.

## Diagnosis (read-only)

At epoch 988 the five smallest |test score| values in that arm are 4.52e-5,
3.82e-4, 3.96e-4, 5.90e-3 and 7.32e-3.

| flipped (sign of the k nearest-to-zero test nodes) | ΔDP vs CSV | ΔEO vs CSV |
|---|---|---|
| none (stored scores as recorded) | 6.41e-3 | 8.40e-3 |
| **k = 1** (score 4.5e-5; group a=0, label y=1) | **5.6e-17** | **5.6e-17** |
| k = 2 | 1.06e-2 | 5.6e-17 |
| k = 3 | 1.70e-2 | 5.6e-17 |

One prediction flip in group a=0 (156 test nodes) moves DP by 1/156 = 0.0064,
exactly the observed difference. Flipping that single node, whose score is
within about 5e-5 of the `score > 0` threshold, reproduces the
restored-checkpoint DP and EO to floating-point precision.

**Reading.** The recorded in-training forward and the later restored-checkpoint
forward of the same epoch's weights differ by GPU numerical noise. Here that
noise moves one node across the decision threshold. This agrees with G1, where
NIFTY/German is not bitwise reproducible even within one process. It is not a
structural restore failure: the epoch, weights and alignment are correct, and 15
of 16 checks are exact. But the pre-registered G3 criterion is exact DP/EO
equality, so the stop stands.

## Consequence for the analysis, if continued

X25 computes every capped, fixed-epoch and telescoping quantity from the stored
in-training scores; no checkpoints exist for those epochs. At a small fraction
of (arm, epoch) points, an outcome can therefore differ from what a restored
checkpoint would give by single near-threshold flips. That is about 0.006–0.011
on DP per flip on German's 250-node test set, the same order as the frozen
`dp_min_step`. Any X25 contrast near that scale would be affected.

## Rerun state

The R10 → R11 rerun was launched before the stop and is still running (`/tmp`
only, nothing copied or committed). Nothing new has been launched. Keeping it
running preserves the trajectories in case the decision below allows the
analysis; the job can be stopped at any time without affecting anything frozen.

## Decision needed (not taken)

1. **Keep the stop.** Stop the rerun, report X25 as not executed, and record
   that NIFTY/German fixed-epoch and capped-selector outcomes cannot be
   recovered without restore-level noise under exact-equality criteria.
2. **Amend G3 before any X25 quantity is computed** (a new pre-registration
   commit, criterion fixed without seeing any decomposition result):
   * epochs must match;
   * AUC within 2 pair swaps, as now;
   * DP/EO may differ from the restored checkpoint only by prediction flips of
     test nodes with |score| ≤ ε, fixed in advance (for example ε = 1e-3;
     here the flip is at 4.5e-5), verified per mismatch by the flip-reproduction
     test above;
   * the number of such flips is reported, and any mismatch not explained this
     way remains a hard stop.

   This follows the X20 precedent, where the AUC restore bound was replaced by a
   noise-derived criterion, recorded openly.
3. **Redefine the canonical outcome.** State that all X25 outcomes, including
   full support, are computed from stored in-training scores. Treat G3 as a
   consistency report with the noise-margin explanation of option 2, and report
   the full-support differences from the CSV alongside.

Options 2 and 3 are pre-registration changes and are not applied without
approval.

## Addendum: the rerun finished; full G3 scope (still no X25 estimate computed)

The rerun that was already running completed. Its outputs are kept in `/tmp`
only; nothing has been copied to `harness/results` or committed.

| run | started | finished | rc | cells | trajectory files |
|---|---|---|---|---|---|
| R10 (H=1000, D0) | 02:52:02 | 03:39:29 | 0 | 30/30 (60 rows) | 60, all valid (complete 0..1000, finite) |
| R11 (H=1000, D1) | 03:39:29 | 04:28:56 | 0 | 30/30 (60 rows) | 60, all valid |

* CSV SHA-256: `x25_R10.csv` fc0a8c49…, `x25_R11.csv` 7487795d….
* The completion marker's "npz files: 0" line is a counting bug in the driver.
  It filtered out lines containing "tmp", and every path lives under `/tmp`.
  The files themselves are complete, as the table shows.

**G3 over the complete rerun (30 cells × 2 arms × 2 selectors × 2 runs = 240
checks):**
* **Replayed epochs** equal the CSV epochs and the ValidationHistory slots in
  all 240.
* **Mismatches: 1.** It is the arm already diagnosed above (R10 s20 r3 M1
  σ_c^BCE, epoch 988). Flipping the single test node with |score| 4.5e-5
  explains it exactly.
* **The other 239** agree on DP and EO to ≤ 1e-12 and on AUC within 2 pair
  swaps.

These are only the checks needed to scope the stopping cause. No
selector-support, fixed-epoch, bridge, telescoping or reproduction-gate quantity
has been computed. The decision above is still pending.
