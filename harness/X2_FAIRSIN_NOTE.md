# FairSIN — excluded from the primary attribution roster

**Status: excluded.** Preserved here as an implementation and reproducibility
note, not as a finding. It is deliberately not developed into anything larger.

## What was verified

The official `FairSIN-main/in-train.py` was run **unmodified**, with its own
loader, its own data path and the German+GCN configuration from its own
`experiment.sh` (`--delta=0.5 --d='yes' --c_lr=0.1 --e_lr=0.1 --hidden=18
--c_epochs=10`). Neither the repository nor the `dev` environment was changed:

  * `dataset/german` is a symlink to this study's `data/german`, which supplies
    the files the loader expects. Data placement, not code.
  * `memory_profiler` is imported by `in-train.py` and is not installed, so a
    stub was injected into `sys.modules` inside the runner only — the same
    device this study used for `dgl`.
  * pandas 3 makes the loader's `df['Gender'][mask] = 1` a no-op, because
    chained assignment cannot modify the frame under copy-on-write and
    copy-on-write can no longer be disabled (`Pandas4Warning`, pandas 3.0.5).
    Verified directly: after that statement the column is still
    `['Male', 'Female', 'Male']`. So the run was done in a throwaway venv with
    `--system-site-packages` and `pandas<3` installed over it, which inherits
    torch 2.6.0 and leaves `dev` at pandas 3.0.5 untouched.

## What the native run does

At its published configuration the model collapses to constant prediction and
oscillates between the two classes:

    ep31   Acc 0.3   F1 0.0000   Parity 0.0   Equality 0.0     all-negative
    ep32   Acc 0.7   F1 0.8235   Parity 0.0   Equality 0.0     all-positive
    ep34   Acc 0.3   F1 0.0000   Parity 0.0   Equality 0.0
    final  Acc 70.0  F1 82.27    Parity 1.28  Equality 0.84

Parity and Equality are exactly 0.0 on the collapsed epochs because a constant
predictor has no disparity by construction. That is the degenerate case this
study's Pareto reading exists to catch, appearing here in the native path.

So the instability seen through the adapter — validation logits reaching 2.5e30
— is not an artefact of the wrapper. The official path shows the same behaviour
in a different register, which is what the native verification was for.

## Why it is excluded rather than fixed

`--prop` defaults to `'scatter'`, which is `propagate2(h, edge_index) + bias`:
sum aggregation that ignores `adj_norm_sp`. German averages 44.5 neighbours, so
magnitudes grow by roughly that factor per layer. `experiment.sh` does not
override `--prop`, so this **is** the published configuration.

Switching to `prop='spmm'` (normalised propagation) would remove the problem and
is therefore *not* used as a FairSIN result. If it is ever run, it is a
diagnostic sensitivity check — "does the numerical problem disappear under
normalised propagation" — and is labelled as one.

## What is not claimed

Nothing about FairSIN's method. One configuration, one dataset, one backbone,
one seed, in an environment the code predates. The adapter reached three of four
verification conditions (trajectory completeness, code-native selector replay,
test isolation) and failed only instrumentation invariance, for the reason
above. That is enough to exclude it from the primary roster and not enough to
say anything else.

## Status: terminated (2026-09-14)

FairSIN is removed from the primary roster and kept only as a reproduction
note. No further engineering time is spent on it. The roster for the audit is
FairGNN, NIFTY, FairGB, FairVGNN, with GNN as the audited baseline.

Reason of record: its published configuration collapses to constant prediction
natively (see above), so an intervention contrast under the common protocol
would be measuring the collapse, not the intervention. That is a finding about
the reproduction, not a defect this study is equipped to attribute.
