# The 3x5 pilot is a published/package view, not an audit decomposition

The 3 splits x 5 runs pilot (`harness/results/pilot_tau_3x5.csv`, and the
splits 23-25 continuation `pilot_tau_s23_25.csv`) recorded the common baseline
B **only at its own code-native selector** (`b.best_state`). Its `b_*` columns
are therefore a published-protocol reading of B.

What that run can support:

* `tau_pkg^pub` -- the published configuration against the baseline, each at
  its own selector. A preliminary signal, on 3 and then 6 splits.
* `tau_int` -- M1 minus M0, both at sigma_c. This term never involves B and is
  unaffected.

What it cannot support:

* `tau_base^audit = Y(M0) - Y(B)` and `tau_pkg^audit = Y(M1) - Y(B)`, because
  B was never read at sigma_c in that run. Recovering it by retraining B in a
  second process was attempted and failed -- see
  `X3_BASELINE_NONDETERMINISM.md`.

These results are kept, not discarded. They are relabelled: everything
previously reported from them as `tau_pkg` is `tau_pkg^pub`, and none of it is
reused as an audit decomposition. The audit decomposition comes only from
`pilot_tau_6x5_audit.csv`, whose `bc_*` columns are written inside the process
that trains B.

The 3x5 split/run variance ratios stay a pilot estimate and are not carried
into any headline; they are recomputed on the 6x5 audit run.
