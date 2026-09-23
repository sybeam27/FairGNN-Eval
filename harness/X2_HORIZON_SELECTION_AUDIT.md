# Phase 1C — Horizon (H) and checkpoint selection (σ), separated

`H` decides how long training runs and whether it stops early. `σ` decides
which of the produced epochs is returned. Several implementations handle both
inside one `if` block, so they look like one thing in code; they are two
operators and are separated here.

Roster only: GNN (baseline), FairGNN, NIFTY, FairVGNN, FairGB, FairSIN, FMP.

## The table

| method | H: horizon | early stop? | σ: selector | candidates | fairness in σ | claim status |
|---|---|---|---|---|---|---|
| **GNN** | fixed epochs | **no** | lowest validation loss | all epochs | no | evaluation convention |
| **FairGNN** | fixed epochs | **no** | highest validation accuracy, above a threshold that adapts over the first 10 epochs | epochs above the threshold | no | evaluation convention |
| **NIFTY** | fixed epochs (`range(epochs+1)`) | **no** | lowest `BCE_val + sim_coeff·invariance_val` | all epochs | **yes** | **unclear** |
| **FairVGNN** | `epochs=200`, nested `d_epochs=5`, `g_epochs=5`, `c_epochs=10` | **no** | `auc + f1 + acc − α(ΔDP + ΔEO)`, α default **1** | all epochs | **yes** | **unclear** |
| **FairGB** | fixed epochs | **no** | same composite, α default **1** | all epochs | **yes** | **unclear** |
| **FairSIN** | `epochs=20` outer (published 80–150), nested `c_epochs=10`, `d_epochs=5`, `m_epoch=20` | **no** | same composite, α default **1**; **plus a nested selector** choosing the neutralisation MLP by validation loss | all epochs | **yes** | **unclear** |
| **FMP** | fixed epochs | **no** | **none** — the final epoch is reported | 1 | no | evaluation convention |

## Four findings

**1. No method in the roster uses early stopping.** Every one runs a fixed
horizon and selects afterwards. That makes `H` and `σ` cleanly separable and a
common horizon feasible in principle — which was not guaranteed, and is the
thing 1C existed to establish before 1D's trajectory logging is designed.

**2. Three audited implementations use the same fairness-aware composite
checkpoint score.** FairVGNN, FairGB and FairSIN all select on
`auc + f1 + acc − α(ΔDP + ΔEO)` with α defaulting to 1.

That is the whole claim for now. Calling it a propagated convention would need
the code lineage actually established, and calling it a problem would need the
selector shown to move the results — neither of which is done. Whether it
matters is answered after 1D, by replacing σ_orig with σ_c on the same
trajectories and measuring how far the fairness gain moves.

**3. In FairSIN's own `experiment.sh`, the method and its baseline get
different selectors.** The vanilla runs pass `--alpha=0`; the FairSIN runs pass
no `--alpha`, so the default of 1 applies. The baseline therefore selects on
`auc + f1 + acc` and the method selects on `auc + f1 + acc − (ΔDP + ΔEO)` —
inside one repository, one script, one experiment.

What this licenses saying, and no more: *the published comparison allows the
fair method and its baseline to select checkpoints using different validation
objectives.* It does **not** yet license "part of FairSIN's fairness gain comes
from selection". That requires applying a common selector to the same
trajectories and measuring the shift, which is what 1D makes possible.

**4. FairSIN contains an internal model-selection step, which is not a second
checkpoint selector.** The neutralisation MLP is chosen by lowest validation
loss (in-train.py:159-168) *inside* the outer loop, and its choice changes what
the classifier is subsequently trained on. So it sits inside the training
algorithm, not after it:

    A(z, σ_inner) → H → σ_final → δ → G

The distinction matters for attribution. `σ_final` picking a checkpoint on a
fairness-aware score is a reporting effect and can be separated out. `σ_inner`
altering the downstream training input is part of the mechanism, and removing it
would change the method rather than the way it is reported.

## Published configuration is not the parser default

`--d` defaults to `'no'`, but `experiment.sh` shows the headline runs are not
uniform:

    german + GCN    --delta=0.5 --d='yes'          D on
    german + GIN    --delta=1   --d='yes'          D on
    bail   + GCN    --delta=1                      D off
    credit          train_mlp.py --delta=0.25      D off, and a different script

So `M₁` for FairSIN is **per (dataset, encoder)**, and on Credit it is a
different entry point entirely. Treating the parser default as the published
method would have mis-specified half the settings, and treating one `M₁` as
valid across settings would mis-specify the rest.

## Three things that must be kept apart per method

    claimed components      what the paper calls its fairness mechanism
    published configuration what was actually switched on for the headline table
    factorial audit space   which combinations the code can meaningfully run

FairSIN is the case where all three differ. That is not a defect in the method;
it is the reason the framework is needed.

## Claim status, and why it is "unclear" rather than "contamination"

A fairness-aware selector is not automatically a confound. If a paper claims its
selection rule as part of its fairness contribution, the rule belongs in `z` and
must vary with the intervention. If the rule is only an evaluation convention
never claimed as a mechanism, it belongs in the evaluation operator and must be
held fixed.

For FairVGNN, FairGB and FairSIN the code alone cannot settle this, so the
status is **unclear** pending a reading of each paper's own description. It is
recorded as unclear rather than assumed either way.

## Consequence for the arms

    published arm     (H_p, σ_p) per method, per setting — what the method
                      actually produces
    attribution arm   (H_c, σ_c) common — what varying z alone produces

    τ_package   = Y(M₁; H_p, σ_p) − Y(B; H_B, σ_B)
    τ_int       = Y(M₁; H_c, σ_c) − Y(M₀; H_c, σ_c)
    τ_selection = Y(M₁; H_c, σ_p) − Y(M₁; H_c, σ_c)

`H_c` is **not** fixed to one epoch count yet. No method early-stops, so a common
horizon is possible, but whether it is scientifically sound — some designs
presuppose their epoch budget — is decided after 1D instruments the trajectories
and the budgets are visible, not now.

---

## Addendum from the 1D dry run — `epochs = N` does not mean N training steps

Found while the completeness check failed on FairGNN for a reason that was not a
logging fault:

    GNN        for epoch in range(epochs + 1)      N + 1 steps
    NIFTY      for epoch in range(epochs + 1)      N + 1 steps
    FairGNN    for epoch in range(args.epochs)     N steps
    FairVGNN   for epoch in range(0, args.epochs)  N steps
    FairGB     for epoch in range(args.epochs)     N steps
    FairSIN    for epoch in range(0, args.epochs)  N steps

So passing `epochs = 1000` to the baseline and to a fair method gives the
baseline one extra training step and one extra selection candidate.

At N = 1000 the effect is 0.1% and almost certainly immaterial. It is recorded
anyway, for two reasons. It is a third unexamined convention in `H` after the
selector and the metric, found in the same place as the others — the seam
between implementations. And the trajectory logger has to declare a horizon per
method to check completeness at all, so the convention cannot be left implicit
even if its effect is negligible.
