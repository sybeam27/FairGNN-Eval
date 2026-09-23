# Decisions frozen before 1D, 2026-09-13

Three, fixed here so that instrumentation cannot quietly settle them later.

## 1. No global common horizon. Within-method equality is the requirement.

The primary estimand is within-method:

    τ_int(m) = E_{S,R}[ Y(M₁(m); H_m, σ_c, δ_c, G_c) − Y(M₀(m); H_m, σ_c, δ_c, G_c) ]

so the condition that matters is that `M₀` and `M₁` of the **same** method share
one training budget `H_m`. Across methods, `H_m` may differ.

Forcing every method onto one epoch count would change training algorithms whose
design presupposes their own budget, and would invite exactly the objection the
framework exists to avoid — that standardising the protocol altered the methods.
Horizon sensitivity, if wanted, is a separate analysis on top.

## 2. FairSIN's `M₁` is the published configuration per (dataset, encoder).

Not the parser defaults, and **not** the maximal configuration the code can
express. `experiment.sh` runs `--d='yes'` on German with GCN and GIN, omits it
on Bail, and uses a different script on Credit. So `M₁` is per setting, and
`M₀` is that setting's configuration with its fairness components removed.

    Audit the method that was actually evaluated, not the largest configuration
    the code can express.

Turning `D` on where the published run had it off would produce a model nobody
reported and call it FairSIN. The `D` on/off contrast still runs — as a
**component study**, kept distinct from published-package reproduction.

## 3. 1D stores validation trajectories; test is touched only after selection.

The purpose is not to evaluate every epoch. It is to make several selection
policies applicable **post hoc to one training trajectory**, so that a
difference between them is attributable to the policy and not to a different
run.

    Training → validation trajectory → selector(s) → selected checkpoint(s) → test

Enforced by the API shape, not by discipline:

    select(validation_history) -> epoch      # cannot see test
    evaluate_test(checkpoint)  -> metrics    # cannot influence selection

A selector that is structurally unable to receive test scores cannot leak them.

## Three selectors, kept distinct

    σ_code        what the audited repository actually implements
    σ_published   what the paper or official script intended
    σ_c           the shared selector used for attribution

`σ_code` and `σ_published` usually agree; FairGNN is the known case where they
may not. `σ_c` is

    σ_c = argmin_t  BCE_val(q_t, y)

computed externally by the unified operator for every method, with two-class
models converted to a positive-class probability first. It is called the
**shared fairness-unaware selector** — not "neutral", which would overclaim: it
is one predictive criterion among several, chosen because it does not read the
sensitive attribute.

## Original selectors are re-implemented on the unified evaluator

FairGB's, FairVGNN's and FairSIN's composite score is recomputed from unified
AUC/ΔDP/ΔEO rather than read back from each repository's own metric code. That
separates two things worth keeping apart:

    published-code reproduction     the checkpoint the original code chose
    semantically reproduced selector the checkpoint the same formula chooses
                                     under one metric definition

Any gap between them is a provenance check, not a headline.

## Three regression tests 1D must pass

    instrumentation invariance  logging must not change training. Same seed and
                                configuration, identical final state with and
                                without the logger.
    selector replay             applying a method's own rule to the stored
                                validation trajectory must reproduce the epoch
                                its unmodified code chose. If it does not, the
                                trajectory is missing information.
    test isolation              the selector API must be unable to accept test
                                labels or scores.

---

## Two more, added 2026-09-13 after the NIFTY correction

### 4. `NIFTY_off` is not an intervention-off control

`sim_coeff = 0` changes the training objective **and** the checkpoint selector,
because `sim_coeff` weights the invariance term in both. E11's contrast is
therefore a **training + selection bundle effect**, not a training-intervention
effect, and stays pending unified re-evaluation with everything else.

In the new primary analysis NIFTY is measured like every other method:

    M₁(H, σ_c, δ_c, G_c) − M₀(H, σ_c, δ_c, G_c)

with the fairness component switched off in training only and **both** arms
scored under the same shared selector. NIFTY's own selector is used only in the
selection and package analyses.

The trajectory keeps `val_s_loss` for `M₀` as well, even though `sim_coeff = 0`
makes it zero there. Storing it regardless means the original NIFTY selector can
still be replayed on an `M₀` trajectory — the selection rule must not be welded
to the training flag in our code the way it is in theirs.

### 5. "Initialization uncertainty" is the wrong name for what the seed varies

Two uninstrumented NIFTY runs at the same seed selected epoch 59 and epoch 37 on
GPU. Whatever that is, it is not initialisation: the initialisation was
identical. The seed axis mixes

    initialisation and sampling randomness  +  runtime nondeterminism

so it is recorded as **run-level stochastic uncertainty** until a deterministic
execution path is shown to be available. If one is, the two can be separated and
the narrower name earned. If not, the paper uses the wider name, because
reporting `σ²_init` for a quantity that also contains cuSPARSE reduction order
would be wrong.

This is not cosmetic. The law-of-total-variance decomposition planned for the
uncertainty analysis assigns a name to each facet; one of those names has to be
correct.
