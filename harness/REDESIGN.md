# Redesign — "What Makes a Fair GNN Fair?"

Written 2026-09-13 after a literature review showed the framing this study had
been drifting toward is too strong to defend. This document is the experimental
design that follows from the corrected framing.

---

## The core claim this design exists to support

Everything below serves one combination. If an experiment does not serve it, it
does not belong in the paper.

> **Define a common estimand for fairness-intervention attribution, identify it
> by nested control, and evaluate how stable that attribution is under split,
> initialisation and protocol uncertainty.**

Three parts, each with a job:

**1. The estimand.** For a method `m`, the intervention effect is a vector, not
a score:

        E_int(m) = E(M1, M0) = [ U(M1) − U(M0),  D(M0) − D(M1) ]

    oriented so both coordinates are better when larger. Not scalarised, because
    any weight between utility and disparity is a free parameter and with one in
    hand the ranking can be chosen. Package effect `E(M1, B)` and base effect
    `E(M0, B)` are the same object at different anchors, and on mean differences
    they add exactly: `E_package = E_base + E_int`.

**2. Identification by nested control.** `B -> M0 -> M1`, where `M0` is the
method's *own* system with every component it claims as fairness switched off.
The estimand is identified only if `M0` differs from `M1` in those components
and nothing else — which is a property of each codebase, not an assumption.
Certifying it is what X1's fidelity grades are for, and why two mislabelled
controls in one day (NIFTY's objective read as all of NIFTY, FairGNN measured at
an inherited coefficient) had to stop the study rather than be worked around.

**3. Stability of the attribution.** An attribution computed once is not a
finding. The same estimand is measured across crossed facets — split,
initialisation, incidental preprocessing, backbone family — and reported as a
variance decomposition and a conclusion-stability rate, not as a point.

What this rules out: adding methods for breadth, building a new leaderboard,
proposing a scalar fairness score, and reporting an attribution at a single
protocol setting.

---

## 0. The correction that forces the redesign

The claim this study was heading for —

> current fair-GNN evaluation cannot identify whether reported gains come from
> the fairness mechanism or from the method's base pipeline

— **is false as stated, and a reviewer can refute it with the original papers.**

| paper | control it already has |
|---|---|
| FairGNN (WSDM'21) | GCN↔FairGCN, GAT↔FairGAT, plus estimator / adversarial / covariance ablations |
| NIFTY (UAI'21) | vanilla vs NIFTY on 5 backbones, **and** a 4-way split of objective vs architecture |
| EDITS (WWW'22) | original graph vs debiased graph, same downstream GNN |
| FairVGNN (KDD'22) | GCN/GIN/SAGE vanilla + both modules ablated, singly and together |
| FairGB (KDD'24) | vanilla SAGE, w/o CAL, w/o CNM, full; plus GCN/GIN robustness |
| FairGT (IJCAI'24) | w/o eigenvector selection, w/o k-hop encoding (but not both off) |

The defensible claim is narrower and better:

> **Controls exist, but they are not commensurable.** Each paper defines
> "fairness off" differently, so the cross-method leaderboard does not expose a
> common, comparable estimate of the fairness intervention's incremental effect.
> The field lacks a common *estimand* for mechanism attribution.

### What must be retracted from this study's own notes

**NIFTY.** This study wrote "turning NIFTY's fairness off makes it fairer".
That is wrong. `sim_coeff = 0` removes the counterfactual-invariance
*objective*; NIFTY's other fairness component — Lipschitz constraint via
`spectral_norm` — sits in the projector, predictor and classifier heads
(`algorithms/NIFTY.py` lines 29, 232, 237, 243, 247) and stays active. Verified
in our own copy: `Encoder(base_model='gcn')` uses a plain `GCNConv`, but every
head around it is spectrally normalised.

What was measured is the **incremental contribution of the objective, with the
architectural component held on**. Which is still interesting — the original
paper found both components contribute, and under our protocol the objective's
increment is ≤ 0 — but it is a different statement.

**`total = backbone + mechanism`.** Pareto win counts are not additive and must
not be presented as a decomposition. Report the two comparisons side by side.

**Terminology.** "backbone vs mechanism" breaks on FairGT, where the
architecture modification *is* the claimed mechanism. Use instead:

    benchmark baseline   ->   method-specific base system   ->   full method
           B                           M0                          M1

and two effects, never summed:

    package effect        Q(M1) - Q(B)     what the leaderboard reports
    intervention effect   Q(M1) - Q(M0)    what we actually want

**Preprocessing.** Split the word in two. *Method-prescribed preprocessing* is a
mechanism (EDITS debiases the graph; that IS the method). *Incidental benchmark
preprocessing* is a nuisance variable (feature normalisation, self-loops,
symmetrisation, loader transforms). Finding 14 is about the second.

---

## 1. What is reusable and must not be re-run

| asset | status |
|---|---|
| Audit protocol: 9 settings × 6 splits × 5 inits, paired per cell, Pareto + Holm | keep |
| Preprocessing-unified arm (incidental normalisation forced on) | keep — this is Finding 14 |
| E10 ladder → FairGate's `M0` (λ=0) and its intervention effect | keep |
| E11 `FairGNN_off` (α=β=0) | keep — the literature accepts this as an all-off |
| E11 `GNN_sage` | keep, but re-label: a matched *encoder family* control, not `M0` |
| E11 `NIFTY_off` | keep, **re-label as objective-only removal** |
| Split variance: split moves the number 0.034 vs a claimed effect of 0.011 | keep |

Roughly 12,000 runs are reusable. The redesign adds controls; it does not
restart.

---

## 2. The core table: what `M0` must be, per method

This is the paper. Each row is a different *type* of fairness mechanism, which
is the point — a field-level claim needs mechanism diversity, not more datasets.

| method | mechanism type | `M0` = full method minus intervention | have it? |
|---|---|---|---|
| **FairGNN** | two loss coefficients | α = 0, β = 0 | ✅ E11 |
| **NIFTY** | objective **+** architecture | 2×2: {obj on/off} × {spectral_norm on/off} | ❌ half done |
| **FairGB** | CAL loss + CNM augmentation | w/o CAL, w/o CNM, both off | ❌ needs switches |
| **FairVGNN** | feature-view generation + weight clamping | both modules off | ❌ not in audit |
| **FairGT** | eigenvector selection + k-hop sensitive encoding | both off, same Transformer scaffold | ❌ needs switches |
| **EDITS** | preprocessing *is* the mechanism | original graph, same downstream GNN | ❌ special case |
| **FairGate** | 3-level loss + node allocation | λ_fair = 0 | ✅ E10 |

**FairGT and EDITS are the two that most need this study.** FairGT ablates each
component but never both together, so a mechanism-off Transformer scaffold does
not exist in the literature. EDITS inverts the preprocessing question entirely.

---

## 3. Experiments to run, in priority order

### P1 — NIFTY 2×2 (blocking; a knowledgeable reviewer will check this first)

Four arms, our protocol, reproducing the original paper's factorisation:

    vanilla GCN  |  architecture-only  |  objective-only  |  full NIFTY

Needs: a switch to build NIFTY's heads without `spectral_norm`. Additive,
one flag, no change to any existing path.

If the original paper's conclusion (both components contribute) does **not**
reproduce under 6 splits with unified incidental preprocessing, that is the
strongest single result available: *mechanism attribution is itself
protocol-dependent.*

Cost: 4 arms × 54 cells ≈ 216 invocations, ~6 h.

### P2 — FairGB component ablation (turns a weak control into a real one)

Current control is a plain GraphSAGE, which also differs in training loop, so
`+34` over-states the intervention. Replace with the authors' own factorisation:
w/o CAL, w/o CNM, both off.

Needs: switches in `algorithms/FairGB_alg.py` around the mixup call and the
alignment loss. Cost: 3 arms × 54 ≈ 162 invocations, ~8 h.

FairGB is the **positive control** for the whole paper — the case where the
intervention survives. It has to be measured properly or the paper has only
negative results.

### P3 — FairGT both-off scaffold

The mechanism-off Transformer that the original paper does not report. Needs
switches for eigenvector selection and k-hop sensitive encoding.
Cost: 1–3 arms × 54, ~6 h. Note the 17 GB adjacency cache per Pokec graph, and
that `/home` is at 98%.

### P4 — FairVGNN into the audit + its two-module ablation

Adds a fifth mechanism type and a method whose own paper controls backbone
family well — so it tests whether *backbone-family control alone* is enough,
which is exactly the gap we claim.
Cost: 3 arms × 54, ~8 h.

### P5 — EDITS as the inverted case

Original vs EDITS-debiased graph, same downstream encoder. Here `M0` is the
original graph and the "preprocessing" is the method. Including it is what
stops the paper from being read as "preprocessing is always a confound".
Cost: 2 arms × 54, ~5 h.

### P6 — backbone-family robustness (GraphSAGE, SGC)

Deferred deliberately. Until P1–P2 are done there is no result to test the
robustness *of*. Run on a subset of settings.

---

## 4. The two diagnostics, and the one to refuse

Report per outcome (AUC, ΔDP, ΔEO) separately. Never scalarise across them —
this study's own audit found EDITS scoring ΔDP = 0.000 by predicting one class.

**Attribution Gap.** For outcome Q, the difference between the package effect
and the intervention effect. Says how much of a leaderboard gain is not
explained by the claimed mechanism.

**Conclusion Reversal Rate.** Over (dataset × split) cells: the fraction where
the package-level conclusion is statistically supported but the
intervention-level conclusion is null or opposite. Unit-free, no cross-metric
arithmetic, and it is the number a reviewer will remember.

**Refuse: a Mechanism Contribution Ratio** (`intervention / package`). It
explodes when the denominator is near zero, inverts when the two effects have
opposite signs, and re-introduces exactly the arbitrary scalarisation this paper
criticises. The `91.3% / 4.7%` shares in Finding 16 stay as a descriptive
reading of one method's ladder and are not generalised into a statistic.

---

## 5. What the paper is, in one paragraph

> Fair-graph papers frequently include useful method-specific ablations, but the
> field's headline cross-method comparisons do not expose a standardised,
> commensurable fairness-intervention effect. Different methods define and
> control their base systems differently, while incidental preprocessing,
> model selection and data splits can further alter the observed comparison. We
> distinguish package-level performance from intervention-level attribution and
> measure both under a common control hierarchy across methods whose mechanisms
> differ in kind.

FairGate is not a method in this paper. It is the self-audit: the procedure was
applied first to the authors' own method and falsified its central claim. That
ordering is worth stating explicitly, because it is evidence the protocol was
not built to attack other people's work.

---

## 6. Minimum bar

A field-level claim needs **four or five methods whose mechanisms differ in
kind**, not more datasets:

    FairGate   intervention effect ≈ 0            (null case, self-audit)
    NIFTY      component-dependent, possibly ≤ 0  (needs the 2×2)
    FairGNN    clean coefficient-off              (done)
    FairGB     intervention survives              (positive control, needs real ablation)
    FairGT     architecture *is* the mechanism    (the case the dichotomy breaks on)

Two methods would be "we found something odd in two codebases". Five of
different mechanism types is a claim about the field.

**Total new compute: roughly 33 h across P1–P5**, on top of ~12,000 runs already
banked.

---

## 7. The second axis: uncertainty, and how to measure it without inventing a score

The framework has two axes and they answer different questions.

    attribution uncertainty   is the gain caused by the claimed intervention?
    protocol uncertainty      would that answer survive a different split,
                              a different incidental preprocessing, a
                              different backbone family?

A method can be strong on one and fail on the other, and the current literature
reports neither separately. This section defines both from data the study
already has.

### 7.1 Order of contributions (this ordering is the argument)

1. **Framework.** Two questions the field conflates: does the *package* beat a
   benchmark, and does the *claimed intervention* cause it. Operationalised as
   `B -> M0 -> M1`.
2. **Diagnostics.** Three statistics, defined per outcome, never summed across
   outcomes.
3. **Discovery.** Re-measuring representative methods under one protocol yields
   conclusions the leaderboard cannot express.

An abstract opening "we propose a novel metric" is the weak version. The
statistic is the instrument; the finding is the contribution.

### 7.2 The three diagnostics

Let `Q ∈ {AUC, ΔDP, ΔEO}`, each reported separately, sign-oriented so that
higher is better.

**(A) Intervention Effect.** `IE_Q = Q(M1) − Q(M0)`.
The primary quantity. "FairGate has lower ΔDP than GCN" is a package statement;
"FairGate full and FairGate mechanism-off differ by nothing" is the attribution
statement, and only the second bears on the mechanism.

**(B) Attribution Gap.** `AG_Q = [Q(M1) − Q(B)] − [Q(M1) − Q(M0)] = Q(M0) − Q(B)`.
How much of what a benchmark credits to the method is not explained by the
claimed intervention. Reported per outcome; never combined into one number.

**(C) Conclusion Reversal Rate.** Over conditions `c` (below), the fraction in
which the package-level conclusion is statistically supported while the
intervention-level conclusion is null or opposite. Unit-free, no cross-metric
arithmetic, and it is the number a reviewer remembers.

Refused, as before: any ratio `IE / package`, which explodes near a zero
denominator and inverts when the two disagree in sign.

### 7.3 Protocol uncertainty as a variance decomposition, not a new index

This is the part to take from measurement theory rather than invent. The
observed intervention effect is measured across four facets:

    split s        6 levels   (the field reports 1)
    init r         5 levels
    preprocessing p 2 levels  (as-submitted, incidental-unified)
    backbone b     2 levels   (GCN, SAGE; where the method permits)

Decompose the variance of `IE` over those facets:

    Var(IE) = σ²_split + σ²_init + σ²_prep + σ²_backbone + σ²_resid

and report **σ²_protocol = σ²_split + σ²_prep + σ²_backbone** against
**σ²_init**, the replicate noise a single-split paper implicitly treats as the
only source of error.

This study already has the first comparison for the uniform arm's ΔDP (E3):
between-split sd against within-split sd, per setting —

    pokec_z 0.0368 / 0.0140 = 2.6x     recidivism 0.0233 / 0.0089 = 2.6x
    pokec_z_g 0.0254 / 0.0097 = 2.6x   pokec_n 0.0216 / 0.0149 = 1.5x
    nba 0.0397 / 0.0312 = 1.3x         credit 0.0219 / 0.0176 = 1.2x
    income 0.0128 / 0.0121 = 1.1x      pokec_n_g 0.0180 / 0.0212 = 0.8x
    german 0.0256 / 0.0362 = 0.7x

Against a claimed effect size of 0.011, the protocol facet is the larger term
on most settings.

**Generalizability coefficient.** Treating the method as the object of
measurement and the protocol facets as measurement conditions,

    G_Q = σ²_method / (σ²_method + σ²_protocol/n_c + σ²_init/n_r)

`G` is the proportion of observed variance in measured intervention effects
that is genuinely between methods. A low `G` says the field's rankings are not
reproducible at the resolution it reports them — stated as a reliability
coefficient, which is standard in measurement theory rather than a metric we
made up. This is computable from the existing design with no new runs; the
facets are already crossed.

**Conclusion Stability.** For a reviewer who will not read a variance table:
label each condition's intervention-level conclusion `L ∈ {+, 0, −}` and report
the modal agreement rate across conditions. `CS = 1.0` means every protocol
choice gives the same answer.

### 7.4 The result is a taxonomy, not a ranking

Plot methods on `IE` (x) against `CS` (y). Four regions, each a different kind
of claim:

    IE high, CS high     credible mechanism
    IE high, CS low      real but protocol-sensitive
    IE ≈ 0,  CS high     package-driven — the gain is the base system
    IE ≈ 0,  CS low      unsupported

The conclusion is then **not** "method A is best" but:

> Methods that look similarly strong under conventional evaluation correspond
> to qualitatively different causal stories, and the leaderboard cannot
> distinguish them.

Current placements, provisional and incomplete: FairGate is package-driven
(`IE` net 0 against its own `M0`); NIFTY needs the 2×2 before it can be placed
at all; FairGB is the candidate for credible mechanism and the paper's positive
control; FairGT is architecturally entangled and may not be placeable on this
plot, which is itself worth reporting.

### 7.5 What this adds to the run list

Nothing new to run for §7.3 — the facets are already crossed in the banked
data. What it needs is an analyser that decomposes variance over
(split × init × preprocessing arm) for each method's `IE`, and a
condition-level conclusion table feeding `CRR` and `CS`.

The backbone facet is the exception: `σ²_backbone` needs P6, which is why P6
moves from "robustness appendix" to **a facet of the main result**. It is still
sequenced after P1 and P2, because a facet needs something to vary.

---

## 8. A second incommensurability, found while reading H7: the strength knob

H7's first complete read said FairGNN's intervention does nothing: net −4,
p = 0.52, ΔΔP −0.0028 against its own α = β = 0 control. On Pokec-z our plain
GCN scores ΔDP 0.1058, which matches the original paper's 9.9% for GCN almost
exactly — but our FairGNN scores 0.1220 where the paper reports 0.9%.

That gap is the shape of a reimplementation bug, so it was checked before being
reported. Sweeping α on Pokec-z, split 20:

    alpha    4 (param.json)   AUC 0.6731   ΔDP 0.1117
    alpha   20                AUC 0.6619   ΔDP 0.0901
    alpha  100                AUC 0.6473   ΔDP 0.0693

**The implementation is not broken.** It responds monotonically to the fairness
coefficient and pays accuracy for disparity in the way the method intends. What
is true is narrower: *at the coefficient this benchmark inherited*, FairGNN's
intervention is not measurable; raising it works, at a cost.

### Why this matters beyond FairGNN

The intervention effect is not a scalar per method. It is a function of that
method's own fairness-strength knob — α and β here, `sim_coeff` for NIFTY,
`lambda_fair` for FairGate, α for FairGB. So there is a **second**
incommensurability underneath the one this paper was already about:

    level 1   each paper defines "fairness off" differently
    level 2   each method is evaluated at a strength setting nobody standardised

`utils/param.json` supplies these per dataset — FairGNN gets α = 8 on German,
4 on Pokec-z, 2 on Recidivism — with no recorded provenance, and **no entry at
all for NBA or Income**, where the code silently falls back to α = β = 1. Two
of nine settings therefore ran at untuned defaults, and nothing in the output
records which.

Comparing methods at one arbitrary point per method is what a leaderboard does.
It is the thing this paper objects to. Doing it ourselves, one level down,
would be the same error in a new place.

### Consequence for the design

**The intervention effect must be measured as a frontier, not a point.** For
each method, sweep its own strength parameter, trace the (AUC, ΔDP) curve, and
compare curves. Questions that then become answerable and are not otherwise:

  * Does the method's frontier dominate its own `M0` anywhere, or only at
    settings that cost more accuracy than they buy?
  * Do different methods' frontiers coincide? If they do, the methods are
    points on one trade-off curve and the leaderboard ordering is a choice of
    operating point, not a finding.
  * Is the benchmark's inherited setting on, above, or below each frontier's
    useful region?

This subsumes the λ-sweep already registered as branch D's condition in the E10
outcome map, and promotes it from a follow-up to a requirement.

It also changes what H7's completed numbers mean. They are the intervention
effect **at the benchmark's inherited settings** — which is exactly the quantity
a reader of the published tables is implicitly given, so it is worth reporting
as such, but it must be labelled that way and never as "the method's mechanism
does nothing".

### Cost

One sweep axis per method, 4–5 points, on a subset of settings rather than all
nine — the frontier's shape does not need six splits at every point, though its
endpoints do. Estimated 10–15 h, and it replaces rather than adds to P6, since
a frontier makes the backbone facet cheaper to interpret.

---

## 9. Design check against the refined theory, 2026-09-13

### 9.1 Two defects found while checking the "paired evaluation" condition

The validity conditions say `M0` and `M1` must share the split, the
initialisation schedule, the preprocessing **and the model-selection rule**.
The last one had not been audited. It should have been first.

**Defect 1 — model selection is itself a fairness intervention in four of six
methods, and the baseline does not have it.**

    GNN (plain baseline)   val_loss < best_loss                       no fairness
    NIFTY                  val_loss < best_loss                       no fairness
    FairGB                 auc + f1 + acc − α·(parity + equality)     fairness, weighted
    FairGT                 acc_sp = val_acc − val_dp                  fairness
    FairGate               acc − ρ·dp − (1−ρ)·eo                      fairness
    BIND                   val_dp + val_eo                            fairness

Every package comparison in the audit therefore contains an unmatched
component: the fair methods choose *which epoch to keep* using disparity, and
plain GCN does not. That is a fairness intervention by any reasonable reading,
and it sits outside every ablation switch we have catalogued.

**Defect 2 — FairGB's α does double duty.** At `FairGB_alg.py:162` the same α
that weights the alignment loss also weights the disparity term in the
checkpoint-selection criterion. An α = 0 ablation would therefore change the
training objective *and* which model is returned. That violates isolation, and
the ablation I was about to build would have measured two things at once.

Consequence for a number already reported: FairGB's `mechanism` contrast of
net +34 against a plain GraphSAGE is over-stated by more than the training-loop
difference already noted. The control also lacks fairness-aware checkpoint
selection.

**This is not a nuisance to control away. It is a result**, and it is the
thesis at a level nobody has looked at: the intervention contrast is not
standardised, and part of the intervention is in the selection rule.

### 9.2 Correction to my own uncertainty proposal

Section 7.3 proposed treating split, initialisation, preprocessing and backbone
as crossed facets of one variance decomposition, with a generalizability
coefficient over all four. **The last two do not belong there.** Splits and
initialisations are drawn from a distribution; preprocessing conventions and
backbone families are deliberate design choices, not samples. Averaging over
them produces a number with no referent.

The corrected treatment splits in two:

  * **Statistical uncertainty** — split and initialisation, by the law of total
    variance on the intervention effect Δ:

        Var(Δ) = Var_S( E_I[Δ | S] )  +  E_S[ Var_I(Δ | S) ]
                 ─────────────────────    ──────────────────
                 split uncertainty        initialisation uncertainty

    extended to the covariance matrix for the vector-valued Δ, with joint
    confidence regions from a hierarchical bootstrap clustered on split.

  * **Protocol sensitivity** — preprocessing and backbone, as a *sensitivity
    set* `{ τ_int(p) : p ∈ P }` around one declared primary protocol, reported
    as "does the conclusion hold under p?" and never averaged.

### 9.3 Claim wording

"The field lacks a common estimand" is refutable — Unprocessing (ICLR'24)
raises unmatched base models directly, FFB and ABCFair standardise pipelines,
and Pareto fairness–utility analysis long predates us. The defensible form:

> Existing fair-GNN evaluations standardise outcome metrics, and some
> benchmarks standardise evaluation pipelines, but they do not standardise the
> **intervention-level contrast**: what constitutes a method's no-intervention
> control differs across methods. Cross-method leaderboard gains are therefore
> not directly interpretable as estimates of a common fairness-intervention
> effect.

Short form for the introduction: **the missing common object is not a fairness
metric, but an intervention contrast.**

And explicitly not claimed: the accuracy–fairness vector is not our
contribution. The contribution is where that vector is anchored — `M1` against
its own intervention-off control, rather than against a competitor.

### 9.4 Validity conditions, and what each control can be called

    Isolation                M0 and M1 differ only in the claimed intervention
    Paired evaluation        same split, init schedule, preprocessing, and
                             model-selection rule
    Intervention completeness  every component the paper claims is removed

    Exact control       one coefficient to zero, everything else identical
    Factorial control   several claimed components, each on/off
    Proxy control       no off-switch; approximated by a matched model

**A proxy control never licenses the phrase "mechanism effect".** It yields an
*intervention contrast* and is reported as one. This settles FairGB's caveat by
naming it rather than hedging it.

### 9.5 Work this adds

  1. **Model-selection audit and unification** — a shared selection rule across
     `B`, `M0`, `M1` within each method, plus a run under each method's own
     rule. The difference between the two is a result, not a control.
  2. **FairGB needs α separated** into a loss coefficient and a selection
     coefficient before any ablation of it is meaningful.
  3. **Hierarchical bootstrap** clustered on split, for joint confidence
     regions of the vector effect.
  4. **Sensitivity-set reporting** for preprocessing and backbone, replacing
     the generalizability coefficient over all four facets.
  5. Diagnostics reduce to **Attribution Reversal Rate** and **Attribution
     Stability**. The Attribution Gap is `τ_base`, a term of the decomposition
     rather than a new statistic, and is reported as such.

---

## 10. Work order, rebuilt 2026-09-13

### 10.1 Three things fixed on paper before more GPU time

**(a) Methods have roles, not a flat roster.** The common-coverage set is
Pokec-z and NBA only, and a field-level claim resting on two datasets is weak.
So coverage stops being a constraint on everyone and becomes a property of each
role:

    Core audit            FairGNN, NIFTY, FairGB, FairGT   9 settings
                          → the field-level empirical evidence
    Self-audit            FairGate                          9 settings
                          → motivation and falsification case; EXCLUDED from
                            any cross-method aggregate, because aggregating our
                            own failed method with others inflates the result
    Mechanistic validation FMP                              3 settings
                          → the message-passing bridge
    Structural extension   BeMap                            4 settings
                          → the same framework on a sampling intervention

The estimand is within-method, so unequal coverage does not confound it. **No
single pooled rate across all methods is reported.**

**(b) Two different things were being called "control fidelity".** Split them:

    Original control availability   did the paper/code already provide the off-switch?
                                    Native | Reconstructable | Proxy-only
    Audit control fidelity          does our M0 remove the claimed intervention
                                    and nothing else?
                                    Exact | Factorial | Dependency-constrained | Proxy

    Factorial closure               are ALL component combinations meaningful
                                    and executable?   yes / no

Whether a flag shipped in the original code is a fact about that repository, not
a property of our control. Factorial closure is a separate column because it
gates whether interaction terms and Shapley attribution may be computed at all.

**(c) The estimand and the taxonomy are fixed now, before any result.**

Primary outcome vector, fixed and not revisited:

        primary      ( ΔAUC , −ΔDP )
        secondary    ( ΔAUC , −ΔEO )
        supplementary  accuracy

Four-dimensional vectors make almost everything Pareto-incomparable, so the
primary pair is declared rather than discovered.

Attribution taxonomy — four classes, and **null is not reversal**:

    intervention-supported     package gain and intervention gain agree in the
                               favourable direction
    base-driven / attenuated   package gain present, intervention effect not
                               resolved or near zero
    attribution reversal       package improves, intervention credibly worsens
    protocol-sensitive         the class itself changes with split or protocol

`ARR` counts **only the third class**. Counting the second would let absence of
evidence masquerade as evidence of reversal, which is the first thing a reviewer
would attack.

### 10.2 Not everything is a 2×2

    FairGNN   cov × adv                   2² = 4    closure: yes
    NIFTY     obj × lip                   2² = 4    closure: yes
    FairGB    cal × cnm                   2² = 4    closure: pending α split
    FairGT    eig × sgr × khp             2³ = 8    closure: to check
    BeMap     bal                         single toggle, not a factorial
    FMP       propagation × fairness      graph-specific, see below
    FairGate  3 loss levels × allocation  done (E10 ladder)

A factorial exists when each combination is *scientifically meaningful*, not
when a code switch happens to exist.

**FMP is not two fairness components.** λ₂ gates propagation, λ₁ gates the
fairness projection, giving

    M00  no propagation, no fairness      ← call it the no-propagation control,
    M10  propagation only                   NOT "MLP" unless verified equivalent
    M11  propagation + fairness

and two contrasts that are the paper's GNN-specific bridge:

    τ_prop = Y(M10) − Y(M00)     what message passing does to the utility–fairness state
    τ_fair = Y(M11) − Y(M10)     how much of that the fairness step recovers

`M01` (fairness without propagation) is run only if it is meaningful and
executable; it is not manufactured to complete a square.

### 10.3 The order

    Phase 0   documents above                                  no GPU
    Phase 1   X2 — control engineering
              1a  model-selection audit + unified-selection arm      ← BLOCKER
              1b  FairGB: separate α into loss and selection weights ← BLOCKER
              1c  component flags: FairGB cal/cnm, FairGT eig/sgr/khp, BeMap bal
              1d  two regression tests per flag (below)
    Phase 2   X3 — core factorial attribution: FairGNN, NIFTY, FairGB, FairGT
              **analyse as soon as these four are done**, do not wait for the rest
    Phase 3   FMP graph-specific decomposition (τ_prop, τ_fair)
    Phase 4   X5 — split/init uncertainty, mostly from banked data
    Phase 5   X4 — fairness-strength trajectory, one primary handle per method
    Phase 6   BeMap extension + protocol sensitivity set

1a and 1b block everything: four of six methods select their checkpoint using
disparity while the plain baseline does not, and FairGB's α sits in both its
loss and its selection rule. Any factorial built before those are fixed measures
two things at once.

### 10.4 Every flag gets two tests, not one

    default preservation   with the flag unset, the patched code does what the
                           original did. Bit-identical where determinism allows;
                           otherwise structural (module graph, state_dict keys)
                           or within numerical tolerance.
    intervention removal   with the flag set, the named component actually stops
                           happening — asserted on the object itself, not
                           inferred from the metrics moving.

The second is the one that matters. A `--no_cnm` that silently does nothing
passes the first test perfectly, and today's `uniform_budget` bug — a flatten
placed at one of three return paths — is exactly that failure caught late.

### 10.5 Positioning: build on, do not compete with

FFB and ABCFair standardise evaluation pipelines; Unprocessing (ICLR'24)
directly raises unmatched base models and constraint levels; graph-fairness
benchmarks (KDD'24, KDD'25, PyGDebias) standardise datasets, metrics and
implementations. These are **foundations, acknowledged early in the
introduction**, not competitors buried in related work.

    they ask     how do we compare complete methods fairly?
    we ask       once compared under a credible protocol, what part of an
                 observed gain is attributable to the intervention the method
                 claims as its contribution?

Never claim graph fairness lacks standardised benchmarks. The accurate line:
outcome metrics and increasingly pipelines are standardised; **the
intervention-off contrast is not**.

---

## 11. FairGate is out of the paper, and what replaces its role

### 11.1 Removed

FairGate appears nowhere in the paper — not the title, abstract, introduction,
experiments or appendix. The claim is carried by external methods; FairGate's
results are not needed as evidence, and including them invites the question
"is this a failed method repackaged as a measurement paper?", which takes
attention away from the argument.

E10's ladder and the FairGate rows of E7/E11 stay in the repository as the work
that motivated the framework. They are not results in the paper.

### 11.2 The anti-cherry-picking evidence, scoped to what is true

The intended replacement was the sentence "we fixed the intervention definitions
and primary audit protocol before evaluating the external methods."

**That sentence is false for this study and must not be written.** The commit
timeline:

    09-13 15:12   H7 registered, with priors
    09-13 16:58   H7's external-method results read — FairGB +34, NIFTY −2,
                  FairGNN −4
    09-13 19:26   X1 control inventory
    09-13 20:13   attribution taxonomy and primary estimand fixed
    09-13 20:19   model-selection audit

Intervention-level results for three external methods were seen **before** the
taxonomy was fixed. Git makes this checkable by anyone.

What is true, and is stronger because it is verifiable:

    exploratory    E7 package audit, E11 first-pass decomposition (H7).
                   Motivated the framework. Controls were incomplete — NIFTY's
                   was partial, FairGB's was a proxy.
    confirmatory   the factorial phase and the selection arms. Component
                   definitions, control-fidelity criteria, primary outcome
                   vector, attribution taxonomy and analysis plan were fixed
                   first, and **not one cell has been run.**

The sentence the paper may carry:

> Our exploratory audit motivated the framework. We then fixed the component
> definitions, control-fidelity criteria, primary outcome vector, attribution
> taxonomy and analysis plan before running any confirmatory factorial
> experiment; the specification is timestamped in the repository.

H7 therefore becomes the pilot rather than a result, superseded by the factorial
with proper controls. Reporting it as a pilot is honest and costs nothing: its
FairGB row was a proxy control and its NIFTY row removed only one of two
components, both of which the factorial fixes.

### 11.3 What must be fixed before the confirmatory phase runs

Already committed and timestamped:

    component definitions and M0 per method     X1, 19:26
    control-fidelity criteria                   §10.1(b), 20:13
    primary outcome vector (ΔAUC, −ΔDP)         §10.1(c), 20:13
    attribution taxonomy, ARR strict            §10.1(c), 20:13
    split/init vs protocol treatment            §9.2, 9.5
    primary vs sensitivity analyses             §10.3

Nothing in that list may be revised once a confirmatory cell has run. If a
revision proves necessary, it is recorded as a deviation with the number of
cells already observed, as D1–D5 were.
