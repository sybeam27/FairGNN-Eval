# Pre-registration — the discriminability hypothesis

Written 2026-09-09 while E2 was still running, with **2 of 9 settings** in
`harness/results/e2_signal_swap_R30.csv` (480 of 2160 runs: `pokec_z`,
`pokec_z_g`). The remaining seven settings — including Credit, the only
degree-skewed one, and the three saturated ones — had not been observed. The
commit that adds this file is the timestamp; nothing below may be revised after
the matrix is complete, and any deviation must be recorded in a `## Deviations`
section rather than by editing the text above it.

The reason for the ceremony is specific. The NeurIPS version claimed phi(v) was
a node-level fairness-risk score, which is not an identifiable quantity, and was
marked down for it. The failure mode to avoid now is the adjacent one: choosing
the allocation criterion by looking at the results, and then reporting, on those
same results, that our rule chooses well. Fixing the hypothesis and the test
before the data exists is the cheapest available defence.

## What is being claimed

Not that any particular node-level signal is correct. The signal library is a
replaceable part; the claim is about the procedure that picks from it.

> **H1 (mechanism).** How much a signal's allocation helps on a graph is
> predicted by a *discriminability* statistic of that signal, computable on the
> graph before fairness training begins.
>
> **H2 (usefulness).** Selecting the signal by that statistic beats every fixed
> signal, evaluated leave-one-setting-out.
>
> **H3 (the negative half).** Where no signal is discriminable, no allocation
> helps, and this is knowable in advance — so the rule that allocates fairness
> pressure also says when not to.

H3 is the part a universal account cannot make. It is also the part that makes
the paper falsifiable in an interesting direction rather than only in a boring
one.

## The primary statistic, fixed now

For signal `g` on setting `G`:

    D(g, G) = MI(g(v); s_v)  over v in I,  10 equal-width bins

`I` is the fairness node set (the nodes with an observed sensitive attribute
over which every fairness term is computed). Mutual information, not variance,
is primary, and that choice is not free: `harness/README.md` Finding 2 already
established on the diagnostics alone — before any benefit was measured — that
`r_b` fails to track degeneracy and that MI separates the settings that
degenerate from those that do not, and `core/signals.py` already documents that
the variance proxy fails specifically in the clustered case, where `w_bdry` is
near zero for the interior majority so its variance is small even though it is
the signal that tracks the sensitive partition.

Secondary statistics, reported alongside but not used for any decision:
`Var(g)`, `tie_frac(g)`, `arb_frac(g)`, `gini(g)`, and `MI(1[v in G_gate]; s)`
for the gate indicator at the operating `q_gate`.

## When D is computed, and why that is not cheating

Three of the eight signals (`loss`, `fairgb_group`, `bind_influence`) are
functions of a trained model and cannot be computed on the raw graph. The
decision point is therefore **after warm-up and before fairness training
begins** — a boundary the method already has, since `T_warm` epochs run with
`lambda_fair = 0` regardless. For `w_bdry`, `w_deg`, `w_lhd`, `uniform` and
`random`, D is strictly pre-training and costs nothing.

So the honest cost claim is: **one warm-up pass** to choose among eight signals,
against **eight full trainings** for the validation-selection control below. It
is not "free", and it will not be described as free.

## How each hypothesis is tested

**H1.** Over the 9 x 8 cells, let `delta(g, G)` be the paired difference in test
ΔDP against `uniform` at matched seeds (negative = helped). Test the association
between `D(g, G)` and `delta(g, G)` by Spearman correlation, with a permutation
null that shuffles `D` **within each setting** — settings differ enormously in
overall disparity and in noise, and a test that ignores that structure would
find an association from the between-setting spread alone. 10,000 permutations,
two-sided. Reported with a bootstrap CI over settings.

*H1 is falsified if that CI contains zero.*

**H2.** Leave one setting out. On the eight retained settings, fit nothing more
than the rule `argmax_g D(g, G)`; apply it to the held-out setting; record the
ΔDP it obtains there. Repeat for all nine. Compare against four references, all
under the identical pipeline and the same paired seeds:

| reference | what it is | why it is here |
|---|---|---|
| `uniform` | no allocation | the premise: does allocation help at all |
| `best-fixed` | the single signal with the best mean over the eight retained settings, applied to the held-out one | if this ties the rule, the rule is unnecessary |
| `val-select` | per setting, train all eight and keep the best validation ΔDP | the strongest competitor, and absent from the current draft |
| `oracle` | per setting, the best test ΔDP | upper bound; not a claimable number |

*H2 is falsified if the rule does not beat `best-fixed` with a paired interval
excluding zero.* If it beats `best-fixed` but only ties `val-select`, the claim
narrows — honestly and in the paper — to cost: one warm-up instead of eight
trainings, plus applicability to a graph never trained on.

**H3.** Rank settings by `max_g D(g, G)`. The prediction is that settings in the
bottom of that ranking are the settings where no signal's interval excludes zero.
Reported as the association between `max_g D` and the number of signals with a
`helps` verdict, and read qualitatively — nine settings will not support more.

## Decisions already fixed, and not revisitable

* **Replicates are paired on seed.** All eight signals share seeds 27–56.
  Unpaired comparison is not reported.
* **`R = 30` is a starting point, not a claim.** `analyze_e2.py` recomputes the
  needed R from the observed paired sd. Settings found short are topped up by
  *extending* the seed list (57, 58, ...), never by re-running with different
  seeds and keeping the better result.
* **Holm correction across the whole family** of 63 signal-vs-uniform
  comparisons. A cell whose interval excludes zero but does not survive Holm is
  `undecided*` and is not evidence.
* **German is excluded from every headline average**, and this was decided in
  `harness/README.md` Finding 4 from the noise measurement alone, before any
  benefit was observed: its within-seed sd is 0.0351 against an effect of 0.011,
  giving `R_paired = 326`. It is reported as uninformative. Note this cuts
  *against* us — German is where the draft claims its largest margin.
* **Only `split_seed = 20` is used.** The 30 seeds vary initialisation,
  dropout and edge-drop, not the train/val/test split. Split variance is a third
  noise source, unmeasured here, and the draft's own text reports German
  flipping between first and last across split seeds. Any claim surviving H1–H3
  must then be re-checked on split seeds; results before that check are stated
  as conditional on the split.
* **The signal library is closed at eight**: `uniform`, `random`, `w_bdry`,
  `w_deg`, `w_lhd`, `loss`, `fairgb_group`, `bind_influence`. No signal is added
  after seeing the matrix. `bind_influence` is currently a first-order
  approximation of BIND, not the published estimator; if it is a load-bearing
  result it must be replaced by the official implementation via `adapters/`
  before publication, and until then it is labelled as a reimplementation.

## What would end the project

* No signal beats `uniform` anywhere with an interval excluding zero after
  Holm → allocation does not help; there is no paper on this axis.
* One signal beats `uniform` almost everywhere and `best-fixed` ties the rule →
  the graph-conditional story is wrong; the honest output is a short negative
  report that a fixed criterion suffices.
* H1 holds but `val-select` dominates the rule on both disparity and any
  reasonable cost accounting → the mechanism is real but the method is not
  needed.

Each of these is a four-month saving and will be written up as such rather than
worked around.

## Deviations

Recorded as they are made, after the text above, never by editing it.

### D1 — the primary statistic is permutation-debiased MI, not plug-in MI

*Made 2026-09-09, with 3 of 9 settings observed in E2 (`pokec_z`, `pokec_z_g`,
`pokec_n`). Decided from a property of the estimator that does not involve the
benefit matrix.*

Plug-in MI is biased upward, and the bias grows as the sample shrinks. Measured
on `idx_fair`, MI between a **random** signal and the sensitive attribute is:

    |I| = 4605 (Income)   MI(random; s) = 0.0008
    |I| =  200 (Pokec)                    0.0129 .. 0.0568
    |I| =  100 (German, Recidivism)       0.0230 .. 0.0475
    |I| =   50 (NBA)                      0.1931

A 240-fold spread with no signal content in it. The benchmark's sensitive-label
budgets differ by two orders of magnitude, so this artefact would have inflated
D on exactly the small-|I| settings, by an amount that also differs per signal
(a signal with more distinct values is inflated more). Both halves of H1 would
have been corrupted: the across-setting association and the within-setting
argmax of H2.

The statistic is therefore

    D(g, G) = MI(g; s) - E_perm[ MI(g; pi(s)) ]     200 permutations

Shuffling `s` preserves both marginals and the sample size and destroys any real
dependence, so the permutation mean estimates the bias directly. D is ~0 in
expectation for a signal unrelated to `s` at any |I| and is **not clipped at
zero**; negative values are expected and are reported as they come.

Only the mean of the bias is removed, not its spread. A single draw still
fluctuates by the null sd, so `mi_null_sd` is recorded next to every D and no D
is read without it. On NBA the null sd is 0.042, large enough that a random
signal lands near 0.09. That setting's D is imprecise -- and so is its benefit
measurement, for the same underlying reason -- and the two are read together.

What is unchanged: the bin count (10, equal width), the node set (`I`), the
decision point (after warm-up), the closed signal library, and every test in
H1-H3. The plug-in value is still recorded as `mi_raw` in the output so the
correction is auditable rather than assumed.

An early consistency check, on the three settings then available: on `pokec_z`,
D(`w_bdry`) = 0.016 against a null sd of 0.012 -- indistinguishable from noise --
and `w_bdry` there did not help (+0.0058, 11 of 30 wins). This is the direction
H1 predicts, on a setting where the draft's regime rule says `w_bdry` is the
correct signal. It is one cell and is not evidence; it is recorded because it
was seen before the deviation was made and so belongs in the audit trail.

### D2 — signals that are measurable functions of `s` are inadmissible for the argmax rule

*Made 2026-09-09, with 3 of 9 settings confirmed in E2 (`pokec_z`, `pokec_z_g`,
`pokec_n`). The five settings where this matters most -- Credit, German, Income,
NBA, Recidivism -- had no benefit measurement at the time. Decided from an
algebraic identity, not from any outcome.*

E1 gives `fairgb_group` the largest D on five of nine settings (Income 0.607,
German 0.587, Recidivism 0.446, Credit 0.317). That is a tautology, and it is
exact rather than approximate:

    german  |I| = 100   H(s) = 0.6022   MI(fairgb_group; s) = 0.6022
    credit  |I| = 6000  H(s) = 0.3170   MI(fairgb_group; s) = 0.3170

`fairgb_group` takes four values, one per `(y, s)` subgroup, and is constant
within each -- a property `tests/test_core.py` has asserted since before any of
this was measured. It therefore determines `s`, so `H(s | g) = 0` and
`MI(g; s) = H(s)` identically. The statistic is at its ceiling for a signal that
carries no information about the graph whatsoever.

So D as written conflates two things: a topological signal that happens to align
with the sensitive structure, which is what the hypothesis is about, and a
signal built out of `s` in the first place, which cannot rank nodes at all.

The fix is a scope condition, stated once and checked exactly:

> A signal `g` is **admissible** for allocation iff it is not
> `sigma(s)`-measurable, tested as `MI(g; s) < H(s)`.

This is not a judgement call and not a threshold: it is an equality that holds
or does not. All eight signals are tested; `fairgb_group` is the only one that
fails, and it fails on every setting.

The condition also states what the paper is about rather than merely excluding
an inconvenient competitor. A signal constant within `(y, s)` is a *group*
reweighting -- which is precisely what FairGB (KDD'24) is, and `harness/README.md`
already records that as the dividing line between it and node-level allocation.
Excluding it from a rule that picks a *node-level* criterion is a definition of
scope, not a result.

Consequences, all fixed now:

* `fairgb_group` is excluded from the `argmax_g D` rule of H2 and from the
  primary association test of H1.
* It is **not** dropped from E2. Its paired benefit is reported in full, because
  reproducing an existing method's criterion under our pipeline is one of the
  things E2 is for. If it turns out to beat every node-level signal, that is a
  result against us and will be reported as one.
* H1 is reported twice: primary over the seven admissible signals, and, next to
  it, the inclusive version over all eight. Both numbers appear in the paper.
* No other signal's status changes, and no signal is added.

### D2a — the admissibility test is "varies within a (y, s) cell", not `MI(g; s) < H(s)`

*Made 2026-09-09, with 4 of 9 settings confirmed in E2 (the four Pokec ones).
Credit, German, Income, NBA and Recidivism still unobserved. This corrects the
operational test in D2; the condition D2 states in words is unchanged.*

D2 says a signal constant within `(y, s)` is a group reweighting and cannot rank
nodes. That is the condition. The test it proposed for it, `MI(g; s) < H(s)`,
does not detect it reliably:

    setting   unique values   H(s)     MI(g;s)   constant within (y,s)?
    pokec_z         5        0.6443    0.0011           yes
    german          4        0.6022    0.6022           yes
    credit          4        0.3170    0.3170           yes

On Pokec-z the four subgroup weights happen to collide across `s`, so `g` does
not numerically determine `s` and the MI test passes it -- while `g` still takes
one value per `(y, s)` cell and carries no node-level information at all. The MI
test detects a numerical coincidence; the condition is structural.

The test is therefore the condition itself, checked directly on `I`:

> `g` is admissible iff it varies within at least one `(y, s)` cell.

Exact, setting-independent for a signal defined this way, and already asserted
for `fairgb_group` by `tests/test_core.py`. Under it `fairgb_group` is
inadmissible on every setting, including the four where the MI test let it
through. No other signal is affected and nothing else in D2 changes.

### D3 — the rule may abstain when no signal is distinguishable from noise

*Made 2026-09-09, same point, 4 of 9 settings observed.*

On Pokec-z every D lies inside the permutation null: `w_bdry` 0.016, `w_deg`
0.005, `bind_influence` 0.024, against a null sd of ~0.011, and the largest of
them is `random` at 0.032. A plain `argmax_g D` there does not select a signal;
it selects a fluctuation, and on that fold the LOSO rule duly picked `random`.

Reporting that as the rule's behaviour would be testing a straw man, because the
pre-registration's own H3 already asserts that a setting with no discriminable
signal should not be allocated at all. A rule that cannot express "none of
these" cannot implement H3.

    g*   = argmax_g D(g, G)   over the k admissible candidates
    rule_abstain(G) = uniform,   if  k * p_perm(g*, G) >= 0.05
                      g*,        otherwise

`p_perm` is the share of permutations reaching the observed MI, computed from
the same null already used to debias D. It adds no fitted parameter. The factor
`k` is there because `g*` is not a signal fixed in advance: it is the largest of
`k` statistics, so its unadjusted p-value is not the probability that the best
of them exceeds noise. Bonferroni over the candidates is the conservative form
of that adjustment and introduces nothing to tune.

*Amended twice the same day, both times before the rule was applied to any
unobserved setting, and both times because the criterion failed on Pokec-z --
the one case then visible where every D sits at noise level and abstention is
the correct behaviour.*

First form, `2 * sd_null`: wrong, because the null distribution of MI is
right-skewed and a multiple of its sd corresponds to no tail probability. It let
`random` through at D = 0.032 against `2 * sd = 0.024`.

Second form, an uncorrected `p_perm < 0.05`: still wrong, and for a more
interesting reason. `random` on Pokec-z has `p_perm = 0.014`. That is not a
defect of the permutation test -- across 72 cells a p of 0.014 is unremarkable --
but the rule does not test a signal chosen in advance. It tests the largest of
`k` statistics, and the largest of `k` draws from a null exceeds any fixed
quantile far more often than one draw does. Ignoring that is the same selection
error the Holm correction already guards against elsewhere in this document.

`n_perm` is raised from 200 to 500 so the p-value has usable resolution; the
debiasing is unaffected.

Both variants are reported. `rule` (plain argmax) remains the primary, because
it is what was pre-registered; `rule_abstain` is secondary and labelled as
specified after seeing that D can be uniformly at noise level on a setting --
which is a fact about the statistic, not about any benefit. If the two differ,
both numbers appear, and the difference is itself the evidence for H3.

## Outcome — recorded 2026-09-09, on the complete 9 x 8 matrix (2160 runs)

Stated against the tests fixed above, before any reinterpretation.

**H1 is falsified.** Spearman between D and the paired benefit over the 54
admissible cells is **-0.074**, permutation p = 0.64, bootstrap CI over settings
**[-0.483, +0.416]** -- it contains zero, which is the falsification condition
written above. Excluding German does not rescue it (-0.238, CI [-0.540, +0.179]).
Discriminability as defined does not predict benefit.

**H2 is falsified.** Leave-one-setting-out, `rule - best_fixed = +0.0008`,
CI [-0.0037, +0.0054], which does not exclude zero. The abstaining variant is
no better (-0.0002, CI [-0.0034, +0.0031]). Fold means:

    rule 0.0435   rule_abstain 0.0425   best_fixed 0.0426
    val_select 0.0377   oracle 0.0354   uniform 0.0450

**H3 is not supported, and the sign runs the wrong way.** Spearman between
`max_g D` and the number of signals that help is -0.209 (p = 0.59). The three
settings with the *highest* `max_g D` -- NBA 0.336, German 0.180, Income 0.135 --
have **zero** helping signals between them, while Credit (0.082) has two.

### What did survive

**The ceiling is real and significant.** `oracle - uniform = -0.0096`,
CI **[-0.0165, -0.0027]**, over nine folds. Choosing the right signal per graph
beats uniform weighting by about 21% of its disparity. The axis the project is
built on is not empty; what fails is our way of finding the choice.

**No signal dominates**, which was the other necessary condition. `w_deg` helps
on Credit (-0.0160) and Recidivism (-0.0078); `bind_influence` helps on Credit
(-0.0230) and Pokec-z (-0.0244) and *hurts* on Recidivism (+0.0051).

**Validation-based selection recovers most of the ceiling but is not
significant at nine folds**: `val_select - uniform = -0.0073`,
CI [-0.0159, +0.0013] -- 76% of the oracle gain, interval still touching zero.

### Two reasons this negative result is not yet clean

**Power.** Only 22 of 63 cells were adequately replicated at R = 30, and 37 of
the 54 undecided cells have `R_needed > 30`. "86% undecided" is therefore partly
a statement about the budget, not about the effect. German (366) and NBA (314)
are hopeless; Pokec, Credit and Pokec-n-g need 43-94 and are reachable by
extending seeds.

**Confounding -- checked, and absent.** This slot first held a second caveat:
that `mean(phi) = 1` does not fix the fairness pressure, so the comparisons were
placement confounded with strength. It was tested the same day and is
**withdrawn**. `trainer.py`'s scale calibration already makes the fairness
gradient scale-free, and the effective contribution `lambda*gamma*L_pred` lands
in 0.068-0.081 across signals whose raw `L_pred` differs sevenfold
(`harness/README.md` Finding 5). The falsifications above therefore carry one
caveat, not two, and power is it.

### What this fixes about the paper's shape

The problem is now well posed in a way it was not this morning: there is a
verified 0.0096 available from per-graph selection, an oracle that attains it,
a validation-based selector that reaches 76% of it, and a pre-training selector
that reaches none of it. That is a sharper object than "allocation helps", and
it is stated with the failure of our own proposal in it.

### D4 — three operator-matched structural signals are added, and why that is not a fishing expedition

*Made 2026-09-09 with 5 of 9 settings observed in E4 (the four Pokec settings
and Credit). Recidivism, Income, German and NBA were unobserved. The three
signals were implemented and this entry written before any of their benefit was
measured on any setting.*

The pre-registration closes the library at eight and says no signal is added
after seeing the matrix. That rule exists to stop exactly the move this section
could be mistaken for: structural signals lost, so add structural signals until
one wins. The following is admissible only because its justification does not
refer to any benefit, and it must be read against that standard rather than
against how reasonable the signals sound.

**The defect, established without outcomes.**
`harness/e0_diagnostics/signal_scale_check.py` trains nothing and reads no result.
It compares `w_bdry` -- a one-hop, equally-counted cross-group neighbour ratio --
against cross-group mass computed on `A_hat = D^-1/2 (A+I) D^-1/2`, the operator
the backbone is defined with, at one and two hops. Since `allocate()` keeps the
top `1 - q_gate`, the quantity that matters is the overlap of the node sets each
version would gate:

    at one hop     Jaccard median 0.86, Spearman 0.90-1.00
    at two hops    Jaccard median 0.67, range 0.22-0.88

The degree weighting is not the problem; the hop count is. A two-layer GCN mixes
over two hops and `w_bdry` looks at one. On the two settings with `h ~ 0.48` the
two measurements are close to unrelated (Jaccard 0.22-0.30, Spearman 0.29-0.37),
so there E4's verdict on `w_bdry` is not a verdict on cross-group exposure at
all. On the other seven roughly a third of the gated set differs.

This is the same footing as D1 (MI's finite-sample bias, shown with a random
signal) and D2a (the `(y,s)`-cell test, shown with an algebraic identity): a
property of the statistic, demonstrated on the graph alone.

**The signals.** One principle -- *measure the signal on the operator the model
uses* -- and one candidate per existing structural signal it repairs. No signal
is added that does not repair a specific one.

    prop_cross  S1  cross-group share of the mass A_hat^h aggregates
                    -> repairs w_bdry's depth
    prop_dev    S3  |receptive-field group composition - graph-wide rate|
                    -> repairs w_lhd's depth and weighting
    prop_sens   S2  ||(A_hat^h - A_tilde^h)_v X||, A_tilde on the intra-group
                    subgraph -> the feature-weighted version of the same idea

`h` is the backbone depth, 2, not a tuned parameter: it is read off the model.
All three are structural -- no training, no `state`, identical across seeds,
which the output makes checkable. The library is now eleven and closes again
here.

**What this changes in the tests.** The Holm family grows from 63 to
9 x 10 = 90 signal-vs-uniform comparisons and every reported correction uses the
larger family; nothing is compared against the old, smaller one. The
admissibility condition of D2a applies unchanged (all three vary within `(y,s)`
cells). Any cutoff these signals might need is fixed on synthetic graphs and
frozen before the nine settings are touched.

**Stopping rule, fixed now.** If none of S1-S3 beats `bind_influence` on any
setting with a paired interval excluding zero under the 90-comparison Holm
family, the structural line is closed and the conclusion is recorded as
*topology-based allocation does not work on this benchmark; a model-dependent
influence estimate does*. In that case no further structural signal is proposed,
and the finding is written up as the negative result it is -- including that our
own draft's central signal is among the ones that failed.

**Prior, stated before the result.** The three signals E4 has tested so far lost
every one of the 35 paired comparisons against `bind_influence` over five
settings. Repairing the measurement makes `w_bdry` less wrong; it does not
supply a reason to expect it to beat a signal that reads the model's own error.
The expected outcome is therefore that the stopping rule fires. It is being run
because "a correctly measured structural signal also loses" is a far stronger
sentence than "our structural signal lost", and both papers need it.

## Outcome II — the split-robust matrix (E4, 2160 runs). The graph-conditional claim is dead.

E2 repeated with the replicate redefined as a (split, init) cell: 6 splits x 5
initialisations, the same 30 replicates, paired within each cell. Everything
below therefore generalises to the dataset rather than to `split_seed = 20`.

**`bind_influence` is never beaten.** Across all 9 settings x 7 competing
signals = 63 paired comparisons: **0 losses**, 34 ties, 29 wins. Not one signal,
on any graph, beats it.

**Allocation helps, and by a lot.**

    uniform      0.0468
    best_fixed   0.0347      -0.0122  [-0.0228, -0.0016]   significant
    oracle       0.0341      -0.0127  [-0.0230, -0.0025]   significant

A 26% reduction in mean demographic parity, with the interval excluding zero.
This is the strongest positive result the project has produced, and it survives
re-splitting.

**Per-graph selection adds nothing.**

    oracle - best_fixed = -0.0006  [-0.0015, +0.0004]   undecided

`best_fixed` chooses `bind_influence` on all nine folds; the oracle chooses it
on seven and `w_lhd` on two, by margins of 0.0013 and 0.0037. The entire gap
between "always use one criterion" and "know the right answer for each graph" is
six ten-thousandths, and its interval contains zero.

This is the second stopping condition, quoted from above: *"One signal beats
uniform almost everywhere and best-fixed ties the rule -> the graph-conditional
story is wrong; the honest output is a short negative report that a fixed
criterion suffices."* It is met. **D2, the contribution the project was built
on, is dead** -- not underpowered, not confounded, but measured at 2160 runs
across six splits and found to be worth 0.0006.

**Almost nothing else is decided.** Of 63 cells, 3 help (`bind_influence` on
Credit and Pokec-z, `w_deg` on Credit), 1 hurts (`w_lhd` on German), and 59 are
undecided. The four decided cells at split 20 that involved Recidivism are all
gone: `w_deg` there was 27 of 30 paired wins at Holm p < 0.0001, and across
splits it is +0.0031, undecided.

### What the paper is now

Not "which criterion, conditioned on the graph". The findings that survive are:

1. Node-wise allocation of a fairness regulariser reduces mean ΔDP by 26%
   against uniform weighting, split-robustly.
2. The criterion that does it is a model-dependent influence estimate. Every
   topological criterion tested -- cross-group exposure, degree, local-homophily
   deviation, all three central to this literature and to our own draft -- fails
   to beat uniform anywhere except `w_deg` on Credit, and none beats influence
   anywhere.
3. Choosing per graph is not worth doing.

(2) is a negative result about the field's premise and it is sharp. (1) is
positive but belongs to an existing method's criterion, used as a soft weight
rather than as node deletion, so **the official BIND estimator now has to
replace our first-order approximation before any of this is claimed** -- it is
no longer one signal among eight but the entire positive result.

### Still open, and now the only things worth running

* **S1-S3** (deviation D4, pre-registered before this outcome was known). The
  stopping rule fires unless one of them beats `bind_influence` somewhere. Worth
  the hour because "a correctly measured structural signal also loses" is the
  sentence (2) needs.
* **Official BIND**, and the soft-versus-hard comparison it enables: BIND
  deletes harmful nodes, we weight them. Nobody has compared the two.
* Power: 59 of 63 cells remain undecided at R = 30 with split variation.

## Outcome III — the stopping rule of D4 fires (E5, 810 runs; 2970 with E4)

Eleven signals on the same (split, init) cells, Holm over the enlarged family of
9 x 10 = 90 comparisons, exactly as D4 fixed before the answer was known.

**No operator-matched structural signal beats `bind_influence` anywhere.** Of
the 27 head-to-head cells, 0 wins, 18 ties, 9 losses. The rule as written fires,
and the structural line is closed. The prediction recorded in D4 -- that it
would fire -- holds.

**The depth correction is real but only removes harm.** It was pre-registered as
a repair of a measured defect, and it repairs it:

    setting     gate overlap   w_bdry    prop_cross   change
    pokec_n_g       0.224      +0.0024     -0.0012    +0.0036
    pokec_z_g       0.304      +0.0097     -0.0004    +0.0102
    pokec_z         0.622      +0.0034     +0.0077    -0.0043
    pokec_n         0.690      +0.0036     +0.0001    +0.0035
    credit          0.671      -0.0115     -0.0165    +0.0050

On Pokec-z-g, where `w_bdry` and the operator-matched version gate almost
disjoint node sets, `w_bdry` significantly *hurts* (+0.0097, CI [+0.0036,
+0.0158]) and `prop_cross` is neutral (-0.0004). The two settings with the
lowest gate overlap show the largest repair. That is the direction Finding 7
predicted, on four points, and it is not claimed as more than that.

What it is not is a route to beating influence. Fixing the measurement moves a
harmful structural signal to neutral; it does not move it past a signal that
reads the model's own error. `prop_cross` ties `bind_influence` on six settings
and loses on three, and wins nowhere.

**The full eleven-signal matrix decides almost nothing.** 3 helps
(`bind_influence` and `prop_cross` and `w_deg`, all on Credit), 2 hurts
(`prop_sens` and `w_lhd`, both on German), 85 undecided of 90, at R = 30 with
split variation.

### The recorded conclusion

*Topology-based allocation does not work on this benchmark; a model-dependent
influence estimate does.* Every topological criterion in this literature was
tested -- cross-group exposure, degree, local-homophily deviation -- at one hop
as the field defines them and at the backbone's own depth after a measured
defect was repaired. None beats uniform anywhere except on Credit, and none
beats influence anywhere. That includes the signal at the centre of our own
draft.

The library closes at eleven. No further structural signal is proposed.

## Design hypotheses — registered 2026-09-11, with E7 at 210 of 378 cells

Written while the audit is incomplete: `FairGT` is mid-run and `FairGB`,
`FairGate` and `BIND` are unobserved. Nothing below may be revised once those
rows exist.

### Why the earlier conclusions are being redone

Every result in E2-E5 was read off ΔDP alone. That is the failure the paper's
own appendix warns about -- "a model may reduce fairness violations simply by
making overly conservative predictions" -- and it is not hypothetical here:
EDITS scored ΔDP = 0.000 on Income and NBA in the submitted table by predicting
one class. Re-reading E4/E5 on the (AUC, ΔDP) Pareto front changes the answers:

    signal            Pareto-dominates uniform   ΔDP-only win rate   mean ΔAUC
    bind_influence            23.0%                   63.7%          -0.0110
    prop_cross                21.9%                   45.9%          +0.0013
    random                    20.7%                   45.9%          +0.0031
    w_bdry                    17.8%                   40.4%          -0.0008

and head-to-head against `random` on matched cells, 270 each:

    bind_influence  +29  (p = 0.009)     the only signal that beats random
    w_lhd            +4  (p = 0.75)
    uniform          -5  (p = 0.70)      uniform is not distinguishable from random
    w_deg           -21  (p = 0.053)
    prop_dev        -22  (p = 0.028)
    w_bdry          -31  (p = 0.004)     significantly *worse* than random

So the E4/E5 finding stands but hardens: topological allocation does not merely
fail to beat influence, `w_bdry` -- this literature's canonical signal, and our
own draft's -- allocates worse than assigning weights at random. And the one
signal that works buys its ΔDP with accuracy: `bind_influence` is the only
signal whose mean AUC cost is not ~0.

**All Pareto statements from here use (AUC up, ΔDP down)**, dominance counted
per (split, init) cell, with a sign test over the cells that are comparable.
No scalarisation: any weighting of accuracy against disparity is a free
parameter and would let the conclusion be chosen.

### H4 — soft weighting Pareto-dominates hard deletion of the same score

BIND estimates per-node influence and deletes the harmful nodes. Finding 9
measured that estimate across initialisations and found the deletion set
unstable to the point of sign reversal (German, seeds 28 vs 29: Spearman -0.800,
top-30% Jaccard 0.02 against a chance level of 0.18). A discrete deletion cannot
recover from a wrong estimate; a continuous weight degrades in proportion to it.

> **H4.** Using influence as a bounded continuous weight Pareto-dominates using
> the same influence to delete nodes, more often than the reverse.

The hypothesis comes from a property of the estimator, measured without
reference to any benefit -- the footing of deviations D1, D2a and D4. It
introduces no tuned parameter: both arms consume the same influence values, and
the weighting bounds (`phi_min` 0.5, `phi_max` 2.0) are already fixed across
every experiment in this study.

*Test.* Both arms on the same (setting, split, init) cells, official BIND
estimator for both. Sign test on Pareto outcomes. **Falsified if the soft arm
does not win with p < 0.05**, and equally if it wins on ΔDP while losing on the
Pareto count -- that would be the same mistake in a new place.

### H5 — allocation under an accuracy constraint dominates unconstrained allocation

`bind_influence` is dominated by uniform in 13.0% of cells while dominating in
23.0%. If those 13% are accuracy losses rather than disparity failures, then an
allocation that refuses to spend accuracy should convert part of that 13% into
the incomparable or dominating column.

> **H5.** Scaling the fairness term back whenever validation AUC falls more than
> `tau` below the uniform-weighted run raises the Pareto dominance rate over
> uniform, without lowering it below the unconstrained arm's.

*The free parameter is the problem.* `tau` cannot be chosen by looking at the
nine settings; that is what every other guard in this document exists to
prevent. It is fixed at **`tau` = 0.005**, the value `adaptive_auc_tol` already
carries in FairGate's own published configuration
(`outputs/ours/exp_fairgate_fiw_v1.csv`), so it is inherited rather than tuned.
If a sweep over `tau` is ever run, it is reported as a sensitivity analysis and
never as the headline.

*Test.* Constrained against unconstrained, same cells, sign test on Pareto
outcomes against uniform. **Falsified if the constrained arm's dominance rate
is not higher.**

### What is deliberately not being done

No new allocation signal. The stopping rule of D4 closed that and the Pareto
re-reading strengthens the reason: the structural family allocates worse than
random, so a better member of it is not the missing piece.

No design fitted to the nine settings. H4 and H5 both come from measured
properties -- estimator instability, accuracy cost -- not from which cells
happened to win. If either needs a parameter that the nine settings would have
to choose, it is set on synthetic graphs and frozen first, per the plan
`harness/HANDOFF.md` recorded before any of this.

### Prior, recorded before the result

H4 is expected to hold, weakly: the instability is large but the deletion
budgets observed so far are small (German k = 0-1), so there may be too little
deletion for the two arms to separate. H5 is expected to hold on the dominance
rate and to *lower* the ΔDP-only win rate -- and if that trade is what happens,
it is the point rather than a disappointment.

### D5 — the audit gets a second arm with feature normalisation forced on

Registered 2026-09-12, with the unified arm at 26 of 162 cells and no
cross-arm comparison yet computed. What has been seen at the time of writing is
one smoke cell (German, split 20) used to verify the flag reaches the branches.

The comparison this study audits does not share preprocessing. GNN and NIFTY
load all nine settings unnormalised; FairGNN normalises NBA and German only;
FairGB, FairGT and FairGate take `get_dataset`'s default and normalise
everywhere. The protocol section lists datasets, splits, seeds, learning rate
and weight decay as held in common. Preprocessing is not on that list, and is
not in fact held in common.

This matters because it is confounded with the audit's own headline. The three
methods that Pareto-dominate plain GCN -- FairGate 0.815, FairGT 0.704,
FairGB 0.685 -- are the three that raise AUC by 0.043 to 0.076, and they are
also three of the four that normalise. A direct measurement puts normalisation
at +0.074 AUC on Income, -0.050 on Credit and -0.003 on Pokec-z: the same order
as the between-method gaps, and not the same sign across settings. Nothing in
the as-submitted arm separates the fairness intervention from the loader.

So: `--force_feature_normalize 1`, every method normalised, a separate cache
and CSV. Only GNN, NIFTY and FairGNN change; the other four already normalise
and their rows are carried over rather than re-run, because re-running a cell
whose configuration did not change can only add sampling noise to a comparison
that should be exact.

**The quantity reported is the difference between the arms, per method.** Not
the unified arm's dominance table on its own. GNN is itself one of the three
methods the override moves, so the unified table is a different comparison, not
a cleaner version of the same one, and a reader needs both columns.

**Prior expectation, recorded before the difference is computed.** If the three
dominating methods' margins are largely preprocessing, their dominance rates
fall substantially in the unified arm and the audit's headline weakens --
including FairGate's, which is ours and has the highest rate. If the margins
survive, the as-submitted reading stands and this arm has closed a hole a
reviewer would otherwise open. I do not know which, and the reason for running
it is that I cannot tell from the as-submitted arm by construction.

**What will not be done.** Normalisation is not uniformly beneficial, so some
method will lose here. That loss is reported as measured. No arm is selected
after the fact as the one to present, and neither arm is described as the
correct one: the finding is that the answer depends on a step the protocol
never fixed.

**Scope.** This holds feature normalisation fixed, not preprocessing in
general. BIND is excluded -- it runs its own published pipeline, and was never
part of the comparison being audited.

**Correction to D5, 2026-09-12, with GNN's 54 cells complete.** D5 above cites
Income +0.074, Credit -0.050, Pokec-z -0.003 as the size of the normalisation
effect. Those came from an ad-hoc test that trained a fixed 500 epochs with no
patience and read the final epoch. Its Credit unnormalised run does not
converge -- validation loss 0.78, 0.96, 2.82, 0.70, 1.02, 1.22 -- so the epoch
it happened to read decided the answer. Re-measured under this study's own
protocol (six splits, five inits, early stopping, nine settings), the effect on
plain GCN is Income +0.059, Pokec-z +0.002, **Credit +0.024** -- a sign
reversal -- and Recidivism **-0.160**, larger than any of the three the old
test looked at and negative. Mean over settings: -0.007.

The prior expectation recorded in D5 stands as written and is not revised; it
was recorded before the difference was computed and revising it now would
defeat the point of recording it. But the direction it assumed is now known to
be incomplete: forcing normalisation makes the *baseline* worse on average, so
on Recidivism the unified arm should raise the fairness methods' dominance
rates rather than lower them. Both directions are live, and that is the finding
-- not that one arm is correct.

## H6 — is the node-wise allocation doing anything? Registered 2026-09-12

Registered before any arm of E10 has been run. E8 is stopped at 98 of 210 cells
and E9 has not started; neither informs what follows.

### Why this is now the central question

Two results in this study point in opposite directions and both are measured.

  * FairGate Pareto-dominates plain GCN on 75.9% of cells with preprocessing
    unified, Holm p < 0.0001 (Finding 14). It wins, and not because of its
    loader.
  * The signal it allocates on is worse than allocating at random. `w_bdry` --
    this literature's canonical structural signal and our own draft's -- loses
    to `random` by net -31, p = 0.004 (E4/E5, Pareto). Of eleven signals only
    `bind_influence` beats random, and it buys ΔDP with accuracy.

So the method wins and its stated mechanism is falsified. A paper cannot be
submitted in that state, and no amount of further signal search resolves it:
deviation D4's stopping rule already closed that search.

The question that resolves it is not which signal is best. It is whether the
allocation machinery contributes anything at all over spending the same
fairness budget uniformly.

### The confound that makes the existing ablation unusable

`utils/model_fairgate.py` already ships a `fiw_weight_mode="uniform"` ablation.
It returns `ones(N)`. Measured 2026-09-12, the allocated weights have mean

    german 0.7117    credit 0.5713    pokec_z 0.5678

so that ablation applies **1.4x to 1.75x more total fairness pressure** than the
arm it is compared against, at the same `lambda_fair`. It varies where the
pressure goes and how much of it there is, simultaneously. Any conclusion drawn
from it about allocation is unattributable — and this is precisely the invariant
(`mean(phi) = 1`, `allocate.py`) that the study harness was built to enforce and
that the published model does not.

A new mode, `uniform_budget`, computes the allocation and replaces it by a
constant equal to its own mean: same sum over nodes by construction, zero
variance. Verified on German: allocated sum 711.7 std 0.381, uniform_budget sum
711.7 std 0.000, published `uniform` sum 1000.0.

### The arms

All four are the published FairGate at its recorded configuration, differing
only in `fiw_weight_mode`:

    full             allocated                       budget B
    perm             same weight multiset, permuted  budget B   (matched_random_perm)
    uniform_budget   no allocation                   budget B
    uniform          no allocation                   budget 1.4-1.75 B  (as shipped)

`perm` is the sharper control: it holds the *distribution* of weights fixed and
destroys only which node gets which. If `full` beats `uniform_budget` but not
`perm`, the benefit is in having a spread of weights at all, not in the ranking
the signal produces.

> **H6.** `full` Pareto-dominates `uniform_budget` on more cells than the
> reverse.

*Test.* 9 settings x 6 splits x 5 inits, paired on the cell, Pareto on
(AUC up, ΔDP down), sign test, Holm over the three comparisons against `full`.
**Falsified if `full` does not win with p < 0.05 after correction** — and, as
with H4, equally falsified if it wins on ΔDP while losing the Pareto count.

### Prior, recorded before the result

I expect H6 to be **falsified**: `full` and `uniform_budget` indistinguishable.
The reason is E2-E5, where `uniform` is not distinguishable from `random`
(net -5, p = 0.70) and ten of eleven signals fail to beat `random`. If the
allocation carried the effect, that pattern could not hold.

I expect `uniform` (the shipped ablation) to look *worse* than `full` on ΔDP
and possibly better on AUC, because it is a different fairness budget rather
than a different allocation — and if the published ablation was read as
evidence for allocation, that is what it was measuring.

### What is decided by the outcome

If H6 is falsified, the claim this study can defend is that the node-wise
weighting apparatus is not what makes these methods work — the regulariser is —
and the evidence is the audit already collected. That is a positive, falsifiable
claim about the literature, not a negative result about FairGate.

If H6 holds, the claim is narrower and the question becomes which signal, with
the constraint that the structural family is already excluded by D4.

Either way this is registered before the data exist, and the arms are fixed
here. No arm may be added or dropped after the first cell runs.

### H6 extended to the full component ladder — registered 2026-09-12, before any cell

H6 as registered above asks one question: does the allocation add anything over
the same budget spent uniformly. That question is necessary and not sufficient.
If it is falsified alone, the paper says "the method wins and we have shown one
reason it is not" — and the first reviewer question, *why does it win then*,
has no answer.

FairGate is not only a weighting scheme. Its intervention is three-level --
structural consistency, representation alignment, prediction alignment -- an
extension of prior work to classification, and the weighting sits on top of
that. Testing only the weighting leaves the rest unmeasured.

So H6 becomes the fourth rung of a ladder, and the whole ladder is registered
here before any of it runs.

    arm            ablation_mode   phi              the step below -> this rung adds
    backbone       none            --               (GCN, lambda_fair = 0)
    out            out_only        uniform_budget   prediction-level DP/EO
    rep_out        rep_out         uniform_budget   representation alignment
    all_uniform    full_loss       uniform_budget   structural consistency
    all_alloc      full_loss       continuous       the allocation           <- H6
    all_perm       full_loss       permuted         (control for all_alloc)
    all_uniform1   full_loss       ones             (what the shipped ablation measures)

Every rung carrying a phi uses `uniform_budget`, so rungs 1-3 spend exactly the
budget `all_alloc` spends. Without that, a step up the ladder would change the
components and the budget together and no step would be attributable -- the same
defect that makes the shipped `uniform` ablation unusable.

**Direction.** This is an ablation downward from a finished method, not a search
upward. Building up and keeping whatever helps would let the nine settings pick
the design, which is what every other guard in this document exists to prevent.
No rung may be added, dropped or reordered once the first cell runs.

**Backbone.** GCN, on all nine settings, because FairGate's recorded
configuration is GCN on all nine and so is every other method in the audit.
The top rung has to *be* the method measured at 75.9% dominance in Finding 14,
or the ladder connects to nothing. GraphSAGE is a robustness appendix to be run
after the ladder is settled, never before it -- a ladder on a backbone no
published number used would answer no question anyone asked.

**Protocol.** 9 settings x 6 splits x 5 inits, the audit's own, paired on the
cell. Pareto on (AUC up, ΔDP down), sign test, Holm across the ladder's
comparisons.

**Registered priors, before any cell.**

  * `all_alloc` vs `all_uniform` (H6): falsified, i.e. indistinguishable. In
    E2-E5, `uniform` is not distinguishable from `random` (net -5, p = 0.70) and
    ten of eleven signals fail to beat `random`. The allocation cannot be
    carrying the effect and produce that pattern.
  * `all_uniform` vs `out`: the larger step. If the multi-level intervention is
    what this method contributes, it shows here.
  * `all_uniform1`: differs from `all_alloc` mostly in budget, not allocation,
    so if the shipped ablation was read as evidence for allocation, this is what
    it was reading.
  * `backbone` -> `out`: large ΔDP gain bought with accuracy. Whether it
    Pareto-dominates is the open part.

**What each outcome licenses.** If the ladder's mass is in `out -> all_uniform`
and `all_uniform -> all_alloc` is flat, the defensible claim is that the
multi-level intervention is what works and the node-wise weighting apparatus the
literature has been elaborating is not. If the mass is in `backbone -> out`
alone, the claim is sharper and costs us more: a standard prediction-level
regulariser is the whole effect. Both are positive, falsifiable claims. Neither
is chosen after the fact.

### E10 outcome map — registered 2026-09-12, with the ladder at 168 of 378 cells

Four settings of nine are complete (German, NBA, Recidivism, Income) and have
not been read past a plumbing check: arm-level means were inspected once to
confirm no arm had collapsed, and the reader was dry-run. No comparison in this
section has been computed.

This map exists so that the paper the result produces is fixed before the result
is. Each branch names what may be claimed and what must be run next. A branch
that is unfavourable to our own method is written here in the same detail as the
favourable one, because the temptation the guard is against -- adjust lambda and
look again -- arrives only when the answer is unwelcome.

**A. Mass in `out -> all_uniform`, `all_uniform -> all_alloc` flat.**
Claim: the multi-level intervention is what works; the node-wise weighting
apparatus is not distinguishable from spending the same budget uniformly.
The contribution survives, the negative result points outward. Next: backbone
robustness, and fill `struct_only` / `struct_rep` / `struct_out` to say which
level is load-bearing.

**B. Mass in `backbone -> out` alone, everything above flat.**
Claim: a prediction-level regulariser is the whole effect and everything the
field has built on top of it is unmeasured. Harsher and stronger, and it costs
us the method contribution -- the audit becomes the contribution. **Required
before it can be claimed:** the same ladder on NIFTY, FairGB and FairGT. A
statement about the literature cannot rest on ablating our own method only.

**C. `all_uniform -> all_alloc` significant (H6 holds).**
Claim: allocation works, and previous comparisons could not show it because
their budgets did not match. **Immediately conditional on `all_perm`:** if
`all_alloc` beats `all_uniform` but not `all_perm`, the effect is having a
spread of weights, not the ranking, and the structural-signal claim still falls.
Next: which signal -- with the structural family already closed by D4, that
means E8's influence arm, resumed.

**D. No rung Pareto-dominates the one below; most cells incomparable.**
Claim: these interventions are trades, not Pareto improvements, and the
convention of reading ΔDP alone hides it. This is Finding 14's result reproduced
inside our own method, which makes the paper unusually coherent: the same thing
is true outside the method and inside it. Next: lambda_fair sweeps per arm to
draw the frontiers and test whether they coincide -- if they do, every method is
a different point on one curve.

**E. `backbone` dominates the ladder.**
Claim: no fairness intervention here is a Pareto improvement. Most exposed to
the external-validity attack, since all nine settings come from one small family
of benchmarks. **Required before it can be claimed:** a synthetic graph where
the intervention is constructed to be necessary, showing the pipeline can detect
a real effect. Without that the branch is indistinguishable from a broken
implementation.

**Attribution check, to be run the moment the ladder completes, on data that
already exists.** E10's `backbone` arm is not the audit's `GNN`: it is
FairGate's own architecture with lambda_fair = 0, while `GNN` is the baseline
codebase's GCN. Finding 14 reports FairGate dominating `GNN` on 75.9% of cells.
If `backbone` already dominates `GNN`, part of that is the backbone and not the
fairness intervention, and every rung above it inherits the correction. This is
a paired comparison on cells both experiments already have.

**Fixed regardless of branch.** No hyper-parameter is changed in response to the
outcome. No arm is added, dropped or reordered. If a branch requires an
experiment named above, it is run; if the result is unwelcome, it is reported.

## H7 — how much of each method's advantage is its backbone? Registered 2026-09-13

Registered before any cell of E11. What is known at the time of writing is
Finding 16, which answered this for our own method only, and one observation
about FairGB's source that has not been measured.

### Why this replaces the registered form of branch B

The E10 outcome map required, for branch B, "the same ladder on NIFTY, FairGB
and FairGT". That is not implementable and the registration should not have
promised it. These methods do not share FairGate's component structure:
FairGB's fairness mechanism is a sampling augmentation, FairGT's is an adjacency
transform, NIFTY's is a counterfactual contrastive term. There are no matching
rungs to ablate.

What *is* uniform across them is the question Finding 16 actually answered:
of the gap a method shows against plain GCN, how much is its fairness mechanism
and how much is the encoder and training loop it happens to ship with.

Finding 16 measured this for FairGate: `backbone` alone beats the audit's GNN by
net +31, dAUC +0.0800, while the entire fairness stack beats its own backbone by
net **+0** (9 win, 9 lose, 36 incomparable).

### The observation that motivates it, and its status

`FairGB_alg.fit()` defaults to `encoder='SAGE'`, and the audit's call site does
not override it. So FairGB ran on GraphSAGE while the baseline it was compared
against ran on GCN. Lining that up against Finding 14:

    method      backbone                dAUC vs GCN   dominates GCN
    FairGate    its own architecture        +0.0738        81.5%
    FairGB      GraphSAGE                   +0.0760        68.5%
    FairGT      transformer                 +0.0432        70.4%
    NIFTY       GCN                         -0.0221         5.6%
    FairGNN     GCN                         -0.0046        14.8%

Every method that dominates plain GCN runs on something other than GCN; both
that run on GCN do not. **This is a correlation over five methods and is not
evidence.** It is written here so that it cannot later be presented as though
the measurement had confirmed it.

### The arms

Three, on the audit's own protocol (9 settings x 6 splits x 5 inits), paired
cell by cell against rows that already exist:

    GNN_sage      the plain baseline on GraphSAGE   -> FairGB's matched control
    NIFTY_off     sim_coeff = 0                     -> NIFTY's own backbone
    FairGNN_off   alpha = beta = 0                  -> FairGNN's own backbone

FairGB and FairGT get no `_off` arm, because neither exposes a coefficient whose
removal leaves the method intact. They are compared against `GNN_sage` and
against plain GCN respectively, and the weaker inference that permits is
reported as weaker rather than dressed up.

> **H7.** For each method with an `_off` arm, the method Pareto-dominates its own
> backbone on more cells than the reverse.

*Test.* Pareto on (AUC up, dP down), sign test, Holm across the family.
**Falsified for a method if it does not beat its own backbone at p < 0.05.**

### Prior, recorded before the result

I expect H7 to be falsified for NIFTY and FairGNN, on the same pattern Finding
16 found for FairGate: the fairness mechanism moves ΔDP and costs accuracy, and
the per-cell Pareto count nets near zero. I expect `GNN_sage` to account for a
substantial part of FairGB's +0.0760 — that is the point of running it — but I
do not know whether it accounts for most of it, and the correlation above is not
a reason to think it does.

If H7 is falsified across methods, the claim the paper can defend widens from
"our method's advantage is mostly its backbone" to "this comparison's advantages
are mostly backbones", which is a statement about the literature rather than
about us. If it holds for some method, that method is the counter-example and is
reported as one.

**H7 partial, recorded 2026-09-13 with `GNN_sage` complete and the two ablation
arms unfinished (NIFTY_off at 27 of 54, FairGNN_off not started).**

FairGB's decomposition is complete and **my registered prior for it was wrong.**
I wrote that I expected `GNN_sage` to "account for a substantial part of
FairGB's +0.0760". It does not:

    total      FairGB vs GNN        net +36  p<0.0001  rate 0.685  dAUC +0.0760
    backbone   GNN_sage vs GNN      net +18  p=0.0014  rate 0.444  dAUC +0.0163
    mechanism  FairGB vs GNN_sage   net +34  p<0.0001  rate 0.704  dAUC +0.0597

The encoder accounts for about a fifth. With it matched, FairGB still dominates
on 70.4% of cells. FairGB is a counter-example to the pattern Finding 16 found
in FairGate, and the simple claim -- "these advantages are mostly backbones" --
is not available.

Two things must travel with this. The encoder effect is real: a plain SAGE
Pareto-dominates a plain GCN at net +18, p = 0.0014, which is itself a
significant advantage obtained by changing nothing but the architecture. And
`mechanism` here is an over-estimate by construction, because the control is a
matched encoder rather than an ablation -- FairGB and `GNN_sage` also differ in
their training loop, and that difference sits inside the +0.0597. The
registration said this in advance and it is not a retrofit.

The claim the evidence now supports is narrower and, I think, better: the
decomposition differs by method -- FairGate's advantage is entirely its backbone
(mechanism net +0), FairGB's is not (net +34) -- and **the published tables
cannot tell these two cases apart**, because both appear as roughly +0.07 AUC
and a dominance rate near 0.7. Whether that holds is still open: NIFTY and
FairGNN are unfinished and either could break it.
