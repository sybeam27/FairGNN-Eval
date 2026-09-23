# FairGate — where the project stands

Written 2026-09-10, after two days of re-experimentation on RTX6000.
6,120 training runs across five sweeps, 22 commits, one pre-registration with
five recorded deviations. Everything below is reproducible from
`harness/results/*.csv`; nothing is transcribed by hand.

The short version: **the claim the project was built on is dead, a different
claim survived, and it belongs to someone else's method.**

---

## 1. What was inherited

Two submissions. NeurIPS 2026 scored low for claiming `phi(v)` was a node-level
fairness-risk score, which is not identifiable. WSDM 2027 was withdrawn before
submission after an experimental error was found. Its framing was better --
risk claim retracted, "allocation" made the axis, controls added -- and its
central contributions were:

    D1  group -> node, discrete -> continuous bounded weights     (low novelty)
    D2  the choice of allocation criterion conditioned on the graph  <- the paper
    D3  deciding before training when allocation is pointless        <- top-venue

## 2. The data and the code

* Credit had been loading the wrong graph (E = 304,754 instead of 2,873,716,
  which moves it from *clustered* to *degree-skewed* -- the only degree-skewed
  setting in the benchmark). Recovered before this work; re-verified here from
  the raw files with the cache deleted. All nine settings now match Tables 5/6
  exactly, and `datasets.load()` raises if they ever stop matching.
* A second Credit graph is present in the vendored FairGT clone and is **not**
  read by anything. An earlier note here claimed baselines were evaluated on a
  different graph; that was asserted from the file's presence and is withdrawn.
* The repository is now under git with every data file's sha256 in
  `harness/data_manifest.tsv`.

## 3. Three measurement problems, all of which change conclusions

**The same seed does not give the same answer.** cuSPARSE reorders its
reduction, so a forward pass differs by ~1e-7 and 300 epochs amplify that into
ΔDP swings of up to 0.05. CPU is deterministic only within a fixed thread count.
`use_deterministic_algorithms(True)` does not help. The draft's headline
difference is 0.011.

**Replicate counts were an order of magnitude short.** Measured per setting, the
paired replicates needed to resolve 0.011 run from 4 to 366. The draft used
five. Pairing every comparison on the seed is what makes the experiment
affordable at all -- it cuts the requirement by 10-60x.

**The split matters more than the seed, and was never varied.** Re-splitting
leaves 4-29% of the training set in common. Between-split sd is 0.7-2.6x the
initialisation sd, dominating on five of nine settings. The `uniform` mean at
split 20 differs from the six-split mean by **0.018 on average**, against the
0.011 being claimed. Everything computed at one split is a statement about that
split.

The cost of ignoring this is concrete. `w_deg` on Recidivism was the cleanest
cell in the entire matrix at split 20 -- 27 of 30 paired wins, Holm p < 0.0001,
6 replicates needed -- and across six splits it is +0.0031, undecided. It was a
property of split 20.

## 4. What was run

    E0  180    noise floor: within-seed vs across-seed variance
    E1  360    discriminability of each signal, MI(g;s) debiased
    E2  2160   9 settings x 8 signals x 30 init seeds, split 20
    E3  810    split variance, and whether E2's verdicts survive re-splitting
    E4  2160   9 x 8 x (6 splits x 5 inits) -- the split-robust matrix
    E5  810    three operator-matched structural signals, same cells
    E6  162    what the winning signal is made of

Under one protocol throughout: paired on the replicate, Holm over the whole
comparison family, and any setting whose interval contains zero reported as
undecided rather than averaged into a headline.

## 5. What is established

**Allocation works, and by a lot.** Split-robust, over nine settings:

    uniform      0.0468
    best_fixed   0.0347     -0.0122  [-0.0228, -0.0016]   significant
    oracle       0.0341     -0.0127  [-0.0230, -0.0025]   significant

A 26% reduction in mean demographic parity. This is the strongest positive
result the project has.

**The criterion that does it is a model-dependent influence estimate, not
topology.** `bind_influence` is never beaten: 63 paired comparisons against the
other seven signals, **0 losses**, 34 ties, 29 wins.

**Every topological criterion in this literature fails.** Cross-group exposure,
degree, local-homophily deviation -- tested as the field defines them, and again
at the backbone's own depth after a measured defect in the one-hop formulation
was repaired. None beats uniform anywhere except `w_deg` on Credit; none beats
influence anywhere. Our own draft's central signal is among them.

**Choosing per graph is worth nothing.**

    oracle - best_fixed = -0.0006  [-0.0015, +0.0004]   undecided

`best_fixed` picks `bind_influence` on all nine folds. The entire gap between
"always use one criterion" and "know the right answer for each graph" is six
ten-thousandths. **This kills D2**, and it kills it on measurement rather than
on power or confounding.

**Almost nothing else is decidable.** Of 90 signal-vs-uniform cells under split
variation, 3 help, 2 hurt, 85 undecided.

## 6. What was tried and did not work

**The discriminability hypothesis (H1-H3).** Pre-registered before the matrix
existed: that a signal's benefit is predicted by how well it discriminates on
the graph. Falsified. Spearman -0.074, bootstrap CI [-0.483, +0.416]. The LOSO
rule built on it ties `best_fixed` (+0.0008, undecided).

**Repairing the structural signals.** `w_bdry` is one-hop and unweighted while
the backbone mixes two hops with degree normalisation; at the backbone's depth
the gated node sets overlap by a median Jaccard of 0.67 and by 0.22-0.30 on the
two settings with h ~ 0.48. Repairing it helps -- on Pokec-z-g `w_bdry`
significantly hurts (+0.0097) and the corrected version is neutral (-0.0004) --
but it never beats influence. The pre-registered stopping rule fired.

**Explaining why topology fails (E6).** Topology does carry information about
the winning signal: rank correlation up to 0.68 and gate overlap up to 0.53
against a chance level of 0.18, with the sensitive attribute alone contributing
nothing. But whether topology explains influence on a graph does **not** predict
whether a structural signal works there -- Spearman +0.55 (p = 0.13), the wrong
sign. The mechanism is not found.

**One retraction.** A claim that `mean(phi) = 1` fails to hold fairness pressure
fixed, and that E2 was therefore confounded, was posted here and then withdrawn
the same day: `trainer.py`'s scale calibration makes the fairness gradient
scale-free, and the effective contribution lands in 0.068-0.081 across signals
whose raw loss differs sevenfold.

## 7. Two things worth keeping from the failures

**A ceiling near one half.** Given every structural quantity, all pairwise
interactions and full regression freedom -- more than any ranking signal can
use -- topology recovers at most 0.53 of the gate. The best conceivable
structural criterion would intervene on about half the nodes influence chooses.

**Where features beat topology, the edges were built from features.** Credit and
German are the two settings where node features dominate, and both construct
edges by feature similarity rather than from observed relations. `w_deg` won on
Credit and nowhere else, so that win may be degree proxying for feature density.
Untested.

## 8. The three papers this could become

### A. Method paper — thin, and the novelty is not ours

"Node-wise allocation helps by 26%." **BIND (AAAI'23) already differentiates
nodes by influence**, and our winning signal is their criterion. The only
unclaimed piece is soft versus hard: BIND *deletes* harmful nodes, we *weight*
them, and nobody has compared. If soft wins that is a concrete improvement to an
existing method. If it does not, this route is closed.

Prerequisite either way: `bind_influence` is our first-order approximation.
It is no longer one signal among eight, it is the entire positive result, so the
official estimator must replace it before anything is claimed.

### B. Negative result about the field's premise — real, and sharp

The literature assumes topology tells you where fairness risk sits (Dong et al.
KDD'22 and AAAI'23, FairDrop, ComFairGNN, and our own draft). We test that
head-to-head under a protocol the field does not use, and topology loses
everywhere. E6 sharpens it: topology is not ignorant, it simply does not
transfer. What is missing is *why*, and E6 did not find it.

### C. Measurement and reproducibility — best supported, least glamorous

Nobody has audited fair-GNN allocation at multiple seeds and multiple splits.
We have: the noise floor per setting, the replicate counts actually required,
the demonstration that the split flips the cleanest verdict in the matrix, 85 of
90 cells undecided, and a protocol that makes the difference visible. The asset
is the 6,120 runs, and the strongest move available is to include our own prior
submission among the claims that do not survive.

These are not exclusive. C is the frame; B is its main finding; A's soft-vs-hard
test is half a day and would give C a positive result to carry.

## 9. The decision

    soft vs hard        half a day    the only positive contribution still open
    official BIND       0.5-1 day     prerequisite for any claim about A
    re-audit baselines  data mostly held    the body of C

Not recommended: proposing further structural signals. The stopping rule closed
that, and reopening it after five sweeps would cost more credibility than any
signal could return.

---

# Update, 2026-09-11 — the criterion changed, and so did several conclusions

Everything above section 9 was written reading ΔDP alone. That is unsound and
this study's own appendix says so: a model that predicts one class scores
ΔDP = 0, and EDITS did exactly that on Income and NBA in the table being
audited. **All comparisons are now Pareto comparisons on (AUC up, ΔDP down)**,
counted per replicate cell, with no scalarisation -- any weighting of accuracy
against disparity is a free parameter, and holding one lets the ranking be
chosen.

Reproduced by `harness/experiments/analyze_allocation_pareto.py` (allocation
signals) and `analyze_audit.py` (published methods).

## What the re-reading changes

**The allocation axis.** Against `random` on matched cells, 270 each:

    bind_influence   +29   p = 0.009     the only signal that beats random
    w_lhd             +4   p = 0.75
    uniform           -5   p = 0.70      not distinguishable from random
    w_deg            -21   p = 0.053
    prop_dev         -22   p = 0.028
    w_bdry           -31   p = 0.004     significantly *worse* than random

The earlier finding that topological allocation fails becomes sharper rather
than weaker: `w_bdry` -- this literature's canonical signal and our own draft's
-- allocates *worse than assigning weights at random*. And the one signal that
does work pays for it: `bind_influence` is alone in having a real accuracy cost
(mean ΔAUC -0.011 against uniform).

Against `uniform`, only `bind_influence` separates at all (net +27, p = 0.008);
every other signal, including `random`, is undecided. The ΔDP-only reading had
`bind_influence` winning 63.7% of cells; it Pareto-dominates in 23.0%.

**The audit.** Six of seven methods complete, 324 cells. Reading the same cells
two ways:

    method      Pareto-dominates GCN   ΔDP-only win   mean ΔAUC
    FairGate           81.5%              85.2%        +0.074
    FairGT             70.4%              88.9%        +0.043
    FairGB             68.5%              68.5%        +0.076
    FairGNN            14.8%              50.0%        -0.005
    NIFTY               5.6%              61.1%        -0.022

NIFTY wins 61% of cells on ΔDP and is *dominated* by plain GCN in 31.5% of them
(p = 0.003). That gap between the two columns is the audit's finding, not a
detail of presentation.

A caveat that must travel with this table: the three dominating methods all
*raise* AUC over plain GCN by 0.04-0.08, so part of what the Pareto count
credits to them may be model capacity or preprocessing rather than the fairness
intervention. Preprocessing is not shared across methods -- FairGB, FairGT and
FairGate normalise features, GNN and NIFTY do not -- and a direct test showed
the effect is real but neither uniform nor always positive. A
preprocessing-unified arm is needed before any of these dominance rates is
claimed as a fairness result.

*Corrected 2026-09-12.* This paragraph previously cited Income +0.074, Credit
**-0.050**, Pokec-z -0.003 from an ad-hoc test. That test trained for a fixed
500 epochs with no patience and read the final epoch; its Credit unnormalised
run has a validation loss that goes 0.78, 0.96, 2.82, 0.70, 1.02, 1.22, so
which epoch is read decides the answer. Measured under the audit's own protocol
instead -- six splits, five inits, early stopping, all nine settings -- the
effect on plain GCN is:

    recidivism -0.160   pokec_n_g -0.016   pokec_n -0.016   german -0.013
    nba        -0.003   pokec_z_g +0.001   pokec_z +0.002   credit +0.024
    income     +0.059                                  mean over settings -0.007

Income and Pokec-z survive; **Credit reverses sign** (-0.050 to +0.024), and
Recidivism, which the old test never looked at, is the largest effect of all
and is negative.

This changes the direction of the concern as well as its size. Forcing
normalisation makes plain GCN *worse* on average, so on settings like
Recidivism the unified arm should *raise* the fairness methods' dominance
rates, not lower them. Unifying preprocessing does not repair the comparison;
it shows that the answer moves with a step the protocol never fixed.

## What is settled, and what the numbers now are

* **Node-wise allocation**: only an influence-based criterion separates from
  random, and it trades accuracy for disparity rather than dominating.
* **Topology**: fails, and worse than random. Repairing the measurement to the
  backbone's operator (deviation D4) did not change it; the stopping rule fired.
* **Per-graph criterion selection (D2)**: dead. oracle - best_fixed = 0.0006.
* **The protocol**: the split is the larger variance component and moves the
  reported number by 0.034 on average against a claimed effect of 0.011.

## Open, in priority order

1. **BIND**, 27 of 42 feasible cells. Income and Credit are reported as not
   computable at this protocol -- 2.9 h per cell and rising with the label
   budget (Finding 13).
2. **H4 / H5**, pre-registered before the audit finished: soft weighting against
   hard deletion, and allocation under an accuracy constraint. These are the
   only remaining candidates for a positive contribution.
3. **Preprocessing-unified arm**, needed to separate the fairness intervention
   from the preprocessing in the audit table above.
4. Write-up.

## Corrections made to this document's earlier claims

The 26% allocation gain in section 5 was a ΔDP-only number. Under Pareto the
honest statement is that `bind_influence` dominates uniform in 23% of cells and
is dominated in 13%, at an accuracy cost. Section 8's option A ("soft versus
hard is the only unclaimed piece") survives and is now H4.
