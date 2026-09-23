# harness/ — allocation re-experiments

Rebuild of the FairGate experiments around one question:

> Non-uniform allocation of fairness regularization already exists (FairGB
> re-weights per (y,s) subgroup, BIND deletes harmful training nodes,
> ComFairGNN selects a coreset per community). Every one of them **fixes the
> allocation criterion a priori**. Does the right criterion depend on the
> graph, and can that be determined before training?

Nothing here imports `utils/model_fairgate.py`. The point of the rebuild is
that `compute_fiw_weights()` fuses regime selection, gating, structural
ranking and uncertainty modulation into one 345-line function, which makes it
impossible to swap the ranking signal while holding everything else fixed —
and that swap *is* the central experiment.

## Layout

```
core/
  datasets.py   torch-free graph loading, .npz cache, provenance guard
  signals.py    every allocation criterion behind one interface   <- the spine
  allocate.py   score -> phi(v): gate, Norm_G, bounds, budget normalisation
tests/
  test_core.py  run this after any change to core/
e0_diagnostics/ signal dispersion study + graph provenance (already run)
```

## Running

```bash
python harness/tests/test_core.py            # synthetic, seconds
python harness/tests/test_core.py --data     # loads data/, ~1 min (Credit is 108 MB)
```

CPU only. `core/` needs numpy, pandas and scipy; nothing in it imports torch.

## The three invariants `allocate()` must hold

These are not implementation details — the paper's argument rests on them, so
they are asserted in `tests/test_core.py`:

1. **Fixed budget.** `mean(phi) == 1` exactly over the fairness node set. If a
   signal also changed the total amount of fairness pressure, E2 would be
   measuring two things at once and "uniform vs allocated" would be
   meaningless. (The first draft of `allocate()` added an epsilon here and was
   off by 1e-8; small, but it is the invariant the comparison is built on.)
2. **Positive floor.** `phi_min > 0`, so ungated nodes keep pressure.
3. **Degrades to uniform.** When the signal cannot separate the gated nodes,
   allocation becomes flat rather than handing an arbitrary top-k the maximum
   weight. Sec. 4.3 of the WSDM draft *claims* this property; here it is
   implemented explicitly (`degenerate_tol`) rather than falling out of an
   epsilon by accident.

## What e0_diagnostics found (2026-09-08)

Run before writing any of this, and it changed the plan:

* **`data/credit/credit_edges.txt` was not the paper's graph.** E = 304,754 vs
  2,873,716; it put Credit in the *clustered* regime, deleting the only
  degree-skewed setting in the benchmark. Restored from
  `FairGB-main/dataset/credit.zip`, which reproduces all five of the paper's
  Credit statistics exactly. The old files are in
  `data/credit/_quarantine_wrong_graph/`. `datasets.load()` now refuses to
  return a graph that does not match `PAPER_STATS`.
* **"saturated ⟹ w_bdry is near-constant" does not hold.** Of the three
  saturated settings, only Recidivism degenerates. NBA has the *highest*
  `MI(w_bdry; s)` of all nine settings (0.629) and German the second (0.350);
  mean MI is 0.329 on saturated vs 0.060 on non-saturated — the opposite of
  the draft's claim.
* **The exact-top-k justification is inverted.** Sec. 4.2 says saturated graphs
  produce many ties. Measured `arb_frac` is *lowest* there (German 0.000,
  NBA 0.000, Recidivism 0.014) and highest on non-saturated graphs
  (Pokec-z gender 0.065, Income 0.045). The guard is still worth having; the
  stated reason for it is not.
* Degeneracy tracks `h ≈ 0.5` combined with high degree, not `r_b`. When
  homophily is near one half, `r_v^x` concentrates around 0.5 and its variance
  collapses. That is a testable prediction for the synthetic sweep.

Consequence: `r_b` is a binary indicator and a poor proxy for whether a signal
can rank. The regime rule should be driven by dispersion — `Var(g)` or
`MI(g; s)` — which also removes three hand-set cutoffs.

## Finding 3 — a second Credit graph is present but, as traced, unused

`algorithms/FairGT/data/credit/credit_edges.txt` has 200,526 lines, the same
count as the quarantined `credit_edges.WRONG_E304754.txt`, while the loader
FairGate uses reads the 2,174,014-line file (E = 2,873,716, Tables 5/6). Two
Credit graphs circulate in this literature and they are not interchangeable:
E = 304,754 (deg 10.2, delta_deg 0.106, *clustered*) versus E = 2,873,716
(deg 95.8, delta_deg 0.315, *degree-skewed*), and Credit is the only
degree-skewed setting in the benchmark.

Traced, however, no run reads the smaller file. Every baseline reaches the data
through `utils/data.py::get_dataset` -- FairGNN, NIFTY, EDITS, FairEdit,
FairVGNN, FairWalk and CrossWalk via the `utils/dataloading.py::load_data`
adapter, FairGB and FairGate directly, and FairGT through
`algorithms/FairGT_alg.py`, which imports only `algorithms.FairGT.model` and
never the vendored `algorithms/FairGT/utils.py::load_dataset`. The smaller file
is dead weight shipped with the vendored clone. (An earlier version of this
note claimed the baselines were evaluated on a different graph. That was
asserted from the file's presence alone and is withdrawn.)

Two things still follow. First, the file should not be deleted but should be
recorded, because its presence is exactly how the original accident happened:
`harness/data_manifest.tsv` now carries a sha256 for every data file. Second,
`algorithms/FairGT_alg.py::_get_same_sens_complete_graph` caches a derived
tensor at `algorithms/adj_files/{dataset}_same_sens_complete_adj.pt`, keyed by
dataset *name* with no content hash. Today that is harmless -- the tensor is a
function of `sens` alone, and the Credit swap changed only the edge list -- but
any name-keyed cache over a file that has already been silently replaced once
is a repeat of the same failure waiting to happen. Anything derived in
`harness/` is keyed by content.

## Finding 4 — the same seed does not give the same answer, and the spread is the size of the claim

Setting up the GPU path surfaced this. On `german / uniform / seed 27`:

    CPU, 1 thread    dp = 0.0164     (reproducible)
    CPU, 8 threads   dp = 0.0792     (reproducible)
    CUDA             dp = 0.0996 .. 0.1442, different every run

CPU is deterministic within a thread count and not across them; CUDA is not
deterministic at all. The cause is reduction order in cuSPARSE spmm: a single
forward pass differs by ~1e-7 relative, and neither the COO nor the CSR kernel
has a deterministic CUDA implementation, so `use_deterministic_algorithms(True)`
does not catch it (it raises nothing and changes nothing). This is not a bug
introduced by the device port -- the CPU thread-count spread is the same
phenomenon and predates it.

`e0_noise_floor.py` quantifies it: 10 repeats at one fixed seed (float noise
only) against 10 different seeds run once each, `uniform` signal, 180 runs.

    setting      regime          sd_within  sd_across   R_paired  R_unpaired
    german       saturated          0.0351     0.0386        326         395
    pokec_z_g    mixed              0.0098     0.0059         26          10
    pokec_z      clustered          0.0097     0.0121         25          39
    pokec_n_g    mixed              0.0087     0.0199         21         105
    credit       degree-skewed      0.0046     0.0245          6         159
    pokec_n      clustered          0.0011     0.0067          1          12
    recidivism   saturated          0.0003     0.0035          1           4
    nba          saturated          0.0000     0.0420          0         467
    income       clustered          0.0000     0.0093          0          23

Four things follow, and they constrain every experiment from here.

**The noise is the size of the claim.** The draft's headline is a 9-setting mean
ΔDP of 0.039 under uniform weighting against 0.028 under adaptive, a difference
of 0.011. Recomputing that same 9-setting mean per seed gives 0.0480 with an
across-seed sd of 0.0075. At five seeds the standard error of each of the two
numbers is 0.0034, so a 95% interval on either is about ±0.0075 -- wider than
half the difference being claimed. Five seeds cannot support the headline, and
the pilot's inability to resolve the effect at three seeds was this, not a
pilot design problem.

**Pairing is not optional.** Running both arms at the same seeds cancels the
seed main effect and shrinks the required replicate count by one to two orders
of magnitude on most settings (credit 159 -> 6, nba 467 -> 0, pokec_n_g 105
-> 21). `R_paired` here assumes no signal x seed interaction and is therefore a
lower bound; the empirical paired sd must be recomputed from the first real
two-arm run rather than assumed.

**German cannot settle anything.** Its `R_paired` is 326: repeating one seed
moves ΔDP by 0.035, three times the effect. Note also that seed 27 alone gives
0.0996 while the ten-seed mean is 0.0358 -- it is both an unlucky seed and an
unstable one. German is where the draft reports its largest fairness margin
(0.028 against 0.076 for the next method) and its largest ablation swing
(0.028 -> 0.359 without the prediction term). Those numbers are inside the
noise. German must be reported as uninformative, not as the headline.

**Instability is not a property of graph size or regime.** Income (n=14,821)
and NBA (n=403) are both bit-exact across repeats while pokec_z (n=67,796) is
not; among the three saturated settings, recidivism has the lowest sd of all
nine and german the highest. Whether cuSPARSE takes a split-K path depends on
the sparsity pattern, so device noise has to be measured per setting and cannot
be argued from the graph.

Consequence for the design: every comparison is paired on seed, the replicate
count is set per setting from its measured sd rather than fixed at five for
all, and any setting whose interval does not separate the arms is reported as
such instead of being averaged into a headline.

## Finding 5 — the objective and the metric come apart; the *strength* claim was wrong

**Post hoc, and partly retracted the same day.** What follows states the measured
part, then the part that was claimed and is false, then what is left.

### The measurement (holds)

`allocate()` enforces `mean(phi) = 1`, which `HANDOFF.md` justifies as keeping
the budget fixed so that changing the signal does not change `lambda_fair`. DP is
a difference of two group means, so if `phi` correlates with the predicted score
*within* a group, the weighted group mean moves even at `mean(phi) = 1`. At the
end of warm-up, seed 27, `q_gate` 0.7:

    setting      signal            true gap   phi-weighted gap   ratio
    recidivism   bind_influence      0.0055        0.1443          26x
    pokec_z      bind_influence      0.0423        0.2502           6x
    credit       bind_influence      0.0751        0.1922         2.6x
    recidivism   w_deg               0.0055        0.0113           2x

and after 300 epochs the two still differ, now in both directions:

    credit       bind_influence      0.0323        0.0035          0.1x
    recidivism   bind_influence      0.0363        0.2150          5.9x
    pokec_z      bind_influence      0.0268        0.1188          4.4x
    credit       w_deg               0.0057        0.0013          0.2x

Credit is the sharp case: the weighted gap was driven to 0.0035 while the gap
actually reported stayed at 0.0323. The optimiser succeeded at a different
quantity from the one being scored.

### The claim that was wrong

This note first asserted that the inflation acts as more fairness pressure --
that `L_pred` under `bind_influence` on Recidivism pushes 26 times harder, and
that E2 therefore measures placement confounded with effective strength. **That
is false, and `trainer.py` already prevents it.** The scale calibration
`gamma_k = min(L_task / (L_k + eps), 100)` makes the term contributed to the
loss `lambda_fair * gamma_k * L_k ~ lambda_fair * L_task` whatever the magnitude
of `L_k`; equivalently the gradient is `lambda_fair * L_task * grad log L_k`,
which is scale-free. Traced over training:

    setting/signal                L_pred    gamma   lambda*gamma*L_pred   gamma clipped
    recidivism/uniform           0.0277    14.61        0.0774                 0%
    recidivism/w_deg             0.0183    22.37        0.0787                 0%
    recidivism/bind_influence    0.1265     2.82        0.0683                 0%
    credit/uniform               0.0076    81.07        0.0813                70%
    credit/w_deg                 0.0104    79.27        0.0796                70%
    credit/bind_influence        0.0664     7.14        0.0799                 0%

A sevenfold larger `L_pred` buys a sevenfold smaller `gamma`, and the effective
contribution lands in 0.068-0.081 for every signal, including where the cap
binds 70% of the time. **The confound this note claimed does not exist**, and
the E2 falsifications of H1-H3 are cleaner than the retracted version said:
power, not strength-confounding, is the remaining caveat on them.

### What is left

That the optimised and evaluated quantities differ by 0.1x to 5.9x after
training is real and is not something the draft measures -- it says only that
its objectives are "differentiable relatives of, but not identical to" the
evaluation metrics. Whether that gap *causes* anything is not established:
`bind_influence` shows the largest divergence on all three settings and helps on
two of them while hurting on the third, so divergence alone predicts neither
sign nor size.

Caveat on these traces: they run `L_pred` alone at a fixed 300 epochs with no
early stopping, so they are not the E2 runs and are indicative only.

The usable residue is a question, not a finding: *what does a node-weighted
fairness objective converge to, and when is that the thing you wanted?* If it is
pursued it needs its own pre-registration, and the strength story must not be
resurrected -- it was tested and it is dead.

## Finding 6 — the split is the larger source of variation, and it changes verdicts

E3, 810 runs, 6 split seeds x 5 init seeds x 3 signals x 9 settings, paired
inside each (split, init) cell so the replicate count matches E2 exactly. Only
the population the 30 replicates are drawn from changes.

Re-splitting leaves 4-29% of the training set in common, so these are different
learning problems, not perturbations. Between-split sd against within-split
(initialisation) sd, on the `uniform` arm:

    pokec_z 2.6x   pokec_z_g 2.6x   recidivism 2.6x   pokec_n 1.5x   nba 1.3x
    credit 1.2x    income 1.1x      pokec_n_g 0.8x    german 0.7x

The split therefore dominates on five of nine settings and is comparable on the
rest. It was the axis we were not varying. Its consequence is direct: the mean
`uniform` disparity at split 20 differs from the six-split mean by **0.018 on
average**, against the 0.011 effect the draft claims. A number computed at one
split is not a number about the dataset.

### Which E2 verdicts survived

Three of the four, and they grew:

    signal          setting      E2 (split 20)   E3 (6 splits)   splits helping
    bind_influence  credit         -0.0230          -0.0396           6/6
    bind_influence  pokec_z        -0.0244          -0.0305           6/6
    w_deg           credit         -0.0160          -0.0256           5/6
    w_deg           recidivism     -0.0078          +0.0033           3/6   died

`w_deg` on Recidivism was decisive at split 20 -- 27 of 30 paired wins, Holm
p < 0.0001, `R_needed` 6, the cleanest cell in the whole matrix -- and it does
not survive re-splitting. It was a property of split 20. Symmetrically,
`bind_influence` on Recidivism was `hurts` at split 20 and is undecided across
splits, and `bind_influence` on German appears only once splits vary (-0.0304,
4/6).

### What the split-robust picture says

`w_deg` helps on **Credit alone** -- the one degree-skewed setting -- and
actively **hurts on four others** (German +0.0303, Pokec-n +0.0090, Pokec-z-g
+0.0061, Income +0.0040). A criterion that wins where the graph's structure
calls for it and does damage elsewhere is exactly the regime-specific behaviour
the project is about, and it is visible only after the split is varied.

`bind_influence` helps on three settings and hurts on none. Between these two
signals it dominates: it is at least as good as `w_deg` on all nine. That is a
threat, not a result in our favour -- if one criterion is never worse, a fixed
choice suffices and no selection rule is needed. It is also a reimplementation
of an existing method (BIND, AAAI'23) used as a soft weight rather than as node
deletion, so if it stays dominant the honest finding is about that method, and
the official estimator has to replace ours.

Only two of eight signals were run here. Whether the domination holds across the
full library is the next thing to measure.

### The price

`R_needed` rises with the added variance component: median about 120 against
about 60 for the init-only design, and 14-433 across cells. R = 30 is no longer
adequate anywhere except Recidivism (18) and Income (14). Generalising from
"at split 20" to "on this dataset" costs roughly four times the replicates, and
that has to be in the budget rather than discovered later.

## Finding 7 — `w_bdry` is measured at the wrong depth for a 2-layer backbone

Diagnostic only: `harness/e0_diagnostics/signal_scale_check.py` trains nothing and
reads no benefit. It compares `w_bdry` against cross-group mass computed on the
operator the backbone is actually defined with, `A_hat = D^-1/2 (A+I) D^-1/2`,
at one and at two hops. What matters is the gate, since `allocate()` keeps the
top `1 - q_gate`, so the table reports the overlap of the node sets each version
would select.

                                gate Jaccard        Spearman
    setting     regime        1 hop   2 hops     1 hop   2 hops   mean deg
    pokec_n_g   mixed         0.622    0.224     0.909    0.285      16.5
    pokec_z_g   mixed         0.791    0.304     0.903    0.371      19.2
    pokec_z     clustered     0.935    0.622     0.996    0.738      19.2
    recidivism  saturated     0.875    0.622     0.957    0.841      34.0
    income      clustered     0.746    0.666     0.986    0.815       6.8
    credit      deg-skewed    0.863    0.671     0.991    0.846      95.8
    pokec_n     clustered     0.967    0.690     0.996    0.760      16.5
    german      saturated     0.818    0.765     0.983    0.943      44.5
    nba         saturated     1.000    0.875     0.930    0.822      53.7

**The degree normalisation is not the problem; the hop count is.** Adding the
`1 / sqrt(d_u d_v)` weighting at one hop leaves `w_bdry` essentially unchanged
(Jaccard median 0.86, Spearman 0.90-1.00), including on Credit where the mean
degree is 96. Going to the backbone's actual depth changes the selected set
substantially everywhere (median 0.67) and almost completely on the two mixed
settings, where the two measurements are close to unrelated (Jaccard 0.22-0.30,
Spearman 0.29-0.37).

Those two are the settings with `h ~ 0.48`. Finding 2 predicted that a
one-hop cross-group ratio collapses when homophily is near one half, because
`r_v^x` concentrates around 0.5; what this adds is that the two-hop propagated
version does not collapse there. The signal died at one hop, not the quantity.

Read carefully, this is partial. On the two mixed settings E4's verdict on
`w_bdry` is not a verdict on cross-group exposure -- it measured something else.
On the other seven `w_bdry` is a serviceable proxy and about a third of the
gated set differs, so the verdict is mostly about exposure, weakened at the
margin. It does not license "structural signals fail" and it does not license
"they were only mis-measured".

What it does license is a revision on the same footing as deviations D1 and D2a:
a defect of the statistic, established without reference to any outcome. The
principle is one line -- *measure the signal on the operator the model uses* --
and it yields three candidates, `prop_hop2` (fixes the depth), a cross-edge
removal sensitivity `||(A_hat^2 - A_tilde^2)_v X||`, and a receptive-field
composition deviation replacing `w_lhd`. They must be pre-registered before the
benefit is measured, and any cutoff they need must be fixed on synthetic graphs
and frozen, or the addition is a fishing expedition however well motivated.

## Finding 8 — topology does know about influence, and that does not help

E6, 162 fits. For each setting the winning signal `bind_influence` is regressed
out-of-fold on blocks of predictors: the sensitive attribute alone, all twelve
structural quantities in the study, those plus every square and pairwise product
(90 columns), the node features, and everything together. This proposes no new
allocation signal; it asks what the winner is made of. Chance overlap for two
independent top-30% sets is about 0.18.

    setting       h     TOPO2+SENS        FEAT          best structural   bind
                        rho    gate       rho   gate     benefit          benefit
    pokec_n     0.956   0.68   0.48       0.25  0.31      +0.0001        -0.0044
    pokec_z     0.953   0.63   0.48       0.20  0.27      +0.0034        -0.0209
    nba         0.729   0.59   0.53       0.31  0.25      -0.0070        -0.0125
    credit      0.960   0.48   0.43       0.70  0.52      -0.0246        -0.0394
    german      0.809   0.46   0.29       0.39  0.57      +0.0192        -0.0256
    recidivism  0.536   0.42   0.26       0.07  0.27      -0.0077        -0.0002
    pokec_n_g   0.489   0.30   0.32       0.31  0.22      -0.0070        -0.0032
    income      0.884   0.29   0.29       0.40  0.35      -0.0025        +0.0003
    pokec_z_g   0.479   0.25   0.25       0.27  0.26      -0.0027        -0.0037

The sensitive attribute alone carries nothing (rho -0.08 to -0.00, gate 0.11 to
0.17), so everything above is topology, not group membership.

**"Topology knows nothing about where fairness pressure should go" is false.**
Rank correlation reaches 0.68 and gate overlap 0.53, both far above chance. The
information is there.

**It does not translate.** Whether topology explains influence on a graph does
not predict whether a structural signal works on it. Over the nine settings,
Spearman between the topology readouts and the gap between the best structural
signal and `bind_influence` is +0.55 (rho, p = 0.13) and +0.45 (gate, p = 0.23)
-- positive, meaning if anything the better topology explains influence the
*further behind* structural signals are, and neither is significant. The
mechanism this experiment was run to find is not there.

Restricting post hoc to the four settings where `bind_influence` actually helps
by more than 0.01 -- NBA, Credit, Pokec-z, German -- the gate overlap orders the
gap correctly in three of four (0.53 -> 0.0055, 0.43 -> 0.0149, 0.29 -> 0.0449,
with Pokec-z out of place). That is four points chosen after seeing the
pre-specified test fail, and it is recorded as a hypothesis for a future
pre-registered test, not as a result.

**A ceiling near one half.** Given every structural quantity, all their pairwise
interactions and full regression freedom -- far more than any ranking signal can
use -- topology recovers at most 0.53 of the gate and typically 0.25-0.48. So
even the best conceivable structural criterion would intervene on roughly half
the nodes influence would choose. Whether that is why they fail is exactly what
the paragraph above could not establish.

**Where features beat topology, the edges were built from features.** Credit
(FEAT rho 0.70 against TOPO 0.48) and German (FEAT gate 0.57 against 0.29) are
the two settings where node features dominate, and both have edges constructed
by feature similarity rather than observed relations. `w_deg` won on Credit and
nowhere else, so that win may be degree acting as a proxy for feature density
rather than as a fairness-relevant structural role. Untested.

Two readouts were needed. Rank correlation and gate overlap disagree in
direction on German (topology 0.46/0.29 against features 0.39/0.57): topology
orders the whole population better while features identify the extreme better,
and it is the extreme that `allocate()` consumes.

## Known issues in the old code (not fixed here)

* `signal_diagnostics.py:414` — `--node-set sens` treats `get_dataset`'s second
  return value as a node index array, but it is a *feature column* index, so
  the evaluation set collapses to one node. Moot in practice: no setting has
  `sens < 0`, so `sens` and `all` coincide.
* `utils/data.py:349` — the German loader assigns ints into a string column and
  raises under pandas 3.
* `algorithms/` contains `FairGB_alg copy.py`, `FairWalk copy.py`,
  `CrossWalk copy.py` alongside the originals. Which one produced the submitted
  numbers is not recoverable. Nothing in `harness/` may have a `copy` twin.

## Still to build

`core/objectives.py`, `core/trainer.py`, `experiments/e2_signal_swap.py`,
`adapters/{bind,fairgb,fairsin}.py`. Backbone will be a plain-torch GCN using
the pre-normalised adjacency (numerically the same propagation as PyG
`GCNConv`, but with no PyG dependency and CPU-friendly).

## Finding 9 — BIND's deletion set is not stable across initialisations

Measured while budgeting the audit, on `harness/adapters/bind.py`, which calls the
published estimator unmodified. Influence recomputed at three initialisation
seeds on one split, everything else held fixed:

    setting      seeds    Spearman(influence)   top-30% Jaccard
    german       27-28          -0.252               0.15
    german       27-29          +0.665               0.50
    german       28-29          -0.800               0.02
    recidivism   27-28          +0.061               0.28
    recidivism   27-29          +0.313               0.40
    recidivism   28-29          +0.932               0.76

On German at seeds 28 and 29 the rank correlation is **negative**: the nodes one
run calls most harmful are among those the other calls most helpful, and the
sets that would actually be deleted overlap by 0.02 against a chance level of
about 0.18. Recidivism is better behaved but still ranges from 0.28 to 0.76.

Two consequences.

**For the budget.** Influence has to be recomputed per (split, initialisation),
not once per split. Computing it once would freeze BIND's deletion set while
every other method in the audit re-runs its whole pipeline per replicate, which
both understates BIND's variance and is the asymmetry the table above shows is
large. 270 computations, about 20 hours; the measured cost is 0.5 min on German
to an estimated 15 min on Credit, per computation.

**As a finding.** BIND estimates per-node influence and then *deletes* the
harmful nodes -- a discrete, irreversible intervention -- and which nodes those
are depends on the random seed. Note the connection to E4, where the same score
used as a *continuous weight* was the only criterion that worked: if the
estimate is this unstable, a soft weighting degrades gracefully where a deletion
does not. That is a mechanism worth testing and is not tested here; it is
recorded as a post-hoc observation.


## Finding 11 — the audit pilot: 5 methods, 3 settings, 6 splits (450 runs)

### Split 20 mostly reproduces the paper, and where it does not is informative

    setting     method    paper   ours     diff
    german      FairGNN  0.2288  0.2288   0.0000  exact
    german      FairGB   0.1495  0.1495   0.0000  exact
    nba         FairGNN  0.2074  0.2074   0.0000  exact
    nba         FairGB   0.0488  0.0488   0.0000  exact
    recidivism  FairGNN  0.0770  0.0770   0.0000  exact
    recidivism  NIFTY    0.0723  0.0732   0.0009
    recidivism  FairGT   0.0100  0.0069  -0.0031
    nba         FairGT   0.0053  0.0130   0.0077
    recidivism  FairGB   0.0268  0.0350   0.0082
    nba         NIFTY    0.1026  0.1181   0.0155
    german      NIFTY    0.1036  0.1299   0.0263
    german      FairGT   0.1352  0.0677  -0.0675

Five cells reproduce to four decimal places, which says the pipeline is reading
the same problem the paper read. The rest do not, and the two largest misses are
on German -- the setting E0 measured as having a within-seed sd of 0.035, so a
0.027 or 0.068 gap there is not evidence of a different pipeline. Both methods
are also known to be nondeterministic on this hardware.

### The winner changes with the split, on some settings

    german      four different methods win across six splits
                (FairGT x2, FairGB x2, FairGNN, and plain GCN once)
    recidivism  three (FairGT x4, FairGB, NIFTY)
    nba         one  (FairGT all six)

On German a plain GCN with no fairness intervention at all takes first place on
one of the six splits. A table built from one split reports a ranking; it does
not report *the* ranking.

NBA is the counter-case and matters as much: FairGT wins on all six splits, so
split variation does not dissolve every conclusion. The claim is not that
rankings are arbitrary but that which ones survive is currently unmeasured.

### Stability is a property of the method, not only of the setting

Ratio of largest to smallest ΔDP over the six splits:

    method     german   nba   recidivism
    FairGB        7.1   6.4      8.2
    FairGT        7.4   2.4     11.8
    FairGNN       3.5   5.1      1.2
    GNN           5.7   2.8      1.3
    NIFTY         3.9   2.6      1.4

On Recidivism, FairGNN, GNN and NIFTY move by 1.2-1.4x while FairGB and FairGT
move by 8-12x on the same six splits. So "how many splits do you need" has no
single answer per dataset: it depends on the method being measured, which is not
something a fixed protocol can absorb.

### BIND's intervention size is decided by graph density

Candidates surviving the non-overlapping filter, and the resulting deletion
budget at BIND's own 30%:

    setting     mean deg  |train|  candidates  max_num  budget
    income           6.8     4605        1902     1089     326
    pokec_z         19.2      500         459       62      18
    recidivism      34.0      100          94       37      11
    german          44.5      100          18       10       3
    nba             53.7      100           4        2       0

BIND ranks training nodes by influence, then keeps only those whose one-hop
computation graphs do not overlap, then stops where the influence changes sign
and deletes 30% of that. On a dense graph almost every node overlaps every
other, so the filter empties. On NBA the budget is **zero** -- BIND is exactly
the unmodified model there, and would appear in a comparison table as a method
that happened not to help.

This is not a defect in BIND. Its own evaluation used bail (deg 34), income
(6.8) and pokec, where the mechanism has room. It is a defect in reporting a
method on settings outside the regime it was demonstrated in, and it is
invisible once a single number reaches a table.


## Finding 12 — BIND's own numbers come from a model its code meant to discard

Validating the adapter's transcription against BIND's published log, using
BIND's loader, BIND's label budget (1000 on Income), BIND's seed (10) and the
`scale = 25` its README specifies for Income -- none of our data, none of our
protocol:

                       ours      BIND README
    vanilla GCN Acc   0.7080       0.7070
                SP    0.2945       0.3125
                EO    0.3402       0.3598
    "At most effective number"  351   327

Close enough to call a reproduction across a five-year gap in hardware and CUDA
(theirs: Titan RTX / CUDA 10.1). The transcription is faithful, so BIND can go
in the audit table.

Getting there turned up two things.

**`weight_decay` is 1e-4**, not the 1e-5 we first assumed -- read off
`1_training.py:31` rather than guessed.

**The model selection is overwritten.** `1_training.py:150-157`:

    for epoch in tqdm(range(args.epochs)):
        loss_mid = train(epoch)
        if loss_mid < loss_val_global:
            loss_val_global = loss_mid
            torch.save(model, 'gcn_' + dataset_name + '.pth')
    torch.save(model, 'gcn_' + dataset_name + '.pth')      # unconditional

The second write lands after the loop and overwrites the checkpoint the first
one selected, so every downstream stage loads the **last epoch**, not the best
validation epoch. Our first attempt returned the best-validation model and got
SP 0.0287 against the published 0.3125 -- a tenfold gap that is not noise.
Selecting properly gives a *much fairer* starting model, which means BIND's
reported debiasing gains are measured from a worse baseline than its own code
intended. `_train_gcn(select=...)` defaults to `"final"` because that is what
runs; `"best_val"` is available so the two can be compared.

**Also from the README, and not in the code's defaults**: the influence
estimator's `scale` is per dataset -- 25 for Income and Recidivism, 100 for
Pokec1, 60 for Pokec2 -- while `s_test_graph_cost` defaults to 60. Running
everything at the default would run Income and Recidivism at a value the authors
say is wrong for them. `SCALE` now carries the published values. Three of our
nine settings (German, NBA, Credit) are not in BIND's paper, so no value exists
for them; we use the code default and flag it, and `SCALE_FROM_PAPER` records
which settings rest on the authors' guidance and which on ours.


## Note — FairGB fragments the GPU on the Pokec graphs

E7 run, 2026-09-11. FairGB completed German, NBA, Recidivism, Income and Credit
(30 of 54 cells) and then failed on every Pokec cell:

    tried to allocate 17.12 GiB; 34.75 GiB in use, 12.74 GiB free
    of which 17.11 GiB is "reserved but unallocated"

Half the memory the process holds is reserved and unusable, so this is
fragmentation, not a 47 GiB card being too small for a 35 GiB method. Our own
Table 12 already recorded FairGB peaking at 35,126 MB on Pokec, so the headroom
was always thin; running five initialisations inside one process is what turns
thin into fragmented.

The runner does not cache failed cells, so those 24 cells are simply missing and
are re-run rather than recovered. Two fixes, in order of preference:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which is what the error
itself suggests and addresses the reservation pattern directly; failing that,
one process per initialisation so the allocator starts clean each time.

Not retried while the sweep is still running -- a second process on the same GPU
is how the first one ran out of room.

**Resolved.** Restarting with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
put `FairGB / pokec_z / split 20` -- the cell that had failed -- through in 332 s.
The diagnosis holds: the card was not too small, the allocator was holding half
its memory in unusable reservations. Nothing about the method or the data
changed, so the 24 cells are recovered rather than approximated, and the method
order was changed to FairGate -> FairGB -> BIND so that the fix could be
confirmed in minutes instead of after BIND's nineteen hours.


## Finding 13 — BIND's cost and its intervention both track `max_num`, in opposite directions

Measured on the audit, five initialisations per cell, BIND's own published
settings:

    setting      min/cell    max_num    retrainings (0.3 x max_num)
    nba              5.3          3            0
    german           5.5         14            4
    recidivism      11.8         38           11
    income         173.6      1,105          331

The whole cost is the deletion sweep: BIND retrains once per candidate deletion
count up to `0.3 * max_num`, and `max_num` is where the influence estimate
changes sign, which grows with the number of training nodes. Income has 4,605 of
them and 331 retrainings; NBA has 100 and **zero**.

The two ends fail differently and both are diagnosable before running anything.

**Large label budget: not computable at this protocol.** Income takes 2.9 hours
per cell, so six splits is 17 hours and Credit (6,000 training nodes) is worse.
One Income cell timed out at the 2-hour limit before the limit was raised. The
audit therefore reports Income and Credit as *not computable within four hours
per cell*, alongside the measurement above -- the same slot our own paper uses
for FairEdit and EDITS failing with OOM, and for the same reason: a method whose
cost is set by the benchmark's label budget rather than by the graph.

**Small label budget: no intervention at all.** NBA's `max_num` is 3, so the
budget `int(0.3 * 3)` is 0 and BIND deletes nothing; its row is the untouched
model. German deletes 4 of 100 training nodes. Finding 9 gave the reason -- the
non-overlapping filter drops any candidate whose 1-hop computation graph
touches an already-selected one, and what drives that is `degree x |train| / n`,
the expected number of training-node neighbours: 13.3 on NBA, 4.5 on German,
0.18 on Recidivism, 0.14 on Pokec-z.

So BIND is squeezed from both sides by the benchmark's label budget, and a
number computable from the graph and the split -- before any training -- says
which side a setting is on. A table row reporting only ΔDP shows none of this;
`bind_k_mean` and `bind_max_num_mean` are recorded beside every cell so that a
row where nothing was deleted cannot be read as a method that did not work.


## Finding 14 — unifying preprocessing keeps three methods' dominance and dissolves the audit's sharpest claim

The comparison being audited does not share feature normalisation: GNN and
NIFTY load all nine settings unnormalised, FairGNN normalises NBA and German
only, and FairGB, FairGT and FairGate normalise everywhere. The protocol
section lists datasets, splits, seeds, learning rate and weight decay as
shared. Preprocessing is not on that list.

Deviation D5 adds a second arm with normalisation forced on for every method
(`--arm normunified`; 162 cells re-run, the four already-normalised methods
carried over rather than re-run). Both arms, dominance over plain GCN on
(AUC up, ΔDP down), Holm-corrected across the five comparisons:

    method      as_submitted              normunified
    FairGate    net +43  Holm 0.0000      net +39  Holm 0.0000
    FairGT      net +37  Holm 0.0000      net +37  Holm 0.0000
    FairGB      net +36  Holm 0.0000      net +36  Holm 0.0000
    NIFTY       net -14  Holm 0.0052      net  +5  Holm 0.7666
    FairGNN     net  -1  Holm 1.0000      net  +4  Holm 0.7666

**The three dominating methods survive, and that is the stronger half of the
result.** Their as-submitted comparison against GCN was preprocessing-*mis*matched
-- they normalised, the baseline did not -- so the caveat this study attached
to that table was the right one to raise. The unified arm matches them, and
the dominance is unchanged: per cell, FairGate moves in 9 of 54 (3 up, 6 down,
p = 0.51), FairGT in 11 (p = 1.00), FairGB in 6 (p = 1.00). The confound was
real and it is not the explanation. Their margins are not bought with the
loader.

**NIFTY does not survive, and this is not a confound at all.** NIFTY and GNN
were *both* unnormalised in the as-submitted arm and are *both* normalised in
the unified one, so the comparison is preprocessing-matched in each arm
separately. Nothing about it was unfair. The conclusion moves anyway: 23 of 54
cells improve for NIFTY against 8 that worsen (p = 0.011), and "plain GCN
Pareto-dominates NIFTY, p = 0.003" becomes "no detectable difference,
Holm 0.77". The sharpest finding this audit had is a finding about one
preprocessing setting, not about NIFTY.

Most of that movement is the baseline, not the method. Forcing normalisation
costs plain GCN 0.0072 AUC on average and 0.131 on Recidivism; NIFTY loses
less (0.0041), so it gains ground without improving. That is the mechanism, and
it is the reason the arms cannot be ranked: neither is the correct one.

Per-setting, averaged over the three methods the override changes:

    recidivism -0.131   pokec_n_g -0.010   german  -0.008   pokec_n -0.008
    nba        -0.002   pokec_z_g +0.009   pokec_z +0.011   income  +0.042
    credit     +0.073

Normalisation is not a fix that was skipped. It helps on Credit and Income and
costs an eighth of an AUC on Recidivism. A protocol that leaves it to each
implementation leaves this much on the table, and the field's convention of one
split and five seeds cannot see it.

Reproduce:

    python harness/experiments/e7_baseline_audit.py --arm normunified \
        --models GNN NIFTY FairGNN
    python harness/experiments/analyze_arms.py

## Finding 15 — a third of this study's influence estimates had silently diverged

BIND estimates per-node influence with LiSSA, which approximates `H^-1 v` by

    h <- v + (I - H / scale) h

The recursion converges only when `scale` exceeds the largest eigenvalue of the
Hessian. Below it the iterate grows without bound, reaches `inf`, and the next
subtraction makes it `NaN`. Nothing raises.

What makes this worse than an ordinary numerical failure is where the NaNs go.
`_order_and_budget` (`3_removing_and_testing.py:243-268`) walks the influence
vector looking for the first sign change and reports that index as `max_num`,
the deletion budget. A vector of NaNs has no sign change, so it returns
`max_num = 0` — which is also exactly what a *converged* estimate reports when
it finds nothing harmful to delete. In every column the audit recorded, a
diverged estimator and an empty budget are the same row.

This study read those rows the wrong way. Pokec-n and Pokec-n_g were described
as settings where "BIND's own budget rule deletes nothing", and the NBA
failures were described as a reliability problem in BIND's estimator. Measured
directly, at the scales in `adapters/bind.py`, seed 27, split 20:

    pokec_n      0 of 500 finite      converges at scale 150
    pokec_n_g    0 of 500 finite
    nba          0 of 100 finite on splits 21, 22 (seed 29) and 23 (seeds 29, 30)
    german       finite at every seed tested (3 splits x 5 seeds)
    recidivism   finite at every seed tested

And NBA's `scale` of 60 is not in `SCALE_FROM_PAPER`: it was **our** guess. The
divergence is a consequence of a hyper-parameter this study chose, not a defect
in the method being audited. Writing it up as the latter would have put a fault
in someone else's method that their method does not have.

**The cache made it permanent.** `influence()` wrote its result to disk before
returning, NaNs included, keyed by a content hash. Once a cell diverged, every
later run read the NaN vector back instantly and never recomputed. Of the 336
influence vectors cached here, **117 were non-finite** — a third of every
influence computation this study had made, being handed back as results.

Fixes, all in our adapter and none in BIND-main:

  * `influence_converged()` escalates `scale` along a fixed ladder
    `[25, 60, 100, 150, 250, 400, 700, 1000]` until the estimate is finite and
    raises if none is. The criterion is finiteness, which cannot depend on or
    tune any accuracy or fairness outcome.
  * `influence()` no longer caches non-finite results and warns with the count
    and the scale. It still returns rather than raises, because the escalation
    needs to see the failure.
  * `scale_used` and `scale_steps` are recorded per cell, so a run at other
    than the published scale is visible in the results rather than inferred.

The 117 stale vectors are deleted. The affected BIND cells — Pokec-n, Pokec-n_g
and NBA — are quarantined under `harness/results/e7_raw_diverged/` with a note
recording what was verified and what was not, and are being re-run. Pokec-z and
Pokec-z_g are **not** verified either way: their `max_num` is above zero on
every cell, which establishes only that at least one of five seeds converged,
and a diverged seed contributes 0 to a cell mean rather than zeroing it. They
are re-run rather than assumed clean.

First re-run cell, Pokec-n split 20, against the quarantined one:

    max_num   0    ->  219.2        nodes actually deleted   0 -> 22.8
    AUC       0.6521 -> 0.6519      dP  0.0588 -> 0.0592
    sec/cell  160    -> 590

The row barely moves. What moves is what can be said about it: not "BIND had
nothing to delete here" — a statement about the budget rule — but "BIND deleted
23 nodes here and it changed nothing", which is a measurement of the method.
The old 160 s/cell was the cost of performing zero retrainings.

## Finding 16 — INVALIDATED PENDING RE-EVALUATION (2026-09-13)

> **Do not cite the +0.0800 backbone figure.** The attribution check below
> compared FairGate's true AUC (probabilities, `harness/core/metrics.py`) against
> the audit's GNN, whose "AUC" is computed from hard 0/1 predictions and is
> therefore balanced accuracy. Measured on one trained model, that definitional
> gap averages +0.056 — the same order as the +0.0800 being claimed. See
> `harness/X2_OUTCOME_AUDIT.md`.
>
> **What is suspended:** every row of the attribution check that crosses the
> metric boundary (`backbone vs GNN`, `all_alloc vs GNN`), and with it the
> reading that FairGate's advantage is mostly architecture.
>
> **What is not suspended:** `all_alloc vs backbone` — net +0, 9 win, 9 lose,
> 36 incomparable. Both arms are E10 and both use the same true-AUC computation,
> so the ladder's own conclusion stands: the fairness stack does not
> Pareto-dominate its own backbone.
>
> The rung table below (dΔDP per component) is unaffected in its dΔDP column and
> suspended in its dAUC column.

## Finding 16 (original text, retained for the record) — the ladder: 91% of the fairness effect is the first rung, and the headline was the backbone

E10, 378 invocations, 1,890 runs, 8.5 hours. Seven arms of the published
FairGate differing only in `ablation_mode` and `fiw_weight_mode`, on the audit's
own protocol: 9 settings x 6 splits x 5 inits, 54 cells, every cell complete in
every arm.

### The attribution check first, because it moves everything else

Finding 14 reports FairGate Pareto-dominating plain GCN on 75.9% of cells. The
two rows differ in more than the fairness term: the audit's `GNN` is the
baseline codebase's GCN, while E10's `backbone` is FairGate's own architecture
and training loop with `lambda_fair = 0`. Paired on the same 54 cells:

    backbone   vs GNN        net +31  p < 0.0001   rate 0.574   dAUC +0.0800
    all_alloc  vs GNN        net +43  p < 0.0001   rate 0.796   dAUC +0.0728
    all_alloc  vs backbone   net  +0  p = 1.0000   rate 0.167   dAUC -0.0072

**The fairness intervention, measured against its own backbone, Pareto-dominates
it on net zero cells** — 9 wins, 9 losses, 36 incomparable. Against the
normalisation-unified GNN the backbone offset is larger still, +0.0872 AUC, so
it is not preprocessing either. It is architecture and training loop.

The 75.9% is real and it is mostly not the fairness intervention. A comparison
that crosses codebases attributes to the method whatever else differs between
the two implementations, and here that is +0.08 AUC — larger than any fairness
effect measured anywhere in this study.

### Where the fairness effect actually is

Every rung below carries `uniform_budget`, so all of them spend the same total
fairness pressure and each step is attributable to its own component:

    rung                       dΔDP      share    dAUC      Wilcoxon p
    backbone -> out          -0.0497     91.3%   -0.0056     0.0000
    out -> rep_out           -0.0026      4.7%   +0.0006     0.0966
    rep_out -> all_uniform   -0.0024      4.5%   -0.0010     0.1891
    all_uniform -> all_alloc +0.0003     -0.6%   -0.0012     0.4461
    total                    -0.0544    100.0%   -0.0072

A prediction-level DP/EO regulariser — the thing everyone already does — is
**91% of the effect**. Representation alignment and structural consistency
together are 9% and neither reaches significance. The node-wise allocation is
-0.6%: it moves ΔDP the wrong way.

### And none of it is a Pareto improvement

Pareto per cell, Holm across the family:

    backbone -> out              net  +6   Holm 0.6015   inc 36/54
    out -> rep_out               net  +3   Holm 0.6900   inc 29/54
    rep_out -> all_uniform       net  -8   Holm 0.6015   inc 24/54
    all_uniform -> all_alloc     net  -8   Holm 0.4531   inc 36/54     <- H6
    all_alloc vs all_perm        net  -9   Holm 0.4531   inc 33/54
    all_alloc vs all_uniform1    net -10   Holm 0.4531   inc 28/54

**No rung Pareto-dominates the one below it.** The whole stack does cut
disparity — ΔDP 0.0985 to 0.0441, a 55% reduction at p = 4.2e-05 — and it pays
0.0072 AUC for it at p = 1.1e-03. Both effects are real; their combination is a
trade, and on a per-cell Pareto count it nets to zero.

### H6, as registered

**Falsified.** `all_alloc` does not beat `all_uniform` (net -8, Holm 0.45). The
registered prior said it would be, and said why: `uniform` is already
indistinguishable from `random` in E2-E5.

It also loses to the permutation control (net -9). `all_perm` holds the weight
multiset exactly and destroys only which node receives which weight — so
whatever the allocation does, doing it to the *wrong* nodes is not worse. That
is the same result E4/E5 reached from the other direction, where `w_bdry`
allocated worse than random.

And `all_uniform1`, the ablation the code ships, is not worse than `all_alloc`
either (net -10). It differs from it by a 1.4x-1.75x larger fairness budget, not
by the allocation, so a paper reading it as evidence for allocation was reading
a budget.

### Branch

The registered outcome map treated B ("mass in `backbone -> out` alone") and D
("no rung Pareto-dominates; most cells incomparable") as alternatives. Both
hold: the mass is at rung one *and* no rung dominates. They are not exclusive --
B is about where the effect is, D about whether it is an improvement or a trade
-- and the map should have said so. Recorded as a deviation in the map's
structure, not in the analysis.

The map did not anticipate the backbone offset at all, and that is the largest
number here.
