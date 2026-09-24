# study/ — allocation re-experiments

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
python study/tests/test_core.py            # synthetic, seconds
python study/tests/test_core.py --data     # loads data/, ~1 min (Credit is 108 MB)
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
`study/data_manifest.tsv` now carries a sha256 for every data file. Second,
`algorithms/FairGT_alg.py::_get_same_sens_complete_graph` caches a derived
tensor at `algorithms/adj_files/{dataset}_same_sens_complete_adj.pt`, keyed by
dataset *name* with no content hash. Today that is harmless -- the tensor is a
function of `sens` alone, and the Credit swap changed only the edge list -- but
any name-keyed cache over a file that has already been silently replaced once
is a repeat of the same failure waiting to happen. Anything derived in
`study/` is keyed by content.

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

## Known issues in the old code (not fixed here)

* `signal_diagnostics.py:414` — `--node-set sens` treats `get_dataset`'s second
  return value as a node index array, but it is a *feature column* index, so
  the evaluation set collapses to one node. Moot in practice: no setting has
  `sens < 0`, so `sens` and `all` coincide.
* `utils/data.py:349` — the German loader assigns ints into a string column and
  raises under pandas 3.
* `algorithms/` contains `FairGB_alg copy.py`, `FairWalk copy.py`,
  `CrossWalk copy.py` alongside the originals. Which one produced the submitted
  numbers is not recoverable. Nothing in `study/` may have a `copy` twin.

## Still to build

`core/objectives.py`, `core/trainer.py`, `experiments/e2_signal_swap.py`,
`adapters/{bind,fairgb,fairsin}.py`. Backbone will be a plain-torch GCN using
the pre-normalised adjacency (numerically the same propagation as PyG
`GCNConv`, but with no PyG dependency and CPU-friendly).
