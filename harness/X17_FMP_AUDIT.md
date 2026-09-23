# FMP mechanistic extension: provenance, intervention structure, feasibility

This is a preparation document. **No FMP training has been launched**, and none
will be before the core experiments and Phase 1 native validation are finished.

**Role.** FMP is **not** a fifth benchmark method. In the paper it is a
*mechanistic case study separating graph-specific propagation from the claimed
fairness intervention*. It gets no win counts and no ranking against the four
core methods.

## 1. Official artifact

| item | value | source |
|---|---|---|
| paper | Jiang et al., *Chasing Fairness in Graphs: A GNN Architecture Perspective*, AAAI 2024 (arXiv 2312.12369) | paper |
| official repo | https://github.com/zhimengj0326/FMP | paper, "code is available at" |
| commit | `7c528413cf110c68becc9cd4f3b7a85cf496447b` (2023-08-17, "update paper"), HEAD | GitHub API |
| local copy `FMP-main/` | **byte-identical** to that commit for `fmp.py`, `main.py`, `fairgnn.py`, `utils.py`, `module.py`, `run_fgnn.sh`, `README.md` | `harness/provenance/fmp/official_*` |
| supported datasets | `pokec_z`, `pokec_n`, `nba` only (`main.py:34-35`, `choices=[...]`) | code |
| execution script | `run_fgnn.sh`: only `pokec_n` is active; `pokec_z` and `nba` appear only in comments | code |
| backbone | MLP, `num_gnn_layer` Linear layers with 64 hidden units, output 2 logits, then the FMP layer on the logits | `fairgnn.py:8-39` |
| FMP layer | K = `--num-layers` = **5** propagation-debiasing steps, no trainable parameters | `fairgnn.py:47-53`, `main.py:42` |
| optimizer | Adam, lr 0.001, **no weight decay passed** (`optim.Adam(model.parameters(), lr=args.lr)`) | `main.py:200` |
| horizon | 300 epochs | `run_fgnn.sh` |
| native checkpoint selector | **none: the last epoch is reported** | `main.py:202-267` |
| metrics | acc (threshold 0.5), AP, AUC on P(y=1), D_SP and D_EO at threshold 0.5 | `main.py`, `utils.py:438-470` |
| preprocessing | pokec: none; nba: `feature_norm` to [-1, 1] | `main.py:119-120` |
| split | `load_pokec(..., train_ratio=0.8, seed=20)`: 80/10/10 on pokec; on **nba `test_idx=True` makes validation = test** | `utils.py:65-147` |
| runs | 5 runs in one process, one split; `np.random.seed` and `torch.cuda.manual_seed` only, and the CPU generator that initializes `nn.Linear` is never seeded | `main.py:76-77, 167` |

### Paper vs official code: discrepancies, both recorded

Rule applied: the executable official configuration takes precedence, and
every discrepancy is stated.

| item | paper (Appendix I, verbatim) | official code | taken |
|---|---|---|---|
| split | "50%/25%/25% for training, validation, and test" | 80/10/10; nba val = test | code |
| propagation / debiasing stacks | "stack 2 layers" | K = 5 (`--num-layers` default, not set by `run_fgnn.sh`) | code |
| projection | l-infinity ball (element-wise clamp) | `--L2 True` default: `L2_projection`, a scaling by `2λ/(2λ+β)` | code |
| weight decay | "1×10⁻⁵ weight decay" | parsed but not passed to Adam, so 0 | code |
| MLP depth | "2 layers of MLP" | `num_gnn_layer` in {2, 5} in the script | undetermined; see §7 |
| runs | "55 times" (apparently a typo for 5) | `running_times` 5 | code |
| **λ values** | grids only: λ_f ∈ {0,5,10,15,20,30,100}, λ_s ∈ {0,0.01,0.1,0.5,1,2,3,5,10,15,20} | `run_fgnn.sh` sweeps λ1 ∈ {5,15,20,30} × λ2 ∈ {0,…,20}; parser defaults 3 and 3 are not published | **undetermined** |
| selection | not stated | last epoch | code |

**Neither the paper nor the repository reports a single chosen (λ1, λ2) for any
dataset, or a rule for choosing one.** The native λ is therefore
**provenance unresolved**. It is not replaced by the parser default (3, 3),
which nothing publishes.

## 2. λ1 and λ2 traced through the code

`FMP.emp_forward` (`fmp.py:122-186`), for each of K steps, with
`γ = 1/(1+λ2)` and `β = 1/(2γ)`:

```
propagation   λ2 > 0 : y = γ·hh + (1-γ)·propa(g, x)     # APPNP-style, DGL GraphConv (norm='both')
              λ2 = 0 : y = γ·hh + (1-γ)·x = hh           # γ = 1: no propagation, not even the propa call
debiasing     λ1 > 0 : z   = sen·softmax(y) / (γ·sen·senᵀ)
                       correction from sen, z, softmax
                       z_bar = z + β·sen·softmax(x_bar)
                       z  = L2_projection(z_bar, λ1, β)  (or L1 clamp)
                       x  = y - γ·correct
              λ1 = 0 : x = y                               # "z = 0"
dropout (p = 0 by default)
```

The paper's objective agrees with this reading:
`min_F λ_s/2·tr(FᵀL̃F) + ½‖F − X_trans‖² + λ_f·‖Δ_s SF(F)‖₁`, with λ_s = λ2
(smoothness) and λ_f = λ1 (fairness).

Verified:

* **λ1 = 0 removes the fairness intervention.** The `if lambda1 > 0` branch
  is skipped and `x = y`. This is also the continuous limit: under the L2
  projection `coeff = 2λ1/(2λ1+β) → 0` gives `z = 0`, which gives
  `correct = 0`, which gives `x = y`. Under the L∞ clamp, `clamp(±0) = 0` gives
  the same. The zero setting is therefore a valid parameterization, not a
  degenerate edge.
* **λ2 = 0 removes propagation.** `γ = 1` gives `y = hh`, so F00 is exactly
  the MLP.
* **λ2 is not purely propagation.** γ also enters the debiasing step:
  `z = …/(γ·sen·senᵀ)`, `x = y − γ·correct`, `β = 1/(2γ)`. Changing λ2
  therefore changes both whether and how much propagation happens and the
  scale of the fairness correction.
* The two interleave: debiasing at step k feeds propagation at step k+1. That
  is the method, not a defect.

Consequences for the three target states:

| state | code path | valid |
|---|---|---|
| F00 = (0, 0) | γ = 1, `y = hh`, `x = y` at every step: pure MLP | **yes** |
| F01 = (0, λ2) | γ = 1/(1+λ2), K-step propagation, no debiasing | **yes** |
| F11 = (λ1, λ2) | propagation and debiasing interleaved, same γ as F01 | **yes** |
| F10 = (λ1, 0) | γ = 1, no propagation, debiasing at step scale γ = 1, β = 0.5 | valid path, **confounded**: its debiasing runs at a different step scale than F11's; not added (see §3) |

## 3. Primary mechanistic design

* **Fairness contrast, `τ_fair = Y(F11) − Y(F01)`: scientifically valid.** Both
  share λ2, hence γ and the exact propagation operator, and differ only in the
  `λ1 > 0` debiasing branch. It is the effect of debiasing **given native
  propagation**, a conditional effect, because debiasing and propagation
  interleave across steps.
* **Propagation contrast, `τ_prop = Y(F01) − Y(F00)`: scientifically valid.**
  Debiasing is off in both. F00 is the MLP; F01 adds K-step propagation.
* **`τ_total = Y(F11) − Y(F00) = τ_prop + τ_fair`** is a decomposition check
  only.
* **Optional interaction cell F10 = (λ1, 0): not added.** The path runs, but
  at λ2 = 0 the debiasing step uses γ = 1 and β = 0.5 rather than F11's γ, so
  `F10 − F00` is not the same intervention as `F11 − F01`. An interaction built
  from it mixes "no propagation" with "differently scaled debiasing".

## 4. Official-code defects found

These do not change the design, but they bind the adapter.

1. **`get_sen` mutates the caller's `sens` in place** (`fmp.py:18`, `sens_1 = sens` is an alias).
   * Every model's first forward calls it, including λ1 = 0 models, because
     the cache is empty.
   * CPU probe: a second call on the same tensor gives a sensitive vector
     that differs from a fresh one by up to 0.005, about twice the ±1/n entry
     size.
   * Mechanism: after call 1 a positive training node holds 1/n, so on call 2
     `sens_0 = 1 − 1/n` makes it read as negative, and its group signal is
     erased.
   * Hence runs 2-5 of the official 5-run loop debias against a corrupted
     group vector.
   * Validation and test entries are not touched: `idx_sens_train` excludes
     them, and sign is preserved. So the reported metrics use the right
     groups.
   * With an integer `sens` the call raises; the official loader passes a
     float, so it does not.
   * **Adapter obligation:** hand each model a fresh copy of `sens`.
     Otherwise F01 built after F11 in the same cell debiases, or later
     evaluates, against a different vector.
2. **The CPU generator is never seeded**, so MLP initialization is not
   reproducible across launches.
3. **`FairGNN.reset_parameters` references `lin1` and `lin2`, which do not
   exist.** It is never called.
4. **On nba, validation is test.**

## 5. Can the current evaluation contract be reused?

| contract element | FMP status |
|---|---|
| trajectory logging | **yes**, via an adapter re-implementing `main.py`'s 30-line loop; the eval forward already happens every epoch |
| raw validation scores | **yes**: `logit[:,1] − logit[:,0]`; `> 0` is identical to `softmax > 0.5` |
| checkpoint restoration | **yes**: the trainable state is the MLP; FMP has no parameters. The cached `sen` is deterministic given the fresh-copy fix |
| native selector replay | **yes**, trivially: last epoch (`select_final`) |
| sigma_c^BCE, sigma_c^AUC | **pokec_z, pokec_n: yes** (separate validation). **nba: no**, because validation = test, so any validation selector selects on test |
| unified evaluator G_c | **yes**, `score > 0`; DP/EO groups `sens == 0` vs `> 0` match `fair_metric` |
| stochastic inference / RNG bundle | **not needed**: FMP dropout is 0 and the MLP has no dropout, so inference is deterministic |
| test isolation | pokec: yes (test metrics are logged by the native code but never used to select); nba: violated for any validation-based selector |
| module import | **blocked as-is**: `fmp.py` imports `dgl` at module level, and `dgl` is not in this env (torch 2.6; the official pin is torch 1.12 + dgl 1.0.1+cu113). A `dgl-1.0.1` CPU wheel exists for cp311 but is not ABI-matched to torch 2.6. Nothing installed |

Adapter feasibility, two faithful routes. Neither edits `FMP-main/`.

* **(a) Separate env:** torch 1.12, dgl 1.0.1 (CUDA build), running the
  official modules directly. Highest fidelity; needs a new environment.
* **(b) This env:** a torch/`torch_sparse` reimplementation of exactly
  `GraphConv(weight=False, bias=False, norm='both')` on the graph after
  `remove_self_loop`/`add_self_loop`, i.e. `D^-1/2 (A+I) D^-1/2 x`, and
  `get_sen` with a copy. It must be verified numerically against DGL, in env
  (a) or a throwaway env, before use.

## 6. Datasets

* **Official FMP datasets ∩ {german, bail, credit} = ∅.**
* No port is made: `load_pokec` has no german/bail/credit branch, and none is
  invented.
* **Proposal:** a separate mechanistic case study on FMP's own datasets.
  * pokec_z and pokec_n are the primary candidates: σ_c-auditable, and our
    `data/pokec/region_job*.csv` and `*_relationship.txt` match the loader's
    file names.
  * **nba is excluded** from any σ_c-based audit because validation = test.
    At most it gets the native last-epoch view.

## 7. Table requested before any run

| dataset | official source | H | preprocessing | λ1 native | λ2 native | native selector | F00 valid? | F01 valid? | F11 valid? | expected runtime (est., 3 arms/cell) |
|---|---|---|---|---|---|---|---|---|---|---|
| pokec_z | zhimengj0326/FMP `7c52841` (commented in `run_fgnn.sh`) | 300 | none | **unresolved** | **unresolved** | last epoch | yes | yes | yes, once λ fixed | 3x5: ~0.4-0.8 h; 6x5: ~0.8-1.5 h |
| pokec_n | same, active in `run_fgnn.sh` | 300 | none | **unresolved** (swept 5-30) | **unresolved** (swept 0-20) | last epoch | yes | yes | yes, once λ fixed | 3x5: ~0.4-0.8 h; 6x5: ~0.8-1.5 h |
| nba | same, commented | 300 | feature_norm [-1, 1] | **unresolved** | **unresolved** | last epoch | yes | yes | yes | native view only: minutes; **σ_c audit invalid (val = test)** |

**Runtime is estimated, not measured.** No FMP run is allowed yet, and GPU 2 is
busy with FairGB/bail.
* Basis: pokec_z has 67,796 nodes and 882,765 edge lines; pokec_n has 66,569
  and 729,129.
* Per epoch: a 2-layer MLP forward/backward plus K = 5 sparse propagations
  plus 1×N debiasing algebra plus one eval forward, about 0.1-0.2 s on GPU. The
  adapter records validation scores only, without the native per-epoch sklearn
  metrics on train/val/test.
* Per model: 300 epochs, about 30-60 s. One (split, run) cell has 3 arms.
* One real cell is timed after Phase 1, before any budget decision. Any λ grid
  multiplies these costs by the number of λ points.

## 8. Open decisions, needed before an FMP run can be specified

1. **Native (λ1, λ2).** No artifact fixes one. Options:
   * a pre-registered small grid drawn from the paper's own λ values, reporting
     τ_fair and τ_prop as functions of λ;
   * a single pair named in advance with its rationale, labelled
     *local-choice*, never *published*.

   A parser default is never used as the native value.
2. **Configuration discrepancies.** Current reading: take the executable
   official code (K = 5, L2 projection, 80/10/10, wd 0). The paper's settings
   (K = 2, L∞, 50/25/25, wd 1e-5) could be a named sensitivity arm, not the
   primary.
3. **MLP depth** (`num_gnn_layer` 2 or 5; paper "2 layers"): 2 is consistent
   across paper and script; recommended, to be confirmed.
4. **Adapter route:** (a) separate env or (b) reimplementation verified
   against DGL.
5. **`get_sen` copy fix:** deviates from the official multi-run loop but is
   required for paired arms. Recommended; recorded as a deliberate deviation
   from an official defect.
6. **Splits:** the native loader uses one split (seed = 20). Split IDs 20-25
   would use the loader's own `seed` argument at non-native values. This needs
   confirmation that it counts as native.

Nothing is run until these are decided and the core experiments and Phase 1
are complete.
