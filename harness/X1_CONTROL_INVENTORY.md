# X1 — Control inventory: does each method's "fairness off" switch exist, and what does it turn off?

Run 2026-09-13 by source inspection, before any comparison. **No performance
claim is made here.** X1 asks one question per method: if we set its fairness
mechanism to zero, what exactly stops happening?

This step exists because the study got this wrong twice in one day. `sim_coeff=0`
was called "NIFTY off" when it removes only the objective and leaves the
Lipschitz constraint running. FairGNN's intervention looked like nothing, when
in fact the benchmark's inherited α was simply small — α = 100 moves ΔDP from
0.112 to 0.069. A control that is not what you think it is invalidates every
number computed from it.

---

## Fidelity grades

    A   real ablation — a coefficient goes to zero, nothing else changes
    B   partial — only some of the claimed mechanism is switchable
    C   no switch in the code; one can be added without changing behaviour
    D   not separable — the mechanism is the architecture

## Findings

| method | claimed components | switch | now | achievable |
|---|---|---|---|---|
| **FairGNN** | α·covariance, β·adversarial | `G_loss = cls + α·cov − β·adv` (FairGNN.py:131) — **fully separable** | **A** | A (2×2) |
| **FMP** | λ₁ fairness projection, λ₂ propagation | `if lambda1 > 0:` / `if lambda2 > 0:` (fmp.py:135,141) — **explicit branches** | **A** | A (2×2) |
| **FairGate** | 3-level loss, node allocation | `lambda_fair`, `fiw_weight_mode` | **A** | done (E10) |
| **NIFTY** | counterfactual objective, Lipschitz | `sim_coeff` yes; `spectral_norm` hard-wired in fc1–fc4 and Classifier | **B** | A with one flag |
| **FairGB** | CAL alignment, CNM mixup | neither has a flag: mixup at FairGB_alg.py:107-109, group weights at :142-143 | **C** | A with two flags |
| **FairGT** | eigenvector PE, same-sens graph, k-hop features | three pieces, all inline: `eigsh` :35, `_get_same_sens_complete_graph` :254, `_re_features` :262 | **C** | A with three flags |
| **BeMap** | balanced neighbour sampling | `lam`, `beta`, `save_num` tune one mechanism; no bypass found | **C** | A with one flag |

### The FMP finding

FMP is worth more than a row. Its two coefficients are not two fairness
components — `lambda2` gates the *propagation* itself and `lambda1` gates the
fairness projection. So one codebase yields the whole GNN-specific ladder with
no cross-implementation confound:

    λ₁ = 0, λ₂ = 0    no propagation, no fairness      ≈ MLP
    λ₁ = 0, λ₂ > 0    propagation, no fairness         ≈ GNN base
    λ₁ > 0, λ₂ > 0    full FMP

Graph Bias Amplification and Bias Recovery can be measured inside a single
implementation rather than across three. Nothing else in the roster can do this.

### Dataset coverage — a hard constraint

    FairGNN, NIFTY, FairGB, FairGT, FairGate   9 settings
    BeMap                                      pokec_z, nba, credit, bail (4)
    FMP                                        pokec_z, pokec_n, nba (3)

Common to all: **pokec_z and nba only.**

This does not damage the primary estimand. `E_intervention = E(M1, M0)` is a
*within-method* comparison, so unequal coverage cannot confound it. Coverage
matters only when aggregating across methods, so:

    Tier 1   pokec_z, nba          cross-method statements, pooled ARR
    Tier 2   each method's own     per-method attribution, taxonomy placement

Any pooled rate is reported per method as well, because unequal coverage
weights methods unequally in a pooled number.

---

## What X1 changes in the plan

1. **FairGNN, FMP and FairGate can be run now.** Three A-grade controls, two of
   them 2×2, one already complete.
2. **NIFTY needs one flag** (spectral_norm off) to reach A and complete its 2×2.
3. **FairGB, FairGT and BeMap need flags added** before they can be run at all.
   Every added flag must be verified to leave the default path byte-identical.
4. **FMP moves to the front of the GNN-specific analysis**, replacing the
   cross-codebase MLP/GNN-base construction originally planned.
5. No method is grade D. The dichotomy "backbone vs mechanism" was going to
   break on FairGT; with three separable pieces it may not have to.

---

## Addendum 2026-09-13 — factorial closure, measured

Two methods turn out **not** to have crossed components, so a 2×2 is not
available for them and is not manufactured.

**FairGB: CAL is nested inside CNM.** The contribution-alignment reweighting
(`group_weight_list`, FairGB_alg.py:143-151) sits *inside* the mixup branch
`if epoch >= args.warmup:`, and it reweights `loss_src` / `loss_dst`, which only
exist when the mixup has produced them. With CNM off the CAL code is
unreachable, so `(CAL=1, CNM=0)` and `(CAL=0, CNM=0)` are the same model by
construction. Measured on German, 25 epochs: both give acc 0.6520, auc 0.7323,
dp 0.6859, identical to the full method — the selected checkpoint there predates
the warmup in every arm, which is a separate matter and is why the pilot uses a
longer horizon.

**FairVGNN: already recorded** — fair-view generation and weight clamping
interact through the generated views.

    method     components            closure   treatment
    FairGNN    cov  × adv            yes       full 2×2
    NIFTY      obj  × lip            yes       full 2×2
    FairGB     CAL ⊂ CNM             **no**    full vs both-off; CAL as a
                                               conditional effect given CNM
    FairVGNN   mask ~ clip           **no**    full vs both-off; per-component
                                               as conditional effects

Recording closure per method is what keeps the framework from forcing every
method into the same table. Where the combinations are meaningful, a factorial
runs; where they are not, the primary contrast is the whole intervention against
its absence and the parts are reported as conditional.

One more line reproduced rather than repaired: FairGB_alg.py:147 indexes `y` by
`sampling_src_idx` and `sens` by `sampling_dst_idx`, while the `grad_count` loop
twelve lines above uses `src` for both. Left as published, with the
inconsistency noted in a comment at the site.
