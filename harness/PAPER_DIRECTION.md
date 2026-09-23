# FairGate — where the paper stands, and what is still open

Self-contained brief, written 2026-09-13 so this can be discussed in a fresh
session with no prior context. Everything below is measured unless it says
otherwise. Numbers are reproducible from `harness/experiments/` against the CSVs
in `harness/results/`.

---

## 1. What happened to the original paper

FairGate was rejected at NeurIPS 2026 and withdrawn from WSDM 2027. Its claim
was: **weight the fairness regulariser per node, choosing which nodes by a
topological signal** (boundary-ness, degree, local homophily).

That claim is dead, and it was killed by our own experiments, not by reviewers.

| test | result |
|---|---|
| Does the topological signal beat allocating at random? | **No.** `w_bdry` — this literature's canonical signal and our draft's — loses to `random` by net −31, p = 0.004 |
| Do any of 11 signals beat random? | One (`bind_influence`), and it buys ΔDP by spending accuracy |
| Does per-graph signal selection help? | No. oracle − best fixed = **0.0006** |
| Does allocation beat spending the same budget uniformly? | **No.** net −8, Holm 0.45 (hypothesis H6, falsified) |
| Does allocation beat the *same weights on random nodes*? | **No.** net −9 |

The last line is the sharpest. `all_perm` keeps the exact multiset of weights
and destroys only *which node gets which*. Giving the same weights to the wrong
nodes is not worse. So the ranking the signal computes carries nothing.

**Attempts to rescue it were made and are recorded**: three operator-matched
structural signals were added under a pre-registered stopping rule (deviation
D4). The rule fired. The search is closed.

---

## 2. What replaced it

Two findings, one from auditing other people's methods, one from dissecting our
own. They reach the same conclusion from opposite directions, which is the
structural reason the paper is defensible.

### Finding 14 — preprocessing decides published conclusions

The comparison the field runs does not share feature normalisation. GNN and
NIFTY load unnormalised; FairGNN normalises two settings of nine; FairGB,
FairGT and FairGate normalise everywhere. The protocol section lists datasets,
splits, seeds, lr and weight decay as shared. Preprocessing is not on that list.

Re-running the audit with normalisation forced on for everyone:

| method | as submitted | normalisation unified |
|---|---|---|
| FairGate | net +43 (Holm 0.0000) | net +39 (0.0000) |
| FairGT | +37 (0.0000) | +37 (0.0000) |
| FairGB | +36 (0.0000) | +36 (0.0000) |
| **NIFTY** | **−14 (0.0052, significant)** | **+5 (0.77, nothing)** |

NIFTY and GNN were *both* unnormalised in one arm and *both* normalised in the
other — the comparison was fair in each arm separately. The answer changes
anyway. A step nobody fixed decides whether a published method beats a plain
GCN.

### Finding 16 — the headline was the backbone

E10 dissects FairGate downward from the finished method, one component at a
time, with every rung spending the **same** total fairness budget (this
required adding a `uniform_budget` mode — the `uniform` ablation the code ships
is not budget-matched and applies 1.4×–1.75× more pressure).

    rung                       dΔDP      share    p
    backbone -> out          -0.0497     91.3%   0.0000    ← the standard regulariser
    out -> rep_out           -0.0026      4.7%   0.0966
    rep_out -> all_uniform   -0.0024      4.5%   0.1891
    all_uniform -> all_alloc +0.0003     -0.6%   0.4461    ← the allocation

Then the attribution check, paired on the same 54 cells:

    backbone   vs audit GNN    net +31   p<0.0001   dAUC +0.0800
    all_alloc  vs audit GNN    net +43   p<0.0001   dAUC +0.0728
    all_alloc  vs backbone     net  +0   p=1.0000   9 win, 9 lose, 36 incomparable

**FairGate's entire fairness stack Pareto-dominates its own backbone on net zero
cells.** Finding 14's 75.9% is mostly architecture and training loop, because
the audit's `GNN` is a different codebase's GCN.

---

## 3. The current claim, and why it is not the obvious one

The obvious claim after Finding 16 is *"these advantages are mostly backbones"*.
**That claim is false and we have the counter-example.** H7 decomposes each
method into `total = backbone + mechanism`:

| method | total | backbone | mechanism | control quality |
|---|---|---|---|---|
| FairGate | +43 | +31 | **+0** | real ablation (λ=0) |
| NIFTY | −14 | −10 | **−2** | real ablation (sim_coeff=0) |
| **FairGB** | +36 | +18 | **+34** | matched encoder, *not* an ablation |
| FairGNN | −1 | *(running)* | | real ablation (α=β=0) |

FairGB survives with its encoder matched. Its mechanism is real. (Caveat kept
attached: its control is a plain GraphSAGE, not FairGB-minus-fairness, because
its mechanism is a sampling augmentation with no coefficient to zero — so
`+34` over-states it by whatever the training loop contributes.)

NIFTY is the opposite and sharper: its ΔDP advantage over GCN (−0.0210) comes
entirely from the backbone (−0.0278), and the counterfactual-invariance
objective *raises* ΔDP by +0.0068. **Turning NIFTY's fairness off makes it
fairer.**

So the claim is:

> The decomposition differs fundamentally by method — FairGate's advantage is
> entirely its backbone, FairGB's is not — **and the published tables cannot
> tell these two cases apart**, because both appear as roughly +0.07 AUC and a
> dominance rate near 0.7.

The contribution is the procedure that tells them apart, not a better method.

---

## 4. "Measurement paper", concretely

A **method paper**'s contribution is an artefact: *here is a new method, it
beats the others*. A **measurement paper**'s contribution is an instrument:
*the way this field measures decides its conclusions; here is how to measure*.

The instrument here is four procedures, each separating a confusion the current
protocol cannot:

| procedure | separates |
|---|---|
| budget-matched component ladder | "the allocation helped" vs "we applied more fairness pressure" |
| method vs **its own** backbone | "the fairness mechanism" vs "the encoder it ships with" |
| Pareto on (AUC↑, ΔDP↓), never scalarised | "fairer" vs "predicting one class" |
| preprocessing-unified arm | "the method" vs "a loader nobody standardised" |

The fourth matters because the field's own appendix warns about the third and
still ranks on ΔDP. EDITS scored ΔDP = 0.000 on two settings by predicting one
class, in the table this study audits.

---

## 5. Scale, and why it matters for credibility

The field's convention is **1 split × 5 seeds**. This study runs **6 splits × 5
inits**, paired per cell, Holm-corrected, ~12,000 training runs across E0–E11.
Measured: the split is the larger variance component and moves the reported
number by 0.034 on average, against a claimed effect size of 0.011. **The
standard protocol cannot resolve the effects it reports.**

Everything is pre-registered: hypotheses H1–H7 and deviations D1–D5 were written
*before* the results they concern, with falsification conditions fixed in
advance. Five claims were retracted mid-study and the retractions are in the
repo.

---

## 6. Open decisions — what to discuss

1. **Venue.** ICML main track needs a claim, which we now have. NeurIPS D&B is
   the fallback. ICML has no benchmark track.
2. **How much of Finding 15 to include.** 117 of 336 cached influence vectors
   had silently diverged to NaN, and `max_num = 0` cannot distinguish "nothing
   to delete" from "the estimator failed". *But* the scale that diverged on NBA
   was **our** guess, not the paper's. Headlining this invites *"that's your
   reimplementation bug"*. Recommendation: appendix, reframed as a structural
   property of the code rather than a defect of BIND.
3. **External validity.** All nine settings are German/NBA/Credit/Bail/Pokec/
   Income — one small family. No answer. Recommendation: state it ourselves in
   Limitations before a reviewer finds it.
4. **What to do with FairGate itself.** It still dominates plain GCN. Do we
   present it as a method at all, given its mechanism is falsified and its
   advantage is its backbone? My view: no — present it as the worked example of
   the procedure, which is honest and makes the paper coherent.
5. **GraphSAGE/SGC robustness.** Required: the finding will otherwise be
   attacked as GCN-specific. Not yet registered or run.

---

## 7. Still running / not finished

| item | state |
|---|---|
| E11 (H7): FairGNN_off | in progress, 120/162 |
| E8 (H4): soft weighting vs hard deletion | queued; a previous run died at 38/210 when `/home` hit 100% |
| BIND repair (Pokec-n/n_g/NBA cells, diverged estimator) | queued |
| GraphSAGE robustness | not registered |

**Server note:** `/home` is at 98% with 2.8 T used, of which ~2.7 T is outside
this account. 73 G was reclaimed by deleting FairGT's regenerable adjacency
cache. This will recur and is not fixable from here.
