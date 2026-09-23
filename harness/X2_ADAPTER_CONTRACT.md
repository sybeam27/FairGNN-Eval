# Phase 1B — Output-semantics audit and the adapter contract

What each implementation actually outputs, established from its **loss function
and label encoding**, never from which orientation scores better. Completed
2026-09-13.

## Why the loss is the evidence

A single output tensor does not mean "logit". `BCELoss` expects a probability,
`BCEWithLogitsLoss` expects a logit, and the two are indistinguishable by shape.
Reading the loss is the only way to know whether a sigmoid has already been
applied, and it is the only evidence used here.

## The contract

### Roster, closed 2026-09-13

    core audit     FairGNN, NIFTY, FairVGNN, FairGB, FairSIN
    mechanistic    FMP
    baseline       GNN — audited like everything else, not counted as a method
    excluded       FairGT, BeMap, BIND, FairGate, FairWalk/CrossWalk, EDITS

The baseline is inside the audit, not outside it. The hard-prediction AUC defect
was found *in the baseline*, so treating it as "just the reference" is exactly
the mistake to avoid. BeMap is dropped because FMP covers the message-passing
role with a cleaner control; it is the fallback only if FMP's
no-propagation arm turns out not to be constructible.

| method | loss | model output | positive class | raw_score to pass to G | original δ |
|---|---|---|---|---|---|
| **GNN** (baseline) | `binary_cross_entropy_with_logits` (GNN.py:339) | single logit | `y = 1` | `forwarding_predict(emb).squeeze()` | `logit > 0` |
| **FairGNN** | `nn.BCEWithLogitsLoss` (FairGNN.py:79) | single logit | `y = 1`; higher logit → higher P(y=1) | `output.squeeze()` | `logit > 0` |
| **NIFTY** | `binary_cross_entropy_with_logits` (NIFTY.py:396-397) | single logit | `y = 1` | `forwarding_predict(emb).squeeze()` | `logit > 0` |
| **FairVGNN** | `binary_cross_entropy_with_logits` (FairVGNN.py:44,64) | single logit | `y = 1` | `output.squeeze()` | `logit > 0` |
| **FairGB** | `binary_cross_entropy_with_logits` (FairGB_alg.py:118,150) | single logit | `y = 1` | `out.squeeze()` | `logit > 0` |
| **FairSIN** | classifier: `binary_cross_entropy_with_logits` (in-train.py:181) | single logit | `y = 1` | `classifier(h).squeeze()` | `logit > 0` |
| **FMP** | `CrossEntropyLoss` (main.py:199) | 2-class logits | index 1 = label 1 | `softmax(all_logit,1)[:,1]` | `prob > 0.5` |
| ~~FairGT~~ | `F.cross_entropy` (FairGT_alg.py:303) | 2-class logits | index 1 | `softmax(logits,1)[:,1]` | `argmax` |

FairGT's row is kept as an audit note. It is out of the roster and receives no
further engineering — no selection audit, no trajectory logging, no flags.

**FairSIN needed care.** Its repository contains both `nn.BCELoss()` (expects a
probability) and `binary_cross_entropy_with_logits` (expects a logit). Tracing
the use: `BCELoss` is `loss_d`, the **discriminator** predicting the sensitive
attribute (in-train.py:201); the label classifier uses `loss_c` with logits
(in-train.py:181). So the raw score is a logit, and the other loss never touches
it. Shape alone would not have distinguished these.

**How positive class was determined, with no reference to results.** For the
single-logit methods, `BCEWithLogitsLoss(logit, y)` with `y ∈ {0,1}` makes the
logit monotone in `P(y = 1)` by construction. For the two-class methods,
`cross_entropy(logits, y)` with `y ∈ {0,1}` makes column *i* the score for
label *i*, so column 1 is the positive class — and both implementations already
index `[:, 1]`, which corroborates without being the reason.

`assert_score_orientation()` exists to catch a wrong declaration. It is
diagnostic and explicitly refuses to negate a score: choosing an orientation
because the other one scores better is a decision made with the evaluation
labels.

## Threshold provenance and sensitive-attribute involvement

| method | threshold source | uses A in δ? | group-specific threshold? |
|---|---|---|---|
| FairGNN, NIFTY, FairGB, BeMap | fixed at 0 (logit) | no | no |
| FairGT | fixed (argmax) | no | no |
| FMP | fixed at 0.5 (probability) | no | no |

**Across the implementations audited here**, decision rules are operationally
aligned: one operating point, no threshold tuned on validation, no sensitive
attribute in δ. That is a statement about these codebases, not about the
literature — it is recorded as a finding and not generalised, on the same rule
that stopped the earlier "four of six methods" claim.

Reporting it matters anyway: an audit that finds fault everywhere it looks is
less credible than one that does not.

## Score semantics do not vary by dataset

Checked rather than assumed: the loaders encode labels identically across
settings, and no method carries a per-dataset output convention. One adapter per
method is therefore sufficient, and the contract collapses to the single table
above.

**But the labels are not all `{0, 1}`.** NBA, Pokec-z and Pokec-n use `-1` for
unlabelled nodes. Measured: the train, validation and test index sets exclude
them (`balanced_split` filters on `labels == 0` and `labels == 1`), so nothing
is currently wrong. Nothing enforced it either — an unlabelled node reaching the
evaluator would be counted as a negative and corrupt AUC and EO together. The
evaluator now refuses any label outside `{0, 1}`.

## Original metric inputs, for the reproduction arm

| method | what its own code fed to `roc_auc_score` |
|---|---|
| FairGNN, NIFTY, GNN | `(logit > 0)` — hard 0/1, so the reported "AUC" is balanced accuracy |
| FairGB | `output[mask]` raw scores — **a real AUC** (`FairGB/eval.py:39`), and this is the number the audit reads. Its *per-sensitive-group* AUC in `predict_sens_group` does use hard predictions, so one implementation computes AUC both ways depending on which number it is producing. Corrected 2026-09-13; the earlier "hard 0/1" entry had been inferred from the group function alone |
| FairGT | `softmax(...)[:,1]` — a real AUC |
| FMP | no AUC selection; reports the final epoch |

These are kept for the reproduction view. The audit view recomputes everything
from `raw_score` through the unified operator.

## Exit criteria, answered per method from code

1. *What score ranks positive examples?* — the column in the contract table.
2. *How is that score converted to a prediction?* — the `original δ` column, all
   equivalent to probability > 0.5.
3. *Did the original implementation use the same objects for its reported
   metrics?* — no, for several: they report AUC from thresholded predictions
   while ranking with the score. **The exact count is deferred** until the
   roster's remaining implementations are audited the same way; quoting a
   fraction now would be generalising from a partial roster, which is the error
   that produced the earlier "four of six" claim.

That third answer is the finding, and it is a statement about the evaluation
operator rather than about anyone's bug.

## Status of earlier results

Everything computed before the unified operator is **pending unified
re-evaluation**. No further per-finding verdicts are issued until 1B, 1C and 1D
are complete and the recomputation runs once, as a whole.

---

## Component switches found while auditing outputs

**FairSIN has both of its claimed components as native flags** — the cleanest
case in the roster.

    N  neutralization   `--delta` (default 5); the term is
                        `encoder(x + delta * model(x), ...)` at in-train.py:179,
                        so delta = 0 removes it exactly
    D  discriminator    `--d` (default **'no'**), gating in-train.py:189

The default matters: **the discriminator is off by default.** "Full FairSIN"
therefore requires `--d yes`, and a run at defaults is already `(N=1, D=0)` —
one cell of the factorial rather than the full method. Any comparison that
assumes defaults are the published configuration would be measuring the wrong
arm.

**FairVGNN's components are not independent and will not be forced into a 2×2.**
Its fair-view generation (node and edge masking, `mask_node_ratio`,
`mask_edge_ratio`) and adaptive weight clamping (`clip_e`, applied at
FairVGNN.py:783) interact through the generated views. So:

    primary contrast      full vs both-off      → τ_total
    per-component arms    reported as **conditional effects**, not marginals

This is a feature of the framework rather than a gap in it: a factorial is
applied where the combinations are meaningful, and withheld where they are not.
FairVGNN is the case that shows the difference is principled.

## Published view vs audit view — FairSIN specifically

FairSIN's own paper reports Accuracy, F1, ΔDP and ΔEO; AUC is not among its
headline metrics. So the paper must not say we "reproduced FairSIN's AUC". The
accurate statement:

> We computed the common audit metric (AUC) externally from the original
> implementation's raw scores.

which is precisely what the unified operator is for, and the distinction is the
paper's own argument applied to itself.
