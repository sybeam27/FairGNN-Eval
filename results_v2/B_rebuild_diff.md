# B rebuild — what changed, per `B_rebuild_decision.md` Deliverables

Four baselines. The method arms are identical in all four; only B differs, so `tau_{-I->+I}` is the same everywhere and is not repeated here.

| name | rule | role |
|---|---|---|
| frozen | B re-trained separately per method; German at H = 200 | the published bundle |
| **B_rep1** | one B per unit, resolved published horizon where one exists (German 1000) | **reported** |
| B_rep2 | the identical rule, independently re-trained | nondeterminism |
| B_H1000 | H = 1000 on every dataset | baseline strength |

## 1. B's own metrics, per dataset

| dataset | horizon rep1 | distinct draws frozen | AUC frozen | rep1 | rep2 | H1000 |
|---|---|---|---|---|---|---|
| german | 1000 | 9 | 0.4415 | 0.6543 | 0.6544 | 0.6525 |
| bail | 200 | 11 | 0.7734 | 0.7734 | 0.7734 | 0.9069 |
| credit | 200 | 9 | 0.7186 | 0.7177 | 0.7179 | 0.7304 |
| income | 200 | 1 | 0.6530 | 0.6520 | 0.6538 | 0.7027 |
| pokec_z | 200 | 4 | 0.7038 | 0.7038 | 0.7038 | 0.7290 |
| pokec_n | 200 | 3 | 0.7112 | 0.7112 | 0.7112 | 0.7145 |
| pokec_z_g | 200 | 1 | 0.7038 | 0.7038 | 0.7038 | 0.7290 |
| pokec_n_g | 200 | 1 | 0.7110 | 0.7110 | 0.7110 | 0.7144 |

German is the only dataset whose horizon changes, and the only one whose frozen baseline was below chance. Everywhere else frozen and rep1 differ only in that B is now one draw per unit rather than one per method.

## 2. Headline blocks, four baselines

**package larger** (of 36)

| coordinate | frozen | B_rep1 | B_rep2 | B_H1000 |
|---|---|---|---|---|
| dAUC | 31 | 27 | 27 | 20 |
| negDP | 26 | 26 | 26 | 19 |
| negEO | 29 | 29 | 29 | 20 |

**tau_pkg < 0** (of 36)

| coordinate | frozen | B_rep1 | B_rep2 | B_H1000 |
|---|---|---|---|---|
| dAUC | 9 | 18 | 18 | 30 |
| negDP | 23 | 19 | 19 | 13 |
| negEO | 24 | 20 | 20 | 16 |

**opposite sign** (of 36)

| coordinate | frozen | B_rep1 | B_rep2 | B_H1000 |
|---|---|---|---|---|
| dAUC | 19 | 16 | 16 | 14 |
| negDP | 18 | 18 | 18 | 10 |
| negEO | 15 | 15 | 15 | 11 |

**median |tau_{B->-I}|**

| coordinate | frozen | B_rep1 | B_rep2 | B_H1000 |
|---|---|---|---|---|
| dAUC | 0.1042 | 0.0368 | 0.0368 | 0.0164 |
| negDP | 0.0387 | 0.0396 | 0.0388 | 0.0157 |
| negEO | 0.0279 | 0.0276 | 0.0279 | 0.0153 |

## 3. Cells whose 'package larger' verdict changes

### frozen -> B_rep1

*dAUC* — 4 cells

* EDITS/credit [default] — larger -> not larger
* EDITS/german [default] — larger -> not larger
* FairSIN/german [GCN] — larger -> not larger
* SFG/german [default] — larger -> not larger

*negDP* — 2 cells

* FairGNN/german [default] — larger -> not larger
* SFG/german [default] — not larger -> larger

*negEO* — 2 cells

* FairGNN/german [default] — larger -> not larger
* SFG/german [default] — not larger -> larger

### B_rep1 -> B_rep2

No cell changes on any coordinate.

### B_rep1 -> B_H1000

*dAUC* — 7 cells

* EDITS/bail [default] — larger -> not larger
* FairGB/bail [default] — larger -> not larger
* FairGNN/pokec_z [default] — larger -> not larger
* FairGNN/pokec_z_g [default] — larger -> not larger
* FairSIN/pokec_z [GCN] — larger -> not larger
* FairVGNN/bail [default] — larger -> not larger
* FairVGNN/credit [default] — larger -> not larger

*negDP* — 9 cells

* BIND/income [1pct] — larger -> not larger
* EDITS/bail [default] — larger -> not larger
* FairGB/credit [default] — larger -> not larger
* FairGNN/credit [default] — not larger -> larger
* FairGNN/pokec_z [default] — larger -> not larger
* FairSIN/bail [GCN] — larger -> not larger
* FairSIN/pokec_z [GCN] — larger -> not larger
* GEAR/bail [default] — larger -> not larger
* SFG/bail [default] — larger -> not larger

*negEO* — 11 cells

* BIND/income [1pct] — larger -> not larger
* BeMap/bail [default] — larger -> not larger
* EDITS/bail [default] — larger -> not larger
* FairGNN/credit [default] — not larger -> larger
* FairGNN/pokec_z [default] — larger -> not larger
* FairGNN/pokec_z_g [default] — larger -> not larger
* FairSIN/bail [GCN] — larger -> not larger
* FairSIN/credit [GCN] — larger -> not larger
* FairSIN/pokec_z [GCN] — larger -> not larger
* FairVGNN/bail [default] — larger -> not larger
* GEAR/bail [default] — larger -> not larger

## 4. tau_{-I->+I} is untouched

All 36 primary cells (and all 90 cells in the bundle) reproduce every `tau_I` mean, bound and resolved flag at max |diff| 0.0. Mismatches found: **0**.

This is structural — `tau_I` is formed from `m1` and `m0` alone and never reads B — but the rebuild asserts it rather than relying on the argument, and stops if it fails.

## 5. What carries a B-referenced number

Tables: `tab:exp1-summary`, `tab:exp1-cells-{dAUC,negDP,negEO}`, `tab:exp1-1-fnrgnn`, `tab:aggregate_robustness`, `tab:controlled_protocol_contract`.

Figures: `fig1_intervention_attribution` panel (a) and its EO variant. `fig3_protocol_variation` and `fig3_selection_support_trajectory` plot `tau_I` only and do not move.

Appendix K (the three FMP figures) is **not** affected: FMP's baseline is built by its own runner at H = 300 on pokec_z and pokec_n, where `published()` resolves `horizon = None`, and it is already one draw per unit.

