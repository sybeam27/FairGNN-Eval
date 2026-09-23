# Two arms

**Arm A -- Controlled Audit Arm.** One audit horizon H_audit = 200 for every (method, dataset). Nothing here is called a published protocol. Configuration is the best-attested one per cell -- the authors' own script where an artifact exists, the harness record otherwise -- and M0 differs from M1 only in the claimed intervention. Every tau_int is an effect **conditional on H = 200**.

**Arm B -- Native/Published Protocol Validation Arm.** Only cells whose setting is confirmed in the authors' own repository or paper. Official horizon, preprocessing, hyperparameters and published selector, unchanged.

## Cell labels and arm assignment

| Method | Dataset | Provenance | Source | Commit | H_native | Arm A | Arm B |
|---|---|---|---|---|---:|---|---|
| GNN | german | **third-party-benchmark** | harness/provenance/nifty_README.md (NIFTY GCN baseline) | 6c270c5 | 1000 | yes (as B) | no -- not a published protocol |
| GNN | bail | **local-unverified** | utils/param.json | 6c270c5 | -- | yes (as B) | no -- not a published protocol |
| GNN | credit | **local-unverified** | utils/param.json | 6c270c5 | -- | yes (as B) | no -- not a published protocol |
| FairGNN | german | **third-party-benchmark** | harness/provenance/nifty_README.md (NIFTY, as a baseline) | 13cdca7 | 1000 | yes | no -- not a published protocol |
| FairGNN | bail | **local-unverified** | utils/param.json | 13cdca7 | -- | yes | no -- not a published protocol |
| FairGNN | credit | **local-unverified** | utils/param.json | 13cdca7 | -- | yes | no -- not a published protocol |
| NIFTY | german | **official-repo** | harness/provenance/nifty_README.md | 6c270c5 | 1000 | yes | **yes** |
| NIFTY | bail | **local-unverified** | utils/param.json | 6c270c5 | -- | yes | no -- not a published protocol |
| NIFTY | credit | **local-unverified** | utils/param.json | 6c270c5 | -- | yes | no -- not a published protocol |
| FairVGNN | german | **official-repo** | harness/provenance/fairvgnn_run_german.sh | 938f2e8 | 200 | yes | **yes** |
| FairVGNN | bail | **official-repo** | harness/provenance/fairvgnn_run_bail.sh | 938f2e8 | 300 | yes | **yes** |
| FairVGNN | credit | **official-repo** | harness/provenance/fairvgnn_run_credit.sh | 938f2e8 | 200 | yes | **yes** |
| FairGB | german | **official-repo** | FairGB-main/run.sh | no commit in the distribution | 1500 | yes | **yes** |
| FairGB | bail | **official-repo** | FairGB-main/run.sh | no commit in the distribution | 1500 | yes | **yes** |
| FairGB | credit | **official-repo** | FairGB-main/run.sh | no commit in the distribution | 2000 | yes | **yes** |

## Arm B membership

* **in Arm B (7)**: NIFTY/german, FairVGNN/german, FairVGNN/bail, FairVGNN/credit, FairGB/german, FairGB/bail, FairGB/credit
* **excluded (8)**: GNN/german, GNN/bail, GNN/credit, FairGNN/german, FairGNN/bail, FairGNN/credit, NIFTY/bail, NIFTY/credit

FairGNN/german and GNN/german are `third-party-benchmark`: the setting comes from NIFTY's README, not from FairGNN's authors. They are **not** admitted to Arm B as a published protocol of their own method.

## Arm A cost at H_audit = 200, both arms of every cell

| Dataset | design | FairGNN | NIFTY | FairGB | FairVGNN | total |
|---|---|---:|---:|---:|---:|---:|
| german | 6x5 | 0.37 h | 0.22 h | 0.24 h | 2.27 h | **3.10 h** |
| bail | 3x5 | 0.06 h | 0.12 h | 0.86 h | 0.76 h | **1.80 h** |
| credit | 3x5 | 0.06 h | 0.13 h | 2.62 h | 1.47 h | **4.27 h** |

**Arm A total ≈ 9.2 h** (plus the GNN baseline, ~3-5 s per cell).

## Arm B cost at the native horizon, if it were run in full

| Dataset | FairGNN | NIFTY | FairGB | FairVGNN | total |
|---|---:|---:|---:|---:|---:|
| german | 1.85 h | 1.10 h | 1.77 h | 2.27 h | **7.00 h** |
| bail | n/a | n/a | 6.45 h | 1.14 h | **7.59 h** |
| credit | n/a | n/a | 26.17 h | 1.47 h | **27.63 h** |

**Arm B total ≈ 42.2 h** if every admitted cell ran. Not launched: the cross-dataset Arm A result decides which Arm B cells are worth it.

## Is the existing german 6x5 reusable in Arm A?

**No.** `harness/results/pilot_tau_6x5_audit.csv` was run at H = 200, which matches, but under a harness-uniform configuration: hidden 128 for GNN and NIFTY, NIFTY sim_coeff 0.5, FairVGNN with no top_k/alpha and learning rates an order of magnitude below the authors', FairGB with no alpha or eta. Arm A's policy is the best-attested configuration per cell, so german is re-run under the same policy as bail and credit. The existing run is kept as the configuration-sensitivity arm, not discarded.

## Arm B obligations, recorded before it is run

* **FairGB / german preprocessing** -- upstream normalizes bail and credit and deliberately **not** german (`FairGB-main/data_utils.py:289`), while the harness normalizes all three (`train_baselines.py:639`). Arm B must follow upstream. A harness-normalized german result is not a published protocol.
* **FairGNN published selector** -- the official rule needs two thresholds, `acc_val > args.acc AND roc_val > args.roc` (`train_fairGNN.py:172`); the harness sets only `acc`. Arm B must implement the official rule exactly. This is unimplemented work, not a configuration change.
* **FairVGNN official ablation** -- it retunes clip_e, e_lr, ratio, c_lr, g_epochs and the horizon between the full and both-off lines, so it measures a retuned model. It stays a provenance note. The isolation-preserving M0 remains the audit control in both arms.
