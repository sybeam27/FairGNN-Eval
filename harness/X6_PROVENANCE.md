| Method | Dataset | Source | Commit | Config path | Preproc | Horizon | Published selector | M0/M1 | Provenance |
|---|---|---|---|---|---|---:|---|---|---|
| GNN | german | https://github.com/chirag126/nifty | 6c270c5 | harness/provenance/nifty_README.md (NIFTY GCN baseline) | raw | 1000 | argmin(val loss)  [nifty_sota_gnn.py:265] | baseline, no intervention | **paper-specified** |
| GNN | bail | https://github.com/chirag126/nifty | 6c270c5 | utils/param.json | raw | **undetermined** | argmin(val loss)  [nifty_sota_gnn.py:265] | baseline, no intervention | **local-unverified** |
| GNN | credit | https://github.com/chirag126/nifty | 6c270c5 | utils/param.json | raw | **undetermined** | argmin(val loss)  [nifty_sota_gnn.py:265] | baseline, no intervention | **local-unverified** |
| FairGNN | german | https://github.com/EnyanDai/FairGNN | 13cdca7 | harness/provenance/nifty_README.md (NIFTY, as a baseline) | norm | 1000 | first epoch with acc_val>args.acc AND roc_val>args.roc  [train_fairGNN.py:172] | M1 alpha,beta published; M0 alpha=0,beta=0 (claimed fairness components only) | **paper-specified** |
| FairGNN | bail | https://github.com/EnyanDai/FairGNN | 13cdca7 | utils/param.json | raw | **undetermined** | first epoch with acc_val>args.acc AND roc_val>args.roc  [train_fairGNN.py:172] | M1 alpha,beta published; M0 alpha=0,beta=0 (claimed fairness components only) | **local-unverified** |
| FairGNN | credit | https://github.com/EnyanDai/FairGNN | 13cdca7 | utils/param.json | raw | **undetermined** | first epoch with acc_val>args.acc AND roc_val>args.roc  [train_fairGNN.py:172] | M1 alpha,beta published; M0 alpha=0,beta=0 (claimed fairness components only) | **local-unverified** |
| NIFTY | german | https://github.com/chirag126/nifty | 6c270c5 | harness/provenance/nifty_README.md | raw | 1000 | argmin(val_c_loss + val_s_loss)  [nifty_sota_gnn.py:325] | M1 sim_coeff published; M0 sim_coeff=0, compared at the common selector so OFF does not change selector semantics | **official-repo** |
| NIFTY | bail | https://github.com/chirag126/nifty | 6c270c5 | utils/param.json | raw | **undetermined** | argmin(val_c_loss + val_s_loss)  [nifty_sota_gnn.py:325] | M1 sim_coeff published; M0 sim_coeff=0, compared at the common selector so OFF does not change selector semantics | **local-unverified** |
| NIFTY | credit | https://github.com/chirag126/nifty | 6c270c5 | utils/param.json | raw | **undetermined** | argmin(val_c_loss + val_s_loss)  [nifty_sota_gnn.py:325] | M1 sim_coeff published; M0 sim_coeff=0, compared at the common selector so OFF does not change selector semantics | **local-unverified** |
| FairVGNN | german | https://github.com/YuWVandy/FairVGNN | 938f2e8 | harness/provenance/fairvgnn_run_*.sh | norm | 200 | auc+F1+acc - alpha*(parity+equality), floor 0  [fairvgnn.py:185] | dependency-constrained: M1 f_mask=yes, weight_clip=yes; M0 both no, every other setting held at M1's | **official-repo** |
| FairVGNN | bail | https://github.com/YuWVandy/FairVGNN | 938f2e8 | harness/provenance/fairvgnn_run_*.sh | raw | 300 | auc+F1+acc - alpha*(parity+equality), floor 0  [fairvgnn.py:185] | dependency-constrained: M1 f_mask=yes, weight_clip=yes; M0 both no, every other setting held at M1's | **official-repo** |
| FairVGNN | credit | https://github.com/YuWVandy/FairVGNN | 938f2e8 | harness/provenance/fairvgnn_run_*.sh | raw | 200 | auc+F1+acc - alpha*(parity+equality), floor 0  [fairvgnn.py:185] | dependency-constrained: M1 f_mask=yes, weight_clip=yes; M0 both no, every other setting held at M1's | **official-repo** |
| FairGB | german | vendored zip, FairGB-main/ | no commit in the distribution | FairGB-main/run.sh | norm | 1500 | alpha-weighted parity+equality tradeoff  [FairGB-main] | nested: M1 CAL on + CNM on; M0 both off (full vs all-off) | **official-repo** |
| FairGB | bail | vendored zip, FairGB-main/ | no commit in the distribution | FairGB-main/run.sh | norm | 1500 | alpha-weighted parity+equality tradeoff  [FairGB-main] | nested: M1 CAL on + CNM on; M0 both off (full vs all-off) | **official-repo** |
| FairGB | credit | vendored zip, FairGB-main/ | no commit in the distribution | FairGB-main/run.sh | norm | 2000 | alpha-weighted parity+equality tradeoff  [FairGB-main] | nested: M1 CAL on + CNM on; M0 both off (full vs all-off) | **official-repo** |


## Hyperparameters, as parsed

* **GNN / german** lr=0.001 num_hidden=16 num_proj_hidden=16 weight_decay=0.0
* **GNN / bail** lr=0.001 num_hidden=8 num_proj_hidden=4 weight_decay=0.0
* **GNN / credit** lr=0.001 num_hidden=32 num_proj_hidden=4 weight_decay=0.0
* **FairGNN / german** acc=0.36000000000000004 alpha=8 beta=0.005 lr=0.001 num_hidden=16 weight_decay=0.0
    * note: FairGNN's own repository configures nba, pokec_n and pokec_z only -- never german, bail or credit
    * note: alpha/beta come from param.json and remain local-unverified
* **FairGNN / bail** acc=0.54 alpha=2 beta=0.05 lr=0.001 num_hidden=128 weight_decay=0.0
    * note: FairGNN's own repository configures nba, pokec_n and pokec_z only -- never german, bail or credit
* **FairGNN / credit** acc=0.3 alpha=1 beta=0.005 lr=0.001 num_hidden=128 weight_decay=0.0
    * note: FairGNN's own repository configures nba, pokec_n and pokec_z only -- never german, bail or credit
* **NIFTY / german** lr=0.001 num_hidden=16 num_proj_hidden=16 sim_coeff=0.6 weight_decay=1e-05
* **NIFTY / bail** lr=0.001 num_hidden=8 num_proj_hidden=4 weight_decay=1e-05
    * note: NIFTY's repository states german only; this dataset has no command in any artifact
* **NIFTY / credit** lr=0.001 num_hidden=8 num_proj_hidden=8 weight_decay=1e-05
    * note: NIFTY's repository states german only; this dataset has no command in any artifact
* **FairVGNN / german** K=10 alpha=1 c_epochs=10 c_lr=0.01 c_wd=0 clip_e=0.1 d_epochs=5 dropout=0.5 e_lr=0.001 e_wd=0 g_epochs=5 hidden=16 prop=scatter ratio=0 top_k=10
    * note: official ablation retunes clip_e:0.1->1, e_lr:0.001->0.01, ratio:0->1
    * note: param.json disagrees (top_k/alpha [5, 0.5] vs official 10/1)
* **FairVGNN / bail** K=10 alpha=1 c_epochs=10 c_lr=0.01 c_wd=0 clip_e=1 d_epochs=5 dropout=0.5 e_lr=0.001 e_wd=0 g_epochs=10 hidden=16 prop=scatter ratio=1 top_k=10
    * note: official ablation retunes c_lr:0.01->0.001, epochs:300->200, g_epochs:10->5
    * note: param.json disagrees (top_k/alpha [5, 1] vs official 10/1)
* **FairVGNN / credit** K=10 alpha=1 c_epochs=5 c_lr=0.01 c_wd=0 clip_e=1 d_epochs=5 dropout=0.5 e_lr=0.01 e_wd=0 g_epochs=10 hidden=16 prop=scatter ratio=0 top_k=10
    * note: official ablation retunes g_epochs:10->5, ratio:0->1
    * note: param.json disagrees (top_k/alpha [5, 0.5] vs official 10/1)
* **FairGB / german** alpha=4 c_lr=0.01 e_lr=0.01 eta=0.7 hidden=16
* **FairGB / bail** alpha=2 c_lr=0.01 e_lr=0.01 hidden=16
* **FairGB / credit** alpha=1 c_lr=0.01 c_wd=0.0001 e_lr=0.01 e_wd=0.0001 hidden=16


## Local code fidelity against the official source

* FairGNN: 51.0% of official non-trivial lines present in the local wrapper -> `local-unverified`
* FairVGNN: 83.4% of official non-trivial lines present in the local wrapper -> `local-mirror-verified`
* GNN: 38.0% of official non-trivial lines present in the local wrapper -> `local-unverified`
* NIFTY: 86.7% of official non-trivial lines present in the local wrapper -> `local-mirror-verified`
* FairGB: algorithms/FairGB/data_utils.py is byte-identical to FairGB-main (diff -q clean) -> `local-mirror-verified`


## Blocking items

* GNN / bail: no artifact states this method's setting on this dataset; horizon undetermined
* GNN / credit: no artifact states this method's setting on this dataset; horizon undetermined
* FairGNN / bail: no artifact states this method's setting on this dataset; horizon undetermined
* FairGNN / credit: no artifact states this method's setting on this dataset; horizon undetermined
* NIFTY / bail: no artifact states this method's setting on this dataset; horizon undetermined
* NIFTY / credit: no artifact states this method's setting on this dataset; horizon undetermined

6 of 15 combinations are blocking.


## Conflicts between paper/repo sources, recorded not hidden

| combination | source A | source B | chosen | note |
|---|---|---|---|---|
| FairGNN horizon | official repo, pokec_z/nba scripts: `--epochs=2000` | NIFTY README baseline on german: `--epochs 1000` | 1000 for german | the 2000 is for datasets FairGNN actually published; no artifact covers german/bail/credit from its own authors |
| FairGNN selector | official `acc_val > args.acc AND roc_val > args.roc` (two thresholds) | harness sets only `acc = temp_accs[ds] - 0.3`, never `roc` | neither is faithful | the local selector is not the published one; FairGNN's published view cannot be reproduced as-is |
| FairVGNN top_k/alpha | official GCN lines never override them: `top_k=10, alpha=1` | `utils/param.json`: `top_k=5, alpha=0.5` (german/credit), `alpha=1` (bail) | official | param.json is a local record that disagrees with the authors' scripts |
| FairVGNN learning rates | official `c_lr=0.01, e_lr=0.001` (german) | harness `c_lr=lr=0.001, e_lr=lr*0.1=0.0001` | official | the harness ran both an order of magnitude low |
| NIFTY sim_coeff | official german command: `--sim_coeff 0.6` | class default `0.5`, used by the harness | official | |
| FairGB preprocessing | upstream normalizes bail and credit, **not** german (`FairGB-main/data_utils.py:289`) | harness `get_dataset` default normalizes everything (`train_baselines.py:639`) | harness behaviour kept, recorded | changing it would alter every german number already collected |
| FairVGNN ablation | official both-off lines retune `clip_e`, `e_lr`, `ratio`, `c_lr`, `g_epochs` and the horizon | our M0 holds M1's configuration and flips only the two flags | ours | the official ablation measures a retuned model, so it cannot serve as an isolation-preserving control; its values are recorded as `official_ablation` |

## Settings that still rest on a parser default

* FairVGNN `top_k=10`, `alpha=1`, `hidden=16`, `dropout=0.5`, `prop='scatter'`,
  `K=10`, and german's horizon 200 -- the official GCN lines do not set them,
  so they come from `fairvgnn.py`'s argparse. This is the authors' own default
  applied by their own script, which is the weakest tier of the priority order
  but is still what running their script does.
* FairGB `hidden=16` -- `run.sh` does not set it.
* FairGNN `alpha`/`beta` on all three datasets -- from `utils/param.json` only.
