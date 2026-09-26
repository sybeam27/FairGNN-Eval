# R3 — SFG: controlled vs native execution (german, credit)

Scope: why `results/3b_protocol_native_horizon_selector.csv:22` (SFG/german) reports
tau_I(-DP) = +0.0734 (resolved) in the controlled cell and +0.0277 (unresolved) in the
native cell, although the horizon is unchanged (200 -> 200) and the selector is unchanged.

Read-only on `results/*.csv`, `results/per_unit_metrics.csv.gz`, `harness/results/**`.
All statements carry a file:line or the command that produced them.

---

## 0. The two rows under audit

```
$ command grep -n "SFG" results/3b_protocol_native_horizon_selector.csv
20:SFG,SAGE,bail,...,200,160,horizon 200 -> 160; ... 0.0202 ... 0.0165 ...
21:SFG,SAGE,credit,...,200,200,horizon unchanged (200); ... -0.0015 ... -0.0010 ...
22:SFG,SAGE,german,...,200,200,horizon unchanged (200); ...,0.07339332584567933,0.0198,0.1270,False,...,0.027683462149950367,-0.0367,0.0933,False,...,0.045709863695728964
```
(`results/3b_protocol_native_horizon_selector.csv:21,22`; column order per its header line 1.)
`x_mean` = controlled tau_I(-DP), `y_mean` = native tau_I(-DP):
german 0.07339333 -> 0.02768346; credit -0.00146451 -> -0.00095443.

Both means are reproduced exactly from the two per-cell stores (section g), so the two rows
are nothing but the mean over the 30 `common_bce` units of each store.

### Correction to the premise of this request

The request states SFG/german moves "from +0.073 (**resolved**) to +0.028 (**unresolved**)".
[확인됨] **That is not what the shipped table says. Both sides are `resolved = False`, and
`resolution_changed = False`.**

```
$ python -c "pd.read_csv('results/3b_protocol_native_horizon_selector.csv') ..."
dataset   x_mean     x_lo    x_hi  x_resolved   y_mean     y_lo    y_hi  y_resolved  sign_changed  resolution_changed  abs_shift
   bail  0.020155 0.000389 0.044564    False  0.016459 -0.004556 0.042942   False        False              False      0.003696
 credit -0.001465 -0.022553 0.015370    False -0.000954 -0.015265 0.011499   False        False              False      0.000510
 german  0.073393 0.019775 0.127029    False  0.027683 -0.036719 0.093273   False        False              False      0.045710
```

The controlled german interval [0.0198, 0.1270] does exclude zero, but `resolved` also requires
sign stability >= 0.75 and |mean| >= 0.01 (`harness/experiments/build_results.py:240`,
constants imported at `build_results.py:60`; stated at `build_results.py:829-830`):

```
$ python -c "pd.read_csv('results/cell_results.csv') ... method=='SFG'"
dataset protocol    tau_I_negDP_mean       lo        hi   sign_stability  resolved
 german controlled          0.073393  0.019775  0.127029        0.700000     False
 german     native          0.027683 -0.036719  0.093273        0.600000     False
 credit controlled         -0.001465 -0.022553  0.015370        0.533333     False
 credit     native         -0.000954 -0.015265  0.011499        0.666667     False
   bail controlled          0.020155  0.000389  0.044564        0.733333     False
   bail     native          0.016459 -0.004556  0.042942        0.700000     False
```

Controlled german fails on sign stability 0.700 < 0.75 (only 21 of 30 units agree in sign) —
which is itself the signature of the per-unit instability documented in (g). So the real
difference between the two rows is a mean shift of 0.0457 within an already-unresolved cell,
not a change of status.


---

## (a) Configuration dict actually passed to the method

**SFG never passes through `published()` / `native_config()`.** Both raise:

```
$ python -c "from core.published_config import published, native_config; published('SFG','german')"
published(SFG,german)    -> KeyError 'SFG'
native_config(SFG,german)-> KeyError 'SFG'
```
`harness/core/published_config.py:274` (`raise KeyError(method)` in `published`) and
`harness/core/published_config.py:378` (`raise KeyError(method)` in `native_config`).
[확인됨] `published_config.py` contributes **nothing** to SFG's method arms. It is called only
for the **B** arm, as `published("GNN", dataset)` (`harness/experiments/x30_run.py:94,193`),
identically in both arms of this audit.

SFG's configuration is parsed by the adapter from the authors' `run.sh`
(`harness/adapters/x31_sfg.py:75-118`, `config()`), with parser defaults from the named script
(`x31_sfg.py:58-72`). It takes **no protocol / native argument**: `train_arm`'s signature is
`train_arm(dataset, encoder, arm, split, seed, epochs, device="cuda")`
(`harness/adapters/x31_sfg.py:157`). The only value that could differ between the two arms is
`epochs`.

```
$ python -c "... adapters.x31_sfg.config(ds) ..."
== german  script sfg.py         native_horizon 200
 plus : {'dataset':'german','runs':1,'epochs':200,'d_epochs':5,'g_epochs':10,'c_epochs':10,
         'g_lr':0.001,'g_wd':0.0,'d_lr':0.001,'d_wd':0.0,'c_lr':0.001,'c_wd':0.0,'e_lr':0.01,
         'e_wd':0.0,'early_stopping':5,'prop':'scatter','dropout':0.5,'hidden':16,'seed':1,
         'encoder':'SAGE','K':10,'top_k':10,'clip_e':0.1,'f_mask':'yes','weight_clip':'no',
         'ratio':0.5,'alpha':1.0,'with_constraint':True,'pretrain':200,'loss_alpha':0.5,
         'eps':0.0,'rho':2.0}
 minus: identical except with_constraint False, loss_alpha 1.0, rho 100.0
== credit  script sfg_credit.py  native_horizon 200
 plus : {...,'epochs':200,'g_epochs':5,'c_lr':0.01,'e_lr':0.001,'early_stopping':0,'clip_c':1.0,
         'ratio':0.0,'with_constraint':True,'loss_alpha':0.5,'rho':2.0, ...}
 minus: identical except with_constraint False, loss_alpha 1.0, rho 1.0
== bail    script sfg.py         native_horizon 160     <- only cell where the horizon moves
```

`config()` is a pure function of `SFG-main/run.sh` + the script's argparse defaults; it is
called with `dataset` only (`x31_sfg.py:95,121,159`). **There is exactly one dict per
(dataset, arm)**, and both executions call it.

Confirmation from the stores — the `config` column, written at `x30_run.py:272`
(`config=repr(arms['plus']['config'])`), is byte-identical across all 60 matched rows of each
dataset:

```
german : config identical set: True   (1 distinct value)
  {'script':'sfg.py','with_constraint':True,'rho':2.0,'loss_alpha':0.5,'encoder':'SAGE',
   'c_lr':0.001,'e_lr':0.01,'ratio':0.5,'native_epochs':200}
credit : config identical set: True   (1 distinct value)
  {'script':'sfg_credit.py','with_constraint':True,'rho':2.0,'loss_alpha':0.5,'encoder':'SAGE',
   'c_lr':0.01,'e_lr':0.001,'ratio':0.0,'native_epochs':200}
```

**[확인됨] (a): the configuration dict is identical. Diff = {} (empty), for both german and
credit.** For bail it would also be identical except `epochs` 200 -> 160.

---

## (b) Data loader path

Two loaders are used, both in both executions:

1. **B arm / common split reference** — `x30_run.py:193-199`:
   `load(dataset, split, dev, feature_normalize=bool(published("GNN",ds)["config"]["feature_normalize"]))`,
   where `load` is `harness/experiments/pilot_tau.py:62-73` -> `utils.dataloading.load_data`
   -> `utils.data.get_dataset(dataset, feature_normalize=..., split_seed=split)`
   (`utils/dataloading.py:50-51`).
2. **SFG method arms** — `harness/adapters/x31_sfg.py:158,163-168`:
   `get_dataset(dataset, feature_normalize=(dataset!='german'), split_seed=split)` plus a second
   unnormalised call for `x_min`/`x_max` (`x31_sfg.py:165-166`, `dataset.py:442`) and
   `sens_correlation` for `corr_idx` (`x31_sfg.py:167-168`).

Neither call site takes a protocol/native argument. `norm = dataset != "german"`
(`x31_sfg.py:163`) is a function of the dataset only. Node ordering, graph construction and
splits come from the same `get_dataset(..., split_seed=split)` with the same `split`.

Cross-check inside the runner: every arm's masks are asserted equal to the common split, or the
run hard-stops (`x30_run.py:207-210`). Both executions completed 30/30 units, so both passed
that assertion on the same split.

Store-level cross-check (identical on all 60 matched rows, both datasets):
`n_test_a1`, `n_test_a0`, `dp_min_step`, `feature_normalize`, `backbone`, `provenance`,
`rng_contract`, `eval_rng_seed`, `b_epochs`, `method_epochs`, `seed`.

**[확인됨] (b): same loader, same preprocessing, same normalization policy, same graph, same
node ordering, same split.**

---

## (c) Preprocessing / wrapper steps applied in one arm and not the other

None. `x30_run.py` has exactly **three** uses of `a.native`:
- `x30_run.py:179` `protocol = a.protocol_name + ("native" if a.native else "")` — a *string
  written into the row*, not a computation;
- `x30_run.py:180-181` a guard that the adapter exposes `native_horizon`;
- `x30_run.py:182` `m_epochs = ad.native_horizon(ds, enc) if a.native else a.epochs`.

Everything downstream (loader, B training, selector `select()` at `x30_run.py:112-117`,
`STORE_SELECTORS = ("common_bce","common_auc")` at `pilot_tau.py:315`, outcome computation,
row construction) is unconditional.

For german and credit `native_horizon() == 200` and the controlled command passes
`--epochs 200` (`harness/experiments/x31_sfg_stream.sh:12-13`), so **`m_epochs` is 200 on both
sides** — confirmed in the stores: `method_epochs` ∈ {200} and `b_epochs` ∈ {200} in all four
files.

One cosmetic, non-semantic difference: the adapter runs the upstream script inside a
per-PID scratch cwd, `CACHE/cwd/<os.getpid()>` (`x31_sfg.py:241-242,249`). The two executions
are different processes, so they used different directories. That directory only receives the
upstream script's `logs/` appends; stdout is discarded (`x31_sfg.py:250`). No input is read
from it.

**[확인됨] (c): no preprocessing or wrapper step differs; the `--native` flag is inert for
german and credit because the two horizons coincide.**

---

## (d) Script and entry point

| | controlled (`protocol=x31`) | native (`protocol=x31native`) |
|---|---|---|
| launcher | `harness/experiments/x31_sfg_stream.sh:9-16` | `harness/experiments/x31_scheduler.py:39-42,106` |
| entry point | `harness/experiments/x30_run.py` | `harness/experiments/x30_run.py` (same file) |
| command | `x31_sfg_stream.sh:12-13`: `python x30_run.py --method SFG --dataset <ds> --protocol-name x31 --splits 20 21 22 23 24 25 --runs 5 --epochs 200 --out .../x31_SFG_<ds>.csv` | `x31_scheduler.py:41-42`: `python x30_run.py --method SFG --dataset <ds> --native --protocol-name x31 --out .../x31native_SFG_<ds>.csv` |
| interpreter | `x31_sfg_stream.sh:7` `/home/sypark/miniconda3/envs/dev/bin/python` | `x31_scheduler.py:19` `DEV = /home/sypark/miniconda3/envs/dev/bin/python` (same binary) |
| defaults relied on | — | `--splits` default `[20..25]`, `--runs` default 5, `--epochs` default 200, `--seed0` default 27 (`x30_run.py:160-163`) — numerically identical to the explicit controlled flags |
| output store | `harness/results/x31/x31_SFG_<ds>.csv` | `harness/results/x31/x31native_SFG_<ds>.csv` |
| process | separate `bash` lane | separate `subprocess.Popen` from the scheduler (`x31_scheduler.py:106`) |

**[확인됨] (d): same script, same interpreter, identical effective CLI apart from `--native`
(inert here, see (c)) and `--out`. They are two different OS processes.**

---

## (e) Execution time and device

```
$ cat harness/results/x31/logs/stream_sfg.log
=== 2026-09-21T17:40:18+09:00 START SFG german
=== 2026-09-21T17:57:50+09:00 START SFG bail
=== 2026-09-21T17:57:50+09:00 START SFG credit
=== 2026-09-22T01:24:27+09:00 END rc=0 SFG credit
=== 2026-09-22T02:02:39+09:00 END rc=0 SFG bail
$ command grep -n "SFG" harness/results/x31/logs/scheduler.log
6:=== 2026-09-21T18:25:39 START native_SFG_german
14:=== 2026-09-21T21:47:39 START native_SFG_bail
16:=== 2026-09-21T22:00:39 START native_SFG_credit
```

Store mtimes (`ls -l --time-style=full-iso`):

| store | last write |
|---|---|
| `x31_SFG_credit.csv` (controlled) | 2026-09-22 01:24:25 |
| `x31_SFG_german.csv` (controlled) | 2026-09-22 03:43:08 |
| `x31native_SFG_credit.csv` | 2026-09-22 03:23:11 |
| `x31native_SFG_german.csv` | 2026-09-22 04:07:28 |

So controlled german ran 17:40 -> 03:43 and native german 18:25 -> 04:07: **they overlapped for
~9 hours on the same GPU**, alongside other x31 cells (the scheduler runs up to 5 concurrently,
`x31_scheduler.py:82,88`, and the stream lanes add more).

Device: both are pinned to **CUDA device index 2**.
- controlled: `export CUDA_VISIBLE_DEVICES=2` (`x31_sfg_stream.sh:6`);
- native: `env = dict(os.environ, CUDA_VISIBLE_DEVICES="2", ...)` (`x31_scheduler.py:87`);
- enforced by the runner itself: `if a.device.startswith("cuda") and
  os.environ.get("CUDA_VISIBLE_DEVICES") != "2": raise SystemExit(...)`
  (`x30_run.py:175-176`). Neither run raised it, since both persisted 30/30 units.

[확인됨] both executions set `CUDA_VISIBLE_DEVICES=2` and both passed the runner's GPU-2 gate.
[추정] that this is physically the same GPU on the host — no per-run `nvidia-smi` UUID is
recorded in `harness/results/x31/logs/`, so the identity rests on the index alone.

Wall-clock per unit differs strongly between the two (e.g. the native log's first units take
25-27 min per arm, the last ones 3-4 min), i.e. different GPU contention — the expected
consequence of the overlap above.

---

## (f) Seed propagation

Formula, single code path, no `native` branch: `x30_run.py:189` `seed = a.seed0 + run`
(`--seed0` default 27, `x30_run.py:162`), then both method arms are seeded with
`seed_all(seed * 1000 + split)` (`x30_run.py:203`) and B with
`torch.manual_seed(seed); np.random.seed(seed)` (`x30_run.py:95`). The controlled launcher never
overrides `--seed0`, and the native launcher never sets it, so both are 27.

Recorded seeds, unit by unit (both stores, both datasets):

```
seed formula check  (seed == 27 + run_id)          : True   (all 60 rows, each file)
seed equal controlled vs native, per (split,run)   : True   (all 60 matched rows, each dataset)
```
i.e. `(20..25) x (r0..r4) -> 27,28,29,30,31` repeated per split, identically on both sides.

**[확인됨] (f): identical seed formula and identical recorded seed for every (split_id, run_id).**

`seed_all` (`harness/core/trajectory.py:141-147`) seeds `random`, `numpy`, `torch` and
`torch.cuda`, but **does not set `torch.backends.cudnn.deterministic`, `cudnn.benchmark=False`
or `torch.use_deterministic_algorithms(True)`**. [확인됨] Identical seeds therefore do **not**
guarantee identical CUDA results.

---

## (g) Stored per-unit values, unit by unit

Join key `(split_id, run_id, selector)`; 60 matched rows per dataset, of which 30 are
`common_bce` (the slot both 3b columns use — `build_results.py:200` "the frozen per-cell
bootstrap for one cell (BCE selector, as frozen)").

### SFG / german — 60 matched rows

| column | rows equal | max abs diff |
|---|---|---|
| `m1_epoch` | 5 / 60 | 176 |
| `m0_epoch` | 1 / 60 | 184 |
| `code_epoch` | 12 / 60 | 164 |
| `bc_epoch` | **60 / 60** | 0 |
| `m1_auc` | 0 / 60 | 0.1105 |
| `m1_dp` | 1 / 60 | 0.5262 |
| `m1_eo` | 3 / 60 | 0.5975 |
| `m0_auc` | 1 / 60 | 0.1355 |
| `m0_dp` | 0 / 60 | 0.4031 |
| `m0_eo` | 0 / 60 | 0.3735 |
| `b_auc` | 26 / 60 | **1.5e-4** |
| `b_dp` | **60 / 60** | 0 |
| `bc_auc` | 46 / 60 | **2.3e-4** |
| `bc_dp` | **60 / 60** | 0 |
| `m1pub_dp` | 8 / 60 | 0.1045 |
| `int_ndp` | 0 / 60 | 0.6244 |

**All 30 `common_bce` units differ in at least one of
{m1_epoch, m0_epoch, code_epoch, bc_epoch, m1_*, m0_*, b_*, bc_*} — 30 / 30.**

The **B arm is the control**: german's `b_dp`, `bc_dp` and `bc_epoch` are *identical on all 60
rows*, and `b_auc`/`bc_auc` differ by at most 1.5e-4 / 2.3e-4 (last-bit float reduction order).
B is trained by the harness itself (`x30_run.py:227`, `train_B`) from the same data and the same
seed. [확인됨] This shows the data, split and seed reaching each process were the same, and that
the harness-side arm is essentially reproducible; the divergence is confined to the SFG method
arms (the upstream adversarial `sfg.py` loop).

Per-unit table (german, `common_bce`), controlled -> native:

```
 sp run seed  m1e_c m1e_n  m0e_c m0e_n  ce_c ce_n  bce_c bce_n  int_ndp_c int_ndp_n    diff
 20  0   27      21    30     49   112     2   25     91    91     0.0274   -0.0022  -0.0296
 20  1   28      33    40     27   142    22  184    200   200    -0.0831   -0.1980  -0.1150
 20  2   29      11    11    108   169     1    1    200   200     0.0150    0.0297   0.0147
 20  3   30      27    27    144    47     2    2    200   200     0.1803   -0.0749  -0.2552
 20  4   31      51   132     95    81     7   16    200   200     0.3017   -0.0810  -0.3827
 21  0   27      76    35     49    64   199  104     91    91     0.0541    0.0806   0.0266
 21  1   28      18    55     94   184    48   23     76    76     0.0900    0.1210   0.0310
 21  2   29      16    17    169   123    16    1    200   200     0.3629    0.0514  -0.3114
 21  3   30      72    51    119   173    50   14    200   200    -0.0075    0.0542   0.0617
 21  4   31      52    27     88   183    43   16    200   200    -0.0653   -0.1789  -0.1136
 22  0   27      77    93      5   189   138   71    100   100    -0.1977    0.2996   0.4973
 22  1   28       3    18    135   124     3    3     88    88     0.1965    0.1049  -0.0916
 22  2   29     199    23    137    73   199   35    200   200     0.2077   -0.1330  -0.3407
 22  3   30      18    29    140   187   176   14    200   200     0.2597    0.3073   0.0476
 22  4   31     130    54     82    76    51   51    200   200    -0.0443   -0.0334   0.0109
 23  0   27      57    38    103    47    57   18     89    89     0.0787    0.0287  -0.0500
 23  1   28      23    23    135   151    14   18     80    80     0.0640    0.1574   0.0934
 23  2   29      34    16     95    38    34    8    200   200     0.0801    0.0051  -0.0750
 23  3   30      31     4    180    29     4    4    200   200     0.1632   -0.0096  -0.1728
 23  4   31      26    27     93    32    38   77     72    72    -0.0831   -0.0993  -0.0162
 24  0   27      55    86    179   111    53   99    102   102     0.2672    0.2027  -0.0645
 24  1   28       3    22    135   190    14   14    200   200     0.1567    0.0240  -0.1327
 24  2   29      33    16    117   148    75   36    200   200    -0.3244   -0.1955   0.1289
 24  3   30       4    12     60   174     4   12    200   200     0.1594    0.2380   0.0786
 24  4   31      18     9    107    34    58   23    200   200     0.0538   -0.2781  -0.3319
 25  0   27     185    47    103   108   185  197     85    85     0.1779    0.1974   0.0195
 25  1   28      11    15    146    81    27   14    200   200     0.0287   -0.0107  -0.0395
 25  2   29      11    16     69    94    32    8    200   200    -0.0265    0.0380   0.0644
 25  3   30      76    35    170   133    74   35    200   200     0.1910    0.0107  -0.1803
 25  4   31      19    22     73   192   110   38    200   200    -0.0823    0.1744   0.2567
 mean                                                              0.073393  0.027683 -0.045710
```

Paired difference (native - controlled) over the 30 units:
`mean -0.0457, sd 0.1818, se 0.0332`; 20 000-resample percentile bootstrap CI
**[-0.1078, +0.0194] — contains 0**. The 3b `abs_shift` 0.04571 is reproduced exactly
(`results/3b_protocol_native_horizon_selector.csv:22`, `abs_shift=0.045709863695728964`).
The shift is ~1.4 standard errors of the run-to-run noise. [확인됨] the 0.073 -> 0.028 move is
statistically indistinguishable from a re-run of the same configuration.

### SFG / credit — 60 matched rows

| column | rows equal | max abs diff |
|---|---|---|
| `m1_epoch` | 36 / 60 | 161 |
| `m0_epoch` | 19 / 60 | 155 |
| `code_epoch` | 22 / 60 | 143 |
| `bc_epoch` | 22 / 60 | 39 |
| `m1_dp` | 2 / 60 | 0.1186 |
| `m0_dp` | 0 / 60 | 0.2413 |
| `b_auc` | 0 / 60 | 0.0277 |
| `b_dp` | 0 / 60 | 0.0770 |
| `int_ndp` | 0 / 60 | 0.1890 |

30 / 30 `common_bce` units differ in at least one stored value.
Paired difference: `mean +0.00051, sd 0.0230, se 0.0042`, bootstrap CI **[-0.0070, +0.0091]**,
contains 0; matches `abs_shift=0.0005100786473307926`
(`results/3b_protocol_native_horizon_selector.csv:21`).
Note that on credit even the B arm is nondeterministic (`b_dp` 0/60 equal, max 0.077) — bigger
graph, non-deterministic scatter/atomic reductions — whereas on german B is stable. That is a
property of the dataset, not of the arm.

`results/per_unit_metrics.csv.gz` carries SFG/german 180 `controlled` + 180 `native` rows,
consistent with 30 units x (M+I, M-I, B) x 2 selectors per protocol.

---

## Mechanism

[확인됨] The selection epochs, not the trajectories' endpoints, are what moves. On german the
`common_bce` epoch of the M+I arm changes in 27 / 30 units and of the M-I arm in 29 / 30, with
swings up to 184 epochs. SFG's validation-BCE curve over a 200-epoch adversarial (generator /
discriminator / classifier) schedule is flat and multi-modal, so an arbitrarily small numerical
perturbation relocates the arg-min by a hundred epochs and moves the reported DP by tenths.

[확인됨] The perturbation is available: `seed_all` (`harness/core/trajectory.py:141-147`) never
enables deterministic cuDNN/CUDA kernels, and the two processes ran concurrently on the same
GPU, so non-deterministic reduction orders differ between them. The german B arm's
`b_auc` difference of 1.5e-4 at identical `b_dp`/`bc_epoch` is a direct measurement of that
floating-point non-determinism in this harness.

## Secondary finding (bookkeeping, not causal)

`results/3b_protocol_native_horizon_selector.csv` line 22's `factors_changed` reads
`"...; selector: sigma_c -> the method's own published rule"`, while the current builder emits
`"selector unchanged (sigma_c^BCE, as in the controlled arm); ..."`
(`harness/experiments/build_results.py:1051-1052`, applied at `1062`). The shipped CSV text is
therefore **stale relative to the code**. The numbers are not affected: both columns are built
from the `common_bce` slot (`build_results.py:200`), which the store comparison above confirms
(both files carry `common_bce` and `common_auc`, and both 3b means reproduce from `common_bce`).

---

## Verdict

**[확인됨] Nothing differs between the controlled and the native execution of SFG on german
except the execution itself.** Specifically:

| factor | controlled | native | differs? |
|---|---|---|---|
| method config dict | see (a) | identical (`config` column byte-equal on 60/60 rows) | no |
| horizon `method_epochs` | 200 | 200 | no |
| `b_epochs` | 200 | 200 | no |
| selector slot used by 3b | `common_bce` | `common_bce` | no |
| data loader / normalization / graph / node order / split | `get_dataset(ds, norm=ds!='german', split_seed=split)` | same | no |
| preprocessing / wrapper | none extra | none extra | no |
| script & interpreter | `x30_run.py`, `envs/dev/bin/python` | same | no |
| CLI | explicit `--splits/--runs/--epochs 200` | the same values as argparse defaults, plus `--native` | only `--native`, which is inert at H=200 |
| seed formula & recorded seeds | `27+run`, `seed_all(seed*1000+split)` | identical, equal per unit on 60/60 | no |
| GPU | `CUDA_VISIBLE_DEVICES=2` | `CUDA_VISIBLE_DEVICES=2` | no |
| OS process / wall clock | 17:40-03:43, own bash lane | 18:25-04:07, scheduler subprocess | **yes — this is the only difference** |
| protocol string stored in the row | `x31` | `x31native` | label only |
| output file | `x31_SFG_german.csv` | `x31native_SFG_german.csv` | label only |

**These two rows are a pure re-run of the same configuration.** The evidence:
the `config` column is identical on every matched row; `method_epochs`/`b_epochs`/`seed`/
`n_test_a*`/`dp_min_step`/`feature_normalize` are identical on every matched row; the only
`--native`-conditional quantity in `x30_run.py` (`m_epochs`, line 182) evaluates to 200 on both
sides because `x31_sfg.native_horizon('german') == 200` and the controlled command passes
`--epochs 200`; and the german B arm — trained by the harness, not the adapter — returns
identical `b_dp`, `bc_dp` and `bc_epoch` on all 60 rows with `b_auc` agreeing to 1.5e-4.

The +0.073 -> +0.028 move is run-to-run variance of the SFG method arms under non-deterministic
CUDA: the paired per-unit difference has mean -0.0457 with bootstrap CI [-0.1078, +0.0194],
comfortably containing zero. The same holds for credit (-0.00146 -> -0.00095, CI
[-0.0070, +0.0091]).

Consequence for the paper: the 3b SFG/german row does **not** measure a horizon effect. There is
no horizon effect to measure (200 -> 200). It measures the instability of SFG's
validation-BCE checkpoint selection; and there is in fact no "resolved -> unresolved" flip at all: both cells are
unresolved in the shipped table (`resolution_changed=False`).

**[확인 불가]**: whether CUDA index 2 was physically the same device in both processes (no
`nvidia-smi` UUID recorded); the exact library/driver state of each process (not logged).
