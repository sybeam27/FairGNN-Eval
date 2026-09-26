"""X31 cell scheduler (GPU 2 only): starts each queued cell once its smoke block
shows a pass, skips it if the smoke failed, waits while the smoke is still
running. At most --max cells of its own run at once. Every cell writes its own
CellStore file, so no two processes ever share an output.

    nohup python harness/experiments/x31_scheduler.py > harness/results/x31/logs/scheduler.out 2>&1 &
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "harness", "results", "x31")
SM, LOGS = os.path.join(R, "smoke"), os.path.join(R, "logs")
DEV = "/home/sypark/miniconda3/envs/dev/bin/python"
DGL = "/home/sypark/x27_dgl_cuda/bin/python"
X30 = "harness/experiments/x30_run.py"
VG = "harness/experiments/x31_fairvgnn_config_run.py"
FG = "harness/experiments/x31_fairgnn_config_run.py"


def q():
    items = []
    for d in ("pokec_z", "pokec_n"):
        items.append((f"FairGNN-upstreamGAT_{d}", "fairgnn_gat_smoke.log", f"== {d}", "SMOKE PASS",
                      [DGL, FG, "--model", "GAT", "--dataset", d,
                       "--out", f"{R}/x31_FairGNN-upstreamGAT_{d}.csv"]))
    for d, c in (("bail", "GCNspmm"), ("bail", "SAGE"), ("credit", "GIN"), ("credit", "SAGE")):
        items.append((f"FairVGNN-{c}_{d}", "fairvgnn_config_smoke.log", f"== {d} {c}", "SMOKE PASS",
                      [DEV, VG, "--dataset", d, "--config", c,
                       "--out", f"{R}/x31_FairVGNN-{c}_{d}.csv"]))
    items.append(("BeMap-GAT_pokec_z", "bemap_gat_smoke.log", "== pokec_z", "ALL GATES PASS",
                  [DGL, X30, "--method", "BeMap", "--encoder", "GAT", "--dataset", "pokec_z",
                   "--protocol-name", "x31", "--out", f"{R}/x31_BeMap-GAT_pokec_z.csv"]))
    for d in ("german", "bail", "credit"):
        items.append((f"native_SFG_{d}", "sfg_native_smoke.log", f"== {d}", "ALL GATES PASS",
                      [DEV, X30, "--method", "SFG", "--dataset", d, "--native",
                       "--protocol-name", "x31", "--out", f"{R}/x31native_SFG_{d}.csv"]))
    for d, c in (("german", "GCNspmm"), ("german", "GIN"), ("german", "SAGE"), ("bail", "GCNspmm"),
                 ("bail", "SAGE"), ("credit", "GIN"), ("credit", "SAGE")):
        items.append((f"native_FairVGNN-{c}_{d}", "fairvgnn_config_native_smoke.log",
                      f"== {d} {c}", "SMOKE PASS",
                      [DEV, VG, "--dataset", d, "--config", c, "--native",
                       "--out", f"{R}/x31native_FairVGNN-{c}_{d}.csv"]))
    return items


def smoke_state(log, header, pattern):
    p = os.path.join(SM, log)
    if not os.path.exists(p):
        return "pending"
    block, on = [], False
    for ln in open(p, errors="replace"):
        ln = ln.rstrip("\n")
        if ln == header:
            on, block = True, []
            continue
        if on and ln.startswith("== "):
            break
        if on:
            block.append(ln)
    if not on:
        return "pending"
    if any(pattern in ln for ln in block):
        return "pass"
    if any(ln.startswith("rc=") for ln in block):
        return "fail"
    return "pending"


def say(msg):
    with open(os.path.join(LOGS, "scheduler.log"), "a") as fh:
        fh.write(f"=== {dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--only", default="",
                    help="queue only cells whose name starts with this prefix (a second "
                         "scheduler must never share a cell with a running one)")
    a = ap.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="2", PYTHONDONTWRITEBYTECODE="1")
    todo, running = [i for i in q() if i[0].startswith(a.only)], {}
    say(f"START {len(todo)} cells queued, max {a.max} concurrent")
    while todo or running:
        for name, (proc, fh) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                say(f"END {name} rc={proc.returncode}")
                del running[name]
        for item in list(todo):
            if len(running) >= a.max:
                break
            name, log, header, pattern, cmd = item
            st = smoke_state(log, header, pattern)
            if st == "fail":
                say(f"SKIP {name}: smoke failed")
                todo.remove(item)
            elif st == "pass":
                fh = open(os.path.join(LOGS, f"{name}.log"), "a")
                running[name] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh,
                                                  stderr=subprocess.STDOUT), fh)
                say(f"START {name}")
                todo.remove(item)
        time.sleep(60)
    say("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
