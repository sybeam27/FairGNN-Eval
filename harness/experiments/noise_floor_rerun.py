"""Re-execute five frozen cells, unchanged, to measure the noise floor of tau_{-I->+I}.

Nothing about the cell is altered: the same runner, the same method and encoder, the same
6 splits x 5 runs, the same horizon, the same protocol name, the same seeds. The only thing
that differs between a realization and the frozen row is that the process is a new one, so
whatever separates them is CUDA nondeterminism and nothing else.

Two extra realizations per cell (rep1, rep2). With the frozen bundle that gives three
realizations and therefore three pairings per cell:

    frozen vs rep1,  frozen vs rep2,  rep1 vs rep2

Each pairing is a paired per-unit difference of tau_I, read with the project's own
hierarchical bootstrap in `noise_floor_delta.py`. This script only produces the runs.

    CUDA_VISIBLE_DEVICES=2 python harness/experiments/noise_floor_rerun.py --rep 1
    CUDA_VISIBLE_DEVICES=2 python harness/experiments/noise_floor_rerun.py --rep 1 --cells SFG_german

Writes results_v2/noise_floor/rep<k>/<tag>.csv. Nothing under results/ or harness/results/
is read for writing or modified.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(ROOT, "harness", "experiments")
DEV = "/home/sypark/miniconda3/envs/dev/bin/python"
DGL = "/home/sypark/x27_dgl_cuda/bin/python"

SPLITS = ["20", "21", "22", "23", "24", "25"]

# tag -> (interpreter, argv template). Each line is the frozen cell's own command with only
# --out redirected; the provenance of each is recorded beside it.
CELLS = {
    # x31_sfg_stream.sh:13
    "SFG_german": (DEV, [f"{EXP}/x30_run.py", "--method", "SFG", "--dataset", "german",
                         "--protocol-name", "x31", "--splits", *SPLITS, "--runs", "5",
                         "--epochs", "200"]),
    # x30_streams_v2.sh:26 (stream S1, FairSIN GCN credit)
    "FairSIN-GCN_credit": (DEV, [f"{EXP}/x30_run.py", "--method", "FairSIN", "--encoder", "GCN",
                                 "--dataset", "credit", "--splits", *SPLITS, "--runs", "5",
                                 "--epochs", "200"]),
    # the core armA sweep: pilot_tau at its own registry
    "NIFTY_german": (DEV, [f"{EXP}/pilot_tau.py", "--dataset", "german", "--methods", "NIFTY",
                           "--splits", *SPLITS, "--runs", "5", "--epochs", "200",
                           "--protocol", "armA"]),
    "FairVGNN_german": (DEV, [f"{EXP}/pilot_tau.py", "--dataset", "german", "--methods", "FairVGNN",
                              "--splits", *SPLITS, "--runs", "5", "--epochs", "200",
                              "--protocol", "armA"]),
    "FairGB_bail": (DEV, [f"{EXP}/pilot_tau.py", "--dataset", "bail", "--methods", "FairGB",
                          "--splits", *SPLITS, "--runs", "5", "--epochs", "200",
                          "--protocol", "armA"]),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, type=int, choices=[1, 2])
    ap.add_argument("--cells", nargs="*", default=sorted(CELLS))
    ap.add_argument("--out", default=os.path.join(ROOT, "results_v2", "noise_floor"))
    a = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise SystemExit("this rerun runs on GPU 2 only: set CUDA_VISIBLE_DEVICES=2")

    out_dir = os.path.join(a.out, f"rep{a.rep}")
    log_dir = os.path.join(a.out, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    for tag in a.cells:
        if tag not in CELLS:
            raise SystemExit(f"unknown cell {tag}; known: {sorted(CELLS)}")
        py, argv = CELLS[tag]
        csv = os.path.join(out_dir, f"{tag}.csv")
        log = os.path.join(log_dir, f"rep{a.rep}_{tag}.log")
        cmd = [py, *argv, "--out", csv]
        t0 = time.time()
        print(f"=== rep{a.rep} {tag}: start", flush=True)
        with open(log, "w") as fh:
            fh.write(" ".join(cmd) + "\n\n")
            fh.flush()
            rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        n = 0
        if os.path.exists(csv):
            import pandas as pd
            n = pd.read_csv(csv).groupby(["split_id", "run_id"]).ngroups
        print(f"=== rep{a.rep} {tag}: rc={rc} {n}/30 units {dt:.0f}s -> {csv}", flush=True)
        if rc != 0 or n != 30:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
