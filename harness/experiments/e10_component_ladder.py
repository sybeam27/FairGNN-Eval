"""
H6 -- does the node-wise allocation contribute anything over spending the same
fairness budget uniformly?

Registered in harness/PREREGISTRATION.md on 2026-09-12, before any arm ran.

This is the experiment the paper turns on. FairGate Pareto-dominates plain GCN
on 75.9% of cells with preprocessing unified (Finding 14), and the signal it
allocates on loses to allocating at random (w_bdry net -31, p = 0.004). Both are
measured. The method wins and its stated mechanism is falsified, and no further
signal search can resolve that -- deviation D4's stopping rule closed it.

Four arms, all the published FairGate at its recorded configuration, differing
in `fiw_weight_mode` and nothing else:

    full             allocated                      budget B
    perm             same weights, permuted         budget B
    uniform_budget   no allocation                  budget B
    uniform          no allocation                  budget 1.4-1.75 B (as shipped)

Why `uniform` is not the control
--------------------------------
It is the ablation the code ships, and it is not budget-matched. It returns
ones(N) while the allocated weights average 0.57 to 0.71, so it applies 1.4x to
1.75x more total fairness pressure at the same lambda. It varies the allocation
and the budget together. It is run anyway, as the fourth arm, because showing
what it actually measures is part of the result.

`perm` is the sharper control: the weight multiset is identical and only the
assignment to nodes is destroyed. If `full` beats `uniform_budget` but not
`perm`, then what helps is having a spread of weights, not the ranking the
signal produces.

Configuration provenance
------------------------
Per-setting hyper-parameters come from the same recorded run the audit reads,
`outputs/ours/exp_fairgate_fiw_v1.csv`, through the identical code path as
`e7_baseline_audit._fairgate_cmd` -- so the `full` arm here is the same
configuration as the audit's FairGate row, not a re-tuned one.

The adaptive policy selection (`--fiw_adaptive`) is switched **off** in every
arm. It chooses among four allocation policies on validation, which is part of
the allocation machinery under test and would otherwise differ between arms
that have an allocation and arms that do not. D2 already measured what that
selection is worth: 0.0006.

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python harness/experiments/e10_h6_allocation.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "harness", "results")
RAW = os.path.join(RESULTS, "e10_raw")
PROVENANCE = os.path.join(ROOT, "outputs", "ours", "exp_fairgate_fiw_v1.csv")

# The ladder, bottom to top. Read downward it is an ablation of a finished
# method, not an upward search: each rung removes one component from the
# published FairGate. That direction matters. Building upward and keeping what
# helps would let the nine settings choose the design, which is the failure
# every guard in PREREGISTRATION.md exists to prevent.
#
#   arm            ablation_mode   phi              what the step above it adds
#   backbone       none            --               (GCN, lambda_fair = 0)
#   out            out_only        uniform_budget   prediction-level DP/EO
#   rep_out        rep_out         uniform_budget   representation alignment
#   all_uniform    full_loss       uniform_budget   structural consistency
#   all_alloc      full_loss       continuous       the allocation            <- H6
#   all_perm       full_loss       permuted         (control for all_alloc)
#   all_uniform1   full_loss       ones             (what the shipped ablation measures)
#
# Every rung that has a phi uses uniform_budget, so rungs 1-3 spend exactly the
# fairness budget all_alloc spends. Without that, moving up the ladder would
# change the budget as well as the components and no step would be attributable.
ARMS = ["backbone", "out", "rep_out", "all_uniform", "all_alloc", "all_perm",
        "all_uniform1"]
ABLATION = {"backbone": "none", "out": "out_only", "rep_out": "rep_out",
            "all_uniform": "full_loss", "all_alloc": "full_loss",
            "all_perm": "full_loss", "all_uniform1": "full_loss"}
MODE = {"backbone": "uniform_budget", "out": "uniform_budget",
        "rep_out": "uniform_budget", "all_uniform": "uniform_budget",
        "all_alloc": "continuous_uncert", "all_perm": "matched_random_perm",
        "all_uniform1": "uniform"}
SETTINGS = ["german", "nba", "recidivism", "income", "credit",
            "pokec_z", "pokec_z_g", "pokec_n", "pokec_n_g"]
SPLITS = [20, 21, 22, 23, 24, 25]
INIT0, N_INIT = 27, 5

_PER_SETTING = ["lambda_fair", "sbrs_quantile", "struct_drop", "warm_up"]
_SHARED = ["hidden_dim", "dropout", "lr", "weight_decay", "epochs", "patience",
           "fips_lam", "mmd_alpha", "dp_eo_ratio", "uncertainty_type",
           "ramp_epochs", "recal_interval", "alpha_beta_mode",
           "edge_intervention", "gating_mode_override",
           "boundary_sat_thr"]        # ablation_mode is set by the arm


def cmd_for(arm: str, setting: str, split: int, python: str, out: str) -> list[str]:
    cfg = pd.read_csv(PROVENANCE).set_index("dataset").loc[setting]
    cmd = [python, "-u", "-m", "utils.train_fairgate",
           "--dataset", setting, "--split_seed", str(split),
           "--seed", str(INIT0), "--runs", str(N_INIT),
           "--device", "cuda", "--output_file", out,
           # the two knobs that define the arm, and the only things that
           # differ between arms
           "--fiw_weight_mode", MODE[arm],
           "--ablation_mode", ABLATION[arm]]
    for k in _PER_SETTING + _SHARED:
        if k in cfg.index and pd.notna(cfg[k]):
            v = cfg[k]
            cmd += [f"--{k}", str(int(v) if isinstance(v, float) and v.is_integer()
                                  and k not in ("lr", "weight_decay", "dropout",
                                                "lambda_fair", "sbrs_quantile",
                                                "struct_drop", "fips_lam",
                                                "mmd_alpha", "dp_eo_ratio",
                                                "boundary_sat_thr")
                                  else v)]
    return cmd


def one(arm: str, setting: str, split: int, python: str, timeout: int) -> dict | None:
    out = os.path.join(RAW, f"{arm}__{setting}__s{split}.csv")
    if os.path.exists(out):
        try:
            # status must be set here too. Without it a resumed cell carries
            # NaN, and every analyser in this study filters on status == "ok"
            # -- so a resumed run would silently drop exactly the cells that
            # completed, which looks like missing data rather than a bug.
            return pd.read_csv(out).assign(arm=arm, setting=setting,
                                           split_seed=split, status="ok",
                                           resumed=True).to_dict("records")[0]
        except Exception:                                        # noqa: BLE001
            os.remove(out)
    t = time.time()
    try:
        p = subprocess.run(cmd_for(arm, setting, split, python, out),
                           cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {timeout}s")
        return {"arm": arm, "setting": setting, "split_seed": split,
                "status": "timeout", "timeout_sec": timeout}
    if p.returncode != 0 or not os.path.exists(out):
        tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
        print("    FAILED: " + " | ".join(tail))
        return {"arm": arm, "setting": setting, "split_seed": split,
                "status": "failed"}
    r = pd.read_csv(out).to_dict("records")[0]
    r.update(arm=arm, setting=setting, split_seed=split, status="ok",
             wall_sec=round(time.time() - t, 1))
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--settings", nargs="*", default=SETTINGS)
    ap.add_argument("--splits", nargs="*", type=int, default=SPLITS)
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=os.path.join(RESULTS, "e10_ladder.csv"))
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    n = len(args.arms) * len(args.settings) * len(args.splits)
    print(f"arms={args.arms}\nsettings={args.settings}\nsplits={args.splits}")
    print(f"invocations = {n}   ({n * N_INIT} runs)\n")

    rows, t0, done = [], time.time(), 0
    for setting in args.settings:
        for split in args.splits:
            for arm in args.arms:
                done += 1
                r = one(arm, setting, split, args.python, args.timeout)
                if r:
                    rows.append(r)
                    print(f"  [{done}/{n}] {arm:<14} {setting:<10} split={split}  "
                          f"auc={r.get('roc_auc_mean', float('nan')):.4f} "
                          f"dp={r.get('dp_mean', float('nan')):.4f} "
                          f"({r.get('wall_sec', 0):.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[saved] {args.out}  ({len(df)} cells, {time.time() - t0:.0f}s)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
