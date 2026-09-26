"""
E7 -- do the published methods survive the protocol?

Everything in harness/ so far compares *signals* inside our own pipeline. A
reproducibility claim has to test the literature's methods, not our
reimplementations of their criteria, so this drives `utils/train_baselines.py`
over the same (split, init) cells that E4 used.

That comparison is only meaningful because the two loaders agree exactly.
`harness/core/datasets.py` and `utils/data.py::get_dataset` were checked on all
nine settings at split seeds 20, 21 and 22: identical train/val/test indices,
feature matrix, labels and sensitive attribute. So a baseline run and a FairGate
run at the same (split_seed, seed) are on the same problem and belong in one
table.

Budget, measured on Pokec-z (the largest graph), one run each:

    GNN 11s / 230MB      FairGNN 13s / 921MB     NIFTY 14s / 756MB
    FairGT 51s / 1.9GB   (870s on the first run per setting, which builds an
                          18GB same-sens adjacency cache keyed by dataset name)
    FairGB 74s / 35GB    (nothing else can share the GPU while it runs)

`--runs R` inside `train_baselines.py` uses seeds `seed .. seed+R-1`, which is
exactly the init-seed block we want, so one invocation covers a whole
(setting, split) cell and the process start-up cost is paid 54 times per method
rather than 270.

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python harness/experiments/e7_baseline_audit.py \
        --models GNN FairGNN NIFTY FairGT FairGB
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
RAW = os.path.join(RESULTS, "e7_raw")

# ordered cheapest-first so a failure surfaces before the expensive methods run
MODELS = ["GNN", "FairGNN", "NIFTY", "FairGT", "FairGB", "FairGate", "BIND"]

# FairGate's published configuration is not retyped from the paper. It is read
# out of the record its own submitted run left behind, `PROVENANCE` below, which
# stores all 43 parameters per setting alongside the numbers Table 1 reports.
# Two things this settles that the paper alone could not.
#
# `run/run_minimal_variant.py` labels a different configuration ("lambda 0.30,
# q 0.70") as "the final settings that produced Table 1"; that comment is wrong
# -- reproducing German from it gives Acc 0.6328 against the reported 0.6896,
# while the recorded configuration gives 0.6856 +- 0.0173 and selects the same
# adaptive policy ('boundary'). And of the nine settings, eight match the
# paper's Table 10 exactly while NBA does not: the run used
# lambda 0.15 / q 0.80 / p 0.2, the table says 0.40 / 0.50 / 0.30.
#
# Also recorded there, and contradicting the paper's own protocol section:
# every run, FairGate and baselines alike, used weight_decay = 0.0, not the
# 1e-5 the text states. The two are consistent with each other, so the
# comparison is fair; the reported setting is simply not the one that ran.
PROVENANCE = os.path.join(ROOT, "outputs", "ours", "exp_fairgate_fiw_v1.csv")
_PER_SETTING = ["lambda_fair", "sbrs_quantile", "struct_drop", "warm_up"]
_SHARED = ["hidden_dim", "dropout", "lr", "weight_decay", "epochs", "patience",
           "fips_lam", "mmd_alpha", "dp_eo_ratio", "uncertainty_type",
           "ramp_epochs", "recal_interval", "alpha_beta_mode",
           "edge_intervention", "gating_mode_override", "fiw_weight_mode",
           "adaptive_probe_epochs", "adaptive_eta", "adaptive_auc_tol",
           "boundary_sat_thr", "ablation_mode"]
# Backbone-attribution variants (E11). Each is an existing method run with one
# thing changed, so the audit's own protocol, splits and seeds carry over and the
# comparison is paired cell by cell.
#
#   GNN_sage      the plain baseline on GraphSAGE instead of GCN. FairGB runs on
#                 SAGE by its own default (FairGB_alg.fit(encoder='SAGE')) and
#                 the audit compared it against a GCN baseline, so its +0.0760
#                 AUC mixes the encoder with the method.
#   NIFTY_off     sim_coeff = 0: NIFTY's encoder, protocol and selection with no
#                 fairness objective.
#   FairGNN_off   alpha = beta = 0: same, for FairGNN's adversary and covariance
#                 terms.
#
# FairGB and FairGT have no such coefficient -- FairGB's fairness is a sampling
# augmentation and FairGT's is an adjacency transform -- so they get no _off arm
# and are compared against the matched plain encoder instead. That limit is
# reported, not worked around.
VARIANTS = {"GNN_sage":    ("GNN",     ["--gnn_encoder", "sage"]),
            "NIFTY_off":   ("NIFTY",   ["--fairness_off"]),
            "FairGNN_off": ("FairGNN", ["--fairness_off"])}

SETTINGS = ["german", "nba", "recidivism", "income", "credit",
            "pokec_z", "pokec_z_g", "pokec_n", "pokec_n_g"]
SPLITS = [20, 21, 22, 23, 24, 25]
INIT0, N_INIT = 27, 5


def _fairgate_cmd(setting: str, split: int, python: str, out: str) -> list[str]:
    cfg = pd.read_csv(PROVENANCE).set_index("dataset").loc[setting]
    cmd = [python, "-u", "-m", "utils.train_fairgate",
           "--dataset", setting, "--backbone", "GCN", "--device", "cuda",
           "--split_seed", str(split), "--seed", str(INIT0),
           "--runs", str(N_INIT), "--output_file", out]
    for k in _PER_SETTING + _SHARED:
        v = cfg[k]
        cmd += [f"--{k}", str(int(v) if isinstance(v, float) and v.is_integer()
                              and k not in ("lr", "weight_decay", "dropout",
                                            "lambda_fair", "struct_drop",
                                            "sbrs_quantile", "adaptive_eta",
                                            "adaptive_auc_tol", "mmd_alpha",
                                            "dp_eo_ratio", "fips_lam",
                                            "boundary_sat_thr") else v)]
    if bool(cfg["fiw_adaptive"]):          # store_true: present or absent
        cmd += ["--fiw_adaptive"]
    return cmd


def one(model: str, setting: str, split: int, python: str, timeout: int,
        arm: str = "as_submitted") -> dict | None:
    out = os.path.join(RAW if arm == "as_submitted" else RAW + "_" + arm,
                       f"{model}__{setting}__s{split}.csv")
    if os.path.exists(out):                       # resumable: 12h of runs
        try:
            # status belongs here as much as on a fresh run. Without it a
            # resumed cell carries NaN, and 342 of this audit's 366 rows are
            # resumed -- so any analyser written the obvious way,
            # df[df.status == "ok"], would silently compute the whole audit on
            # the 24 rows that happened not to be cached, with no error.
            return pd.read_csv(out).assign(model=model, setting=setting,
                                           split_seed=split, status="ok",
                                           resumed=True).to_dict("records")[0]
        except Exception:                                        # noqa: BLE001
            os.remove(out)
    if model == "FairGate":
        cmd = _fairgate_cmd(setting, split, python, out)
    elif model == "BIND":
        # BIND runs under its own published settings (1-layer GCN, 16 hidden,
        # wd 1e-4, 1000 epochs, per-dataset influence scale), not the shared
        # protocol the other rows use. It was never in the comparison being
        # audited, so there is no "as submitted" configuration for it and the
        # authors' is the only defensible one -- at the cost of its row
        # differing in backbone as well as in method.
        cmd = [python, "-u", "-m", "study.experiments._run_bind_cell",
               "--dataset", setting, "--split_seed", str(split),
               "--seed", str(INIT0), "--runs", str(N_INIT),
               "--device", "cuda", "--output_file", out]
    else:
        real, extra = VARIANTS.get(model, (model, []))
        cmd = [python, "-u", "-m", "utils.train_baselines",
               "--model", real, "--dataset", setting,
               "--split_seed", str(split), "--seed", str(INIT0),
               "--runs", str(N_INIT), "--device", "cuda", "--output_file", out]
        cmd += extra
        if arm == "normunified":
            # The comparison being audited does not share preprocessing: GNN and
            # NIFTY run unnormalised on all nine settings, FairGNN normalises
            # only NBA and German, and FairGB/FairGT/FairGate take
            # get_dataset's default, which normalises everywhere. So three of
            # the seven rows differ from the rest in a step that is not the
            # fairness intervention. This arm sets every row to normalised; the
            # quantity to report is the difference between the two arms, per
            # method, not this arm on its own.
            #
            # Normalisation is not uniformly helpful -- measured directly, it is
            # +0.074 AUC on Income, -0.050 on Credit and -0.003 on Pokec-z -- so
            # a method may lose here, and that loss is part of the result.
            cmd += ["--force_feature_normalize", "1"]
    t = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        # A cell that runs out of time is a result, not a crash. BIND's cost is
        # set by its own deletion sweep -- 0.3 * max_num retrainings, which is
        # 3 on German and 326 on Income -- so "did not finish in N hours at this
        # protocol" is exactly the kind of thing the audit should record rather
        # than something to be worked around. Before this, the exception
        # propagated and killed the sweep.
        print(f"    TIMEOUT after {timeout}s")
        return {"model": model, "setting": setting, "split_seed": split,
                "status": "timeout", "timeout_sec": timeout,
                "wall_sec": round(time.time() - t, 1)}
    if p.returncode != 0 or not os.path.exists(out):
        tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
        print(f"    FAILED rc={p.returncode}: {' | '.join(tail)}")
        return {"model": model, "setting": setting, "split_seed": split,
                "status": "failed", "wall_sec": round(time.time() - t, 1)}
    r = pd.read_csv(out).to_dict("records")[0]
    r |= {"model": model, "setting": setting, "split_seed": split,
          "status": "ok", "wall_sec": round(time.time() - t, 1)}
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--settings", nargs="*", default=SETTINGS)
    ap.add_argument("--splits", nargs="*", type=int, default=SPLITS)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--arm", default="as_submitted",
                    choices=["as_submitted", "normunified"],
                    help="as_submitted keeps each method's own preprocessing; "
                         "normunified forces feature normalisation on all.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(RESULTS, "e7_baseline_audit.csv"
                                if args.arm == "as_submitted"
                                else f"e7_baseline_audit_{args.arm}.csv")
    os.makedirs(RAW if args.arm == "as_submitted"
                else RAW + "_" + args.arm, exist_ok=True)
    n = len(args.models) * len(args.settings) * len(args.splits)
    print(f"arm={args.arm}\nmodels={args.models}\nsettings={args.settings}\nsplits={args.splits}")
    print(f"invocations = {n}   ({N_INIT} init seeds each, "
          f"{n * N_INIT} runs)\n")

    rows, t0, done = [], time.time(), 0
    for model in args.models:
        for setting in args.settings:
            for split in args.splits:
                done += 1
                r = one(model, setting, split, args.python, args.timeout,
                        arm=args.arm)
                if r:
                    rows.append(r)
                    print(f"  [{done}/{n}] {model:<8} {setting:<10} split={split}  "
                          f"acc={r.get('acc_mean', float('nan')):.4f} "
                          f"dp={r.get('dp_mean', float('nan')):.4f} "
                          f"eo={r.get('eo_mean', float('nan')):.4f} "
                          f"({r.get('wall_sec', 0):.0f}s)")
                if rows:
                    pd.DataFrame(rows).to_csv(args.out, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[saved] {args.out}  ({len(df)} cells, {time.time() - t0:.0f}s)\n")
    if not df.empty and "dp_mean" in df.columns:
        bad = df[df.get("status", "ok") != "ok"]
        if len(bad):
            print(f"not completed: {len(bad)} cells")
            print(bad[["model", "setting", "split_seed", "status"]].to_string(index=False))
        piv = df[df.get("status", "ok") == "ok"].pivot_table(
            index="setting", columns="model", values="dp_mean", aggfunc="mean")
        print("mean dp over splits\n" + piv.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
