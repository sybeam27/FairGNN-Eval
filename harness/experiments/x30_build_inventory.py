"""Write harness/METHOD_EXTENSION_INVENTORY.csv and
harness/METHOD_EXTENSION_COMPATIBILITY_MATRIX.csv for X30 from one table.

Rebuilt after the X30 recovery pass (the earlier draft listed a method excluded
by user decision and carried pre-recovery verdicts). Written with csv.writer and
read back with csv.reader: every row must have the header's field count, every
(method, dataset) pair must appear exactly once, and labels must come from the
fixed label set. Exits non-zero otherwise, so a malformed file is never left
behind silently.

    python harness/experiments/x30_build_inventory.py
"""
from __future__ import annotations

import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INV = os.path.join(ROOT, "harness", "METHOD_EXTENSION_INVENTORY.csv")
MAT = os.path.join(ROOT, "harness", "METHOD_EXTENSION_COMPATIBILITY_MATRIX.csv")
DATASETS = ("german", "bail", "credit", "pokec_z", "pokec_z_g", "pokec_n", "pokec_n_g",
            "nba", "income")
LABELS = {"A. VALID-CONTROLLED", "A. VALID-NATIVE", "B. NATIVE-INVALID",
          "C. STRUCTURALLY-INVALID", "D. UNSUPPORTED", "E. ENGINEERING/ASSET-BLOCKED",
          "F. COVERED-ELSEWHERE", "G. CORE (frozen)", "-"}
NBA = "nba common split: validation contained in test (val∩test = all 213 validation nodes)"

INV_HEADER = ["method", "category", "source", "environment", "intervention_I",
              "M_plus_I", "M_minus_I", "configuration_provenance", "datasets_admitted",
              "native", "recovery_attempts", "admission", "evidence"]

INVENTORY = [
    ["FairSIN", "reconsidered (X2 exclusion not inherited)",
     "FairSIN-main/in-train.py, train_mlp.py, experiment.sh (official)",
     "dev (torch 2.6.0, PyG 2.8.0)",
     "neutralisation x + delta*mlp(x) (+ discriminator step when d=yes)",
     "first ablation row (experiment.sh:50-116) with delta>0 and d=yes, else delta>0",
     "same row with delta=0, d=no",
     "official-repo (authors' ablation block)",
     "german; bail; credit; pokec_z; pokec_n (GCN primary; GIN, SAGE variants)",
     "VALID-NATIVE (validation tradeoff selector, native horizon)",
     "run() executed unmodified; seeded train_test_split; evaluate wrapper; hadj cache namespace; utils import isolation; CUDA bind before script's CUDA_VISIBLE_DEVICES=6; memory_profiler stub",
     "ADMITTED controlled + native",
     "smoke/fairsin_smoke.log; X2 rationale refuted at utils.py:29-41"],
    ["EDITS", "new", "algorithms/EDITS.py (repository wrapper; upstream not vendored)",
     "x30_edits_env (dev + deeprobust, aif360)",
     "debiasing stage EDITS.fit (attribute + structural debiasing)",
     "fit() then predict() at param.json dropout / threshold_proportion",
     "stage bypassed: original features and adjacency into the same predict()",
     "local-unverified (utils/param.json); routing repaired",
     "german; bail; credit",
     "D (no official configuration)",
     "venv with deeprobust/aif360; venv-local torch 2.14 removed (CUDA 13 mismatch); param routing repaired; seed injection before predict (optimize reseeds to 10)",
     "ADMITTED controlled",
     "smoke logs; x30_env_edits_fairedit.log; train_baselines.py:567-587, 800-830"],
    ["FairEdit", "new", "algorithms/FairEdit.py (repository wrapper)", "x30_edits_env",
     "fair_graph_edit while epoch < edit_num (FairEdit.py:780)",
     "edit_num=10 (fit default)", "edit_num=0",
     "local-unverified (utils/param.json)", "german; bail; credit",
     "D (no official configuration)",
     "venv with aif360; private cwd per process; trainer.train(epochs=H) instead of predict()'s hard-coded 100",
     "ADMITTED controlled", "smoke logs; train_baselines.py:550-565"],
    ["BeMap", "new", "BeMap-main (official)", "x27_dgl_cuda (torch 2.2.2, dgl 1.1.3)",
     "balance-aware per-epoch subgraph sampling",
     "train_bemap.train(epoch, ...)", "train_bemap.train(-1, ...) (full-graph branch)",
     "official-repo (README command, parser defaults)", "bail; credit; pokec_z",
     "B. NATIVE-INVALID (selects on test, train_bemap.py:136-142)",
     "DGL env; utils import isolation; import-time seeding isolated",
     "ADMITTED controlled", "smoke/bemap_smoke.log (bail in session log)"],
    ["GEAR", "new", "GEAR-main/src (official) + released aug assets", "x30_edits_env",
     "counterfactual-consistency objective sim_coeff",
     "sim_coeff=0.6", "sim_coeff=0",
     "official-repo (parser defaults); released assets verified as sensitive-attribute flips",
     "bail",
     "E (published counterfactual assets not reproducible)",
     "PyG1 pickle I/O adapter; torch.load weights_only=False scoped to GEAR cache; utils import isolation (utils_mp kept registered for multiprocessing); private cwd/models_save; __main__ globals n/device/sens provided; evaluate stub (print/native checkpoint only)",
     "ADMITTED controlled (bail), asset caveat",
     "asset verification in adapters/x30_gear.py:_assets; credit asset graph overlap 3.0%"],
    ["FairGT", "new", "FairGT-main (official)", "-",
     "same-sensitive complete graph + eigenvector PE", "-", "none in authors' code",
     "official-repo README commands", "-", "C",
     "switch search in authors' code and wrapper; components_off eig/sgr is study-authored help text (b396f76), unimplemented",
     "EXCLUDED (C)", "train_fairgt.py:84-103"],
    ["BIND", "new", "BIND-main/implementations (official)", "x27_dgl_cuda",
     "influence-guided deletion of training nodes (helpfulness_collection=1 ranking)",
     "k = round(p*|train|) deleted, p=0.01 primary (BIND-1pct), p=0.10 variant (BIND-10pct)",
     "k = 0 (same Stage-A model, same retraining)",
     "official-repo (parser defaults; README scale 25)",
     "bail; income",
     "B. NATIVE-INVALID (published fairness cost reads test labels)",
     "device shims S5 (sens to device); Windows dll no-op; ipdb stub; L1 fairness cost on validation nodes (leakage removal, estimator unchanged); W1-W3 common-data loaders/get_adj/del_adj; Stage-A cache namespace with hashes",
     "ADMITTED controlled (bail, income)",
     "bind_recovery/RECOVERY_LOG.md; approximator.py:131-133,149; README.md:55; smoke/bind_smoke.log"],
    ["FMP", "existing (X27 component case study)", "FMP-main (official)", "x27_dgl_cuda",
     "fair message passing (lambda1, lambda2)", "grid only (run_fgnn.sh)", "lambda=0",
     "official grid, no single configuration", "-", "-",
     "bundled adv/reg/Fmix baselines: hyper_reg default 0, no published value",
     "NO NEW CELL (X27 stands)", "X27_RESULTS.md; run_fgnn.sh"],
    ["FairWalk", "new", "algorithms/FairWalk.py", "-", "fair random-walk sampling", "-", "-",
     "-", "-", "C", "-", "EXCLUDED (C)", "random sensitive groups in wrapper; dead run()"],
    ["CrossWalk", "new", "algorithms/CrossWalk.py", "-", "cross-group walk reweighting",
     "-", "-", "-", "-", "C", "-", "EXCLUDED (C)", "random sensitive groups in wrapper; dead run()"],
    ["GNN_cf", "variant", "algorithms/GNN_cf.py", "-", "counterfactual evaluation variant",
     "-", "-", "-", "-", "D", "-", "NOT A CANDIDATE",
     "variant of the core NIFTY mechanism"],
    ["NIFTY_cf", "variant", "algorithms/NIFTY_cf.py", "-", "counterfactual evaluation variant",
     "-", "-", "-", "-", "D", "-", "NOT A CANDIDATE",
     "variant of the core NIFTY mechanism"],
    ["FairGNN", "core (X22, frozen)", "algorithms/FairGNN.py", "dev", "alpha, beta", "-", "-",
     "-", "-", "-", "-", "CORE (not re-run)", "X22"],
    ["NIFTY", "core (X22, frozen)", "algorithms/NIFTY.py", "dev", "sim_coeff", "-", "-",
     "-", "-", "-", "-", "CORE (not re-run)", "X22"],
    ["FairVGNN", "core (X22, frozen)", "algorithms/FairVGNN.py", "dev", "f_mask, weight_clip",
     "-", "-", "-", "-", "-", "-", "CORE (not re-run)", "X22"],
    ["FairGB", "core (X22, frozen)", "algorithms/FairGB_alg.py", "dev", "CAL, CNM", "-", "-",
     "-", "-", "-", "-", "CORE (not re-run)", "X22"],
    ["GNN", "baseline B", "algorithms/GNN.py", "every X30 env", "-", "-", "-",
     "published('GNN', dataset)", "-", "-", "-", "BASELINE", "pilot_tau.main"],
]


def mat(method, controlled, native, reason):
    return [[method, ds, controlled.get(ds, ("D. UNSUPPORTED",))[0],
             native.get(ds, ("D. UNSUPPORTED",))[0],
             controlled.get(ds, ("", reason.get(ds, "not supported by the method's repository")))[-1]
             if ds in controlled else reason.get(ds, "not supported by the method's repository")]
            for ds in DATASETS]


def matrix_rows(bind_rows):
    A, VN = "A. VALID-CONTROLLED", "A. VALID-NATIVE"
    rows = []
    fs = {d: (A, "ablation-block configuration; official run(); gates pass for GCN, GIN, SAGE")
          for d in ("german", "bail", "credit", "pokec_z", "pokec_n")}
    rows += mat("FairSIN", fs, {d: (VN,) for d in fs},
                {"pokec_z_g": "sensitive index hard-coded to region", "pokec_n_g":
                 "sensitive index hard-coded to region", "nba": NBA, "income": "no loader"})
    ed = {d: (A, "stage bypass; param.json routing repaired; gates pass") for d in ("german", "bail", "credit")}
    rows += mat("EDITS", ed, {}, {"nba": NBA})
    fe = {d: (A, "edit_num 10 vs 0; gates pass") for d in ("german", "bail", "credit")}
    rows += mat("FairEdit", fe, {}, {"nba": NBA})
    bm = {d: (A, "subgraph vs full-graph training step; gates pass") for d in ("bail", "credit", "pokec_z")}
    bm["nba"] = ("C. STRUCTURALLY-INVALID", NBA)
    rows += mat("BeMap", bm, {d: ("B. NATIVE-INVALID",) for d in ("bail", "credit", "pokec_z")}, {})
    ge = {"bail": (A, "released assets verified as sensitive-attribute flips of common data (caveat)"),
          "credit": ("E. ENGINEERING/ASSET-BLOCKED", "released assets on a different graph (3.0% edge overlap)"),
          "german": ("E. ENGINEERING/ASSET-BLOCKED", "no released assets; generator checkpoints absent")}
    rows += mat("GEAR", ge, {d: ("E. ENGINEERING/ASSET-BLOCKED",) for d in ("bail",)}, {})
    gt = {d: ("C. STRUCTURALLY-INVALID", "no off-state in authors' code")
          for d in ("german", "bail", "credit", "nba", "income")}
    rows += mat("FairGT", gt, {d: ("C. STRUCTURALLY-INVALID",) for d in gt}, {})
    rows += bind_rows
    fm = {d: ("F. COVERED-ELSEWHERE", "X27 component case study; grid, no single configuration")
          for d in ("pokec_z", "pokec_n")}
    fm["nba"] = ("C. STRUCTURALLY-INVALID", NBA)
    rows += mat("FMP", fm, {d: ("F. COVERED-ELSEWHERE",) for d in ("pokec_z", "pokec_n")}, {})
    for m in ("FairWalk", "CrossWalk"):
        rows += mat(m, {d: ("C. STRUCTURALLY-INVALID", "random sensitive groups; dead run()") for d in DATASETS},
                    {d: ("C. STRUCTURALLY-INVALID",) for d in DATASETS}, {})
    for m in ("GNN_cf", "NIFTY_cf"):
        rows += mat(m, {}, {}, {d: "variant of the core NIFTY mechanism, not a separate intervention"
                                for d in DATASETS})
    return rows


def write_and_check(path, header, rows, key_cols=None):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    with open(path, newline="") as fh:
        rd = list(csv.reader(fh))
    bad = [i for i, r in enumerate(rd) if len(r) != len(header)]
    if rd[0] != header or bad:
        raise SystemExit(f"[inventory] {path}: malformed rows {bad}")
    if key_cols:
        keys = [tuple(r[c] for c in key_cols) for r in rd[1:]]
        if len(keys) != len(set(keys)):
            raise SystemExit(f"[inventory] {path}: duplicate keys")
    return len(rd) - 1


def main(bind_rows=None) -> int:
    bi = {d: ("A. VALID-CONTROLLED", "influence deletion k=round(p|train|) vs 0; README scale 25; fairness cost on validation nodes; gates pass")
          for d in ("bail", "income")}
    bi["pokec_z"] = ("D. UNSUPPORTED", "BIND pokec1 zip is not the repository-common pokec loader; not ported")
    bi["pokec_n"] = ("D. UNSUPPORTED", "BIND pokec2 zip is not the repository-common pokec loader; not ported")
    bind_rows = mat("BIND", bi, {d: ("B. NATIVE-INVALID",) for d in ("bail", "income")}, {"nba": NBA})
    rows = matrix_rows(bind_rows)
    for r in rows:
        if r[2] not in LABELS or r[3] not in LABELS:
            raise SystemExit(f"[inventory] unknown label in {r}")
    n_inv = write_and_check(INV, INV_HEADER, INVENTORY, key_cols=(0,))
    n_mat = write_and_check(MAT, ["method", "dataset", "controlled_label", "native_label", "reason"],
                            rows, key_cols=(0, 1))
    methods = {r[0] for r in rows}
    if n_mat != len(methods) * len(DATASETS):
        raise SystemExit(f"[inventory] matrix has {n_mat} rows, expected "
                         f"{len(methods)} x {len(DATASETS)}")
    print(f"[inventory] {INV}: {n_inv} rows; {MAT}: {n_mat} rows, {len(methods)} methods")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
