"""Deterministic regeneration of the paper-facing `results/` bundle.

One command rebuilds every file the paper uses:

    python harness/experiments/build_results.py

Nothing scientific is recomputed or reinterpreted. The paired unit construction
(`analyze_armA.build`), the paired hierarchical bootstrap (`bootstrap_armA.boot`,
seed 20260914, 10,000 replicates, splits resampled first and then runs within a
drawn split, each matched (split, run) unit carried whole) and the resolution
rule (sign stability >= 0.75 AND |mean| >= 0.010 AND the 95% percentile interval
excluding zero) are imported and used exactly as frozen. Raw stores are read
only.

What this script does change is *presentation*: the bookkeeping names of the
runs (the internal stage identifiers and group labels) never reach the output.
Each cell carries only scientific facts:

    configuration_role   primary | robustness
    protocol             controlled | native
    count_in_primary_summary   true only for primary controlled cells

Every completed primary controlled method x dataset configuration forms one
final evaluation set, whatever order the runs happened in. Robustness
configurations (alternative backbones, alternative intervention budgets) and
native-protocol rows are reported but never pooled into primary counts.

Inclusion depends only on completeness (30 matched units, 60 rows) and on the
admission decisions fixed before any outcome existed -- never on the magnitude
or sign of any result.

Outputs (replaced atomically after the validation gates pass):

    results/cell_results.csv           canonical per-cell estimates
    results/1_ ... 5_*.csv             the numbered paper sections
    results/experiment_index.csv       what each section holds fixed and varies
    results/method_configurations.csv  what each cell ran
    results/coverage.csv               completed, partial and pending
    results/README.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))
sys.path.insert(0, os.path.join(ROOT, "harness", "adapters"))

from analyze_armA import build, sign_stability, SIGN_MIN, NEAR_ZERO   # noqa: E402
from analyze_x29 import CORE_CSVS, per_cell                           # noqa: E402
from bootstrap_armA import boot, SEED                                 # noqa: E402

RES = os.path.join(ROOT, "harness", "results")
UNITS_REQUIRED, ROWS_REQUIRED = 30, 60
COORDS = (("dAUC", "auc"), ("negDP", "ndp"), ("negEO", "neo"))
QUANTS = (("tau_nonint", "abase"), ("tau_I", "int"), ("tau_pkg", "apkg"))


def nonint_larger(r, coord):
    """Is the surrounding package larger than the intervention, on this coordinate?

    One rule for all three coordinates: |tau_{B->-I}| > |tau_{-I->+I}|, a strict
    inequality on the unrounded cell means. Returns None when either mean is absent.

    This used to be computed in two places. negDP and negEO were written here, before
    the CSV round-trip; dAUC was recomputed downstream from the two mean columns after
    reading the CSV, and the table recorded which path it had taken in
    `nonint_larger_source`. The two agreed on the frozen bundle -- 26/26 on negDP and
    29/29 on negEO, no disagreement -- because the rule and the inputs were already the
    same. They are now one function so they cannot drift apart.
    """
    a, b = r.get(f"tau_nonint_{coord}_mean"), r.get(f"tau_I_{coord}_mean")
    if a is None or b is None:
        return None
    return bool(abs(a) > abs(b))

# Configuration identity and role, fixed before any outcome existed. The suffix
# in a stored method name is a configuration label, not a separate method.
CONFIG_ROLE = {
    ("FairSIN", "GCN"): "primary", ("FairSIN", "GIN"): "robustness",
    ("FairSIN", "SAGE"): "robustness",
    ("BIND", "1pct"): "primary", ("BIND", "10pct"): "robustness",
    # further published rows of a method whose primary row is already in the set
    ("FairVGNN", "GCNspmm"): "robustness", ("FairVGNN", "GIN"): "robustness",
    ("FairVGNN", "SAGE"): "robustness", ("BeMap", "GAT"): "robustness",
    ("FairGNN", "upstreamGCN"): "robustness", ("FairGNN", "upstreamGAT"): "robustness",
}
# Methods released for another task whose training loop the harness had to
# complete: reported on their own, never counted in the primary summary.
TASK_ADAPTED = {"FnRGNN": "released for node regression: loss MSE -> BCE, losses on training "
                          "nodes only, mmd_sample and H set by the harness (no loop released)"}
# Native protocols that are scientifically valid (leakage-free), fixed at
# admission. Others have no native counterpart by construction.
#
# Two native families, never pooled, because their sampling rationale differs:
#   systematic -- every configuration of a method that has a leakage-free
#                 published procedure is run natively (FairSIN).
#   targeted   -- a validation set fixed in advance to probe protocol
#                 sensitivity on already-frozen controlled cells.
NATIVE_VALID = {"FairSIN", "NIFTY", "FairVGNN", "FairGB", "SFG"}
NATIVE_ROLE = {"FairSIN": "systematic", "NIFTY": "targeted",
               "FairVGNN": "targeted", "FairGB": "targeted", "SFG": "systematic"}


def native_role(method, configuration):
    """FairVGNN's further published rows were run natively as a sweep (every
    row), so they are systematic; its primary row belongs to the frozen targeted set."""
    if method == "FairVGNN" and configuration != "default":
        return "systematic"
    return NATIVE_ROLE.get(method, "")
# The frozen targeted native store, one file per cell.
TARGETED_NATIVE = [("NIFTY", "german"), ("FairVGNN", "german"), ("FairVGNN", "bail"),
                   ("FairVGNN", "credit"), ("FairGB", "german"), ("FairGB", "bail"),
                   ("FairGB", "credit")]
FROZEN_NATIVE_ANALYSIS = f"{os.path.join(ROOT, 'harness', 'results')}/armB_phase1_seven_cell_analysis.txt"
# `backbone` names the message-passing encoder the cell actually trained, and
# nothing else: the harness's stored label carries a method's configuration
# suffix (BIND's budget) or its own generic default, which would mislabel a plot
# grouped by backbone. One canonical map decides it for every file, and a gate
# below refuses to write if two files disagree. FairSIN is the one method whose
# configuration *is* the encoder, so it takes the configuration.
CANONICAL_BACKBONE = {
    "B (common baseline)": "GCN",
    "FairGNN": "GCN", "NIFTY": "GCN", "FairVGNN": "GCN",
    "FairGB": "SAGE",          # FairGB_alg.py:47, its own default encoder
    "EDITS": "GCN",            # EDITS.py's downstream GCN
    "FairEdit": "GCN",         # FairEdit.py model_name='gcn'
    "BeMap": "GCN",            # BeMap_GCN (DGL GraphConv)
    "GEAR": "SAGE",            # GEAR main.py parser default encoder='sage'
    "BIND": "GCN",             # BIND GNNs/gcn.py; 1pct / 10pct is the budget
    "SFG": "SAGE",             # SFG run.sh rows, --encoder SAGE
    "FnRGNN": "GCN",           # FnRGNN two GCNConv layers
}
# configurations whose label names the encoder the cell trained
CONFIG_BACKBONE = {("FairVGNN", "GCNspmm"): "GCN", ("FairVGNN", "GIN"): "GIN",
                   ("FairVGNN", "SAGE"): "SAGE", ("BeMap", "GAT"): "GAT",
                   ("FairGNN", "upstreamGCN"): "GCN", ("FairGNN", "upstreamGAT"): "GAT"}


def canonical_backbone(method, configuration, fallback="GCN"):
    if method == "FairSIN":
        return configuration
    if (method, configuration) in CONFIG_BACKBONE:
        return CONFIG_BACKBONE[(method, configuration)]
    return CANONICAL_BACKBONE.get(method, fallback)


def split_name(stored: str):
    """'FairSIN-GCN' -> ('FairSIN', 'GCN'); 'EDITS' -> ('EDITS', 'default')."""
    if "-" in stored:
        m, c = stored.rsplit("-", 1)
        return m, c
    return stored, "default"


def role_of(method, configuration):
    return CONFIG_ROLE.get((method, configuration), "primary")


def _inputs(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        b = os.path.basename(f)
        if b.endswith(("_summary.csv", "_cell_table.csv")) or "_interim_" in b:
            continue
        out.append(f)
    return out


def load_store():
    """Every controlled and native row, from the untouched raw stores.

    Native rows come from two frozen stores: the systematic sweep and the
    targeted validation set, each read as it was written."""
    # the FMP baseline stage and FnRGNN's regression task have their own schemas and
    # sections; only classification cells are read here
    x31 = [f for f in _inputs(f"{RES}/x31/x31_*.csv")
           if "_fmp_" not in os.path.basename(f) and "-regression_" not in os.path.basename(f)]
    controlled = [build(CORE_CSVS), build(_inputs(f"{RES}/x29/x29_*.csv")),
                  build(_inputs(f"{RES}/x30/x30_*.csv")), build(x31)]
    c = pd.concat([d for d in controlled if not d.empty], ignore_index=True)
    c["protocol"] = "controlled"

    sysnat = (_inputs(f"{RES}/x30native_*.csv") + _inputs(f"{RES}/x30/x30native_*.csv")
              + _inputs(f"{RES}/x31/x31native_*.csv"))
    parts = [c]
    if sysnat:
        n = build(sysnat); n["protocol"] = "native"; parts.append(n)
    tgt = [f"{RES}/armB_native_{m}_{d}.csv" for m, d in TARGETED_NATIVE]
    missing = [f for f in tgt if not os.path.exists(f)]
    if missing:
        raise SystemExit("[final] BLOCKED: frozen targeted native artifacts missing: "
                         + ", ".join(os.path.basename(f) for f in missing))
    t = build(tgt); t["protocol"] = "native"
    parts.append(t)
    return pd.concat(parts, ignore_index=True)


def cell_key(d):
    return list(d.groupby(["method", "dataset", "protocol"]).groups)


def completeness(g):
    return g.groupby(["split_id", "run_id"]).ngroups, len(g)


def frozen_estimates(g):
    """The frozen per-cell bootstrap for one cell (BCE selector, as frozen)."""
    s = per_cell(g, io.StringIO(), "cell")
    out = {}
    for q_name, q_key in QUANTS:
        for c_name, _ in COORDS:
            lbl = {"dAUC": "dAUC", "negDP": "-dDP", "negEO": "-dEO"}[c_name]
            r = s[(s.quantity == q_name) & (s.coord == lbl)]
            if r.empty:
                continue
            r = r.iloc[0]
            out[f"{q_name}_{c_name}_mean"] = float(r["mean"])
            out[f"{q_name}_{c_name}_lo"] = float(r.lo)
            out[f"{q_name}_{c_name}_hi"] = float(r.hi)
            if q_name == "tau_I":
                out[f"{q_name}_{c_name}_sign_stability"] = float(r["sign"])
                out[f"{q_name}_{c_name}_resolved"] = bool(r.resolved)
    return out


def auc_selector_estimates(g):
    """The same frozen bootstrap applied to the AUC-selected readings of the
    same matched units -- the robustness selector fixed in the design."""
    a = g[g.selector == "common_auc"]
    if a.empty:
        return {}
    cells = (a.set_index(["split_id", "run_id"])[["int_auc", "int_ndp", "int_neo"]]
             .reset_index())
    cols = ["int_auc", "int_ndp", "int_neo"]
    reps = boot(cells, cols, np.random.default_rng(SEED))
    out = {}
    for c_name, c_key in COORDS:
        k = cols.index(f"int_{c_key}")
        lo, hi = np.percentile(reps[:, k], [2.5, 97.5])
        mu = float(cells[f"int_{c_key}"].mean())
        sgn = float(sign_stability(a[f"int_{c_key}"]))
        out[f"tau_I_{c_name}_auc_selector_mean"] = mu
        out[f"tau_I_{c_name}_auc_selector_lo"] = float(lo)
        out[f"tau_I_{c_name}_auc_selector_hi"] = float(hi)
        out[f"tau_I_{c_name}_auc_selector_sign_stability"] = sgn
        out[f"tau_I_{c_name}_auc_selector_resolved"] = bool(
            sgn >= SIGN_MIN and abs(mu) >= NEAR_ZERO and lo * hi > 0)
    return out


CAVEAT = {
    ("GEAR", "bail"): "the released counterfactual assets are three identical files that flip "
                      "the sensitive attribute on the unchanged graph, so the cell measures the "
                      "similarity objective given an attribute flip, not the published generator",
    ("FairEdit", "german"): "the repository default deletes 10 undirected edges per run "
                          "(20 of 44,484 directed entries); the edge-addition branch is disabled "
                          "(add=False), so a near-zero estimate reflects the size of the intervention",
    ("FairEdit", "bail"): "the repository default deletes 10 undirected edges per run "
                          "(20 of 642,616 directed entries); the edge-addition branch is disabled "
                          "(add=False), so a near-zero estimate reflects the size of the intervention",
    ("FairEdit", "credit"): "the repository default deletes 10 undirected edges per run "
                          "(20 of 2,873,716 directed entries); the edge-addition branch is disabled "
                          "(add=False), so a near-zero estimate reflects the size of the intervention",
    ("BIND", "bail"): "the published 1% / 10% budgets are applied as fractions of the training "
                      "set; on bail (|train| = 100) that is 1 and 10 deleted nodes",
    ("BIND", "income"): "the published 1% / 10% budgets are applied as fractions of the training "
                        "set; on income (|train| = 4605) that is 46 and 461 deleted nodes",
    ("EDITS", "german"): "the wrapper's parameter routing was repaired from its own stored values "
                         "so the structural half of the intervention is active",
    ("EDITS", "bail"): "the wrapper's parameter routing was repaired from its own stored values "
                       "so the structural half of the intervention is active",
    ("EDITS", "credit"): "the wrapper's parameter routing was repaired from its own stored values "
                         "so the structural half of the intervention is active",
}
NATIVE_NOTE = {
    "BeMap": "no native cell: the official training loop selects on the test split",
    "BIND": "no native cell: the published influence estimator reads test labels; the controlled "
            "cells use the same estimator with its fairness cost on validation nodes",
    "GEAR": "no native cell: the published counterfactual assets cannot be reproduced",
    "EDITS": "no native cell: the repository carries no official configuration",
    "FairEdit": "no native cell: the repository carries no official configuration",
}


def build_cells(store, report):
    rows, pending = [], []
    for (stored, ds, proto), g in store.groupby(["method", "dataset", "protocol"], sort=True):
        method, configuration = split_name(stored)
        role = role_of(method, configuration)
        units, nrows = completeness(g)
        backbone = (g.backbone.iloc[0] if "backbone" in g.columns
                    and isinstance(g.backbone.iloc[0], str) else "GCN")
        backbone = canonical_backbone(method, configuration, backbone)
        base = dict(method=method, backbone=backbone, dataset=ds,
                    configuration=configuration, configuration_role=role, protocol=proto,
                    native_evaluation_role=(native_role(method, configuration) if proto == "native" else ""),
                    units_done=units, units_required=UNITS_REQUIRED)
        if units != UNITS_REQUIRED or nrows != ROWS_REQUIRED:
            pending.append(dict(base, status="partial" if units else "pending"))
            continue
        pending.append(dict(base, status="complete"))
        nat_role = native_role(method, configuration) if proto == "native" else ""
        # B's protocol, read from the rows themselves rather than assumed
        def _eps(col):
            if col not in g.columns:
                return []
            v = pd.to_numeric(g[col], errors="coerce").dropna().unique()
            return sorted({int(x) for x in v})
        b_ep, m_ep = _eps("b_epochs"), _eps("method_epochs")
        if proto == "controlled":
            base_ref = "same_protocol"
        else:
            base_ref = ("controlled_fixed_reference" if b_ep == [200]
                        else ("native" if b_ep == m_ep else "unknown"))
        r = dict(method=method, backbone=backbone, dataset=ds, configuration=configuration,
                 configuration_role=role, protocol=proto, native_evaluation_role=nat_role,
                 baseline_reference_protocol=base_ref,
                 count_in_primary_summary=bool(role == "primary" and proto == "controlled"
                                               and method not in TASK_ADAPTED),
                 task_adaptation=TASK_ADAPTED.get(method, ""),
                 status="complete", n_units=units)
        r.update(frozen_estimates(g))
        r.update(auc_selector_estimates(g))
        for c, _ in COORDS:
            v = nonint_larger(r, c)
            if v is not None:
                r[f"nonint_larger_{c}"] = v
        note = CAVEAT.get((method, ds), "")
        if proto == "controlled" and method in NATIVE_NOTE:
            note = (note + "; " if note else "") + NATIVE_NOTE[method]
        if proto == "native" and r["baseline_reference_protocol"] == "controlled_fixed_reference":
            note = (note + "; " if note else "") + (
                "native protocol for the method arms only: the baseline is the fixed controlled "
                "reference trained at H = 200, so tau_I is a within-protocol contrast while "
                "tau_nonint and tau_pkg are measured against that fixed reference and are not a "
                "same-protocol B -> M contrast")
        if method in TASK_ADAPTED:
            note = (note + "; " if note else "") + (
                "task-adapted: " + TASK_ADAPTED[method] + "; reported on its own, never "
                "counted in the primary summary")
        if role == "robustness":
            note = (note + "; " if note else "") + (
                "robustness configuration: shares data, splits and baseline with the primary "
                "configuration of the same dataset, never pooled into primary counts")
        r["caveat"] = note
        rows.append(r)
        report.append(f"  mapped {method}/{configuration}/{ds}/{proto} -> "
                      f"{'primary' if r['count_in_primary_summary'] else role + '/' + proto}")
    return pd.DataFrame(rows), pd.DataFrame(pending)


def protocol_pairs(cells):
    out = []
    ctrl = cells[cells.protocol == "controlled"]
    nat = cells[cells.protocol == "native"]
    for r in nat.itertuples():
        if r.method not in NATIVE_VALID:
            continue
        m = ctrl[(ctrl.method == r.method) & (ctrl.dataset == r.dataset)
                 & (ctrl.configuration == r.configuration)]
        if m.empty:
            continue
        m = m.iloc[0]
        out.append(dict(
            comparison_type="controlled_vs_native", method=r.method, backbone=r.backbone,
            dataset=r.dataset, configuration=r.configuration,
            native_evaluation_role=r.native_evaluation_role,
            x_name="controlled_tau_I_negDP", x_mean=m.tau_I_negDP_mean,
            x_lo=m.tau_I_negDP_lo, x_hi=m.tau_I_negDP_hi, x_resolved=bool(m.tau_I_negDP_resolved),
            y_name="native_tau_I_negDP", y_mean=r.tau_I_negDP_mean,
            y_lo=r.tau_I_negDP_lo, y_hi=r.tau_I_negDP_hi, y_resolved=bool(r.tau_I_negDP_resolved),
            sign_changed=bool((m.tau_I_negDP_mean > 0) != (r.tau_I_negDP_mean > 0)),
            resolution_changed=bool(bool(m.tau_I_negDP_resolved) != bool(r.tau_I_negDP_resolved)),
            abs_shift=abs(r.tau_I_negDP_mean - m.tau_I_negDP_mean)))
    # every controlled cell has both selectors: primary and robustness configurations
    for r in ctrl.itertuples():
        if not hasattr(r, "tau_I_negDP_auc_selector_mean") or pd.isna(
                r.tau_I_negDP_auc_selector_mean):
            continue
        out.append(dict(
            comparison_type="bce_vs_auc_selector", method=r.method, backbone=r.backbone,
            dataset=r.dataset, configuration=r.configuration, native_evaluation_role="",
            configuration_role=r.configuration_role, task_adaptation=r.task_adaptation,
            x_name="bce_selected_tau_I_negDP", x_mean=r.tau_I_negDP_mean,
            x_lo=r.tau_I_negDP_lo, x_hi=r.tau_I_negDP_hi, x_resolved=bool(r.tau_I_negDP_resolved),
            y_name="auc_selected_tau_I_negDP", y_mean=r.tau_I_negDP_auc_selector_mean,
            y_lo=r.tau_I_negDP_auc_selector_lo, y_hi=r.tau_I_negDP_auc_selector_hi,
            y_resolved=bool(r.tau_I_negDP_auc_selector_resolved),
            sign_changed=bool((r.tau_I_negDP_mean > 0) != (r.tau_I_negDP_auc_selector_mean > 0)),
            resolution_changed=bool(bool(r.tau_I_negDP_resolved)
                                    != bool(r.tau_I_negDP_auc_selector_resolved)),
            abs_shift=abs(r.tau_I_negDP_auc_selector_mean - r.tau_I_negDP_mean)))
    return pd.DataFrame(out)


MECH = [  # (analysis, method, dataset, frozen summary file, selector column?)
    ("selection-support decomposition", "NIFTY", "german", f"{RES}/x25/x25_summary.csv", False),
    ("selection-support decomposition", "FairGB", "bail", f"{RES}/x26/x26_bail_summary.csv", False),
    ("component analysis", "FMP", "pokec_z", f"{RES}/x27/x27_pokec_z_summary.csv", True),
    ("component analysis", "FMP", "pokec_n", f"{RES}/x27/x27_pokec_n_summary.csv", True),
    # the baseline stage: B trained under the FMP arms' own setting
    ("component analysis", "FMP", "pokec_z", f"{RES}/x31/x31_fmp_B_pokec_z_summary.csv", True),
    ("component analysis", "FMP", "pokec_n", f"{RES}/x31/x31_fmp_B_pokec_n_summary.csv", True),
]
COORD_OF = {"auc": "dAUC", "ndp": "negDP", "neo": "negEO"}


REGRESSION_SUMMARY = f"{RES}/x31/x31_FnRGNN-regression_summary.csv"
REGRESSION_DATASETS = ("german", "pokec_z", "pokec_n")


def regression_results():
    """FnRGNN on its released task (node regression). Regression metrics, so it is
    its own section and never joins the classification tables."""
    if not os.path.exists(REGRESSION_SUMMARY):
        raise SystemExit("[final] BLOCKED: FnRGNN regression summary missing "
                         f"({REGRESSION_SUMMARY}); run analyze_x31_fnrgnn_regression.py")
    r = pd.read_csv(REGRESSION_SUMMARY)
    if sorted(r.dataset) != sorted(REGRESSION_DATASETS) or (r.n_units != UNITS_REQUIRED).any():
        raise SystemExit("[final] BLOCKED: FnRGNN regression summary incomplete")
    r.insert(1, "backbone", "GCN")
    return r


def regression_configurations():
    import x31_fnrgnn as FR
    out = []
    for ds in REGRESSION_DATASETS:
        c = FR.config(ds)
        out.append(dict(
            method="FnRGNN", backbone="GCN", dataset=ds,
            configuration="regression (released task)", protocol="controlled",
            intervention_I="edge reweighting by feature similarity and sensitive difference, "
                           "MMD representation alignment and GWN prediction normalisation",
            M_plus_I="use_edge_weight, use_mmd, use_gwn = True",
            M_minus_I="all three False (same GCN, unit edge weights)",
            horizon="H = 200", selector="smallest validation MSE (strict <, earliest tie)",
            optimizer_and_hyperparameters="; ".join(f"{k}={c[k]:.6g}" for k in FR.APPLIED)
                                          + f"; mmd_sample={c['mmd_sample']}; loss MSE (released)",
            preprocessing="target = " + {"german": "LoanAmount"}.get(ds, "completion_percentage")
                          + ", removed from the features and standardised with the training "
                            "nodes' statistics; features StandardScaler (FnRGNN loader); B: the "
                            "common GCN at published('GNN', ds) trained with MSE",
            configuration_source=f"FnRGNN-master/logs/best_configs/{c['config_file']}",
            provenance="official-repo class and configuration; harness-completed loop",
            caveat="released task kept (node regression); training loop completed by the harness "
                   "(training-node mask, mmd_sample, H, selection); regression metrics, not "
                   "comparable with the classification sections",
            native_evaluation_role="", baseline_reference_protocol="same_protocol"))
    return pd.DataFrame(out)


def mechanistic():
    """The frozen mechanistic analyses, transcribed, never recomputed."""
    out = []
    for analysis, method, ds, path, has_sel in MECH:
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        for r in d.itertuples():
            q = str(r.quantity)
            suffix = q.rsplit(".", 1)[-1]
            if suffix not in COORD_OF:
                continue
            note = f"selector {r.selector}" if has_sel else ""
            out.append(dict(analysis=analysis, method=method, dataset=ds,
                            coordinate=COORD_OF[suffix], term=q.rsplit(".", 1)[0],
                            mean=r.mean, lo=r.lo, hi=r.hi, resolved=bool(r.resolved),
                            note=note))
    return pd.DataFrame(out)


def configurations(cells):
    """Paper-facing configuration table, keyed to the cells that exist."""
    import x30_config_table as CT
    raw = pd.DataFrame(CT.rows())
    raw["protocol"] = raw.protocol.map({"x30": "controlled", "x30native": "native"})
    keep = {(r.method, r.dataset, r.protocol, r.configuration) for r in cells.itertuples()}
    out = []
    for r in raw.itertuples():
        explicit = getattr(r, "configuration", None)
        if isinstance(explicit, str) and explicit:
            meth, cfg = r.method, explicit
        elif r.method == "B (baseline)":
            cfg, meth = "baseline", "B (common baseline)"
        elif r.method.startswith("BIND-"):
            meth, cfg = "BIND", r.method.split("-", 1)[1]
        elif r.method == "FairSIN":
            meth, cfg = "FairSIN", r.backbone
        elif r.method in ("FairGNN", "NIFTY", "FairVGNN", "FairGB"):
            meth, cfg = r.method, "default"
        else:
            meth, cfg = r.method, "default"
        if meth != "B (common baseline)" and (meth, r.dataset, r.protocol, cfg) not in keep:
            continue
        bb = canonical_backbone(meth, cfg, r.backbone)
        horizon = (f"H = {r.horizon_H}" if r.protocol == "controlled"
                   else f"native horizon = {r.native_horizon}")
        sel = (r.selector_controlled if r.protocol == "controlled" else r.native_selector)
        out.append(dict(
            method=meth, backbone=bb, dataset=r.dataset, configuration=cfg,
            protocol=r.protocol, intervention_I=r.intervention_I, M_plus_I=r.M_plus_I,
            M_minus_I=r.M_minus_I, horizon=horizon, selector=sel,
            optimizer_and_hyperparameters=r.optimiser_and_hyperparameters,
            preprocessing=r.preprocessing, configuration_source=r.configuration_source,
            provenance=r.provenance,
            caveat=str(r.caveat).replace("(protocol 2.2)", "").replace("(protocol 2.5)", "")
                   .replace("protocol 2.5", "").strip("; ").strip()))
    out.extend(targeted_native_configurations())
    for r in out:
        r.setdefault("native_evaluation_role",
                     native_role(r["method"], r["configuration"]) if r["protocol"] == "native" else "")
        r.setdefault("baseline_reference_protocol",
                     "controlled_fixed_reference" if r["protocol"] == "native"
                     else "same_protocol")
    return pd.DataFrame(out)


def targeted_native_configurations():
    """The seven frozen targeted native cells, restored from the same native
    configuration interpreter their runs used -- never from a summary table."""
    from core.published_config import native_config
    from pilot_tau import METHODS
    import x30_config_table as CT
    out = []
    for m, ds in TARGETED_NATIVE:
        n = native_config(m, ds)
        c = dict(n["config"])
        fn = c.pop("feature_normalize", False)
        knobs = CT.CORE_I[m][1]
        plus = ", ".join(f"{k}={c[k]}" for k in knobs if k in c) or "repository configuration"
        minus = ", ".join(f"{k}={v}" for k, v in sorted(METHODS[m]["off"].items()))
        changes = "; ".join(n.get("native_changes") or [])
        out.append(dict(
            method=m, backbone=canonical_backbone(m, "default"), dataset=ds,
            configuration="default", protocol="native",
            intervention_I=CT.CORE_I[m][0], M_plus_I=plus, M_minus_I=minus,
            horizon=f"native horizon = {n['horizon']}",
            selector="sigma_c^BCE (the controlled selector) for the reported estimate; the "
                     "method's own published rule is replayed on the stored trajectory and "
                     "recorded as code_epoch / m1pub_*, and sigma_c^AUC is also recorded",
            optimizer_and_hyperparameters="; ".join(f"{k}={v}" for k, v in sorted(c.items())),
            preprocessing=f"feature_normalize={bool(fn)}"
                          + (f"; {changes}" if changes else ""),
            configuration_source=n["source"], provenance=n["provenance"],
            caveat="targeted native validation, fixed in advance to probe protocol "
                   "sensitivity; the baseline is the fixed controlled reference at H = 200"))
    return out


def frozen_native_reference(path):
    """Parse the frozen targeted-native analysis: its own native column, so the
    rebuilt values can be checked against what was published."""
    if not os.path.exists(path):
        return {}
    import re
    num = r"([+-]?\d+\.\d+)"
    pat = re.compile(rf"^\s+(tau_base|tau_int)\s+{num}\s+\[{num},{num}\]\s+s(\d+\.\d+)"
                     rf"\s+{num}\s+\[{num},{num}\]\s+s(\d+\.\d+)")
    head = re.compile(r"^(\w[\w-]*) / (\w+)\s+native H")
    coord = {"-dDP": "negDP", "-dEO": "negEO", "dAUC": "dAUC"}
    out, cell, co = {}, None, None
    for line in open(path):
        h = head.match(line)
        if h:
            cell = (h.group(1), h.group(2)); continue
        t = line.strip().split("  ")[0].strip()
        if t in coord:
            co = coord[t]; continue
        m = pat.match(line)
        if m and cell and co:
            q = "tau_nonint" if m.group(1) == "tau_base" else "tau_I"
            out[(cell[0], cell[1], q, co)] = dict(
                mean=float(m.group(6)), lo=float(m.group(7)), hi=float(m.group(8)),
                sign=float(m.group(9)))
    return out


def gates(cells, cov, pairs, store, prev, report, cfg_for_pairs=None):
    """Every gate is a hard stop: a failure leaves the bundle untouched."""
    fails = []
    cfg_for_pairs = pd.DataFrame() if cfg_for_pairs is None else cfg_for_pairs
    # A / B: every completed cell in the stores maps to exactly one canonical row
    completed = {(m, d, p) for (m, d, p), g in
                 store.groupby(["method", "dataset", "protocol"])
                 if completeness(g) == (UNITS_REQUIRED, ROWS_REQUIRED)}
    mapped = {(f"{r.method}-{r.configuration}" if r.configuration != "default" else r.method,
               r.dataset, r.protocol) for r in cells.itertuples()}
    if mapped != completed:
        fails.append(f"A/B mapping mismatch: missing {completed - mapped}, extra {mapped - completed}")
    if len(cells) != len(cells.drop_duplicates(["method", "dataset", "configuration", "protocol"])):
        fails.append("A duplicate canonical rows")
    # C: robustness and native present, and never counted as primary
    bad = cells[(cells.configuration_role == "robustness") | (cells.protocol == "native")]
    if bad.count_in_primary_summary.any():
        fails.append("C a robustness or native row is counted in the primary summary")
    # D: identity per coordinate
    for c_name, _ in COORDS:
        r = (cells[f"tau_pkg_{c_name}_mean"]
             - (cells[f"tau_nonint_{c_name}_mean"] + cells[f"tau_I_{c_name}_mean"])).abs().max()
        report.append(f"  identity tau_pkg = tau_nonint + tau_I ({c_name}): max |residual| {r:.2e}")
        if not (r <= 1e-9):
            fails.append(f"D identity violated on {c_name} (max |residual| {r:.3e})")
    # E: numerically identical to the values published in the previous bundle
    n_cmp = 0
    if prev is not None and not prev.empty:
        key = ["method", "dataset", "configuration", "protocol"]
        p_ = prev.set_index(key)
        for r in cells.itertuples():
            k = (r.method, r.dataset, r.configuration, r.protocol)
            if k not in p_.index:
                continue
            old = p_.loc[k]
            for col in cells.columns:
                if not col.endswith(("_mean", "_lo", "_hi", "_sign_stability")):
                    continue
                a_, b_ = old.get(col), getattr(r, col, None)
                if a_ is None or b_ is None or pd.isna(a_) or pd.isna(b_):
                    continue
                n_cmp += 1
                if abs(float(a_) - float(b_)) > 1e-12:
                    fails.append(f"E value drift {k} {col}: {a_!r} -> {b_!r}")
            for col in [c for c in cells.columns if c.endswith("_resolved")]:
                a_, b_ = old.get(col), getattr(r, col, None)
                if a_ is None or pd.isna(a_):
                    continue
                n_cmp += 1
                if bool(a_) != bool(b_):
                    fails.append(f"E resolution drift {k} {col}: {a_!r} -> {b_!r}")
    report.append(f"  compared {n_cmp} previously published values (tolerance 1e-12)")
    # F: no pending or partial cell in the scientific files
    part = {(r.method, r.dataset, r.protocol, r.configuration) for r in cov.itertuples()
            if r.status != "complete"}
    for df, name in ((cells, "cell_results"), (pairs, "protocol_pairs")):
        if df.empty:
            continue
        hit = {(r.method, r.dataset, r.protocol if hasattr(r, "protocol") else "controlled",
                r.configuration) for r in df.itertuples()} & part
        if hit:
            fails.append(f"F pending/partial cell present in {name}: {sorted(hit)}")
    # E2: the targeted native cells against the values published with them
    ref = frozen_native_reference(FROZEN_NATIVE_ANALYSIS)
    if ref:
        dmean = dci = 0.0
        n2 = 0
        for r in cells[(cells.protocol == "native")
                       & (cells.native_evaluation_role == "targeted")].itertuples():
            for q in ("tau_nonint", "tau_I"):
                for c_name, _ in COORDS:
                    k = (r.method, r.dataset, q, c_name)
                    if k not in ref:
                        continue
                    n2 += 1
                    o = ref[k]
                    dmean = max(dmean, abs(o["mean"] - getattr(r, f"{q}_{c_name}_mean")))
                    dci = max(dci, abs(o["lo"] - getattr(r, f"{q}_{c_name}_lo")),
                              abs(o["hi"] - getattr(r, f"{q}_{c_name}_hi")))
                    if q == "tau_I":
                        ds_ = abs(o["sign"] - getattr(r, f"{q}_{c_name}_sign_stability"))
                        if ds_ > 5e-3:
                            fails.append(f"E2 sign stability drift {k}: {o['sign']} vs "
                                         f"{getattr(r, f'{q}_{c_name}_sign_stability')}")
        report.append(f"  targeted native vs its published values: {n2} quantities, "
                      f"max |mean| difference {dmean:.2e}, max interval difference {dci:.2e} "
                      f"(published to 4 decimals)")
        if dmean > 5e-5:
            fails.append(f"E2 targeted native means differ from the published values "
                         f"(max {dmean:.3e})")
        if dci > 1e-2:
            fails.append(f"E2 targeted native intervals differ beyond the documented "
                         f"Monte-Carlo stream shift (max {dci:.3e})")
    # H: a controlled/native pair must share one intervention construction
    if not pairs.empty and not cfg_for_pairs.empty:
        ck = {(r.method, r.dataset, r.configuration, r.protocol): (r.M_plus_I, r.M_minus_I)
              for r in cfg_for_pairs.itertuples()}
        for r in pairs[pairs.comparison_type == "controlled_vs_native"].itertuples():
            a_ = ck.get((r.method, r.dataset, r.configuration, "controlled"))
            b_ = ck.get((r.method, r.dataset, r.configuration, "native"))
            if a_ is None or b_ is None:
                fails.append(f"H pair without a configuration row: {r.method}/{r.dataset}")
            elif a_[1] != b_[1]:
                fails.append(f"H pair does not share the off-state construction: "
                             f"{r.method}/{r.dataset}: {a_[1]!r} vs {b_[1]!r}")
    # G: inclusion and classification never look at a result
    if not cells.empty:
        for r in cells.itertuples():
            if r.count_in_primary_summary != (r.configuration_role == "primary"
                                              and r.protocol == "controlled"
                                              and r.method not in TASK_ADAPTED):
                fails.append("G classification does not follow the fixed rule")
            if r.n_units != UNITS_REQUIRED:
                fails.append("G an included cell is not complete")
    return fails


def backbone_gate(cells, cov, cfg):
    """`backbone` must mean the same thing in every file, and must never carry a
    configuration label."""
    fails = []
    ref = {(r.method, r.configuration): canonical_backbone(r.method, r.configuration)
           for r in cells.itertuples()}
    for name, df in (("cell_results", cells), ("coverage", cov), ("method_configurations", cfg)):
        for r in df.itertuples():
            want = canonical_backbone(r.method, r.configuration)
            if r.backbone != want:
                fails.append(f"backbone mismatch in {name}: {r.method}/{r.configuration} "
                             f"is {r.backbone!r}, expected {want!r}")
            if (r.backbone == r.configuration and r.method != "FairSIN"
                    and CONFIG_BACKBONE.get((r.method, r.configuration)) != r.configuration):
                fails.append(f"backbone carries a configuration label in {name}: "
                             f"{r.method}/{r.configuration}")
    del ref
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--dry-run", action="store_true",
                    help="run every gate and print the summary without replacing the bundle")
    a = ap.parse_args()
    report = []

    prev_p = os.environ.get("X_PREV_CELL_RESULTS", os.path.join(a.out, "cell_results.csv"))
    prev = pd.read_csv(prev_p) if os.path.exists(prev_p) else None

    store = load_store()
    cells, cov = build_cells(store, report)
    pairs = protocol_pairs(cells)
    mech = mechanistic()
    reg = regression_results()
    cfg = pd.concat([configurations(cells), regression_configurations()], ignore_index=True)

    fails = gates(cells, cov, pairs, store, prev, report, cfg)
    cov["backbone"] = [canonical_backbone(r.method, r.configuration, r.backbone)
                       for r in cov.itertuples()]
    cov = pd.concat([cov, pd.DataFrame([dict(
        method="FnRGNN", backbone="GCN", dataset=r.dataset,
        configuration="regression (released task)", configuration_role="primary",
        protocol="controlled", native_evaluation_role="", units_done=int(r.n_units),
        units_required=UNITS_REQUIRED, status="complete") for r in reg.itertuples()])],
        ignore_index=True)
    fails += backbone_gate(cells, cov, cfg)
    fails += feasibility_gate(cells)
    print("\n".join(report))
    prim = cells[cells.count_in_primary_summary]
    rob = cells[(cells.configuration_role == "robustness") & (cells.protocol == "controlled")]
    nat = cells[cells.protocol == "native"]
    nat_sys = nat[nat.native_evaluation_role == "systematic"]
    nat_tgt = nat[nat.native_evaluation_role == "targeted"]
    print("\nvalidation summary")
    print(f"  completed primary controlled cells   {len(prim)}")
    print(f"  robustness controlled cells          {len(rob)}")
    print(f"  native comparisons, systematic       {len(nat_sys)}")
    print(f"  native validations, targeted         {len(nat_tgt)}")
    print(f"  pending or partial cells             {int((cov.status != 'complete').sum())}")
    cvn = pairs[pairs.comparison_type == "controlled_vs_native"] if not pairs.empty else pairs
    print(f"  controlled/native pairs              {len(cvn)}"
          + (f"  (systematic {int((cvn.native_evaluation_role == 'systematic').sum())}, "
             f"targeted {int((cvn.native_evaluation_role == 'targeted').sum())})"
             if not cvn.empty else ""))
    print(f"  BCE/AUC selector pairs               "
          f"{int((pairs.comparison_type == 'bce_vs_auc_selector').sum()) if not pairs.empty else 0}")
    print(f"  identity failures                    "
          f"{sum(1 for f in fails if f.startswith('D'))}")
    print(f"  mapping failures                     "
          f"{sum(1 for f in fails if f.startswith(('A', 'B')))}")
    print("  (primary, robustness, systematic native and targeted native are never pooled)")
    if fails:
        print("\nGATE FAILURE -- the bundle was not replaced:")
        for f in fails:
            print(f"  {f}")
        return 3
    if a.dry_run:
        print("\ndry run: gates passed, bundle not written")
        return 0

    os.makedirs(a.out, exist_ok=True)
    sections = section_tables(cells, pairs, mech, reg)
    # protocol_pairs and mechanistic_results are fully contained in the section
    # files (3a-3c, 4-5), so they are not written; the builder still derives the
    # sections from them in memory.
    keep = {"cell_results.csv", "method_configurations.csv", "coverage.csv",
            "README.md"} | set(sections)
    # remove only stale CSV files this builder once wrote; never a directory or
    # any other file someone put here (e.g. figures/)
    for f in os.listdir(a.out):
        p_ = os.path.join(a.out, f)
        if f not in keep and f.endswith(".csv") and os.path.isfile(p_):
            os.remove(p_)
    cells.to_csv(os.path.join(a.out, "cell_results.csv"), index=False)
    cfg.to_csv(os.path.join(a.out, "method_configurations.csv"), index=False)
    cov[["method", "backbone", "dataset", "configuration", "configuration_role", "protocol",
         "native_evaluation_role", "units_done", "units_required", "status"]].to_csv(
        os.path.join(a.out, "coverage.csv"), index=False)
    for name, df in sections.items():
        df.to_csv(os.path.join(a.out, name), index=False)
    write_readme(a.out, cells, pairs, mech, cov)
    for f in sorted(keep):
        p = os.path.join(a.out, f)
        print(f"[written] {p}")
    return 0


def write_readme(out, cells, pairs, mech, cov):
    prim = cells[cells.count_in_primary_summary]
    rob = cells[(cells.configuration_role == "robustness") & (cells.protocol == "controlled")]
    nat = cells[cells.protocol == "native"]
    sysn = nat[nat.native_evaluation_role == "systematic"]
    tgt = nat[nat.native_evaluation_role == "targeted"]
    tad = cells[(cells.task_adaptation != "") & (cells.protocol == "controlled")]
    txt = f"""# final_results

Paper-facing result bundle. Regenerate everything with one command:

    python harness/experiments/build_results.py

## What a cell is

One cell is one method x dataset x configuration x protocol, evaluated on
6 splits x 5 runs = {UNITS_REQUIRED} matched units. Within a unit, three arms share the
split, the seed and the initialisation:

    B       a common GNN baseline
    M^-I    the method with its claimed fairness intervention switched off
    M^+I    the method at its repository configuration

    tau_nonint = Y(M^-I) - Y(B)     everything the package does apart from the
                                    claimed intervention
    tau_I      = Y(M^+I) - Y(M^-I)  the claimed intervention itself
    tau_pkg    = Y(M^+I) - Y(B)     the package as a whole (= tau_nonint + tau_I)

Coordinates are oriented so that larger is better: `dAUC`, `negDP`, `negEO`.
Uncertainty is a paired hierarchical bootstrap (10,000 replicates, splits
resampled first and then runs within a drawn split, each matched unit carried
whole). An estimate is **resolved** only if sign stability >= {SIGN_MIN},
|mean| >= {NEAR_ZERO}, and the 95% percentile interval excludes zero.

## The evaluation sets, and why they are never pooled

| set | cells | what it is |
|---|---|---|
| primary controlled | {len(prim)} | one configuration per method x dataset, `count_in_primary_summary = true` |
| configuration robustness, controlled | {len(rob)} | further published rows, alternative backbones or alternative intervention budgets of a method already in the primary set |
| task-adapted, controlled | {len(tad)} | a method released for another task (FnRGNN, node regression) whose missing training loop the harness completed; reported on its own, `count_in_primary_summary = false` |
| systematic native comparisons | {len(sysn)} | every configuration of a method whose published procedure is leakage-free, re-run under that published procedure |
| targeted native validations | {len(tgt)} | a validation set fixed in advance to probe protocol sensitivity on already-frozen controlled cells |

The two native families answer different questions and were sampled on
different rationales -- one sweeps a method exhaustively, the other targets
specific frozen cells -- so **they are never combined into a single success
rate**, and neither is added to a primary count. Robustness configurations share
data, splits and baseline with the primary configuration of the same dataset, so
they are not independent cells either.

Only a published procedure that selects without touching the test split is
admissible as a native protocol; methods whose published selection reads test
data have no native row at all.

## Schema

* `configuration_role` — `primary` or `robustness`.
* `protocol` — `controlled` (one common horizon and selector for every arm) or
  `native` (the method at its own published horizon; the checkpoint selector stays the controlled
  validation-BCE rule on both sides, and the method's own published rule is recorded but unused).
* `native_evaluation_role` — empty for controlled rows, else `systematic` or
  `targeted`.
* `count_in_primary_summary` — true only for primary controlled cells of methods
  run as released (never for a task-adapted method).
* `task_adaptation` — empty, or what the harness had to change so a method
  released for another task could be evaluated here.
* `baseline_reference_protocol` — `same_protocol` when B ran under the same
  protocol as the method arms, `controlled_fixed_reference` when the row reuses
  the fixed controlled baseline trained at H = 200.
* `backbone` — the message-passing encoder the cell trained (`GCN`, `GIN`,
  `SAGE` or `GAT`), never a configuration label; what distinguishes two
  configurations of one method (another published row, an encoder, a budget)
  lives in `configuration`.

**What a native row's baseline means.** Every native row reuses the fixed
controlled reference B trained at H = 200; the native horizon applies to the
method arms. The scientific target of a controlled-vs-native comparison is
`tau_I = Y(M^+I) - Y(M^-I)`, which is a contrast between two arms that ran under
the same protocol and is therefore unaffected. `tau_nonint` and `tau_pkg` on a
native row are measured against that fixed reference and must not be read as a
same-protocol B -> M contrast. The algebraic identity
`tau_pkg = tau_nonint + tau_I` still holds, because all three contrasts use the
same B.

## How the result files are organised

`experiment_index.csv` lists every section file with what it holds fixed and
what it varies. The families never mix:

| file | section | held fixed | varied |
|---|---|---|---|
| `1_main_package_vs_intervention.csv` | 1. Main | controlled protocol, primary configuration | method vs common baseline, split into `tau_nonint` and `tau_I` |
| `1b_main_task_adapted.csv` | 1b. Main, task-adapted | controlled protocol | as section 1 for FnRGNN, with `task_adaptation` stating what the harness completed |
| `1c_main_released_task_regression.csv` | 1c. Main, released task | FnRGNN on node regression (its own task, MSE), same split and sensitive attribute; B = the common GCN trained with MSE | `tau_nonint`, `tau_I`, `tau_pkg` on regression metrics: `negMSE`, `negMeanGap`, `negWD` (standardised target units); compare only within this file |
| `2_configuration_variation.csv` | 2. Configuration variation | controlled protocol | the configuration: FairSIN GCN -> GIN, SAGE; FairVGNN's further published rows (GCN-spmm, GIN, SAGE); BeMap GCN -> GAT; FairGNN's upstream GCN and GAT rows; BIND 1% -> 10%. `varied_factor` names it |
| `3a_protocol_selector_bce_vs_auc.csv` | 3a. Protocol variation: selector | configuration, horizon, data, units | only the checkpoint selector (validation BCE vs validation AUC), for every controlled cell; `configuration_role` separates primary from robustness rows |
| `3b_protocol_native_horizon_selector.csv` | 3b. Protocol variation: published horizon | configuration, data and the checkpoint selector | the training horizon, set to the method's own published value, and nothing else; validation-BCE selection is retained on both sides, so this is a horizon-only contrast |
| `3c_protocol_native_published_procedure.csv` | 3c. Protocol variation: published procedure | the intervention's configuration and the checkpoint selector | the training horizon plus the preprocessing or training-loop details the published procedure prescribes; validation-BCE selection is retained, so the selector is not among the changed factors; `factors_changed` lists what is |
| `4_mechanistic_case_study.csv` | 4. Mechanistic case study | the frozen cell | selection-support decompositions (NIFTY, FairGB) |
| `5_component_case_study_FMP.csv` | 5. Component-level case study | FMP on its own split and horizon, the baseline trained under the same setting | four stages, one component at a time: B -> base -> +propagation -> +fairness |
| `model_dataset_feasibility.csv` | coverage | -- | every method x dataset the released code configures, by role |

3a and 3b change only the protocol and can be read as protocol effects. 3c
changes the protocol together with what the published procedure bundles with it
(for example FairVGNN on german and FairGB on german switch feature
normalisation off, NIFTY restores its published drop rates, FairVGNN on credit
uses a separate credit training loop), so it is reported as the effect of the
published procedure as a whole. Which of the two a native row belongs to is
decided from the configuration interpreters alone, before any result is read.

## FnRGNN, two views

FnRGNN was released for node regression and ships no training loop. It is
reported twice, never pooled with any other set:

* `1b` evaluates it on the common classification task, with the loss switched
  to BCE;
* `1c` evaluates it on its own regression task, with the released MSE loss.

Both use the same split, the same released configuration and the same
harness-completed loop (training-node mask, mmd_sample = 500, H = 200). `1c`
measures regression quality and fairness in standardised target units, so its
numbers are comparable only with each other.

## Files

| file | what it holds |
|---|---|
| `cell_results.csv` | one row per completed cell: all three estimands on all three coordinates, with intervals, sign stability, resolution, the AUC-selector robustness estimate, the role metadata above, and any caveat |
| `method_configurations.csv` | what each cell actually ran: intervention, both arms, horizon, selector, hyperparameters, preprocessing, source and provenance. Every result cell has exactly one row here, plus the common baseline |
| `coverage.csv` | every cell with units done / required and status; pending and partial cells appear here and nowhere else |

## Reading the caveats

Some cells carry an implementation caveat that belongs beside the number: a
released asset that does not match its own specification, an intervention whose
repository default is very small, a budget expressed as a fraction of the
training set, a wrapper whose parameter routing had to be repaired from its own
stored values, or a native row's fixed-reference baseline. `cell_results.csv`
and `method_configurations.csv` both carry these in a `caveat` column; none of
them is a post-hoc adjustment to a result.

## Status

{len(prim)} primary controlled, {len(tad)} task-adapted controlled,
{len(rob)} configuration robustness, {len(sysn)} systematic native, {len(tgt)} targeted native,
{int((cov.status != 'complete').sum())} pending or partial.
Pending and partial cells are never summarised scientifically.
"""
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write(txt)


# ---------------------------------------------------------------------------
# Numbered paper sections, derived only from the canonical tables above.
# ---------------------------------------------------------------------------
DATASET_ORDER = ("german", "bail", "credit", "pokec_z", "pokec_n", "pokec_z_g", "pokec_n_g",
                 "income", "nba")
NBA_NOTE = "common split: the validation set is contained in the test set"
# Every method x dataset the method's own released code configures, with what
# was run and why anything else was not. "done" rows are cross-checked against
# cell_results.csv by a gate, so this table cannot claim a cell that does not exist.
# One row per method x dataset that the method's released code configures.
# Columns say, for each role, whether the cell is done, feasible but not run,
# or not possible and why. "done" entries are checked against cell_results.csv.
D, F, T, X, NC = "done", "feasible, not run", "task-adapted only", "excluded", "no configuration"
TD, ND = "done, task-adapted", "not run (decision)"
FEASIBILITY = [
    # method, dataset, primary controlled, robustness controlled, native, source, note
    ("FairGNN", "german", D, NC, NC, "utils/param.json", "upstream configures pokec and nba only"),
    ("FairGNN", "bail", D, NC, NC, "utils/param.json", "upstream configures pokec and nba only"),
    ("FairGNN", "credit", D, NC, NC, "utils/param.json", "upstream configures pokec and nba only"),
    ("FairGNN", "pokec_z", D, D, ND, "primary: utils/param.json; robustness: upstream src/scripts/pokec_z (GCN alpha=100 beta=1 H=2000, GAT alpha=10 beta=0.01 H=800)",
     "robustness = the upstream GCN and GAT rows; native not run by decision"),
    ("FairGNN", "pokec_n", D, D, ND, "primary: utils/param.json; robustness: upstream src/scripts/pokec_n (GCN alpha=50 beta=1 H=1800, GAT alpha=4 beta=0.01 H=1800)",
     "robustness = the upstream GCN and GAT rows; native not run by decision"),
    ("FairGNN", "pokec_z_g", D, NC, NC, "utils/param.json", "gender as the sensitive attribute"),
    ("FairGNN", "pokec_n_g", D, NC, NC, "utils/param.json", "gender as the sensitive attribute"),
    ("FairGNN", "nba", X, X, X, "upstream src/scripts/nba", NBA_NOTE),
    ("NIFTY", "german", D, NC, D, "upstream README", ""),
    ("NIFTY", "bail", D, NC, NC, "utils/param.json", "upstream states german only"),
    ("NIFTY", "credit", D, NC, NC, "utils/param.json", "upstream states german only"),
    ("FairVGNN", "german", D, D, D, "upstream run_german.sh", "primary GCN row; robustness: the other full rows (GCN-spmm, GIN, SAGE); native for the primary row only (the other rows' native runs were not run by decision)"),
    ("FairVGNN", "bail", D, D, D, "upstream run_bail.sh", "primary GCN row; robustness: the other full rows (GCN-spmm, SAGE); native for the primary row only (the other rows' native runs were not run by decision)"),
    ("FairVGNN", "credit", D, D, D, "upstream run_credit.sh + fairvgnn_credit.py", "primary GCN row; robustness: the other full rows (GIN, SAGE); native for the primary row only (the other rows' native runs were not run by decision)"),
    ("FairGB", "german", D, NC, D, "upstream run.sh", ""),
    ("FairGB", "bail", D, NC, D, "upstream run.sh", ""),
    ("FairGB", "credit", D, NC, D, "upstream run.sh", ""),
    ("FairSIN", "german", D, D, D, "experiment.sh ablation block", "primary GCN; robustness GIN, SAGE; native for all three"),
    ("FairSIN", "bail", D, D, D, "experiment.sh ablation block", "primary GCN; robustness GIN, SAGE; native for all three"),
    ("FairSIN", "credit", D, D, D, "experiment.sh ablation block (GCN, GIN via train_mlp.py)", "primary GCN; robustness GIN, SAGE; native for all three"),
    ("FairSIN", "pokec_z", D, D, D, "experiment.sh ablation block", "neutralisation only (no d='yes' row published)"),
    ("FairSIN", "pokec_n", D, D, D, "experiment.sh ablation block", "neutralisation only (no d='yes' row published)"),
    ("EDITS", "german", D, NC, NC, "utils/param.json (repository wrapper)", "no official configuration for a native run"),
    ("EDITS", "bail", D, NC, NC, "utils/param.json (repository wrapper)", "no official configuration for a native run"),
    ("EDITS", "credit", D, NC, NC, "utils/param.json (repository wrapper)", "no official configuration for a native run"),
    ("FairEdit", "german", D, NC, NC, "utils/param.json (repository wrapper)", "no official configuration for a native run"),
    ("FairEdit", "bail", D, NC, NC, "utils/param.json (repository wrapper)", "no official configuration for a native run"),
    ("FairEdit", "credit", D, NC, NC, "utils/param.json (repository wrapper)", "no official configuration for a native run"),
    ("BeMap", "bail", D, D, X, "README command, parser defaults", "primary GCN, robustness GAT (--model gat); native excluded: selects on the test split"),
    ("BeMap", "credit", D, D, X, "README command, parser defaults", "primary GCN, robustness GAT (--model gat); native excluded: selects on the test split"),
    ("BeMap", "pokec_z", D, D, X, "README command, parser defaults", "primary GCN, robustness GAT (--model gat); native excluded: selects on the test split"),
    ("BeMap", "nba", X, X, X, "README command", NBA_NOTE),
    ("GEAR", "bail", D, NC, X, "parser defaults + released augmentation assets", "released counterfactuals are attribute flips; published generator not reproducible"),
    ("GEAR", "credit", X, X, X, "released assets", "released credit assets are on a different graph (3% edge overlap)"),
    ("GEAR", "german", X, X, X, "loader only", "no released assets; generator checkpoints absent"),
    ("BIND", "bail", D, D, X, "parser defaults + README scale 25", "primary 1% budget, robustness 10%; native excluded: estimator reads test labels"),
    ("BIND", "income", D, D, X, "parser defaults + README scale 25", "primary 1% budget, robustness 10%; native excluded: estimator reads test labels"),
    ("BIND", "pokec_z", X, X, X, "pokec_dataset.zip (pokec1)", "BIND's pokec1 is a 7,659-node subgraph, not the common pokec_z"),
    ("BIND", "pokec_n", X, X, X, "pokec_dataset.zip (pokec2)", "BIND's pokec2 is a 6,185-node subgraph, not the common pokec_n"),
    ("FMP", "pokec_z", "component study", NC, NC, "run_fgnn.sh grid", "component-level case study: B -> base -> +propagation -> +fairness, the baseline trained on FMP's own split and horizon"),
    ("FMP", "pokec_n", "component study", NC, NC, "run_fgnn.sh grid", "component-level case study: B -> base -> +propagation -> +fairness, the baseline trained on FMP's own split and horizon"),
    ("FMP", "nba", X, X, X, "run_fgnn.sh grid", NBA_NOTE),
    ("SFG", "german", D, NC, D, "run.sh ablation ladder (SAGE)", "off-state is the authors' own unconstrained row; the native test-reading branch is unreachable at the published horizon"),
    ("SFG", "bail", D, NC, D, "run.sh ablation ladder (SAGE)", "off-state is the authors' own unconstrained row; the native test-reading branch is unreachable at the published horizon"),
    ("SFG", "credit", D, NC, D, "run.sh ablation ladder (SAGE, sfg_credit.py)", "off-state is the authors' own unconstrained row; the native test-reading branch is unreachable at the published horizon"),
    ("FnRGNN", "german", TD, NC, X, "logs/best_configs (regression target LoanAmount)", "released for node regression, no training loop: on classification (1b) loss BCE, on its own regression task (1c) loss MSE; both with training-node mask, mmd_sample and H set by the harness; reported on its own"),
    ("FnRGNN", "pokec_z", TD, NC, X, "logs/best_configs (regression target completion_percentage)", "released for node regression, no training loop: on classification (1b) loss BCE, on its own regression task (1c) loss MSE; both with training-node mask, mmd_sample and H set by the harness; reported on its own"),
    ("FnRGNN", "pokec_n", TD, NC, X, "logs/best_configs (regression target completion_percentage)", "released for node regression, no training loop: on classification (1b) loss BCE, on its own regression task (1c) loss MSE; both with training-node mask, mmd_sample and H set by the harness; reported on its own"),
    ("FnRGNN", "nba", X, X, X, "logs/best_configs", NBA_NOTE),
    ("FairGT", "german", X, X, X, "README commands", "no off-state in the authors' code"),
    ("FairGT", "bail", X, X, X, "README commands", "no off-state in the authors' code"),
    ("FairGT", "credit", X, X, X, "README commands", "no off-state in the authors' code"),
    ("FairGT", "income", X, X, X, "README commands", "no off-state in the authors' code"),
    ("FairGT", "nba", X, X, X, "README commands", "no off-state in the authors' code"),
]


def feasibility_gate(cells):
    fails = []
    have = {(r.method, r.dataset, r.protocol) for r in cells.itertuples()}
    roles = {(r.method, r.dataset, r.protocol, r.configuration_role) for r in cells.itertuples()}
    for m, ds, prim, rob, nat, _, _ in FEASIBILITY:
        if prim in (D, TD) and (m, ds, "controlled", "primary") not in roles:
            fails.append(f"feasibility says {m}/{ds} primary is done but no cell exists")
        if rob == D and (m, ds, "controlled", "robustness") not in roles:
            fails.append(f"feasibility says {m}/{ds} robustness is done but no cell exists")
        if nat == D and (m, ds, "native") not in have:
            fails.append(f"feasibility says {m}/{ds} native is done but no cell exists")
    for r in cells.itertuples():
        row = [f for f in FEASIBILITY if f[0] == r.method and f[1] == r.dataset]
        if not row:
            fails.append(f"cell {r.method}/{r.dataset} missing from the feasibility table")
    return fails


def native_factors(method, dataset, configuration):
    """Everything the native published procedure changes relative to the
    controlled cell of the same configuration, read from the interpreters."""
    # The native runs report the sigma_c^BCE slot, exactly as the controlled arm does; the method's
    # own published rule is replayed and stored (code_epoch / m1pub_*) but never enters tau_I, and
    # the M^{-I} arm has no published-rule counterpart, so a published-rule tau_I is not constructible.
    sel = ("selector unchanged (sigma_c^BCE, as in the controlled arm); the method's own published "
           "rule was replayed and recorded as code_epoch / m1pub_*, but does not enter this estimate")
    if method == "FairSIN":
        import x30_fairsin as FS
        h = FS.native_horizon(dataset, configuration)
        return dict(native_horizon=h, pure=True,
                    factors=f"horizon 200 -> {h}; {sel}")
    if method == "SFG":
        import x31_sfg as SF
        h = SF.native_horizon(dataset)
        hs = f"horizon 200 -> {h}" if h != 200 else "horizon unchanged (200)"
        return dict(native_horizon=h, pure=True, factors=f"{hs}; {sel}")
    if method == "FairVGNN" and configuration != "default":
        import x31_fairvgnn_config_run as VG
        cfg, meta = VG.config(dataset, configuration)
        n = VG.native(cfg, meta, dataset)
        h = int(meta["native_epochs"])
        extra = []
        if n.get("fairvgnn_credit_adapter"):
            extra.append(f"training loop from fairvgnn_credit.py (clip_c={n['clip_c']})")
        if bool(cfg.get("feature_normalize")) != bool(n.get("vg_wrapper_normalize")):
            extra.append("feature normalization "
                         + ("on" if n.get("vg_wrapper_normalize") else "off")
                         + " (official dataset.py rule)")
        parts = ([f"horizon 200 -> {h}"] if h != 200 else ["horizon unchanged (200)"]) + [sel] + extra
        return dict(native_horizon=h, pure=not extra, factors="; ".join(parts))
    from core.published_config import native_config
    n = native_config(method, dataset)
    h = int(n["horizon"])
    extra = [c for c in (n.get("native_changes") or []) if not c.startswith("horizon")]
    parts = ([f"horizon 200 -> {h}"] if h != 200 else ["horizon unchanged (200)"]) + [sel] + extra
    return dict(native_horizon=h, pure=not extra, factors="; ".join(parts))


def section_tables(cells, pairs, mech, reg=None):
    """The five paper sections, each a view of the canonical tables."""
    est = lambda q, c: [f"{q}_{c}_mean", f"{q}_{c}_lo", f"{q}_{c}_hi"]   # noqa: E731
    key = ["method", "backbone", "dataset", "configuration"]
    prim = cells[cells.count_in_primary_summary].copy()
    cols1 = key + ["n_units"]
    for c in ("dAUC", "negDP", "negEO"):
        cols1 += est("tau_pkg", c) + est("tau_nonint", c) + est("tau_I", c) + [
            f"tau_I_{c}_sign_stability", f"tau_I_{c}_resolved"]
    cols1 += ["nonint_larger_dAUC", "nonint_larger_negDP", "nonint_larger_negEO", "caveat"]
    s1 = prim[cols1]
    ta = cells[(cells.task_adaptation != "") & (cells.protocol == "controlled")]
    s1b = ta[cols1[:4] + ["task_adaptation"] + cols1[4:]]

    ctl = cells[cells.protocol == "controlled"]
    rows = []
    for r in ctl[ctl.configuration_role == "robustness"].itertuples():
        p_ = ctl[(ctl.method == r.method) & (ctl.dataset == r.dataset)
                 & (ctl.configuration_role == "primary")]
        if p_.empty:
            continue
        p_ = p_.iloc[0]
        for c in ("dAUC", "negDP", "negEO"):
            a, b = p_[f"tau_I_{c}_mean"], getattr(r, f"tau_I_{c}_mean")
            ra, rb = bool(p_[f"tau_I_{c}_resolved"]), bool(getattr(r, f"tau_I_{c}_resolved"))
            rows.append(dict(method=r.method, dataset=r.dataset, coordinate=c,
                             primary_configuration=p_.configuration,
                             robustness_configuration=r.configuration,
                             primary_backbone=p_.backbone, robustness_backbone=r.backbone,
                             primary_tau_I_mean=a, primary_tau_I_lo=p_[f"tau_I_{c}_lo"],
                             primary_tau_I_hi=p_[f"tau_I_{c}_hi"], primary_resolved=ra,
                             robustness_tau_I_mean=b,
                             robustness_tau_I_lo=getattr(r, f"tau_I_{c}_lo"),
                             robustness_tau_I_hi=getattr(r, f"tau_I_{c}_hi"),
                             robustness_resolved=rb,
                             sign_changed=bool((a > 0) != (b > 0)), resolution_changed=ra != rb,
                             abs_shift=abs(b - a)))
    s2 = pd.DataFrame(rows)
    if not s2.empty:
        vf = {("FairVGNN", "GCNspmm"): "propagation (spmm), with that published row's hyperparameters",
              ("FairVGNN", "GIN"): "backbone (encoder), with that published row's hyperparameters",
              ("FairVGNN", "SAGE"): "backbone (encoder), with that published row's hyperparameters",
              ("BeMap", "GAT"): "backbone (encoder)",
              ("FairGNN", "upstreamGCN"): "hyperparameters: the upstream repository's row",
              ("FairGNN", "upstreamGAT"): "backbone (GAT) with the upstream repository's row"}
        s2.insert(2, "varied_factor", [
            vf.get((m, c), {"FairSIN": "backbone (encoder)", "BIND": "intervention budget"}
                   .get(m, "configuration"))
            for m, c in zip(s2.method, s2.robustness_configuration)])

    s3a = pairs[pairs.comparison_type == "controlled_vs_native"].copy()
    s3a = s3a.drop(columns=["comparison_type"])
    facts = [native_factors(r.method, r.dataset, r.configuration) for r in s3a.itertuples()]
    s3a.insert(5, "controlled_horizon", 200)
    s3a.insert(6, "native_horizon", [f["native_horizon"] for f in facts])
    s3a.insert(7, "factors_changed", [f["factors"] for f in facts])
    s3a.insert(8, "horizon_and_selector_only", [f["pure"] for f in facts])
    s3b = pairs[pairs.comparison_type == "bce_vs_auc_selector"].copy().drop(
        columns=["native_evaluation_role"], errors="ignore")

    s4 = mech[mech.analysis == "selection-support decomposition"].copy()

    f = mech[mech.analysis == "component analysis"].copy()
    import re
    stage = {"base": ("B", "base"),
             "prop": ("base", "+propagation"), "fair": ("+propagation", "+propagation+fairness"),
             "total": ("base", "+propagation+fairness"),
             "pkg": ("B", "+propagation+fairness")}
    out = []
    for r in f.itertuples():
        head = r.term.split(".", 1)[0]
        l1 = re.search(r"l1(\d+(?:\.\d+)?)", r.term)
        l2 = re.search(r"l2(\d+(?:\.\d+)?)", r.term)
        a, b = stage.get(head, ("?", "?"))
        out.append(dict(method=r.method, dataset=r.dataset, selector=r.note.replace("selector ", ""),
                        coordinate=r.coordinate, step=head, from_stage=a, to_stage=b,
                        lambda1_fairness=float(l1.group(1)) if l1 else 0.0,
                        lambda2_propagation=float(l2.group(1)) if l2 else 0.0,
                        mean=r.mean, lo=r.lo, hi=r.hi, resolved=r.resolved))
    s5 = pd.DataFrame(out)
    feas = pd.DataFrame(FEASIBILITY, columns=["method", "dataset", "primary_controlled",
                                              "robustness_controlled", "native",
                                              "configuration_source", "note"])
    feas["dataset"] = pd.Categorical(feas.dataset, DATASET_ORDER, ordered=True)
    feas = feas.sort_values(["method", "dataset"]).reset_index(drop=True)
    s3_native_pure = s3a[s3a.horizon_and_selector_only].drop(columns=["horizon_and_selector_only"])
    s3_native_bundle = s3a[~s3a.horizon_and_selector_only].drop(columns=["horizon_and_selector_only"])
    files = {
        "1_main_package_vs_intervention.csv": s1,
        "1b_main_task_adapted.csv": s1b,
        "1c_main_released_task_regression.csv": reg if reg is not None else pd.DataFrame(),
        "2_configuration_variation.csv": s2,
        "3a_protocol_selector_bce_vs_auc.csv": s3b,
        "3b_protocol_native_horizon_selector.csv": s3_native_pure,
        "3c_protocol_native_published_procedure.csv": s3_native_bundle,
        "4_mechanistic_case_study.csv": s4,
        "5_component_case_study_FMP.csv": s5,
        "model_dataset_feasibility.csv": feas,
    }
    idx = [
        ("1_main_package_vs_intervention.csv", "1", "main", "controlled protocol, primary configuration",
         "method vs common baseline, split into non-intervention and intervention parts", "cell"),
        ("1b_main_task_adapted.csv", "1b", "main, task-adapted method", "controlled protocol",
         "as section 1, for a method released for another task whose training loop the harness completed (FnRGNN)", "cell"),
        ("1c_main_released_task_regression.csv", "1c", "main, released task (regression)", "FnRGNN on its own task: node regression with MSE, same split, B = common GCN with MSE",
         "tau_nonint / tau_I / tau_pkg on regression metrics (negMSE, negMeanGap, negWD)", "cell"),
        ("2_configuration_variation.csv", "2", "configuration variation", "controlled protocol, same method and dataset",
         "the configuration: another published row, encoder backbone or intervention budget; varied_factor names it", "robustness vs primary configuration x coordinate"),
        ("3a_protocol_selector_bce_vs_auc.csv", "3a", "protocol variation (selector only)", "configuration, horizon, data and units",
         "the checkpoint selector: validation BCE vs validation AUC", "controlled cell (primary and robustness)"),
        ("3b_protocol_native_horizon_selector.csv", "3b", "protocol variation (published horizon)",
         "configuration, data and the checkpoint selector",
         "the training horizon, to the method's own published value; nothing else (validation-BCE selection retained)",
         "controlled vs native pair"),
        ("3c_protocol_native_published_procedure.csv", "3c", "protocol variation (published procedure bundle)", "configuration of the intervention",
         "the training horizon plus the preprocessing or training-loop details the published procedure prescribes (validation-BCE selection retained)", "controlled vs native pair"),
        ("4_mechanistic_case_study.csv", "4", "mechanistic case study", "the frozen cell",
         "which trajectory and selection support drives the attribution (NIFTY, FairGB)", "decomposition term x coordinate"),
        ("5_component_case_study_FMP.csv", "5", "component-level case study", "FMP, its own split and horizon",
         "four stages, one component at a time: B, base, +propagation, +fairness", "step x lambda x selector x coordinate"),
    ]
    files["experiment_index.csv"] = pd.DataFrame(
        [dict(file=f, section=sec, family=fam, held_fixed=fix, varied=var, row_unit=unit,
              rows=len(files[f])) for f, sec, fam, fix, var, unit in idx])
    return files


if __name__ == "__main__":
    raise SystemExit(main())
