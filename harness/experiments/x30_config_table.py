"""Emit final_results/method_configurations.csv: what each cell actually ran.

Every value is read from the repository it came from -- each script's own
argparse defaults (via `ast`, never transcribed), `utils/param.json`, or
FairSIN's ablation block -- so the table cannot drift from the adapters.

    python harness/experiments/x30_config_table.py --out final_results
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "harness"))
from core.paths import repo as _repo  # noqa: E402  (repositories live under models/)

import argparse
import ast
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "harness", "adapters"))

HEADER = ["method", "backbone", "dataset", "protocol", "environment",
          "intervention_I", "M_plus_I", "M_minus_I", "horizon_H", "native_horizon",
          "selector_controlled", "native_selector", "optimiser_and_hyperparameters",
          "preprocessing", "configuration_source", "provenance", "caveat"]

SEL = "sigma_c^BCE (primary), sigma_c^AUC (robustness), common evaluator, decision score>0"


def parser_defaults(path):
    """Defaults of a script's own argparse, with its declared types."""
    out = {}
    for node in ast.walk(ast.parse(open(path).read())):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)):
            kw = {k.arg: k.value for k in node.keywords}
            if "default" in kw:
                try:
                    v = ast.literal_eval(kw["default"])
                except ValueError:
                    continue
                t = {"int": int, "float": float, "str": str}.get(
                    getattr(kw.get("type"), "id", None))
                out[node.args[0].value.lstrip("-")] = t(v) if (t and v is not None) else v
    return out


# The four methods whose cells were run before this bundle existed. Their
# configuration is read from the same `published()` resolver the runs used, and
# their off-state from the frozen M0 definition, so the appendix table is
# complete without transcribing anything.
CORE_I = {
    "FairGNN": ("adversarial debiasing: a covariance constraint (alpha) and an adversary loss "
                "(beta) against a sensitive-attribute estimator", ("alpha", "beta")),
    "NIFTY": ("counterfactual-and-noise invariance objective (sim_coeff) on a Lipschitz-"
              "constrained encoder", ("sim_coeff",)),
    "FairVGNN": ("generative feature masking (f_mask) and weight clipping (weight_clip)",
                 ("top_k", "alpha", "clip_e", "ratio")),
    "FairGB": ("contribution alignment (CAL) and counterfactual node mixup (CNM)",
               ("alpha", "eta")),
}
CORE_BACKBONE = {"FairGNN": "GCN", "NIFTY": "GCN", "FairVGNN": "GCN",
                 "FairGB": "SAGE"}      # FairGB_alg.py:47, its own default encoder
CORE_DATASETS = {
    "FairGNN": ("german", "bail", "credit", "pokec_z", "pokec_z_g", "pokec_n", "pokec_n_g"),
    "NIFTY": ("german", "bail", "credit"),
    "FairVGNN": ("german", "bail", "credit"),
    "FairGB": ("german", "bail", "credit"),
}


def core_rows():
    from core.published_config import published
    from pilot_tau import METHODS
    out = []
    for m, (what, knobs) in CORE_I.items():
        off = METHODS[m]["off"]
        for ds in CORE_DATASETS[m]:
            p = published(m, ds)
            c = dict(p["config"])
            fn = c.pop("feature_normalize", False)
            plus = ", ".join(f"{k}={c[k]}" for k in knobs if k in c) or "repository configuration"
            minus = ", ".join(f"{k}={v}" for k, v in sorted(off.items()))
            notes = list(p.get("notes") or [])
            out.append(dict(
                method=m, backbone=CORE_BACKBONE[m], dataset=ds, protocol="x30",
                environment="dev (torch 2.6.0, PyG 2.8.0)",
                intervention_I=what, M_plus_I=plus, M_minus_I=minus,
                horizon_H=200, native_horizon=(p["horizon"] if p["horizon"] else "-"),
                selector_controlled=SEL,
                native_selector="own published selector",
                optimiser_and_hyperparameters="; ".join(
                    f"{k}={v}" for k, v in sorted(c.items())),
                preprocessing=f"feature_normalize={bool(fn)}",
                configuration_source=p["source"], provenance=p["provenance"],
                caveat="; ".join(notes)))
    return out


def rows():
    import x30_fairsin as FS
    import x30_edits as ED
    import x30_fairedit as FE
    from core.published_config import published
    out = list(core_rows())

    # ---- baseline B, one row per dataset actually used
    for ds in ("german", "bail", "credit", "pokec_z", "pokec_n", "income"):
        c = dict(published("GNN", ds)["config"])
        fn = c.pop("feature_normalize", False)
        out.append(dict(
            method="B (baseline)", backbone="GCN", dataset=ds, protocol="x30",
            environment="same as the cell it is paired with",
            intervention_I="none (reference arm)", M_plus_I="-", M_minus_I="-",
            horizon_H=200, native_horizon="-", selector_controlled=SEL,
            native_selector="own best-validation-loss epoch",
            optimiser_and_hyperparameters="; ".join(f"{k}={v}" for k, v in sorted(c.items())),
            preprocessing=f"feature_normalize={bool(fn)}",
            configuration_source="published('GNN', dataset) -> utils/param.json",
            provenance=published("GNN", ds)["provenance"], caveat=""))

    # ---- FairSIN: one row per (dataset, encoder), straight from the ablation block
    for ds in FS.DATASETS:
        for enc in FS.ENCODERS:
            c = FS.config(ds, enc); p = c["plus"]
            hp = "; ".join(f"{k}={p[k]}" for k in
                           ("hidden", "c_lr", "e_lr", "c_wd", "e_wd", "d_lr", "d_wd",
                            "c_epochs", "d_epochs", "m_epoch", "dropout", "alpha", "prop")
                           if k in p)
            if "m_lr" in p:
                hp += f"; m_lr={p['m_lr']}"
            for proto in ("x30", "x30native"):
                out.append(dict(
                    method="FairSIN", backbone=enc, dataset=ds, protocol=proto,
                    environment="dev (torch 2.6.0, PyG 2.8.0)",
                    intervention_I="neutralisation x + delta*MLP(x)"
                                   + (" and the discriminator step" if p["d"] == "yes" else ""),
                    M_plus_I=f"delta={p['delta']}, d='{p['d']}'",
                    M_minus_I="delta=0, d='no'",
                    horizon_H=200 if proto == "x30" else p["epochs"],
                    native_horizon=p["epochs"],
                    selector_controlled=SEL,
                    native_selector="val AUC+F1+ACC - alpha*(DP+EO), strict >, floor 0",
                    optimiser_and_hyperparameters=hp,
                    preprocessing="german raw; others feature_norm with the sensitive column kept",
                    configuration_source=f"FairSIN-main/experiment.sh:{c['plus_line']} "
                                         f"(ablation block), script {c['script']}",
                    provenance="official-repo",
                    caveat=("primary backbone" if enc == "GCN" else
                            "backbone variant, reported apart, never pooled")))

    # ---- EDITS / FairEdit: param.json plus the wrapper's own arguments
    for ds in ED.DATASETS:
        c = ED.config(ds)
        out.append(dict(
            method="EDITS", backbone="GCN (EDITS.py)", dataset=ds, protocol="x30",
            environment="x30_edits_env (dev + deeprobust, aif360)",
            intervention_I="debiasing stage EDITS.fit: attribute debiasing (4 lowest-weight "
                           "columns zeroed) + structural debiasing binarised at threshold",
            M_plus_I=f"fit() then predict(), threshold_proportion={c['threshold_proportion']}",
            M_minus_I="stage bypassed: original features and adjacency into the same predict()",
            horizon_H=200, native_horizon="-", selector_controlled=SEL,
            native_selector="n/a (no official configuration)",
            optimiser_and_hyperparameters=f"lr={c['lr']}; weight_decay={c['weight_decay']}; "
                                          f"debiaser dropout={c['dropout']}; "
                                          f"debias fit epochs={c['fit_epochs']}; nhid=50",
            preprocessing=f"load_data(feature_normalize=False)"
                          + ("; features / column norm" if c["column_norm"] else ""),
            configuration_source="utils/param.json[EDITS] + train_baselines.py:567-587",
            provenance="local-unverified (repository wrapper)",
            caveat="wrapper param routing repaired (protocol 2.2); upstream repo not vendored"))
    for ds in FE.DATASETS:
        c = FE.config(ds)
        out.append(dict(
            method="FairEdit", backbone="GCN (FairEdit.py)", dataset=ds, protocol="x30",
            environment="x30_edits_env", intervention_I="gradient-guided edge editing "
                        "(fair_graph_edit while epoch < edit_num)",
            M_plus_I=f"edit_num={FE.EDIT_NUM_PLUS}", M_minus_I="edit_num=0",
            horizon_H=200, native_horizon="-", selector_controlled=SEL,
            native_selector="n/a (no official configuration)",
            optimiser_and_hyperparameters=f"lr={c['lr']}; weight_decay={c['weight_decay']}; "
                                          f"hidden={c['hidden']}; dropout={c['dropout']}; "
                                          f"model={c['model_name']}",
            preprocessing=f"load_data(feature_normalize={c['feature_normalize']}); "
                          "fit() feature_norm keeping the sensitive column",
            configuration_source="utils/param.json[FairEdit] + train_baselines.py:550-565",
            provenance="local-unverified (repository wrapper)",
            caveat="the repository default deletes 10 undirected edges per run (20 directed "
                   "entries of 44,484 on german, 642,616 on bail, 2,873,716 on credit); the "
                   "edge-addition branch is disabled (add=False)"))

    # ---- BeMap: its script's parser defaults
    bm = parser_defaults(os.path.join(_repo("BeMap-main"), "train_bemap.py"))
    for ds in ("bail", "credit", "pokec_z"):
        out.append(dict(
            method="BeMap", backbone="BeMap_GCN (DGL)", dataset=ds, protocol="x30",
            environment="x27_dgl_cuda (torch 2.2.2+cu121, dgl 1.1.3)",
            intervention_I="balance-aware per-epoch subgraph sampling",
            M_plus_I="train(epoch, ...) -- a fair subgraph each epoch",
            M_minus_I="train(-1, ...) -- the model's own full-graph branch",
            horizon_H=200, native_horizon=bm.get("epochs"), selector_controlled=SEL,
            native_selector="INVALID: the official loop selects on the test split",
            optimiser_and_hyperparameters="; ".join(
                f"{k}={bm[k]}" for k in ("lr", "weight_decay", "hidden", "dropout",
                                         "beta", "lam", "save_num", "model") if k in bm),
            preprocessing="BeMap's own feature_norm (all columns)",
            configuration_source="BeMap-main/README.md command + train_bemap.py parser defaults",
            provenance="official-repo", caveat="controlled only (native selects on test)"))

    # ---- GEAR: main.py parser defaults
    ge = parser_defaults(os.path.join(_repo("GEAR-main"), "src", "main.py"))
    out.append(dict(
        method="GEAR", backbone=f"{ge.get('encoder')} (GEAR Encoder)", dataset="bail",
        protocol="x30", environment="x30_edits_env",
        intervention_I="counterfactual-consistency (similarity) objective",
        M_plus_I=f"sim_coeff={ge.get('sim_coeff')}", M_minus_I="sim_coeff=0",
        horizon_H="200 mini-batch steps", native_horizon=ge.get("epochs"),
        selector_controlled=SEL,
        native_selector="val loss_c+loss_s every 100 steps (not run: assets, protocol 2.5)",
        optimiser_and_hyperparameters="; ".join(
            f"{k}={ge[k]}" for k in ("lr", "weight_decay", "hidden_size", "proj_hidden",
                                     "dropout", "batch_size", "subgraph_size", "n_order")
            if k in ge),
        preprocessing="GEAR feature_norm with the sensitive column kept; PPR subgraphs",
        configuration_source="GEAR-main/src/main.py parser defaults, --dataset bail",
        provenance="official-repo (released aug assets)",
        caveat="released counterfactual assets are sensitive-attribute flips on the "
               "unchanged graph, not CFGT outputs"))

    # ---- BIND: its scripts' parser defaults plus the adapter's fixed rules
    b1 = parser_defaults(os.path.join(_repo("BIND-main"), "implementations", "1_training.py"))
    b3 = parser_defaults(os.path.join(_repo("BIND-main"), "implementations",
                                      "3_removing_and_testing.py"))
    for budget, p in (("1pct", 0.01), ("10pct", 0.10)):
        for ds in ("bail", "income"):
            out.append(dict(
                method=f"BIND-{budget}", backbone="GCN (BIND GNNs/gcn.py)", dataset=ds,
                protocol="x30", environment="x27_dgl_cuda",
                intervention_I="influence-guided deletion of training nodes "
                               "(helpfulness_collection=1 ranking)",
                M_plus_I=f"k = round({p} * |train|) most harmful training nodes deleted",
                M_minus_I="k = 0 (same Stage-A model, same retraining)",
                horizon_H=200, native_horizon=b3.get("epochs"), selector_controlled=SEL,
                native_selector="INVALID: the published fairness cost reads test labels",
                optimiser_and_hyperparameters=(
                    f"Stage A pretrain: lr={b1.get('lr')}, weight_decay={b1.get('weight_decay')}, "
                    f"hidden={b1.get('hidden')}, dropout={b1.get('dropout')}, "
                    f"epochs={b1.get('epochs')}; influence: scale=25, damp=0.03, "
                    f"recursion_depth=5000; Stage B retrain: lr={b3.get('lr')}, "
                    f"weight_decay={b3.get('weight_decay')}"),
                preprocessing="each official script's own (1_training full feature_norm; "
                              "2_influence keeps the sensitive column raw)",
                configuration_source="BIND-main parser defaults + README.md:55 (scale 25)",
                provenance="official-repo",
                caveat=("primary budget" if budget == "1pct" else
                        "budget variant, reported apart, never pooled")
                       + "; fairness cost moved to validation nodes (leakage removed, "
                         "estimator unchanged)"))
    out.extend(x31_rows())
    return out


def x31_rows():
    """Configuration rows of the additional cells; each carries its explicit
    `configuration` label so the bundle maps it to exactly one cell."""
    import x31_sfg as SF
    import x31_fnrgnn as FR
    import x31_fairvgnn_config_run as VG
    import x31_fairgnn_config_run as FGR
    from pilot_tau import METHODS
    out = []
    for ds in SF.DATASETS:
        c = SF.config(ds); p = c["plus"]
        hp = "; ".join(f"{k}={p[k]}" for k in ("hidden", "c_lr", "e_lr", "c_wd", "e_wd", "top_k",
                                                "clip_e", "ratio", "d_epochs", "g_epochs",
                                                "c_epochs", "K", "dropout") if k in p)
        for proto in ("x30", "x30native"):
            out.append(dict(
                method="SFG", configuration="default", backbone="SAGE", dataset=ds,
                protocol=proto, environment="dev (torch 2.6.0, PyG 2.8.0)",
                intervention_I="Lipschitz constraint on the encoder (rho) with the stronger "
                               "chi-square DRO generator loss",
                M_plus_I=f"with_constraint, rho={p['rho']}, loss_alpha={p['loss_alpha']}",
                M_minus_I="no constraint, loss_alpha=1 (the authors' FairVGNN row)",
                horizon_H=200 if proto == "x30" else p["epochs"], native_horizon=p["epochs"],
                selector_controlled=SEL,
                native_selector="the script's validation trade-off, strict >, floor 0 "
                                "(validation-only at the published horizon)",
                optimiser_and_hyperparameters=hp,
                preprocessing="german raw; others normalised with the sensitive column kept",
                configuration_source=f"SFG-main/run.sh ablation ladder, script {c['script']}",
                provenance="official-repo", caveat=""))
    for ds, labels in VG.CONFIGS.items():
        for lab in labels:
            cfg, meta = VG.config(ds, lab)
            for proto in ("x30", "x30native"):
                cc = VG.native(cfg, meta, ds) if proto == "x30native" else cfg
                hp = "; ".join(f"{k}={cc[k]}" for k in VG.KEYS if k in cc)
                if proto == "x30native" and "clip_c" in cc:
                    hp += f"; clip_c={cc['clip_c']}"
                out.append(dict(
                    method="FairVGNN", configuration=lab,
                    backbone="GCN" if lab.startswith("GCN") else lab, dataset=ds,
                    protocol=proto, environment="dev (torch 2.6.0, PyG 2.8.0)",
                    intervention_I="generative feature masking (f_mask) and weight clipping "
                                   "(weight_clip)",
                    M_plus_I="f_mask=yes, weight_clip=yes",
                    M_minus_I=", ".join(f"{k}={v}" for k, v in
                                        sorted(METHODS["FairVGNN"]["off"].items())),
                    horizon_H=200 if proto == "x30" else meta["native_epochs"],
                    native_horizon=meta["native_epochs"], selector_controlled=SEL,
                    native_selector="the method's own validation selection",
                    optimiser_and_hyperparameters=f"encoder={cfg['encoder']}; " + hp,
                    preprocessing=(f"feature_normalize={bool(cc.get('feature_normalize'))}"
                                   + ("" if proto == "x30" else
                                      f"; wrapper normalisation={cc.get('vg_wrapper_normalize')}")),
                    configuration_source=f"harness/provenance/fairvgnn_run_{ds}.sh "
                                         f"({lab} full row)",
                    provenance="official-repo",
                    caveat="configuration variant of the primary GCN row, reported apart"
                           + ("; credit trains with fairvgnn_credit.py's loop"
                              if proto == "x30native" and ds == "credit" else "")))
    bm = parser_defaults(os.path.join(_repo("BeMap-main"), "train_bemap.py"))
    for ds in ("bail", "credit", "pokec_z"):
        c = dict(bm, model="gat")
        out.append(dict(
            method="BeMap", configuration="GAT", backbone="GAT", dataset=ds, protocol="x30",
            environment="x27_dgl_cuda",
            intervention_I="balance-aware neighbour sampling: a fresh fair subgraph each epoch",
            M_plus_I="train_bemap.train(epoch, ...) (sampled subgraph)",
            M_minus_I="train_bemap.train(-1, ...) (full graph)",
            horizon_H=200, native_horizon="-", selector_controlled=SEL,
            native_selector="INVALID: the official loop selects on the test split",
            optimiser_and_hyperparameters="; ".join(f"{k}={c[k]}" for k in
                                                     ("model", "lr", "weight_decay", "hidden",
                                                      "dropout", "beta", "lam", "save_num")),
            preprocessing="BeMap feature_norm",
            configuration_source="BeMap-main train_bemap.py --model gat, parser defaults",
            provenance="official-repo",
            caveat="GAT keeps dropout active at evaluation (official forward); "
                   "backbone variant, reported apart"))
    for model in ("GCN", "GAT"):
        for ds in ("pokec_z", "pokec_n"):
            cfg, rec = FGR.config(ds, model)
            out.append(dict(
                method="FairGNN", configuration=f"upstream{model}", backbone=model, dataset=ds,
                protocol="x30",
                environment="dev (torch 2.6.0, PyG 2.8.0)" if model == "GCN" else "x27_dgl_cuda",
                intervention_I=CORE_I["FairGNN"][0],
                M_plus_I=f"alpha={cfg['alpha']:g}, beta={cfg['beta']:g}",
                M_minus_I="alpha=0, beta=0", horizon_H=200, native_horizon=rec["epochs"],
                selector_controlled=SEL, native_selector="not run (no FairGNN native cells)",
                optimiser_and_hyperparameters=f"lr={cfg['lr']:g}; weight_decay={cfg['weight_decay']:g}"
                    + ("" if model == "GCN" else
                       "; GAT_body: 1 layer, heads [1,1], hidden 64, feat_drop 0.5, attn_drop 0"),
                preprocessing="feature_normalize=False",
                configuration_source=f"FairGNN upstream src/scripts/{ds}/train_fair{model}.sh "
                                     "(harness/provenance/fairgnn_upstream)",
                provenance="official-repo",
                caveat="upstream configuration; the primary cell keeps utils/param.json; "
                       "recorded, not applied: --acc/--roc, --epochs, --sens_number"
                       + ("; --num-hidden 128 (the wrapper builds at 64)" if model == "GCN" else "")))
    for ds in FR.DATASETS:
        c = FR.config(ds)
        out.append(dict(
            method="FnRGNN", configuration="default", backbone="GCN", dataset=ds,
            protocol="x30", environment="x30_edits_env (geomloss)",
            intervention_I="edge reweighting by feature similarity and sensitive difference, "
                           "MMD representation alignment and GWN prediction normalisation",
            M_plus_I="use_edge_weight, use_mmd, use_gwn = True",
            M_minus_I="all three False (same GCN, unit edge weights)",
            horizon_H=200, native_horizon="-", selector_controlled=SEL,
            native_selector="none: no training loop released",
            optimiser_and_hyperparameters="; ".join(f"{k}={c[k]:.6g}" for k in FR.APPLIED)
                                          + f"; mmd_sample={c['mmd_sample']}",
            preprocessing="StandardScaler over all nodes' features (FnRGNN loader); "
                          "degree-0 nodes kept",
            configuration_source=f"FnRGNN-master/logs/best_configs/{c['config_file']}",
            provenance="official-repo class and configuration; harness-completed loop",
            caveat="released for node regression: MSE replaced by BCE, losses restricted to "
                   "training nodes, mmd_sample and H set by the harness; configuration tuned "
                   "for a regression target; reported apart, never counted as primary"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "final_results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "method_configurations.csv")
    rs = rows()
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER, extrasaction="ignore")
        w.writeheader()
        for r in rs:
            w.writerow(r)
    with open(p, newline="") as fh:
        rd = list(csv.reader(fh))
    bad = [i for i, r in enumerate(rd) if len(r) != len(HEADER)]
    if rd[0] != HEADER or bad:
        raise SystemExit(f"[config-table] malformed rows {bad}")
    print(f"[written] {p}  ({len(rs)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
