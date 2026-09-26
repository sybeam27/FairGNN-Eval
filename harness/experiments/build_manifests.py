"""Audit manifests: why each cell is in the study, what the intervention toggles, and what a score means.

Three files that let a reader check the paper's boundaries without re-running anything. They are
derived from the frozen ledgers, never written by hand:

    harness/coverage_manifest.csv           every method x dataset candidate, with the eligibility
                                            decision and, when omitted, a fixed exclusion category
    harness/intervention_manifest.csv       what M^{-I} and M^{+I} differ by, and what they share
    harness/score_convention_manifest.csv   the canonical decision margin of each implementation

and one compact export for auditability:

    results/per_unit_metrics.csv.gz         the matched per-unit outcomes (split x run x arm) the
                                            reported contrasts and intervals are computed from

    python harness/experiments/build_manifests.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results")
HARNESS = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# Fixed exclusion categories. A cell is either included, or omitted for exactly one of these
# reasons; the ledger's own wording is carried through in `notes`, so nothing is lost.
CATEGORIES = [
    "included",
    "included_reported_separately",          # in the study, but not part of the primary set
    "unsupported_released_configuration",    # the released code configures no comparable setting
    "required_artifact_unavailable",         # released assets a matched run would need are missing
    "matched_intervention_not_separable",    # no off-state exists, so M^{-I} cannot be constructed
    "invalid_evaluation_split",              # the released setting breaks train/val/test separation
]

# note text -> category. Every excluded row must match exactly one rule; the script stops otherwise.
RULES = [
    ("validation set is contained in the test set", "invalid_evaluation_split"),
    ("no off-state", "matched_intervention_not_separable"),
    ("no released assets", "required_artifact_unavailable"),
    ("different graph", "unsupported_released_configuration"),
    ("subgraph, not the common", "unsupported_released_configuration"),
]

# One row per implementation: what the classifier's loss proves the output to be, and the margin the
# evaluator thresholds. Evidence is the frozen audit document plus the adapter that records it.
SCORE_CONVENTIONS = [
    # method, classifier loss, raw output, canonical margin, decision rule, positive class, evidence
    ("GNN (baseline)", "binary_cross_entropy_with_logits", "scalar_logit", "z", "q > 0", 1,
     "harness/X2_ADAPTER_CONTRACT.md; harness/core/model.py"),
    ("FairGNN", "nn.BCEWithLogitsLoss", "scalar_logit", "z", "q > 0", 1, "harness/X2_ADAPTER_CONTRACT.md"),
    ("NIFTY", "binary_cross_entropy_with_logits", "scalar_logit", "z", "q > 0", 1, "harness/X2_ADAPTER_CONTRACT.md"),
    ("FairVGNN", "binary_cross_entropy_with_logits", "scalar_logit", "z", "q > 0", 1,
     "harness/X2_ADAPTER_CONTRACT.md; harness/adapters/fairvgnn_credit_native.py"),
    ("FairGB", "binary_cross_entropy_with_logits", "scalar_logit", "z", "q > 0", 1, "harness/X2_ADAPTER_CONTRACT.md"),
    ("FairSIN", "binary_cross_entropy_with_logits", "scalar_logit", "z", "q > 0", 1,
     "harness/X2_ADAPTER_CONTRACT.md; harness/adapters/fairsin.py"),
    ("BIND", "binary_cross_entropy_with_logits", "scalar_logit", "z", "q > 0", 1,
     "harness/adapters/bind.py; harness/adapters/x30_bind.py"),
    ("EDITS", "binary_cross_entropy_with_logits", "scalar_logit", "z", "q > 0", 1, "harness/adapters/x30_edits.py"),
    ("FairEdit", "binary_cross_entropy_with_logits", "scalar_logit", "z", "q > 0", 1, "harness/adapters/x30_fairedit.py"),
    ("BeMap", "binary_cross_entropy_with_logits", "scalar_logit", "z", "q > 0", 1, "harness/adapters/x30_bemap.py"),
    ("GEAR", "binary_cross_entropy_with_logits", "scalar_logit", "z", "q > 0", 1, "harness/adapters/x30_gear.py"),
    ("SFG", "binary_cross_entropy_with_logits (classifier; nn.BCELoss is the discriminator)",
     "scalar_logit", "z", "q > 0", 1, "harness/adapters/x31_sfg.py"),
    ("FnRGNN", "BCEWithLogitsLoss (harness-completed loop; MSE on its released regression task)",
     "scalar_logit", "z", "q > 0", 1, "harness/adapters/x31_fnrgnn.py; harness/X31_ADDITIONAL_CELLS_PROTOCOL.md"),
    ("FMP", "CrossEntropyLoss", "two_class_logits", "z1 - z0", "q > 0", 1,
     "harness/X17_FMP_AUDIT.md; harness/experiments/x27_fmp_run.py"),
]


def category(row) -> tuple[str, str]:
    """(category, primary_eligible) for one method x dataset candidate."""
    st = str(row.primary_controlled)
    if st == "done":
        return "included", "yes"
    if st in ("done, task-adapted", "component study"):
        return "included_reported_separately", "no"
    note = str(row.note).lower()
    hit = [c for k, c in RULES if k in note]
    if len(hit) != 1:
        raise SystemExit(f"[manifest] no single exclusion rule for {row.method}/{row.dataset}: {row.note!r}")
    return hit[0], "no"


def coverage_manifest() -> pd.DataFrame:
    feas = pd.read_csv(os.path.join(RESULTS, "model_dataset_feasibility.csv"))
    cov = pd.read_csv(os.path.join(RESULTS, "coverage.csv"))
    prim = cov[cov.configuration_role.eq("primary") & cov.protocol.eq("controlled")]
    done = {(r.method, r.dataset): r for r in prim.itertuples()}

    methods, datasets = sorted(feas.method.unique()), sorted(feas.dataset.unique())
    known = {(r.method, r.dataset): r for r in feas.itertuples()}
    rows = []
    for m in methods:
        for ds in datasets:
            r = known.get((m, ds))
            if r is None:          # the released code configures no such pair at all
                cat, elig, src, note, status = ("unsupported_released_configuration", "no", "",
                                                "not configured by the released code", "not_established")
            else:
                cat, elig = category(r)
                src = str(r.configuration_source) if pd.notna(r.configuration_source) else ""
                note = str(r.note) if pd.notna(r.note) else ""
                status = "valid" if cat.startswith("included") else "not_established"
            c = done.get((m, ds))
            rows.append(dict(
                method=m, dataset=ds,
                primary_eligible=elig,
                included="yes" if cat.startswith("included") else "no",
                configuration=(c.configuration if c is not None else ""),
                backbone=(c.backbone if c is not None else ""),
                units_done=(int(c.units_done) if c is not None else 0),
                configuration_source=src,
                matched_construction_status=status,
                exclusion_reason=("" if cat == "included" else cat),
                notes=note))
    d = pd.DataFrame(rows)
    assert len(d) == len(methods) * len(datasets)
    assert set(d.exclusion_reason) <= set(CATEGORIES) | {""}
    assert int((d.included == "yes").sum()) == len(feas[feas.primary_controlled != "excluded"])
    return d


def intervention_manifest() -> pd.DataFrame:
    cfg = pd.read_csv(os.path.join(RESULTS, "method_configurations.csv"))
    c = cfg[cfg.protocol.eq("controlled")].copy()
    out = c.rename(columns={
        "intervention_I": "claimed_intervention",
        "M_minus_I": "m_minus_I_state",
        "M_plus_I": "m_plus_I_state",
        "backbone": "shared_backbone",
        "optimizer_and_hyperparameters": "shared_optimizer_and_hyperparameters",
        "preprocessing": "shared_preprocessing",
        "horizon": "shared_horizon",
        "selector": "shared_checkpoint_selector",
        "configuration_source": "source_file_or_command",
        "caveat": "qualification"})
    cols = ["method", "dataset", "configuration", "shared_backbone", "claimed_intervention",
            "m_minus_I_state", "m_plus_I_state", "shared_horizon", "shared_checkpoint_selector",
            "shared_optimizer_and_hyperparameters", "shared_preprocessing", "source_file_or_command",
            "provenance", "qualification"]
    out = out[cols].sort_values(["method", "dataset", "configuration"])
    # the split, the seed assignment and the evaluator are shared by construction, not per row
    out.insert(len(cols), "shared_split_and_seeds", "6 splits (20-25) x 5 runs, seed = 27 + run, paired M0/M1")
    out.insert(len(cols) + 1, "shared_evaluator", "harness/core/evaluator.py, decision q > 0")
    return out


def per_unit_metrics() -> pd.DataFrame:
    """The matched per-unit outcomes: one row per (cell, split, run, arm), metrics only.

    The raw store encodes the configuration inside the method name (BIND-1pct, FairSIN-GCN, ...) and
    writes the literal backbone "GCN" for every method (pilot_tau.py:547), which is wrong for FairGB
    (SAGE) among others. Here the name is split into method + configuration and the backbone is taken
    from the same canonical map the section tables use, so this file joins 1:1 with cell_results.csv on
    (method, dataset, backbone, configuration, protocol). The stored name is kept as `store_method`.
    """
    import build_results as B
    store = B.load_store().copy()
    store["store_method"] = store["method"]
    split = [B.split_name(m) for m in store["method"]]
    store["method"] = [m for m, _ in split]
    store["configuration"] = [c for _, c in split]
    store["backbone"] = [B.canonical_backbone(m, c) for m, c in split]
    keep = ["method", "dataset", "backbone", "configuration", "protocol", "selector",
            "split_id", "run_id", "store_method"]
    arms = {"B": ("bc_auc", "bc_dp", "bc_eo"), "M_minus_I": ("m0_auc", "m0_dp", "m0_eo"),
            "M_plus_I": ("m1_auc", "m1_dp", "m1_eo")}
    parts = []
    for state, (a, dp, eo) in arms.items():
        p = store[keep + [a, dp, eo, "eo_defined"]].copy()
        p.columns = keep + ["auc", "dp", "eo", "eo_defined"]
        p.insert(len(keep), "state", state)
        parts.append(p)
    d = pd.concat(parts, ignore_index=True).sort_values(keep + ["state"])
    for c in ("auc", "dp", "eo"):
        d[c] = d[c].astype(float)          # full precision: rounding manufactured ties in FairEdit/credit
    return d


def main():
    cov = coverage_manifest()
    cov.to_csv(os.path.join(HARNESS, "coverage_manifest.csv"), index=False)
    print("coverage_manifest.csv:", len(cov), "candidates,",
          int((cov.included == "yes").sum()), "included,",
          int((cov.primary_eligible == "yes").sum()), "primary")
    print(cov[cov.included == "no"].exclusion_reason.value_counts().to_string())

    iv = intervention_manifest()
    iv.to_csv(os.path.join(HARNESS, "intervention_manifest.csv"), index=False)
    print("\nintervention_manifest.csv:", len(iv), "controlled cells")

    sc = pd.DataFrame(SCORE_CONVENTIONS, columns=[
        "method", "classifier_loss", "raw_output", "canonical_margin", "decision_rule",
        "positive_class", "evidence_path"])
    for p in {q.strip() for row in sc.evidence_path for q in row.split(";")}:
        if not os.path.exists(os.path.join(ROOT, p)):
            raise SystemExit(f"[manifest] evidence path missing: {p}")
    sc.to_csv(os.path.join(HARNESS, "score_convention_manifest.csv"), index=False)
    print("score_convention_manifest.csv:", len(sc), "implementations")

    pu = per_unit_metrics()
    out = os.path.join(RESULTS, "per_unit_metrics.csv.gz")
    pu.to_csv(out, index=False, compression="gzip")
    print(f"\nper_unit_metrics.csv.gz: {len(pu)} rows, "
          f"{pu.groupby(['method', 'dataset', 'protocol', 'selector']).ngroups} cell-selector groups, "
          f"{os.path.getsize(out) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
