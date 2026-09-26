"""
H7 — of each method's advantage over plain GCN, how much is the fairness
mechanism and how much is the backbone it ships with?

Finding 16 answered this for FairGate: its backbone at lambda_fair = 0 already
beats the audit's GNN by net +31 (dAUC +0.0800), while the whole fairness stack
beats that backbone by net **+0**. This generalises the question.

Three quantities per method, all Pareto on (AUC up, dP down), paired on
(setting, split_seed):

    total       method      vs plain GCN       reproduces the audit's row
    backbone    backbone    vs plain GCN       what the encoder alone buys
    mechanism   method      vs its backbone    what the fairness mechanism buys

`mechanism` is the one that means something. `total` is what the literature
reports, and it is `backbone` plus `mechanism` plus whatever else differs
between two codebases.

What counts as "its backbone" differs by method, and the difference is the
limit of this analysis rather than a detail:

    FairGate    E10's `backbone` arm — its own model at lambda_fair = 0
    NIFTY       NIFTY_off — sim_coeff = 0
    FairGNN     FairGNN_off — alpha = beta = 0
    FairGB      GNN_sage — a *plain* GraphSAGE, not FairGB with its mechanism
                removed. FairGB's fairness is a sampling augmentation with no
                coefficient to zero, so this is a matched-encoder control, not
                an ablation, and it also differs from FairGB in its training
                loop. Weaker inference, reported as weaker.
    FairGT      no control at all. Its mechanism is an adjacency transform and
                its backbone is a transformer with no plain counterpart here.
                Only `total` is computable and that is what is shown.

Usage
-----
    python harness/experiments/analyze_backbone_attribution.py
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_audit import dominates                      # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
CELL = ["setting", "split_seed"]
BASE = "GNN"

# method -> (its backbone's label, how that label is obtained)
BACKBONE = {"NIFTY":   ("NIFTY_off",   "ablation: sim_coeff = 0"),
            "FairGNN": ("FairGNN_off", "ablation: alpha = beta = 0"),
            "FairGB":  ("GNN_sage",    "matched encoder, NOT an ablation"),
            "FairGT":  (None,          "no control available"),
            "FairGate": ("__e10_backbone", "E10 arm: lambda_fair = 0")}


def paired(x: pd.DataFrame, y: pd.DataFrame) -> dict | None:
    m = x.merge(y, on=CELL, suffixes=("", "_b"))
    if m.empty:
        return None
    c = [dominates(r.roc_auc_mean, r.dp_mean, r.roc_auc_mean_b, r.dp_mean_b)
         for r in m.itertuples()]
    w, l = c.count(1), c.count(-1)
    return {"n": len(c), "w": w, "l": l, "inc": c.count(0), "net": w - l,
            "p": stats.binomtest(w, w + l, 0.5).pvalue if w + l else 1.0,
            "rate": w / len(c),
            "d_auc": float((m.roc_auc_mean - m.roc_auc_mean_b).mean()),
            "d_dp": float((m.dp_mean - m.dp_mean_b).mean())}


def row(tag: str, r: dict | None) -> None:
    if r is None:
        print(f"    {tag:<12s} (not computable)")
        return
    print(f"    {tag:<12s} n={r['n']:3d}  net={r['net']:+4d}  p={r['p']:.4f}  "
          f"rate={r['rate']:.3f}  inc={r['inc']:3d}  "
          f"dAUC={r['d_auc']:+.4f}  ddP={r['d_dp']:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=os.path.join(RESULTS, "e7_audit_merged.csv"))
    ap.add_argument("--e11", default=os.path.join(RESULTS, "e11_backbone.csv"))
    ap.add_argument("--ladder", default=os.path.join(RESULTS, "e10_ladder.csv"))
    a = ap.parse_args()

    A = pd.read_csv(a.audit)
    frames = [A]
    if os.path.exists(a.e11):
        frames.append(pd.read_csv(a.e11))
    D = pd.concat(frames, ignore_index=True)
    if "status" in D:
        D = D[D.status.fillna("ok").eq("ok")]            # NaN == resumed
    D = D.dropna(subset=["roc_auc_mean", "dp_mean"])

    if os.path.exists(a.ladder):
        L = pd.read_csv(a.ladder)
        if "status" in L:
            L = L[L.status.fillna("ok").eq("ok")]
        L = L.dropna(subset=["roc_auc_mean", "dp_mean"])
        bb = L[L.arm == "backbone"].assign(model="__e10_backbone")
        al = L[L.arm == "all_alloc"].assign(model="FairGate")
        D = pd.concat([D[D.model != "FairGate"], bb, al], ignore_index=True)

    base = D[D.model == BASE]
    print(f"\nplain GCN cells: {len(base)}")
    print("arms present: " + ", ".join(sorted(D.model.unique())) + "\n")

    print("=" * 100)
    print("Each method decomposed. `mechanism` is the fairness intervention with")
    print("the backbone held fixed; `total` is what a cross-codebase table reports.")
    print("=" * 100)
    for m, (bname, how) in BACKBONE.items():
        meth = D[D.model == m]
        if meth.empty:
            continue
        print(f"\n  {m}   (backbone control: {how})")
        row("total", paired(meth, base))
        if bname:
            bbf = D[D.model == bname]
            if not bbf.empty:
                row("backbone", paired(bbf, base))
                row("mechanism", paired(meth, bbf))
            else:
                print("    backbone     (arm not run yet)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
