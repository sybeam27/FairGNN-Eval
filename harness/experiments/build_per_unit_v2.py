"""The per-unit export for a rebuilt bundle, with the two columns C-1 asks for.

Same content and schema as `build_manifests.per_unit_metrics()` -- one row per (cell, split, run,
arm) with auc/dp/eo -- plus:

    seed      27 + run_id. Verified exact on all 3,900 frozen rows, so it is a derived column, not
              a new fact. It is written out because a reader should not have to know the rule.
    rng_rule  the seeding that path actually used, which is *not* uniform across the study:

                  manual_seed(seed)              B, and the pilot_tau method arms
                  seed_all(seed*1000 + split)    the x30/x31 method arms
                  seed_all(seed*1000 + split+1)  EDITS (x30_edits.py:131)

              pilot_tau.py:503 and :506 do seed per (seed, split), but train() at :109 reseeds with
              `seed` alone as its first statement, so for FairGNN, NIFTY, FairVGNN and FairGB the
              effective seeding is split-independent. Recorded, not fixed: the frozen results
              depend on it.

`backbone` and `configuration` are already separate columns upstream and are carried through
unchanged. When `--baseline` is given, B's rows carry the rebuilt baseline.

    python harness/experiments/build_per_unit_v2.py --baseline results_v2/baselines/B_rep1 \
        --out results_v2/bundle
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

# store_method prefix -> the seeding its runner used
PILOT_METHODS = {"FairGNN", "NIFTY", "FairVGNN", "FairGB"}
RULE_PILOT = "manual_seed(seed)"
RULE_X30 = "seed_all(seed*1000+split)"
RULE_EDITS = "seed_all(seed*1000+split+1)"
RULE_B = "manual_seed(seed)"


def rng_rule(method, state):
    if state == "B":
        return RULE_B
    if method == "EDITS":
        return RULE_EDITS
    return RULE_PILOT if method in PILOT_METHODS else RULE_X30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import build_results as B
    store = B.load_store()
    if a.baseline:
        store = B.swap_baseline(store, a.baseline)

    store = store.copy()
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
    for state, (auc, dp, eo) in arms.items():
        p = store[keep + [auc, dp, eo, "eo_defined"]].copy()
        p.columns = keep + ["auc", "dp", "eo", "eo_defined"]
        p.insert(len(keep), "state", state)
        parts.append(p)
    d = pd.concat(parts, ignore_index=True).sort_values(keep + ["state"])
    for c in ("auc", "dp", "eo"):
        d[c] = d[c].astype(float)

    d["seed"] = 27 + d["run_id"].astype(int)
    d["rng_rule"] = [rng_rule(m, s) for m, s in zip(d["method"], d["state"])]

    # seed = 27 + run is a claim about the store, so it is checked against it rather than assumed
    chk = store[["run_id", "seed"]].drop_duplicates()
    bad = chk[chk.seed != 27 + chk.run_id]
    if len(bad):
        raise SystemExit(f"[per-unit] seed != 27 + run_id on {len(bad)} distinct pairs:\n{bad}")

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "per_unit_metrics.csv.gz")
    d.to_csv(out, index=False, compression="gzip")
    print(f"per_unit_metrics.csv.gz: {len(d)} rows, "
          f"{d.groupby(['method', 'dataset', 'protocol', 'selector']).ngroups} cell-selector groups")
    print("  seed values:", sorted(d.seed.unique()))
    print("  rng_rule:", d.rng_rule.value_counts().to_dict())
    print("  baseline:", a.baseline or "frozen (per-method)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
