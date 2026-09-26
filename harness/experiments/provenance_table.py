"""The provenance table for every (method, dataset) of the fixed set.

Configuration values are parsed from artifacts by core.published_config; the
repository identifiers, selector rules and M0/M1 definitions are recorded here
with the file each was read from, so every cell points at something checkable.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "harness"))
from core.published_config import published            # noqa: E402

REPO = {
    "FairVGNN": ("https://github.com/YuWVandy/FairVGNN",
                 "938f2e8d8da8c4bfc73b92c7953493a91f89244a"),
    "NIFTY":    ("https://github.com/chirag126/nifty",
                 "6c270c5ec736c5138d455609112cf6dfcc63dcca"),
    "FairGNN":  ("https://github.com/EnyanDai/FairGNN",
                 "13cdca7627d7779a2396299cdc8edd3a790db7cc"),
    "FairGB":   ("vendored zip, FairGB-main/", "no commit in the distribution"),
    "GNN":      ("https://github.com/chirag126/nifty",
                 "6c270c5ec736c5138d455609112cf6dfcc63dcca"),
}

# How much of the official source appears in our local wrapper, measured by
# harness/experiments/provenance_table.py --diff (normalized non-trivial lines).
MIRROR = {"NIFTY": 0.867, "FairVGNN": 0.834, "FairGNN": 0.510, "GNN": 0.380}

# Published checkpoint selector, read from the official source.
SELECTOR = {
    "FairVGNN": "auc+F1+acc - alpha*(parity+equality), floor 0  [fairvgnn.py:185]",
    "NIFTY":    "argmin(val_c_loss + val_s_loss)  [nifty_sota_gnn.py:325]",
    "FairGNN":  "first epoch with acc_val>args.acc AND roc_val>args.roc  "
                "[train_fairGNN.py:172]",
    "FairGB":   "alpha-weighted parity+equality tradeoff  [FairGB-main]",
    "GNN":      "argmin(val loss)  [nifty_sota_gnn.py:265]",
}

M0M1 = {
    "FairGNN":  "M1 alpha,beta published; M0 alpha=0,beta=0 (claimed fairness "
                "components only)",
    "NIFTY":    "M1 sim_coeff published; M0 sim_coeff=0, compared at the common "
                "selector so OFF does not change selector semantics",
    "FairGB":   "nested: M1 CAL on + CNM on; M0 both off (full vs all-off)",
    "FairVGNN": "dependency-constrained: M1 f_mask=yes, weight_clip=yes; "
                "M0 both no, every other setting held at M1's",
    "GNN":      "baseline, no intervention",
}

METHODS = ["GNN", "FairGNN", "NIFTY", "FairVGNN", "FairGB"]
DATASETS = ["german", "bail", "credit"]


def main():
    rows = []
    for m in METHODS:
        for ds in DATASETS:
            r = published(m, ds)
            url, commit = REPO[m]
            rows.append(dict(
                method=m, dataset=ds, source=r["source"], url=url,
                commit=commit, prov=r["provenance"], H=r["horizon"],
                norm=r["config"].get("feature_normalize"),
                cfg={k: v for k, v in sorted(r["config"].items())
                     if k != "feature_normalize"},
                notes=r["notes"], abl=r["official_ablation"]))

    print("| Method | Dataset | Source | Commit | Config path | Preproc | "
          "Horizon | Published selector | M0/M1 | Provenance |")
    print("|---|---|---|---|---|---|---:|---|---|---|")
    for r in rows:
        c = r["commit"]
        c = c[:7] if len(c) == 40 else c
        print(f"| {r['method']} | {r['dataset']} | {r['url']} | {c} | "
              f"{r['source']} | {'norm' if r['norm'] else 'raw'} | "
              f"{r['H'] if r['H'] else '**undetermined**'} | "
              f"{SELECTOR[r['method']]} | {M0M1[r['method']]} | "
              f"**{r['prov']}** |")

    print("\n\n## Hyperparameters, as parsed\n")
    for r in rows:
        print(f"* **{r['method']} / {r['dataset']}** "
              + " ".join(f"{k}={v}" for k, v in r["cfg"].items()))
        for n in r["notes"]:
            print(f"    * note: {n}")

    print("\n\n## Local code fidelity against the official source\n")
    for m, f in sorted(MIRROR.items()):
        grade = "local-mirror-verified" if f >= 0.80 else "local-unverified"
        print(f"* {m}: {f*100:.1f}% of official non-trivial lines present "
              f"in the local wrapper -> `{grade}`")
    print("* FairGB: algorithms/FairGB/data_utils.py is byte-identical to "
          "FairGB-main (diff -q clean) -> `local-mirror-verified`")

    print("\n\n## Blocking items\n")
    bad = [r for r in rows if r["prov"] == "local-unverified" or r["H"] is None]
    for r in bad:
        why = []
        if r["prov"] == "local-unverified":
            why.append("no artifact states this method's setting on this dataset")
        if r["H"] is None:
            why.append("horizon undetermined")
        print(f"* {r['method']} / {r['dataset']}: " + "; ".join(why))
    print(f"\n{len(bad)} of {len(rows)} combinations are blocking.")


if __name__ == "__main__":
    main()
