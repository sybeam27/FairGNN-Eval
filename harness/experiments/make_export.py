"""Build the anonymous public checkout, from this working tree rather than from git.

Git is not the source: several files this study needs were never committed, because `.gitignore`
excludes every runner (`:49`), all of `models/` (`:9`), `harness/provenance/` (`:48`) and
`figures/src/` (`:66`). A clone of the repository cannot execute a single one of the 36 primary
cells. The export is therefore assembled from the working tree by an explicit manifest, and the
result carries no git history at all.

Anonymisation is part of the copy, not a later pass: author names, e-mail addresses and absolute
user paths are rewritten as each file is written, and `--check` greps the result for anything that
survived.

    python harness/experiments/make_export.py --out ../FairGNN-Eval-export
    python harness/experiments/make_export.py --out ../FairGNN-Eval-export --check
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directories copied whole, minus the excludes below.
TREES = [
    "harness/core", "harness/adapters", "harness/experiments", "harness/provenance",
    "models/algorithms", "figures/src", "utils",
    "results/phase0_audit", "results/tables", "results/superseded",
    # Stage B: the two vendored upstreams whose MIT licence permits redistribution. Their
    # dataset/ and figures/ directories are left out -- 226 MB and 13 MB of data that the fetch
    # script retrieves instead.
    "models/BeMap-main", "models/FairGB-main",
]
# Individual files.
# src -> dst, where the exported name differs from the source name
RENAMED = {
    "paper/export_README.md": "README.md",
    "harness/experiments/fetch_upstream_template.sh": "fetch_upstream.sh",
    "patches/fairgb_local_edits.patch": "patches/fairgb_local_edits.patch",
}
FILES = [
    "phase0_verify.py", "LICENSE", "LICENSE-DATA", "CHANGELOG.md",
    "PAPER_ARTIFACT_MAP.md",
    "harness/coverage_manifest.csv", "harness/intervention_manifest.csv",
    "harness/score_convention_manifest.csv", "harness/external_repos.tsv",
    "harness/METHOD_INVENTORY.csv", "harness/METHOD_EXTENSION_INVENTORY.csv",
    "results/per_unit_metrics.csv.gz", "results/README.md", "results/experiment_index.csv",
    "results/cell_results.csv", "results/coverage.csv", "results/method_configurations.csv",
    "results/model_dataset_feasibility.csv",
    "results/paper_numbers.csv", "results/B_rebuild_diff.md",
    "results/C18_FINAL_REPORT.md", "results/phase0_verify_v2.txt",
]
# Glob-ish additions: the frozen section tables and the rebuilt bundle.
# the nine figures the submitted version prints; the generators also write variants that this
# version does not use, and shipping those invites a reviewer to look for them in the paper
PAPER_FIGURES = (r"^(fig1_intervention_attribution|fig3_protocol_variation|"
                 r"fig3_selection_support_trajectory|fig4_selection_support_bridge|"
                 r"fig5_fmp_component_bridge|figS1_intervention_attribution_negEO|"
                 r"figS4_fixed_epoch_trajectory|figS5_fmp_component_bridge_eo_auc|"
                 r"figS5_fmp_fair_by_selector)\.pdf$")
GLOBS = [("results", r"^\d[a-z]?_.*\.csv$"),
         ("figures", PAPER_FIGURES), ("paper/table_template", r".*\.(tex|md)$"),
         ("results/appendix_robustness", r".*\.csv$")]

EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", "preview", ".git",
                "dataset", "figures", "Figures"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".npz", ".pt", ".pth", ".zip"}
# this builder carries the anonymisation patterns as literals, so it would fail its own check;
# a reviewer does not need the tool that produced the checkout
EXCLUDE_FILES = {"harness/experiments/make_export.py"}

# Anonymisation. Order matters: the longest patterns first.
SUBS = [
    (re.compile(r"/home/sypark/workspace/FairGNN-Eval"), "."),
    (re.compile(r"/home/sypark/workspace/FairGate"), "."),
    (re.compile(r"/home/sypark/miniconda3/envs/dev/bin/python"), "python"),
    (re.compile(r"/home/sypark/x27_dgl_cuda/bin/python"), "python-dgl"),
    (re.compile(r"/home/sypark/x30_edits_env/bin/python"), "python-edits"),
    (re.compile(r"/home/sypark/[A-Za-z0-9_./-]*"), "<path>"),
    (re.compile(r"sypark1452@o\.cnu\.ac\.kr"), "<anonymised>"),
    (re.compile(r"wlsl89822@gmail\.com"), "<anonymised>"),
    (re.compile(r"\bsybeam27\b"), "<anonymised>"),
    (re.compile(r"\bsypark\b"), "<anonymised>"),
    (re.compile(r"github\.com/sybeam27/[A-Za-z0-9_.-]+"), "<anonymised-mirror>"),
    (re.compile(r"\bFairGate\b"), "FairGNN-Eval"),
]
TEXT_SUFFIX = {".py", ".md", ".txt", ".tex", ".csv", ".tsv", ".sh", ".json", ".yml", ".cfg"}

# Strings that must not survive; --check greps for these.
FORBIDDEN = [r"/home/sypark", r"sypark1452", r"wlsl89822", r"sybeam27", r"\bFairGate\b"]


def anonymise(text):
    for pat, rep in SUBS:
        text = pat.sub(rep, text)
    return text


def copy_one(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ext = os.path.splitext(src)[1].lower()
    if ext in TEXT_SUFFIX:
        try:
            with open(src, encoding="utf-8") as f:
                t = f.read()
        except (UnicodeDecodeError, ValueError):
            shutil.copy2(src, dst); return "binary"
        with open(dst, "w", encoding="utf-8") as f:
            f.write(anonymise(t))
        return "text"
    shutil.copy2(src, dst)
    return "binary"


def walk(rel):
    base = os.path.join(ROOT, rel)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in EXCLUDE_SUFFIX:
                continue
            p = os.path.join(dirpath, fn)
            r = os.path.relpath(p, ROOT)
            if r in EXCLUDE_FILES:
                continue
            yield p, r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="grep the export for identifying strings")
    a = ap.parse_args()
    out = os.path.abspath(a.out)

    if not a.check:
        if os.path.exists(out):
            shutil.rmtree(out)
        n_text = n_bin = 0
        for rel in TREES:
            if not os.path.isdir(os.path.join(ROOT, rel)):
                print(f"  [missing tree] {rel}"); continue
            for src, r in walk(rel):
                kind = copy_one(src, os.path.join(out, r))
                n_text += kind == "text"; n_bin += kind == "binary"
        for src_rel, dst_rel in RENAMED.items():
            src = os.path.join(ROOT, src_rel)
            if not os.path.exists(src):
                print(f"  [missing renamed] {src_rel}"); continue
            kind = copy_one(src, os.path.join(out, dst_rel))
            n_text += kind == "text"; n_bin += kind == "binary"
        for rel in FILES:
            src = os.path.join(ROOT, rel)
            if not os.path.exists(src):
                print(f"  [missing file] {rel}"); continue
            kind = copy_one(src, os.path.join(out, rel))
            n_text += kind == "text"; n_bin += kind == "binary"
        for rel, pat in GLOBS:
            d = os.path.join(ROOT, rel)
            if not os.path.isdir(d):
                print(f"  [missing dir] {rel}"); continue
            rx = re.compile(pat)
            for fn in sorted(os.listdir(d)):
                if rx.match(fn) and os.path.isfile(os.path.join(d, fn)):
                    kind = copy_one(os.path.join(d, fn), os.path.join(out, rel, fn))
                    n_text += kind == "text"; n_bin += kind == "binary"
        print(f"export written to {out}: {n_text} text files anonymised, {n_bin} copied verbatim")
        print("no .git directory is created, so no history is carried")

    # ---- the check runs either way -----------------------------------------------------
    hits = []
    for dirpath, dirnames, filenames in os.walk(out):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if os.path.splitext(fn)[1].lower() not in TEXT_SUFFIX:
                continue
            try:
                t = open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, ValueError):
                continue
            for pat in FORBIDDEN:
                for m in re.finditer(pat, t):
                    line = t[:m.start()].count("\n") + 1
                    hits.append((os.path.relpath(p, out), line, m.group(0)))
    print(f"\nanonymisation check: {len(hits)} identifying string(s) remain")
    for h in hits[:40]:
        print(f"  {h[0]}:{h[1]}  {h[2]!r}")
    if os.path.exists(os.path.join(out, ".git")):
        print("  .git PRESENT -- history not removed")
        return 1
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
