"""The manuscript's tables, generated against the Overleaf files rather than replacing them.

Until now every table lived only in `results/plot.ipynb`, which this repository does not carry, and
the Overleaf copies had drifted from it by hand: `table1_summary` was rewritten into a different
shape entirely, the per-cell tables gained `\\scriptsize` and a corrected footnote ("unit-level sign
consistency", not "sign stability"), and `tableS3b`/`tableS3c` carry the R2 captions. Regenerating
from the notebook would silently undo all of that.

So the Overleaf file is the template: **caption, footnotes, column spec and row order are copied
through unchanged, and only the numbers are substituted.** Each generator below states which cells
it fills and in what order; the row order is read from the template itself, so a row the template
does not have is never invented and one it has is never dropped.

Verification, run by `--check`: fill every template from the **pre-rebuild** bundle in
`results/superseded/` -- the one the Overleaf files were built from -- and diff against the
template. A difference is either a real disagreement or a rounding rule this file got wrong; either
way it is reported and nothing is written.

    python harness/experiments/build_tables.py --check                 # frozen -> templates
    python harness/experiments/build_tables.py --results results --out results/tables
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATE = os.path.join(ROOT, "paper", "table_template")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

COORDS = ["dAUC", "negDP", "negEO"]
FAMILY = {"Modification": ["BIND", "EDITS", "FairEdit"],
          "Constraint": ["FairGNN", "FairVGNN", "SFG"],
          "Regularization": ["BeMap", "FairGB", "FairSIN", "GEAR", "NIFTY"]}
FAM_OF = {m: f for f, ms in FAMILY.items() for m in ms}


def num(v, digits=3):
    """The manuscript's number format: a sign always, `digits` decimals, U+2212 never (LaTeX $-$)."""
    return f"{v:+.{digits}f}".replace("+", "+").replace("-", "-")


def primary(results_dir):
    d = pd.read_csv(os.path.join(results_dir, "cell_results.csv"))
    d = d[d.count_in_primary_summary == True].copy()          # noqa: E712
    d["family"] = d.method.map(FAM_OF)
    assert len(d) == 36, len(d)
    return d


# ---------------------------------------------------------------- table1_summary (tab:exp1-summary)
def table1_summary(results_dir, template):
    """Fill the 3x3 grid: rows Resolved / Package larger / Opposite sign / two medians, columns
    coordinate x family. The template's own row and column order is followed exactly."""
    import build_results as B
    d = primary(results_dir)
    order = ["Modification", "Constraint", "Regularization"]
    vals = {}
    for c in COORDS:
        for fam in order:
            g = d[d.family == fam]
            larger = (g[f"tau_nonint_{c}_mean"].abs() > g[f"tau_I_{c}_mean"].abs()).sum()
            res = g[f"tau_I_{c}_resolved"].astype(str).str.lower().eq("true").sum()
            # sign(pkg) * sign(tau_I) < 0, as the notebook had it: an estimate of exactly 0
            # (FairEdit, whose intervention is numerically inert) counts as neither sign
            opp = int(((np.sign(g[f"tau_pkg_{c}_mean"]) * np.sign(g[f"tau_I_{c}_mean"])) < 0).sum())
            vals[("Resolved", c, fam)] = str(int(res))
            vals[("Package larger", c, fam)] = str(int(larger))
            vals[("Opposite sign", c, fam)] = str(int(opp))
            vals[("med_nonint", c, fam)] = f"{g[f'tau_nonint_{c}_mean'].abs().median():.3f}"
            vals[("med_I", c, fam)] = f"{g[f'tau_I_{c}_mean'].abs().median():.3f}"
    rowkey = {"Resolved": "Resolved", "Package larger": "Package larger",
              "Opposite sign": "Opposite sign",
              r"Median \(|\tau_{B\rightarrow -I}|\)": "med_nonint",
              r"Median \(|\tau_{-I\rightarrow +I}|\)": "med_I"}
    out, i, lines = [], 0, template.split("\n")
    while i < len(lines):
        ln = lines[i]
        hit = next((k for k in rowkey if ln.strip() == k), None)
        if hit is None:
            out.append(ln); i += 1; continue
        out.append(ln)
        # the next three lines carry the nine values, three per coordinate
        body = [vals[(rowkey[hit], c, f)] for c in COORDS for f in order]
        for j, c in enumerate(COORDS):
            trio = body[j * 3:(j + 1) * 3]
            sep = r" \\" if j == 2 else ""
            out.append(("& " if j == 0 else "& ") + " & ".join(trio) + sep)
        i += 4                                   # skip the template's three value lines
    return "\n".join(out)


GENERATORS = {"table1_summary": table1_summary}

# ------------------------------------------------- table1_cells_{coord} (tab:exp1-cells-*)
ROW = re.compile(r"^([A-Za-z]+) & ([a-z_\\]+(?: \[[^\]]+\])?) & .*\\\\$")
DS_TEX = {"pokec\\_n": "pokec_n", "pokec\\_z": "pokec_z",
          "pokec\\_n\\_g": "pokec_n_g", "pokec\\_z\\_g": "pokec_z_g"}


def _cellrow(r, c):
    """The five numeric fields of one per-cell row, in the template's column order."""
    def f3(v):
        # a value that rounds to zero prints as +0.000, never -0.000: the sign of a rounded-away
        # magnitude is not information, and the manuscript writes it that way
        t = f"{v:+.3f}"
        return "+0.000" if t == "-0.000" else t

    def n(v):
        return f"${f3(v)}$"

    def ci(lo, hi):
        return f"[${f3(lo)}$, ${f3(hi)}$]"
    tI = f3(r[f'tau_I_{c}_mean'])
    bold = str(r[f"tau_I_{c}_resolved"]).strip().lower() == "true"
    tI_tex = (rf"$\mathbf{{{tI}}}$" if bold else f"${tI}$")
    return [n(r[f"tau_nonint_{c}_mean"]), ci(r[f"tau_nonint_{c}_lo"], r[f"tau_nonint_{c}_hi"]),
            tI_tex, ci(r[f"tau_I_{c}_lo"], r[f"tau_I_{c}_hi"]), n(r[f"tau_pkg_{c}_mean"])]


def table1_cells(coord):
    def gen(results_dir, template):
        d = primary(results_dir).set_index(["method", "dataset"])
        out = []
        for ln in template.split("\n"):
            m = ROW.match(ln.strip())
            if not m:
                out.append(ln); continue
            meth, dslab = m.group(1), m.group(2)
            ds = dslab.split(" [")[0]
            ds = DS_TEX.get(ds, ds).replace("\\_", "_")
            if (meth, ds) not in d.index:
                out.append(ln); continue
            r = d.loc[(meth, ds)]
            out.append(f"{meth} & {dslab} & " + " & ".join(_cellrow(r, coord)) + r" \\")
        return "\n".join(out)
    return gen


GENERATORS.update({f"table1_cells_{c}": table1_cells(c) for c in COORDS})



# ------------------------------------------------- tableS1_1_fnrgnn_two_tasks (tab:exp1-1-fnrgnn)
FNR_ROW = re.compile(r"^(?:\$[^$]+\$|)\s*&?\s*([a-z_\\]+) & .*\\\\$")


def tableS1_1_fnrgnn_two_tasks(results_dir, template):
    """Only the classification block is B-dependent; the regression block has its own baseline and
    is copied through. Rows are keyed by (coordinate, dataset) in the template's own order."""
    cls = pd.read_csv(os.path.join(results_dir, "1b_main_task_adapted.csv"))
    key = {}
    for r in cls.itertuples():
        for c in COORDS:
            key[(c, r.dataset)] = r
    coord_of = {r"$\Delta$AUC": "dAUC", r"$-\Delta_{\mathrm{DP}}$": "negDP",
                r"$-\Delta_{\mathrm{EO}}$": "negEO"}
    out, cur, in_reg = [], None, False
    for ln in template.split("\n"):
        t = ln.strip()
        if "Regression (released task" in t:
            in_reg = True
        for lbl, c in coord_of.items():
            if t.startswith(lbl + " &"):
                cur = c
        if in_reg or cur is None or not t.endswith(r"\\") or "&" not in t:
            out.append(ln); continue
        cells = [x.strip() for x in t.rstrip("\\").split("&")]
        if len(cells) != 8:
            out.append(ln); continue
        ds = cells[1].replace("\\_", "_")
        r = key.get((cur, ds))
        if r is None:
            out.append(ln); continue
        def n(v):
            x = f"{v:+.3f}"
            return "$+0.000$" if x == "-0.000" else f"${x}$"
        def ci(lo, hi):
            a, b = (f"{v:+.3f}" for v in (lo, hi))
            a = "+0.000" if a == "-0.000" else a
            b = "+0.000" if b == "-0.000" else b
            return f"[${a}$, ${b}$]"
        ti = f"{getattr(r, f'tau_I_{cur}_mean'):+.3f}"
        ti = "+0.000" if ti == "-0.000" else ti
        bold = str(getattr(r, f"tau_I_{cur}_resolved")).strip().lower() == "true"
        cells[2] = n(getattr(r, f"tau_nonint_{cur}_mean"))
        cells[3] = ci(getattr(r, f"tau_nonint_{cur}_lo"), getattr(r, f"tau_nonint_{cur}_hi"))
        cells[4] = rf"$\mathbf{{{ti}}}$" if bold else f"${ti}$"
        cells[5] = ci(getattr(r, f"tau_I_{cur}_lo"), getattr(r, f"tau_I_{cur}_hi"))
        cells[6] = n(getattr(r, f"tau_pkg_{cur}_mean"))
        cells[7] = ci(getattr(r, f"tau_pkg_{cur}_lo"), getattr(r, f"tau_pkg_{cur}_hi"))
        out.append(" & ".join(cells) + r" \\")
    return "\n".join(out)


GENERATORS["tableS1_1_fnrgnn_two_tasks"] = tableS1_1_fnrgnn_two_tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true",
                    help="fill each template from results/superseded/ (the pre-rebuild bundle the "
                         "templates were built from) and diff against it")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    if a.check and a.results == os.path.join(ROOT, "results"):
        sup = os.path.join(ROOT, "results", "superseded")
        if not os.path.isdir(sup):
            print("--check needs the pre-rebuild bundle, which this checkout does not publish "
                  "(results/ carries only what the paper reports).\n"
                  "Regenerate it with:\n"
                  "  python harness/experiments/build_results.py --out results/superseded\n"
                  "then re-run --check. Without --check the generators still run against any "
                  "bundle given by --results.")
            return 1
        a.results = sup
    names = a.only or sorted(GENERATORS)
    fails = 0
    for name in names:
        tpl_path = os.path.join(TEMPLATE, f"{name}.tex")
        if not os.path.exists(tpl_path):
            print(f"[{name}] no template at {tpl_path}"); fails += 1; continue
        tpl = open(tpl_path).read()
        got = GENERATORS[name](a.results, tpl)
        if a.check:
            same = got.strip() == tpl.strip()
            print(f"[{name}] {'reproduces the template' if same else 'DIFFERS from the template'}")
            if not same:
                fails += 1
                g, t = got.strip().split("\n"), tpl.strip().split("\n")
                for k, (x, y) in enumerate(zip(g, t)):
                    if x != y:
                        print(f"    line {k+1}\n      generated: {x}\n      template : {y}")
                if len(g) != len(t):
                    print(f"    line count {len(g)} vs {len(t)}")
        else:
            os.makedirs(a.out, exist_ok=True)
            open(os.path.join(a.out, f"{name}.tex"), "w").write(got)
            print(f"[{name}] written to {a.out}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
