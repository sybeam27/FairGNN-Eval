"""Shared parsing, formatting and checking for the X28 publication tables.

No scientific value is defined here. Every number is parsed from a frozen
artifact. Two kinds of computation are allowed and no others:

  * the frozen resolved rule (sign stability >= 0.75 AND |mean| >= 0.010 AND the
    95% interval excluding 0) applied to frozen ingredients, and
  * differences that a frozen identity already defines.

The resolved computation is validated against the seven Arm A resolved flags
that armB_phase1_seven_cell_analysis.txt states explicitly; if any of those
seven disagrees, nothing is written.

The artifact parsers are imported from the X28 figure package rather than
rewritten, because those were verified by 401 checks when the figures were
built.
"""
from __future__ import annotations

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

ROOT = "/home/sypark/workspace/FairGate"
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments"))
sys.path.insert(0, os.path.join(ROOT, "harness", "experiments", "x28_figures"))

import common as FIG                                                 # noqa: E402
from analyze_armA import SIGN_MIN, NEAR_ZERO                         # noqa: E402

OUTDIR = os.path.join(ROOT, "harness/results/tables/paper")
SRCDIR = os.path.join(OUTDIR, "source_data")

ARMA_REPORT = f"{ROOT}/harness/results/armA_final_report.txt"
X24_SUMMARY = f"{ROOT}/harness/results/x24_nifty_german_summary.csv"

METHODS = ("FairGNN", "NIFTY", "FairGB", "FairVGNN")
DATASETS = ("german", "bail", "credit")
DS_LABEL = {"german": "German", "bail": "Bail", "credit": "Credit"}

# frozen X22/X23 classes; no class outside this set may appear in any table
TRANSFER_CLASSES = ("maintained", "relation changed", "resolution gained",
                    "resolution lost", "stable reversal")


# --------------------------------------------------------------- frozen parsing
_PD_ROW = re.compile(
    r"^(\w+)\s+(\w+)\s+"
    r"\[([+-][\d.]+), ([+-][\d.]+)\]\s+"
    r"\[([+-][\d.]+), ([+-][\d.]+)\]\s+"
    r"\[([+-][\d.]+), ([+-][\d.]+)\]\s+(yes|no)\s+([\d.]+)\s*$")


def parse_armA_per_dataset(path=ARMA_REPORT):
    """Section [4]: per-cell [dAUC, -dDP] means and the sign stability of tau_int.

    This is the only frozen artifact carrying sign stability for all 12
    controlled cells, and sign stability is one of the three ingredients of the
    frozen resolved rule.
    """
    rows, inside = [], False
    for line in open(path):
        if line.startswith("[4] per dataset"):
            inside = True
            continue
        if inside and line.startswith("[5]"):
            break
        m = _PD_ROW.match(line.rstrip("\n"))
        if inside and m:
            rows.append(dict(
                dataset=m.group(1), method=m.group(2),
                base_auc=float(m.group(3)), base_ndp=float(m.group(4)),
                int_auc=float(m.group(5)), int_ndp=float(m.group(6)),
                pkg_auc=float(m.group(7)), pkg_ndp=float(m.group(8)),
                base_gt_int=m.group(9) == "yes", sign_int_ndp=float(m.group(10))))
    d = pd.DataFrame(rows)
    if len(d) != 12:
        raise SystemExit(f"parsed {len(d)} rows from Arm A section [4], expected 12")
    return d


def resolved(mean, lo, hi, sign):
    """The frozen rule, unchanged: stable sign, non-trivial size, interval off 0."""
    return bool(sign >= SIGN_MIN and abs(mean) >= NEAR_ZERO and lo * hi > 0)


def x24():
    return pd.read_csv(X24_SUMMARY)


def qsel(d, selector, quantity, coord):
    r = d[(d.selector == selector) & (d.quantity == quantity) & (d.coord == coord)]
    if len(r) != 1:
        raise KeyError(f"{selector}/{quantity}/{coord}: {len(r)} rows")
    return r.iloc[0]


# ------------------------------------------------------------------- formatting
def num(v, nd=3, small=True):
    """A signed fixed-precision number; a nonzero value that would print as
    zero is shown as <0.001 so its sign is not silently lost."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    q = round(float(v), nd)
    if small and q == 0 and float(v) != 0:
        return ("+" if v > 0 else "-") + "$<$" + f"{10**-nd:.{nd}f}"
    return f"{q:+.{nd}f}"


def ci(lo, hi, nd=3):
    if lo is None or hi is None:
        return "--"
    return f"[{float(lo):+.{nd}f}, {float(hi):+.{nd}f}]"


def res(flag):
    return r"\textbf{R}" if flag else "u"


def esc(s):
    return (str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
            .replace("#", r"\#"))


# ------------------------------------------------------------------ latex emit
def latex_table(colspec, header, body_rows, caption, label, notes=(),
                width="single", midrules=()):
    """A booktabs table. `header` is a list of rows (each a list of cells)."""
    env = "table*" if width == "double" else "table"
    out = [f"% generated by harness/experiments/x28_tables/make_tables.py",
           f"% every number is parsed from a frozen artifact; do not edit by hand",
           rf"\begin{{{env}}}[t]", r"\centering", r"\small",
           rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    for h in header:
        out.append(" & ".join(h) + r" \\")
    out.append(r"\midrule")
    for i, row in enumerate(body_rows):
        if i in midrules:
            out.append(r"\midrule")
        out.append(" & ".join(row) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    # notes go inside the caption: a bare \\ after \caption sits outside any
    # tabular and makes LaTeX report "There's no line here to end"
    cap = caption
    if notes:
        cap = caption + r" {\footnotesize " + " ".join(notes) + "}"
    out.append(rf"\caption{{{cap}}}")
    out.append(rf"\label{{{label}}}")
    out.append(rf"\end{{{env}}}")
    return "\n".join(out) + "\n"


def _split_amp(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def detex(cell):
    """Strip LaTeX markup so a preview shows what the rendered table will say."""
    s = cell.strip()
    m = re.match(r"^\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$", s)
    if m:
        s = m.group(2)
    for cmd in ("textbf", "emph", "mathrm", "text"):
        s = re.sub(r"\\" + cmd + r"\{([^{}]*)\}", r"\1", s)
    for cmd in ("scriptsize", "footnotesize", "small"):
        s = s.replace("\\" + cmd, "")
    for a, b in ((r"\Delta", "D"), (r"\tau", "tau"), (r"\lambda", "lambda"),
                 (r"\sigma", "sigma"), (r"\geq", ">="), (r"\leq", "<="),
                 (r"\times", "x"), (r"\rightarrow", "->")):
        s = s.replace(a, b)
    s = s.replace("$", "").replace(r"\_", "_").replace(r"\%", "%")
    s = re.sub(r"[_^]\{([^{}]*)\}", r"_\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def rows_from_tex(tex):
    """The header and body of the generated table, as plain text."""
    body = tex.split(r"\begin{tabular}")[1].split(r"\end{tabular}")[0]
    header, rows, seen_mid = [], [], False
    for line in body.splitlines():
        t = line.strip()
        if t.startswith(r"\midrule"):
            seen_mid = True
            continue
        if not t.endswith(r"\\"):
            continue
        cells = []
        for c in _split_amp(re.sub(r"\\\\$", "", t)):
            m = re.search(r"\\multicolumn\{(\d+)\}", c)
            # a spanning cell occupies n columns: keep it in place and pad after
            # it, so a group label sits over the columns it actually covers
            cells += [detex(c)] + [""] * (int(m.group(1)) - 1 if m else 0)
        (rows if seen_mid else header).append(cells)
    return header, rows


def preview(tex, name, title):
    """A plain rendering of the table as generated.

    This is NOT LaTeX output -- no TeX toolchain is installed here. It previews
    the rows the .tex actually contains, not the full-precision source CSV, so
    what is checked by eye is what a reader would see.
    """
    header, rows = rows_from_tex(tex)
    ncol = max(len(r) for r in header + rows)
    pad = lambda r: r + [""] * (ncol - len(r))            # noqa: E731
    # header rows read top-down: the first is the column-label row of the
    # rendered table, any further ones sit beneath it
    labels = pad(header[0]) if header else [""] * ncol
    text = [pad(r) for r in header[1:]] + [pad(r) for r in rows]
    fig_w = min(26, max(6.0, 1.35 * ncol + 1.0))
    fig_h = max(1.6, 0.32 * (len(text) + 3))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    t = ax.table(cellText=text, colLabels=labels, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(7)
    t.auto_set_column_width(col=list(range(ncol)))
    t.scale(1, 1.3)
    ngroup = len(header) - 1
    for (r, c), cell in t.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#d8d8d8")
        if r == 0 or (ngroup and r <= ngroup):
            cell.set_text_props(weight="bold", color="#101010")
            cell.set_facecolor("#f2f2f0")
    ax.set_title(title + "   [preview of the generated table; not LaTeX output]",
                 fontsize=9, loc="left", color="#101010", pad=10)
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, f"{name}_preview.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def write(name, df, tex, title, outs):
    """CSV keeps full precision as the machine-readable source; the preview
    shows the rendered table."""
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(SRCDIR, exist_ok=True)
    csv_p = os.path.join(SRCDIR, f"{name}_source.csv")
    df.to_csv(csv_p, index=False)
    tex_p = os.path.join(OUTDIR, f"{name}.tex")
    with open(tex_p, "w") as fh:
        fh.write(tex)
    png_p = preview(tex, name, title)
    outs += [tex_p, csv_p, png_p]
    return tex_p, csv_p, png_p


# ---------------------------------------------------------------------- checks
class Checks(FIG.Checks):
    """Same fail-loudly semantics as the figure checks: a mismatch stops the
    run instead of writing a table over it."""

    def interval(self, what, mean, lo, hi):
        self.true(f"{what}: lo <= mean <= hi",
                  float(lo) <= float(mean) <= float(hi))

    def unique(self, what, df, cols):
        self.true(f"{what}: no duplicated {'/'.join(cols)}",
                  not df.duplicated(list(cols)).any())

    def complete(self, what, df, n):
        self.true(f"{what}: {n} rows, no missing cell",
                  len(df) == n and not df.isna().any().any())

    def klass(self, what, value):
        self.true(f"{what}: '{value}' is a frozen transfer class",
                  value in TRANSFER_CLASSES)

    def latex_matches_csv(self, what, tex, df, cols, nd=3):
        """Every formatted number in the chosen columns must appear in the .tex."""
        missing = []
        for c in cols:
            for v in df[c]:
                if isinstance(v, str) and v not in tex:
                    missing.append((c, v))
        self.true(f"{what}: all {sum(len(df[c]) for c in cols)} rendered cells "
                  f"present in the LaTeX source", not missing)
