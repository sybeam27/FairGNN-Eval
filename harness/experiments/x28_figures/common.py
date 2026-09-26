"""Shared style, source-data emission and frozen-value checking for the X28
paper figure candidates.

Nothing scientific is defined in this module. The resolution rule, the cell
construction and the bootstrap come from the frozen modules (analyze_armA,
bootstrap_armA); the X25/X26/X27 summaries are read exactly as written.

Every figure plots numbers parsed from a frozen artifact, writes a
<name>_source.csv holding exactly those numbers, and registers checks that
re-derive them from the raw per-cell CSVs. A figure that cannot reproduce its
own numbers fails loudly instead of rendering.
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

OUTDIR = os.path.join(ROOT, "harness/results/figures/paper")
SRCDIR = os.path.join(OUTDIR, "source_data")

# ---------------------------------------------------------------- frozen inputs
ARMA_CSVS = [f"{ROOT}/harness/results/armA_german.csv",
             f"{ROOT}/harness/results/armA_bail.csv",
             f"{ROOT}/harness/results/armA_bail_s23_25.csv",
             f"{ROOT}/harness/results/armA_credit.csv",
             f"{ROOT}/harness/results/armA_credit_s23_25.csv",
             f"{ROOT}/harness/results/armA_fairvgnn_german.csv",
             f"{ROOT}/harness/results/armA_fairvgnn_bail.csv",
             f"{ROOT}/harness/results/armA_fairvgnn_credit.csv"]
ARMA_BOOT_TXT = f"{ROOT}/harness/results/armA_final_bootstrap.txt"
ARMB_SEVEN_TXT = f"{ROOT}/harness/results/armB_phase1_seven_cell_analysis.txt"
X25_SUMMARY = f"{ROOT}/harness/results/x25/x25_summary.csv"
X26_SUMMARY = f"{ROOT}/harness/results/x26/x26_bail_summary.csv"
X27_SUMMARY = {ds: f"{ROOT}/harness/results/x27/x27_{ds}_summary.csv"
               for ds in ("pokec_z", "pokec_n")}

METHODS = ("FairGB", "FairGNN", "FairVGNN", "NIFTY")
DATASETS = ("german", "bail", "credit")

# ------------------------------------------------------------------ appearance
# Categorical slots are assigned in this fixed order and never cycled. Colour is
# never the only channel: every method also carries its own marker, and every
# vector kind its own line style.
SURFACE = "#ffffff"
INK, INK2, MUTED = "#101010", "#4a4a4a", "#8a8a8a"
GRIDC, BASEC = "#e6e6e6", "#bfbfbf"
C = {"FairGB": "#2a78d6", "FairGNN": "#eb6834",
     "FairVGNN": "#1baf7a", "NIFTY": "#9a6ad6"}
MK = {"FairGB": "o", "FairGNN": "s", "FairVGNN": "^", "NIFTY": "D"}
NEUTRAL = "#9a9a9a"

FS = dict(base=8, tick=7.5, label=8.5, panel=9, title=9.5, note=7)


def rc():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FS["base"],
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # real vector text, not paths
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def style(ax, grid="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASEC)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, width=0.8, length=2.5,
                   labelsize=FS["tick"])
    if grid:
        kw = dict(color=GRIDC, linewidth=0.7, linestyle="-")
        ax.grid(axis=grid if grid != "both" else "both", **kw)
    ax.set_axisbelow(True)


def zero_lines(ax, x=True, y=True):
    """The zero reference is drawn explicitly on every effect-space axis."""
    if y:
        ax.axhline(0, color=BASEC, linewidth=0.9, zorder=1)
    if x:
        ax.axvline(0, color=BASEC, linewidth=0.9, zorder=1)


def quadrant_labels(ax, fontsize=None, inset=0.045):
    """Four very faint corner tags: what each quadrant of the effect space means.

    Drawn in axes fraction so they never move with the data, inset from the
    corners so they clear tick labels, and kept far lighter than any mark so
    they read as background, not as a finding. On a multi-panel figure the axes
    meaning is shared, so call this on one panel only.
    """
    fs = fontsize or FS["note"] - 1.0
    i = inset
    tags = [(1 - i, 1 - i, "right", "top", "utility ↑\nfairness ↑"),
            (i, 1 - i, "left", "top", "utility ↓\nfairness ↑"),
            (1 - i, i, "right", "bottom", "utility ↑\nfairness ↓"),
            (i, i, "left", "bottom", "utility ↓\nfairness ↓")]
    for x, y, ha, va, t in tags:
        ax.text(x, y, t, transform=ax.transAxes, ha=ha, va=va,
                fontsize=fs, color="#d2d2d2", linespacing=1.1, zorder=0)


def errcross(ax, x, y, xlo, xhi, ylo, yhi, color, lw=0.9, alpha=1.0, zorder=4):
    """Compact 95% interval cross at a vector endpoint."""
    ax.plot([xlo, xhi], [y, y], color=color, linewidth=lw, alpha=alpha,
            solid_capstyle="butt", zorder=zorder)
    ax.plot([x, x], [ylo, yhi], color=color, linewidth=lw, alpha=alpha,
            solid_capstyle="butt", zorder=zorder)


def resolved_marker(ax, x, y, color, mk, resolved, size=6.5, zorder=6, lw=1.4):
    """Filled = resolved, hollow = unresolved. Never colour alone."""
    ax.plot(x, y, marker=mk, markersize=size, linestyle="none",
            markerfacecolor=color if resolved else SURFACE,
            markeredgecolor=color, markeredgewidth=lw, zorder=zorder)


def save(fig, name, outs):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUTDIR, f"{name}.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None)
        outs.append(p)
    plt.close(fig)


# -------------------------------------------------------------- source data + checks
class Source:
    """Collects exactly the numbers a figure plots, then writes them beside it."""

    def __init__(self, name):
        self.name = name
        self.rows = []

    def add(self, **kw):
        self.rows.append(kw)
        return kw

    def write(self):
        os.makedirs(SRCDIR, exist_ok=True)
        p = os.path.join(SRCDIR, f"{self.name}_source.csv")
        pd.DataFrame(self.rows).to_csv(p, index=False)
        return p


class Checks:
    """Programmatic comparison of plotted coordinates against frozen values."""

    def __init__(self):
        self.items = []
        self.worst = 0.0

    def close(self, what, got, want, tol=1e-4):
        d = abs(float(got) - float(want))
        self.worst = max(self.worst, d)
        ok = d <= tol
        self.items.append((what, ok, d))
        if not ok:
            raise SystemExit(f"[check FAILED] {what}: plotted {got:+.6f} vs "
                             f"frozen {want:+.6f} (|d| {d:.2e} > {tol:g})")
        return ok

    def true(self, what, cond):
        self.items.append((what, bool(cond), 0.0))
        if not cond:
            raise SystemExit(f"[check FAILED] {what}")

    def limits(self, ax, xs, ys, what):
        """Assert every plotted coordinate, interval end included, is drawn."""
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xs = [v for v in np.ravel(xs) if np.isfinite(v)]
        ys = [v for v in np.ravel(ys) if np.isfinite(v)]
        bad = ([v for v in xs if not (min(x0, x1) <= v <= max(x0, x1))]
               + [v for v in ys if not (min(y0, y1) <= v <= max(y0, y1))])
        self.true(f"{what}: all {len(xs) + len(ys)} coordinates inside axes limits",
                  not bad)

    def report(self):
        n = len(self.items)
        bad = [i for i in self.items if not i[1]]
        print(f"  [checks] {n - len(bad)}/{n} passed, worst |difference| "
              f"{self.worst:.2e}")
        return not bad


# ------------------------------------------------------------- frozen artifact parsers
_ARMA_ROW = re.compile(
    r"^(\S+)\s+(tau_base\^audit|tau_int|tau_pkg\^audit|D_selector)\s+"
    r"(dAUC|-dDP)\s+([+-][\d.]+)\s+\[([+-][\d.]+), ([+-][\d.]+)\]\s+(yes|no)\s*$")


def parse_armA_bootstrap(path=ARMA_BOOT_TXT):
    """The frozen 12-cell controlled bootstrap table, exactly as written."""
    rows, ds, prev = [], None, ""
    for line in open(path):
        line = line.rstrip("\n")
        if set(line.strip()) == {"="} and prev.strip() in DATASETS:
            ds = prev.strip()
        m = _ARMA_ROW.match(line)
        if m and ds:
            rows.append(dict(dataset=ds, method=m.group(1), quantity=m.group(2),
                             coord=m.group(3), mean=float(m.group(4)),
                             lo=float(m.group(5)), hi=float(m.group(6)),
                             excludes0=m.group(7) == "yes"))
        prev = line
    d = pd.DataFrame(rows)
    if len(d) != 12 * 4 * 2:
        raise SystemExit(f"parsed {len(d)} Arm A bootstrap rows, expected 96")
    return d


_B_CELL = re.compile(r"^(\w+) / (\w+)\s+native H = ")
_B_COORD = re.compile(r"^  (-dDP|-dEO|dAUC)\b")
_B_VAL = re.compile(
    r"^    (tau_base|tau_int)\s+([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\] s([\d.]+)"
    r"\s+([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\] s([\d.]+)")
_B_RES = re.compile(r"^    tau_int resolved\s+(\w+)\s+(\w+)\s*$")
_B_CLASS = re.compile(r"^    transfer class \(X22\)\s+(.+?)\s*$")


def parse_armB_seven(path=ARMB_SEVEN_TXT):
    """The frozen 7 native cells: Arm A and native side by side, per coordinate."""
    rows, cell, coord, cur = [], None, None, {}
    for line in open(path):
        line = line.rstrip("\n")
        m = _B_CELL.match(line)
        if m:
            cell, coord = (m.group(1), m.group(2)), None
            continue
        m = _B_COORD.match(line)
        if m:
            coord = m.group(1)
            continue
        m = _B_VAL.match(line)
        if m and cell and coord:
            cur[(cell, coord, m.group(1))] = dict(
                method=cell[0], dataset=cell[1], coord=coord, quantity=m.group(1),
                a_mean=float(m.group(2)), a_lo=float(m.group(3)),
                a_hi=float(m.group(4)), a_sign=float(m.group(5)),
                n_mean=float(m.group(6)), n_lo=float(m.group(7)),
                n_hi=float(m.group(8)), n_sign=float(m.group(9)))
            continue
        m = _B_RES.match(line)
        if m and cell and coord:
            k = (cell, coord, "tau_int")
            if k in cur:
                cur[k]["a_resolved"] = m.group(1) == "True"
                cur[k]["n_resolved"] = m.group(2) == "True"
            continue
        m = _B_CLASS.match(line)
        if m and cell and coord:
            for q in ("tau_base", "tau_int"):
                if (cell, coord, q) in cur:
                    cur[(cell, coord, q)]["transfer_class"] = m.group(1)
    rows = list(cur.values())
    d = pd.DataFrame(rows)
    ncell = d.groupby(["method", "dataset"]).ngroups
    if ncell != 7:
        raise SystemExit(f"parsed {ncell} native cells, expected 7")
    return d


def summary(path):
    """An X25/X26/X27 summary CSV, indexed for lookup by quantity."""
    d = pd.read_csv(path)
    return d


def q(s, quantity, selector=None):
    r = s[s.quantity == quantity]
    if selector is not None:
        r = r[r.selector == selector]
    if len(r) != 1:
        raise KeyError(f"{quantity} (selector={selector}): {len(r)} rows")
    return r.iloc[0]


# ------------------------------------------------------- independent re-derivation
def arma_recompute():
    """Re-run the frozen controlled bootstrap, reproducing its RNG stream.

    Same inputs, same cell table, same dataset-then-method order and one shared
    generator seeded exactly as bootstrap_armA.main() seeds it, so the intervals
    must land on the frozen ones rather than merely near them.
    """
    from analyze_armA import build
    from bootstrap_armA import cell_table, boot, SEED
    d = build(ARMA_CSVS)
    cols = ["abase_auc", "abase_ndp", "int_auc", "int_ndp",
            "apkg_auc", "apkg_ndp", "D_auc", "D_ndp"]
    rng = np.random.default_rng(SEED)
    out = []
    for ds_ in sorted(d.dataset.unique()):
        for m in sorted(d.method.unique()):
            cells = cell_table(d, m, ds_)
            if cells.empty:
                continue
            reps = boot(cells, cols, rng)
            for qn, lab in (("abase", "tau_base^audit"), ("int", "tau_int"),
                            ("apkg", "tau_pkg^audit"), ("D", "D_selector")):
                for c, cl in (("auc", "dAUC"), ("ndp", "-dDP")):
                    k = cols.index(f"{qn}_{c}")
                    lo, hi = np.percentile(reps[:, k], [2.5, 97.5])
                    out.append(dict(dataset=ds_, method=m, quantity=lab, coord=cl,
                                    mean=float(cells[f"{qn}_{c}"].mean()),
                                    lo=float(lo), hi=float(hi)))
    return pd.DataFrame(out)
