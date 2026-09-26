"""Experiment 2 -- how often a configuration change flips the sign of tau_{-I->+I}(P) or changes whether it
is resolved, by the factor that changed. Diverging bars around 0:

    up    share of comparisons whose point estimate changes sign        (CSV: sign_changed)
    down  share whose resolution status changes, split into
          resolved -> unresolved (lost) and unresolved -> resolved (gained)   (CSV: primary/robustness_resolved)

Counts are shown as k/n inside each bar segment; the coordinate is named under each bar. Comparisons share data, splits and baseline with their primary
configuration, so they are not independent.

    python figures/src/make_all.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
from style import apply_tone, tint as _tint, TINT, TINT_RES, OUTLINE, ROLE     # the Fig. 2 tone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parents[1]           # output: figures/ (PDF only)
DATA = HERE.parent / "results"
cfg = pd.read_csv(DATA / "2_configuration_variation.csv")
assert len(cfg) == 78, len(cfg)                       # 26 comparisons x 3 coordinates


def changed_factor(v: str) -> str:
    v = v.lower()
    if "budget" in v:
        return "budget"
    if "upstream" in v or "spmm" in v:
        return "upstream"
    if "backbone" in v or "encoder" in v:
        return "encoder"
    raise ValueError(v)


as_true = lambda s: s.astype(str).str.strip().str.lower().eq("true")
cfg["factor"] = cfg["varied_factor"].map(changed_factor)
cfg["sign"] = as_true(cfg["sign_changed"])
cfg["lost"] = as_true(cfg["primary_resolved"]) & ~as_true(cfg["robustness_resolved"])
cfg["gained"] = ~as_true(cfg["primary_resolved"]) & as_true(cfg["robustness_resolved"])
# the CSV's resolution_changed flag must equal lost | gained
assert (as_true(cfg["resolution_changed"]) == (cfg["lost"] | cfg["gained"])).all()

FACTORS = [("encoder", "Encoder"), ("upstream", "Upstream config."), ("budget", "Budget")]
COORDS = [("negDP", r"$-\Delta_{\mathrm{DP}}$"), ("negEO", r"$-\Delta_{\mathrm{EO}}$"), ("dAUC", r"$\Delta$AUC")]
summary = (cfg.groupby(["coordinate", "factor"])
              .agg(n=("sign", "size"), sign=("sign", "sum"), lost=("lost", "sum"), gained=("gained", "sum"))
              .reset_index())
pass  # counts CSV stays in results
print(summary.to_string(index=False))

plt.rcParams.update({
    "font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "xtick.labelcolor": "#0b0b0b", "ytick.labelcolor": "#0b0b0b",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "savefig.dpi": 300, "pdf.fonttype": 42,
})
apply_tone()
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
C_SIGN, C_LOST, C_GAIN = ROLE["sign_flip"], ROLE["lost"], ROLE["gained"]


# one panel; per changed factor three adjacent bars, one per coordinate, named directly under each bar.
# All bars share one style: light fill, coloured outline; the count k/n sits inside the end of each segment.
COORD_SHORT = {"negDP": r"$-\Delta_{\mathrm{DP}}$", "negEO": r"$-\Delta_{\mathrm{EO}}$", "dAUC": r"$\Delta$AUC"}


def tint(c, t=0.82):
    r, g, b_ = matplotlib.colors.to_rgb(c)
    return (r + (1 - r) * t, g + (1 - g) * t, b_ + (1 - b_) * t)


def bar(ax, x, h, bottom, w, color, k, n):
    ax.bar(x, h, bottom=bottom, width=w, facecolor=tint(color), edgecolor=color, lw=0.8, zorder=3)
    end = bottom + h
    ax.text(x, end - np.sign(h) * 0.015, f"{k}/{n}", ha="center", va="top" if h > 0 else "bottom",
            fontsize=6, color=INK, zorder=5)


def grouped(ax):
    w, gap = 0.25, 0.03
    x = np.arange(len(FACTORS))
    ax.axhline(0, color=INK2, lw=0.8, zorder=4)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    minor_x, minor_lab = [], []
    for j, (coord, _) in enumerate(COORDS):
        s = summary[summary.coordinate.eq(coord)].set_index("factor")
        for xi, (f, _) in zip(x + (j - 1) * (w + gap), FACTORS):
            n, k_s, k_l, k_g = (int(s.loc[f, c]) for c in ("n", "sign", "lost", "gained"))
            up, lo, ga = k_s / n, k_l / n, k_g / n
            if up: bar(ax, xi, up, 0, w, C_SIGN, k_s, n)
            if lo: bar(ax, xi, -lo, 0, w, C_LOST, k_l, n)
            if ga: bar(ax, xi, -ga, -lo, w, C_GAIN, k_g, n)
            minor_x.append(xi); minor_lab.append(COORD_SHORT[coord])
    ax.xaxis.remove_overlapping_locs = False     # the middle bar sits on the factor tick
    ax.set_xticks(minor_x, minor=True)
    ax.set_xticklabels(minor_lab, minor=True, fontsize=6.2)
    ax.tick_params(axis="x", which="minor", length=0, pad=2)
    ax.set_xticks(x)
    n_of = summary[summary.coordinate.eq("negDP")].set_index("factor")["n"]
    ax.set_xticklabels([f"{lab} (n={int(n_of[f])})" for f, lab in FACTORS], fontsize=7.5)
    ax.tick_params(axis="x", which="major", length=0, pad=9)
    for b in x[:-1] + 0.5:
        ax.axvline(b, color=GRID, lw=0.6, zorder=0)
    ax.set_xlim(-0.55, len(FACTORS) - 0.45)
    ax.set_ylim(-0.6, 0.78)
    t = np.arange(-0.5, 0.76, 0.25)
    ax.set_yticks(t)
    ax.set_yticklabels([f"{abs(v):.0%}" for v in t])
    ax.set_ylabel("Share of paired comparisons", fontsize=7.9)
    ax.text(0.015, 0.985, "Sign flips ↑", transform=ax.transAxes, ha="left", va="top", fontsize=7.2)
    ax.text(0.015, 0.015, "Resolution transitions ↓", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2)


fig, ax = plt.subplots(figsize=(3.35, 2.4))
grouped(ax)
# the coordinate is named under each bar, so the legend only explains the three event colours
handles = [Patch(facecolor=tint(C_SIGN), edgecolor=C_SIGN, lw=0.8, label="sign flip (point estimate)"),
           Patch(facecolor=tint(C_LOST), edgecolor=C_LOST, lw=0.8, label="resolved$\\rightarrow$unresolved"),
           Patch(facecolor=tint(C_GAIN), edgecolor=C_GAIN, lw=0.8, label="unresolved$\\rightarrow$resolved")]
leg = fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.035), frameon=True,
                 fancybox=False, edgecolor="#d0cfca", framealpha=1.0, columnspacing=0.9, handlelength=1.2,
                 handletextpad=0.4, borderpad=0.4, fontsize=7)
leg.get_frame().set_linewidth(0.4)
fig.tight_layout(rect=(0, 0.08, 1, 1), pad=0.3)
fig.savefig(HERE / "fig2_sign_resolution.pdf", bbox_inches="tight", pad_inches=0.02)
print("saved", HERE / "fig2_sign_resolution.pdf")
