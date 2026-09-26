"""X24 figures: interaction plots of NIFTY/German tau_int over H, one line per D.

Reads only the analysis summary written by analyze_x24_factorial.py (means and
95% paired hierarchical bootstrap intervals). It computes nothing new.

Main figure: tau_int(-dDP) under sigma_c^BCE (the pre-registered primary).
Supplementary: small multiples, rows = selector (BCE, AUC), columns = coordinate
(-dDP, -dEO, dAUC). Each panel has its own y-axis; there is never a second y-scale.

Colour: categorical slots 1-2 of the reference palette (blue, orange). These are
documented as passing all-pairs CVD and normal-vision checks in both modes. D
also carries marker shape (circle / square) as secondary encoding, a legend, and
direct end labels, so identity never rests on colour alone.

    python harness/experiments/plot_x24_factorial.py --summary harness/results/x24_nifty_german_summary.csv --outdir harness/results/figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import pandas as pd                                                   # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
SERIES = {"D0": dict(color="#2a78d6", marker="o",
                     name="D0: Arm-A NIFTY configuration"),
          "D1": dict(color="#eb6834", marker="s",
                     name="D1: official/native augmentation/validation-view config.")}
CELL = {("D0", 200): "P00", ("D0", 1000): "P10", ("D1", 200): "P01", ("D1", 1000): "P11"}
COORD = {"ndp": "τ_int (−ΔDP)", "neo": "τ_int (−ΔEO)", "auc": "τ_int (ΔAUC)"}
SEL = {"common_bce": "σ_c^BCE", "common_auc": "σ_c^AUC"}
XPOS = {200: 0, 1000: 1}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK2, width=1, length=3)
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.axhline(0, color=BASE, linewidth=1, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["H = 200", "H = 1000"])
    ax.set_xlim(-0.45, 1.55)


def panel(ax, s, sel, coord, label_points, end_labels, dodge=0.03):
    style(ax)
    for D, sp in SERIES.items():
        xs, ys, los, his, res = [], [], [], [], []
        for H in (200, 1000):
            r = s[(s.selector == sel) & (s.coord == coord) & (s.quantity == CELL[(D, H)])].iloc[0]
            xs.append(XPOS[H]); ys.append(r["mean"]); los.append(r["lo"]); his.append(r["hi"])
            res.append(bool(r["resolved"]))
        off = -dodge if D == "D0" else dodge
        xo = [x + off for x in xs]
        ax.plot(xo, ys, color=sp["color"], linewidth=2, solid_capstyle="round", zorder=3)
        ax.errorbar(xo, ys, yerr=[[y - l for y, l in zip(ys, los)], [h - y for y, h in zip(ys, his)]],
                    fmt="none", ecolor=sp["color"], elinewidth=1.5, capsize=0, zorder=2, alpha=0.9)
        for x, y, r in zip(xo, ys, res):
            # filled marker = resolved under the frozen rule; hollow = unresolved
            ax.plot(x, y, marker=sp["marker"], markersize=8, zorder=4,
                    markerfacecolor=sp["color"] if r else SURFACE,
                    markeredgecolor=sp["color"], markeredgewidth=2)
        if label_points:
            # D0 values below-left, D1 values up-right: away from the other
            # series' bars and from each line's own approach; the H=1000 value
            # also carries the series name (direct label)
            for x, y, H in zip(xo, ys, (200, 1000)):
                txt = f"{y:+.3f}" + (f"  {D}" if (H == 1000 and D == "D1") else "")
                txt = (f"{D}  " if (H == 1000 and D == "D0") else "") + txt
                if D == "D0" and H == 1000:
                    # the D1 line crosses to the left of this point, so the label
                    # goes to its right, level with it and clear of the D1 bar top
                    ax.annotate(txt, (x, y), textcoords="offset points", xytext=(10, 0),
                                ha="left", va="center", fontsize=9, color=INK2)
                elif D == "D0":
                    ax.annotate(txt, (x, y), textcoords="offset points", xytext=(-12, -11),
                                ha="right", va="top", fontsize=9, color=INK2)
                else:
                    ax.annotate(txt, (x, y), textcoords="offset points", xytext=(9, 9),
                                ha="left", va="bottom", fontsize=9, color=INK2)
        if end_labels:
            ax.annotate(f"{D} ({CELL[(D, 1000)]})", (xo[1], ys[1]), textcoords="offset points",
                        xytext=(12, 0), ha="left", va="center", fontsize=9, color=INK)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    s = pd.read_csv(a.summary)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})

    outs = []
    # ---- main figure ----
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    panel(ax, s, "common_bce", "ndp", label_points=True, end_labels=False, dodge=0.06)
    ax.set_ylabel(COORD["ndp"] + "   (> 0: intervention improves DP)", color=INK2)
    ax.set_title("NIFTY / German: intervention effect across the 2×2 protocol factors",
                 color=INK, fontsize=11, loc="left", pad=30)
    ax.text(0, 1.015, "σ_c^BCE · mean over 6 splits × 5 runs\n"
            "bars = 95% paired hierarchical bootstrap · filled marker = resolved (frozen rule)",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    handles = [plt.Line2D([], [], color=sp["color"], marker=sp["marker"], linewidth=2,
                          markersize=7, label=sp["name"]) for sp in SERIES.values()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.1), frameon=False,
              fontsize=8.5, labelcolor=INK2, ncol=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(a.outdir, f"x24_nifty_german_interaction_dp_bce.{ext}")
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
        fig.savefig(p, dpi=200)
        outs.append(p)
    plt.close(fig)

    # ---- supplementary small multiples ----
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for i, sel in enumerate(("common_bce", "common_auc")):
        for j, coord in enumerate(("ndp", "neo", "auc")):
            ax = axes[i, j]
            panel(ax, s, sel, coord, label_points=False, end_labels=(j == 2))
            ax.set_title(f"{COORD[coord]} · {SEL[sel]}", color=INK, fontsize=10, loc="left")
    handles = [plt.Line2D([], [], color=sp["color"], marker=sp["marker"], linewidth=2,
                          markersize=7, label=sp["name"]) for sp in SERIES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9,
               labelcolor=INK2)
    fig.suptitle("NIFTY / German 2×2: τ_int by coordinate and selector (filled = resolved; "
                 "bars = 95% bootstrap)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    for ext in ("png", "pdf"):
        p = os.path.join(a.outdir, f"x24_nifty_german_interaction_supplementary.{ext}")
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
        fig.savefig(p, dpi=200)
        outs.append(p)
    plt.close(fig)
    for p in outs:
        print("[written]", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
