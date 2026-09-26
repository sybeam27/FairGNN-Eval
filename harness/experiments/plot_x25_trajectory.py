"""X25 figures: NIFTY/German trajectory–selection decomposition.

Reads only the analyzer outputs (x25_summary.csv and x25_selected_epochs.csv):
means and 95% paired hierarchical bootstrap intervals, plus per-cell selected
epochs. It computes no new estimate.

Main: τ_int(−ΔDP) at the pre-registered fixed epochs for D0 and D1. A strip
panel below, on the same x scale with its own categorical y, shows where
σ_c^BCE selects over the full support {0..1000} for each arm.
Second: cap ≤ 200 vs full ≤ 1000 τ_int per D and selector, with the frozen
H=200 value as a reference tick.
Supplementary: the same two views for −ΔEO and ΔAUC.

D is identified by colour (reference palette slots 1-2, documented as passing
all-pairs checks) plus marker shape, a legend and direct labels. There is never
a second y-scale.

    python harness/experiments/plot_x25_trajectory.py --indir harness/results/x25 --outdir harness/results/figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_C, BASE = "#e1e0d9", "#c3c2b7"
DS = {"D0": dict(color="#2a78d6", marker="o", name="D0: Arm-A NIFTY configuration (R10)"),
      "D1": dict(color="#eb6834", marker="s", name="D1: official/native aug./validation-view config. (R11)")}
EPOCHS = (25, 50, 100, 150, 200, 300, 400, 600, 800, 1000)
LAB = {"ndp": "τ_int (−ΔDP)", "neo": "τ_int (−ΔEO)", "auc": "τ_int (ΔAUC)"}
SIGL = {"bce": "σ_c^BCE", "auc": "σ_c^AUC"}


def style(ax, xgrid=False):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK2, width=1, length=3)
    ax.grid(axis="x" if xgrid else "y", color=GRID_C, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def q(s, name):
    r = s[s.quantity == name]
    if r.empty:
        raise KeyError(name)
    return r.iloc[0]


def trajectory_panel(ax, s, coord, label=True):
    style(ax)
    ax.axhline(0, color=BASE, linewidth=1, zorder=1)
    ax.axvline(200, color=GRID_C, linewidth=1, zorder=0)
    for D, sp in DS.items():
        rows = [q(s, f"{D}.fx{t}.{coord}") for t in EPOCHS]
        m = np.array([r["mean"] for r in rows])
        lo = np.array([r["lo"] for r in rows])
        hi = np.array([r["hi"] for r in rows])
        x = np.array(EPOCHS, float) + (-6 if D == "D0" else 6)
        ax.fill_between(x, lo, hi, color=sp["color"], alpha=0.10, linewidth=0, zorder=2)
        ax.plot(x, m, color=sp["color"], linewidth=2, zorder=3)
        ax.plot(x, m, linestyle="none", marker=sp["marker"], markersize=7, markerfacecolor=sp["color"],
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        if label:
            # the two series can end at nearly the same value (dAUC), so the
            # labels are staggered vertically instead of overprinting
            ax.annotate(f"{D}  {m[-1]:+.3f}", (x[-1], m[-1]), textcoords="offset points",
                        xytext=(10, -9 if D == "D0" else 9), ha="left",
                        va="top" if D == "D0" else "bottom", fontsize=9, color=INK2)
    ax.set_xlim(0, 1130)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])


def strip_panel(ax, se):
    style(ax, xgrid=True)
    ax.axvline(200, color=BASE, linewidth=1, zorder=1)
    rows = [("D1", "m0_epoch", "D1 · M0"), ("D1", "m1_epoch", "D1 · M1"),
            ("D0", "m0_epoch", "D0 · M0"), ("D0", "m1_epoch", "D0 · M1")]
    rng = np.random.default_rng(0)
    for i, (D, col, _) in enumerate(rows):
        g = se[(se.D == D) & (se.sigma == "bce") & (se.cap == 1000)]
        e = g[col].to_numpy(float)
        y = i + rng.uniform(-0.18, 0.18, size=len(e))
        ax.plot(e, y, linestyle="none", marker=DS[D]["marker"], markersize=5,
                markerfacecolor=DS[D]["color"] if col == "m1_epoch" else SURFACE,
                markeredgecolor=DS[D]["color"], markeredgewidth=1.2, alpha=0.9, zorder=3)
        ax.plot([np.median(e)] * 2, [i - 0.32, i + 0.32], color=INK2, linewidth=2, zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[2] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 1130)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_xlabel("epoch", color=INK2)


def capfull_panel(ax, s, coord):
    style(ax)
    ax.axhline(0, color=BASE, linewidth=1, zorder=1)
    groups = [("D0", "bce"), ("D0", "auc"), ("D1", "bce"), ("D1", "auc")]
    for gi, (D, sig) in enumerate(groups):
        sp = DS[D]
        for k, (cap, mk, dx) in enumerate(((200, "^", -0.15), (1000, sp["marker"], 0.15))):
            r = q(s, f"{D}.{sig}.c{cap}.{coord}")
            ax.errorbar(gi + dx, r["mean"], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                        fmt="none", ecolor=sp["color"], elinewidth=1.5, zorder=2)
            ax.plot(gi + dx, r["mean"], marker=mk, markersize=8, linestyle="none", zorder=4,
                    markerfacecolor=sp["color"] if cap == 1000 else SURFACE,
                    markeredgecolor=sp["color"], markeredgewidth=2)
        h = q(s, f"{D}.{sig}.h200.{coord}")
        ax.plot([gi - 0.32, gi - 0.02], [h["mean"]] * 2, color=MUTED, linewidth=2, zorder=3)
        S = q(s, f"{D}.{sig}.S.{coord}")
        ax.annotate(f"S {S['mean']:+.3f}\n[{S['lo']:+.3f}, {S['hi']:+.3f}]", (gi, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 4 if gi % 2 == 0 else 26), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{D}\n{SIGL[sig]}" for D, sig in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)


def save(fig, outdir, name, outs):
    for ext in ("png", "pdf"):
        p = os.path.join(outdir, f"{name}.{ext}")
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
        fig.savefig(p, dpi=200)
        outs.append(p)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    s = pd.read_csv(os.path.join(a.indir, "x25_summary.csv"))
    se = pd.read_csv(os.path.join(a.indir, "x25_selected_epochs.csv"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    outs = []
    handles = [plt.Line2D([], [], color=sp["color"], marker=sp["marker"], linewidth=2, markersize=7,
                          label=sp["name"]) for sp in DS.values()]

    # ---- main ----
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.8), sharex=False,
                                  gridspec_kw=dict(height_ratios=[3, 1.35], hspace=0.35))
    trajectory_panel(ax, s, "ndp")
    ax.set_ylabel(LAB["ndp"] + "  at fixed epoch", color=INK2)
    ax.set_title("NIFTY / German: intervention effect along the H = 1000 training trajectory",
                 color=INK, fontsize=11, loc="left", pad=30)
    ax.text(0, 1.015, "no checkpoint selection · mean over 6 splits × 5 runs · band = 95% paired "
            "hierarchical bootstrap\nvertical line at epoch 200 = the H = 200 horizon",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    strip_panel(ax2, se)
    ax2.set_title("where σ_c^BCE selects over the full support {0..1000} (filled = M1, hollow = M0; bar = median)",
                  color=INK2, fontsize=8.5, loc="left")
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.0), frameon=False,
               fontsize=8.5, labelcolor=INK2, ncol=1)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.17)
    save(fig, a.outdir, "x25_nifty_german_trajectory_dp", outs)

    # ---- second ----
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    capfull_panel(ax, s, "ndp")
    ax.set_ylabel(LAB["ndp"], color=INK2)
    ax.set_title("Same H = 1000 trajectories, selector support capped at 200 vs full 1000",
                 color=INK, fontsize=11, loc="left", pad=30)
    ax.text(0, 1.015, "▲ hollow = cap ≤ 200 · filled = full ≤ 1000 · bars = 95% bootstrap · "
            "gray tick = frozen H = 200 run\nS = full − cap (selection-support contrast)",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.0), frameon=False,
               fontsize=8.5, labelcolor=INK2)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.26)
    save(fig, a.outdir, "x25_nifty_german_cap_vs_full_dp", outs)

    # ---- supplementary ----
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for j, coord in enumerate(("ndp", "neo", "auc")):
        trajectory_panel(axes[0, j], s, coord, label=(j == 2))
        axes[0, j].set_title(f"{LAB[coord]} at fixed epoch", color=INK, fontsize=10, loc="left")
        capfull_panel(axes[1, j], s, coord)
        axes[1, j].set_title(f"{LAB[coord]}: cap ≤ 200 (hollow ▲) vs full ≤ 1000 (filled)",
                             color=INK, fontsize=10, loc="left")
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("NIFTY / German X25: trajectory and selection-support views (bands/bars = 95% bootstrap; "
                 "gray tick = frozen H = 200)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save(fig, a.outdir, "x25_nifty_german_supplementary", outs)
    for p in outs:
        print("[written]", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
