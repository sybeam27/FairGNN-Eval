"""Figure B -- how the intervention contrast moves when the protocol changes.

RQ2: the same seven method-dataset cells, measured first under the controlled
protocol (H = 200) and then under each method's native protocol. Each cell is
one arrow in effect space, from its controlled endpoint to its native endpoint.

Numbers are parsed from the frozen armB_phase1_seven_cell_analysis.txt and the
means are independently re-derived from the per-cell CSVs. Marker fill is the
frozen `tau_int resolved` flag on -dDP.

The crossing metadata written into the source CSV is descriptive, for caption
writing only. It defines no new classification: the frozen X22 transfer class
is carried alongside it unchanged.

    python harness/experiments/x28_figures/fig_b_native_shift.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K                                                   # noqa: E402


# The seven native endpoints crowd the centre of the plane, so their labels are
# placed by hand rather than at a shared offset that would overprint.
LABEL_OFF = {("FairVGNN", "german"): (10, 4), ("FairGB", "credit"): (9, 7),
             ("FairVGNN", "credit"): (9, 8), ("FairVGNN", "bail"): (-9, -8),
             ("FairGB", "german"): (9, 6), ("FairGB", "bail"): (-9, -9),
             ("NIFTY", "german"): (10, -3)}


def cells(frozen):
    """The seven (method, dataset) cells with both coordinates side by side."""
    out = []
    t = frozen[frozen.quantity == "tau_int"]
    for (m, ds), g in t.groupby(["method", "dataset"], sort=True):
        x = g[g.coord == "dAUC"].iloc[0]
        y = g[g.coord == "-dDP"].iloc[0]
        out.append(dict(method=m, dataset=ds, x=x, y=y))
    return sorted(out, key=lambda r: (r["dataset"], r["method"]))


def crossing(c):
    """Descriptive geometry of one arrow -- for the caption, not a finding."""
    ax_, ay = c["x"]["a_mean"], c["y"]["a_mean"]
    nx, ny = c["x"]["n_mean"], c["y"]["n_mean"]
    fair = (ay > 0) != (ny > 0)
    util = (ax_ > 0) != (nx > 0)
    return dict(crossed_fairness_sign=bool(fair), crossed_utility_sign=bool(util),
                crossed_both=bool(fair and util),
                quadrant_stayed_same=bool(not fair and not util))


def limits(vals, pad=0.16):
    lo, hi = float(np.min(vals)), float(np.max(vals))
    m = max((hi - lo) * pad, 1e-4)
    return lo - m, hi + m


def figure_b(cs, src, ck, outs):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    K.style(ax)
    xs, ys = [0.0], [0.0]
    for c in cs:
        xs += [c["x"]["a_mean"], c["x"]["n_mean"]]
        ys += [c["y"]["a_mean"], c["y"]["n_mean"],
               c["y"]["a_lo"], c["y"]["a_hi"], c["y"]["n_lo"], c["y"]["n_hi"]]
    ax.set_xlim(*limits(xs))
    ax.set_ylim(*limits(ys))
    K.zero_lines(ax)
    K.quadrant_labels(ax)

    for c in cs:
        col, mk = K.C[c["method"]], K.MK[c["method"]]
        ax_, ay = c["x"]["a_mean"], c["y"]["a_mean"]
        nx, ny = c["x"]["n_mean"], c["y"]["n_mean"]
        ax.annotate("", xy=(nx, ny), xytext=(ax_, ay),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.17,head_length=0.34",
                                    color=col, linewidth=1.5, alpha=0.9,
                                    shrinkA=3.2, shrinkB=3.2), zorder=3)
        # controlled start: hollow circle, always the same shape so the arrow
        # direction is read from the shape change, not only from the arrow head
        ax.plot(ax_, ay, marker="o", markersize=5.0, linestyle="none",
                markerfacecolor=K.SURFACE, markeredgecolor=col,
                markeredgewidth=1.3, zorder=5)
        K.resolved_marker(ax, nx, ny, col, mk, bool(c["y"]["n_resolved"]),
                          size=6.5, zorder=6)
        dx, dy = LABEL_OFF.get((c["method"], c["dataset"]), (7, 6))
        ax.annotate(f"{c['method']} / {c['dataset']}", (nx, ny),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=K.FS["note"], color=K.INK2,
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top", zorder=7)
        src.add(figure="B", method=c["method"], dataset=c["dataset"],
                controlled_dAUC=ax_, controlled_ndDP=ay,
                controlled_ndDP_lo=c["y"]["a_lo"], controlled_ndDP_hi=c["y"]["a_hi"],
                controlled_resolved=bool(c["y"]["a_resolved"]),
                native_dAUC=nx, native_ndDP=ny,
                native_ndDP_lo=c["y"]["n_lo"], native_ndDP_hi=c["y"]["n_hi"],
                native_resolved=bool(c["y"]["n_resolved"]),
                transfer_class_x22=c["y"].get("transfer_class", ""),
                **crossing(c))
    ck.limits(ax, xs, ys, "Figure B")
    ax.set_xlabel("change in AUC", color=K.INK2, fontsize=K.FS["label"])
    ax.set_ylabel("change in demographic parity (−ΔDP; higher is fairer)",
                  color=K.INK2, fontsize=K.FS["label"])
    handles = [plt.Line2D([], [], color=K.INK2, marker="o", lw=0, markersize=5,
                          markerfacecolor=K.SURFACE, markeredgecolor=K.INK2,
                          label="controlled H = 200 (arrow tail)"),
               plt.Line2D([], [], color=K.INK2, marker="s", lw=0, markersize=6,
                          markerfacecolor=K.INK2, markeredgecolor=K.INK2,
                          label="native protocol (arrow head)"),
               plt.Line2D([], [], color=K.INK2, marker="s", lw=0, markersize=6,
                          markerfacecolor=K.SURFACE, markeredgecolor=K.INK2,
                          label="hollow head: unresolved on −ΔDP")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=K.FS["note"], labelcolor=K.INK2,
               bbox_to_anchor=(0.5, -0.015))
    ax.set_title("Changing the protocol moves the intervention contrast",
                 color=K.INK, fontsize=K.FS["title"], loc="left", pad=16)
    ax.text(0, 1.015, "seven native-validation cells · marker shape and colour "
            "identify the method", transform=ax.transAxes,
            fontsize=K.FS["note"], color=K.MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    K.save(fig, "figB_native_shift", outs)


def figure_b2(cs, src, ck, outs):
    """B2: the same shift on -dDP alone, as a paired slope plot with intervals."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    K.style(ax, grid="y")
    ys, labels = [], []
    for i, c in enumerate(cs):
        col, mk = K.C[c["method"]], K.MK[c["method"]]
        a, n = c["y"]["a_mean"], c["y"]["n_mean"]
        xa, xn = 0.0, 1.0
        ax.plot([xa, xn], [a, n], color=col, linewidth=1.4, alpha=0.9, zorder=3)
        ax.plot([xa, xa], [c["y"]["a_lo"], c["y"]["a_hi"]], color=col,
                linewidth=1.0, alpha=0.55, zorder=2)
        ax.plot([xn, xn], [c["y"]["n_lo"], c["y"]["n_hi"]], color=col,
                linewidth=1.0, alpha=0.55, zorder=2)
        ax.plot(xa, a, marker="o", markersize=5, linestyle="none",
                markerfacecolor=K.SURFACE, markeredgecolor=col,
                markeredgewidth=1.3, zorder=5)
        K.resolved_marker(ax, xn, n, col, mk, bool(c["y"]["n_resolved"]), size=6.5)
        labels.append([n, f"{c['method']} / {c['dataset']}", col, n])
        ys += [c["y"]["a_lo"], c["y"]["a_hi"], c["y"]["n_lo"], c["y"]["n_hi"]]
        src.add(figure="B2", method=c["method"], dataset=c["dataset"],
                controlled_ndDP=a, controlled_lo=c["y"]["a_lo"],
                controlled_hi=c["y"]["a_hi"], native_ndDP=n,
                native_lo=c["y"]["n_lo"], native_hi=c["y"]["n_hi"])
    ax.axhline(0, color=K.BASEC, linewidth=0.9, zorder=1)
    ax.set_xlim(-0.25, 1.75)
    ax.set_ylim(*limits(ys, 0.10))
    # push the labels apart so none overprints another, then run a short leader
    # from each point to its displaced label
    y0, y1 = ax.get_ylim()
    sep = (y1 - y0) * 0.052
    labels.sort(key=lambda r: r[0])
    for i in range(1, len(labels)):
        if labels[i][3] - labels[i - 1][3] < sep:
            labels[i][3] = labels[i - 1][3] + sep
    shift = max(0.0, max(r[3] for r in labels) - (y1 - sep * 0.6))
    for r in labels:
        r[3] -= shift
    for yv, text, col, ylab in labels:
        ax.plot([1.02, 1.09], [yv, ylab], color=col, linewidth=0.7, alpha=0.6,
                zorder=2)
        ax.annotate(text, (1.10, ylab), fontsize=K.FS["note"], color=K.INK2,
                    ha="left", va="center")
    ck.limits(ax, [0.0, 1.0], ys, "Figure B2")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["controlled\nH = 200", "native\nprotocol"],
                       fontsize=K.FS["label"])
    ax.set_ylabel("intervention effect on −ΔDP", color=K.INK2,
                  fontsize=K.FS["label"])
    ax.set_title("Same cells, two protocols", color=K.INK,
                 fontsize=K.FS["title"], loc="left", pad=16)
    ax.text(0, 1.015, "bars are 95% paired hierarchical bootstrap intervals · "
            "filled marker = resolved under the native protocol",
            transform=ax.transAxes, fontsize=K.FS["note"], color=K.MUTED, va="bottom")
    fig.tight_layout()
    K.save(fig, "figB2_native_shift_slope", outs)


def main() -> int:
    K.rc()
    frozen = K.parse_armB_seven()
    cs = cells(frozen)
    ck, outs = K.Checks(), []

    # independent re-derivation of both endpoints' means from the per-cell CSVs
    from analyze_armA import build
    arma = build(K.ARMA_CSVS)
    arma = arma[arma.selector == "common_bce"]
    for c in cs:
        m, ds = c["method"], c["dataset"]
        g = arma[(arma.method == m) & (arma.dataset == ds)]
        ck.close(f"{m}/{ds} controlled -dDP mean", g.int_ndp.mean(),
                 c["y"]["a_mean"], 1e-4)
        ck.close(f"{m}/{ds} controlled dAUC mean", g.int_auc.mean(),
                 c["x"]["a_mean"], 1e-4)
        p = f"{K.ROOT}/harness/results/armB_native_{m}_{ds}.csv"
        n = pd.read_csv(p)
        n = n[n.selector == "common_bce"]
        ck.true(f"{m}/{ds} native cells = 30", len(n) == 30)
        ck.close(f"{m}/{ds} native -dDP mean", n.int_ndp.mean(),
                 c["y"]["n_mean"], 1e-4)
        ck.close(f"{m}/{ds} native dAUC mean", n.int_dauc.mean(),
                 c["x"]["n_mean"], 1e-4)

    src = K.Source("figB_native_shift")
    figure_b(cs, src, ck, outs)
    figure_b2(cs, src, ck, outs)
    p = src.write()
    ok = ck.report()
    for o in outs + [p]:
        print("[written]", o)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
