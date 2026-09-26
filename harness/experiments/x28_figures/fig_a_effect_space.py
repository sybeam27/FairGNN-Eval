"""Figure A -- controlled package effect vs intervention effect, in effect space.

RQ1: within the controlled protocol (H = 200 for every cell), how large is the
fairness-method intervention beside the rest of the package it ships in?

Both vectors start at the origin, as registered:

    tau_nonint = Y(M_off) - Y(B)      the package without the intervention
    tau_I      = Y(M_on) - Y(M_off)   the intervention itself

with x = dAUC and y = -dDP. Vector length is the measured effect; nothing is
normalised. Numbers are parsed from the frozen armA_final_bootstrap.txt and are
independently re-derived from the eight Arm A CSVs through the frozen bootstrap.

Marker fill encodes "95% interval excludes 0" -- the column that exists in the
frozen artifact. No Arm A artifact carries a combined resolved flag for these
12 cells, so none is invented here.

    python harness/experiments/x28_figures/fig_a_effect_space.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K                                                   # noqa: E402

NONINT, INT = "tau_base^audit", "tau_int"
VEC = ((NONINT, "non-intervention package  Y(M_off) − Y(B)"),
       (INT, "intervention  Y(M_on) − Y(M_off)"))


def cell_xy(frozen, ds, method, quantity):
    """The frozen (x = dAUC, y = -dDP) endpoint and its two intervals."""
    g = frozen[(frozen.dataset == ds) & (frozen.method == method)
               & (frozen.quantity == quantity)]
    x = g[g.coord == "dAUC"].iloc[0]
    y = g[g.coord == "-dDP"].iloc[0]
    return x, y


def limits(vals, pad=0.14):
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        lo, hi = lo - 1e-3, hi + 1e-3
    m = (hi - lo) * pad
    return lo - m, hi + m


def figure_a(frozen, src, ck, outs):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.05))
    for ax, ds in zip(axes, K.DATASETS):
        K.style(ax)
        xs, ys = [0.0], [0.0]
        for method in K.METHODS:
            for quantity, _ in VEC:
                x, y = cell_xy(frozen, ds, method, quantity)
                xs += [x["mean"], x["lo"], x["hi"]]
                ys += [y["mean"], y["lo"], y["hi"]]
        ax.set_xlim(*limits(xs))
        ax.set_ylim(*limits(ys))
        K.zero_lines(ax)
        if ds == K.DATASETS[0]:
            # the axes meaning is shared across panels; one legend of quadrants
            # is enough and keeps the other panels clear of background text
            K.quadrant_labels(ax)

        for method in K.METHODS:
            col, mk = K.C[method], K.MK[method]
            for quantity, _ in VEC:
                x, y = cell_xy(frozen, ds, method, quantity)
                intervention = quantity == INT
                c = col if intervention else K.NEUTRAL
                ax.annotate("", xy=(x["mean"], y["mean"]), xytext=(0, 0),
                            arrowprops=dict(
                                arrowstyle="-|>,head_width=0.16,head_length=0.32",
                                color=c, linewidth=1.7 if intervention else 0.9,
                                linestyle="-" if intervention else (0, (3, 2)),
                                alpha=1.0 if intervention else 0.75,
                                shrinkA=0, shrinkB=0), zorder=3)
                K.errcross(ax, x["mean"], y["mean"], x["lo"], x["hi"],
                           y["lo"], y["hi"], c,
                           lw=0.9 if intervention else 0.7,
                           alpha=1.0 if intervention else 0.6, zorder=4)
                K.resolved_marker(ax, x["mean"], y["mean"], c, mk,
                                  bool(x["excludes0"] and y["excludes0"]),
                                  size=6.0 if intervention else 4.6,
                                  lw=1.4 if intervention else 1.0)
                src.add(figure="A", dataset=ds, method=method,
                        vector="intervention" if intervention else "non_intervention",
                        x_dAUC=x["mean"], x_lo=x["lo"], x_hi=x["hi"],
                        y_ndDP=y["mean"], y_lo=y["lo"], y_hi=y["hi"],
                        x_excludes0=bool(x["excludes0"]),
                        y_excludes0=bool(y["excludes0"]))
        ck.limits(ax, xs, ys, f"Figure A / {ds}")
        ax.set_title(ds, color=K.INK, fontsize=K.FS["panel"], loc="left", pad=4)
        ax.set_xlabel("change in AUC", color=K.INK2, fontsize=K.FS["label"])
        if ds == K.DATASETS[0]:
            ax.set_ylabel("change in demographic parity\n(−ΔDP; higher is fairer)",
                          color=K.INK2, fontsize=K.FS["label"])

    handles = [plt.Line2D([], [], color=K.NEUTRAL, lw=1.0, linestyle=(0, (3, 2)),
                          label=VEC[0][1]),
               plt.Line2D([], [], color=K.INK2, lw=1.7, label=VEC[1][1])]
    handles += [plt.Line2D([], [], color=K.C[m], marker=K.MK[m], lw=0,
                           markersize=5.5, markerfacecolor=K.C[m],
                           markeredgecolor=K.C[m], label=m) for m in K.METHODS]
    handles += [plt.Line2D([], [], color=K.INK2, marker="o", lw=0, markersize=5.5,
                           markerfacecolor=K.SURFACE, markeredgecolor=K.INK2,
                           label="hollow: 95% interval covers 0")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=K.FS["note"], labelcolor=K.INK2,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Controlled protocol (H = 200): the intervention beside the package "
                 "it ships in", color=K.INK, fontsize=K.FS["title"], x=0.008, ha="left")
    fig.text(0.008, 0.895, "arrows from the origin · crosses are 95% paired "
             "hierarchical bootstrap intervals · axes scale per dataset",
             fontsize=K.FS["note"], color=K.MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.14, 1, 0.88))
    K.save(fig, "figA_effect_space", outs)


def figure_a2(frozen, src, ck, outs):
    """A2: the same 12 cells as paired small multiples, one panel per cell."""
    fig, axes = plt.subplots(3, 4, figsize=(7.4, 5.4))
    for r, ds in enumerate(K.DATASETS):
        for c, method in enumerate(K.METHODS):
            ax = axes[r][c]
            K.style(ax)
            xs, ys = [0.0], [0.0]
            for quantity, _ in VEC:
                x, y = cell_xy(frozen, ds, method, quantity)
                xs += [x["mean"], x["lo"], x["hi"]]
                ys += [y["mean"], y["lo"], y["hi"]]
            ax.set_xlim(*limits(xs, 0.18))
            ax.set_ylim(*limits(ys, 0.18))
            K.zero_lines(ax)
            for quantity, _ in VEC:
                x, y = cell_xy(frozen, ds, method, quantity)
                intervention = quantity == INT
                col = K.C[method] if intervention else K.NEUTRAL
                K.errcross(ax, x["mean"], y["mean"], x["lo"], x["hi"],
                           y["lo"], y["hi"], col, lw=0.9,
                           alpha=1.0 if intervention else 0.65)
                K.resolved_marker(ax, x["mean"], y["mean"], col, K.MK[method],
                                  bool(x["excludes0"] and y["excludes0"]),
                                  size=6.0 if intervention else 4.8)
                src.add(figure="A2", dataset=ds, method=method,
                        vector="intervention" if intervention else "non_intervention",
                        x_dAUC=x["mean"], y_ndDP=y["mean"])
            ck.limits(ax, xs, ys, f"Figure A2 / {ds} / {method}")
            ax.set_title(f"{method} · {ds}", color=K.INK, fontsize=K.FS["note"] + 0.5,
                         loc="left", pad=3)
            ax.tick_params(labelsize=K.FS["tick"] - 0.8)
            if c == 0:
                ax.set_ylabel("−ΔDP", color=K.INK2, fontsize=K.FS["label"] - 0.5)
            if r == 2:
                ax.set_xlabel("ΔAUC", color=K.INK2, fontsize=K.FS["label"] - 0.5)
    handles = [plt.Line2D([], [], color=K.NEUTRAL, marker="o", lw=0, markersize=5,
                          markerfacecolor=K.NEUTRAL, markeredgecolor=K.NEUTRAL,
                          label="non-intervention package endpoint"),
               plt.Line2D([], [], color=K.INK2, marker="o", lw=0, markersize=5.5,
                          markerfacecolor=K.INK2, markeredgecolor=K.INK2,
                          label="intervention endpoint (method colour/marker)"),
               plt.Line2D([], [], color=K.INK2, marker="o", lw=0, markersize=5.5,
                          markerfacecolor=K.SURFACE, markeredgecolor=K.INK2,
                          label="hollow: 95% interval covers 0")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=K.FS["note"], labelcolor=K.INK2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Controlled protocol: package and intervention endpoints per cell",
                 color=K.INK, fontsize=K.FS["title"], x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    K.save(fig, "figA2_effect_space_small_multiples", outs)


def main() -> int:
    K.rc()
    frozen = K.parse_armA_bootstrap()
    ck, outs = K.Checks(), []

    # independent re-derivation: same inputs, same frozen bootstrap, same RNG
    # stream -- the intervals must land on the frozen ones, not merely near them
    re_ = K.arma_recompute()
    for _, row in frozen.iterrows():
        g = re_[(re_.dataset == row.dataset) & (re_.method == row.method)
                & (re_.quantity == row.quantity) & (re_.coord == row.coord)]
        if g.empty:
            raise SystemExit(f"re-derivation missing {tuple(row[:4])}")
        g = g.iloc[0]
        tag = f"{row.dataset}/{row.method}/{row.quantity}/{row.coord}"
        ck.close(f"{tag} mean", g["mean"], row["mean"], 1e-4)
        ck.close(f"{tag} lo", g["lo"], row["lo"], 1e-4)
        ck.close(f"{tag} hi", g["hi"], row["hi"], 1e-4)

    src = K.Source("figA_effect_space")
    figure_a(frozen, src, ck, outs)
    figure_a2(frozen, src, ck, outs)
    p = src.write()
    ok = ck.report()
    for o in outs + [p]:
        print("[written]", o)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
