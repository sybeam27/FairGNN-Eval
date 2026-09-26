"""Figure F -- does the choice of checkpoint selector change the conclusion?

Every frozen intervention-effect cell, plotted as its effect on -dDP under
sigma_c^BCE against the same effect under sigma_c^AUC. Points on the y = x
diagonal are cells where the selector does not matter; distance from the
diagonal is how much it does.

Controlled and native cells are drawn in one panel, separated by marker fill,
so no cell is faded out to make another stand out.

The per-cell means are derived from the frozen per-cell CSVs. For the controlled
cells the difference is checked against the frozen D_selector entry, which is
exactly bce - auc, so those points are verified against the frozen bootstrap
rather than merely recomputed.

    python harness/experiments/x28_figures/fig_f_selector_scatter.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K                                                   # noqa: E402

NATIVE = [("FairGB", "bail"), ("FairGB", "credit"), ("FairGB", "german"),
          ("FairVGNN", "bail"), ("FairVGNN", "credit"), ("FairVGNN", "german"),
          ("NIFTY", "german")]


def main() -> int:
    K.rc()
    from analyze_armA import build
    ck, outs = K.Checks(), []
    src = K.Source("figF_selector_scatter")
    frozen = K.parse_armA_bootstrap()

    d = build(K.ARMA_CSVS)
    pts = []
    for ds in K.DATASETS:
        for m in K.METHODS:
            g = d[(d.method == m) & (d.dataset == ds)]
            b = g[g.selector == "common_bce"].int_ndp.mean()
            a = g[g.selector == "common_auc"].int_ndp.mean()
            fb = frozen[(frozen.dataset == ds) & (frozen.method == m)
                        & (frozen.quantity == "tau_int") & (frozen.coord == "-dDP")]
            fd = frozen[(frozen.dataset == ds) & (frozen.method == m)
                        & (frozen.quantity == "D_selector") & (frozen.coord == "-dDP")]
            ck.close(f"{m}/{ds} controlled BCE mean", b, fb.iloc[0]["mean"], 1e-4)
            ck.close(f"{m}/{ds} controlled D_selector = bce - auc", b - a,
                     fd.iloc[0]["mean"], 1e-4)
            pts.append(dict(method=m, dataset=ds, protocol="controlled",
                            bce=b, auc=a))
    for m, ds in NATIVE:
        n = pd.read_csv(f"{K.ROOT}/harness/results/armB_native_{m}_{ds}.csv")
        b = n[n.selector == "common_bce"].int_ndp.mean()
        a = n[n.selector == "common_auc"].int_ndp.mean()
        pts.append(dict(method=m, dataset=ds, protocol="native", bce=b, auc=a))

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    K.style(ax)
    vals = [p[k] for p in pts for k in ("bce", "auc")]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.10
    lim = (lo - pad, hi + pad)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.plot(lim, lim, color=K.BASEC, linewidth=0.9, linestyle=(0, (4, 3)),
            zorder=1)
    # both ends of the diagonal are wanted: the legend moves to the empty
    # lower-right quadrant so this label can keep the top-right end
    ax.annotate("y = x  (selector does not change the value)", (lim[1], lim[1]),
                textcoords="offset points", xytext=(-8, -10), ha="right",
                va="top", fontsize=K.FS["note"], color=K.MUTED)
    K.zero_lines(ax)
    for p in pts:
        col, mk = K.C[p["method"]], K.MK[p["method"]]
        native = p["protocol"] == "native"
        K.resolved_marker(ax, p["bce"], p["auc"], col, mk, native,
                          size=7.0 if native else 6.0, lw=1.4)
        if abs(p["bce"] - p["auc"]) > 0.05:
            ax.annotate(f"{p['method']} / {p['dataset']}\n({p['protocol']})",
                        (p["bce"], p["auc"]), textcoords="offset points",
                        xytext=(8, -2), fontsize=K.FS["note"], color=K.INK2,
                        ha="left", va="top")
        src.add(figure="F", **p, gap=p["bce"] - p["auc"])
    ck.limits(ax, [p["bce"] for p in pts], [p["auc"] for p in pts], "Figure F")
    ax.set_xlabel("intervention effect on −ΔDP, σ_c^BCE", color=K.INK2,
                  fontsize=K.FS["label"])
    ax.set_ylabel("intervention effect on −ΔDP, σ_c^AUC", color=K.INK2,
                  fontsize=K.FS["label"])
    handles = [plt.Line2D([], [], color=K.C[m], marker=K.MK[m], lw=0,
                          markersize=6, markerfacecolor=K.SURFACE,
                          markeredgecolor=K.C[m], label=m) for m in K.METHODS]
    handles += [plt.Line2D([], [], color=K.INK2, marker="o", lw=0, markersize=6,
                           markerfacecolor=K.SURFACE, markeredgecolor=K.INK2,
                           label="hollow = controlled"),
                plt.Line2D([], [], color=K.INK2, marker="o", lw=0, markersize=6.5,
                           markerfacecolor=K.INK2, markeredgecolor=K.INK2,
                           label="filled = native")]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2,
              fontsize=K.FS["note"], labelcolor=K.INK2)
    ax.set_title("The selector matters in some cells and not others",
                 color=K.INK, fontsize=K.FS["title"], loc="left", pad=15)
    ax.text(0, 1.015, "19 frozen cells · mean over 6 splits × 5 runs · "
            "cells more than 0.05 from the diagonal are named",
            transform=ax.transAxes, fontsize=K.FS["note"], color=K.MUTED,
            va="bottom")
    fig.tight_layout()
    K.save(fig, "figF_selector_scatter", outs)
    p = src.write()
    ok = ck.report()
    for o in outs + [p]:
        print("[written]", o)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
