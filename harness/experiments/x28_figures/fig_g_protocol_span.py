"""Figure G -- the span of the intervention effect across the protocols we ran.

For each method-dataset cell, the smallest and largest tau_I(-dDP) observed over
the protocol points this project actually evaluated. Every point is drawn, so
the span is visibly the range of a handful of measurements.

This is NOT an identified set and NOT full protocol uncertainty. The protocol
points here are sparse and were chosen for other reasons; a cell with one point
has a span of zero because it was measured once, not because it is stable.

    python harness/experiments/x28_figures/fig_g_protocol_span.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K                                                   # noqa: E402

PROTO_MK = {"controlled H=200": "o", "native": "s"}


def main() -> int:
    K.rc()
    ck, outs = K.Checks(), []
    src = K.Source("figG_protocol_span")
    arma = K.parse_armA_bootstrap()
    armb = K.parse_armB_seven()

    cells = []
    for ds in K.DATASETS:
        for m in K.METHODS:
            pts = []
            r = arma[(arma.dataset == ds) & (arma.method == m)
                     & (arma.quantity == "tau_int") & (arma.coord == "-dDP")].iloc[0]
            pts.append(("controlled H=200", float(r["mean"])))
            n = armb[(armb.dataset == ds) & (armb.method == m)
                     & (armb.quantity == "tau_int") & (armb.coord == "-dDP")]
            if len(n):
                pts.append(("native", float(n.iloc[0]["n_mean"])))
            cells.append(dict(method=m, dataset=ds, pts=pts))
    ck.true("7 cells have a native protocol point",
            sum(len(c["pts"]) == 2 for c in cells) == 7)

    order = sorted(cells, key=lambda c: -(max(p[1] for p in c["pts"])
                                          - min(p[1] for p in c["pts"])))
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    K.style(ax, grid="x")
    y = np.arange(len(order))[::-1]
    xs, ticks = [], []
    for c, yy in zip(order, y):
        col = K.C[c["method"]]
        vals = [p[1] for p in c["pts"]]
        if len(vals) > 1:
            ax.plot([min(vals), max(vals)], [yy, yy], color=col, linewidth=1.6,
                    alpha=0.45, zorder=2, solid_capstyle="round")
        for name, v in c["pts"]:
            ax.plot(v, yy, marker=PROTO_MK[name], markersize=6.0,
                    linestyle="none", markerfacecolor=col if name == "native"
                    else K.SURFACE, markeredgecolor=col, markeredgewidth=1.4,
                    zorder=4)
        xs += vals
        ticks.append(f"{c['method']} / {c['dataset']}")
        src.add(figure="G", method=c["method"], dataset=c["dataset"],
                n_protocol_points=len(c["pts"]),
                span_min=min(vals), span_max=max(vals),
                span_width=max(vals) - min(vals),
                **{f"point_{n.replace(' ', '_').replace('=', '')}": v
                   for n, v in c["pts"]})
    ax.axvline(0, color=K.BASEC, linewidth=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ticks, fontsize=K.FS["note"] + 0.5)
    ax.set_ylim(-0.7, len(order) - 0.3)
    lo, hi = min(xs), max(xs)
    m = (hi - lo) * 0.10
    ax.set_xlim(lo - m, hi + m)
    ck.limits(ax, xs, y, "Figure G")
    ax.set_xlabel("intervention effect on −ΔDP", color=K.INK2,
                  fontsize=K.FS["label"])
    handles = [plt.Line2D([], [], color=K.INK2, marker="o", lw=0, markersize=6,
                          markerfacecolor=K.SURFACE, markeredgecolor=K.INK2,
                          label="controlled H = 200"),
               plt.Line2D([], [], color=K.INK2, marker="s", lw=0, markersize=6,
                          markerfacecolor=K.INK2, markeredgecolor=K.INK2,
                          label="native protocol")]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              fontsize=K.FS["note"], labelcolor=K.INK2)
    ax.set_title("Observed intervention-effect span across evaluated protocols",
                 color=K.INK, fontsize=K.FS["title"], loc="left", pad=15)
    ax.text(0, 1.015, "every evaluated point is drawn · five cells have one "
            "protocol point only, so their span is zero by construction",
            transform=ax.transAxes, fontsize=K.FS["note"], color=K.MUTED,
            va="bottom")
    fig.tight_layout()
    K.save(fig, "figG_protocol_span", outs)
    p = src.write()
    ok = ck.report()
    for o in outs + [p]:
        print("[written]", o)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
