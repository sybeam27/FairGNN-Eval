"""Figure E -- FMP component decomposition, paper style.

Two ordered components of one package, at a stated configuration:

    propagation        F00 -> F01(lambda2)
    fairness correction F01 -> F11(lambda1, lambda2)

Top row: the two vectors at true scale in the (dAUC, -dDP) plane, F00 at the
origin. The F01 state is drawn as a real marker -- annotate() arrows are
invisible to autoscale, and an earlier version of this figure lost an endpoint
that way.

Bottom row: the fairness-correction component alone, with its intervals, since
at true scale it is two orders of magnitude shorter than the propagation vector.
That is a zoom of the same numbers, never a substitute for the top row.

FMP has no native lambda, so every quantity here is configuration-conditional.

    python harness/experiments/x28_figures/fig_e_fmp_components.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K                                                   # noqa: E402

DATASETS = ("pokec_z", "pokec_n")
LAM1 = (5.0, 30.0)
LAM2 = (0.01, 20.0)
SEL = "last"
PROP_C, FAIR_C = "#2a78d6", "#eb6834"


def limits(vals, pad=0.16):
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-12:
        lo, hi = lo - 1e-3, hi + 1e-3
    m = (hi - lo) * pad
    return lo - m, hi + m


def figure_e(sums, src, ck, outs, coord="ndp"):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4),
                             gridspec_kw=dict(height_ratios=[1.55, 1.0]))
    for c, ds in enumerate(DATASETS):
        s = sums[ds]
        ax = axes[0][c]
        K.style(ax)
        xs, ys = [0.0], [0.0]
        pts = []
        for l2 in LAM2:
            px = K.q(s, f"prop.l2{l2:g}.auc", SEL)
            py = K.q(s, f"prop.l2{l2:g}.{coord}", SEL)
            grp = [(px["mean"], py["mean"])]
            for l1 in LAM1:
                fx = px["mean"] + K.q(s, f"fair.l1{l1:g}.l2{l2:g}.auc", SEL)["mean"]
                fy = py["mean"] + K.q(s, f"fair.l1{l1:g}.l2{l2:g}.{coord}", SEL)["mean"]
                grp.append((fx, fy))
            pts.append((l2, px, py, grp))
            xs += [p[0] for p in grp]
            ys += [p[1] for p in grp]
        ax.set_xlim(*limits(xs))
        ax.set_ylim(*limits(ys))
        K.zero_lines(ax)

        ax.plot(0, 0, marker="o", markersize=6.5, linestyle="none",
                markerfacecolor=K.SURFACE, markeredgecolor=K.INK,
                markeredgewidth=1.5, zorder=6)
        ax.annotate("F00 (MLP)", (0, 0), textcoords="offset points",
                    xytext=(-10, 4), ha="right", va="bottom",
                    fontsize=K.FS["note"], color=K.INK2)
        for i, (l2, px, py, grp) in enumerate(pts):
            ls = "-" if i == 0 else (0, (4, 2))
            ax.annotate("", xy=grp[0], xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.32",
                                        color=PROP_C, linewidth=1.6, linestyle=ls,
                                        shrinkA=0, shrinkB=0), zorder=3)
            K.resolved_marker(ax, grp[0][0], grp[0][1], PROP_C, "o",
                              bool(py["resolved"]), size=6.0, lw=1.4)
            for j, end in enumerate(grp[1:]):
                ax.annotate("", xy=end, xytext=grp[0],
                            arrowprops=dict(arrowstyle="-|>,head_width=0.14,head_length=0.28",
                                            color=FAIR_C, linewidth=1.4,
                                            linestyle="-" if j == 0 else (0, (3, 2)),
                                            shrinkA=0, shrinkB=0), zorder=4)
                r = K.q(s, f"fair.l1{LAM1[j]:g}.l2{l2:g}.{coord}", SEL)
                K.resolved_marker(ax, end[0], end[1], FAIR_C, "s",
                                  bool(r["resolved"]), size=5.2, lw=1.3)
            top = max(grp, key=lambda t: t[1])
            right = grp[0][0] < (min(xs) + max(xs)) / 2
            ax.annotate(f"λ2 = {l2:g}", top, textcoords="offset points",
                        xytext=(9 if right else -9, 9), fontsize=K.FS["note"],
                        color=K.INK2, ha="left" if right else "right")
            src.add(figure="E", dataset=ds, coord=coord, component="propagation",
                    lam2=l2, x_dAUC=px["mean"], y=py["mean"], lo=py["lo"],
                    hi=py["hi"], resolved=bool(py["resolved"]))
        ck.limits(ax, xs, ys, f"Figure E top / {ds}")
        ax.set_title(f"FMP / {ds}", color=K.INK, fontsize=K.FS["panel"],
                     loc="left", pad=4)
        ax.set_xlabel("change in AUC", color=K.INK2, fontsize=K.FS["label"])
        if c == 0:
            ax.set_ylabel("change in demographic parity\n(−ΔDP; higher is fairer)"
                          if coord == "ndp" else coord,
                          color=K.INK2, fontsize=K.FS["label"])

        # ---- secondary panel: the fairness component alone, with intervals
        ax2 = axes[1][c]
        K.style(ax2, grid="x")
        labs, xs2 = [], []
        cells = [(l2, l1) for l2 in LAM2 for l1 in LAM1]
        y = np.arange(len(cells))[::-1]
        for (l2, l1), yy in zip(cells, y):
            r = K.q(s, f"fair.l1{l1:g}.l2{l2:g}.{coord}", SEL)
            ax2.plot([r["lo"], r["hi"]], [yy, yy], color=FAIR_C, linewidth=1.3,
                     alpha=0.8, zorder=2, solid_capstyle="butt")
            K.resolved_marker(ax2, r["mean"], yy, FAIR_C, "s",
                              bool(r["resolved"]), size=5.5, lw=1.3)
            labs.append(f"λ1={l1:g}, λ2={l2:g}")
            xs2 += [r["lo"], r["hi"]]
            src.add(figure="E_zoom", dataset=ds, coord=coord,
                    component="fairness_correction", lam1=l1, lam2=l2,
                    mean=r["mean"], lo=r["lo"], hi=r["hi"],
                    resolved=bool(r["resolved"]))
        ax2.axvline(0, color=K.BASEC, linewidth=0.9, zorder=1)
        ax2.set_yticks(y)
        ax2.set_yticklabels(labs, fontsize=K.FS["note"])
        ax2.set_ylim(-0.7, len(cells) - 0.3)
        ax2.set_xlim(*limits(xs2, 0.12))
        ck.limits(ax2, xs2, y, f"Figure E zoom / {ds}")
        ax2.set_xlabel("fairness-correction component alone (−ΔDP)",
                       color=K.INK2, fontsize=K.FS["label"])

    handles = [plt.Line2D([], [], color=PROP_C, lw=1.6, marker="o", markersize=5.5,
                          markerfacecolor=PROP_C, markeredgecolor=PROP_C,
                          label="propagation  F00 → F01(λ2)"),
               plt.Line2D([], [], color=FAIR_C, lw=1.4, marker="s", markersize=5,
                          markerfacecolor=FAIR_C, markeredgecolor=FAIR_C,
                          label="fairness correction  F01 → F11(λ1, λ2)"),
               plt.Line2D([], [], color=K.INK2, marker="s", lw=0, markersize=5,
                          markerfacecolor=K.SURFACE, markeredgecolor=K.INK2,
                          label="hollow = unresolved")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=K.FS["note"], labelcolor=K.INK2, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("FMP: which component moves fairness", color=K.INK,
                 fontsize=K.FS["title"], x=0.008, ha="left")
    fig.text(0.008, 0.935, "σ_last · 6 splits × 5 runs · no native λ exists, so "
             "every value is conditional on the stated configuration",
             fontsize=K.FS["note"], color=K.MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.925))
    K.save(fig, f"figE_fmp_components_{coord}", outs)


def main() -> int:
    K.rc()
    sums = {ds: K.summary(p) for ds, p in K.X27_SUMMARY.items()}
    ck, outs = K.Checks(), []
    # the registered identity, re-checked on the values this figure plots
    for ds, s in sums.items():
        for l2 in LAM2:
            for l1 in LAM1:
                for c in ("ndp", "auc"):
                    p = K.q(s, f"prop.l2{l2:g}.{c}", SEL)["mean"]
                    f = K.q(s, f"fair.l1{l1:g}.l2{l2:g}.{c}", SEL)["mean"]
                    t = K.q(s, f"total.l1{l1:g}.l2{l2:g}.{c}", SEL)["mean"]
                    ck.close(f"{ds} λ1={l1:g} λ2={l2:g} {c}: total = prop + fair",
                             p + f, t, 1e-9)
    src = K.Source("figE_fmp_components")
    figure_e(sums, src, ck, outs, "ndp")
    p = src.write()
    ok = ck.report()
    for o in outs + [p]:
        print("[written]", o)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
