"""X27 figures: FMP mechanistic decomposition (Pokec-z, Pokec-n).

Reads only the analyzer outputs. Datasets are never mixed into one curve: each
gets its own panel.

Figure 1  F00 -> F01 -> F11 vector decomposition in the dAUC x -dDP plane: one
          propagation vector per lambda2, then the fairness-correction vector
          added at that same propagation state, per lambda1.
Figure 2  tau_fair(-dDP) over the preregistered grid (x = lambda2, y = lambda1),
          value as text with the interval, and resolution shown by marker border
          and an explicit label -- never by colour alone.
Supplement  the -dEO version and the selector-robustness comparison.

    python harness/experiments/plot_x27_fmp.py --indir harness/results/x27 --outdir harness/results/figures
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
PROP = "#2a78d6"      # categorical slot 1: propagation vectors
FAIR = "#eb6834"      # categorical slot 2: fairness-correction vectors
LAM1 = (5.0, 30.0)
LAM2 = (0.01, 20.0)
DATASETS = ("pokec_z", "pokec_n")
LAB = {"auc": "τ (ΔAUC)", "ndp": "τ (−ΔDP)", "neo": "τ (−ΔEO)"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK2, width=1, length=3)
    ax.grid(color=GRID_C, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def q(s, sel, name):
    r = s[(s.selector == sel) & (s.quantity == name)]
    if r.empty:
        raise KeyError(f"{sel} {name}")
    return r.iloc[0]


def fig1(summaries, outdir, outs, coord="ndp"):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12.4, 5.4))
    for ax, ds in zip(np.atleast_1d(axes), DATASETS):
        s = summaries[ds]
        style(ax)
        ax.axhline(0, color=BASE, linewidth=1)
        ax.axvline(0, color=BASE, linewidth=1)
        ax.plot(0, 0, marker="o", markersize=9, markerfacecolor=SURFACE,
                markeredgecolor=INK, markeredgewidth=2, zorder=5)
        # the propagation vectors leave the origin downward or sideways, so F00's
        # label goes straight left of the origin, clear of every vector
        ax.annotate("F00 (MLP)", (0, 0), textcoords="offset points", xytext=(-14, 4),
                    fontsize=9, color=INK2, ha="right", va="bottom")
        xs = [0.0]
        for li, l2 in enumerate(LAM2):
            px = q(s, "last", f"prop.l2{l2:g}.auc")["mean"]
            py = q(s, "last", f"prop.l2{l2:g}.{coord}")["mean"]
            ax.annotate("", xy=(px, py), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=PROP, lw=2,
                                        linestyle="-" if li == 0 else "--"))
            # the F01 state is drawn as a real marker, not only as an arrow head:
            # annotate() arrows are invisible to autoscale, so without this a
            # propagation endpoint can fall outside the drawn range.
            ax.plot(px, py, marker="o", markersize=8, zorder=5,
                    markerfacecolor=PROP if q(s, "last", f"prop.l2{l2:g}.{coord}")["resolved"]
                    else SURFACE, markeredgecolor=PROP, markeredgewidth=2)
            pts = [(px, py)]
            for lj, l1 in enumerate(LAM1):
                fx = px + q(s, "last", f"fair.l1{l1:g}.l2{l2:g}.auc")["mean"]
                fy = py + q(s, "last", f"fair.l1{l1:g}.l2{l2:g}.{coord}")["mean"]
                ax.annotate("", xy=(fx, fy), xytext=(px, py),
                            arrowprops=dict(arrowstyle="-|>", color=FAIR, lw=2,
                                            linestyle="-" if lj == 0 else "--"))
                res = q(s, "last", f"fair.l1{l1:g}.l2{l2:g}.{coord}")["resolved"]
                ax.plot(fx, fy, marker="s", markersize=8, zorder=5,
                        markerfacecolor=FAIR if res else SURFACE,
                        markeredgecolor=FAIR, markeredgewidth=2)
                # no per-lambda1 labels here: the fairness vectors are two orders
                # of magnitude shorter than the propagation vector, so the four
                # F11 endpoints coincide at this scale. The grid figure carries
                # every value and interval.
                pts.append((fx, fy))
            xs += [pt[0] for pt in pts]
            # one label per lambda2 group, anchored on the group's topmost point
            # and offset away from the bulk of the data in x
            lx, ly = max(pts, key=lambda t: t[1])
            right = px < (min(xs) + max(xs)) / 2
            ax.annotate(f"F01, F11  λ2={l2:g}", (lx, ly), textcoords="offset points",
                        xytext=(12 if right else -12, 14), fontsize=8.5, color=INK2,
                        ha="left" if right else "right")
        ax.set_xlabel("ΔAUC  (utility)", color=INK2)
        ax.set_ylabel(LAB[coord] + "  (fairness; > 0 better)", color=INK2)
        ax.margins(0.22)
        ax.set_title(f"FMP / {ds}", color=INK, fontsize=11, loc="left")
        mx = max(abs(q(s, "last", f"fair.l1{l1:g}.l2{l2:g}.{coord}")["mean"])
                 for l1 in LAM1 for l2 in LAM2)
        ax.text(0.99, 0.02, f"fairness vectors ≤ {mx:.3f} in this plane;\n"
                "values and intervals in the grid figure",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
    handles = [plt.Line2D([], [], color=PROP, lw=2, marker="o", markersize=8,
                          markerfacecolor=SURFACE, markeredgecolor=PROP,
                          label="propagation: F00 → F01(λ2)"),
               plt.Line2D([], [], color=FAIR, lw=2, label="fairness correction: F01 → F11(λ1, λ2)"),
               plt.Line2D([], [], color=FAIR, marker="s", lw=0, markerfacecolor=FAIR,
                          markeredgecolor=FAIR, label="filled = resolved"),
               plt.Line2D([], [], color=FAIR, marker="s", lw=0, markerfacecolor=SURFACE,
                          markeredgecolor=FAIR, label="hollow = unresolved")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK2)
    fig.suptitle("FMP mechanistic decomposition: propagation then fairness correction, "
                 "σ_last, mean over 6 splits × 5 runs", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save(fig, outdir, f"x27_fmp_vectors_{coord}", outs)


def fig2(summaries, outdir, outs, coord="ndp", selector="last"):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12.4, 4.6))
    for ax, ds in zip(np.atleast_1d(axes), DATASETS):
        s = summaries[ds]
        style(ax)
        ax.set_xticks(range(len(LAM2)));  ax.set_xticklabels([f"λ2 = {v:g}" for v in LAM2])
        ax.set_yticks(range(len(LAM1)));  ax.set_yticklabels([f"λ1 = {v:g}" for v in LAM1])
        ax.set_xlim(-0.6, len(LAM2) - 0.4); ax.set_ylim(-0.6, len(LAM1) - 0.4)
        for i, l2 in enumerate(LAM2):
            for j, l1 in enumerate(LAM1):
                r = q(s, selector, f"fair.l1{l1:g}.l2{l2:g}.{coord}")
                res = bool(r["resolved"])
                ax.plot(i, j, marker="s", markersize=46, zorder=2,
                        markerfacecolor=SURFACE,
                        markeredgecolor=FAIR if res else MUTED,
                        markeredgewidth=3 if res else 1.5)
                ax.annotate(f"{r['mean']:+.3f}\n[{r['lo']:+.3f}, {r['hi']:+.3f}]\n"
                            f"{'resolved' if res else 'unresolved'}",
                            (i, j), ha="center", va="center", fontsize=8, color=INK2, zorder=3)
        ax.set_title(f"FMP / {ds}", color=INK, fontsize=11, loc="left")
    fig.suptitle(f"τ_fair({LAB[coord].split('(')[1].rstrip(')')}) over the preregistered grid "
                 f"endpoints, selector σ_{selector}  (thick border = resolved; text carries the "
                 f"value and interval)", color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save(fig, outdir, f"x27_fmp_grid_{coord}_{selector}", outs)


def fig_selector(summaries, outdir, outs, coord="ndp"):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12.4, 4.8))
    marks = {"last": ("o", "#2a78d6"), "bce": ("s", "#eb6834"), "auc": ("^", "#1baf7a")}
    for ax, ds in zip(np.atleast_1d(axes), DATASETS):
        s = summaries[ds]
        style(ax)
        ax.axhline(0, color=BASE, linewidth=1)
        labels = []
        for k, (l2, l1) in enumerate([(l2, l1) for l2 in LAM2 for l1 in LAM1]):
            labels.append(f"λ1={l1:g}\nλ2={l2:g}")
            for m, (sel, (mk, col)) in enumerate(marks.items()):
                r = q(s, sel, f"fair.l1{l1:g}.l2{l2:g}.{coord}")
                x = k + (m - 1) * 0.22
                ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                            fmt="none", ecolor=col, elinewidth=1.5, zorder=2)
                ax.plot(x, r["mean"], marker=mk, markersize=8, zorder=4,
                        markerfacecolor=col if r["resolved"] else SURFACE,
                        markeredgecolor=col, markeredgewidth=2)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylabel(f"τ_fair {LAB[coord]}", color=INK2)
        ax.set_title(f"FMP / {ds}", color=INK, fontsize=11, loc="left")
    handles = [plt.Line2D([], [], color=c, marker=m, lw=0, markersize=8, label=f"σ_{k}")
               for k, (m, c) in marks.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK2)
    fig.suptitle("Selector robustness of the fairness-correction contrast "
                 "(filled = resolved; σ_last is primary)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    save(fig, outdir, f"x27_fmp_selector_{coord}", outs)


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
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    summaries = {ds: pd.read_csv(os.path.join(a.indir, f"x27_{ds}_summary.csv"))
                 for ds in DATASETS}
    outs = []
    fig1(summaries, a.outdir, outs, "ndp")
    fig2(summaries, a.outdir, outs, "ndp", "last")
    fig1(summaries, a.outdir, outs, "neo")
    fig2(summaries, a.outdir, outs, "neo", "last")
    fig_selector(summaries, a.outdir, outs, "ndp")
    for p in outs:
        print("[written]", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
