"""Experiment 5 -- FMP component case study: how much of the change comes from graph propagation and how much
from the claimed fairness correction?

Four stages on FMP's own split and horizon (Pokec-z, Pokec-n; 30 units):

    B (common GNN) --base--> F00 (MLP) --prop--> F01 (+propagation) --fair--> F11 (+propagation+fairness)

    pkg = F11 - B = base + prop + fair           (identity checked on the point estimates)
    in the paper notation M^{-I} = F01, M^{+I} = F11, so fair = tau_{-I->+I}

No native lambda exists: lambda1 in {5, 30} (fairness) and lambda2 in {0.01, 20} (propagation) are the
pre-registered endpoints of the official search space, so every value is configuration-conditional.
F10 (fairness only) is not run, because lambda2 also scales the debiasing step.

    fig5_fmp_component_bridge          main: -Delta_DP, sigma_last; 2 x 2 (dataset x lambda2), one y-scale
    figS5_fmp_component_bridge_eo_auc  appendix: the same bridge for -Delta_EO and Delta AUC
    figS5_fmp_fair_by_selector         appendix: the fairness step under sigma_last / sigma_c^BCE / sigma_c^AUC

Resolved status is the CSV flag, never recomputed. FMP is excluded from every benchmark aggregate.

    python figures/src/make_fig5.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parents[1]           # output: figures/ (PDF only)
DATA = HERE.parent / "results"
d = pd.read_csv(DATA / "5_component_case_study_FMP.csv")
assert len(d) == 270, len(d)
as_true = lambda s: s.astype(str).str.strip().str.lower().eq("true")
d["res"] = as_true(d["resolved"])

plt.rcParams.update({
    "font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.labelcolor": "#0b0b0b", "ytick.labelcolor": "#0b0b0b",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "savefig.dpi": 300, "pdf.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})
apply_tone()
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e0", "#9a9993"
# roles (as in Fig. 4, kept apart from the family colours blue / orange / green)
C_LEVEL, C_BASE = ROLE["level"], ROLE["baseline_step"]
C_PROP, C_FAIR = ROLE["surrounding"], ROLE["intervention"]
COORD = {"negDP": r"$-\Delta_{\mathrm{DP}}$", "negEO": r"$-\Delta_{\mathrm{EO}}$", "dAUC": r"$\Delta$AUC"}
DS = {"pokec_z": "Pokec-z", "pokec_n": "Pokec-n"}
L1, L2 = (5.0, 30.0), (0.01, 20.0)


def tint(c, t=0.78):
    r, g, b = matplotlib.colors.to_rgb(c)
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


def get(ds, step, coord, sel="last", l1=0.0, l2=0.0):
    r = d[(d.dataset == ds) & (d.step == step) & (d.coordinate == coord) & (d.selector == sel)
          & np.isclose(d.lambda1_fairness, l1) & np.isclose(d.lambda2_propagation, l2)]
    assert len(r) == 1, (ds, step, coord, sel, l1, l2, len(r))
    return r.iloc[0]


for ds in DS:        # pkg = base + prop + fair on the point estimates, every coordinate / selector / lambda
    for c in COORD:
        for sel in ("last", "bce", "auc"):
            for l2 in L2:
                for l1 in L1:
                    v = (get(ds, "base", c, sel)["mean"] + get(ds, "prop", c, sel, 0.0, l2)["mean"]
                         + get(ds, "fair", c, sel, l1, l2)["mean"])
                    assert abs(v - get(ds, "pkg", c, sel, l1, l2)["mean"]) < 1e-6, (ds, c, sel, l1, l2)


def bar(ax, x, base, r, color, w, sgn=1):
    """one bar from `base` by r['mean']; whisker = the step's own 95% interval, anchored at `base`"""
    mean = r["mean"]
    end = base + mean
    ax.bar(x, mean, bottom=base, width=w, facecolor=_tint(color, TINT_RES),        # resolved is marked by the star only
           edgecolor=color, lw=OUTLINE, zorder=3)
    lo, hi = base + r["lo"], base + r["hi"]
    ax.plot([x, x], [lo, hi], color=INK, lw=0.6, zorder=4)
    for yy in (lo, hi):
        ax.plot([x - w * 0.18, x + w * 0.18], [yy, yy], color=INK, lw=0.6, zorder=4)
    return end, (lo, hi)


def label(ax, x, ci, end, base, r, fs=5.0, rot=0):
    up = end >= base
    y = ci[1] if up else ci[0]
    txt = f"{r['mean']:+.3f}".replace("-0.000", "0.000").replace("+0.000", "0.000") + (r" $\star$" if r["res"] else "")
    ax.annotate(txt, (x, y), xytext=(0, 2 if up else -2), textcoords="offset points", ha="center",
                va="bottom" if up else "top", fontsize=fs, rotation=rot,
                fontweight="bold" if r["res"] else "normal", zorder=6)


def bridge(ax, ds, l2, coord, sel="last", fs=4.7):
    """B (=0) -> base -> +prop -> +fair (lambda1 = 5, 30 side by side) -> pkg (lambda1 = 5, 30)"""
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    w, w2 = 0.62, 0.34
    rb = get(ds, "base", coord, sel)
    e_base, ci = bar(ax, 0, 0.0, rb, C_BASE, w)
    label(ax, 0, ci, e_base, 0.0, rb, fs)
    rp = get(ds, "prop", coord, sel, 0.0, l2)
    ax.plot([0 - w / 2, 1 + w / 2], [e_base] * 2, color=MUTED, lw=0.5, ls=(0, (2, 1.5)), zorder=2)
    e_prop, ci = bar(ax, 1, e_base, rp, C_PROP, w)
    label(ax, 1, ci, e_prop, e_base, rp, fs)
    ax.plot([1 - w / 2, 2 + w / 2], [e_prop] * 2, color=MUTED, lw=0.5, ls=(0, (2, 1.5)), zorder=2)
    for k, l1 in enumerate(L1):
        xf, xp = 2 + (k - 0.5) * (w2 + 0.02), 3 + (k - 0.5) * (w2 + 0.02)
        rf = get(ds, "fair", coord, sel, l1, l2)
        e_fair, ci = bar(ax, xf, e_prop, rf, C_FAIR, w2)
        label(ax, xf, ci, e_fair, e_prop, rf, fs - 0.4, rot=90)
        rk = get(ds, "pkg", coord, sel, l1, l2)
        e_pkg, ci = bar(ax, xp, 0.0, rk, C_LEVEL, w2)
        label(ax, xp, ci, e_pkg, 0.0, rk, fs - 0.4, rot=90)
    # lambda1 of the side-by-side bars as minor tick labels, the stage names underneath
    narrow = [2 + (k - 0.5) * (w2 + 0.02) for k in range(2)] + [3 + (k - 0.5) * (w2 + 0.02) for k in range(2)]
    ax.xaxis.remove_overlapping_locs = False
    ax.set_xticks(narrow, minor=True)
    ax.set_xticklabels(["5", "30"] * 2, minor=True, fontsize=5.8)
    ax.tick_params(axis="x", which="minor", length=0, pad=1.5)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["base", "+prop", "+fair", "pkg"], fontsize=7.2)
    ax.tick_params(axis="x", which="major", length=0, pad=7)
    ax.set_xlim(-0.55, 3.55)


def legend(fig, y, ncol=6):
    handles = [Patch(facecolor=_tint(C_BASE, TINT_RES), edgecolor=C_BASE, label=r"base: B → $M^{\mathrm{base}}$"),
               Patch(facecolor=_tint(C_PROP, TINT_RES), edgecolor=C_PROP, label=r"+prop: $M^{\mathrm{base}}$ → $M^{\mathrm{prop}}$"),
               Patch(facecolor=_tint(C_FAIR, TINT_RES), edgecolor=C_FAIR,
                     label=r"+fair: $M^{\mathrm{prop}}$ → $M^{\mathrm{prop+fair}}$  ($\tau_{{-}I\rightarrow {+}I}$)"),
               Patch(facecolor=_tint(C_LEVEL, TINT_RES), edgecolor=C_LEVEL, label=r"pkg: B → $M^{\mathrm{prop+fair}}$"),
               Line2D([], [], ls="none", label=r"$\star$ = resolved"),
               Line2D([], [], ls="none", label=r"paired bars: $\lambda_1$ = 5 | 30")]
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=True,
                     fancybox=False, edgecolor="#d0cfca", framealpha=1.0, fontsize=6.6, handlelength=1.2,
                     columnspacing=0.8, handletextpad=0.35)
    leg.get_frame().set_linewidth(0.4)


def fit_labels(fig, row, pad=0.02):
    """set the row's y-limits to the data extent plus exactly what the value labels need"""
    for _ in range(3):
        fig.canvas.draw()
        lo, hi = row[0].get_ylim()
        need_lo, need_hi = lo, hi
        for ax in row:
            inv = ax.transData.inverted()
            for t in ax.texts:
                bb = t.get_window_extent()
                (_, y0), (_, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
                need_lo, need_hi = min(need_lo, y0), max(need_hi, y1)
        span = need_hi - need_lo
        new = (need_lo - pad * span, need_hi + pad * span)
        if np.allclose(new, (lo, hi), atol=1e-6 * span):
            break
        row[0].set_ylim(*new)


def grid_of_bridges(coords, name, row_h):
    """one row per coordinate, four panels per row: (Pokec-z, Pokec-n) x (lambda2 = 0.01, 20)"""
    cols = [(ds, l2) for ds in DS for l2 in L2]
    fig, axes = plt.subplots(len(coords), 4, figsize=(7.0, row_h * len(coords) + 0.35), sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    for i, c in enumerate(coords):
        for j, (ds, l2) in enumerate(cols):
            ax = axes[i, j]
            bridge(ax, ds, l2, c)
            tag = "abcdefgh"[i * 4 + j]
            ax.set_title(f"({tag})  {DS[ds]},  " + r"$\lambda_2$" + f" = {l2:g}", loc="left", fontsize=8.2)
            if j == 0:
                # bar ends sit at cumulative levels from B; bar heights (and their numbers) are the stage steps
                ax.set_ylabel("Component contrast in " + COORD[c] + r" ($\sigma_{\mathrm{last}}$)")
            else:
                ax.tick_params(axis="y", labelleft=False)
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    for i in range(len(coords)):                       # tight y (shared in the row), widened only to fit the labels
        fit_labels(fig, axes[i])
    legend(fig, 0.0, ncol=7)
    for ext in ("pdf",):
        fig.savefig(HERE / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    print("saved", HERE / f"{name}.pdf")


grid_of_bridges(["negDP"], "fig5_fmp_component_bridge", 2.25)
grid_of_bridges(["negEO", "dAUC"], "figS5_fmp_component_bridge_eo_auc", 2.25)

# --------------------------------------------------------------- appendix: the fairness step by selector
SEL = [("last", r"$\sigma_{\mathrm{last}}$", "o"), ("bce", r"$\sigma_c^{\mathrm{BCE}}$", "s"),
       ("auc", r"$\sigma_c^{\mathrm{AUC}}$", "^")]
CFG = [(l1, l2) for l2 in L2 for l1 in L1]
fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6), sharey=True)
for i, ds in enumerate(DS):
    for j, c in enumerate(COORD):
        ax = axes[i, j]
        ax.axvspan(-0.01, 0.01, color="#efeeea", lw=0, zorder=0)       # |mean| < 0.010: never resolved
        ax.axvline(0, color=INK2, lw=0.6, zorder=1)
        for k, (l1, l2) in enumerate(CFG):
            for s, (sel, _, mk) in enumerate(SEL):
                r = get(ds, "fair", c, sel, l1, l2)
                y = len(CFG) - 1 - k + (1 - s) * 0.22
                ax.plot([r["lo"], r["hi"]], [y, y], color=C_FAIR, lw=0.8, zorder=2)
                ax.plot(r["mean"], y, marker=mk, ms=3.6, color=C_FAIR, mfc=C_FAIR if r["res"] else "white",
                        mew=0.8, ls="none", zorder=3)
        ax.set_yticks(range(len(CFG)))
        ax.set_yticklabels([r"$\lambda_1$=" + f"{l1:g}, " + r"$\lambda_2$=" + f"{l2:g}" for l1, l2 in CFG][::-1],
                           fontsize=7)
        ax.tick_params(axis="y", length=0)
        if c == "dAUC":                       # Delta AUC intervals lie within [-0.0022, +0.0022]: own, narrower range
            ax.set_xlim(-0.0025, 0.0025)
            ax.set_xticks([-0.002, 0, 0.002])
            ax.set_xticklabels(["−0.002", "0", "+0.002"])
        else:                                 # fairness coordinates lie within [-0.0112, +0.0098]
            ax.set_xlim(-0.0125, 0.0125)
            ax.set_xticks([-0.01, 0, 0.01])
        if i == 0:
            ax.set_title(COORD[c], loc="left", fontsize=8.8)
        if i == 1:
            ax.set_xlabel(r"fairness step  $\tau_{{-}I\rightarrow {+}I}$ = F11 − F01")
        if j == 0:
            ax.set_ylabel(DS[ds], fontsize=8.8)
n_res = int(d[d.step.eq("fair")]["res"].sum())
assert n_res == 0, n_res          # the claim this figure carries
handles = [Line2D([], [], marker=mk, color=C_FAIR, mfc="white", ms=3.6, lw=0.8, label=lab) for _, lab, mk in SEL]
handles += [Patch(facecolor="#efeeea", label=r"$|\mathrm{mean}| < 0.010$ (below the resolution threshold)"),
            Line2D([], [], color=C_FAIR, lw=0.8, label="95% interval; open = unresolved (all 72)")]
leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=True,
                 fancybox=False, edgecolor="#d0cfca", framealpha=1.0, fontsize=7, handlelength=1.5,
                 columnspacing=1.0)
leg.get_frame().set_linewidth(0.4)
fig.tight_layout(h_pad=0.8, w_pad=0.8)
for ext in ("pdf",):
    fig.savefig(HERE / f"figS5_fmp_fair_by_selector.{ext}", bbox_inches="tight", pad_inches=0.02)
print("saved", HERE / "figS5_fmp_fair_by_selector.pdf")
