"""Experiment 4 -- where does the controlled -> native shift of tau_{-I->+I}(P) come from?

Two frozen cells, each re-run at its native horizon (NIFTY/German, native drop rates D1, H = 1000;
FairGB/Bail, H = 1500), 30 units. On the SAME long trajectories the selector is allowed either the first 200
epochs (cap) or the whole run (full):

    tau_H200 --R_traj--> tau_cap --E_{+I}--> . --(-E_{-I})--> tau_full
    S_sel = full - cap = E_{+I} - E_{-I},   H_shift = S_sel + R_traj     (CSV terms: T = R_traj, S = S_sel, E1/E0)

R_traj = prefix/run difference (incl. GPU nondeterminism); E_{+I} / E_{-I} = how far moving M^{+I} / M^{-I} from its cap
checkpoint to its full-support checkpoint changes the contrast (an algebraic identity, not mediation).

    fig4_selection_support_bridge      main: bridge on -Delta_DP, sigma_c^BCE, one panel per cell
    figS4_fixed_epoch_trajectory       appendix: tau_{-I->+I} at fixed epochs (no selection), 3 coordinates x
                                       {NIFTY D0, NIFTY D1, FairGB}; the cap (epoch 200) is marked; a strip
                                       underneath shows where the validation-BCE selector lands over the full
                                       support, per unit (source: harness/results/x25, x26 selected-epoch files)

Resolved status is the CSV flag, never recomputed. Not claimed: that selection causes unfairness, that the BCE
selector is wrong, mediation, or anything beyond these two cells.

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parents[1]           # output: figures/ (PDF only)
DATA = HERE.parent / "results"
d = pd.read_csv(DATA / "4_mechanistic_case_study.csv")
assert len(d) == 255, len(d)
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
# roles, kept apart from the family colours (blue / orange / green) and from Figs. 2-3
C_LEVEL, C_T = ROLE["level"], ROLE["run_noise"]
C_E1, C_E0 = ROLE["intervention"], ROLE["surrounding"]
TAU = r"\tau^{%s}_{{-}I\rightarrow {+}I}"
COORD_SUP = {"negDP": r"-\Delta_{\mathrm{DP}}", "negEO": r"-\Delta_{\mathrm{EO}}", "dAUC": r"\Delta\mathrm{AUC}"}


def tint(c, t=0.78):
    r, g, b = matplotlib.colors.to_rgb(c)
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


def get(method, term, coord="negDP"):
    r = d[(d.method == method) & (d.term == term) & (d.coordinate == coord)]
    assert len(r) == 1, (method, term, coord, len(r))
    return r.iloc[0]


# term names per cell (sigma_c^BCE)
CELLS = [("NIFTY", "NIFTY / German  (published drop rates, H = 1000)",
          dict(h200="D1.bce.h200", T="D1.bce.T", cap="D1.bce.c200", E1="D1.tele.m1ext", E0="D1.tele.m0ext",
               full="D1.bce.c1000", S="D1.bce.S", H="D1.bce.H")),
         ("FairGB", "FairGB / Bail  (H = 1500)",
          dict(h200="bce.h200", T="bce.T", cap="bce.cap", E1="bce.E1", E0="bce.E0", full="bce.full",
               S="bce.S", H="bce.Hshift"))]

for m, _, t in CELLS:          # identities on the point estimates
    for c in COORD_SUP:
        v = {k: get(m, n, c)["mean"] for k, n in t.items()}
        assert abs(v["h200"] + v["T"] - v["cap"]) < 1e-6, (m, c, "T")
        assert abs(v["cap"] + v["E1"] - v["E0"] - v["full"]) < 1e-6, (m, c, "E")
        assert abs(v["E1"] - v["E0"] - v["S"]) < 1e-6 and abs(v["S"] + v["T"] - v["H"]) < 1e-6, (m, c, "S/H")


def bridge(ax, method, title, t, coord="negDP"):
    g = {k: get(method, n, coord) for k, n in t.items()}
    steps = [("level", r"$\tau_{\mathrm{H200}}$", g["h200"], 1),
             ("T", r"$R_{\mathrm{traj}}$", g["T"], 1),
             ("level", r"$\tau_{\mathrm{cap}}$", g["cap"], 1),
             ("E1", r"$E_{+I}$", g["E1"], 1),
             ("E0", r"$-E_{-I}$", g["E0"], -1),
             ("level", r"$\tau_{\mathrm{full}}$", g["full"], 1)]
    color = {"level": C_LEVEL, "T": C_T, "E1": C_E1, "E0": C_E0}
    w = 0.62
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    run = 0.0
    tops = []
    for i, (kind, lab, r, sgn) in enumerate(steps):
        c = color[kind]
        mean, lo, hi = sgn * r["mean"], *(sorted((sgn * r["lo"], sgn * r["hi"])))
        fc = _tint(c, TINT_RES) if r["res"] else _tint(c, TINT)
        if kind == "level":
            base, end = 0.0, mean
            run = mean
            ci = (lo, hi)
        else:
            base, end = run, run + mean
            ci = (run + lo, run + hi)
            ax.plot([i - 1 - w / 2, i + w / 2], [run, run], color=MUTED, lw=0.5, ls=(0, (2, 1.5)), zorder=2)
            run = end
        ax.bar(i, end - base, bottom=base, width=w, facecolor=fc, edgecolor=c, lw=OUTLINE, zorder=3)
        ax.plot([i, i], ci, color=INK, lw=0.7, zorder=4)
        ax.plot([i - 0.08, i + 0.08], [ci[0]] * 2, color=INK, lw=0.7, zorder=4)
        ax.plot([i - 0.08, i + 0.08], [ci[1]] * 2, color=INK, lw=0.7, zorder=4)
        up = end >= base
        yv = ci[1] if up else ci[0]
        ax.text(i, yv + (0.008 if up else -0.008), f"{mean:+.3f}".replace("-0.000", "0.000") + (r" $\star$" if r["res"] else ""), ha="center", va="bottom" if up else "top",
                fontsize=6.6, fontweight="bold" if r["res"] else "normal")
        tops.append((ci, end))
    # S bracket over E1 and -E0
    S = g["S"]
    lo_all = min(min(tops[3][0]), min(tops[4][0]), tops[2][1])
    hi_all = max(max(tops[3][0]), max(tops[4][0]), tops[2][1])
    yb = hi_all + 0.035 if S["mean"] < 0 else lo_all - 0.035
    ax.plot([2.7, 2.7, 4.3, 4.3], [yb - 0.008 * np.sign(yb - 0), yb, yb, yb - 0.008 * np.sign(yb - 0)],
            color=INK2, lw=0.6)
    ax.text(3.5, yb + (0.006 if yb > 0 else -0.006),
            r"$S_{\mathrm{sel}}$" + f" = {S['mean']:+.3f}" + (r" $\star$" if S["res"] else " (unresolved)"),
            ha="center", va="bottom" if yb > 0 else "top", fontsize=6.9)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[1] for s in steps], fontsize=7.2, linespacing=1.1)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(steps) - 0.4)
    ax.set_title(title, loc="left", fontsize=8.8)
    return g


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
for ax, (m, title, t), tag in zip(axes, CELLS, "ab"):
    bridge(ax, m, f"({tag})  {title}", t)
axes[0].set_ylabel(f"${TAU % COORD_SUP['negDP']}$" + "  (validation-BCE selector)")
axes[0].set_ylim(-0.26, 0.14)
axes[1].set_ylim(-0.155, 0.1)
handles = [Patch(facecolor=_tint(C_LEVEL, TINT_RES), edgecolor=C_LEVEL, label="level"),
           Patch(facecolor=_tint(C_T, TINT_RES), edgecolor=C_T, label=r"$R_{\mathrm{traj}}$ (prefix / run)"),
           Patch(facecolor=_tint(C_E1, TINT_RES), edgecolor=C_E1, label=r"$E_{+I}$ ($M^{+I}$ checkpoint moves)"),
           Patch(facecolor=_tint(C_E0, TINT_RES), edgecolor=C_E0, label=r"$-E_{-I}$ ($M^{-I}$ checkpoint moves)"),
           Patch(facecolor=_tint(INK2, TINT_RES), edgecolor=INK2, lw=OUTLINE, label=r"$\star$ stronger fill = resolved"),
           Patch(facecolor=_tint(INK2, TINT), edgecolor=INK2, lw=OUTLINE, label="light fill = unresolved"),
           Line2D([], [], color=INK, lw=0.7, label="95% interval")]
leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=True,
                 fancybox=False, edgecolor="#d0cfca", framealpha=1.0, fontsize=7, handlelength=1.3,
                 columnspacing=1.0, handletextpad=0.4)
leg.get_frame().set_linewidth(0.4)
fig.tight_layout(w_pad=2.0)
for ext in ("pdf",):
    fig.savefig(HERE / f"fig4_selection_support_bridge.{ext}", bbox_inches="tight", pad_inches=0.02)
print("saved", HERE / "fig4_selection_support_bridge.pdf")

# --------------------------------------------------------------- appendix: fixed-epoch trajectories (no selection)
def traj(method, prefix, coord):
    r = d[(d.method == method) & (d.coordinate == coord) & d.term.str.startswith(prefix + "fx")].copy()
    r["epoch"] = r.term.str.replace(prefix + "fx", "", regex=False).astype(int)
    return r.sort_values("epoch")


# selected epochs per unit (validation-BCE selector, full support) from the frozen study outputs; the selector
# does not depend on the coordinate, so one strip per column
STUDY = HERE.parent / "harness" / "results"
x25 = pd.read_csv(STUDY / "x25" / "x25_selected_epochs.csv")
x26 = pd.read_csv(STUDY / "x26" / "x26_bail_selected_epochs.csv")
assert len(x25) == 240 and len(x26) == 240


def picks(method, pre):
    if method == "NIFTY":
        q = x25[(x25.D == pre.strip(".")) & (x25.sigma == "bce") & (x25.cap == 1000)]
        return q.m1_epoch.to_numpy(), q.m0_epoch.to_numpy()
    q = x26[(x26.selector == "bce") & (x26.support == "full")]
    return q[q.arm == "M1"].epoch.to_numpy(), q[q.arm == "M0"].epoch.to_numpy()


assert np.median(picks("NIFTY", "D1.")[0]) == 947 and np.median(picks("FairGB", "")[1]) == 63   # as documented

COLS = [("NIFTY", "D0.", "NIFTY / German, D0"),
        ("NIFTY", "D1.", "NIFTY / German, D1 (published)"),
        ("FairGB", "", "FairGB / Bail")]
fig = plt.figure(figsize=(7.0, 5.9))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.42], hspace=0.28, wspace=0.28)
axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(4)])
for j in range(3):
    for i in range(4):
        if i < 3:
            axes[i, j].sharex(axes[3, j])
            axes[i, j].tick_params(labelbottom=False)
rng = np.random.default_rng(0)
for j, (m, pre, title) in enumerate(COLS):
    for i, c in enumerate(COORD_SUP):
        ax = axes[i, j]
        r = traj(m, pre, c)
        ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        ax.axvline(200, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.fill_between(r.epoch, r.lo, r.hi, color=C_LEVEL, alpha=0.12, lw=0, zorder=2)
        ax.plot(r.epoch, r["mean"], color=C_LEVEL, lw=1.0, zorder=3)
        ax.scatter(r.epoch[~r.res], r["mean"][~r.res], s=9, facecolor="white", edgecolor=C_LEVEL, lw=0.7, zorder=4)
        ax.scatter(r.epoch[r.res], r["mean"][r.res], s=11, color=C_LEVEL, zorder=5)
        if i == 0:
            ax.set_title(f"({'abc'[j]})  {title}", loc="left", fontsize=8.5)
            ax.text(205, 1, "cap = 200", transform=ax.get_xaxis_transform(), fontsize=6.2, color=INK2, va="top")
        if j == 0:
            ax.set_ylabel(f"${TAU % COORD_SUP[c]}$")
    # strip: where the BCE selector lands over the full support, per unit (30 each)
    ax = axes[3, j]
    m1, m0 = picks(m, pre)
    for yy, e, col, lab in ((1, m1, C_E1, r"$M^{+I}$"), (0, m0, C_E0, r"$M^{-I}$")):
        ax.scatter(e, yy + rng.uniform(-0.18, 0.18, len(e)), s=6, color=col, alpha=0.75, lw=0, zorder=3)
        med = np.median(e)
        ax.plot([med, med], [yy - 0.32, yy + 0.32], color=INK, lw=1.1, zorder=4)
        ax.text(med, yy + 0.36, f"{med:g}", ha="center", va="bottom", fontsize=6)
    ax.axvline(200, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([r"$M^{+I}$", r"$M^{-I}$"] if j == 0 else ["", ""])
    ax.set_ylim(-0.55, 1.75)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("epoch")
    if j == 0:
        ax.set_ylabel("selected epoch", fontsize=7.5)
for i in range(3):   # common y within a coordinate row
    lo = min(a_.get_ylim()[0] for a_ in axes[i]); hi = max(a_.get_ylim()[1] for a_ in axes[i])
    for a_ in axes[i]:
        a_.set_ylim(lo, hi)
handles = [Line2D([], [], color=C_LEVEL, lw=1.0, marker="o", ms=3.5, label="mean over 30 units, resolved"),
           Line2D([], [], color=C_LEVEL, lw=1.0, marker="o", ms=3.2, mfc="white", label="unresolved"),
           Patch(facecolor=C_LEVEL, alpha=0.14, label="95% interval"),
           Line2D([], [], color=MUTED, lw=0.6, ls=(0, (3, 2)), label="cap of the controlled selector"),
           Line2D([], [], marker="o", ls="none", ms=3, color=C_E1, label=r"selected epoch, $M^{+I}$"),
           Line2D([], [], marker="o", ls="none", ms=3, color=C_E0, label=r"selected epoch, $M^{-I}$"),
           Line2D([], [], color=INK, lw=1.1, label="median")]
leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=4, frameon=True,
                 fancybox=False, edgecolor="#d0cfca", framealpha=1.0, fontsize=7.2, handlelength=1.5,
                 columnspacing=1.2)
leg.get_frame().set_linewidth(0.4)
for ext in ("pdf",):
    fig.savefig(HERE / f"figS4_fixed_epoch_trajectory.{ext}", bbox_inches="tight", pad_inches=0.02)
print("saved", HERE / "figS4_fixed_epoch_trajectory.pdf")
