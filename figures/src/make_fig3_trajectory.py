"""Main-text selection-support figure: the two arms are checkpointed at different epochs.

Replaces the selection-support bar chart in the main text (kept, unchanged, as the appendix figure
figS_selection_support_bars.pdf). The bars gave the algebraic decomposition; this shows the thing the
decomposition is about -- once the selector may look past epoch 200, sigma_f^BCE lands the two arms
in different places, and the contrast measured at a fixed epoch is not the contrast that gets
reported.

No new training and no re-run: every number is read from the same frozen artifacts that produced the
appendix trajectory figure (figS4_fixed_epoch_trajectory, `make_fig4.py`).

    results/4_mechanistic_case_study.csv            tau_{-I->+I} at each fixed checkpoint, and tau_full
    harness/results/x25/x25_selected_epochs.csv     NIFTY per-unit selected epochs (sigma_f^BCE)
    harness/results/x26/x26_bail_selected_epochs.csv  FairGB per-unit selected epochs (sigma_f^BCE)

Two cells, one coordinate (-Delta_DP):
    (a) NIFTY / German, D1 (published drop rates), H = 1000
    (b) FairGB / Bail, H = 1500

Top row: tau at the fixed checkpoints of Table 28, mean over 30 units with the 95% paired
hierarchical-bootstrap interval, filled marker when resolved. The open diamond at the right is
tau_full, the contrast actually reported once the selector has the whole trajectory -- it is not a
point on the fixed-epoch curve, and that is the point of the panel.

Bottom row: where the selector lands, per unit, for each arm.

    python figures/src/make_fig3_trajectory.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
from style import apply_tone, ROLE                                   # the shared Fig. 2 tone

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parents[1]           # figures/
DATA = HERE.parent / "results"
STUDY = HERE.parent / "harness" / "results"

# ---------------------------------------------------------------- style, identical to make_fig4.py
plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.labelcolor": "#0b0b0b", "ytick.labelcolor": "#0b0b0b",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "savefig.dpi": 300, "pdf.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})
apply_tone()
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e0", "#9a9993"
C_LEVEL = ROLE["level"]                       # the tau curve and tau_full
C_E1, C_E0 = ROLE["intervention"], ROLE["surrounding"]      # M^{+I} / M^{-I}, same as Fig. S4
# the coordinate rides beside the symbol, not as a subscript inside a superscript: mathtext
# shrinks 70% per nesting level, and the doubly nested form printed "DP" at 3.9 pt
TAU = r"\tau_{{-}I\rightarrow {+}I}"
YLAB = rf"${TAU}$  ($-\Delta_{{\mathrm{{DP}}}}$)"

d = pd.read_csv(DATA / "4_mechanistic_case_study.csv")
assert len(d) == 255, len(d)
as_true = lambda s: s.astype(str).str.strip().str.lower().eq("true")
d["res"] = as_true(d["resolved"])

x25 = pd.read_csv(STUDY / "x25" / "x25_selected_epochs.csv")
x26 = pd.read_csv(STUDY / "x26" / "x26_bail_selected_epochs.csv")
assert len(x25) == 240 and len(x26) == 240

CAP = 200                                      # the controlled selector's horizon


def traj(method, prefix):
    """tau_{-I->+I}(-Delta_DP) at each fixed checkpoint, as make_fig4.traj reads it."""
    r = d[(d.method == method) & (d.coordinate == "negDP")
          & d.term.str.startswith(prefix + "fx")].copy()
    r["epoch"] = r.term.str.replace(prefix + "fx", "", regex=False).astype(int)
    return r.sort_values("epoch")


def picks(method, pre):
    """per-unit selected epochs (sigma_f^BCE, full support) -> (M^{+I}, M^{-I}); as make_fig4.picks."""
    if method == "NIFTY":
        q = x25[(x25.D == pre.strip(".")) & (x25.sigma == "bce") & (x25.cap == 1000)]
        return q.m1_epoch.to_numpy(), q.m0_epoch.to_numpy()
    q = x26[(x26.selector == "bce") & (x26.support == "full")]
    return q[q.arm == "M1"].epoch.to_numpy(), q[q.arm == "M0"].epoch.to_numpy()


def tau_full(method, pre):
    """the contrast reported once the selector has the whole trajectory."""
    term = f"{pre}bce.c1000" if method == "NIFTY" else "bce.full"
    r = d[(d.method == method) & (d.coordinate == "negDP") & (d.term == term)]
    return float(r["mean"].iloc[0]), float(r.lo.iloc[0]), float(r.hi.iloc[0]), bool(r.res.iloc[0])


# ---------------------------------------------------------------- the numbers this figure asserts
COLS = [("NIFTY", "D1.", "(a) NIFTY / German (published drop rates, $H = 1000$)", 1000,
         [25, 50, 100, 150, 200, 300, 400, 600, 800, 1000],
         [0, 200, 400, 600, 800, 1000]),
        ("FairGB", "", "(b) FairGB / Bail ($H = 1500$)", 1500,
         [25, 50, 100, 150, 200, 300, 500, 750, 1000, 1250, 1499],
         [0, 500, 1000, 1500])]

for meth, pre, _t, _h, want, _tk in COLS:                  # Table 28's checkpoint list, verbatim
    got = traj(meth, pre).epoch.tolist()
    assert got == want, (meth, got, want)
assert np.median(picks("NIFTY", "D1.")[0]) == 947 and np.median(picks("NIFTY", "D1.")[1]) == 692.5
assert np.median(picks("FairGB", "")[0]) == 999 and np.median(picks("FairGB", "")[1]) == 63
assert round(tau_full("NIFTY", "D1.")[0], 3) == -0.148
assert round(tau_full("FairGB", "")[0], 3) == -0.019

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(6.25, 1.99))
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0], hspace=0.12, wspace=0.14)
top = [fig.add_subplot(gs[0, j]) for j in range(2)]
bot = [fig.add_subplot(gs[1, j], sharex=top[j]) for j in range(2)]
top[1].sharey(top[0])                                   # top row shares y: the sizes are comparable

rng = np.random.default_rng(4)                          # jitter only; touches no reported number

for j, (meth, pre, title, hmax, _ck, xticks) in enumerate(COLS):
    a, b = top[j], bot[j]
    t = traj(meth, pre)
    xs, ys = t.epoch.to_numpy(), t["mean"].to_numpy()

    a.axhline(0, color=INK2, lw=0.5, zorder=1)
    for ax in (a, b):
        ax.axvline(CAP, color=MUTED, lw=0.7, ls=(0, (2.5, 2)), zorder=1)

    a.fill_between(xs, t.lo.to_numpy(), t.hi.to_numpy(), color=C_LEVEL, alpha=0.13, lw=0, zorder=2)
    a.plot(xs, ys, color=C_LEVEL, lw=0.9, zorder=3)
    r = t.res.to_numpy()
    a.plot(xs[r], ys[r], ls="none", marker="o", ms=3.0, mfc=C_LEVEL, mec=C_LEVEL, mew=0.7, zorder=4)
    a.plot(xs[~r], ys[~r], ls="none", marker="o", ms=3.0, mfc="white", mec=C_LEVEL, mew=0.7, zorder=4)

    # tau_full is not a fixed-epoch value at all: it is what the selector returns once it may see
    # the whole trajectory. It gets its own strip past the end of training, fenced off so it cannot
    # be misread as the curve continuing.
    fm, flo, fhi, fres = tau_full(meth, pre)
    sep, fx = hmax * 1.045, hmax * 1.145
    for ax in (a, b):
        ax.axvline(sep, color=MUTED, lw=0.6, zorder=2)
    a.errorbar(fx, fm, yerr=[[fm - flo], [fhi - fm]], fmt="none", ecolor=C_LEVEL, elinewidth=0.7,
               capsize=1.5, capthick=0.7, zorder=4)
    a.plot([fx], [fm], marker="D", ms=3.4, mfc="white", mec=C_LEVEL, mew=0.9, ls="none", zorder=5)
    # above the upper cap, where neither the interval nor the curve's tail can reach it
    a.annotate(f"{fm:.3f}".replace("-", "−"), (fx, fhi), textcoords="offset points",
               xytext=(0, 3.5), ha="center", va="bottom", fontsize=7, color=INK, zorder=6)
    # named inside the strip itself, at the foot of the upper panel, clear of the epoch ticks
    a.text(fx, 0.02, "full\nselection", transform=a.get_xaxis_transform(), ha="center", va="bottom",
           fontsize=7, color=INK2, linespacing=0.95, zorder=6,
           bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

    m1, m0 = picks(meth, pre)
    for k, (v, c) in enumerate(((m1, C_E1), (m0, C_E0))):
        y = 1 - k + rng.uniform(-0.13, 0.13, size=len(v))
        b.plot(v, y, ls="none", marker="o", ms=1.9, mfc=c, mec=c, alpha=0.72, zorder=3)
        md = float(np.median(v))
        b.plot([md, md], [1 - k - 0.30, 1 - k + 0.30], color=c, lw=1.4, solid_capstyle="butt",
               zorder=4)
        lab = f"{md:g}"
        b.annotate(lab, (md, 1 - k + 0.34), ha="center", va="bottom", fontsize=6.5, color=c,
                   zorder=5)

    # left-aligned: (a)'s title fills its panel while (b)'s is short, so centring made the
    # two look misaligned against each other
    a.set_title(title, loc="left", fontsize=8, pad=3.5)
    a.set_xlim(-hmax * 0.03, hmax * 1.235)
    a.set_xticks(xticks)                  # ticks stop at the training horizon, not at the strip
    b.set_ylim(-0.75, 1.85)
    b.set_yticks([1, 0])
    b.set_yticklabels([r"$M^{+I}$", r"$M^{-I}$"], fontsize=6.5)
    b.set_xlabel("epoch", labelpad=1.5)
    a.tick_params(labelbottom=False)
    if j == 0:
        a.set_ylabel(YLAB)
    else:
        a.tick_params(labelleft=False)
    for ax in (a, b):
        ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)

handles = [
    Line2D([], [], color=C_LEVEL, lw=0.9, marker="o", ms=3.0, mfc=C_LEVEL, mec=C_LEVEL,
           label="mean, resolved"),
    Line2D([], [], color=C_LEVEL, lw=0.9, marker="o", ms=3.0, mfc="white", mec=C_LEVEL,
           label="unresolved"),
    Patch(facecolor=C_LEVEL, alpha=0.13, edgecolor="none", label="95% interval"),
    Line2D([], [], color=MUTED, lw=0.7, ls=(0, (2.5, 2)), label="selector cap"),
    Line2D([], [], color=C_LEVEL, lw=0, marker="D", ms=3.4, mfc="white", mec=C_LEVEL, mew=0.9,
           label=r"$\tau_{\mathrm{full}}$"),
    Line2D([], [], color=C_E1, lw=0, marker="o", ms=2.6, label=r"selected, $M^{+I}$"),
    Line2D([], [], color=C_E0, lw=0, marker="o", ms=2.6, label=r"selected, $M^{-I}$"),
    Line2D([], [], color=INK2, lw=1.4, label="median"),
]
leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.165), ncol=8,
                 frameon=True, fancybox=False, borderpad=0.35, handlelength=1.2,
                 columnspacing=0.5, handletextpad=0.25)
leg.get_frame().set_linewidth(0.4)
leg.get_frame().set_edgecolor(GRID)

fig.savefig(HERE / "fig3_selection_support_trajectory.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(HERE / "src" / "preview" / "fig3_selection_support_trajectory.png",
            bbox_inches="tight", pad_inches=0.02, dpi=300)
print("saved", HERE / "fig3_selection_support_trajectory.pdf")
