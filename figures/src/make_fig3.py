"""Experiment 3 -- does tau_{-I->+I}(P) on -Delta_DP survive a change of protocol P?

    (a) 3a  selector only:           validation BCE (x)  vs  validation AUC (y), primary cells of Experiment 1
    (b) 3b  horizon + selector:      controlled (x)      vs  the method's own published horizon and selector (y)
    (c) 3c  published procedure:     one row per pair, controlled -> native arrow, 95% intervals (the protocol plus
                                     what the published procedure bundles with it; read as one joint shift)

(a) and (b) are identity scatters: on the diagonal the protocol does not matter; in the shaded quadrants the
sign flips. Colour = how the resolved status (taken from the CSV) moves from x to y, with the same colours as
Fig. 2. FnRGNN (task-adapted) is left out of (a), as in Experiment 1. In (b) the 3 targeted native validations
are drawn as triangles, apart from the 18 systematic native comparisons; the two sets have different sampling
rationales, so the embedded count table never pools them. 3a and 3b change only the protocol; 3c changes the
protocol together with the preprocessing / training-loop details its published procedure bundles, so it is not a
further step on the same scale and no monotonic "wider protocol -> more instability" reading is implied.
The counts (from the CSV flags) are written to fig3_protocol_counts.csv for the text; no table is drawn.

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

import os as _os
HERE = Path(__file__).resolve().parents[1]           # output: figures/ (PDF only)
# FAIRGNN_RESULTS points the same script at the frozen bundle or a rebuilt one.
DATA = Path(_os.environ.get("FAIRGNN_RESULTS", str(HERE.parent / "results")))
ROOT = DATA
sel = pd.read_csv(ROOT / "3a_protocol_selector_bce_vs_auc.csv")
hs = pd.read_csv(ROOT / "3b_protocol_native_horizon_selector.csv")
pub = pd.read_csv(ROOT / "3c_protocol_native_published_procedure.csv")
assert (len(sel), len(hs), len(pub)) == (65, 21, 4)
for d in (sel, hs, pub):
    assert d["x_name"].str.endswith("tau_I_negDP").all() and d["y_name"].str.endswith("tau_I_negDP").all()

as_true = lambda s: s.astype(str).str.strip().str.lower().eq("true")
sel_p = sel[sel["configuration_role"].eq("primary") & sel["method"].ne("FnRGNN")].copy()
assert len(sel_p) == 36, len(sel_p)

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.labelcolor": "#0b0b0b", "ytick.labelcolor": "#0b0b0b",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "savefig.dpi": 300, "pdf.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})
apply_tone()
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e0", "#9a9993"
C_SIGN, C_LOST, C_GAIN = ROLE["sign_flip"], ROLE["lost"], ROLE["gained"]
C_BOTH, C_NONE = ROLE["level"], ROLE["run_noise"]
LINTHRESH = 0.02      # symlog: linear within +-0.02, log beyond, so the bulk near 0 and the few large cells both show
# unnested: mathtext shrinks 70% per level, so a subscript inside a superscript
# printed "DP" at 3.9 pt. The coordinate now rides beside the symbol.
TAU = r"\tau_{{-}I\rightarrow {+}I}"
COORD = r"(-\Delta_{\mathrm{DP}})"


def status(d):
    x, y = as_true(d["x_resolved"]), as_true(d["y_resolved"])
    return np.select([x & y, x & ~y, ~x & y], ["both", "lost", "gained"], "neither")


STATUS_STYLE = {  # facecolor, edgecolor, size, z -- Fig. 2 tone: tinted fill, outline in the full colour
    "neither": ("white", C_NONE, 14, 3), "both": (_tint(C_BOTH, 0.35), C_BOTH, 18, 4),
    "lost": (_tint(C_LOST, 0.35), C_LOST, 22, 5), "gained": (_tint(C_GAIN, 0.35), C_GAIN, 22, 5)}


def identity(ax, d, lim, xlabel, ylabel, marker=None):
    lo, hi = lim
    # sign-flip quadrants (II and IV)
    ax.fill_between([lo, 0], 0, hi, color=C_SIGN, alpha=0.07, lw=0, zorder=0)
    ax.fill_between([0, hi], lo, 0, color=C_SIGN, alpha=0.07, lw=0, zorder=0)
    ax.plot(np.linspace(lo, hi, 400), np.linspace(lo, hi, 400), color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1); ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    st = status(d)
    mk = np.full(len(d), "o") if marker is None else np.asarray(marker)
    for k, (fc, ec, s, z) in STATUS_STYLE.items():
        for m_ in np.unique(mk):
            m = (st == k) & (mk == m_)
            ax.scatter(d["x_mean"][m], d["y_mean"][m], s=s * (1.3 if m_ == "^" else 1), marker=m_, facecolor=fc,
                       edgecolor=ec, lw=0.7, zorder=z)
    for axis in (ax.xaxis, ax.yaxis):
        axis._set_axes_scale("symlog", linthresh=LINTHRESH, linscale=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ticks = [-0.1, -0.02, 0, 0.02, 0.1]
    ticks = [t for t in ticks if lo <= t <= hi]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_box_aspect(1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for tx, ty, h, v in ((0.03, 0.97, "left", "top"), (0.97, 0.16, "right", "bottom")):
        ax.text(tx, ty, "sign flip", transform=ax.transAxes, ha=h, va=v, fontsize=7, color=C_SIGN, style="italic")
    ax.text(0.03, 0.03, f"symlog, linear |τ|<{LINTHRESH:g}", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7, color=INK2, zorder=6, bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))
    return st


def annotate(ax, d, keep, offsets):
    for (_, r), k in zip(d.iterrows(), keep):
        if not k:
            continue
        name = f"{r.method}" + (f"-{r.backbone}" if r.method == "FairSIN" else "") + f" / {r.dataset}"
        dx, dy = offsets.get(name, (8, -8))
        ax.annotate(name, (r.x_mean, r.y_mean), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    ha="left" if dx >= 0 else "right", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2, shrinkA=0, shrinkB=2))


fig = plt.figure(figsize=(5.40, 1.95))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.30], wspace=0.34,
                      left=0.06, right=0.985, top=0.95, bottom=0.33)
ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

# (a) selector only
st_a = identity(ax_a, sel_p, (-0.15, 0.35), f"BCE-selected ${TAU}$", f"AUC-selected ${TAU}$")
# cells whose status changes are named in the caption, not in the panel (too crowded near 0)
ax_a.set_title(f"(a)  Selector only  ${COORD}$", loc="center", fontsize=8)

# (b) horizon + selector to the published values
# Decision 2: SFG/German and SFG/Credit differ by re-execution alone, so they are drawn
# apart (open square) and are not among the systematic comparisons.
ROLE_MARKER = {"systematic": "o", "targeted": "^", "re_execution": "s"}
assert hs["native_evaluation_role"].value_counts().to_dict() == {
    "systematic": 16, "targeted": 3, "re_execution": 2}
st_b = identity(ax_b, hs, (-0.15, 0.35), f"Controlled ${TAU}$", f"Published ${TAU}$",
                marker=hs["native_evaluation_role"].map(ROLE_MARKER))
for tag, d_, st_ in (("a", sel_p, st_a), ("b", hs, st_b)):   # for the caption
    m = np.isin(st_, ["lost", "gained"])
    print(tag, [f"{r.method}/{r.dataset} ({k})" for r, k in zip(d_[m].itertuples(), st_[m])])
ax_b.set_title(f"(b)  Published horizon  ${COORD}$", loc="center", fontsize=8)

# (c) published procedure: one row per pair; controlled (upper, grey) -> native (lower) with 95% intervals.
# The arrow joins the two point estimates; purple = it crosses 0 (sign flips). Filled = resolved (CSV flag).
d = pub.reset_index(drop=True)
rows_y = np.arange(len(d))[::-1].astype(float)
ax_c.axvline(0, color=INK2, lw=0.6, zorder=1)
ax_c.grid(axis="x", color=GRID, lw=0.5, zorder=0)
OFF = 0.17
for yi, (_, r) in zip(rows_y, d.iterrows()):
    flip = bool(as_true(pd.Series([r["sign_changed"]]))[0])
    c = C_SIGN if flip else C_BOTH
    for side, yy, col in (("x", yi + OFF, C_NONE), ("y", yi - OFF, c)):
        res = bool(as_true(pd.Series([r[f"{side}_resolved"]]))[0])
        ax_c.plot([r[f"{side}_lo"], r[f"{side}_hi"]], [yy, yy], color=col, lw=0.8, alpha=0.6,
                  solid_capstyle="butt", zorder=2)
        ax_c.plot(r[f"{side}_mean"], yy, marker="o", ms=3.8, color=col, mfc=col if res else "white", mew=0.9, zorder=4)
    ax_c.annotate("", xy=(r.y_mean, yi - OFF), xytext=(r.x_mean, yi + OFF),
                  arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, mutation_scale=6, shrinkA=2.5, shrinkB=2.5),
                  zorder=3)
    ax_c.text(-0.245, yi + 0.40, f"{r.method} / {r.dataset}", fontsize=7, ha="left", va="center")
ax_c.set_yticks([])
ax_c.spines["left"].set_visible(False)
ax_c.set_xlim(-0.25, 0.32)
ax_c.set_ylim(-0.5, len(d) - 0.25)
ax_c.set_xlabel(f"${TAU}$ ${COORD}$")
ax_c.set_box_aspect(0.72)
ax_c.set_title("(c)  Published procedure bundle", loc="center", fontsize=8)
# one compact legend: resolved-status colours (a, b; filled/open also in c), the targeted marker of (b), and
# the sign-flip arrow of (c). The sign-flip quadrants are labelled inside the panels.
handles = [Line2D([], [], marker="o", ls="none", ms=4.4, mfc=_tint(C_BOTH, 0.35), mec=C_BOTH, mew=0.7, label="resolved in both"),
           Line2D([], [], marker="o", ls="none", ms=4.4, mfc=_tint(C_LOST, 0.35), mec=C_LOST, mew=0.7, label=r"resolved$\rightarrow$unresolved"),
           Line2D([], [], marker="o", ls="none", ms=4.4, mfc=_tint(C_GAIN, 0.35), mec=C_GAIN, mew=0.7, label=r"unresolved$\rightarrow$resolved"),
           Line2D([], [], marker="o", ls="none", ms=3.8, mfc="white", mec=C_NONE, mew=0.8, label="unresolved in both"),
           Line2D([], [], marker="^", ls="none", ms=4.2, mfc="white", mec=INK2, mew=0.8, label="targeted (b)"),
           Line2D([], [], marker="s", ls="none", ms=4.0, mfc="white", mec=INK2, mew=0.8,
                  label="re-execution (b)"),
           Line2D([], [], color=C_SIGN, lw=1.0, marker=">", ms=3, label="sign flip (c)")]
leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.15), ncol=4, frameon=True,
                 fancybox=False, edgecolor="#d0cfca", framealpha=1.0, fontsize=7, handlelength=1.3,
                 columnspacing=1.6, handletextpad=0.5, borderpad=0.6, labelspacing=0.55)
leg.get_frame().set_linewidth(0.4)

# --------------------------------------------------------------- counts for the text (from the CSV flags)
def counts(stage, change, d):
    x, y = as_true(d["x_resolved"]), as_true(d["y_resolved"])
    return dict(stage=stage, protocol_change=change, pairs=len(d), sign_flips=int(as_true(d["sign_changed"]).sum()),
                status_changes=int(as_true(d["resolution_changed"]).sum()),
                lost=int((x & ~y).sum()), gained=int((~x & y).sum()),
                sign_or_status=int((as_true(d["sign_changed"]) | as_true(d["resolution_changed"])).sum()))


assert pub["native_evaluation_role"].eq("targeted").all()
tab = pd.DataFrame([
    counts("(a)", "Selector only (validation BCE -> AUC), primary cells", sel_p),
    counts("(b)", "Published horizon, systematic", hs[hs.native_evaluation_role.eq("systematic")]),
    counts("(b)", "Horizon + selector, targeted native", hs[hs.native_evaluation_role.eq("targeted")]),
    counts("(c)", "Published procedure bundle, targeted native", pub)])
assert ((tab.lost + tab.gained) == tab.status_changes).all()
assert tab.iloc[3][["sign_flips", "status_changes", "sign_or_status"]].tolist() == [3, 3, 4]
pass  # counts CSV stays in results
print(tab.to_string(index=False))

for ext in ("pdf",):
    fig.savefig(HERE / f"fig3_protocol_variation.{ext}", bbox_inches="tight", pad_inches=0.02)
_os.makedirs(HERE / "src" / "preview", exist_ok=True)
fig.savefig(HERE / "src" / "preview" / "fig3_protocol_variation.png",
            bbox_inches="tight", pad_inches=0.02, dpi=300)
print("saved", HERE / "fig3_protocol_variation.pdf")
