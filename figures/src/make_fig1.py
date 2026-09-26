"""Experiment 1 and 1.1 figures for figures/, in the tone of Fig. 2.

Ported from results/plot.ipynb (cells 1, 2, 5-9, 12; tables left in the notebook). Only the look changes:
Fig. 2's rcParams (6.5 pt text, black labels, grey axes), 7.0 in full width, and output to figures/.
Every number, filter, label rule and resolved flag is as in the notebook.

    python figures/src/make_fig1.py
"""
import matplotlib
matplotlib.use("Agg")
# ======== notebook cell 1 ========
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from matplotlib.colors import to_rgba

import os as _os
# FAIRGNN_RESULTS lets the same script draw the frozen bundle or a rebuilt one
# (results_v2/bundle) without editing it; the default is the frozen results/.
DATA_DIR = Path(_os.environ.get("FAIRGNN_RESULTS",
                                str(Path(__file__).resolve().parents[2] / "results")))
OUT = Path(__file__).resolve().parents[1]
FILES = {
    "main": "1_main_package_vs_intervention.csv",
    "main_ta": "1b_main_task_adapted.csv",
    "main_reg": "1c_main_released_task_regression.csv",
    "config": "2_configuration_variation.csv",
    "selector": "3a_protocol_selector_bce_vs_auc.csv",
    "native_hs": "3b_protocol_native_horizon_selector.csv",
    "native_pub": "3c_protocol_native_published_procedure.csv",
    "mech": "4_mechanistic_case_study.csv",
    "fmp": "5_component_case_study_FMP.csv",
}
dfs = {k: pd.read_csv(DATA_DIR / v) for k, v in FILES.items()}
# row counts are checked against the bundle's own index instead of being hard-coded
EXPECTED_ROWS = pd.read_csv(DATA_DIR / "experiment_index.csv").set_index("file")["rows"].to_dict()
for k, d in dfs.items():
    exp_n = EXPECTED_ROWS.get(FILES[k])
    assert exp_n is None or len(d) == exp_n, (FILES[k], len(d), exp_n)
    print(f"{k:10s} {FILES[k]:45s} rows={len(d)}  (index: {exp_n})")

EXP_DIRS = {
    1: DATA_DIR / "exp1_package_vs_intervention",
    "1_1": DATA_DIR / "exp1_1_fnrgnn_task_adapted",
    2: DATA_DIR / "exp2_configuration_variation",
    3: DATA_DIR / "exp3_protocol_variation",
    4: DATA_DIR / "exp4_mechanistic_case_study",
    5: DATA_DIR / "exp5_fmp_component_case_study",
}


# ======== notebook cell 2 ========
from style import apply_tone, tint as _tint, ROLE
apply_tone()

# Colours encode the *role* of an estimate (estimand / arm / component),
# never the method. The same role keeps the same colour in every figure.
INK, INK2, MUTED, GRID, ZERO = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0", "#52514e"
C_A = ROLE["intervention"]    # plum : tau_{-I->+I} and every step on the M^{+I} side
C_B = ROLE["surrounding"]     # slate: tau_{B->-I}, the surrounding package
C_PKG = INK       # black  : tau_{B->+I} (package total), Delta_long

METRICS = ["dAUC", "negDP", "negEO"]
METRIC_LABEL = {
    "dAUC": r"$\Delta$AUC",
    "negDP": r"$-\Delta_{\mathrm{DP}}$",
    "negEO": r"$-\Delta_{\mathrm{EO}}$",
}
METRIC_TEX = {"dAUC": r"$\Delta$AUC", "negDP": r"$-\Delta_{\mathrm{DP}}$", "negEO": r"$-\Delta_{\mathrm{EO}}$"}
# Paper notation (eq. intervention_path). {-}/{+} keep mathtext from spacing the signs as binary operators.
#   B --tau_{B->-I}(P)--> M^-I --tau_{-I->+I}(P)--> M^+I,   B --tau_{B->+I}(P)--> M^+I
#   tau_{B->+I}(P) = tau_{B->-I}(P) + tau_{-I->+I}(P)
TAU_NONINT = r"\tau_{B\rightarrow {-}I}"
TAU_I = r"\tau_{{-}I\rightarrow {+}I}"
TAU_PKG = r"\tau_{B\rightarrow {+}I}"
M_MINUS, M_PLUS = r"M^{{-}I}", r"M^{{+}I}"
# the same symbols for LaTeX tables
TEX_NONINT, TEX_I, TEX_PKG = r"$\tau_{B\rightarrow -I}$", r"$\tau_{-I\rightarrow +I}$", r"$\tau_{B\rightarrow +I}$"
RESOLVED_NOTE = "filled = resolved, hollow = unresolved"
# Method families by what the intervention does:
#   Modification   -- changes the data, features or graph directly (BIND, EDITS, FairEdit)
#   Constraint     -- imposes a fairness constraint or an adversarial / generative restriction during training
#   Regularization -- adds a training-time objective or procedure: sampling, augmentation, invariance, alignment Judged from intervention_I in
# method_configurations.csv. Used for grouping only: families hold few, non-independent cells.
FAMILY = {"EDITS": "pre", "FairEdit": "pre", "BIND": "pre",
          "FairGNN": "adv", "FairVGNN": "adv", "SFG": "adv",
          "NIFTY": "reg", "GEAR": "reg", "FairGB": "reg", "FairSIN": "reg", "BeMap": "reg"}
FAMILY_ORDER = ["pre", "adv", "reg"]
FAMILY_LABEL = {"pre": "Modification", "adv": "Constraint", "reg": "Regularization"}
DATASET_ORDER = ["bail", "credit", "german", "income",
                 "pokec_n", "pokec_n_g", "pokec_z", "pokec_z_g"]


# every paper figure draws its legend in a thin, light box
LEGEND_BOX = dict(frameon=True, fancybox=False, edgecolor="#d0cfca", framealpha=1.0, borderpad=0.5)


def legend_style(name):
    return LEGEND_BOX


def boxed_legend(fig, name, **kw):
    """fig.legend with the appendix box drawn in a thin, light line."""
    leg = fig.legend(**kw, **legend_style(name))
    leg.get_frame().set_linewidth(0.4)
    return leg


def savefig(fig, exp, name):
    for ext in ("pdf",):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    _os.makedirs(OUT / "src" / "preview", exist_ok=True)
    fig.savefig(OUT / "src" / "preview" / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("saved", OUT / f"{name}.pdf")


def as_bool(s):
    """Resolved flags as stored in the CSV (no recomputation)."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def zero_line(ax, axis="x"):
    (ax.axvline if axis == "x" else ax.axhline)(0, color=ZERO, lw=0.8, zorder=1)


def xgrid(ax):
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def point(ax, x, y, color, resolved, marker="o", size=5.5, z=4, mew=1.1):
    """Filled marker if resolved, hollow otherwise."""
    ax.plot(x, y, marker=marker, ms=size, ls="none", zorder=z,
            mec=color, mew=mew, mfc=_tint(color, 0.35) if resolved else "white")


def ci(ax, lo, hi, y, color, horizontal=True, lw=1.1, alpha=0.8):
    if horizontal:
        ax.plot([lo, hi], [y, y], color=color, lw=lw, alpha=alpha, zorder=3, solid_capstyle="butt")
    else:
        ax.plot([y, y], [lo, hi], color=color, lw=lw, alpha=alpha, zorder=3, solid_capstyle="butt")


def status_bar(ax, x, height, color, resolved=None, width=0.6, bottom=0.0):
    """Vertical bar from `bottom`. Filled = resolved (or a quantity without a resolution
    status), hollow = unresolved. The sign is read from the bar's direction, no hatch."""
    filled = resolved is not False
    ax.bar(x, height, bottom=bottom, width=width, facecolor=color if filled else "white",
           edgecolor=color, lw=1.0, zorder=3)


def cell_label(r):
    """method / dataset, plus backbone and configuration when they are informative."""
    s = f"{r['method']} / {r['dataset']}"
    extra = []
    if "backbone" in r and pd.notna(r["backbone"]) and r["backbone"] != "GCN":
        extra.append(str(r["backbone"]))
    if ("configuration" in r and pd.notna(r["configuration"])
            and r["configuration"] not in ("default", r.get("backbone"))):
        extra.append(str(r["configuration"]))
    return s + (f" [{', '.join(extra)}]" if extra else "")

# ======== notebook cell 5 ========
main = dfs["main"].copy()
main["_ds"] = main["dataset"].map({d: i for i, d in enumerate(DATASET_ORDER)}).fillna(99)
main["family"] = main["method"].map(FAMILY)
assert main["family"].notna().all(), main.loc[main["family"].isna(), "method"].unique()
main["_fam"] = main["family"].map({f: k for k, f in enumerate(FAMILY_ORDER)})
main = main.sort_values(["_fam", "method", "_ds", "configuration"]).reset_index(drop=True)
print(main.groupby("family", sort=False)["method"].unique().to_dict())
main["label"] = main.apply(cell_label, axis=1)

# task-adapted cells (1b): separate, never pooled into the primary set
main_ta = dfs["main_ta"].copy()
main_ta["label"] = main_ta.apply(cell_label, axis=1)
assert not set(main_ta["method"]) & set(main["method"])
print(f"primary cells: {len(main)} | task-adapted cells (reported separately): {len(main_ta)}")

# identity check: tau_{B->+I} = tau_{B->-I} + tau_{-I->+I} (point estimates)
for m in METRICS:
    gap = (main[f"tau_pkg_{m}_mean"] - main[f"tau_nonint_{m}_mean"] - main[f"tau_I_{m}_mean"]).abs().max()
    print(f"{m}: max |tau_B->+I - (tau_B->-I + tau_-I->+I)| = {gap:.2e}")


def dominance(d, metric):
    """|tau_{B->-I}| > |tau_{-I->+I}| per cell: the CSV flag when it exists, else from the two means."""
    col = f"nonint_larger_{metric}"
    if col in d.columns:
        return as_bool(d[col]).to_numpy(), "CSV"
    return (d[f"tau_nonint_{metric}_mean"].abs() > d[f"tau_I_{metric}_mean"].abs()).to_numpy(), "means"

# ======== notebook cell 6 ========
# Arm-level means (B, M^-I, M^+I) per primary cell, from the raw result stores via the bundle's own
# loader (build_results.load_store). Their differences reproduce the tau estimates in 1_main exactly.
import sys, io, contextlib
# the harness lives beside the repository root, not beside DATA_DIR, which may point at a
# rebuilt bundle under results_v2/
sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "harness" / "experiments")))
import build_results as BFR

with contextlib.redirect_stdout(io.StringIO()):
    store = BFR.load_store()
    # When DATA_DIR is a rebuilt bundle, its tau_{B->*} were formed against a rebuilt baseline, so
    # the store has to carry the same B or the consistency check below compares two different
    # baselines. FAIRGNN_BASELINE names the directory that was used.
    _bl = _os.environ.get("FAIRGNN_BASELINE")
    if _bl:
        store = BFR.swap_baseline(store, _bl)
ctl = store[store.protocol.eq("controlled") & store.selector.eq("common_bce")].copy()
ctl[["method_", "configuration_"]] = [BFR.split_name(m) for m in ctl["method"]]
# B = bc_*: the common baseline under the same checkpoint selector as the method arms, which is the B
# of tau_{B->-I} and tau_{B->+I} in the bundle (analyze_6x5.build: apkg = m1 - bc, abase = m0 - bc)
ARM_COLS = [f"{a}_{m}" for a in ("bc", "m0", "m1") for m in ("auc", "dp", "eo")]
arm = (ctl.groupby(["method_", "dataset", "configuration_"])
          .agg(units=("split_id", "size"), **{c: (c, "mean") for c in ARM_COLS})
          .reset_index().rename(columns={"method_": "method", "configuration_": "configuration"}))
main = main.drop(columns=[c for c in ARM_COLS + ["units"] if c in main.columns]).merge(
    arm, on=["method", "dataset", "configuration"], how="left", validate="1:1")
assert main[ARM_COLS].notna().all().all() and main["units"].eq(30).all()
# consistency with the bundle: M^+I - M^-I and M^+I - B reproduce tau_{-I->+I} and tau_{B->+I}
SIGN = {"dAUC": ("auc", 1), "negDP": ("dp", -1), "negEO": ("eo", -1)}
for m, (col, sg) in SIGN.items():
    gi = (sg * (main[f"m1_{col}"] - main[f"m0_{col}"]) - main[f"tau_I_{m}_mean"]).abs().max()
    gp = (sg * (main[f"m1_{col}"] - main[f"bc_{col}"]) - main[f"tau_pkg_{m}_mean"]).abs().max()
    print(f"{m}: max gap tau_-I->+I {gi:.1e}, tau_B->+I {gp:.1e}")
    assert gi < 1e-9 and gp < 1e-9


# ======== notebook cell 7 ========
# Fig 1d: direction of tau(P) in the (fairness, utility) plane, both higher-is-better. No title (LaTeX caption).
# a colour always means a ROLE; the method family is told apart by marker shape alone
FAM_STYLE = {"pre": (ROLE["level"], "s"), "adv": (ROLE["level"], "^"), "reg": (ROLE["level"], "o")}
T_NON, T_I_P, T_PKG_P = rf"{TAU_NONINT}(P)", rf"{TAU_I}(P)", rf"{TAU_PKG}(P)"
DASHED = (0, (3, 2))
QUAD = [("fairer &\nmore accurate", 1, 1, "right", "top"), ("less fair,\nmore accurate", -1, 1, "left", "top"),
        ("fairer,\nless accurate", 1, -1, "right", "bottom"), ("less fair &\nless accurate", -1, -1, "left", "bottom")]


def res_any(d, fair):
    """In the 2-D view a step counts as resolved if it is resolved on either coordinate (CSV flags)."""
    return (as_bool(d["tau_I_dAUC_resolved"]) | as_bool(d[f"tau_I_{fair}_resolved"])).to_numpy()


def arrow(ax, x0, y0, x1, y1, col, dashed=False, lw=1.4, z=3, head="filled"):
    style = ("-|>,head_length=0.45,head_width=0.22" if head == "filled"
             else "->,head_length=0.5,head_width=0.3")          # open head: direction only
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=z,
                arrowprops=dict(arrowstyle=style, color=col, lw=lw, linestyle=DASHED if dashed else "-",
                                shrinkA=0, shrinkB=0))


def shade_quadrants(ax, xl, yl):
    ax.fill_between([0, xl[1]], 0, yl[1], color="#2a78d6", alpha=0.07, lw=0, zorder=0)    # both better
    ax.fill_between([xl[0], 0], yl[0], 0, color="#e34948", alpha=0.07, lw=0, zorder=0)    # both worse


def fig1d(fair="negDP", name="fig1_tradeoff_direction", zoom=0.03):
    fx = main[f"tau_pkg_{fair}_mean"].to_numpy(); fy = main["tau_pkg_dAUC_mean"].to_numpy()
    ix = main[f"tau_I_{fair}_mean"].to_numpy(); iy = main["tau_I_dAUC_mean"].to_numpy()
    solid = res_any(main, fair)
    fam = main["family"].to_numpy()

    def centred(v, pad=0.25):
        """Symmetric about 0, so the origin sits in the middle and all four quadrants are visible."""
        L = np.abs(v).max() * (1 + pad)
        return (-L, L)
    # each panel on its own scale (tick labels carry it); x and y scaled separately in a square box
    LIMS = [(centred(fx), centred(fy)), (centred(ix), centred(iy))]

    def draw(ax, xs, ys, status, lw_scale=1.0, color=ROLE["level"]):
        # direction only, in the panel's role colour; resolution is reported in Table 1
        for x, y in zip(xs, ys):
            arrow(ax, 0, 0, x, y, color, lw=0.9 * lw_scale)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.7), gridspec_kw=dict(wspace=0.28))
    panels = [(fx, fy, np.ones(len(main), bool), f"(a)  Package  ${T_PKG_P}$:  $B \\rightarrow {M_PLUS}$",
               ROLE["level"]),
              (ix, iy, solid, f"(b)  Intervention  ${T_I_P}$:  ${M_MINUS} \\rightarrow {M_PLUS}$",
               ROLE["intervention"])]
    for k, (ax, (xs, ys, status, title, col)) in enumerate(zip(axes, panels)):
        XL, YL = LIMS[k]
        shade_quadrants(ax, XL, YL)
        ax.axhline(0, color=INK2, lw=0.7, zorder=1); ax.axvline(0, color=INK2, lw=0.7, zorder=1)
        ax.grid(True, color=GRID, lw=0.5, zorder=0)
        draw(ax, xs, ys, status, color=col)
        on_axis = int(((np.sign(xs) == 0) | (np.sign(ys) == 0)).sum())
        print(f"{name} panel {'ab'[k]}: {on_axis} cell(s) on an axis, excluded from the quadrant counts")
        for qn, sx, sy, ha, va in QUAD:
            n = int(((np.sign(xs) == sx) & (np.sign(ys) == sy)).sum())
            ax.text(0.98 if ha == "right" else 0.02, 0.98 if va == "top" else 0.02, f"{qn}: {n}",
                    transform=ax.transAxes, ha=ha, va=va, fontsize=7.5, color=INK, zorder=9, linespacing=1.1,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))
        ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_box_aspect(1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5)); ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_xlabel(f"{METRIC_LABEL[fair]} contrast  (→ fairer)")
        if k == 1:   # inset: the many short intervention steps near the origin
            ins = ax.inset_axes([0.37, 0.645, 0.27, 0.27])  # empty area above the zero line
            ins.axhline(0, color=INK2, lw=0.5); ins.axvline(0, color=INK2, lw=0.5)
            shade_quadrants(ins, (-zoom, zoom), (-zoom, zoom))
            draw(ins, xs, ys, status, lw_scale=0.75, color=col)
            ins.set_xlim(-zoom, zoom); ins.set_ylim(-zoom, zoom); ins.set_aspect("equal")
            ins.set_xticks([]); ins.set_yticks([])
            ins.set_title(f"zoom ±{zoom:g}", fontsize=7.5, loc="left", pad=2)
            for sp in ins.spines.values():
                sp.set_visible(True); sp.set_color("#9a9993"); sp.set_linewidth(0.6)
            ax.indicate_inset_zoom(ins, edgecolor="#9a9993")
    axes[0].set_ylabel(r"$\Delta$AUC contrast  (→ more accurate)")
    handles = [Line2D([], [], color=ROLE["level"], lw=1.2, label=f"${T_PKG_P}$, one arrow per cell"),
               Line2D([], [], color=ROLE["intervention"], lw=1.2, label=f"${T_I_P}$, one arrow per cell")]
    boxed_legend(fig, name, handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.045), fontsize=7.2,
               columnspacing=1.2, handlelength=1.8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    savefig(fig, 1, name)


fig1d("negDP", "figS1_tradeoff_direction")                 # appendix: direction in the two-coordinate plane
fig1d("negEO", "figS1_tradeoff_direction_negEO")          # appendix, -Delta_EO

# ======== notebook cell 8 ========
# Fig S1 (appendix): absolute AUC and -Delta_DP (-Delta_EO), family x dataset (datasets with >= MIN_CELLS cells).
# Labels: only cells whose intervention step is resolved; placed to clear markers, arrows and other labels.
MIN_CELLS = 4   # bail (11), credit (9), german (8); smaller datasets are left out
OFFSETS = [(dx * s, dy * s, ha, va, s) for s in (2, 3, 4)
           for dx, dy, ha, va in [(5, -5, "left", "top"), (5, 5, "left", "bottom"), (-5, -5, "right", "top"),
                                  (-5, 5, "right", "bottom"), (0, 8, "center", "bottom"), (0, -8, "center", "top"),
                                  (9, 0, "left", "center"), (-9, 0, "right", "center")]]


def _segment_points(ax, segs, n=24):
    """Display-space sample points along line segments (arrows, dotted steps)."""
    pts = []
    for (x0, y0), (x1, y1) in segs:
        t = np.linspace(0, 1, n)[:, None]
        pts.append(ax.transData.transform(np.c_[x0 + t[:, 0] * (x1 - x0), y0 + t[:, 0] * (y1 - y0)]))
    return np.vstack(pts) if pts else np.empty((0, 2))


# hand-set offsets (points) where the automatic search cannot clear a crowded spot: (dataset, label) -> offset
LABEL_OVERRIDE = {("attr_b", "NIFTY / Credit"): (-22, -16, "right", "top")}


def place_labels(ax, items, points, segs, fontsize=7.2, pad_px=3, dataset=None):
    rend = ax.figure.canvas.get_renderer()
    obstacles = np.vstack([ax.transData.transform(np.asarray(points)), _segment_points(ax, segs)])
    box = ax.get_window_extent(rend)
    taken = []
    for text, x, y in items:
        best, best_hits = None, None
        forced = LABEL_OVERRIDE.get((dataset, text))
        for dx, dy, ha, va, s in ([forced + (2,)] if forced else OFFSETS):
            # every label sits a little away from its point with a thin leader line, so it is
            # unambiguous which marker it names even where markers crowd
            leader = dict(arrowstyle="-", lw=0.5, color="black", shrinkA=0, shrinkB=2.5)
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, color="black", zorder=8, arrowprops=leader)
            bb = t.get_window_extent(rend)
            inside = (obstacles[:, 0] > bb.x0 - pad_px) & (obstacles[:, 0] < bb.x1 + pad_px) & \
                     (obstacles[:, 1] > bb.y0 - pad_px) & (obstacles[:, 1] < bb.y1 + pad_px)
            hits = int(inside.sum()) + 100 * sum(bb.overlaps(o) for o in taken) \
                + (1000 if not (box.x0 <= bb.x0 and bb.x1 <= box.x1 and box.y0 <= bb.y0 and bb.y1 <= box.y1) else 0)
            if best is None or hits < best_hits:
                if best is not None:
                    best[0].remove()
                best, best_hits = (t, bb), hits
            else:
                t.remove()
            if hits == 0:
                break
        taken.append(best[1])


def fig1e(fair="negDP", name="figS1_arms_by_family_dataset"):
    """Appendix: rows = method family, columns = dataset (>= MIN_CELLS primary cells).
    Axes are shared within a column, so the three families on one dataset sit on one scale."""
    col, _ = SIGN[fair]
    counts = main["dataset"].value_counts()
    keep = [ds for ds in counts.index if counts[ds] >= MIN_CELLS]
    dropped = {ds: int(counts[ds]) for ds in counts.index if counts[ds] < MIN_CELLS}
    print(f"{fair}: columns {keep}; left out (n < {MIN_CELLS}): {dropped}")
    solid_all = res_any(main, fair)
    # each panel scaled to its own points so the shifts are visible
    fig, axes = plt.subplots(len(FAMILY_ORDER), len(keep), figsize=(7.0, 7.2))
    jobs = []
    for j, ds in enumerate(keep):
        dsel = main["dataset"].eq(ds).to_numpy()
        bx, by = -main.loc[dsel, f"bc_{col}"].mean(), main.loc[dsel, "bc_auc"].mean()
        for i, fam in enumerate(FAMILY_ORDER):
            ax = axes[i, j]
            sel = dsel & main["family"].eq(fam).to_numpy()
            g, solid = main[sel], solid_all[sel]
            c, mk = FAM_STYLE[fam]
            ax.grid(True, color=GRID, lw=0.5, zorder=0)
            ax.scatter(bx, by, marker="*", s=60, color="black", zorder=6)
            items, points, segs = [], [(bx, by)], []
            for (_, r), s in zip(g.iterrows(), solid):
                x0, y0, x1, y1 = -r[f"m0_{col}"], r.m0_auc, -r[f"m1_{col}"], r.m1_auc
                ax.plot([bx, x0], [by, y0], color="#9a9993", lw=0.8, ls=":", zorder=1)      # tau_{B->-I}
                ax.scatter(x0, y0, marker=mk, s=26, facecolor="white", edgecolor=C_B, lw=0.7, zorder=4)
                ax.scatter(x1, y1, marker=mk, s=28, facecolor=_tint(C_A, 0.35), edgecolor=C_A, lw=0.7, zorder=5)
                arrow(ax, x0, y0, x1, y1, C_A, dashed=not s, lw=1.1 if s else 0.8, head="open")
                points += [(x0, y0), (x1, y1)]; segs += [((bx, by), (x0, y0)), ((x0, y0), (x1, y1))]
                if s:
                    items.append((r.method + ("" if r.backbone == "GCN" else f"-{r.backbone}"), x1, y1))
            xs, ys = zip(*points)
            px, py = 0.15 * (max(xs) - min(xs)) or 0.01, 0.15 * (max(ys) - min(ys)) or 0.01
            ax.set_xlim(min(xs) - px, max(xs) + px); ax.set_ylim(min(ys) - py, max(ys) + py)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
            if i == 0:
                ax.set_title(f"{ds}  (n = {int(dsel.sum())})", loc="left", fontsize=8)
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
            if i == len(FAMILY_ORDER) - 1:
                ax.set_xlabel(f"{METRIC_LABEL[fair]}  (→ fairer)")
            if j == 0:
                ax.set_ylabel(f"{FAMILY_LABEL[fam]}\nAUC  (→ more accurate)")
            jobs.append((ax, items, points, segs, ds))
    handles = [Line2D([], [], marker="*", ls="none", color="black", ms=7, label="$B$ (common baseline)"),
               Line2D([], [], marker="o", ls="none", color=INK2, mfc="white", mew=0.6, ms=5.5,
                      label=f"${M_MINUS}$ (intervention off)"),
               Line2D([], [], marker="o", ls="none", color=INK2, mfc=_tint(INK2, 0.35), mew=0.6, ms=5.5, label=f"${M_PLUS}$ (intervention on)"),
               Line2D([], [], color="#9a9993", lw=0.8, ls=":", label=f"${T_NON}$:  $B \\rightarrow {M_MINUS}$"),
               Line2D([], [], color=INK2, lw=1.3, label=f"${T_I_P}$:  ${M_MINUS} \\rightarrow {M_PLUS}$, resolved"),
               Line2D([], [], color=INK2, lw=0.9, ls=DASHED, label=f"${T_I_P}$, unresolved")]
    handles += [Line2D([], [], marker=FAM_STYLE[f][1], ls="none", color=FAM_STYLE[f][0], mfc=_tint(FAM_STYLE[f][0], 0.35), mew=0.6, ms=5.5,
                       label=FAMILY_LABEL[f]) for f in FAMILY_ORDER]
    boxed_legend(fig, name, handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=7.2,
               columnspacing=1.2, handlelength=1.8)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.canvas.draw()
    for ax, items, points, segs, ds in jobs:
        ax.set_xlim(ax.get_xlim()); ax.set_ylim(ax.get_ylim())    # freeze limits before placing labels
        place_labels(ax, items, points, segs, dataset=ds)
    savefig(fig, 1, name)


fig1e("negDP", "figS1_arms_by_family_dataset")               # appendix
fig1e("negEO", "figS1_arms_by_family_dataset_negEO")         # appendix, -Delta_EO

# ======== notebook cell 9 ========
# Fig 1 (alternative): intervention attribution and intervention effect space, one point per primary cell.
#   (a) x = tau_{B->-I} (surrounding package) on -Delta_DP, y = tau_{-I->+I}; grey wedge = |x| > |y|
#   (b) x = tau_{-I->+I} on Delta AUC, y = tau_{-I->+I} on -Delta_DP
# Colour / marker = method family; filled = tau_{-I->+I} resolved on the fairness coordinate (CSV flag).
# The main view is a zoom; the inset shows the full space with the zoom window dashed.
DATASET_PRETTY = {"bail": "Bail", "credit": "Credit", "german": "German", "income": "Income",
                  "pokec_n": "Pokec-n", "pokec_z": "Pokec-z", "pokec_n_g": "Pokec-n-g", "pokec_z_g": "Pokec-z-g"}
# zoom windows chosen to hold the bulk of the cells; cells outside are named in the panel and shown in the inset
# the main view is a zoom on the bulk; resolved cells outside it are named in the caption (printed below)
ATTR_VIEW = {"negDP": {"a": ((-0.17, 0.12), (-0.085, 0.105)), "b": ((-0.052, 0.032), (-0.045, 0.103))},
             "negEO": {"a": ((-0.17, 0.12), (-0.085, 0.105)), "b": ((-0.052, 0.032), (-0.045, 0.103))}}


def cell_name(r):
    m = r.method + (f"-{r.configuration}" if r.configuration not in ("default", r.backbone) else "")
    return f"{m} / {DATASET_PRETTY.get(r.dataset, r.dataset)}"


def fig1_attribution(fair="negDP", name="fig1_intervention_attribution"):
    y = main[f"tau_I_{fair}_mean"].to_numpy()
    xa = main[f"tau_nonint_{fair}_mean"].to_numpy()
    xb = main["tau_I_dAUC_mean"].to_numpy()
    res = as_bool(main[f"tau_I_{fair}_resolved"]).to_numpy()
    fam = main["family"].to_numpy()
    names = [cell_name(r) for r in main.itertuples()]
    fl = METRIC_LABEL[fair].strip("$")
    panels = [("a", xa, f"Surrounding-package fairness contrast  ${TAU_NONINT}$ (${{{fl}}}$)",
               "(a)  Fairness attribution"),
              ("b", xb, rf"Intervention utility contrast  ${TAU_I}$ ($\Delta$AUC)",
               "(b)  Intervention utility–fairness trade-off")]
    fig, axes = plt.subplots(1, 2, figsize=(5.66, 2.38))
    jobs = []
    for ax, (key, x, xlabel, title) in zip(axes, panels):
        (x0, x1), (y0, y1) = ATTR_VIEW[fair][key]
        if key == "a":   # wedge |x| > |y| and its diagonals
            big = 1.0
            ax.fill([-big, big, big, -big], [-big, big, -big, big], color="#9a9993", alpha=0.12, lw=0, zorder=0)
            # say what the shaded region means inside it, so the legend is not needed to read the panel
            ax.text(0.02, 0.985, "surrounding package larger\n" +
                    r"$|\tau_{B\rightarrow {-}I}| > |\tau_{{-}I\rightarrow {+}I}|$", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7, color=INK2, linespacing=1.3, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white", alpha=0.85,
                              edgecolor="none"))
            for s in (1, -1):
                ax.plot([-big, big], [-s * big, s * big], color=MUTED, lw=0.7, ls="--", zorder=1)
        ax.axhline(0, color=MUTED, lw=0.7, zorder=1); ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
        for f in FAMILY_ORDER:
            c, mk = FAM_STYLE[f]
            s = fam == f
            ax.scatter(x[s & ~res], y[s & ~res], marker=mk, s=26, facecolor="white", edgecolor=MUTED, lw=0.8, zorder=3)
            ax.scatter(x[s & res], y[s & res], marker=mk, s=44, facecolor=_tint(C_A, 0.3), edgecolor=C_A, lw=0.9,
                       zorder=4)
        if key == "b":   # x = utility, y = fairness: shade the quadrants where both coordinates agree
            shade_quadrants(ax, (x0, x1), (y0, y1))
            on_axis = int(((np.sign(x) == 0) | (np.sign(y) == 0)).sum())
            q = {(sx, sy): int(((np.sign(x) == sx) & (np.sign(y) == sy)).sum()) for sx in (1, -1) for sy in (1, -1)}
            print(f"{name} (b) quadrants (utility, fairness): {q}; {on_axis} cell(s) on an axis")
            for (sx, sy), lab, tx, ty, ha, va in (((1, 1), "fairer &\nmore accurate", 0.98, 0.98, "right", "top"),
                                                  ((-1, -1), "less fair &\nless accurate", 0.02, 0.02, "left", "bottom")):
                ax.text(tx, ty, f"{lab}: {q[(sx, sy)]}", transform=ax.transAxes, ha=ha, va=va, fontsize=7,
                        color=INK2, linespacing=1.2, zorder=9,
                        bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6)); ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontsize=8)
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        out = [n for n, r_, i_ in zip(names, res, inside) if r_ and not i_]
        print(f"{name} ({key}): {int((~inside).sum())} cell(s) outside the view"
              + (f"; resolved among them: {', '.join(out)}" if out else ""))
        ins = ax.inset_axes([0.72, 0.035, 0.26, 0.26] if key == "a"      # clear of the note in (a)
                            else [0.035, 0.70, 0.26, 0.26])              # clear of the points in (b)
        ins.axhline(0, color=MUTED, lw=0.5); ins.axvline(0, color=MUTED, lw=0.5)
        for f in FAMILY_ORDER:
            _, mk = FAM_STYLE[f]
            m = fam == f
            ins.scatter(x[m & ~res], y[m & ~res], marker=mk, s=4, facecolor="white", edgecolor=MUTED, lw=0.35)
            ins.scatter(x[m & res], y[m & res], marker=mk, s=7, facecolor=_tint(C_A, 0.3), edgecolor=C_A, lw=0.4)
        ins.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=0.5, ec=INK2))
        px, py = 0.07 * float(np.ptp(x)), 0.07 * float(np.ptp(y))
        ins.set_xlim(min(x.min(), x0) - px, max(x.max(), x1) + px)
        ins.set_ylim(min(y.min(), y0) - py, max(y.max(), y1) + py)
        ins.set_xticks([]); ins.set_yticks([])
        ins.set_facecolor("white")
        for sp in ins.spines.values():
            sp.set_visible(True); sp.set_color(MUTED); sp.set_linewidth(0.5)
        ins.set_title("full range", fontsize=7, pad=1.5, loc="left", color=INK2)
    # both panels plot the same quantity on y, so the label is written once, on the left. Their
    # ranges differ, so the tick labels stay on both.
    axes[0].set_ylabel(f"Intervention fairness contrast ${TAU_I}$ (${{{fl}}}$)", fontsize=7.5)
    handles = [Line2D([], [], marker=FAM_STYLE[f][1], ls="none", color=MUTED, mfc="white", mew=0.8, ms=5.5,
                      label=FAMILY_LABEL[f]) for f in FAMILY_ORDER]
    handles += [Line2D([], [], marker="o", ls="none", color=C_A, mfc=_tint(C_A, 0.3), mew=0.9, ms=6,
                       label=f"${TAU_I}$ resolved"),
                Line2D([], [], marker="o", ls="none", color=MUTED, mfc="white", mew=0.8, ms=5, label="unresolved")]
    boxed_legend(fig, name, handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=7.2,
                 columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=2.0)
    fig.canvas.draw()
    for (ax, items, pts), key in zip(jobs, ("a", "b")):
        # cells are not named in the panel (too crowded); the resolved ones are listed for the caption
        print(f"{name} ({key}) resolved, in view:", ", ".join(n for n, _, _ in items))
    savefig(fig, 1, name)


fig1_attribution("negDP", "fig1_intervention_attribution")          # main text: Figure 1
fig1_attribution("negEO", "figS1_intervention_attribution_negEO")   # appendix, -Delta_EO

# ======== notebook cell 12 ========
# Experiment 1.1: FnRGNN under both tasks, kept apart from Experiment 1
fnr = {"classification": dfs["main_ta"].copy(), "regression": dfs["main_reg"].copy()}
TASK_COORDS = {"classification": ["dAUC", "negDP", "negEO"], "regression": ["negMSE", "negMeanGap", "negWD"]}
COORD_LABEL = {**METRIC_LABEL, "negMSE": r"$-$MSE", "negMeanGap": r"$-$MeanGap", "negWD": r"$-$WD"}
COORD_TEX = {**METRIC_TEX, "negMSE": r"$-$MSE", "negMeanGap": r"$-$MeanGap", "negWD": r"$-$WD"}
TASK_TITLE = {"classification": "(a)  Classification", "regression": "(b)  Regression"}
FNR_DATASETS = ["german", "pokec_n", "pokec_z"]
for task, d in fnr.items():
    assert set(d["method"]) == {"FnRGNN"} and sorted(d["dataset"]) == sorted(FNR_DATASETS), task
    for m in TASK_COORDS[task]:   # identity check on the point estimates
        gap = (d[f"tau_pkg_{m}_mean"] - d[f"tau_nonint_{m}_mean"] - d[f"tau_I_{m}_mean"]).abs().max()
        assert gap < 1e-9, (task, m, gap)
    print(task, d[["dataset", "n_units"]].to_dict("records"))


def fig_fnr(name="figS1_1_fnrgnn_two_tasks"):
    fig, axes = plt.subplots(1, 2, figsize=(5.66, 2.38))
    for ax, task in zip(axes, ["classification", "regression"]):
        d = fnr[task].set_index("dataset").loc[FNR_DATASETS]
        rows = [(m, ds) for m in TASK_COORDS[task] for ds in FNR_DATASETS]
        y = np.arange(len(rows))[::-1].astype(float)
        zero_line(ax)
        xgrid(ax)
        for yi, (m, ds) in zip(y, rows):
            r = d.loc[ds]
            ci(ax, r[f"tau_nonint_{m}_lo"], r[f"tau_nonint_{m}_hi"], yi + 0.16, C_B, lw=1.0)
            point(ax, r[f"tau_nonint_{m}_mean"], yi + 0.16, C_B, True, "o", 5.5, mew=0.6)  # no status: filled
            ci(ax, r[f"tau_I_{m}_lo"], r[f"tau_I_{m}_hi"], yi - 0.16, C_A, lw=1.0)
            point(ax, r[f"tau_I_{m}_mean"], yi - 0.16, C_A, bool(as_bool(pd.Series([r[f"tau_I_{m}_resolved"]]))[0]),
                  "s", 5.5, mew=0.6)
            ax.plot(r[f"tau_pkg_{m}_mean"], yi, marker="D", ms=3.5, color="black", ls="none", zorder=6)
        for k in range(1, len(TASK_COORDS[task])):
            ax.axhline(y[k * len(FNR_DATASETS)] + 0.5, color=INK2, lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(FNR_DATASETS * len(TASK_COORDS[task]) if ax is axes[0] else FNR_DATASETS * 3, fontsize=7.5)
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for k, m in enumerate(TASK_COORDS[task]):
            yc = y[k * len(FNR_DATASETS) + 1]
            ax.text(1.01, yc, COORD_LABEL[m], transform=trans, va="center", ha="left", fontsize=7.8, rotation=90)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(TASK_TITLE[task], loc="left", fontsize=8)
        ax.set_xlabel("effect (higher is better)")
    handles = [Line2D([], [], marker="o", color=C_B, mfc=_tint(C_B, 0.35), ls="-", lw=1.0, ms=5.5, mew=0.6, label=f"${TAU_NONINT}(P)$"),
               Line2D([], [], marker="s", color=C_A, mfc=_tint(C_A, 0.35), ls="-", lw=1.0, ms=5.5, mew=0.6,
                      label=f"${TAU_I}(P)$, resolved"),
               Line2D([], [], marker="s", color=C_A, mfc="white", ls="-", lw=1.0, ms=5.5, mew=0.6,
                      label=f"${TAU_I}(P)$, unresolved"),
               Line2D([], [], marker="D", color="black", ls="none", ms=3.5, label=f"${TAU_PKG}(P)$"),
               Line2D([], [], color=INK2, lw=1.0, label="95% interval")]
    boxed_legend(fig, name, handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.01), fontsize=7.2,
               columnspacing=1.2, handlelength=2.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.5)
    savefig(fig, "1_1", name)


fig_fnr()

