"""X26 figures: FairGB selection-support replication (Bail).

Reads only the analyzer outputs (x26_<ds>_summary.csv, x26_<ds>_cell_table.csv,
x26_<ds>_selected_epochs.csv). It computes no new estimate. Datasets are never
merged into one curve; each gets its own figure.

Main:   tau_int(-dDP) at the pre-registered fixed epochs, with the cap<=199 and
        full<=H-1 selected values marked, and a strip panel showing where
        sigma_c^BCE selects for each arm.
Second: the three-step bridge, frozen H=200 -> cap<=199 on the native-length
        trajectory -> full support, with S and T annotated.
Supplement: -dEO and dAUC, and the sigma_c^AUC versions.

    python harness/experiments/plot_x26_fairgb.py --indir harness/results/x26 --dataset bail \
        --outdir harness/results/figures
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
C_TRAJ = "#2a78d6"      # slot 1: fixed-epoch trajectory
C_SEL = "#eb6834"       # slot 2: selected checkpoints
GRID = {"bail": (25, 50, 100, 150, 200, 300, 500, 750, 1000, 1250, 1499),
        "credit": (25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 1999)}
CAP_LAST = 199
LAB = {"ndp": "τ_int (−ΔDP)", "neo": "τ_int (−ΔEO)", "auc": "τ_int (ΔAUC)"}
SIGL = {"bce": "σ_c^BCE", "auc": "σ_c^AUC"}


def style(ax, xgrid=False):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK2, width=1, length=3)
    ax.grid(axis="x" if xgrid else "y", color=GRID_C, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def q(s, name):
    r = s[s.quantity == name]
    if r.empty:
        raise KeyError(name)
    return r.iloc[0]


def marker(ax, x, r, color, mk, label=None, dx=0):
    ax.errorbar(x + dx, r["mean"], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                fmt="none", ecolor=color, elinewidth=1.5, zorder=3)
    ax.plot(x + dx, r["mean"], marker=mk, markersize=9, zorder=4,
            markerfacecolor=color if r["resolved"] else SURFACE,
            markeredgecolor=color, markeredgewidth=2)
    if label:
        ax.annotate(label, (x + dx, r["mean"]), textcoords="offset points",
                    xytext=(10, 0), ha="left", va="center", fontsize=8.5, color=INK2)


def trajectory_fig(s, se, ds, coord, sig, outdir, outs):
    H = GRID[ds][-1] + 1
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.6, 6.8),
                                  gridspec_kw=dict(height_ratios=[3, 1.2], hspace=0.38))
    style(ax)
    ax.axhline(0, color=BASE, linewidth=1)
    ax.axvline(CAP_LAST + 1, color=GRID_C, linewidth=1, zorder=0)
    rows = [q(s, f"fx{t}.{coord}") for t in GRID[ds]]
    m = np.array([r["mean"] for r in rows])
    lo = np.array([r["lo"] for r in rows])
    hi = np.array([r["hi"] for r in rows])
    x = np.array(GRID[ds], float)
    ax.fill_between(x, lo, hi, color=C_TRAJ, alpha=0.10, linewidth=0, zorder=2)
    ax.plot(x, m, color=C_TRAJ, linewidth=2, zorder=3)
    ax.plot(x, m, linestyle="none", marker="o", markersize=6, markerfacecolor=C_TRAJ,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
    marker(ax, CAP_LAST + 1, q(s, f"{sig}.cap.{coord}"), C_SEL, "^", "cap ≤ 199")
    full_r = q(s, f"{sig}.full.{coord}")
    marker(ax, H - 1, full_r, C_SEL, "s")
    ax.annotate(f"full ≤ {H - 1}", (H - 1, full_r["mean"]), textcoords="offset points",
                xytext=(-12, 14), ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xlim(0, H * 1.12)
    ax.set_ylabel(LAB[coord] + "  at fixed epoch", color=INK2)
    ax.set_title(f"FairGB / {ds}: intervention effect along the native-length trajectory",
                 color=INK, fontsize=11, loc="left", pad=28)
    ax.text(0, 1.015, f"blue: no checkpoint selection · orange: {SIGL[sig]} selected, "
            f"capped vs full support\nmean over 6 splits × 5 runs · bands/bars = 95% paired "
            "hierarchical bootstrap · filled = resolved", transform=ax.transAxes,
            fontsize=8, color=MUTED, va="bottom")

    style(ax2, xgrid=True)
    ax2.axvline(CAP_LAST + 1, color=BASE, linewidth=1)
    g = se[se.selector == sig]
    rng = np.random.default_rng(0)
    rowspec = [("full", "M0"), ("full", "M1"), ("cap", "M0"), ("cap", "M1")]
    for i, (supp, arm) in enumerate(rowspec):
        e = g[(g.support == supp) & (g.arm == arm)].epoch.to_numpy(float)
        ax2.plot(e, i + rng.uniform(-0.16, 0.16, len(e)), linestyle="none",
                 marker="s" if supp == "full" else "^", markersize=5,
                 markerfacecolor=C_SEL if arm == "M1" else SURFACE,
                 markeredgecolor=C_SEL, markeredgewidth=1.2, alpha=0.9)
        ax2.plot([np.median(e)] * 2, [i - 0.3, i + 0.3], color=INK2, linewidth=2)
    ax2.set_yticks(range(len(rowspec)))
    ax2.set_yticklabels([f"{s_} · {a_}" for s_, a_ in rowspec], fontsize=8.5)
    ax2.set_ylim(-0.6, len(rowspec) - 0.4)
    ax2.set_xlim(0, H * 1.12)
    ax2.set_xlabel("epoch", color=INK2)
    ax2.set_title(f"where {SIGL[sig]} selects (filled = M1, hollow = M0; bar = median)",
                  color=INK2, fontsize=8.5, loc="left")
    fig.tight_layout()
    save(fig, outdir, f"x26_{ds}_trajectory_{coord}_{sig}", outs)


def bridge_fig(s, ds, coord, outdir, outs):
    H = GRID[ds][-1] + 1
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    style(ax)
    ax.axhline(0, color=BASE, linewidth=1)
    for k, sig in enumerate(("bce", "auc")):
        base = k * 4
        h200 = q(s, f"{sig}.h200.{coord}")
        cap = q(s, f"{sig}.cap.{coord}")
        full = q(s, f"{sig}.full.{coord}")
        ax.plot([base, base + 1, base + 2], [h200["mean"], cap["mean"], full["mean"]],
                color=C_SEL, linewidth=2, zorder=2)
        ax.plot(base, h200["mean"], marker="o", markersize=9, markerfacecolor=MUTED,
                markeredgecolor=MUTED, zorder=4)
        marker(ax, base + 1, cap, C_SEL, "^")
        marker(ax, base + 2, full, C_SEL, "s")
        S, T = q(s, f"{sig}.S.{coord}"), q(s, f"{sig}.T.{coord}")
        ax.annotate(f"T {T['mean']:+.3f}\n[{T['lo']:+.3f}, {T['hi']:+.3f}]",
                    ((2 * base + 1) / 2, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
        ax.annotate(f"S {S['mean']:+.3f}\n[{S['lo']:+.3f}, {S['hi']:+.3f}]",
                    (base + 1.5, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, 26), textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
    ax.set_xticks([0, 1, 2, 4, 5, 6])
    ax.set_xticklabels(["frozen\nH = 200", f"cap ≤ {CAP_LAST}", f"full ≤ {H - 1}",
                        "frozen\nH = 200", f"cap ≤ {CAP_LAST}", f"full ≤ {H - 1}"], fontsize=8.5)
    ax.set_xlim(-0.6, 6.8)
    ax.set_ylabel(LAB[coord], color=INK2)
    for k, sig in enumerate(("bce", "auc")):
        ax.annotate(SIGL[sig], (k * 4 + 1, 0.97), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=9.5, color=INK)
    ax.set_title(f"FairGB / {ds}: three-step attribution bridge",
                 color=INK, fontsize=11, loc="left", pad=30)
    ax.text(0, 1.015, "gray: the frozen controlled run · orange: the native-length trajectory, "
            "selector capped then full\nT = cap − H200 (prefix/run), S = full − cap "
            "(selection support) · filled = resolved", transform=ax.transAxes,
            fontsize=8, color=MUTED, va="bottom")
    fig.tight_layout()
    save(fig, outdir, f"x26_{ds}_bridge_{coord}", outs)


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
    ap.add_argument("--dataset", default="bail")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    s = pd.read_csv(os.path.join(a.indir, f"x26_{a.dataset}_summary.csv"))
    se = pd.read_csv(os.path.join(a.indir, f"x26_{a.dataset}_selected_epochs.csv"))
    outs = []
    trajectory_fig(s, se, a.dataset, "ndp", "bce", a.outdir, outs)
    bridge_fig(s, a.dataset, "ndp", a.outdir, outs)
    for coord in ("neo", "auc"):
        trajectory_fig(s, se, a.dataset, coord, "bce", a.outdir, outs)
        bridge_fig(s, a.dataset, coord, a.outdir, outs)
    trajectory_fig(s, se, a.dataset, "ndp", "auc", a.outdir, outs)
    for p in outs:
        print("[written]", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
