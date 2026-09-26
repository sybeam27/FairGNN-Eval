"""Figure D -- NIFTY / German intervention effect along the fixed-epoch grid.

The X25 trajectory artifact, read as written. Only the pre-registered epoch grid
is drawn: consecutive points are joined for legibility, but no value is
interpolated and no point exists that was not measured.

D0 is the Arm-A augmentation/validation-view configuration, D1 the NIFTY native
one. The strip below shows where sigma_c^BCE actually selects, per arm, on the
full-support trajectory.

The sigma_c^AUC version is written as a supplementary figure, not the main one.

    python harness/experiments/x28_figures/fig_d_trajectory.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K                                                   # noqa: E402

GRID = (25, 50, 100, 150, 200, 300, 400, 600, 800, 1000)
FULL_CAP = 1000
DCOL = {"D0": "#2a78d6", "D1": "#9a6ad6"}
DMK = {"D0": "o", "D1": "D"}
DLS = {"D0": (0, (4, 2)), "D1": "-"}
DLAB = {"D0": "D0  Arm-A configuration", "D1": "D1  NIFTY native configuration"}


def trajectory(s, D, coord):
    rows = [K.q(s, f"{D}.fx{e}.{coord}") for e in GRID]
    return (np.array([r["mean"] for r in rows]),
            np.array([r["lo"] for r in rows]),
            np.array([r["hi"] for r in rows]))


def figure_d(s, se, src, ck, outs, sigma="bce", coord="ndp", name="figD_nifty_trajectory"):
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(5.8, 4.4),
        gridspec_kw=dict(height_ratios=[3.0, 1.0], hspace=0.34))
    K.style(ax, grid="y")
    ax.axhline(0, color=K.BASEC, linewidth=0.9, zorder=1)
    x = np.array(GRID, float)
    ys = []
    for D in ("D0", "D1"):
        m, lo, hi = trajectory(s, D, coord)
        col = DCOL[D]
        ax.fill_between(x, lo, hi, color=col, alpha=0.10, linewidth=0, zorder=2)
        ax.plot(x, m, color=col, linewidth=1.5, linestyle=DLS[D], zorder=3)
        ax.plot(x, m, linestyle="none", marker=DMK[D], markersize=4.2,
                markerfacecolor=col, markeredgecolor=K.SURFACE,
                markeredgewidth=0.8, zorder=4)
        ys += list(lo) + list(hi)
        for e, mm, ll, hh in zip(GRID, m, lo, hi):
            src.add(figure="D", selector=sigma, coord=coord, config=D, epoch=e,
                    mean=mm, lo=ll, hi=hh)
    ax.set_xlim(0, GRID[-1] * 1.06)
    ck.limits(ax, x, ys, f"Figure D main ({sigma})")
    ax.set_ylabel("intervention effect on −ΔDP" if coord == "ndp"
                  else f"intervention effect ({coord})",
                  color=K.INK2, fontsize=K.FS["label"])
    handles = [plt.Line2D([], [], color=DCOL[D], lw=1.5, linestyle=DLS[D],
                          marker=DMK[D], markersize=4.5, label=DLAB[D])
               for D in ("D0", "D1")]
    ax.legend(handles=handles, loc="lower left", frameon=False,
              fontsize=K.FS["note"], labelcolor=K.INK2)
    ax.set_title("NIFTY / German: the effect along the training trajectory",
                 color=K.INK, fontsize=K.FS["title"], loc="left", pad=22)
    # the fixed-epoch curves involve no checkpoint selection at all, so naming a
    # selector here would be wrong -- that is why the top panel is identical in
    # the BCE and AUC versions. The selector belongs to the strip below.
    ax.text(0, 1.015, "pre-registered epoch grid only · no checkpoint selection: "
            "each point is the effect at that fixed epoch\nbands are 95% paired "
            "hierarchical bootstrap intervals",
            transform=ax.transAxes, fontsize=K.FS["note"], color=K.MUTED, va="bottom")

    # where the selector lands, on the full-support trajectory
    K.style(ax2, grid="x")
    g = se[(se.sigma == sigma) & (se.cap == FULL_CAP)]
    rows = [("D1", "m1_epoch", "M_on"), ("D1", "m0_epoch", "M_off"),
            ("D0", "m1_epoch", "M_on"), ("D0", "m0_epoch", "M_off")]
    rng = np.random.default_rng(0)
    for i, (D, col_, arm) in enumerate(rows):
        e = g[g.D == D][col_].to_numpy(float)
        ax2.plot(e, i + rng.uniform(-0.16, 0.16, len(e)), linestyle="none",
                 marker=DMK[D], markersize=3.6,
                 markerfacecolor=DCOL[D] if arm == "M_on" else K.SURFACE,
                 markeredgecolor=DCOL[D], markeredgewidth=0.9, alpha=0.9)
        ax2.plot([np.median(e)] * 2, [i - 0.30, i + 0.30], color=K.INK2,
                 linewidth=1.4)
        src.add(figure="D_strip", selector=sigma, config=D, arm=arm,
                n=len(e), median_epoch=float(np.median(e)),
                min_epoch=float(e.min()), max_epoch=float(e.max()))
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([f"{D} · {arm}" for D, _, arm in rows],
                        fontsize=K.FS["note"])
    ax2.set_ylim(-0.6, len(rows) - 0.4)
    ax2.set_xlim(0, GRID[-1] * 1.06)
    ax2.set_xlabel("epoch", color=K.INK2, fontsize=K.FS["label"])
    ax2.set_title(f"where σ_c^{sigma.upper()} selects on the full-support trajectory "
                  "(filled = M_on, hollow = M_off; bar = median)",
                  color=K.INK2, fontsize=K.FS["note"], loc="left", pad=3)
    fig.tight_layout()
    K.save(fig, name, outs)


def main() -> int:
    K.rc()
    s = K.summary(K.X25_SUMMARY)
    se = pd.read_csv(f"{K.ROOT}/harness/results/x25/x25_selected_epochs.csv")
    ck, outs = K.Checks(), []
    # the endpoint of the grid is the full-support selector value already frozen
    for D in ("D0", "D1"):
        ck.true(f"{D}: grid epochs present",
                all(len(s[s.quantity == f"{D}.fx{e}.ndp"]) == 1 for e in GRID))
    ck.true("selected-epoch rows = 2 configs x 2 caps x 30 cells",
            len(se) == 2 * 2 * 30 * 2)   # two selectors as well

    src = K.Source("figD_nifty_trajectory")
    figure_d(s, se, src, ck, outs, sigma="bce")
    figure_d(s, se, src, ck, outs, sigma="auc",
             name="figD_supp_nifty_trajectory_auc_selector")
    p = src.write()
    ok = ck.report()
    for o in outs + [p]:
        print("[written]", o)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
