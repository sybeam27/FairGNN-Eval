"""Figure C -- selection-support replication across two methods.

X25 (NIFTY / German, native configuration D1) and X26 (FairGB / Bail) ask the
same question of different methods: when the training horizon is extended, is
the attribution shift carried by the longer trajectory itself, or by the
selector gaining access to later checkpoints?

Three points per panel, in the order they are defined:

    tau at the frozen short horizon        h200
    tau on the long trajectory, capped     cap  (<= 200 / <= 199)
    tau on the long trajectory, full       full

    T = cap - h200      the prefix/run term
    S = full - cap      the selection-support term

All numbers come from the frozen summaries; the identities S = full - cap and
H_shift = S + T are re-checked on the plotted values.

    python harness/experiments/x28_figures/fig_c_selection_support.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K                                                   # noqa: E402

# (label, summary path, quantity prefix, cap label, full label, colour key)
CASES = [
    dict(key="x25", title="NIFTY / German  (X25)", path=K.X25_SUMMARY,
         q=dict(h200="D1.bce.h200.ndp", cap="D1.bce.c200.ndp",
                full="D1.bce.c1000.ndp", S="D1.bce.S.ndp", T="D1.bce.T.ndp",
                H="D1.bce.H.ndp"),
         ticks=("frozen\nH = 200", "long run\ncap ≤ 200", "long run\nfull ≤ 1000"),
         colour="NIFTY"),
    dict(key="x26", title="FairGB / Bail  (X26)", path=K.X26_SUMMARY,
         q=dict(h200="bce.h200.ndp", cap="bce.cap.ndp", full="bce.full.ndp",
                S="bce.S.ndp", T="bce.T.ndp", H="bce.Hshift.ndp"),
         ticks=("frozen\nH = 200", "long run\ncap ≤ 199", "long run\nfull ≤ 1499"),
         colour="FairGB"),
]


def load(case):
    s = K.summary(case["path"])
    return {k: K.q(s, v) for k, v in case["q"].items()}


def figure_c(vals, src, ck, outs, shared=True):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    allv = []
    for case in CASES:
        v = vals[case["key"]]
        for k in ("h200", "cap", "full"):
            allv += [v[k]["mean"], v[k]["lo"], v[k]["hi"]]
    lo, hi = float(np.min(allv)), float(np.max(allv))
    pad = (hi - lo) * 0.30
    ylim = (lo - pad * 0.45, hi + pad)

    for ax, case in zip(axes, CASES):
        v = vals[case["key"]]
        col = K.C[case["colour"]]
        K.style(ax, grid="y")
        ax.axhline(0, color=K.BASEC, linewidth=0.9, zorder=1)
        xs = [0, 1, 2]
        means = [v[k]["mean"] for k in ("h200", "cap", "full")]
        ax.plot(xs, means, color=col, linewidth=1.5, zorder=3)
        for x, k, mk in zip(xs, ("h200", "cap", "full"), ("o", "^", "s")):
            r = v[k]
            ax.plot([x, x], [r["lo"], r["hi"]], color=col, linewidth=1.1,
                    alpha=0.65, zorder=2)
            K.resolved_marker(ax, x, r["mean"], col, mk, bool(r["resolved"]),
                              size=7.0)
            src.add(figure="C", case=case["key"], point=k, x=x, mean=r["mean"],
                    lo=r["lo"], hi=r["hi"], resolved=bool(r["resolved"]))
        if shared:
            ax.set_ylim(*ylim)
        # T and S as brackets between consecutive points
        # the two brackets are staggered in height: anchored to the same point
        # they would sit at one level and their labels would run together
        for bi, (x0, x1, name) in enumerate(((0, 1, "T"), (1, 2, "S"))):
            r = v[name]
            y = (max(means[x0], means[x1])
                 + (ylim[1] - ylim[0]) * (0.055 + 0.085 * bi))
            ax.plot([x0, x0, x1, x1],
                    [y - (ylim[1] - ylim[0]) * 0.018, y, y,
                     y - (ylim[1] - ylim[0]) * 0.018],
                    color=K.INK2, linewidth=0.8, zorder=4)
            ax.annotate(f"{name} {r['mean']:+.3f}"
                        f"{'' if r['resolved'] else ' (unresolved)'}",
                        ((x0 + x1) / 2, y), textcoords="offset points",
                        xytext=(0, 3), ha="center", va="bottom",
                        fontsize=K.FS["note"], color=K.INK2, zorder=5)
            src.add(figure="C", case=case["key"], point=name, x=(x0 + x1) / 2,
                    mean=r["mean"], lo=r["lo"], hi=r["hi"],
                    resolved=bool(r["resolved"]))
        ax.set_xticks(xs)
        ax.set_xticklabels(case["ticks"], fontsize=K.FS["label"] - 0.5)
        ax.set_xlim(-0.45, 2.45)
        ax.set_title(case["title"], color=K.INK, fontsize=K.FS["panel"],
                     loc="left", pad=4)
        ck.limits(ax, xs, [v[k][b] for k in ("h200", "cap", "full")
                           for b in ("lo", "hi")], f"Figure C / {case['key']}")
        if case is CASES[0]:
            ax.set_ylabel("intervention effect on −ΔDP", color=K.INK2,
                          fontsize=K.FS["label"])
    fig.suptitle("Extending the horizon: the selector's checkpoint support is what moves",
                 color=K.INK, fontsize=K.FS["title"], x=0.008, ha="left")
    fig.text(0.008, 0.885, "σ_c^BCE · bars are 95% paired hierarchical bootstrap "
             "intervals · filled = resolved · shared y-axis",
             fontsize=K.FS["note"], color=K.MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.875))
    K.save(fig, "figC_selection_support", outs)


def figure_c2(vals, src, ck, outs):
    """C2: the same two cases as three forest estimates each (H_shift, S, T)."""
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    K.style(ax, grid="x")
    rows, ticks = [], []
    for case in CASES:
        for name, lab in (("H", "H_shift"), ("S", "S  (selection support)"),
                          ("T", "T  (prefix / run)")):
            rows.append((case, name, lab))
    y = np.arange(len(rows))[::-1]
    xs = []
    for (case, name, lab), yy in zip(rows, y):
        r = vals[case["key"]][name]
        col = K.C[case["colour"]]
        ax.plot([r["lo"], r["hi"]], [yy, yy], color=col, linewidth=1.3,
                alpha=0.75, zorder=2, solid_capstyle="butt")
        K.resolved_marker(ax, r["mean"], yy, col, K.MK[case["colour"]],
                          bool(r["resolved"]), size=6.5)
        ticks.append(f"{case['key'].upper()}  {lab}")
        xs += [r["lo"], r["hi"]]
        src.add(figure="C2", case=case["key"], quantity=name, mean=r["mean"],
                lo=r["lo"], hi=r["hi"], resolved=bool(r["resolved"]))
    ax.axvline(0, color=K.BASEC, linewidth=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ticks, fontsize=K.FS["note"] + 0.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    lo, hi = float(np.min(xs)), float(np.max(xs))
    m = (hi - lo) * 0.10
    ax.set_xlim(lo - m, hi + m)
    ck.limits(ax, xs, y, "Figure C2")
    ax.set_xlabel("effect on −ΔDP", color=K.INK2, fontsize=K.FS["label"])
    ax.set_title("Decomposition of the horizon shift", color=K.INK,
                 fontsize=K.FS["title"], loc="left", pad=14)
    ax.text(0, 1.02, "filled = resolved under the frozen rule", transform=ax.transAxes,
            fontsize=K.FS["note"], color=K.MUTED, va="bottom")
    fig.tight_layout()
    K.save(fig, "figC2_selection_support_forest", outs)


def main() -> int:
    K.rc()
    vals = {c["key"]: load(c) for c in CASES}
    ck, outs = K.Checks(), []
    for c in CASES:
        v = vals[c["key"]]
        ck.close(f"{c['key']}: S = full - cap",
                 v["full"]["mean"] - v["cap"]["mean"], v["S"]["mean"], 1e-9)
        ck.close(f"{c['key']}: T = cap - h200",
                 v["cap"]["mean"] - v["h200"]["mean"], v["T"]["mean"], 1e-9)
        ck.close(f"{c['key']}: H_shift = S + T",
                 v["S"]["mean"] + v["T"]["mean"], v["H"]["mean"], 1e-9)
    src = K.Source("figC_selection_support")
    figure_c(vals, src, ck, outs)
    figure_c2(vals, src, ck, outs)
    p = src.write()
    ok = ck.report()
    for o in outs + [p]:
        print("[written]", o)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
