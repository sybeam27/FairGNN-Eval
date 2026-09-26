"""
E6 -- why did topology fail? Decompose the winning signal.

E4/E5 closed the search for a better allocation criterion: across 9 settings and
11 signals, nothing beats `bind_influence`, and the pre-registered stopping rule
of D4 fired. That is a negative result about topology, and the first question a
reader will ask is *why*. This asks it without reopening the search: no new
allocation signal is proposed here, and nothing in this file is used to choose
one. It only asks what the winner is made of.

`bind_influence` is a deterministic function of the model's predicted
probability and the sensitive attribute -- how far a node's score sits from its
own group's mean, signed so that pulling the groups apart scores high. So the
question "is influence topological?" is really "is the model's prediction
*deviation* topological?", and it has two possible answers, both decisive:

  * topology explains little  -> where fairness pressure should go is not a
    function of the graph, and the negative result has its mechanism;
  * topology explains a lot   -> the information was there and the failure is in
    how the structural signals transform it, which would be a surprising result
    and the only legitimate route back to a structural criterion, because the
    quantity it points at would come from a measurement rather than from a
    search over outcomes.

Two readouts, because R^2 is not what the method consumes:

  R^2      out-of-fold, ridge, over all nodes -- the statistical question.
  Jaccard  overlap between the gate `allocate()` would form from the true
           influence and from the topology-only prediction, on `idx_fair` at the
           operating `q_gate` -- the decision question. A block can carry
           middling R^2 and still pick the same nodes, or the reverse.

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python harness/experiments/e6_influence_decomposition.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import datasets as D            # noqa: E402
from core import signals as S             # noqa: E402
from core.trainer import Config, warmup_state   # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _ridge_svd(Z: np.ndarray, yc: np.ndarray, lams):
    """Ridge solutions for every lambda from one economy SVD.

    Not an optimisation. Forming the normal equations and calling `solve` is
    numerically hopeless here: the quadratic expansion has a condition number of
    ~1e17, and at the small end of the lambda grid the solve returns noise. The
    inner cross-validation then sometimes *selects* that noise, and the fold it
    is applied to reports an out-of-fold R^2 of -0.9 -- which is how Recidivism
    and NBA came to look like failures of topology rather than of arithmetic.
    Sweeping lambda directly on the same data gives +0.18 on Recidivism.

    With Z = U S V^T the ridge weights are V diag(S / (S^2 + lam)) U^T yc, which
    is stable for any lam and costs one decomposition for the whole grid.
    """
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    Uty = U.T @ yc
    return [Vt.T @ ((S / (S ** 2 + lam)) * Uty) for lam in lams]


def ridge_oof_r2(X: np.ndarray, y: np.ndarray, folds: int = 5,
                 lams=(1e-2, 1e0, 1e2, 1e4, 1e6, 1e8), seed: int = 0):
    """Out-of-fold R^2 of ridge, lambda picked inside each training fold.

    Out-of-fold rather than in-sample because the feature blocks differ in width
    by two orders of magnitude (12 topological columns against up to 277 feature
    columns) and an in-sample R^2 would reward the wide one for nothing.

    Returns Spearman as well as R^2, and Spearman is the one to read. The
    quadratic expansion is heavy-tailed -- a high-degree node contributes a
    `deg^2` sitting 2900 standard deviations out -- and a single such point in a
    test fold drives that fold's R^2 to -4.1 while the other four sit at +0.17.
    R^2 here measures how badly the tail is extrapolated, not how well the
    ranking is recovered, and the ranking is what `allocate()` consumes.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n) % folds
    pred = np.empty(n)
    for f in range(folds):
        tr, te = fold != f, fold == f
        Xtr, ytr = X[tr], y[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Z, ym = (Xtr - mu) / sd, ytr.mean()

        inner = rng.permutation(len(ytr)) % 3
        sse = np.zeros(len(lams))
        for gg in range(3):
            m_a, m_b = inner != gg, inner == gg
            ya = ytr[m_a]
            ws = _ridge_svd(Z[m_a], ya - ya.mean(), lams)
            for k, w in enumerate(ws):
                sse[k] += float(((Z[m_b] @ w + ya.mean() - ytr[m_b]) ** 2).sum())
        best = lams[int(np.argmin(sse))]

        w = _ridge_svd(Z, ytr - ym, [best])[0]
        pred[te] = ((X[te] - mu) / sd) @ w + ym
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rho = float(stats.spearmanr(pred, y).statistic)
    return r2, rho, pred


def topo_block(g) -> tuple[np.ndarray, list[str]]:
    """Everything structural the study has, plus plain degree quantities.
    No model, no labels, no features."""
    deg = g.stats["deg"].astype(float)
    rx = g.stats["r_cross"].astype(float)
    cols = {
        "deg": deg, "log_deg": np.log1p(deg),
        "r_cross": rx, "local_h": 1.0 - rx,
        "lhd": np.abs((1.0 - rx) - (1.0 - rx).mean()),
        "w_bdry": S.get("w_bdry")(g), "w_deg": S.get("w_deg")(g),
        "w_lhd": S.get("w_lhd")(g),
        "prop_cross": S.get("prop_cross")(g),
        "prop_dev": S.get("prop_dev")(g),
        "prop_sens": S.get("prop_sens")(g),
    }
    # two-hop mass, the denominator of the propagated signals
    _, p1 = S._propagator(g, 2)
    cols["mass2"] = p1
    return np.column_stack(list(cols.values())), list(cols)


def poly2(X: np.ndarray) -> np.ndarray:
    """X plus every square and pairwise product. Ridge is linear, so a low R^2
    from the raw structural columns could mean "not topological" or merely "not
    *linearly* topological". Twelve columns become ninety, which costs nothing
    and removes that reading: if topology still cannot recover the gate with
    every pairwise interaction available, the shortfall is information, not
    functional form."""
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / np.where(sd < 1e-12, 1.0, sd)   # standardise *before* the
    cols = [Z]                                      # products, or deg^2 arrives
    d = Z.shape[1]                                  # at 1e6 beside a 0-1 ratio
    for j in range(d):
        cols.append(Z[:, [j]] * Z[:, j:])
    return np.hstack(cols)


def gate(x: np.ndarray, q: float) -> set:
    k = max(1, int(round((1 - q) * len(x))))
    return set(np.argsort(-x, kind="stable")[:k].tolist())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[27, 28, 29])
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--q-gate", type=float, default=0.7)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=os.path.join(RESULTS, "e6_influence_decomposition.csv"))
    args = ap.parse_args()

    rows = []
    for name in D.ALL_SETTINGS:
        g = D.load(name)
        i = np.asarray(g.idx_fair)
        T, tnames = topo_block(g)
        X = g.x.astype(float)
        s = g.sens.astype(float)[:, None]
        T2 = poly2(T)
        blocks = {"SENS": s, "TOPO": T, "TOPO2": T2,
                  "TOPO2+SENS": np.hstack([T2, s]), "FEAT": X,
                  "TOPO2+FEAT+SENS": np.hstack([T2, X, s])}
        for seed in args.seeds:
            st = warmup_state(g, Config(seed=seed, warmup=args.warmup, device=args.device))
            y = S.get("bind_influence")(g, st)
            gt = gate(y[i], args.q_gate)
            for bname, B in blocks.items():
                r2, rho, pred = ridge_oof_r2(B, y, seed=seed)
                rows.append({"setting": name, "regime": g.regime_paper, "seed": seed,
                             "block": bname, "n_cols": B.shape[1],
                             "r2": r2, "spearman": rho,
                             "gate_jaccard": len(gt & gate(pred[i], args.q_gate))
                                             / max(len(gt | gate(pred[i], args.q_gate)), 1)})
            pd.DataFrame(rows).to_csv(args.out, index=False)
        sub = pd.DataFrame([r for r in rows if r["setting"] == name])
        m = sub.groupby("block")[["spearman", "gate_jaccard"]].mean()
        print(f"{name:<12} " + "   ".join(
            f"{b} rho={m.loc[b,'spearman']:.2f}/J={m.loc[b,'gate_jaccard']:.2f}"
            for b in ("SENS", "TOPO2+SENS", "FEAT") if b in m.index))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[saved] {args.out}  ({len(df)} rows)\n")
    report(df)
    return 0


def report(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    for metric, label in (("spearman", "out-of-fold Spearman(prediction, bind_influence)"),
                          ("r2", "out-of-fold R^2 -- secondary, tail-dominated"),
                          ("gate_jaccard", "overlap of the gate each block would form")):
        print("=" * 96)
        print(label)
        print("=" * 96)
        piv = df.pivot_table(index=["setting", "regime"], columns="block", values=metric)
        order = [c for c in ("SENS", "TOPO", "TOPO2", "TOPO2+SENS", "FEAT",
                             "TOPO2+FEAT+SENS") if c in piv.columns]
        print(piv[order].round(3).to_string())
        print(f"\nmean  " + "  ".join(f"{c}={piv[c].mean():.3f}" for c in order) + "\n")

    t = df[df.block == "TOPO2+SENS"].groupby("setting")["gate_jaccard"].mean()
    print("Reading: TOPO2+SENS is every structural quantity in the study, every "
          "square and\npairwise product of them, and the sensitive attribute, "
          "fit with full regression\nfreedom -- a far stronger use of topology "
          "than any single ranking signal can be.\nIf it cannot recover the "
          "gate, no structural signal could, and the negative result\nof E4/E5 "
          "has its explanation. If it can, the information is there and the "
          "failure\nis in the transform, which would be the one legitimate way "
          "back to a structural\ncriterion.")
    print(f"\nTOPO2+SENS gate overlap: mean {t.mean():.2f}, "
          f"min {t.min():.2f} ({t.idxmin()}), max {t.max():.2f} ({t.idxmax()})")
    r = df[df.block == "TOPO2+SENS"].groupby("setting")["spearman"].mean()
    print(f"TOPO2+SENS Spearman:      mean {r.mean():.2f}, "
          f"min {r.min():.2f}, max {r.max():.2f}")
    print("Chance overlap for two independent top-30% sets is about 0.18.")


if __name__ == "__main__":
    raise SystemExit(main())
