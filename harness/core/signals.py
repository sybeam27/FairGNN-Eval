"""
Allocation signals: every candidate criterion for ranking nodes, behind one
interface.

    signal(graph, state=None) -> np.ndarray of shape (n_nodes,)

Higher score = this node should receive stronger fairness pressure. That is the
*whole* contract. A score is not a probability that the node is treated
unfairly, and not an estimate of node-level fairness risk (see the WSDM draft,
Sec. 3): it only says how a fixed budget of fairness pressure is distributed.

Why this file is the spine of the study
---------------------------------------
The central experiment (E2) holds the entire pipeline fixed and swaps only the
ranking signal, so that "does allocation help?" and "does *this* allocation rule
help?" can be answered separately. That is only possible if uniform weighting,
boundary exposure, degree, per-node loss, BIND's influence score and FairGB's
group weight all expose the same signature. In utils/model_fairgate.py they are
fused into a 345-line compute_fiw_weights() that also does gating, ranking,
uncertainty modulation and regime selection at once, which makes the swap
impossible. Here they are separate: signals rank, allocate.py converts a
ranking into weights, and nothing else knows about regimes.

Model-dependent signals take a `state` (anything with `.probs` and optionally
`.sigma`), so a signal that needs a partially trained model can be recomputed
mid-training without the signal registry knowing about torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

EPS = 1e-8
REGISTRY: dict[str, "SignalSpec"] = {}


@dataclass
class ModelState:
    """The minimum a model must expose for a model-dependent signal."""
    probs: np.ndarray                      # (n,) predicted P(y=1)
    sigma: Optional[np.ndarray] = None     # (n,) learned interval width, if any
    epoch: int = 0


@dataclass
class SignalSpec:
    name: str
    fn: Callable
    needs_model: bool
    doc: str

    def __call__(self, graph, state: Optional[ModelState] = None, **kw) -> np.ndarray:
        if self.needs_model and state is None:
            raise ValueError(f"signal {self.name!r} needs a ModelState")
        s = np.asarray(self.fn(graph, state, **kw), dtype=np.float64)
        if s.shape != (graph.n_nodes,):
            raise ValueError(f"signal {self.name!r} returned {s.shape}, "
                             f"expected {(graph.n_nodes,)}")
        if not np.all(np.isfinite(s)):
            raise ValueError(f"signal {self.name!r} produced non-finite values")
        return s


def register(name: str, needs_model: bool = False):
    def deco(fn):
        REGISTRY[name] = SignalSpec(name, fn, needs_model, (fn.__doc__ or "").strip())
        return fn
    return deco


def get(name: str) -> SignalSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown signal {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def available(needs_model: Optional[bool] = None) -> list[str]:
    return sorted(n for n, s in REGISTRY.items()
                  if needs_model is None or s.needs_model == needs_model)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo + EPS)


def _deg_and_cross(graph):
    return graph.stats["deg"], graph.stats["r_cross"]


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
@register("uniform")
def _uniform(graph, state=None):
    """Every node identical. Recovers the conventional node-agnostic objective."""
    return np.ones(graph.n_nodes)


@register("random")
def _random(graph, state=None, seed: int = 0):
    """Random ranking. Isolates 'concentrating pressure on *some* subset'."""
    return np.random.default_rng(seed).random(graph.n_nodes)


# ---------------------------------------------------------------------------
# Structural signals (Eq. 5 of the draft; formulas match
# utils/model_fairgate.py::_compute_structural_signals)
# ---------------------------------------------------------------------------
@register("w_bdry")
def _w_bdry(graph, state=None):
    """Cross-group exposure, Norm(log(1 + 10 r_v^x)). Informative when such
    exposure is sparse; near-constant when r_v^x concentrates (h ~ 0.5)."""
    _, r_cross = _deg_and_cross(graph)
    return minmax(np.log1p(10.0 * r_cross))


@register("w_deg")
def _w_deg(graph, state=None):
    """Degree, Norm(log(1 + d_v)). Informative when the two groups differ in
    mean degree, since degree governs how strongly aggregation shapes h_v."""
    deg, _ = _deg_and_cross(graph)
    return minmax(np.log1p(deg))


@register("w_lhd")
def _w_lhd(graph, state=None):
    """|local homophily - graph mean|. The signal implied by the local-homophily
    line of work (Loveland et al.) and by ComFairGNN's coreset selection."""
    deg, r_cross = _deg_and_cross(graph)
    local_h = 1.0 - r_cross
    return minmax(np.abs(local_h - local_h.mean()))


# ---------------------------------------------------------------------------
# Operator-matched structural signals
#
# `w_bdry`, `w_deg` and `w_lhd` above are one-hop and count every neighbour
# equally. A two-layer GCN forms A_hat^2 X with A_hat = D^-1/2 (A + I) D^-1/2,
# so it mixes over two hops and weights a neighbour by 1 / sqrt(d_u d_v).
# `harness/e0_diagnostics/signal_scale_check.py` measures the consequence without
# reference to any outcome: at the backbone's depth the node set the gate would
# select overlaps the `w_bdry` one by a median Jaccard of 0.67, and by only
# 0.22-0.30 on the two settings with h ~ 0.48, where the two are close to
# unrelated. The degree weighting alone changes almost nothing (median 0.86);
# it is the hop count.
#
# These three follow one principle -- measure the signal on the operator the
# model uses -- and are pre-registered in harness/PREREGISTRATION.md (deviation
# D4) before their benefit was measured. They are structural: no training, no
# `state`, identical across seeds.
# ---------------------------------------------------------------------------
def _propagator(graph, hops: int) -> tuple[np.ndarray, np.ndarray]:
    """(A_hat^k e, A_hat^k 1) with e = 1[s = 1]. Two sparse matvecs per hop; the
    dense A_hat^k is never formed. Cached on the graph, keyed by hop count."""
    key = f"_prop{hops}"
    if key in graph.stats:
        return graph.stats[key]
    n = graph.n_nodes
    r, c = graph.edge_index
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    dinv = 1.0 / np.sqrt(np.maximum(np.asarray(A.sum(1)).ravel(), 1e-12))
    Ah = (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr()
    pe, p1 = (graph.sens == 1).astype(float), np.ones(n)
    for _ in range(hops):
        pe, p1 = Ah @ pe, Ah @ p1
    graph.stats[key] = (pe, p1)
    return pe, p1


@register("prop_cross")
def _prop_cross(graph, state=None, hops: int = 2):
    """S1. Share of the mass a 2-layer GCN aggregates that comes from the
    opposite sensitive group. The depth-corrected `w_bdry`: same quantity,
    measured on A_hat^2 rather than on a one-hop neighbour count."""
    pe, p1 = _propagator(graph, hops)
    share1 = pe / np.maximum(p1, 1e-12)
    cross = np.where(graph.sens == 0, share1, 1.0 - share1)
    return minmax(np.log1p(10.0 * cross))      # same transform as w_bdry


@register("prop_dev")
def _prop_dev(graph, state=None, hops: int = 2):
    """S3. |receptive-field group composition - the graph-wide rate|. What
    `w_lhd` was reaching for, at the depth and weighting the model uses. Zero
    for a node whose two-hop neighbourhood mirrors the graph; large for one
    drawn disproportionately from either group."""
    pe, p1 = _propagator(graph, hops)
    share1 = pe / np.maximum(p1, 1e-12)
    return minmax(np.abs(share1 - float((graph.sens == 1).mean())))


@register("prop_sens")
def _prop_sens(graph, state=None, hops: int = 2):
    """S2. How much of what the node aggregates would disappear if cross-group
    edges were cut: ||(A_hat^h - A_tilde^h)_v X|| with A_tilde the same operator
    on the intra-group subgraph. Unlike S1 and S3 this weights by the features
    actually being mixed, so a node exposed to a group whose features happen to
    coincide scores low."""
    key = f"_propsens{hops}"
    if key in graph.stats:
        return graph.stats[key]
    n = graph.n_nodes
    r, c = graph.edge_index
    keep = graph.sens[r] == graph.sens[c]

    def op(rr, cc):
        A = sp.coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(n, n)).tocsr()
        dinv = 1.0 / np.sqrt(np.maximum(np.asarray(A.sum(1)).ravel(), 1e-12))
        return (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr()

    Ah, At = op(r, c), op(r[keep], c[keep])
    X = graph.x.astype(np.float64)
    F, G = X, X
    for _ in range(hops):
        F, G = Ah @ F, At @ G
    out = minmax(np.linalg.norm(F - G, axis=1))
    graph.stats[key] = out
    return out


@register("w_bdry_deg_var")
def _w_mix_var(graph, state=None):
    """Variance-weighted mix of w_bdry and w_deg (Eq. 8, 'otherwise' branch)."""
    b, d = _w_bdry(graph), _w_deg(graph)
    vb, vd = float(np.var(b)), float(np.var(d))
    a = vb / (vb + vd + EPS)
    return a * b + (1 - a) * d


@register("w_bdry_deg_mi")
def _w_mix_mi(graph, state=None):
    """Mutual-information-weighted mix (Eq. 8, 'clustered' branch)."""
    b, d = _w_bdry(graph), _w_deg(graph)
    ib, idg = mutual_info(b, graph.sens), mutual_info(d, graph.sens)
    a = ib / (ib + idg + EPS)
    return a * b + (1 - a) * d


# ---------------------------------------------------------------------------
# Model-dependent signals
# ---------------------------------------------------------------------------
@register("loss", needs_model=True)
def _loss(graph, state, **kw):
    """Norm(BCE(y_v, p_v) * (0.5 + r_v^x)) -- Eq. 6. Per-node error weighted by
    cross-group exposure. Unavailable during warm-up."""
    _, r_cross = _deg_and_cross(graph)
    p = np.clip(state.probs, 1e-7, 1 - 1e-7)
    bce = -(graph.y * np.log(p) + (1 - graph.y) * np.log(1 - p))
    return minmax(bce * (0.5 + r_cross))


@register("entropy", needs_model=True)
def _entropy(graph, state, **kw):
    """Predictive entropy. The cheap substitute the draft compares sigma against."""
    p = np.clip(state.probs, 1e-7, 1 - 1e-7)
    return minmax(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


@register("sigma", needs_model=True)
def _sigma(graph, state, **kw):
    """Learned interval width from the auxiliary uncertainty head."""
    if state.sigma is None:
        raise ValueError("ModelState.sigma is None; train the uncertainty head")
    return minmax(state.sigma)


# ---------------------------------------------------------------------------
# Signals that reproduce a *published* allocation rule.
# Each is the criterion that method fixes a priori; E2 asks whether any fixed
# criterion wins outside the regime it suits.
# ---------------------------------------------------------------------------
@register("fairgb_group", needs_model=True)
def _fairgb_group(graph, state, **kw):
    """FairGB (KDD'24) Contribution Alignment Loss: weight per (y, s) subgroup,
    proportional to the inverse of that subgroup's gradient contribution.

    Constant within a subgroup by construction, so it carries no topological
    information -- which is the distinction this study is built around. The
    gradient norm of a BCE head w.r.t. the logit is |p - y|, so the per-node
    contribution is available without touching FairGB's code.
    """
    p = np.clip(state.probs, 1e-7, 1 - 1e-7)
    contrib = np.abs(p - graph.y)
    out = np.zeros(graph.n_nodes)
    for yv in (0, 1):
        for sv in (0, 1):
            m = (graph.y == yv) & (graph.sens == sv)
            if m.sum() == 0:
                continue
            out[m] = 1.0 / (contrib[m].mean() + EPS)
    return minmax(out)


@register("bind_influence", needs_model=True)
def _bind_influence(graph, state, **kw):
    """BIND (AAAI'23): per-training-node influence on model bias.

    BIND estimates the influence of removing node v on a distributional bias
    measure and then *deletes* the harmful nodes -- a hard 0/1 allocation. This
    is a first-order stand-in for that score: how much node v's predicted score
    pulls its own sensitive group's mean away from the other group's, which is
    the leading term of the change in the DP gap when v is removed.

    Replace with the exact estimator via adapters/bind.py when comparing
    head-to-head with the published method; this keeps E2 runnable without
    BIND's retraining loop.
    """
    p = state.probs
    mu = {s: p[graph.sens == s].mean() if np.any(graph.sens == s) else 0.0 for s in (0, 1)}
    direction = np.where(graph.sens == 1, 1.0, -1.0) * (mu[1] - mu[0])
    return minmax(direction * (p - np.where(graph.sens == 1, mu[1], mu[0])))


# ---------------------------------------------------------------------------
# Diagnostics: is a signal usable on this graph at all?
# ---------------------------------------------------------------------------
def mutual_info(x: np.ndarray, s: np.ndarray, n_bins: int = 10) -> float:
    """MI between a (continuous) signal and the binary sensitive attribute."""
    if np.allclose(x, x[0]):
        return 0.0
    edges = np.linspace(x.min(), x.max() + EPS, n_bins + 1)
    b = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    mi, n = 0.0, len(x)
    for bi in range(n_bins):
        pb = np.mean(b == bi)
        if pb == 0:
            continue
        for sv in (0, 1):
            ps = np.mean(s == sv)
            pj = np.mean((b == bi) & (s == sv))
            if pj > 0 and ps > 0:
                mi += pj * np.log(pj / (pb * ps))
    return float(mi)


def mutual_info_debiased(x: np.ndarray, s: np.ndarray, n_bins: int = 10,
                         n_perm: int = 500, seed: int = 0
                         ) -> tuple[float, float, float, float]:
    """Permutation-debiased MI, the bias removed, the null spread, and the
    one-sided permutation p-value.

    Plug-in MI is biased upward and the bias grows as the sample shrinks and as
    the signal takes more distinct values. Measured on `idx_fair`, MI between a
    *random* signal and the sensitive attribute runs 0.0008 on Income
    (|I| = 4605) and 0.1931 on NBA (|I| = 50) -- a 240-fold artefact that has
    nothing to do with the signal. Left in, it would inflate D on exactly the
    small-|I| settings and by an amount that differs per signal, which would
    corrupt both the across-setting association and the within-setting argmax.

    Shuffling `s` destroys any real dependence while preserving both marginals
    and the sample size, so the mean MI under permutation estimates the bias
    directly. Subtracting it leaves a quantity that is ~0 *in expectation* for a
    signal unrelated to `s` at any |I|, and it may be negative, which is expected
    and is not clipped.

    Only the mean of the bias is removed, not its spread: a single draw still
    fluctuates by the null, which is why the spread and the p-value are reported
    alongside every D. The null is right-skewed, so `p` -- the share of
    permutations reaching the observed MI -- is the usable statement of "this D
    is distinguishable from a signal unrelated to s", and a multiple of the sd
    is not. On |I| = 50 (NBA) the null sd is large enough that a
    random signal can land near 0.09, so D there is imprecise -- a consequence of
    the benchmark giving that setting only fifty sensitive labels, not of the
    estimator. Settings with small |I| are imprecise in the benefit measurement
    for the same reason, and the two must be read together.
    """
    obs = mutual_info(x, s, n_bins)
    rng = np.random.default_rng(seed)
    sp = np.asarray(s).copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(sp)
        null[i] = mutual_info(x, sp, n_bins)
    bias = float(null.mean())
    p = float((null >= obs).sum() + 1) / (n_perm + 1)      # never exactly zero
    return float(obs - bias), bias, float(null.std(ddof=1)), p


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative signal. 0 = flat, 1 = all mass on one
    node. Reported as a secondary dispersion statistic."""
    v = np.sort(np.asarray(x, dtype=float) - min(0.0, float(np.min(x))))
    n = len(v)
    tot = v.sum()
    if n == 0 or tot <= 0:
        return 0.0
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(v) / (n * tot))


def describe(signal_name: str, graph, state=None, q_gate: float = 0.7,
             idx: Optional[np.ndarray] = None) -> dict:
    """Dispersion of a signal on a graph -- the necessary condition for the
    signal to rank anything. Zero variance => no ranking information, whatever
    the graph-level statistics say.

    Note this is a *necessary* condition only: high dispersion does not imply
    that allocating by this signal reduces disparity. That is what E2 measures.

    `idx` restricts every statistic to a node subset. Pass `graph.idx_fair` to
    measure on the set the allocation actually acts on, which is what
    `harness/PREREGISTRATION.md` fixes as the primary statistic; the default of
    None keeps the whole-graph behaviour the e0 diagnostics were computed with.
    """
    full = get(signal_name)(graph, state)
    sub = np.asarray(idx) if idx is not None else np.arange(graph.n_nodes)
    s, sens = full[sub], graph.sens[sub]

    n = len(s)
    k = max(1, int(round((1 - q_gate) * n)))
    order = np.argsort(-s, kind="stable")
    thresh = s[order[k - 1]]
    tied = int(np.sum(s == thresh))
    above = int(np.sum(s > thresh))

    in_gate = np.zeros(n, dtype=int)
    in_gate[order[:k]] = 1

    _d_mi, _d_bias, _d_sd, _d_p = mutual_info_debiased(s, sens)

    return dict(
        signal=signal_name, setting=graph.name, regime=graph.regime_paper,
        n_eval=n, on="idx_fair" if idx is not None else "all",
        var=float(np.var(s)), std=float(np.std(s)),
        n_unique=int(len(np.unique(s))),
        uniq_ratio=float(len(np.unique(s)) / n),
        gini=gini(s),
        mi_raw=mutual_info(s, sens),                # plug-in, biased at small |I|
        mi_with_sens=_d_mi,                         # <- the primary statistic D
        mi_bias=_d_bias, mi_null_sd=_d_sd, mi_p_perm=_d_p,   # precision of D
        mi_gate_with_sens=mutual_info_debiased(in_gate.astype(float), sens)[0],
        gate_k=k,
        tie_frac=tied / n,
        arb_frac=max(0, k - above) / k,   # share of G decided by tie-breaking
    )
