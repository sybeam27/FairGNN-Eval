"""
Turning a ranking into node-wise intervention weights phi(v).

This is the second half of what utils/model_fairgate.py::compute_fiw_weights()
does in one 345-line function. Splitting it matters: allocate() must depend
only on (score, hyper-parameters) and know nothing about regimes, graphs or
models, otherwise the E2 signal swap silently changes two things at once.

    phi(v) = phi_min + (phi_max - phi_min) * Norm_G[s(v) * (1 + sigma(v))]   v in G
    phi(v) = phi_min                                                          v not in G

Three properties are deliberate (WSDM draft, Sec. 4.3):

  * sigma enters *after* selection and only inside G, so uncertainty alone can
    never promote a node; it rescales an ordering that structure already fixed.
  * the modulation is multiplicative and monotone in s, so it reduces to the
    pure structural score as sigma -> 0. An additive term would let uncertainty
    override structural exposure.
  * phi_min > 0, so nodes outside G keep fairness pressure. When no signal is
    discriminative the mechanism degrades toward uniform weighting -- the
    node-agnostic baseline -- rather than toward a harmful allocation.

Gating uses an exact top-k rather than a quantile threshold. With a quantile,
every node tied at the cut-off is admitted, which on a graph with many ties
can gate far more than the intended fraction. Note the diagnostics found ties
to be *most* prevalent on non-saturated graphs (Pokec-z gender, Income), not
saturated ones as the draft asserts -- so this guard matters, but the stated
reason for it does not hold.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-8


def gate(score: np.ndarray, q_gate: float, rng: np.random.Generator | None = None
         ) -> np.ndarray:
    """Boolean mask of the top (1 - q_gate) fraction of nodes by `score`.

    Ties at the boundary are broken deterministically by index unless an rng is
    given, in which case they are broken at random. Pass an rng when the
    tie-breaking itself could bias the comparison (e.g. degree-ordered inputs).
    """
    if not 0.0 <= q_gate < 1.0:
        raise ValueError(f"q_gate must be in [0, 1), got {q_gate}")
    n = len(score)
    k = max(1, int(round((1.0 - q_gate) * n)))
    if k >= n:
        return np.ones(n, dtype=bool)

    if rng is None:
        order = np.argsort(-score, kind="stable")
    else:
        order = np.lexsort((rng.random(n), -score))

    m = np.zeros(n, dtype=bool)
    m[order[:k]] = True
    return m


def allocate(score: np.ndarray,
             q_gate: float = 0.7,
             phi_min: float = 0.5,
             phi_max: float = 2.0,
             modulation: np.ndarray | None = None,
             rng: np.random.Generator | None = None,
             normalize_mean: bool = True,
             degenerate_tol: float = 1e-12) -> np.ndarray:
    """score -> phi(v).

    modulation
        Optional non-negative per-node array (e.g. learned sigma). Applied as
        s * (1 + modulation), inside the gate only.
    normalize_mean
        If True, rescale so mean(phi) == 1 over all nodes. This is what keeps
        "uniform" and "allocated" comparable: the total fairness pressure is
        held fixed and only its distribution changes. Without it, a different
        signal would also change the effective lambda_fair, and E2 would be
        measuring two things at once.
    """
    score = np.asarray(score, dtype=np.float64)
    if phi_min <= 0:
        raise ValueError("phi_min must be > 0 so ungated nodes keep pressure")
    if phi_max < phi_min:
        raise ValueError("phi_max must be >= phi_min")

    n = len(score)
    g = gate(score, q_gate, rng=rng)

    s = score.copy()
    if modulation is not None:
        modulation = np.asarray(modulation, dtype=np.float64)
        if np.any(modulation < 0):
            raise ValueError("modulation must be non-negative")
        s = s * (1.0 + modulation)

    phi = np.full(n, phi_min, dtype=np.float64)
    if g.any():
        sg = s[g]
        lo, hi = float(sg.min()), float(sg.max())
        spread = hi - lo
        if spread > degenerate_tol * max(abs(hi), abs(lo), 1.0):
            normed = (sg - lo) / spread           # Norm_G: over the gate only
            phi[g] = phi_min + (phi_max - phi_min) * normed
        else:
            # The signal cannot separate the gated nodes at all. This is not an
            # edge case to paper over: it is the condition the whole study is
            # about, and the draft (Sec. 4.3) claims the mechanism "degrades
            # towards uniform weighting rather than towards a harmful
            # allocation". That property has to be implemented, not asserted --
            # leaving the gate at phi_max would hand an arbitrary top-k 4x the
            # pressure, i.e. exactly the `random` arm of E2.
            phi[:] = phi_min

    if normalize_mean:
        # phi >= phi_min > 0, so the mean is strictly positive and no epsilon
        # guard is needed. Adding one here would break the invariant that the
        # total fairness budget is held exactly fixed across arms, which is the
        # only reason uniform and allocated weighting are comparable at all.
        phi = phi / phi.mean()
    return phi


def allocate_on(score: np.ndarray, idx: np.ndarray, **kw) -> np.ndarray:
    """allocate() restricted to the fairness node set I.

    Every fairness term is computed over I, so gating and mean-normalisation
    must also be over I -- otherwise the effective budget on I depends on how
    many nodes outside I happened to be gated, and the arms of E2 are no longer
    comparable. Returns a full-length array; entries outside I are phi_min
    and are never read.
    """
    phi = np.full(len(score), kw.get("phi_min", 0.5), dtype=np.float64)
    phi[idx] = allocate(score[idx], **kw)
    return phi


def summarize(phi: np.ndarray, idx: np.ndarray | None = None) -> dict:
    """Sanity numbers for a logged run: did the allocation actually vary?"""
    p = phi if idx is None else phi[idx]
    return dict(
        mean=float(p.mean()), std=float(p.std()),
        min=float(p.min()), max=float(p.max()),
        frac_at_floor=float(np.mean(np.isclose(p, p.min()))),
        gini=_gini(p),
    )


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
