"""
The three node-weighted fairness objectives.

No novelty is claimed for any of them; each is adapted from existing work. They
are used because each one decomposes over nodes and therefore accepts the same
phi(v), which is what makes "uniform" and "allocated" comparable under an
otherwise identical pipeline. The contribution under test is the allocation,
not the regularizers it is applied to.

Every loss is computed over the fairness node set I (graph.idx_fair) and takes
phi already restricted and mean-normalised on I, so that sum(phi) == |I| and
the total fairness budget is identical across arms.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


# ---------------------------------------------------------------------------
# structure level
# ---------------------------------------------------------------------------
def structural_consistency(h: torch.Tensor, h_pert: torch.Tensor,
                           idx: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """phi-weighted agreement between the real and the perturbed graph.

    L = 1/|I| sum_I phi(v) * ||h_v - h~_v||^2 / d
    """
    d = h.size(1)
    diff = (h[idx] - h_pert[idx]).pow(2).sum(dim=1) / d
    return (phi * diff).mean()


# ---------------------------------------------------------------------------
# representation level
# ---------------------------------------------------------------------------
def _weighted_mean_var(h: torch.Tensor, w: torch.Tensor):
    wsum = w.sum() + EPS
    mu = (w.unsqueeze(1) * h).sum(0) / wsum
    var = (w * (h - mu).pow(2).sum(1)).sum() / wsum
    return mu, var


def _mmd(a: torch.Tensor, b: torch.Tensor, max_n: int = 512) -> torch.Tensor:
    """Multi-bandwidth RBF MMD^2, subsampled so it stays affordable on Pokec."""
    if a.size(0) > max_n:
        a = a[torch.randperm(a.size(0))[:max_n]]
    if b.size(0) > max_n:
        b = b[torch.randperm(b.size(0))[:max_n]]
    if a.size(0) < 2 or b.size(0) < 2:
        return a.new_zeros(())

    z = torch.cat([a, b], 0)
    d2 = torch.cdist(z, z).pow(2)
    med = d2.detach().flatten().median().clamp(min=EPS)     # median heuristic

    k = sum(torch.exp(-d2 / (2.0 * med * s)) for s in (0.25, 0.5, 1.0, 2.0, 4.0))
    na = a.size(0)
    return k[:na, :na].mean() + k[na:, na:].mean() - 2 * k[:na, na:].mean()


def representation_alignment(h: torch.Tensor, sens: torch.Tensor,
                             idx: torch.Tensor, phi: torch.Tensor,
                             mmd_alpha: float = 0.3) -> torch.Tensor:
    """phi-weighted moment matching blended with a multi-bandwidth RBF MMD."""
    hi, si = h[idx], sens[idx]
    m0, m1 = si == 0, si == 1
    if m0.sum() < 2 or m1.sum() < 2:
        return h.new_zeros(())

    mu0, v0 = _weighted_mean_var(hi[m0], phi[m0])
    mu1, v1 = _weighted_mean_var(hi[m1], phi[m1])
    moment = (mu0 - mu1).pow(2).sum() + (v0 - v1).abs()

    mmd = _mmd(hi[m0], hi[m1]) if mmd_alpha > 0 else h.new_zeros(())
    return (1 - mmd_alpha) * moment + mmd_alpha * mmd


# ---------------------------------------------------------------------------
# prediction level
# ---------------------------------------------------------------------------
def _wmean(p: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (w * p).sum() / (w.sum() + EPS)


def prediction_alignment(prob: torch.Tensor, y: torch.Tensor, sens: torch.Tensor,
                         idx: torch.Tensor, phi: torch.Tensor,
                         rho0: float = 0.3):
    """phi-weighted DP and EO gaps on *soft* scores, adaptively mixed.

    These are differentiable relatives of the reported metrics, not the metrics
    themselves: they act on probabilities rather than thresholded predictions,
    they carry the node weights, and the conditional term aligns both label
    classes rather than only the positive one. Optimising them therefore does
    not directly minimise the reported DP/EO.
    """
    p, yy, ss = prob[idx], y[idx], sens[idx]
    m0, m1 = ss == 0, ss == 1
    if m0.sum() == 0 or m1.sum() == 0:
        return prob.new_zeros(()), prob.new_zeros(()), prob.new_zeros(())

    d_marg = (_wmean(p[m0], phi[m0]) - _wmean(p[m1], phi[m1])).abs()

    d_cond, n_cls = prob.new_zeros(()), 0
    for cls in (0, 1):
        a, b = m0 & (yy == cls), m1 & (yy == cls)
        if a.sum() > 0 and b.sum() > 0:
            d_cond = d_cond + (_wmean(p[a], phi[a]) - _wmean(p[b], phi[b])).abs()
            n_cls += 1
    d_cond = d_cond / max(n_cls, 1)

    tot = d_marg.detach() + d_cond.detach() + EPS
    w_dp = 0.5 * rho0 + 0.5 * d_marg.detach() / tot
    w_eo = 0.5 * (1 - rho0) + 0.5 * d_cond.detach() / tot
    s = w_dp + w_eo
    return (w_dp / s) * d_marg + (w_eo / s) * d_cond, d_marg, d_cond


# ---------------------------------------------------------------------------
# uncertainty head (only needed by the 'sigma' signal)
# ---------------------------------------------------------------------------
def interval_loss(logit: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                  idx: torch.Tensor, width_penalty: float = 0.05) -> torch.Tensor:
    """Coverage-and-width objective for the auxiliary interval head."""
    lo = torch.sigmoid(logit[idx] - sigma[idx])
    hi = torch.sigmoid(logit[idx] + sigma[idx])
    yy = y[idx]
    cover = F.relu(lo - yy) + F.relu(yy - hi)
    return cover.mean() + width_penalty * (hi - lo).mean()
