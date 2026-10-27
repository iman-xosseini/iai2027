"""Representation diagnostics: latent statistics, effective rank, collapse checks, gradient norms."""
import math

import torch


def effective_rank(z):
    """exp(entropy of the normalized covariance eigen-spectrum) of z [N, D] (as in the CIRP study).

    ~1 = all variance along one direction, ~D = variance spread over all directions. It is scale-invariant,
    so a latent collapsed to a POINT is not detected by it: check the per-dimension spread as well.
    """
    eigvals = torch.linalg.eigvalsh(torch.cov(z.T.double())).clamp(min=0)
    p = eigvals / eigvals.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


@torch.no_grad()
def latent_statistics(z):
    """Summary statistics of embeddings z [N, D]."""
    z = z.double()
    mean, std = z.mean(0), z.std(0)
    cov = torch.cov(z.T)
    corr = cov / torch.outer(std, std)
    off_diag = corr[~torch.eye(len(corr), dtype=torch.bool)]
    return {
        "latent_mean_abs": float(mean.abs().mean()),        # average |mean| per dimension (target 0)
        "latent_std_min": float(std.min()),                 # target 1 for every dimension
        "latent_std_median": float(std.median()),
        "latent_std_max": float(std.max()),
        "effective_rank": effective_rank(z),                # target D
        "max_abs_correlation": float(off_diag.abs().max()), # target 0
        "per_dim_variance": std.square().tolist(),
        "covariance": cov.tolist(),
    }


def noncollapse_checks(stats, min_effective_rank, min_latent_std):
    """The predefined non-collapse criteria used for checkpoint selection."""
    return {
        "effective_rank_ok": stats["effective_rank"] >= min_effective_rank,
        "latent_std_ok": stats["latent_std_min"] >= min_latent_std,
    }


def grad_norm(module):
    """L2 norm of all parameter gradients of a module; NaN if no parameter received a gradient."""
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return math.nan
    return float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])))
