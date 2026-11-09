"""Response-permutation nulls for barrier quantities (Draft v0.2, Section 11.3): a statistic
of (z, y) divided by its median over permutations of y with the representation fixed."""
from __future__ import annotations

import numpy as np

from .lipschitz_barrier import barrier_curve, lambda_star


def null_ratio(stat, z, y, n_perm: int = 200, seed=0, rng=None, return_null: bool = False, **kw):
    rng = np.random.default_rng(seed) if rng is None else rng
    s_obs = stat(z, y, **kw)
    s_null = np.array([stat(z, rng.permutation(y), **kw) for _ in range(n_perm)])
    med = np.median(s_null, axis=0)
    ratio = s_obs / med
    return (ratio, s_obs, s_null) if return_null else ratio


def lambda_star_ratio(z, y, eps, n_perm=30, rng=None, seed=0):
    """Lambda*(eps) / median over permutations (pilot: 30 permutations)."""
    return null_ratio(lambda_star, z, y, n_perm=n_perm, rng=rng, seed=seed, eps=eps)


def barrier_curve_ratio(z, y, Lambdas, n_perm=200, rng=None, seed=0):
    """Null-normalized barrier curve e*(L)/median_null e*(L), NaN where the null is zero."""
    rng = np.random.default_rng(seed) if rng is None else rng
    obs = barrier_curve(z, y, Lambdas)
    nul = np.array([barrier_curve(z, rng.permutation(y), Lambdas) for _ in range(n_perm)])
    med = np.median(nul, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(med > 0, obs / med, np.nan)
    return ratio, obs, med
