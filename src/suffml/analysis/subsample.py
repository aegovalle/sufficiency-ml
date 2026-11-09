"""Subsampling intervals for Omega/Omega_null, with paired comparisons between representations
(the same subsamples and the same response permutations for every representation)."""
from __future__ import annotations

import numpy as np

from ..metrics.sufficiency import closest_pairs


def paired_omega_subsample(Zs: dict, y: np.ndarray, rng, frac: float = 0.5, B: int = 100,
                           nperm: int = 100, q: float = 0.01):
    """Zs: name -> (n x d) embeddings ALREADY in the metric (transform fitted on train).
    Returns dict name -> array of B ratios Omega/mean(Omega_null) on random subsamples."""
    n = len(y)
    m = int(round(frac * n))
    out = {k: np.empty(B) for k in Zs}
    for b in range(B):
        sub = rng.choice(n, m, replace=False)
        ys = y[sub]
        perms = [rng.permutation(m) for _ in range(nperm)]
        for k, Z in Zs.items():
            i, j, _ = closest_pairs(Z[sub], q, ties=True)
            om = np.abs(ys[i] - ys[j]).max()
            nul = np.mean([np.abs(ys[p][i] - ys[p][j]).max() for p in perms])
            out[k][b] = om / nul
    return out


def interval(a, lo=5, hi=95):
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))
