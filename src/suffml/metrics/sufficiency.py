"""Equal-coverage worst-case statistic Omega(q) and its response-permutation null
(Draft v0.2, Section 11.2; descriptor study, Section 4)."""
from __future__ import annotations

import numpy as np

from .lipschitz_barrier import pairwise_euclidean


def closest_pairs(z: np.ndarray, q: float = 0.01, ties: bool = True, chunk: int = 1024):
    """Indices (i, j), i < j, of the n_q = ceil(q * n(n-1)/2) closest pairs.
    ties=True retains every pair tied with the cut-off distance (descriptor study);
    ties=False keeps exactly n_q pairs in argsort order (pilot script)."""
    n = len(z)
    npairs = n * (n - 1) // 2
    nq = int(np.ceil(q * npairs - 1e-9))
    if n <= 3000:                                    # dense path, exact argsort order
        iu = np.triu_indices(n, 1)
        d = pairwise_euclidean(z, z)[iu]
        if not ties:
            idx = np.argsort(d, kind="stable")[:nq]
            return iu[0][idx], iu[1][idx], d[idx]
        cut = np.partition(d, nq - 1)[nq - 1]
        sel = d <= cut + 1e-12
        return iu[0][sel], iu[1][sel], d[sel]
    # chunked path: keep the nq smallest candidates
    cand_d, cand_i, cand_j = np.empty(0), np.empty(0, int), np.empty(0, int)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        D = pairwise_euclidean(z[a:b], z)
        ii, jj = np.nonzero(np.arange(n)[None, :] > np.arange(a, b)[:, None])
        dd = D[ii, jj]
        cand_d = np.r_[cand_d, dd]; cand_i = np.r_[cand_i, ii + a]; cand_j = np.r_[cand_j, jj]
        if len(cand_d) > 4 * nq:
            k = np.argpartition(cand_d, nq)[: 2 * nq]
            cand_d, cand_i, cand_j = cand_d[k], cand_i[k], cand_j[k]
    order = np.argsort(cand_d, kind="stable")
    cand_d, cand_i, cand_j = cand_d[order], cand_i[order], cand_j[order]
    if not ties:
        return cand_i[:nq], cand_j[:nq], cand_d[:nq]
    sel = cand_d <= cand_d[nq - 1] + 1e-12
    return cand_i[sel], cand_j[sel], cand_d[sel]


def omega(z, y, q: float = 0.01, ties: bool = True, return_pair: bool = False):
    """Omega(q): largest |y_i - y_j| among the q-fraction closest pairs."""
    i, j, _ = closest_pairs(z, q, ties)
    dy = np.abs(y[i] - y[j])
    k = int(np.argmax(dy))
    return (float(dy[k]), (int(i[k]), int(j[k]))) if return_pair else float(dy[k])


def omega_ratio(z, y, rng, q: float = 0.01, nperm: int = 200, ties: bool = True,
                return_details: bool = False):
    """Omega(q) over the mean of the same statistic under response permutations (z fixed;
    the selected pairs do not depend on y, so they are computed once)."""
    i, j, _ = closest_pairs(z, q, ties)
    om = float(np.abs(y[i] - y[j]).max())
    nul = np.array([np.abs(yp[i] - yp[j]).max() for yp in (rng.permutation(y) for _ in range(nperm))])
    r = om / nul.mean()
    if return_details:
        return {"ratio": r, "omega": om, "null_mean": float(nul.mean()),
                "p_perm": float((1 + np.sum(nul <= om)) / (1 + nperm)), "n_pairs": int(len(i))}
    return r
