"""Conditional incremental sufficiency test (Draft v0.2, Section 13.5, Item I18).

Ports the local-swap conditional permutation test (approximate CPT, Berrett et al. 2020) and
Holm step-down adjustment from the companion descriptor discovery study (v09_stats.py, v09_r3_power.py).

Tests H0: R is conditionally independent of an added representation block U given a baseline
descriptor block Psi_k, using the equal-coverage worst-case insufficiency Omega(q) (and the
mean local dispersion on the n_q closest pairs) of the enlarged representation (Psi_k, U).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


def zscore(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    if A.ndim == 1:
        A = A[:, None]
    s = A.std(0)
    keep = s > 1e-12
    Z = np.zeros_like(A)
    Z[:, keep] = (A[:, keep] - A[:, keep].mean(0)) / s[keep]
    return Z[:, keep]


def whiten_svd(A: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    A = np.asarray(A, float)
    if A.ndim == 1:
        A = A[:, None]
    A = A - A.mean(0)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > tol * s.max()] * np.sqrt(len(A))


def holm_adjust(p: np.ndarray) -> np.ndarray:
    """Holm step-down multiplicity adjustment."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def _stats_from_d(d: np.ndarray, dR: np.ndarray, nq: int) -> np.ndarray:
    sel = d <= np.partition(d, nq - 1)[nq - 1] + 1e-12
    sub = dR[sel]
    return np.array([float(sub.max()), float(sub.mean())])


def _swap_perm_factory(Xk: np.ndarray, k_nn: int = 5):
    Zk = zscore(Xk)
    n = len(Xk)
    Dd = np.sqrt(((Zk[:, None, :] - Zk[None, :, :]) ** 2).sum(-1)) if Zk.shape[1] else np.zeros((n, n))
    np.fill_diagonal(Dd, np.inf)
    nbrs = [
        np.unique(np.r_[np.flatnonzero(Dd[i] <= 1e-12), np.argsort(Dd[i], kind="stable")[:k_nn]])
        for i in range(n)
    ]

    def draw_perm(rg: np.random.Generator) -> np.ndarray:
        perm = np.arange(n)
        free = np.ones(n, bool)
        for i in rg.permutation(n):
            if not free[i]:
                continue
            cand = [j for j in nbrs[i] if free[j] and j != i]
            if not cand:
                continue
            j = cand[rg.integers(len(cand))]
            free[i] = free[j] = False
            if rg.random() < 0.5:
                perm[i], perm[j] = j, i
        return perm

    return draw_perm


def cond_perm_swap(
    Xk: np.ndarray,
    eta: np.ndarray,
    R: np.ndarray,
    rng: np.random.Generator,
    q: float = 0.01,
    B: int = 1000,
    k_nn: int = 5,
    metric: str = "std",
) -> dict:
    """Local-swap conditional permutation test of adding block `eta` to baseline `Xk` for response `R`."""
    Xk = np.asarray(Xk, float)
    if Xk.ndim == 1:
        Xk = Xk[:, None]
    eta = np.asarray(eta, float)
    if eta.ndim == 1:
        eta = eta[:, None]
    R = np.asarray(R, float)
    n = len(R)
    II, JJ = np.triu_indices(n, 1)
    dR = np.abs(R[II] - R[JJ])
    nq = int(np.ceil(q * len(dR) - 1e-9))
    draw_perm = _swap_perm_factory(Xk, k_nn=k_nn)

    if metric == "std":
        Zk = zscore(Xk)
        Ze = zscore(eta)
        dk2 = pdist(Zk, metric="sqeuclidean") if Zk.shape[1] else np.zeros(len(II))
        De2_mat = squareform(pdist(Ze, metric="sqeuclidean")) if Ze.shape[1] else np.zeros((n, n))
        d2_obs = dk2 + De2_mat[II, JJ]
        obs = _stats_from_d(d2_obs, dR, nq)
        nl = np.empty((B, 2))
        for b in range(B):
            perm = draw_perm(rng)
            d2_b = dk2 + De2_mat[perm[II], perm[JJ]]
            nl[b] = _stats_from_d(d2_b, dR, nq)
    elif metric == "white":
        d_obs = pdist(whiten_svd(np.hstack([Xk, eta])))
        obs = _stats_from_d(d_obs, dR, nq)
        nl = np.empty((B, 2))
        for b in range(B):
            perm = draw_perm(rng)
            d_b = pdist(whiten_svd(np.hstack([Xk, eta[perm]])))
            nl[b] = _stats_from_d(d_b, dR, nq)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    nl_mean = nl.mean(0)
    ratio = obs / np.maximum(nl_mean, 1e-12)
    p_val = (1.0 + (nl <= obs).sum(0)) / (1.0 + B)
    return {
        "omega_obs": float(obs[0]),
        "omega_cnull": float(nl_mean[0]),
        "ratio_max": float(ratio[0]),
        "p_max": float(p_val[0]),
        "mean_obs": float(obs[1]),
        "mean_cnull": float(nl_mean[1]),
        "ratio_mean": float(ratio[1]),
        "p_mean": float(p_val[1]),
    }
