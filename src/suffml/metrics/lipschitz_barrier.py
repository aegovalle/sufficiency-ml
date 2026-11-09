"""Exact finite-sample Lipschitz barrier, its constant Lambda*, McShane interpolation and
cross-fitted McShane error (Draft v0.2, Sections 4, 5.2 and 11).

All functions take embeddings that are ALREADY in the metric of the bound, i.e. rows
z_n = N(z)/sqrt(d) (or whitened / sqrt(d)); the Euclidean distance between rows is d_z.
Use `metric.WhiteningTransform` / `metric.Standardizer` to build them from raw embeddings.
Pair computations are chunked so that N = 10^4 fits in memory.
"""
from __future__ import annotations

import numpy as np


def pairwise_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix of Euclidean distances between the rows of a and b."""
    ga = np.einsum("ij,ij->i", a, a)
    gb = np.einsum("ij,ij->i", b, b)
    return np.sqrt(np.maximum(ga[:, None] + gb[None, :] - 2.0 * a @ b.T, 0.0))


def _row_chunks(n: int, chunk: int):
    for a in range(0, n, chunk):
        yield a, min(a + chunk, n)


def max_residual(z: np.ndarray, y: np.ndarray, Lambda: float, chunk: int = 1024,
                 return_pair: bool = False):
    """max_{i<j} |y_i - y_j| - Lambda d_ij (may be negative), optionally with its pair."""
    n = len(y)
    best, pair = -np.inf, (-1, -1)
    for a, b in _row_chunks(n, chunk):
        d = pairwise_euclidean(z[a:b], z)
        r = np.abs(y[a:b, None] - y[None, :]) - Lambda * d
        # exclude i >= j (diagonal and lower triangle)
        rows = np.arange(a, b)[:, None]
        r[np.arange(n)[None, :] <= rows] = -np.inf
        k = int(np.argmax(r))
        v = r.flat[k]
        if v > best:
            best, pair = float(v), (a + k // n, k % n)
    return (best, pair) if return_pair else best


def exact_barrier(z: np.ndarray, y: np.ndarray, Lambda: float, chunk: int = 1024,
                  return_pair: bool = False):
    """e*(Lambda) = 1/2 max_ij [|y_i - y_j| - Lambda d_ij]_+  (Theorem 1)."""
    m, pair = max_residual(z, y, Lambda, chunk, return_pair=True)
    e = 0.5 * max(m, 0.0)
    return (e, pair) if return_pair else e


def barrier_curve(z: np.ndarray, y: np.ndarray, Lambdas, chunk: int = 1024) -> np.ndarray:
    """e*(Lambda) for an array of Lambda values in one pass over the pairs."""
    L = np.asarray(Lambdas, float)
    n = len(y)
    best = np.full(len(L), -np.inf)
    for a, b in _row_chunks(n, chunk):
        d = pairwise_euclidean(z[a:b], z)
        dy = np.abs(y[a:b, None] - y[None, :])
        mask = np.arange(n)[None, :] > np.arange(a, b)[:, None]
        dd, ddy = d[mask], dy[mask]
        if dd.size == 0:
            continue
        for k, lk in enumerate(L):
            best[k] = max(best[k], float((ddy - lk * dd).max()))
    return 0.5 * np.maximum(best, 0.0)


def lambda_star(z: np.ndarray, y: np.ndarray, eps: float, chunk: int = 1024,
                dmin: float = 1e-12) -> float:
    """Smallest Lambda with e*(Lambda) <= eps: max over pairs of (|dy| - 2 eps)/d.
    Exact collisions (d = 0 with |dy| > 2 eps) give values of order 1/dmin."""
    n = len(y)
    best = 0.0
    for a, b in _row_chunks(n, chunk):
        d = pairwise_euclidean(z[a:b], z)
        r = np.abs(y[a:b, None] - y[None, :]) - 2.0 * eps
        mask = (np.arange(n)[None, :] > np.arange(a, b)[:, None]) & (r > 0)
        if mask.any():
            best = max(best, float((r[mask] / np.maximum(d[mask], dmin)).max()))
    return best


def exact_collisions(z: np.ndarray, y: np.ndarray, tol_z: float = 0.0, eps: float = 0.0,
                     chunk: int = 1024):
    """Pairs with d_ij <= tol_z and |y_i - y_j| > 2 eps (e*(inf) > 0 if any)."""
    n = len(y)
    out = []
    for a, b in _row_chunks(n, chunk):
        d = pairwise_euclidean(z[a:b], z)
        dy = np.abs(y[a:b, None] - y[None, :])
        mask = (np.arange(n)[None, :] > np.arange(a, b)[:, None]) & (d <= tol_z) & (dy > 2 * eps)
        for i, j in zip(*np.nonzero(mask)):
            out.append((a + int(i), int(j), float(dy[i, j])))
    return out


def mcshane_mid(z_anchor: np.ndarray, y_anchor: np.ndarray, z_query: np.ndarray,
                Lambda: float, chunk: int = 2048, exclude=None) -> np.ndarray:
    """Midpoint of the upper and lower McShane extensions built on the anchors
    (Definition 1). `exclude` (optional): boolean matrix |Q| x |A| of forbidden anchors
    (e.g. same cell as the query)."""
    out = np.empty(len(z_query))
    for a, b in _row_chunks(len(z_query), chunk):
        d = pairwise_euclidean(z_query[a:b], z_anchor)
        up = y_anchor[None, :] + Lambda * d
        lo = y_anchor[None, :] - Lambda * d
        if exclude is not None:
            ex = exclude[a:b]
            up = np.where(ex, np.inf, up)
            lo = np.where(ex, -np.inf, lo)
        out[a:b] = 0.5 * (up.min(1) + lo.max(1))
    return out


def crossfit_errors(z_anchor, y_anchor, z_query, y_query, Lambda, q_hi: float | None = None):
    """Out-of-sample errors of the McShane midpoint of the anchors on the queries.
    Returns dict with rmse, max (e^cf), mae and optionally a high quantile."""
    e = mcshane_mid(z_anchor, y_anchor, z_query, Lambda) - y_query
    out = {"rmse": float(np.sqrt(np.mean(e ** 2))), "max": float(np.abs(e).max()),
           "mae": float(np.abs(e).mean())}
    if q_hi is not None:
        out[f"q{q_hi:g}"] = float(np.quantile(np.abs(e), q_hi))
    return out


def select_lambda(z_fit, y_fit, z_val, y_val, grid, criterion: str = "rmse") -> float:
    """Lambda from a grid minimizing the validation error of the McShane midpoint."""
    errs = [crossfit_errors(z_fit, y_fit, z_val, y_val, L)[criterion] for L in grid]
    return float(grid[int(np.argmin(errs))])


def crossfit_halving(z, y, rng, grid, nsplit: int = 10):
    """Cross-fitted McShane error by random halving (pilot protocol, Section 6):
    for each split A/B, Lambda is chosen inside A (A1 -> A2), the interpolant is built on A
    and evaluated on B. Returns mean RMSE/sd(y) and mean sup-error/sd(y)."""
    n, sd = len(y), y.std()
    rm, sup = [], []
    for _ in range(nsplit):
        p = rng.permutation(n)
        A, B = p[: n // 2], p[n // 2:]
        A1, A2 = A[: len(A) // 2], A[len(A) // 2:]
        errs = [np.sqrt(np.mean((mcshane_mid(z[A1], y[A1], z[A2], L) - y[A2]) ** 2)) for L in grid]
        L = grid[int(np.argmin(errs))]
        e = mcshane_mid(z[A], y[A], z[B], L) - y[B]
        rm.append(np.sqrt(np.mean(e ** 2)) / sd)
        sup.append(np.abs(e).max() / sd)
    return float(np.mean(rm)), float(np.mean(sup))
