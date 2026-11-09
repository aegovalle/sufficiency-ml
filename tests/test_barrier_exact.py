"""I03: chunked exact e* and Lambda* agree with brute force; McShane midpoint attains e*."""
import numpy as np
import pytest

from suffml.metrics import (exact_barrier, barrier_curve, lambda_star, mcshane_mid, omega,
                            closest_pairs, exact_collisions)


def brute_barrier(z, y, L):
    best = 0.0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            best = max(best, abs(y[i] - y[j]) - L * np.linalg.norm(z[i] - z[j]))
    return 0.5 * best


def brute_lambda_star(z, y, eps):
    best = 0.0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            r = abs(y[i] - y[j]) - 2 * eps
            if r > 0:
                best = max(best, r / max(np.linalg.norm(z[i] - z[j]), 1e-12))
    return best


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_barrier_and_lambda_star_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((60, 3)); y = rng.standard_normal(60)
    for L in (0.0, 0.3, 1.0, 5.0):
        assert exact_barrier(z, y, L, chunk=7) == pytest.approx(brute_barrier(z, y, L), abs=1e-12)
    curve = barrier_curve(z, y, [0.0, 0.3, 1.0, 5.0], chunk=11)
    assert np.allclose(curve, [brute_barrier(z, y, L) for L in (0.0, 0.3, 1.0, 5.0)], atol=1e-12)
    for eps in (0.0, 0.1, 0.5):
        assert lambda_star(z, y, eps, chunk=9) == pytest.approx(brute_lambda_star(z, y, eps), rel=1e-12)


def test_lambda_star_is_feasibility_threshold():
    rng = np.random.default_rng(3)
    z = rng.standard_normal((80, 2)); y = rng.standard_normal(80); eps = 0.1
    Ls = lambda_star(z, y, eps)
    assert exact_barrier(z, y, Ls * (1 + 1e-9)) <= eps + 1e-12
    assert exact_barrier(z, y, Ls * (1 - 1e-3)) > eps


@pytest.mark.parametrize("L", [0.2, 1.0, 4.0])
def test_mcshane_midpoint_attains_barrier(L):
    rng = np.random.default_rng(4)
    z = rng.standard_normal((70, 4)); y = np.sin(z[:, 0]) + 0.1 * rng.standard_normal(70)
    mid = mcshane_mid(z, y, z, L)
    # Gram-matrix distances carry O(1e-8) round-off at zero distance
    assert np.abs(mid - y).max() == pytest.approx(exact_barrier(z, y, L), abs=1e-7)
    # Lipschitz on random query pairs
    q = rng.standard_normal((200, 4)); m = mcshane_mid(z, y, q, L)
    d = np.linalg.norm(q[:100] - q[100:], axis=1)
    assert np.all(np.abs(m[:100] - m[100:]) <= L * d + 1e-12)


def test_omega_dense_and_chunked_paths_agree():
    rng = np.random.default_rng(5)
    z = rng.standard_normal((3100, 2)); y = rng.standard_normal(3100)
    i1, j1, d1 = closest_pairs(z, 0.001, ties=True)
    # force the dense computation for comparison on a subset size limit by direct brute force
    n = len(z)
    G = (z ** 2).sum(1)
    D = np.sqrt(np.maximum(G[:, None] + G[None, :] - 2 * z @ z.T, 0))
    iu = np.triu_indices(n, 1)
    dd = D[iu]
    nq = int(np.ceil(0.001 * len(dd) - 1e-9))
    cut = np.partition(dd, nq - 1)[nq - 1]
    assert len(i1) == int(np.sum(dd <= cut + 1e-12))
    assert d1.max() == pytest.approx(cut)


def test_exact_collisions():
    z = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]); y = np.array([0.0, 1.0, 0.0])
    c = exact_collisions(z, y)
    assert c == [(0, 1, 1.0)]
    assert exact_barrier(z, y, 1e9) == pytest.approx(0.5)
