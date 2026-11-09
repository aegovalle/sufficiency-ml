"""Unit tests for the local-swap conditional permutation test (Item I18)."""
import numpy as np

from suffml.metrics import cond_perm_swap, holm_adjust


def test_cond_perm_swap_signal_and_null():
    rng = np.random.default_rng(42)
    n = 150
    Xk = rng.normal(size=(n, 3))
    z_hidden = rng.normal(size=n)
    # Response depends on both Xk and z_hidden
    R = Xk[:, 0] - 0.5 * Xk[:, 1] + 2.0 * z_hidden + rng.normal(0, 0.05, size=n)

    # 1. Informative block (contains z_hidden)
    eta_good = np.column_stack([z_hidden, rng.normal(size=(n, 2))])
    res_good = cond_perm_swap(Xk, eta_good, R, rng, q=0.01, B=200, metric="std")
    assert res_good["ratio_max"] < 0.65
    assert res_good["p_max"] < 0.02

    # 2. Uninformative block (purely a noisy function of Xk, conditionally independent of R given Xk)
    eta_null = 0.8 * Xk[:, :2] + rng.normal(0, 0.5, size=(n, 2))
    res_null = cond_perm_swap(Xk, eta_null, R, rng, q=0.01, B=200, metric="std")
    assert res_null["p_max"] > 0.05

    # 3. Holm step-down check
    adj = holm_adjust([0.002, 0.04, 0.20])
    assert np.allclose(adj, [0.006, 0.08, 0.20])
