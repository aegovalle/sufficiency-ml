"""I04: the library reproduces Tables 1 and 2 of the pilot (pilot_null_normalization.py),
with the same random draws in the same order."""
import numpy as np
import pytest

from suffml.metrics import (whiten_pilot, omega_ratio, exact_barrier, lambda_star, crossfit_halving)

TABLE1 = {  # Om/Om0, e*(rng), e*(2rng), Lam*, Lam*/null, CF-RMSE, CF-sup
    "A": (0.14, 0.015, 0.010, 2.58, 0.007, 0.09, 0.51),
    "B": (0.98, 0.000, 0.000, 2.62, 0.916, 0.89, 2.22),
    "C": (0.92, 0.015, 0.000, 2.86, 0.995, 1.01, 1.97),
}
TABLE2 = {  # k: Om/Om0, Lam*, Lam*/null, e*(rng), CF-RMSE
    0: (0.14, 2.58, 0.008, 0.015, 0.09), 1: (0.29, 3.08, 0.032, 0.081, 0.24),
    2: (0.48, 3.57, 0.170, 0.194, 0.38), 4: (0.67, 3.62, 0.378, 0.244, 0.53),
    8: (0.91, 3.62, 0.689, 0.279, 0.69), 16: (0.95, 2.97, 0.799, 0.060, 0.83),
    30: (0.99, 2.66, 0.926, 0.000, 0.92),
}


@pytest.fixture(scope="module")
def pilot():
    rng = np.random.default_rng(0)
    N = 600
    u = rng.uniform(-1, 1, (N, 2))
    R = np.sin(3 * u[:, 0]) + u[:, 1] ** 2 + 0.02 * rng.standard_normal(N)
    reps = {"A": whiten_pilot(u),
            "B": whiten_pilot(np.hstack([u, rng.standard_normal((N, 30))])),
            "C": whiten_pilot(rng.standard_normal((N, 32)))}
    rangeR = R.max() - R.min(); eps = 0.05
    om = {k: omega_ratio(z, R, rng, q=0.01, nperm=200, ties=False) for k, z in reps.items()}
    e1 = {k: exact_barrier(z, R, rangeR) for k, z in reps.items()}
    e2 = {k: exact_barrier(z, R, 2 * rangeR) for k, z in reps.items()}
    for k, z in reps.items():                       # same generator consumption as the pilot
        [exact_barrier(z, rng.permutation(R), 2 * rangeR) for _ in range(30)]
    ls, lsn = {}, {}
    for k, z in reps.items():
        ls[k] = lambda_star(z, R, eps)
        lsn[k] = ls[k] / np.median([lambda_star(z, rng.permutation(R), eps) for _ in range(30)])
    rng2 = np.random.default_rng(1)
    grid = rangeR * np.array([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
    cf = {}
    for k, z in reps.items():
        cf[k] = crossfit_halving(z, R, rng2, grid)
        crossfit_halving(z, rng2.permutation(R), rng2, grid)
    t2 = {}
    for kn in (0, 1, 2, 4, 8, 16, 30):
        z = whiten_pilot(np.hstack([u, rng2.standard_normal((N, kn))])) if kn else whiten_pilot(u)
        L0 = lambda_star(z, R, eps)
        Ln = np.median([lambda_star(z, rng2.permutation(R), eps) for _ in range(30)])
        t2[kn] = (omega_ratio(z, R, rng2, ties=False), L0, L0 / Ln, exact_barrier(z, R, rangeR),
                  crossfit_halving(z, R, rng2, grid)[0])
    return om, e1, e2, ls, lsn, cf, t2


def test_table1(pilot):
    om, e1, e2, ls, lsn, cf, _ = pilot
    for k, ref in TABLE1.items():
        got = (om[k], e1[k], e2[k], ls[k], lsn[k], cf[k][0], cf[k][1])
        for g, r, nd in zip(got, ref, (2, 3, 3, 2, 3, 2, 2)):
            assert round(g, nd) == pytest.approx(r, abs=10 ** -nd / 2 + 1e-12), (k, got, ref)


def test_table2(pilot):
    t2 = pilot[-1]
    for k, ref in TABLE2.items():
        for g, r, nd in zip(t2[k], ref, (2, 2, 3, 3, 2)):
            assert round(g, nd) == pytest.approx(r, abs=10 ** -nd / 2 + 1e-12), (k, t2[k], ref)


def test_null_normalized_metrics_rank_correctly(pilot):
    """The conclusion of the pilot: raw e* misranks, null-normalized metrics do not."""
    om, e1, _, _, lsn, cf, t2 = pilot
    assert e1["B"] < e1["A"]                         # raw barrier rewards dilution
    assert om["A"] < om["B"] and om["A"] < om["C"]
    assert lsn["A"] < lsn["B"] and cf["A"][0] < cf["B"][0]
    ks = sorted(t2)
    assert all(t2[a][4] < t2[b][4] for a, b in zip(ks, ks[1:]))   # CF monotone in dilution
