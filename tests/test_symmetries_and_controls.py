"""Canonical hash (I05) and solver symmetries (I06), energy control, GeM identity."""
import numpy as np
import pytest

from suffml.data.canonical_hash import canonical_hash
from suffml.data.geometry import sample_geometry, mask_of, DEFAULT_CORPUS
from suffml.physics.fft_targets import solve_cell, hazard_family


@pytest.fixture(scope="module")
def cells():
    rng = np.random.default_rng(11)
    return [mask_of(sample_geometry(rng, DEFAULT_CORPUS)[0]) for _ in range(3)]


def transforms(m, rng):
    s = tuple(int(v) for v in rng.integers(0, m.shape[0], 2))
    return [np.roll(m, s, (0, 1)), m[::-1, :], m[:, ::-1], np.roll(m[::-1, ::-1], s, (0, 1))]


def test_canonical_hash_invariance_and_separation(cells):
    rng = np.random.default_rng(0)
    for m in cells:
        h = canonical_hash(m)
        assert all(canonical_hash(t) == h for t in transforms(m, rng))
    assert len({canonical_hash(m) for m in cells}) == len(cells)
    m = cells[0].copy(); m[5, 5] = ~m[5, 5]                 # one-pixel change
    assert canonical_hash(m) != canonical_hash(cells[0])


@pytest.mark.slow
def test_solver_symmetries(cells):
    """Targets invariant to integer shifts and reflections within solver tolerance."""
    rng = np.random.default_rng(1)
    for m in cells[:2]:
        ref = solve_cell(m, keep_fields=False)["scalars"]
        for t in transforms(m, rng):
            s = solve_cell(t, keep_fields=False)["scalars"]
            assert s["R_E"] == pytest.approx(ref["R_E"], rel=1e-6)
            for b in (2, 5, 10, 20, 40):
                assert s[f"RH_b{b}"] == pytest.approx(ref[f"RH_b{b}"], abs=1e-4)


def test_energy_positive_control(cells):
    """Hill--Mandel: the phase means reconstruct W_hom exactly (up to solver tolerance)."""
    for m in cells:
        s = solve_cell(m, keep_fields=False)["scalars"]
        assert s["W_HM"] == pytest.approx(s["W_hom"], rel=1e-5)


def test_gem_identity():
    """ln int_Ym s^beta = beta ln GeM_beta(s) + ln|Ym| (Eq. gem-identity) on a cell."""
    rng = np.random.default_rng(2)
    m = rng.random((127, 127)) < 0.3
    s = np.exp(0.3 * rng.standard_normal((127, 127)))
    h = hazard_family(s, m, betas=(2, 5, 10, 20, 40))
    area = (~m).mean()
    for b in (2, 5, 10, 20, 40):
        gem = np.mean(s[~m] ** b) ** (1.0 / b)
        assert h[f"RH_b{b}"] == pytest.approx(b * np.log(gem) + np.log(area), abs=1e-10)
