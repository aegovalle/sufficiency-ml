"""Critical-distance onset intensity J_r of the descriptor study (Draft v0.2, Section 8.4):
J_r = 1/2 ln max_x <psi+/psi_c>_{D_r(x)} - 1/2 ln max(psi+/psi_c)_hom, disk over both phases,
local psi_c of each phase (same code as campaign/tcd_frozen/tcd_frozen.py, any radius)."""
from __future__ import annotations

import numpy as np

from .fft_rve import FFTElastic2D, lame
from .pf_fft import PhaseFieldRVE

ELL = 0.025


class CriticalDistance:
    def __init__(self, N=127, radii_ell=(0.0, 1.0, 1.5, 2.0), ell=ELL, kappa=10.0, nu=0.3):
        self.N, self.radii = N, tuple(radii_ell)
        self.lam_m, self.mu_m = lame(1.0, nu)
        self.lam_i, self.mu_i = lame(kappa, nu)
        self.sol = FFTElastic2D(N)
        x = (np.arange(N) + 0.5) / N
        X1, X2 = np.meshgrid(x, x, indexing="ij")
        d1 = (X1 - 0.5 / N + 0.5) % 1.0 - 0.5
        d2 = (X2 - 0.5 / N + 0.5) % 1.0 - 0.5
        self.KH = {}
        for a in self.radii:
            if a > 0:
                K = ((d1 ** 2 + d2 ** 2) <= (a * ell) ** 2).astype(float)
                self.KH[a] = np.fft.fft2(K / K.sum())
        pf0 = PhaseFieldRVE(np.zeros((N, N), bool))
        one, zero = np.ones((N, N)), np.zeros((N, N))
        self.H0 = 0.5 * np.log((pf0.psi_plus((one, zero, zero)) / pf0.psic).max())

    def __call__(self, mask) -> dict:
        lam = np.where(mask, self.lam_i, self.lam_m)
        mu = np.where(mask, self.mu_i, self.mu_m)
        (e11, e22, e12), _, _, _ = self.sol.solve(lam, mu, (1.0, 0.0, 0.0), tol=1e-7, maxit=4000)
        pf = PhaseFieldRVE(mask)
        ratio = pf.psi_plus((e11, e22, e12)) / pf.psic
        out = {}
        for a in self.radii:
            key = f"J_{a:g}"
            if a == 0:
                out[key] = 0.5 * np.log(ratio.max()) - self.H0
            else:
                out[key] = 0.5 * np.log(np.real(np.fft.ifft2(np.fft.fft2(ratio) * self.KH[a])).max()) - self.H0
        return out
