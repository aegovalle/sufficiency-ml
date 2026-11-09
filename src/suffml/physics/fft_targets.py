"""Elastic targets and a-priori/solved-field descriptors of one periodic cell
(Draft v0.2, Section 8): R_E = W_hom/W_m, R_H(beta) = ln int_Ym s^beta for beta in BETAS,
phase means E0, Jensen gaps (E1), interface density, ligament amplification, two-point map.
The solver is the Moulinec--Suquet scheme of the descriptor study (physics/fft_rve.py)."""
from __future__ import annotations

import numpy as np

from .fft_rve import FFTElastic2D, lame, analyze_state, _disk_kernel_hat
from .descr_tools import ligament_amp

N_DEFAULT = 127
ELL = 0.025
TOL = 1e-7
BETAS = (2, 5, 10, 20, 40)
KAPPA = 10.0
NU = 0.3
EBAR = (1.0, 0.0, 0.0)
S2_CROP = 31


def materials(kappa: float = KAPPA, nu: float = NU):
    lam_m, mu_m = lame(1.0, nu)
    lam_i, mu_i = lame(kappa, nu)
    return lam_m, mu_m, lam_i, mu_i


def hazard_family(sbar: np.ndarray, mask: np.ndarray, betas=BETAS) -> dict:
    """R_H(beta) = ln int_{Y_m} s^beta dx from the normalized nonlocal stress field
    (equal to lnB of the descriptor study for beta = 20)."""
    sm = sbar[~mask].astype(float)
    sp = sm.max()
    h2 = 1.0 / mask.size
    return {f"RH_b{b}": float(b * np.log(sp) + np.log(np.sum((sm / sp) ** b) * h2)) for b in betas}


def two_point_map(mask: np.ndarray, crop: int = S2_CROP) -> np.ndarray:
    """Cropped two-point autocovariance of the phase indicator (MKS baseline)."""
    x = mask.astype(float) - mask.mean()
    ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(np.fft.fft2(x)) ** 2)) / x.size)
    c0 = mask.shape[0] // 2
    return ac[c0 - crop:c0 + crop + 1, c0 - crop:c0 + crop + 1].astype(np.float32)


def solve_cell(mask: np.ndarray, sol: FFTElastic2D | None = None, kappa: float = KAPPA,
               ebar=EBAR, betas=BETAS, keep_fields: bool = True, ell: float = ELL) -> dict:
    """Solve one cell and return targets, descriptors and (optionally) fields."""
    N = mask.shape[0]
    sol = sol or FFTElastic2D(N)
    mats = materials(kappa)
    lam = np.where(mask, mats[2], mats[0])
    mu = np.where(mask, mats[3], mats[1])
    st, ss, it, err = sol.solve(lam, mu, ebar, tol=TOL, maxit=4000)
    out = analyze_state(mask, st, ss, mats, ebar, m=20, ell=ell, keep_field=True)
    sbar = out.pop("_sbar")
    out["R_E"] = out["W_hom"]
    out.update(hazard_family(sbar, mask, betas))
    out["R_H"] = out["RH_b20"]
    out["iters"], out["eq_err"] = int(it), float(err)
    out["A_lig"] = ligament_amp(mask, k=kappa)[0]
    res = {"scalars": out, "S2": two_point_map(mask)}
    if keep_fields:
        res["stress"] = np.stack(ss).astype(np.float16)
        res["sbar"] = sbar.astype(np.float16)
    return res
