"""Evaluation of a representation with the protocol of Draft v0.2, Section 11:
average prediction (ridge head), Omega/Omega_null on a test set, cross-fitted McShane error
(train anchors, Lambda chosen on validation, relative to sd and to permuted training responses)
and Lambda*(eps_noise) relative to its permutation null. Raw e* is reported only descriptively."""
from __future__ import annotations

import numpy as np

from ..metrics import (WhiteningTransform, Standardizer, omega_ratio, crossfit_errors, select_lambda,
                       lambda_star, exact_barrier, mcshane_mid)

NOISE = {"R_E": 0.0025}           # declared numerical-noise floors (descriptor study)


def noise_floor(target: str) -> float:
    if target in NOISE:
        return NOISE[target]
    if target.startswith("RH_b"):
        return 0.004 * float(target[4:])          # 0.08 at beta = 20 (translation level)
    if target == "R_H":
        return 0.08
    return 0.01


def ridge_fit_predict(Ztr, ytr, Zte, alpha=1e-3):
    A = np.column_stack([np.ones(len(Ztr)), Ztr])
    c = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ ytr)
    return np.column_stack([np.ones(len(Zte)), Zte]) @ c


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def evaluate(Z: dict, y: dict, target: str, rng, metric: str = "white", nperm: int = 200,
             q: float = 0.01, n_null_cf: int = 5, n_null_lam: int = 30, eval_sets=("test",)):
    """Z, y: dicts with keys 'train', 'val' and the evaluation sets (e.g. 'test', 'ood')."""
    T = (WhiteningTransform() if metric == "white" else Standardizer()).fit(Z["train"])
    Zn = {k: T.transform(v) for k, v in Z.items()}
    sd = y["train"].std()
    grid = sd * np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])
    Lam = select_lambda(Zn["train"], y["train"], Zn["val"], y["val"], grid)
    eps = noise_floor(target)
    out = {"Lambda_cf": Lam, "d": int(Z["train"].shape[1])}
    for s in eval_sets:
        zs, ys = Zn[s], y[s]
        p = ridge_fit_predict(Zn["train"], y["train"], zs)
        om = omega_ratio(zs, ys, rng, q=q, nperm=nperm, return_details=True)
        cf = crossfit_errors(Zn["train"], y["train"], zs, ys, Lam)
        cfn = [crossfit_errors(Zn["train"], rng.permutation(y["train"]), zs, ys, Lam)["rmse"] for _ in range(n_null_cf)]
        ls = lambda_star(zs, ys, eps)
        lsn = np.median([lambda_star(zs, rng.permutation(ys), eps) for _ in range(n_null_lam)])
        sds = ys.std()
        out[s] = {"R2_ridge": r2(ys, p), "RMSE_ridge": float(np.sqrt(np.mean((ys - p) ** 2))),
                  "R2_mcshane": 1 - cf["rmse"] ** 2 / sds ** 2,
                  "omega_ratio": om["ratio"], "omega": om["omega"], "omega_p": om["p_perm"],
                  "cf_rmse_sd": cf["rmse"] / sds, "cf_max_sd": cf["max"] / sds,
                  "cf_rmse_null_sd": float(np.mean(cfn)) / sds,
                  "lam_star": ls, "lam_star_ratio": float(ls / lsn) if lsn > 0 else np.nan,
                  "e_star_at_range": exact_barrier(zs, ys, float(ys.max() - ys.min())), "n": int(len(ys))}
    return out
