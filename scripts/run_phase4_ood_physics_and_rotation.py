"""Phase 4 (Draft v0.2 Section 9.3, Experiment G):
1. Rotation sanity check (load along e_2 vs 90-deg rotated images under e_1 load on n=300 test cells):
   - Verifies exact continuum/FFT equivalence between load along e_2 on mask X and load along e_1 on rot90(X).
   - Quantifies physical response shift |R_H(X; e_2) - R_H(X; e_1)| on anisotropic cells.
   - Evaluates frozen encoders (trained under e_1 load) zero-shot on rot90(X) against R_H(X; e_2).
2. OOD-physics transfer (stiffness contrast kappa in {5, 20} and biaxial strain Ebar=(1.0, 0.5, 0.0)):
   - Solves 200 adaptation cells (from val split) + 300 evaluation cells (test split) via FFT.
   - Evaluates frozen kappa=10 uniaxial encoders on head-free worst-case sufficiency (Omega/Omega_null)
     and few-shot transfer (n_few in {50, 200}) via Ridge R^2 and cross-fitted McShane interpolation.

Outputs:
- data/processed/ood_physics_and_rotation.npz
- results/ood_physics_rotation_metrics.csv
- results/ood_physics_rotation_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import torch

from suffml.analysis.evaluate import r2, ridge_fit_predict
from suffml.data.datasets import load_pilot
from suffml.metrics import WhiteningTransform, crossfit_errors, omega_ratio, select_lambda
from suffml.physics.fft_rve import FFTElastic2D, _disk_kernel_hat
from suffml.physics.fft_targets import BETAS, ELL, NU, TOL, hazard_family, materials
from suffml.train.train_gate import DEV, Supervised, embed_all

MD = os.path.join(ROOT, "results", "models")

FAMILIES = {
    "MSE": ("gem", ["sup_RH_b20_gem2_d8_s0", "sup_RH_b20_gem2_d8_s1", "sup_RH_b20_gem2_d8_s2"]),
    "SBR-in (lam=1.0)": ("gem", ["obj_RH_b20_sbrin1_d8_s0", "obj_RH_b20_sbrin1_d8_s1", "obj_RH_b20_sbrin1_d8_s2"]),
    "RankSim (gam=1.0)": ("gem", ["obj_RH_b20_ranksimg1_d8_s0", "obj_RH_b20_ranksimg1_d8_s1", "obj_RH_b20_ranksimg1_d8_s2"]),
    "RNC (tau=1.0)": ("gem", ["obj_RH_b20_rnct1_d8_s0", "obj_RH_b20_rnct1_d8_s1", "obj_RH_b20_rnct1_d8_s2"]),
    "SBR-CF (lam=1.0)": ("gem", ["obj_RH_b20_sbrcf1_d8_s0", "obj_RH_b20_sbrcf1_d8_s1", "obj_RH_b20_sbrcf1_d8_s2"]),
    "SBR-CF (lam=3.0)": ("gem", ["obj_RH_b20_sbrcf3_d8_s0", "obj_RH_b20_sbrcf3_d8_s1", "obj_RH_b20_sbrcf3_d8_s2"]),
}


def solve_physics_state(mask: np.ndarray, kappa: float, ebar: tuple[float, float, float],
                        sol: FFTElastic2D | None = None, ell: float = ELL) -> dict[str, float]:
    """Solve periodic cell for arbitrary (kappa, ebar) and compute exact R_E and R_H(beta)."""
    N = mask.shape[0]
    sol = sol or FFTElastic2D(N)
    lam_m, mu_m, lam_i, mu_i = materials(kappa, NU)
    lam = np.where(mask, lam_i, lam_m)
    mu = np.where(mask, mu_i, mu_m)
    (e11, e22, e12), (s11, s22, s12), it, err = sol.solve(lam, mu, ebar, tol=TOL, maxit=4000)

    eb11, eb22, eb12 = ebar
    tr_b = eb11 + eb22
    Wref = 0.5 * lam_m * (tr_b ** 2) + mu_m * (eb11 ** 2 + eb22 ** 2 + 2.0 * (eb12 ** 2))
    w = 0.5 * (s11 * e11 + s22 * e22 + 2.0 * s12 * e12)
    R_E = float(w.mean() / Wref)

    s11_ref = lam_m * tr_b + 2.0 * mu_m * eb11
    s22_ref = lam_m * tr_b + 2.0 * mu_m * eb22
    s12_ref = 2.0 * mu_m * eb12
    sref = 0.5 * (s11_ref + s22_ref) + np.sqrt((0.5 * (s11_ref - s22_ref)) ** 2 + s12_ref ** 2)

    mat = ~mask
    s1 = 0.5 * (s11 + s22) + np.sqrt((0.5 * (s11 - s22)) ** 2 + s12 ** 2)
    s1p = np.maximum(s1, 0.0) * mat
    Kh = _disk_kernel_hat(N, ell)
    num = np.real(np.fft.ifft2(np.fft.fft2(s1p) * Kh))
    den = np.real(np.fft.ifft2(np.fft.fft2(mat.astype(float)) * Kh))
    sbar = np.where(mat & (den > 1e-12), num / np.maximum(den, 1e-12), 0.0) / sref

    out = {"R_E": R_E, "iters": int(it), "eq_err": float(err)}
    out.update(hazard_family(sbar, mask, BETAS))
    return out


def _worker(task):
    idx, mask, kappa, ebar = task
    return idx, solve_physics_state(mask, kappa=kappa, ebar=ebar)


def embed_masks_with_tag(tag: str, pool: str, masks: np.ndarray, d: int = 8) -> np.ndarray:
    pt_path = os.path.join(MD, f"{tag}.pt")
    js_path = os.path.join(MD, f"{tag}.json")
    info = json.load(open(js_path))
    model = Supervised(d, pool, info.get("Lambda", 16.0)).to(DEV)
    state = torch.load(pt_path, map_location=DEV, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    X = torch.from_numpy(masks[:, None].astype(np.float32))
    return embed_all(model.embed, X)


def best_ridge_r2(Z_tr: np.ndarray, y_tr: np.ndarray, Z_te: np.ndarray, y_te: np.ndarray) -> float:
    """Select ridge alpha by 80/20 split on (Z_tr, y_tr) and evaluate R^2 on (Z_te, y_te)."""
    n_a = max(10, int(0.8 * len(y_tr)))
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    best_a, best_mse = 1e-2, float("inf")
    for a in alphas:
        p_val = ridge_fit_predict(Z_tr[:n_a], y_tr[:n_a], Z_tr[n_a:], alpha=a)
        mse = float(np.mean((y_tr[n_a:] - p_val) ** 2))
        if mse < best_mse:
            best_mse, best_a = mse, a
    pred = ridge_fit_predict(Z_tr, y_tr, Z_te, alpha=best_a)
    return r2(y_te, pred)


def eval_transfer(Z_tr_white: np.ndarray, y_tr: np.ndarray,
                  Z_te_white: np.ndarray, y_te: np.ndarray) -> dict[str, float]:
    """Fit Ridge and McShane interpolant on (Z_tr_white, y_tr) and evaluate on (Z_te_white, y_te)."""
    sd = float(y_te.std())
    r2_rg = best_ridge_r2(Z_tr_white, y_tr, Z_te_white, y_te)
    n_a = max(10, int(0.8 * len(y_tr)))
    grid = float(y_tr.std()) * np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    lam = select_lambda(Z_tr_white[:n_a], y_tr[:n_a], Z_tr_white[n_a:], y_tr[n_a:], grid)
    cf = crossfit_errors(Z_tr_white, y_tr, Z_te_white, y_te, lam)
    r2_mc = float(1.0 - (cf["rmse"] ** 2) / (sd ** 2 + 1e-12))
    return {
        "r2_ridge": r2_rg,
        "r2_mcshane": r2_mc,
        "cf_rmse_sd": float(cf["rmse"] / sd),
        "cf_max_sd": float(cf["max"] / sd),
        "lam_mcshane": float(lam),
    }


def main():
    t0 = time.time()
    pilot = load_pilot(ROOT)
    D = pilot["table"]
    masks = pilot["mask"]
    idx_train = D.index[D.split == "train"].values
    idx_val = D.index[D.split == "val"].values[:200]   # 200 adaptation cells for n_few in {50, 200}
    idx_test = D.index[D.split == "test"].values        # 300 held-out test cells

    cache_path = os.path.join(ROOT, "data", "processed", "ood_physics_and_rotation.npz")
    if os.path.exists(cache_path):
        print(f"Loading cached FFT physics solves from {cache_path}...")
        npz = np.load(cache_path)
        y_rot_e2_RE = npz["y_rot_e2_RE"]
        y_rot_e2_RH = npz["y_rot_e2_RH"]
        y_rot_img_RE = npz["y_rot_img_RE"]
        y_rot_img_RH = npz["y_rot_img_RH"]
        y_k5_adapt = npz["y_k5_adapt"]
        y_k5_test = npz["y_k5_test"]
        y_k20_adapt = npz["y_k20_adapt"]
        y_k20_test = npz["y_k20_test"]
        y_biax_adapt = npz["y_biax_adapt"]
        y_biax_test = npz["y_biax_test"]
    else:
        nw = min(24, os.cpu_count() or 8)
        print(f"Solving OOD-physics (kappa=5, kappa=20, biaxial) and Rotation Check (e_2 load) with {nw} workers...")
        masks_adapt = masks[idx_val]
        masks_test = masks[idx_test]
        masks_test_rot90 = np.array([np.rot90(m, 1) for m in masks_test])

        tasks = []
        for i, m in enumerate(masks_test):
            tasks.append((("rot_e2", i), m, 10.0, (0.0, 1.0, 0.0)))
        for i, m in enumerate(masks_test_rot90):
            tasks.append((("rot_img", i), m, 10.0, (1.0, 0.0, 0.0)))
        for i, m in enumerate(masks_adapt):
            tasks.append((("k5_adapt", i), m, 5.0, (1.0, 0.0, 0.0)))
        for i, m in enumerate(masks_test):
            tasks.append((("k5_test", i), m, 5.0, (1.0, 0.0, 0.0)))
        for i, m in enumerate(masks_adapt):
            tasks.append((("k20_adapt", i), m, 20.0, (1.0, 0.0, 0.0)))
        for i, m in enumerate(masks_test):
            tasks.append((("k20_test", i), m, 20.0, (1.0, 0.0, 0.0)))
        for i, m in enumerate(masks_adapt):
            tasks.append((("biax_adapt", i), m, 10.0, (1.0, 0.5, 0.0)))
        for i, m in enumerate(masks_test):
            tasks.append((("biax_test", i), m, 10.0, (1.0, 0.5, 0.0)))

        with Pool(nw) as pool:
            res = dict(pool.map(_worker, tasks, chunksize=8))

        y_rot_e2_RE = np.array([res[("rot_e2", i)]["R_E"] for i in range(300)])
        y_rot_e2_RH = np.array([res[("rot_e2", i)]["RH_b20"] for i in range(300)])
        y_rot_img_RE = np.array([res[("rot_img", i)]["R_E"] for i in range(300)])
        y_rot_img_RH = np.array([res[("rot_img", i)]["RH_b20"] for i in range(300)])

        y_k5_adapt = np.array([res[("k5_adapt", i)]["RH_b20"] for i in range(200)])
        y_k5_test = np.array([res[("k5_test", i)]["RH_b20"] for i in range(300)])
        y_k5_RE_test = np.array([res[("k5_test", i)]["R_E"] for i in range(300)])

        y_k20_adapt = np.array([res[("k20_adapt", i)]["RH_b20"] for i in range(200)])
        y_k20_test = np.array([res[("k20_test", i)]["RH_b20"] for i in range(300)])
        y_k20_RE_test = np.array([res[("k20_test", i)]["R_E"] for i in range(300)])

        y_biax_adapt = np.array([res[("biax_adapt", i)]["RH_b20"] for i in range(200)])
        y_biax_test = np.array([res[("biax_test", i)]["RH_b20"] for i in range(300)])
        y_biax_RE_test = np.array([res[("biax_test", i)]["R_E"] for i in range(300)])

        np.savez_compressed(
            cache_path,
            idx_val=idx_val,
            idx_test=idx_test,
            y_rot_e2_RE=y_rot_e2_RE,
            y_rot_e2_RH=y_rot_e2_RH,
            y_rot_img_RE=y_rot_img_RE,
            y_rot_img_RH=y_rot_img_RH,
            y_k5_adapt=y_k5_adapt,
            y_k5_test=y_k5_test,
            y_k5_RE_test=y_k5_RE_test,
            y_k20_adapt=y_k20_adapt,
            y_k20_test=y_k20_test,
            y_k20_RE_test=y_k20_RE_test,
            y_biax_adapt=y_biax_adapt,
            y_biax_test=y_biax_test,
            y_biax_RE_test=y_biax_RE_test,
        )
        print(f"Saved {cache_path} in {time.time() - t0:.1f}s", flush=True)

    equiv_err_RE = float(np.max(np.abs(y_rot_e2_RE - y_rot_img_RE)))
    equiv_err_RH = float(np.max(np.abs(y_rot_e2_RH - y_rot_img_RH)))
    y_e1_test_RH = D.loc[idx_test, "RH_b20"].values.astype(float)
    y_e1_train_RH = D.loc[idx_train, "RH_b20"].values.astype(float)
    rot_phys_diff_rms = float(np.sqrt(np.mean((y_rot_e2_RH - y_e1_test_RH) ** 2)))
    rot_phys_diff_max = float(np.max(np.abs(y_rot_e2_RH - y_e1_test_RH)))
    print(f"Equivalence max |R(X, e2) - R(rot90(X), e1)|: R_E={equiv_err_RE:.3e}, R_H={equiv_err_RH:.3e}")
    print(f"Physical difference |R_H(X, e2) - R_H(X, e1)|: RMS={rot_phys_diff_rms:.3f}, max={rot_phys_diff_max:.3f}")

    masks_test = masks[idx_test]
    masks_test_rot90 = np.array([np.rot90(m, 1) for m in masks_test])

    rows = []
    for name, (pool, tags) in FAMILIES.items():
        for s, tag in enumerate(tags):
            E = np.load(os.path.join(MD, f"{tag}_emb.npz"))
            Z_all = E["Z"]
            Z_tr = Z_all[idx_train]
            Z_ad = Z_all[idx_val]
            Z_te = Z_all[idx_test]
            Z_te_rot = embed_masks_with_tag(tag, pool, masks_test_rot90, d=8)

            wt = WhiteningTransform().fit(Z_tr)
            Z_tr_w = wt.transform(Z_tr)
            Z_ad_w = wt.transform(Z_ad)
            Z_te_w = wt.transform(Z_te)
            Z_te_rot_w = wt.transform(Z_te_rot)

            # 0. Reference IID test (e_1 load, kappa=10, zero-shot from train)
            nr_iid = omega_ratio(Z_te_w, y_e1_test_RH, np.random.default_rng(100 + s), q=0.01, nperm=200, return_details=True)
            tr_iid = eval_transfer(Z_tr_w, y_e1_train_RH, Z_te_w, y_e1_test_RH)
            rows.append({
                "method": name, "seed": s, "regime": "iid_e1_k10",
                "omega_ratio": float(nr_iid["ratio"]),
                "r2_ridge_zs": tr_iid["r2_ridge"], "r2_mc_zs": tr_iid["r2_mcshane"],
                "cf_max_sd_zs": tr_iid["cf_max_sd"],
            })

            # 1. Rotation check (e_2 load via rot90(X), kappa=10, zero-shot from train)
            nr_rot = omega_ratio(Z_te_rot_w, y_rot_e2_RH, np.random.default_rng(200 + s), q=0.01, nperm=200, return_details=True)
            tr_rot = eval_transfer(Z_tr_w, y_e1_train_RH, Z_te_rot_w, y_rot_e2_RH)
            nr_rot_unrot = omega_ratio(Z_te_w, y_rot_e2_RH, np.random.default_rng(250 + s), q=0.01, nperm=200, return_details=True)
            tr_rot_unrot = eval_transfer(Z_tr_w, y_e1_train_RH, Z_te_w, y_rot_e2_RH)
            rows.append({
                "method": name, "seed": s, "regime": "rot90_e2_k10",
                "omega_ratio": float(nr_rot["ratio"]),
                "r2_ridge_zs": tr_rot["r2_ridge"], "r2_mc_zs": tr_rot["r2_mcshane"],
                "cf_max_sd_zs": tr_rot["cf_max_sd"],
                "unrot_omega_ratio": float(nr_rot_unrot["ratio"]),
                "unrot_r2_ridge": tr_rot_unrot["r2_ridge"],
            })

            # 2-4. OOD-physics regimes: kappa=5, kappa=20, biaxial (1.0, 0.5, 0.0)
            for reg_tag, y_ad, y_te, pseed in [
                ("ood_phys_k5", y_k5_adapt, y_k5_test, 300),
                ("ood_phys_k20", y_k20_adapt, y_k20_test, 400),
                ("ood_phys_biax", y_biax_adapt, y_biax_test, 500),
            ]:
                nr = omega_ratio(Z_te_w, y_te, np.random.default_rng(pseed + s), q=0.01, nperm=200, return_details=True)
                tr50 = eval_transfer(Z_ad_w[:50], y_ad[:50], Z_te_w, y_te)
                tr200 = eval_transfer(Z_ad_w[:200], y_ad[:200], Z_te_w, y_te)
                rows.append({
                    "method": name, "seed": s, "regime": reg_tag,
                    "omega_ratio": float(nr["ratio"]),
                    "r2_ridge_n50": tr50["r2_ridge"], "r2_mc_n50": tr50["r2_mcshane"],
                    "cf_rmse_sd_n50": tr50["cf_rmse_sd"], "cf_max_sd_n50": tr50["cf_max_sd"],
                    "r2_ridge_n200": tr200["r2_ridge"], "r2_mc_n200": tr200["r2_mcshane"],
                    "cf_rmse_sd_n200": tr200["cf_rmse_sd"], "cf_max_sd_n200": tr200["cf_max_sd"],
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(ROOT, "results", "ood_physics_rotation_metrics.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "equivalence_check": {
            "max_abs_diff_RE_e2_vs_rot90_e1": equiv_err_RE,
            "max_abs_diff_RH_e2_vs_rot90_e1": equiv_err_RH,
            "phys_diff_RH_e2_vs_e1_rms": rot_phys_diff_rms,
            "phys_diff_RH_e2_vs_e1_max": rot_phys_diff_max,
            "corr_RH_k10_vs_k5": float(np.corrcoef(y_e1_test_RH, y_k5_test)[0, 1]),
            "corr_RH_k10_vs_k20": float(np.corrcoef(y_e1_test_RH, y_k20_test)[0, 1]),
            "corr_RH_k10_vs_biax": float(np.corrcoef(y_e1_test_RH, y_biax_test)[0, 1]),
        },
        "medians": {},
    }
    for (method, regime), g in df.groupby(["method", "regime"], sort=False):
        num_cols = [c for c in g.columns if c not in ("method", "seed", "regime") and g[c].notna().all()]
        summary["medians"].setdefault(regime, {})[method] = {
            c: float(np.median(g[c].values)) for c in num_cols
        }

    json_path = os.path.join(ROOT, "results", "ood_physics_rotation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {csv_path} and {json_path} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
