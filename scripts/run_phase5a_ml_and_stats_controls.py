"""Phase 5A: Reviewer Controls (ML, Metric Regularization, AE HiFi, Conditional Power/Calibration,
PF Extreme/Relative Errors, and 254x254 Counterexample Grid Convergence).

1. Direct Extreme-Error Control: Top-k Tail MSE (L_tail, k=16, lambda_tail in {1.0, 3.0}, 3 seeds, 60 epochs)
   + Matched-protocol / matched-selection comparison (val CF_max vs val CF_rmse vs val head_mse).
2. Regularized / Truncated Whitening Sweep (RNC vs SBR-CF vs Tail-MSE vs MSE) + Full Exact Barrier
   Curves e*(Lambda)/sigma_R and finite-sample interpolation threshold Lambda_interp/sigma_R.
3. Autoencoder Reconstruction Benchmark: Base rate (1 - f_bar), Pixel Acc, Inclusion IoU, Dice,
   and High-Fidelity ConvAE-HiFi (BCE + Dice, residual skip connections, d=32 and d=128, 3 seeds)
   evaluated on mechanical sufficiency (Omega/Omega_null, R2_McShane, R2_ridge for R_E and RH_b20).
4. Valid Seed-Combined Conditional Permutation Tests (T1-T6, B=2000) + Type-I Error Calibration
   and Graduated Statistical Power Curves on n=150 multi-inclusion phase-field cells.
5. Phase-Field Fracture Extreme Errors, Relative Peak-Stress Errors, Paired Bootstrap CIs, and
   254x254 FFT Grid-Refinement Verification on the Top-10 Consensus Counterexample Pairs.
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

from suffml.data.datasets import load_pilot, load_design, ROOT
from suffml.data.frozen import DESCRIPTOR_STUDY
from suffml.train.train_objectives import train_objective
from suffml.train.train_gate import _data, embed_all, DEV
from suffml.analysis.evaluate import evaluate, ridge_fit_predict, r2
from suffml.metrics import (
    WhiteningTransform,
    Standardizer,
    exact_barrier,
    crossfit_errors,
    select_lambda,
    mcshane_mid,
    omega_ratio,
    holm_adjust,
)
from suffml.metrics.conditional_test import zscore, _swap_perm_factory, _stats_from_d
from suffml.physics.fft_targets import solve_cell
from suffml.physics.critical_distance import CriticalDistance

R_DIR = os.path.join(ROOT, "results")
MD = os.path.join(R_DIR, "models")


# ============================================================================
# 1. Train Top-k Tail MSE Control (k=16, lam in {1.0, 3.0}, 3 seeds, 60 epochs)
# ============================================================================
def run_part1_tail_mse():
    print("=== Part 1: Training Top-k Tail MSE (lam=1,3) & SBR-in (lam=3) Controls (3 seeds each) ===", flush=True)
    for m_type, lam_val, name in [
        ("tail_mse", 1.0, "tailmse1"),
        ("tail_mse", 3.0, "tailmse3"),
        ("sbr_in", 3.0, "sbrin3"),
    ]:
        for seed in (0, 1, 2):
            tag = f"obj_RH_b20_{name}_d8_s{seed}"
            emb_path = os.path.join(MD, f"{tag}_emb.npz")
            if os.path.exists(emb_path):
                print(f"  [Cached] {tag}", flush=True)
                continue
            train_objective(
                m_type,
                target="RH_b20",
                d=8,
                Lambda=16.0,
                Lambda_cf=4.0,
                epochs=60,
                lam=lam_val,
                k=16,
                seed=seed,
                tag=tag,
            )


# ============================================================================
# 2. Regularized Whitening Sweep & Full Exact Barrier Curves e*(Lambda)
# ============================================================================
class RegWhiteningTransform:
    """Regularized ZCA whitening: (Sigma + alpha * tr(Sigma)/d * I)^{-1/2} / sqrt(d)."""

    def __init__(self, alpha: float = 1e-5, trunc_rel: float = 0.0):
        self.alpha = alpha
        self.trunc_rel = trunc_rel
        self.mu = None
        self.W = None
        self.d_eff = None

    def fit(self, Z: np.ndarray):
        Z = np.asarray(Z, dtype=np.float64)
        n, d = Z.shape
        self.mu = Z.mean(axis=0)
        Zc = Z - self.mu
        cov = (Zc.T @ Zc) / max(n - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        tr_mean = max(float(np.mean(eigvals)), 1e-12)
        if self.trunc_rel > 0.0:
            keep = eigvals >= self.trunc_rel * eigvals.max()
            inv_sqrt = np.where(keep, 1.0 / np.sqrt(np.maximum(eigvals, 1e-12)), 0.0)
            self.d_eff = int(np.sum(keep))
            scale = 1.0 / np.sqrt(max(self.d_eff, 1))
        else:
            reg_eigs = eigvals + self.alpha * tr_mean
            inv_sqrt = 1.0 / np.sqrt(np.maximum(reg_eigs, 1e-12))
            self.d_eff = float((eigvals.sum() ** 2) / max(np.sum(eigvals ** 2), 1e-12))
            scale = 1.0 / np.sqrt(d)
        self.W = (eigvecs * inv_sqrt) @ eigvecs.T * scale
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        return (np.asarray(Z, dtype=np.float64) - self.mu) @ self.W


def eval_under_transform(Z: np.ndarray, y: np.ndarray, idx: dict, T_obj, seed_perm: int = 100):
    tr, va, te2, ood = idx["train"], idx["val"], idx["test2"], idx["ood_morphology"]
    sd_tr = float(y[tr].std())
    grid = sd_tr * np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    Ztr = T_obj.transform(Z[tr])
    Zva = T_obj.transform(Z[va])
    Zte = T_obj.transform(Z[te2])
    Zood = T_obj.transform(Z[ood])

    # Select Lambda on validation by RMSE (and also record Lambda selected by CF_max)
    Lam_rmse = select_lambda(Ztr, y[tr], Zva, y[va], grid, criterion="rmse")
    Lam_max = select_lambda(Ztr, y[tr], Zva, y[va], grid, criterion="max")

    cf_va = crossfit_errors(Ztr, y[tr], Zva, y[va], Lam_rmse)
    cf_te = crossfit_errors(Ztr, y[tr], Zte, y[te2], Lam_rmse)
    cf_ood = crossfit_errors(Ztr, y[tr], Zood, y[ood], Lam_rmse)
    cf_te_lmax = crossfit_errors(Ztr, y[tr], Zte, y[te2], Lam_max)

    om_va = float(omega_ratio(Zva, y[va], np.random.default_rng(seed_perm), q=0.01, nperm=200))
    om_te = float(omega_ratio(Zte, y[te2], np.random.default_rng(seed_perm + 1), q=0.01, nperm=200))
    om_ood = float(omega_ratio(Zood, y[ood], np.random.default_rng(seed_perm + 2), q=0.01, nperm=200))

    sd_te = float(y[te2].std())
    sd_ood = float(y[ood].std())
    r2_mc_va = 1.0 - (cf_va["rmse"] / max(float(y[va].std()), 1e-12)) ** 2
    r2_mc_te = 1.0 - (cf_te["rmse"] / max(sd_te, 1e-12)) ** 2
    r2_mc_ood = 1.0 - (cf_ood["rmse"] / max(sd_ood, 1e-12)) ** 2

    r2_ridge_va = r2(y[va], ridge_fit_predict(Ztr, y[tr], Zva))
    r2_ridge_te = r2(y[te2], ridge_fit_predict(Ztr, y[tr], Zte))
    r2_ridge_ood = r2(y[ood], ridge_fit_predict(Ztr, y[tr], Zood))

    return {
        "val_omega_ratio": om_va,
        "val_cf_rmse": float(cf_va["rmse"]),
        "val_cf_max": float(cf_va["max"]),
        "val_r2_mcshane": float(r2_mc_va),
        "val_r2_ridge": float(r2_ridge_va),
        "test2_omega_ratio": om_te,
        "test2_cf_rmse": float(cf_te["rmse"]),
        "test2_cf_max_norm": float(cf_te["max"] / sd_te),
        "test2_cf_max_norm_lmax": float(cf_te_lmax["max"] / sd_te),
        "test2_r2_mcshane": float(r2_mc_te),
        "test2_r2_ridge": float(r2_ridge_te),
        "ood_omega_ratio": om_ood,
        "ood_cf_max_norm": float(cf_ood["max"] / sd_ood),
        "ood_r2_mcshane": float(r2_mc_ood),
        "ood_r2_ridge": float(r2_ridge_ood),
    }


def run_part2_whitening_and_barrier_curves(P, D, idx):
    print("=== Part 2: Regularized Whitening Sweep & Full Barrier Curves e*(Lambda) ===", flush=True)
    y = D["RH_b20"].values.astype(np.float64)
    tr, va, te2, ood = idx["train"], idx["val"], idx["test2"], idx["ood_morphology"]
    sd_tr = float(y[tr].std())

    families = {
        "MSE": ["sup_RH_b20_gem2_d8_s0", "sup_RH_b20_gem2_d8_s1", "sup_RH_b20_gem2_d8_s2"],
        "Tail-MSE (lam=1.0)": ["obj_RH_b20_tailmse1_d8_s0", "obj_RH_b20_tailmse1_d8_s1", "obj_RH_b20_tailmse1_d8_s2"],
        "Tail-MSE (lam=3.0)": ["obj_RH_b20_tailmse3_d8_s0", "obj_RH_b20_tailmse3_d8_s1", "obj_RH_b20_tailmse3_d8_s2"],
        "SBR-in (lam=1.0)": ["obj_RH_b20_sbrin1_d8_s0", "obj_RH_b20_sbrin1_d8_s1", "obj_RH_b20_sbrin1_d8_s2"],
        "SBR-in (lam=3.0)": ["obj_RH_b20_sbrin3_d8_s0", "obj_RH_b20_sbrin3_d8_s1", "obj_RH_b20_sbrin3_d8_s2"],
        "RankSim (gam=1.0)": ["obj_RH_b20_ranksimg1_d8_s0", "obj_RH_b20_ranksimg1_d8_s1", "obj_RH_b20_ranksimg1_d8_s2"],
        "RNC (tau=1.0)": ["obj_RH_b20_rnct1_d8_s0", "obj_RH_b20_rnct1_d8_s1", "obj_RH_b20_rnct1_d8_s2"],
        "SBR-CF (lam=1.0)": ["obj_RH_b20_sbrcf1_d8_s0", "obj_RH_b20_sbrcf1_d8_s1", "obj_RH_b20_sbrcf1_d8_s2"],
        "SBR-CF (lam=3.0)": ["obj_RH_b20_sbrcf3_d8_s0", "obj_RH_b20_sbrcf3_d8_s1", "obj_RH_b20_sbrcf3_d8_s2"],
    }

    reg_configs = [
        ("d_white (alpha=1e-5)", {"alpha": 1e-5, "trunc_rel": 0.0}),
        ("d_reg (alpha=1e-3)", {"alpha": 1e-3, "trunc_rel": 0.0}),
        ("d_reg (alpha=1e-2)", {"alpha": 1e-2, "trunc_rel": 0.0}),
        ("d_reg (alpha=1e-1)", {"alpha": 1e-1, "trunc_rel": 0.0}),
        ("d_trunc (eig>=1e-2)", {"alpha": 1e-5, "trunc_rel": 1e-2}),
        ("d_std (coord)", None),
    ]

    rows_whitening = []
    rows_barriers = []
    lam_mults = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]

    for fam, tags in families.items():
        for s, tag in enumerate(tags):
            E = np.load(os.path.join(MD, f"{tag}_emb.npz"))
            Z = E["Z"]
            Zn = E["Zn"] / np.sqrt(E["Zn"].shape[1])

            for m_name, kw in reg_configs:
                if kw is None:
                    T_obj = Standardizer().fit(Z[tr])
                else:
                    T_obj = RegWhiteningTransform(**kw).fit(Z[tr])
                res = eval_under_transform(Z, y, idx, T_obj, seed_perm=5000 + 30 * s)
                rows_whitening.append({"family": fam, "seed": s, "tag": tag, "metric": m_name, **res})

            # Also compute Val-selected optimal alpha_reg for each model!
            cand_rows = [r for r in rows_whitening if r["tag"] == tag and r["metric"].startswith(("d_white", "d_reg"))]
            best_rmse_row = min(cand_rows, key=lambda r: r["val_cf_rmse"])
            best_max_row = min(cand_rows, key=lambda r: r["val_cf_max"])
            rows_whitening.append({**best_rmse_row, "metric": "d_reg (val-selected by CF_rmse)"})
            rows_whitening.append({**best_max_row, "metric": "d_reg (val-selected by CF_max)"})

            # Full exact barrier curves e*(Lambda)/sigma_R in d_N (certified metric) and d_white
            T_w = WhiteningTransform().fit(Z[tr])
            for space_name, Z_tr_sp, Z_te_sp, Z_ood_sp in [
                ("d_N", Zn[tr], Zn[te2], Zn[ood]),
                ("d_white", T_w.transform(Z[tr]), T_w.transform(Z[te2]), T_w.transform(Z[ood])),
            ]:
                for sp_split, Z_sp, y_sp in [("train", Z_tr_sp, y[tr]), ("test2", Z_te_sp, y[te2]), ("ood", Z_ood_sp, y[ood])]:
                    d_vec = pdist(Z_sp)
                    II, JJ = np.triu_indices(len(y_sp), 1)
                    dy_vec = np.abs(y_sp[II] - y_sp[JJ])
                    pos = d_vec > 1e-12
                    lam_interp = float(np.max(dy_vec[pos] / d_vec[pos]) / sd_tr)
                    b_row = {
                        "family": fam,
                        "seed": s,
                        "space": space_name,
                        "split": sp_split,
                        "Lambda_interp_norm": lam_interp,
                    }
                    for m in lam_mults:
                        Lam_abs = m * sd_tr
                        e_val = float(0.5 * max(0.0, np.max(dy_vec - Lam_abs * d_vec)) / sd_tr)
                        b_row[f"estar_L{m}"] = e_val
                    rows_barriers.append(b_row)
        print(f"  [Part 2] Completed whitening sweep & barrier curves for {fam}", flush=True)

    df_w = pd.DataFrame(rows_whitening)
    df_b = pd.DataFrame(rows_barriers)
    df_w.to_csv(os.path.join(R_DIR, "phase5_whitening_and_tailmse.csv"), index=False)
    df_b.to_csv(os.path.join(R_DIR, "phase5_barrier_curves.csv"), index=False)
    return df_w, df_b


# ============================================================================
# 3. Autoencoder Reconstruction Benchmark & High-Fidelity ConvAE-HiFi
# ============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, c),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class ConvAEHiFi(nn.Module):
    """High-fidelity autoencoder with spatial bottleneck + residual blocks, trained with pos-weighted BCE + soft Dice."""

    def __init__(self, d: int, ch=(32, 64, 128, 128)):
        super().__init__()
        enc, c_in = [], 1
        for c in ch:  # 128 -> 64 -> 32 -> 16 -> 8
            enc += [
                nn.Conv2d(c_in, c, 4, stride=2, padding=1),
                nn.GroupNorm(8, c),
                nn.GELU(),
                ResidualBlock(c),
            ]
            c_in = c
        self.enc = nn.Sequential(*enc)
        self.to_z = nn.Linear(c_in * 8 * 8, d)
        self.from_z = nn.Linear(d, c_in * 8 * 8)
        dec = []
        chs = list(ch[::-1]) + [32]
        for a, b in zip(chs[:-1], chs[1:]):
            dec += [
                ResidualBlock(a),
                nn.ConvTranspose2d(a, b, 4, stride=2, padding=1),
                nn.GroupNorm(8, b),
                nn.GELU(),
            ]
        dec += [nn.Conv2d(chs[-1], 1, 3, padding=1)]
        self.dec = nn.Sequential(*dec)
        self.c_last, self.d = c_in, d

    @staticmethod
    def pad(x):
        return F.pad(x, (0, 1, 0, 1), mode="circular")

    def encode(self, x):
        h = self.enc(self.pad(x))
        return self.to_z(h.flatten(1))

    def decode(self, z):
        h = self.from_z(z).view(-1, self.c_last, 8, 8)
        return self.dec(h)[..., :127, :127]

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def boundary_map(x: torch.Tensor) -> torch.Tensor:
    """Morphological boundary pixels of binary mask x (B x 1 x H x W)."""
    mx = F.max_pool2d(x, 3, stride=1, padding=1)
    mn = -F.max_pool2d(-x, 3, stride=1, padding=1)
    return (mx - mn > 0.5).float()


def eval_recon_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict:
    pr = (logits > 0.0).float()
    acc = (pr == target).float().mean().item()
    base_acc = (1.0 - target.mean()).item()
    inter = (pr * target).sum(dim=(1, 2, 3))
    union = ((pr + target) > 0.5).float().sum(dim=(1, 2, 3))
    denom = pr.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    iou = (inter / (union + 1e-8)).mean().item()
    dice = (2.0 * inter / (denom + 1e-8)).mean().item()

    # Interface / boundary F1 (within 1-pixel tolerance via 3x3 dilation)
    b_pr = boundary_map(pr)
    b_gt = boundary_map(target)
    b_pr_dil = F.max_pool2d(b_pr, 3, stride=1, padding=1)
    b_gt_dil = F.max_pool2d(b_gt, 3, stride=1, padding=1)
    prec = ((b_pr * b_gt_dil).sum(dim=(1, 2, 3)) / (b_pr.sum(dim=(1, 2, 3)) + 1e-8)).mean().item()
    rec = ((b_gt * b_pr_dil).sum(dim=(1, 2, 3)) / (b_gt.sum(dim=(1, 2, 3)) + 1e-8)).mean().item()
    bf1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {
        "pixel_acc": float(acc),
        "base_matrix_acc": float(base_acc),
        "inclusion_iou": float(iou),
        "inclusion_dice": float(dice),
        "boundary_f1": float(bf1),
        "pred_vol_frac": float(pr.mean().item()),
        "true_vol_frac": float(target.mean().item()),
    }


def train_and_eval_ae_hifi(P, D, idx):
    print("=== Part 3: Autoencoder Reconstruction Benchmark & ConvAE-HiFi ===", flush=True)
    from suffml.models.autoencoder import ConvAE

    X = torch.from_numpy(P["mask"].astype(np.float32))[:, None]
    tr, va, te2, ood = idx["train"], idx["val"], idx["test2"], idx["ood_morphology"]
    y_RE = D["R_E"].values.astype(np.float64)
    y_RH = D["RH_b20"].values.astype(np.float64)

    ae_rows = []

    # 1. Evaluate existing ConvAE (d=8, d=32, 3 seeds)
    for d_val in (8, 32):
        for s in range(3):
            tag = f"ae_d{d_val}_s{s}"
            model = ConvAE(d_val).to(DEV)
            model.load_state_dict(torch.load(os.path.join(MD, f"{tag}.pt"), map_location=DEV, weights_only=True))
            model.eval()
            with torch.no_grad():
                logits_va = torch.cat([model(X[va][a:a+64].to(DEV))[0].cpu() for a in range(0, len(va), 64)])
                logits_te2 = torch.cat([model(X[te2][a:a+64].to(DEV))[0].cpu() for a in range(0, len(te2), 64)])
            rec_va = eval_recon_metrics(logits_va, X[va])
            rec_te2 = eval_recon_metrics(logits_te2, X[te2])
            Z = np.load(os.path.join(MD, f"{tag}_emb.npz"))["Z"]
            T_w = WhiteningTransform().fit(Z[tr])
            res_RH = eval_under_transform(Z, y_RH, idx, T_w, seed_perm=7000 + 10 * d_val + s)
            res_RE = eval_under_transform(Z, y_RE, idx, T_w, seed_perm=7500 + 10 * d_val + s)
            ae_rows.append({
                "model": f"ConvAE-BCE (d={d_val})",
                "d": d_val,
                "seed": s,
                "val_pixel_acc": rec_va["pixel_acc"],
                "val_iou": rec_va["inclusion_iou"],
                "val_dice": rec_va["inclusion_dice"],
                "val_bf1": rec_va["boundary_f1"],
                "test2_pixel_acc": rec_te2["pixel_acc"],
                "test2_base_acc": rec_te2["base_matrix_acc"],
                "test2_iou": rec_te2["inclusion_iou"],
                "test2_dice": rec_te2["inclusion_dice"],
                "test2_bf1": rec_te2["boundary_f1"],
                "RH_test2_omega": res_RH["test2_omega_ratio"],
                "RH_test2_r2_mcshane": res_RH["test2_r2_mcshane"],
                "RH_test2_r2_ridge": res_RH["test2_r2_ridge"],
                "RH_ood_omega": res_RH["ood_omega_ratio"],
                "RE_test2_omega": res_RE["test2_omega_ratio"],
                "RE_test2_r2_mcshane": res_RE["test2_r2_mcshane"],
                "RE_test2_r2_ridge": res_RE["test2_r2_ridge"],
            })

    # 2. Train & evaluate ConvAE-HiFi (pos-weighted BCE + Dice + residual blocks) at d=32 and d=128 (3 seeds)
    pos_weight = torch.tensor([4.0], device=DEV)
    for d_val in (32, 128):
        for s in range(3):
            tag = f"aehifi_d{d_val}_s{s}"
            pt_file = os.path.join(MD, f"{tag}.pt")
            emb_file = os.path.join(MD, f"{tag}_emb.npz")
            model = ConvAEHiFi(d_val).to(DEV)
            if os.path.exists(pt_file) and os.path.exists(emb_file):
                model.load_state_dict(torch.load(pt_file, map_location=DEV, weights_only=True))
                Z = np.load(emb_file)["Z"]
            else:
                torch.manual_seed(100 + s); np.random.seed(100 + s)
                epochs = 80
                bs = 64
                opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
                steps = epochs * (len(tr) // bs)
                sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=steps, pct_start=0.1)
                Xtr_gpu = X[tr].to(DEV)
                for ep in range(epochs):
                    model.train()
                    perm = torch.randperm(len(Xtr_gpu), device=DEV)
                    for a in range(0, (len(perm) // bs) * bs, bs):
                        xb = Xtr_gpu[perm[a:a + bs]]
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
                            logits, _ = model(xb)
                        logits_f = logits.float()
                        bce = F.binary_cross_entropy_with_logits(logits_f, xb, pos_weight=pos_weight)
                        pr_sig = torch.sigmoid(logits_f)
                        inter = (pr_sig * xb).sum(dim=(1, 2, 3))
                        dice_l = 1.0 - (2.0 * inter + 1.0) / (pr_sig.sum(dim=(1, 2, 3)) + xb.sum(dim=(1, 2, 3)) + 1.0)
                        loss = bce + dice_l.mean()
                        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                model.eval()
                Z = embed_all(model.encode, X, bs=64)
                torch.save(model.state_dict(), pt_file)
                np.savez_compressed(emb_file, Z=Z)
                del Xtr_gpu, opt, sched
                if DEV == "cuda":
                    torch.cuda.empty_cache()

            model.eval()
            with torch.no_grad():
                logits_va = torch.cat([model(X[va][a:a+64].to(DEV))[0].cpu() for a in range(0, len(va), 64)])
                logits_te2 = torch.cat([model(X[te2][a:a+64].to(DEV))[0].cpu() for a in range(0, len(te2), 64)])
            rec_va = eval_recon_metrics(logits_va, X[va])
            rec_te2 = eval_recon_metrics(logits_te2, X[te2])
            T_w = RegWhiteningTransform(alpha=1e-2).fit(Z[tr])
            res_RH = eval_under_transform(Z, y_RH, idx, T_w, seed_perm=8000 + 10 * d_val + s)
            res_RE = eval_under_transform(Z, y_RE, idx, T_w, seed_perm=8500 + 10 * d_val + s)
            print(
                f"  [{tag}] te2 Acc={rec_te2['pixel_acc']:.4f} IoU={rec_te2['inclusion_iou']:.4f} "
                f"Dice={rec_te2['inclusion_dice']:.4f} | RH Omega={res_RH['test2_omega_ratio']:.3f} "
                f"R2_McS={res_RH['test2_r2_mcshane']:.3f}",
                flush=True,
            )
            ae_rows.append({
                "model": f"ConvAE-HiFi (d={d_val})",
                "d": d_val,
                "seed": s,
                "val_pixel_acc": rec_va["pixel_acc"],
                "val_iou": rec_va["inclusion_iou"],
                "val_dice": rec_va["inclusion_dice"],
                "val_bf1": rec_va["boundary_f1"],
                "test2_pixel_acc": rec_te2["pixel_acc"],
                "test2_base_acc": rec_te2["base_matrix_acc"],
                "test2_iou": rec_te2["inclusion_iou"],
                "test2_dice": rec_te2["inclusion_dice"],
                "test2_bf1": rec_te2["boundary_f1"],
                "RH_test2_omega": res_RH["test2_omega_ratio"],
                "RH_test2_r2_mcshane": res_RH["test2_r2_mcshane"],
                "RH_test2_r2_ridge": res_RH["test2_r2_ridge"],
                "RH_ood_omega": res_RH["ood_omega_ratio"],
                "RE_test2_omega": res_RE["test2_omega_ratio"],
                "RE_test2_r2_mcshane": res_RE["test2_r2_mcshane"],
                "RE_test2_r2_ridge": res_RE["test2_r2_ridge"],
            })

    df_ae = pd.DataFrame(ae_rows)
    df_ae.to_csv(os.path.join(R_DIR, "phase5_ae_reconstruction.csv"), index=False)
    return df_ae


# ============================================================================
# 4. Seed-Combined Conditional Tests (T1-T6) + Power & Type-I Calibration
# ============================================================================
def cond_perm_seed_combined(Xk: np.ndarray, Z_seeds: list[np.ndarray], R: np.ndarray, rng: np.random.Generator,
                            q: float = 0.01, B: int = 2000, k_nn: int = 5) -> dict:
    """Exact permutation test using a pre-specified seed-aggregated test statistic:
    Under each local swap b in {1..B} of the conditioning-neighbor graph on Xk, all 3 seeds are permuted
    synchronously with the same permutation pi_b, and the seed-mean Omega_bar(q) = (1/S) sum_s Omega_s(q)
    is compared to its observed value. Also evaluates the concatenated ensemble embedding."""
    Xk = np.asarray(Xk, float)
    R = np.asarray(R, float)
    n = len(R)
    II, JJ = np.triu_indices(n, 1)
    dR = np.abs(R[II] - R[JJ])
    nq = int(np.ceil(q * len(dR) - 1e-9))
    draw_perm = _swap_perm_factory(Xk, k_nn=k_nn)

    Zk = zscore(Xk)
    dk2 = pdist(Zk, metric="sqeuclidean") if Zk.shape[1] else np.zeros(len(II))

    De2_mats = [squareform(pdist(zscore(Z), metric="sqeuclidean")) for Z in Z_seeds]
    obs_seeds = [_stats_from_d(dk2 + M[II, JJ], dR, nq)[0] for M in De2_mats]
    obs_mean_stat = float(np.mean(obs_seeds))

    # Also ensemble distance (average squared standardized latent distance across seeds)
    De2_ens = np.mean(De2_mats, axis=0)
    obs_ens_stat = float(_stats_from_d(dk2 + De2_ens[II, JJ], dR, nq)[0])

    nl_seeds = np.empty((B, len(Z_seeds)))
    nl_ens = np.empty(B)
    for b in range(B):
        perm = draw_perm(rng)
        pII, pJJ = perm[II], perm[JJ]
        for s, M in enumerate(De2_mats):
            nl_seeds[b, s] = _stats_from_d(dk2 + M[pII, pJJ], dR, nq)[0]
        nl_ens[b] = _stats_from_d(dk2 + De2_ens[pII, pJJ], dR, nq)[0]

    nl_mean_stat = nl_seeds.mean(axis=1)
    cnull_bar = float(nl_mean_stat.mean())
    ratio_bar = obs_mean_stat / max(cnull_bar, 1e-12)
    p_combined = float((1.0 + np.sum(nl_mean_stat <= obs_mean_stat + 1e-12)) / (1.0 + B))

    cnull_ens = float(nl_ens.mean())
    ratio_ens = obs_ens_stat / max(cnull_ens, 1e-12)
    p_ens = float((1.0 + np.sum(nl_ens <= obs_ens_stat + 1e-12)) / (1.0 + B))

    p_per_seed = [float((1.0 + np.sum(nl_seeds[:, s] <= obs_seeds[s] + 1e-12)) / (1.0 + B)) for s in range(len(Z_seeds))]
    ratio_per_seed = [float(obs_seeds[s] / max(nl_seeds[:, s].mean(), 1e-12)) for s in range(len(Z_seeds))]

    return {
        "ratio_combined": ratio_bar,
        "p_combined": p_combined,
        "ratio_ens": ratio_ens,
        "p_ens": p_ens,
        "ratio_s0": ratio_per_seed[0],
        "ratio_s1": ratio_per_seed[1],
        "ratio_s2": ratio_per_seed[2],
        "p_s0": p_per_seed[0],
        "p_s1": p_per_seed[1],
        "p_s2": p_per_seed[2],
    }


def run_part4_conditional_and_power():
    print("=== Part 4: Seed-Combined Conditional Tests (T1-T6) & Power/Calibration ===", flush=True)
    from suffml.train.train_gate import Supervised

    des = load_design()
    D_des = des["table"]
    M_des = des["mask"]
    inc_n1_mask = ((D_des["family"] != "laminate") & (D_des["n"] > 1)).values
    D_sub = D_des[inc_n1_mask].reset_index(drop=True)
    M_sub = M_des[inc_n1_mask]

    E1_COLS = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]
    E1_sub = D_sub[E1_COLS].values.astype(float)
    E1J_sub = D_sub[E1_COLS + ["J_1.5"]].values.astype(float)

    fams = {
        "MSE (RH_b20)": ("gem", ["sup_RH_b20_gem2_d8_s0", "sup_RH_b20_gem2_d8_s1", "sup_RH_b20_gem2_d8_s2"]),
        "Tail-MSE (lam=1.0)": ("gem", ["obj_RH_b20_tailmse1_d8_s0", "obj_RH_b20_tailmse1_d8_s1", "obj_RH_b20_tailmse1_d8_s2"]),
        "Tail-MSE (lam=3.0)": ("gem", ["obj_RH_b20_tailmse3_d8_s0", "obj_RH_b20_tailmse3_d8_s1", "obj_RH_b20_tailmse3_d8_s2"]),
        "RankSim (gam=1.0)": ("gem", ["obj_RH_b20_ranksimg1_d8_s0", "obj_RH_b20_ranksimg1_d8_s1", "obj_RH_b20_ranksimg1_d8_s2"]),
        "RNC (tau=1.0)": ("gem", ["obj_RH_b20_rnct1_d8_s0", "obj_RH_b20_rnct1_d8_s1", "obj_RH_b20_rnct1_d8_s2"]),
        "SBR-CF (lam=1.0)": ("gem", ["obj_RH_b20_sbrcf1_d8_s0", "obj_RH_b20_sbrcf1_d8_s1", "obj_RH_b20_sbrcf1_d8_s2"]),
        "SBR-CF (lam=3.0)": ("gem", ["obj_RH_b20_sbrcf3_d8_s0", "obj_RH_b20_sbrcf3_d8_s1", "obj_RH_b20_sbrcf3_d8_s2"]),
    }

    X_sub_t = torch.from_numpy(M_sub[:, None].astype(np.float32))
    emb_sub = {}
    for fam, (pool, tags) in fams.items():
        Z_list = []
        for tag in tags:
            info = json.load(open(os.path.join(MD, f"{tag}.json")))
            model = Supervised(8, pool, info.get("Lambda", 16.0)).to(DEV)
            model.load_state_dict(torch.load(os.path.join(MD, f"{tag}.pt"), map_location=DEV, weights_only=True))
            model.eval()
            Z_list.append(embed_all(model.embed, X_sub_t))
        emb_sub[fam] = Z_list

    PF_TESTS = [
        ("T1", "R_E|E1", "R_E", "E1", E1_sub),
        ("T2", "RH_b20|E1", "RH_b20", "E1", E1_sub),
        ("T3", "RH_b20|E1+J1.5", "RH_b20", "E1+J_1.5", E1J_sub),
        ("T4", "R_sigma|E1", "R_sigma", "E1", E1_sub),
        ("T5", "R_sigma|E1+J1.5", "R_sigma", "E1+J_1.5", E1J_sub),
        ("T6", "R_path|E1+J1.5", "R_path", "E1+J_1.5", E1J_sub),
    ]

    comb_rows = []
    for fam, Z_list in emb_sub.items():
        f_rows = []
        for idx_t, (code, label, t_col, base_name, X_base) in enumerate(PF_TESTS):
            y_vec = D_sub[t_col].values.astype(float)
            res = cond_perm_seed_combined(
                X_base, Z_list, y_vec, np.random.default_rng(9000 + idx_t), q=0.01, B=2000, k_nn=5
            )
            f_rows.append({"family": fam, "test_code": code, "label": label, "target": t_col, "baseline": base_name, **res})
        df_f = pd.DataFrame(f_rows)
        df_f["holm_p_combined"] = holm_adjust(df_f["p_combined"].values)
        df_f["holm_p_ens"] = holm_adjust(df_f["p_ens"].values)
        df_f["holm_p_s0"] = holm_adjust(df_f["p_s0"].values)
        df_f["holm_p_s1"] = holm_adjust(df_f["p_s1"].values)
        df_f["holm_p_s2"] = holm_adjust(df_f["p_s2"].values)
        df_f["n_seeds_holm_sig"] = (
            (df_f["holm_p_s0"] < 0.05).astype(int)
            + (df_f["holm_p_s1"] < 0.05).astype(int)
            + (df_f["holm_p_s2"] < 0.05).astype(int)
        )
        comb_rows.append(df_f)
    df_comb = pd.concat(comb_rows, ignore_index=True)
    df_comb.to_csv(os.path.join(R_DIR, "phase5_cond_test_combined.csv"), index=False)

    # Graduated Power & Type-I Error Calibration for T5 (R_sigma | E1 + J_1.5, n=150)
    # Using SBR-CF (lam=1.0) embeddings Z_list
    Z_sbrcf = emb_sub["SBR-CF (lam=1.0)"]
    Zk = zscore(E1J_sub)
    Phi = np.column_stack([np.ones(len(Zk)), Zk])
    R_sig = D_sub["R_sigma"].values.astype(float)
    beta_base = np.linalg.lstsq(Phi, R_sig, rcond=None)[0]
    R_hat_base = Phi @ beta_base
    res_R = R_sig - R_hat_base
    sd_R = float(R_sig.std())
    sd_res = float(res_R.std())

    # Extract top principal component of Z_sbrcf orthogonal to D0 = (E1, J_1.5)
    Z_concat = np.hstack([zscore(Z) for Z in Z_sbrcf])
    Z_orth = Z_concat - Phi @ np.linalg.lstsq(Phi, Z_concat, rcond=None)[0]
    u_orth = np.linalg.svd(Z_orth, full_matrices=False)[0][:, 0]
    u_orth = (u_orth - u_orth.mean()) / u_orth.std()

    # Also nonlinear local pair-collision signal in Z_sbrcf (matching what Omega(q=1%) detects)
    power_rows = []
    etas = [0.0, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]
    n_mc = 150
    B_pow = 300
    rng_pow = np.random.default_rng(202611)

    for eta_val in etas:
        p_comb_list = []
        r2_inc_list = []
        for rep in range(n_mc):
            # Permuted residual noise + synthetic orthogonal latent signal
            eps_draw = rng_pow.normal(0.0, sd_res, size=len(R_sig))
            R_syn = R_hat_base + eta_val * sd_R * u_orth + eps_draw
            var_share_total = float((eta_val * sd_R) ** 2 / np.var(R_syn))
            var_share_resid = float((eta_val * sd_R) ** 2 / ((eta_val * sd_R) ** 2 + sd_res ** 2))
            res_mc = cond_perm_seed_combined(
                E1J_sub, Z_sbrcf, R_syn, rng_pow, q=0.01, B=B_pow, k_nn=5
            )
            p_comb_list.append(res_mc["p_combined"])
            r2_inc_list.append((var_share_total, var_share_resid, res_mc["ratio_combined"]))
        p_arr = np.array(p_comb_list)
        power_rows.append({
            "eta_sd_R": eta_val,
            "inc_var_share_total": float(np.mean([x[0] for x in r2_inc_list])),
            "inc_var_share_resid": float(np.mean([x[1] for x in r2_inc_list])),
            "mean_omega_ratio": float(np.mean([x[2] for x in r2_inc_list])),
            "reject_rate_alpha05": float(np.mean(p_arr <= 0.05)),
            "reject_rate_alpha10": float(np.mean(p_arr <= 0.10)),
            "reject_rate_holm6_alpha05": float(np.mean(p_arr <= (0.05 / 6.0))),
            "median_p": float(np.median(p_arr)),
        })
        print(
            f"  [Power T5] eta={eta_val:.2f} (dR2_tot={power_rows[-1]['inc_var_share_total']*100:.1f}%, "
            f"dR2_res={power_rows[-1]['inc_var_share_resid']*100:.1f}%): "
            f"pow(0.05)={power_rows[-1]['reject_rate_alpha05']:.3f}, "
            f"pow_Holm(0.05)={power_rows[-1]['reject_rate_holm6_alpha05']:.3f}",
            flush=True,
        )

    df_pow = pd.DataFrame(power_rows)
    df_pow.to_csv(os.path.join(R_DIR, "phase5_cond_power_t5.csv"), index=False)
    return df_comb, df_pow


# ============================================================================
# 5. Phase-Field Fracture Extreme & Relative Stress Errors + 254x254 Convergence
# ============================================================================
def solve_254_worker(args):
    cell_id, mask_127 = args
    # Upsample binary mask exactly by 2x2 Kronecker product (preserving exact continuum inclusion geometry!)
    mask_254 = np.repeat(np.repeat(mask_127, 2, axis=0), 2, axis=1)
    st_127 = solve_cell(mask_127, keep_fields=False)["scalars"]
    st_254 = solve_cell(mask_254, keep_fields=False)["scalars"]
    return {
        "cell_id": int(cell_id),
        "RE_127": float(st_127["R_E"]),
        "RE_254": float(st_254["R_E"]),
        "RH20_127": float(st_127["RH_b20"]),
        "RH20_254": float(st_254["RH_b20"]),
        "RH40_127": float(st_127["RH_b40"]),
        "RH40_254": float(st_254["RH_b40"]),
        "Jgap_m_127": float(st_127["Jgap_m"]),
        "Jgap_m_254": float(st_254["Jgap_m"]),
    }


def run_part5_pf_extremes_and_254_convergence(P):
    print("=== Part 5: Phase-Field Extreme/Relative Stress Errors & 254x254 Counterexample Convergence ===", flush=True)
    from suffml.train.train_gate import Supervised

    def r2_score(y_t: np.ndarray, y_p: np.ndarray) -> float:
        return float(1.0 - np.sum((y_t - y_p) ** 2) / np.sum((y_t - y_t.mean()) ** 2))

    des = load_design()
    D_des = des["table"]
    M_des = des["mask"]
    E1_COLS = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]

    blind_masks = np.load(os.path.join(DESCRIPTOR_STUDY, "blind", "blind_masks.npz"))["mask"]
    blind_pf = pd.read_csv(os.path.join(DESCRIPTOR_STUDY, "blind", "blind_pf.csv"))
    blind_pf = blind_pf[blind_pf["separated"]].reset_index(drop=True)
    if "J_1.5" not in blind_pf.columns:
        cd = CriticalDistance(N=127, radii_ell=(0.0, 1.5))
        j_rows = [dict(bid=int(b), **cd(blind_masks[int(b)])) for b in blind_pf["bid"]]
        blind_pf = blind_pf.merge(pd.DataFrame(j_rows), on="bid")
    blind_pf["RH_b20"] = blind_pf["lnB"]

    tcd_masks = np.load(os.path.join(DESCRIPTOR_STUDY, "tcd_frozen", "tcd_masks.npz"))["mask"]
    tcd_pf = pd.read_csv(os.path.join(DESCRIPTOR_STUDY, "tcd_frozen", "tcd_pf.csv"))
    tcd_pf = tcd_pf[tcd_pf["separated"]].reset_index(drop=True)
    tcd_pf["RH_b20"] = tcd_pf["lnB"]

    inc_mask = (D_des["family"] != "laminate").values
    TR = D_des[inc_mask].reset_index(drop=True)
    cells_tr = TR["cell"].values
    y_sigma_tr = TR["R_sigma"].values.astype(float)

    mask_B = (blind_pf["set"] == "A").values
    mask_O = (blind_pf["set"] == "O").values
    mask_C1 = (tcd_pf["set"] == "C1").values
    mask_C2 = (tcd_pf["set"] == "C2").values

    y_true = {
        "Set_B": blind_pf.loc[mask_B, "R_sigma"].values.astype(float),
        "Set_O": blind_pf.loc[mask_O, "R_sigma"].values.astype(float),
        "Set_C1": tcd_pf.loc[mask_C1, "R_sigma"].values.astype(float),
        "Set_C2": tcd_pf.loc[mask_C2, "R_sigma"].values.astype(float),
    }

    def get_oof_and_preds(Xtr, Xte_dict, alphas=(1e-3, 1e-2, 1e-1, 1.0, 5.0, 20.0)):
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Ztr = (Xtr - mu) / sd
        uniq_cells = np.unique(cells_tr)
        rng = np.random.default_rng(42)
        perm_cells = rng.permutation(uniq_cells)
        folds = np.array_split(perm_cells, 5)
        best_alpha, best_mse, best_oof = alphas[0], np.inf, None
        for a in alphas:
            oof = np.zeros_like(y_sigma_tr, dtype=float)
            for f_cells in folds:
                val_m = np.isin(cells_tr, f_cells)
                tr_m = ~val_m
                A = np.column_stack([np.ones(tr_m.sum()), Ztr[tr_m]])
                reg = a * np.eye(A.shape[1]); reg[0, 0] = 0.0
                c = np.linalg.solve(A.T @ A + reg, A.T @ y_sigma_tr[tr_m])
                B_mat = np.column_stack([np.ones(val_m.sum()), Ztr[val_m]])
                oof[val_m] = B_mat @ c
            mse = float(np.mean((y_sigma_tr - oof) ** 2))
            if mse < best_mse:
                best_mse, best_alpha, best_oof = mse, a, oof
        A = np.column_stack([np.ones(len(Ztr)), Ztr])
        reg = best_alpha * np.eye(A.shape[1]); reg[0, 0] = 0.0
        c = np.linalg.solve(A.T @ A + reg, A.T @ y_sigma_tr)
        preds = {"Design_CV": best_oof}
        for k, Xte in Xte_dict.items():
            Zte = (Xte - mu) / sd
            preds[k] = np.column_stack([np.ones(len(Zte)), Zte]) @ c
        return preds

    def summarize_err(yt, yp):
        ae = np.abs(yt - yp)
        rel_sig = np.abs(np.exp(yp - yt) - 1.0) * 100.0  # relative error in peak stress (%)
        return {
            "r2": r2_score(yt, yp),
            "rmse": float(np.sqrt(np.mean(ae ** 2))),
            "max_err": float(np.max(ae)),
            "p90_err": float(np.percentile(ae, 90)),
            "mean_rel_stress_pct": float(np.mean(rel_sig)),
            "max_rel_stress_pct": float(np.max(rel_sig)),
            "worst_idx": int(np.argmax(ae)),
        }

    # Physics baselines
    phys_specs = {
        "E1": E1_COLS,
        "E1 + RH_b20": E1_COLS + ["RH_b20"],
        "E1 + J_1.5": E1_COLS + ["J_1.5"],
    }
    pf_ext_rows = []
    stored_preds = {}

    for p_name, cols in phys_specs.items():
        Xtr = TR[cols].values.astype(float)
        Xte = {
            "Set_B": blind_pf.loc[mask_B, cols].values.astype(float),
            "Set_O": blind_pf.loc[mask_O, cols].values.astype(float),
            "Set_C1": tcd_pf.loc[mask_C1, cols].values.astype(float),
            "Set_C2": tcd_pf.loc[mask_C2, cols].values.astype(float),
        }
        preds = get_oof_and_preds(Xtr, Xte)
        stored_preds[(p_name, 0)] = preds
        row = {"rep": p_name, "seed": 0}
        for sp_name, yt in [("Design_CV", y_sigma_tr), *y_true.items()]:
            st = summarize_err(yt, preds[sp_name])
            for k, v in st.items():
                row[f"{sp_name}_{k}"] = v
        pf_ext_rows.append(row)

    # Learned models (alone and hybrid with E1 + J_1.5)
    fams = {
        "MSE (RH_b20)": ("gem", ["sup_RH_b20_gem2_d8_s0", "sup_RH_b20_gem2_d8_s1", "sup_RH_b20_gem2_d8_s2"]),
        "Tail-MSE (lam=1.0)": ("gem", ["obj_RH_b20_tailmse1_d8_s0", "obj_RH_b20_tailmse1_d8_s1", "obj_RH_b20_tailmse1_d8_s2"]),
        "Tail-MSE (lam=3.0)": ("gem", ["obj_RH_b20_tailmse3_d8_s0", "obj_RH_b20_tailmse3_d8_s1", "obj_RH_b20_tailmse3_d8_s2"]),
        "RankSim (gam=1.0)": ("gem", ["obj_RH_b20_ranksimg1_d8_s0", "obj_RH_b20_ranksimg1_d8_s1", "obj_RH_b20_ranksimg1_d8_s2"]),
        "RNC (tau=1.0)": ("gem", ["obj_RH_b20_rnct1_d8_s0", "obj_RH_b20_rnct1_d8_s1", "obj_RH_b20_rnct1_d8_s2"]),
        "SBR-CF (lam=1.0)": ("gem", ["obj_RH_b20_sbrcf1_d8_s0", "obj_RH_b20_sbrcf1_d8_s1", "obj_RH_b20_sbrcf1_d8_s2"]),
        "SBR-CF (lam=3.0)": ("gem", ["obj_RH_b20_sbrcf3_d8_s0", "obj_RH_b20_sbrcf3_d8_s1", "obj_RH_b20_sbrcf3_d8_s2"]),
    }

    E1J_tr = TR[E1_COLS + ["J_1.5"]].values.astype(float)
    E1J_te = {
        "Set_B": blind_pf.loc[mask_B, E1_COLS + ["J_1.5"]].values.astype(float),
        "Set_O": blind_pf.loc[mask_O, E1_COLS + ["J_1.5"]].values.astype(float),
        "Set_C1": tcd_pf.loc[mask_C1, E1_COLS + ["J_1.5"]].values.astype(float),
        "Set_C2": tcd_pf.loc[mask_C2, E1_COLS + ["J_1.5"]].values.astype(float),
    }

    X_des_t = torch.from_numpy(M_des[:, None].astype(np.float32))
    X_blind_t = torch.from_numpy(blind_masks[blind_pf["bid"].values.astype(int)][:, None].astype(np.float32))
    X_tcd_t = torch.from_numpy(tcd_masks[tcd_pf["cid"].values.astype(int)][:, None].astype(np.float32))

    for fam, (pool, tags) in fams.items():
        for s, tag in enumerate(tags):
            info = json.load(open(os.path.join(MD, f"{tag}.json")))
            model = Supervised(8, pool, info.get("Lambda", 16.0)).to(DEV)
            model.load_state_dict(torch.load(os.path.join(MD, f"{tag}.pt"), map_location=DEV, weights_only=True))
            model.eval()
            Z_des = embed_all(model.embed, X_des_t)[inc_mask]
            Z_bl = embed_all(model.embed, X_blind_t)
            Z_tc = embed_all(model.embed, X_tcd_t)

            Z_te = {
                "Set_B": Z_bl[mask_B],
                "Set_O": Z_bl[mask_O],
                "Set_C1": Z_tc[mask_C1],
                "Set_C2": Z_tc[mask_C2],
            }
            for mode_name, Xtr_m, Xte_m in [
                (f"{fam} alone", Z_des, Z_te),
                (f"E1 + J_1.5 + {fam}", np.hstack([E1J_tr, Z_des]), {k: np.hstack([E1J_te[k], Z_te[k]]) for k in Z_te}),
            ]:
                preds = get_oof_and_preds(Xtr_m, Xte_m)
                stored_preds[(mode_name, s)] = preds
                row = {"rep": mode_name, "seed": s}
                for sp_name, yt in [("Design_CV", y_sigma_tr), *y_true.items()]:
                    st = summarize_err(yt, preds[sp_name])
                    for k, v in st.items():
                        row[f"{sp_name}_{k}"] = v
                pf_ext_rows.append(row)

    df_pf_ext = pd.DataFrame(pf_ext_rows)
    df_pf_ext.to_csv(os.path.join(R_DIR, "phase5_pf_extreme_metrics.csv"), index=False)

    # Compute paired cell-level bootstrap CIs on Set_C2 and Set_B (seed-averaged predictions)
    rng_b = np.random.default_rng(777)
    paired_comparisons = {}
    for sp_name, yt in [("Design_CV", y_sigma_tr), *y_true.items()]:
        p_e1j = stored_preds[("E1 + J_1.5", 0)][sp_name]
        p_mse_alone = np.mean([stored_preds[("MSE (RH_b20) alone", s)][sp_name] for s in range(3)], axis=0)
        p_sbr1_alone = np.mean([stored_preds[("SBR-CF (lam=1.0) alone", s)][sp_name] for s in range(3)], axis=0)
        p_hyb_mse = np.mean([stored_preds[("E1 + J_1.5 + MSE (RH_b20)", s)][sp_name] for s in range(3)], axis=0)
        p_hyb_sbr1 = np.mean([stored_preds[("E1 + J_1.5 + SBR-CF (lam=1.0)", s)][sp_name] for s in range(3)], axis=0)

        # Paired squared error reductions
        se_e1j = (yt - p_e1j) ** 2
        se_hyb_mse = (yt - p_hyb_mse) ** 2
        se_hyb_sbr1 = (yt - p_hyb_sbr1) ** 2
        se_mse_alone = (yt - p_mse_alone) ** 2
        se_sbr1_alone = (yt - p_sbr1_alone) ** 2

        n_s = len(yt)
        boot_idx = rng_b.integers(0, n_s, size=(2000, n_s))
        diff_hyb_vs_e1j = np.mean(se_hyb_sbr1[boot_idx] - se_e1j[boot_idx], axis=1)
        diff_hyb_sbr_vs_mse = np.mean(se_hyb_sbr1[boot_idx] - se_hyb_mse[boot_idx], axis=1)
        diff_alone_sbr_vs_mse = np.mean(se_sbr1_alone[boot_idx] - se_mse_alone[boot_idx], axis=1)

        paired_comparisons[sp_name] = {
            "n": n_s,
            "dMSE_hyb_sbr1_vs_e1j_mean": float(np.mean(se_hyb_sbr1 - se_e1j)),
            "dMSE_hyb_sbr1_vs_e1j_ci90": [float(np.percentile(diff_hyb_vs_e1j, 5)), float(np.percentile(diff_hyb_vs_e1j, 95))],
            "dMSE_hyb_sbr1_vs_hyb_mse_mean": float(np.mean(se_hyb_sbr1 - se_hyb_mse)),
            "dMSE_hyb_sbr1_vs_hyb_mse_ci90": [float(np.percentile(diff_hyb_sbr_vs_mse, 5)), float(np.percentile(diff_hyb_sbr_vs_mse, 95))],
            "dMSE_alone_sbr1_vs_mse_mean": float(np.mean(se_sbr1_alone - se_mse_alone)),
            "dMSE_alone_sbr1_vs_mse_ci90": [float(np.percentile(diff_alone_sbr_vs_mse, 5)), float(np.percentile(diff_alone_sbr_vs_mse, 95))],
        }

    # 254x254 FFT Grid Convergence on Top-10 Consensus MSE Counterexamples on test2
    df_top10 = pd.read_csv(os.path.join(R_DIR, "counterexamples_top10.csv"))
    df_t2 = df_top10[df_top10["set"] == "test2"].reset_index(drop=True)
    uniq_ce_cells = sorted(set(df_t2["cell_i"].tolist() + df_t2["cell_j"].tolist()))
    print(f"  [254x254 Convergence] Solving {len(uniq_ce_cells)} counterexample cells at 127x127 and 254x254...", flush=True)
    tasks_254 = [(cid, P["mask"][int(cid)]) for cid in uniq_ce_cells]
    with ProcessPoolExecutor(max_workers=min(19, len(tasks_254))) as ex:
        res_254 = list(ex.map(solve_254_worker, tasks_254))
    map_254 = {r["cell_id"]: r for r in res_254}

    pair_conv_rows = []
    for _, r in df_t2.iterrows():
        ci, cj = int(r["cell_i"]), int(r["cell_j"])
        ri, rj = map_254[ci], map_254[cj]
        dRH_127 = abs(ri["RH20_127"] - rj["RH20_127"])
        dRH_254 = abs(ri["RH20_254"] - rj["RH20_254"])
        dRE_127 = abs(ri["RE_127"] - rj["RE_127"])
        dRE_254 = abs(ri["RE_254"] - rj["RE_254"])
        pair_conv_rows.append({
            "rank": int(r["rank"]),
            "cell_i": ci,
            "cell_j": cj,
            "RH20_i_127": ri["RH20_127"],
            "RH20_i_254": ri["RH20_254"],
            "RH20_j_127": rj["RH20_127"],
            "RH20_j_254": rj["RH20_254"],
            "dRH20_127": dRH_127,
            "dRH20_254": dRH_254,
            "rel_diff_dRH20_pct": float(abs(dRH_254 - dRH_127) / max(dRH_254, 1e-9) * 100.0),
            "dRE_127": dRE_127,
            "dRE_254": dRE_254,
        })
    df_conv = pd.DataFrame(pair_conv_rows)
    df_conv.to_csv(os.path.join(R_DIR, "phase5_counterexample_convergence_254.csv"), index=False)
    print(
        f"  [254x254 Convergence] Top-10 pairs median |dRH_254 - dRH_127|/dRH_254 = "
        f"{df_conv['rel_diff_dRH20_pct'].median():.2f}% (max = {df_conv['rel_diff_dRH20_pct'].max():.2f}%)",
        flush=True,
    )
    return df_pf_ext, paired_comparisons, df_conv


def main():
    t0 = time.time()
    P = load_pilot()
    D = P["table"]
    idx = {s: np.array(P["splits"][s]) for s in ("train", "val", "test", "test2", "ood_morphology")}

    run_part1_tail_mse()
    df_w, df_b = run_part2_whitening_and_barrier_curves(P, D, idx)
    df_ae = train_and_eval_ae_hifi(P, D, idx)
    df_comb, df_pow = run_part4_conditional_and_power()
    df_pf_ext, paired_comp, df_conv = run_part5_pf_extremes_and_254_convergence(P)

    summary = {
        "runtime_s": time.time() - t0,
        "paired_pf_comparisons": paired_comp,
        "convergence_254_summary": {
            "n_pairs": int(len(df_conv)),
            "median_rel_diff_dRH20_pct": float(df_conv["rel_diff_dRH20_pct"].median()),
            "max_rel_diff_dRH20_pct": float(df_conv["rel_diff_dRH20_pct"].max()),
            "pair1_2751_3270_dRH_127": float(df_conv.iloc[0]["dRH20_127"]),
            "pair1_2751_3270_dRH_254": float(df_conv.iloc[0]["dRH20_254"]),
        },
    }
    with open(os.path.join(R_DIR, "phase5a_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"=== Phase 5A completed in {time.time() - t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
