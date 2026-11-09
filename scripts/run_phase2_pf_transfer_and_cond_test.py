"""Phase 2 Analysis (Draft v0.2, Experiment H, Hypothesis H6, Items I18 & I19):
1. Conditional incremental sufficiency tests (local-swap permutation test + Holm adjustment):
   - On test2 (n=1000): adding learned embedding u_theta(X) to E1 for R_E and RH_b20.
   - On the phase-field design campaign (design_cells, n>1 inclusion cells and all 180 inclusion cells):
     T1: R_E | E1
     T2: RH_b20 | E1
     T3: RH_b20 | E1 + J_1.5
     T4: R_sigma | E1
     T5: R_sigma | E1 + J_1.5
     T6: R_path | E1 + J_1.5
2. Frozen Phase-Field Out-of-Sample Transfer (Sets B, O, C1, C2):
   - Embeds all 127x127 periodic phase-field cells with frozen encoders (MSE, SBR-CF lam=1.0, SBR-CF lam=3.0,
     RankSim gam=1.0, RNC tau=1.0) across all 3 seeds.
   - Evaluates out-of-sample R2 and RMSE on Set B, Set O, Set C1, Set C2 and 5-fold cell-grouped CV on design_cells.

Outputs:
  results/cond_test_h6.csv
  results/pf_transfer_i19.csv
  results/pf_transfer_summary.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import torch

from suffml.data.datasets import load_pilot, load_design
from suffml.data.frozen import DESCRIPTOR_STUDY
from suffml.metrics import cond_perm_swap, holm_adjust, WhiteningTransform, omega_ratio
from suffml.physics.critical_distance import CriticalDistance
from suffml.train.train_gate import Supervised, embed_all, DEV

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")
MD = os.path.join(R, "models")
E1_COLS = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]

MODEL_FAMILIES = {
    "MSE (R_E)": ("gap", ["sup_R_E_gap_d8_s0", "sup_R_E_gap_d8_s1", "sup_R_E_gap_d8_s2"]),
    "MSE (RH_b20)": ("gem", ["sup_RH_b20_gem2_d8_s0", "sup_RH_b20_gem2_d8_s1", "sup_RH_b20_gem2_d8_s2"]),
    "SBR-CF (lam=1.0)": ("gem", ["obj_RH_b20_sbrcf1_d8_s0", "obj_RH_b20_sbrcf1_d8_s1", "obj_RH_b20_sbrcf1_d8_s2"]),
    "SBR-CF (lam=3.0)": ("gem", ["obj_RH_b20_sbrcf3_d8_s0", "obj_RH_b20_sbrcf3_d8_s1", "obj_RH_b20_sbrcf3_d8_s2"]),
    "RankSim (gam=1.0)": ("gem", ["obj_RH_b20_ranksimg1_d8_s0", "obj_RH_b20_ranksimg1_d8_s1", "obj_RH_b20_ranksimg1_d8_s2"]),
    "RNC (tau=1.0)": ("gem", ["obj_RH_b20_rnct1_d8_s0", "obj_RH_b20_rnct1_d8_s1", "obj_RH_b20_rnct1_d8_s2"]),
}


def embed_masks_with_tag(tag: str, pool: str, masks: np.ndarray, d: int = 8) -> np.ndarray:
    """Load frozen Supervised checkpoint and compute exact D2-invariant embeddings on masks."""
    pt_path = os.path.join(MD, f"{tag}.pt")
    js_path = os.path.join(MD, f"{tag}.json")
    info = json.load(open(js_path))
    model = Supervised(d, pool, info.get("Lambda", 16.0)).to(DEV)
    state = torch.load(pt_path, map_location=DEV, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    X = torch.from_numpy(masks[:, None].astype(np.float32))
    return embed_all(model.embed, X)


def r2_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def ridge_cv_and_predict(Xtr: np.ndarray, ytr: np.ndarray, cells_tr: np.ndarray, Xte_dict: dict,
                         alphas=(1e-3, 1e-2, 1e-1, 1.0, 5.0, 20.0)):
    """Select ridge penalty alpha by 5-fold cell-grouped CV on Xtr, then predict on Xte_dict."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Ztr = (Xtr - mu) / sd
    uniq_cells = np.unique(cells_tr)
    rng = np.random.default_rng(42)
    perm_cells = rng.permutation(uniq_cells)
    folds = np.array_split(perm_cells, 5)

    best_alpha, best_mse = alphas[0], np.inf
    best_oof = None
    for a in alphas:
        oof = np.zeros_like(ytr, dtype=float)
        for f_cells in folds:
            val_mask = np.isin(cells_tr, f_cells)
            tr_mask = ~val_mask
            A = np.column_stack([np.ones(tr_mask.sum()), Ztr[tr_mask]])
            reg = a * np.eye(A.shape[1]); reg[0, 0] = 0.0
            c = np.linalg.solve(A.T @ A + reg, A.T @ ytr[tr_mask])
            B = np.column_stack([np.ones(val_mask.sum()), Ztr[val_mask]])
            oof[val_mask] = B @ c
        mse = float(np.mean((ytr - oof) ** 2))
        if mse < best_mse:
            best_mse, best_alpha, best_oof = mse, a, oof

    # Fit on full training set with best_alpha
    A = np.column_stack([np.ones(len(Ztr)), Ztr])
    reg = best_alpha * np.eye(A.shape[1]); reg[0, 0] = 0.0
    c = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
    preds = {}
    for k, Xte in Xte_dict.items():
        Zte = (Xte - mu) / sd
        preds[k] = np.column_stack([np.ones(len(Zte)), Zte]) @ c
    return r2_score(ytr, best_oof), best_alpha, preds


def main():
    print("=== Phase 2: Conditional Sufficiency Tests (H6, I18) & Phase-Field Transfer (I19) ===", flush=True)

    # ------------------------------------------------------------------
    # Part 1: Conditional test on test2 (n=1000) of sufficiency-ml corpus
    # ------------------------------------------------------------------
    cond_csv = os.path.join(R, "cond_test_h6.csv")
    need_cond = not os.path.exists(cond_csv)
    cond_rows = []
    if need_cond:
        P_pilot = load_pilot()
        D_pilot = P_pilot["table"]
        te2_idx = np.array(P_pilot["splits"]["test2"])
        E1_te2 = D_pilot.loc[te2_idx, E1_COLS].values.astype(float)
        RE_te2 = D_pilot.loc[te2_idx, "R_E"].values.astype(float)
        RH_te2 = D_pilot.loc[te2_idx, "RH_b20"].values.astype(float)

        for fam, (pool, tags) in MODEL_FAMILIES.items():
            for s, tag in enumerate(tags):
                Z_all = np.load(os.path.join(MD, f"{tag}_emb.npz"))["Z"]
                Z_te2 = Z_all[te2_idx]
                for t_name, y_vec in (("R_E", RE_te2), ("RH_b20", RH_te2)):
                    res = cond_perm_swap(
                        E1_te2, Z_te2, y_vec, np.random.default_rng(3000 + s), q=0.01, B=200, metric="std"
                    )
                    cond_rows.append({
                        "corpus": "pilot_test2_n1000",
                        "test_code": f"{t_name}|E1",
                        "target": t_name,
                        "baseline": "E1",
                        "family": fam,
                        "seed": s,
                        **res,
                    })
            print(f"  [test2] Conditional test finished for {fam}", flush=True)

    # ------------------------------------------------------------------
    # Part 2: Embed 216 design cells + Blind Sets B, O + Frozen Sets C1, C2
    # ------------------------------------------------------------------
    des = load_design()
    D_des = des["table"]
    M_des = des["mask"]

    # Load Blind (Sets A/B and O) and TCD Frozen (Sets C1 and C2)
    blind_masks = np.load(os.path.join(DESCRIPTOR_STUDY, "blind", "blind_masks.npz"))["mask"]
    blind_pf = pd.read_csv(os.path.join(DESCRIPTOR_STUDY, "blind", "blind_pf.csv"))
    blind_pf = blind_pf[blind_pf["separated"]].reset_index(drop=True)

    # Compute J_1.5 on blind cells if not already present
    if "J_1.5" not in blind_pf.columns:
        cd = CriticalDistance(N=127, radii_ell=(0.0, 1.5))
        j_rows = [dict(bid=int(b), **cd(blind_masks[int(b)])) for b in blind_pf["bid"]]
        blind_pf = blind_pf.merge(pd.DataFrame(j_rows), on="bid")

    tcd_masks = np.load(os.path.join(DESCRIPTOR_STUDY, "tcd_frozen", "tcd_masks.npz"))["mask"]
    tcd_pf = pd.read_csv(os.path.join(DESCRIPTOR_STUDY, "tcd_frozen", "tcd_pf.csv"))
    tcd_pf = tcd_pf[tcd_pf["separated"]].reset_index(drop=True)

    # Embed all 3 mask arrays with every frozen model checkpoint
    emb_des, emb_blind, emb_tcd = {}, {}, {}
    for fam, (pool, tags) in MODEL_FAMILIES.items():
        for s, tag in enumerate(tags):
            emb_des[(fam, s)] = embed_masks_with_tag(tag, pool, M_des)
            emb_blind[(fam, s)] = embed_masks_with_tag(tag, pool, blind_masks[blind_pf["bid"].values.astype(int)])
            emb_tcd[(fam, s)] = embed_masks_with_tag(tag, pool, tcd_masks[tcd_pf["cid"].values.astype(int)])
        print(f"  [PF masks] Embedded design (216), blind ({len(blind_pf)}), tcd ({len(tcd_pf)}) for {fam}", flush=True)

    # ------------------------------------------------------------------
    # Part 3: Conditional tests T1-T6 on Phase-Field Design Corpus (n>1 inclusion cells, n=120)
    # ------------------------------------------------------------------
    inc_n1_mask = ((D_des["family"] != "laminate") & (D_des["n"] > 1)).values
    D_sub = D_des[inc_n1_mask].reset_index(drop=True)
    E1_sub = D_sub[E1_COLS].values.astype(float)
    E1J_sub = D_sub[E1_COLS + ["J_1.5"]].values.astype(float)

    PF_TESTS = [
        ("T1", "R_E|E1", "R_E", "E1", E1_sub),
        ("T2", "RH_b20|E1", "RH_b20", "E1", E1_sub),
        ("T3", "RH_b20|E1+J1.5", "RH_b20", "E1+J_1.5", E1J_sub),
        ("T4", "R_sigma|E1", "R_sigma", "E1", E1_sub),
        ("T5", "R_sigma|E1+J1.5", "R_sigma", "E1+J_1.5", E1J_sub),
        ("T6", "R_path|E1+J1.5", "R_path", "E1+J_1.5", E1J_sub),
    ]

    if need_cond:
        for fam, (pool, tags) in MODEL_FAMILIES.items():
            for s in range(len(tags)):
                Z_sub = emb_des[(fam, s)][inc_n1_mask]
                for code, label, t_col, base_name, X_base in PF_TESTS:
                    y_vec = D_sub[t_col].values.astype(float)
                    res = cond_perm_swap(
                        X_base, Z_sub, y_vec, np.random.default_rng(4000 + 100 * s + int(code[1])),
                        q=0.01, B=2000, metric="std"
                    )
                    cond_rows.append({
                        "corpus": "pf_design_n120",
                        "test_code": f"{code}:{label}",
                        "target": t_col,
                        "baseline": base_name,
                        "family": fam,
                        "seed": s,
                        **res,
                    })

        df_cond = pd.DataFrame(cond_rows)
        # Apply Holm step-down across T1-T6 within each (corpus, family, seed)
        df_cond["holm_p_max"] = np.nan
        df_cond["holm_p_mean"] = np.nan
        for (corp, fam, s), g in df_cond.groupby(["corpus", "family", "seed"]):
            df_cond.loc[g.index, "holm_p_max"] = holm_adjust(g["p_max"].values)
            df_cond.loc[g.index, "holm_p_mean"] = holm_adjust(g["p_mean"].values)
        df_cond.to_csv(cond_csv, index=False)
        print("Saved cond_test_h6.csv", flush=True)
    else:
        df_cond = pd.read_csv(cond_csv)
        print("Loaded cached cond_test_h6.csv", flush=True)

    blind_pf["RH_b20"] = blind_pf["lnB"]
    tcd_pf["RH_b20"] = tcd_pf["lnB"]

    # ------------------------------------------------------------------
    # Part 4: Out-of-Sample Phase-Field Transfer (180 design inclusion cells -> Sets B, O, C1, C2)
    # ------------------------------------------------------------------
    inc_mask = (D_des["family"] != "laminate").values
    TR = D_des[inc_mask].reset_index(drop=True)
    cells_tr = TR["cell"].values
    y_sigma_tr = TR["R_sigma"].values.astype(float)
    y_path_tr = TR["R_path"].values.astype(float)

    mask_B = (blind_pf["set"] == "A").values
    mask_O = (blind_pf["set"] == "O").values
    mask_B_path = (blind_pf["set"] == "A").values & np.isfinite(blind_pf["R_path"].values)
    mask_O_path = (blind_pf["set"] == "O").values & np.isfinite(blind_pf["R_path"].values)
    mask_C1 = (tcd_pf["set"] == "C1").values
    mask_C2 = (tcd_pf["set"] == "C2").values

    y_sets = {
        "R_sigma": {
            "Set_B": blind_pf.loc[mask_B, "R_sigma"].values.astype(float),
            "Set_O": blind_pf.loc[mask_O, "R_sigma"].values.astype(float),
            "Set_C1": tcd_pf.loc[mask_C1, "R_sigma"].values.astype(float),
            "Set_C2": tcd_pf.loc[mask_C2, "R_sigma"].values.astype(float),
        },
        "R_path": {
            "Set_B": blind_pf.loc[mask_B_path, "R_path"].values.astype(float),
            "Set_O": blind_pf.loc[mask_O_path, "R_path"].values.astype(float),
        },
    }

    pf_rows = []

    # 1. Explicit descriptor baselines
    EXPLICIT = {
        "E1": E1_COLS,
        "E1+RH_b20": E1_COLS + ["RH_b20"],
        "E1+J_0": E1_COLS + ["J_0"],
        "E1+J_1.5": E1_COLS + ["J_1.5"],
        "J_1.5_alone": ["J_1.5"],
    }
    for exp_name, cols in EXPLICIT.items():
        Xtr = TR[cols].values.astype(float)
        Xte_sigma = {
            "Set_B": blind_pf.loc[mask_B, cols].values.astype(float),
            "Set_O": blind_pf.loc[mask_O, cols].values.astype(float),
            "Set_C1": tcd_pf.loc[mask_C1, cols].values.astype(float),
            "Set_C2": tcd_pf.loc[mask_C2, cols].values.astype(float),
        }
        r2_cv, a_best, preds = ridge_cv_and_predict(Xtr, y_sigma_tr, cells_tr, Xte_sigma, alphas=(1e-3,))
        om_tr = omega_ratio(WhiteningTransform().fit_transform(Xtr), y_sigma_tr, np.random.default_rng(77), q=0.01, nperm=500)
        row = {
            "target": "R_sigma",
            "representation": exp_name,
            "kind": "explicit",
            "seed": 0,
            "alpha_cv": a_best,
            "omega_ratio_design180": om_tr,
            "R2_cv_design180": r2_cv,
        }
        for s_key, y_true in y_sets["R_sigma"].items():
            row[f"R2_{s_key}"] = r2_score(y_true, preds[s_key])
            row[f"RMSE_{s_key}"] = float(np.sqrt(np.mean((y_true - preds[s_key]) ** 2)))
        pf_rows.append(row)

        # Also evaluate R_path on design180, Set_B, Set_O
        Xte_path = {
            "Set_B": blind_pf.loc[mask_B_path, cols].values.astype(float),
            "Set_O": blind_pf.loc[mask_O_path, cols].values.astype(float),
        }
        r2_cv_p, a_best_p, preds_p = ridge_cv_and_predict(
            Xtr, y_path_tr, cells_tr, Xte_path, alphas=(1e-3,)
        )
        om_tr_p = omega_ratio(WhiteningTransform().fit_transform(Xtr), y_path_tr, np.random.default_rng(78), q=0.01, nperm=500)
        row_p = {
            "target": "R_path",
            "representation": exp_name,
            "kind": "explicit",
            "seed": 0,
            "alpha_cv": a_best_p,
            "omega_ratio_design180": om_tr_p,
            "R2_cv_design180": r2_cv_p,
            "R2_Set_B": r2_score(y_sets["R_path"]["Set_B"], preds_p["Set_B"]),
            "RMSE_Set_B": float(np.sqrt(np.mean((y_sets["R_path"]["Set_B"] - preds_p["Set_B"]) ** 2))),
            "R2_Set_O": r2_score(y_sets["R_path"]["Set_O"], preds_p["Set_O"]),
            "RMSE_Set_O": float(np.sqrt(np.mean((y_sets["R_path"]["Set_O"] - preds_p["Set_O"]) ** 2))),
            "R2_Set_C1": np.nan,
            "RMSE_Set_C1": np.nan,
            "R2_Set_C2": np.nan,
            "RMSE_Set_C2": np.nan,
        }
        pf_rows.append(row_p)

    # 2. Learned embeddings alone and combined with E1 and E1+J_1.5
    E1_tr = TR[E1_COLS].values.astype(float)
    E1J_tr = TR[E1_COLS + ["J_1.5"]].values.astype(float)
    E1_te = {
        "Set_B": blind_pf.loc[mask_B, E1_COLS].values.astype(float),
        "Set_O": blind_pf.loc[mask_O, E1_COLS].values.astype(float),
        "Set_C1": tcd_pf.loc[mask_C1, E1_COLS].values.astype(float),
        "Set_C2": tcd_pf.loc[mask_C2, E1_COLS].values.astype(float),
    }
    E1J_te = {
        "Set_B": blind_pf.loc[mask_B, E1_COLS + ["J_1.5"]].values.astype(float),
        "Set_O": blind_pf.loc[mask_O, E1_COLS + ["J_1.5"]].values.astype(float),
        "Set_C1": tcd_pf.loc[mask_C1, E1_COLS + ["J_1.5"]].values.astype(float),
        "Set_C2": tcd_pf.loc[mask_C2, E1_COLS + ["J_1.5"]].values.astype(float),
    }

    for fam, (pool, tags) in MODEL_FAMILIES.items():
        for s in range(len(tags)):
            Z_tr = emb_des[(fam, s)][inc_mask]
            Z_te = {
                "Set_B": emb_blind[(fam, s)][mask_B],
                "Set_O": emb_blind[(fam, s)][mask_O],
                "Set_C1": emb_tcd[(fam, s)][mask_C1],
                "Set_C2": emb_tcd[(fam, s)][mask_C2],
            }
            combos = [
                (f"{fam} alone", "learned_alone", Z_tr, Z_te),
                (f"E1 + {fam}", "E1_plus_learned", np.hstack([E1_tr, Z_tr]), {k: np.hstack([E1_te[k], Z_te[k]]) for k in Z_te}),
                (f"E1+J_1.5 + {fam}", "E1J_plus_learned", np.hstack([E1J_tr, Z_tr]), {k: np.hstack([E1J_te[k], Z_te[k]]) for k in Z_te}),
            ]
            for rep_label, kind_str, Xtr_c, Xte_c in combos:
                r2_cv, a_best, preds = ridge_cv_and_predict(Xtr_c, y_sigma_tr, cells_tr, Xte_c)
                om_tr = omega_ratio(WhiteningTransform().fit_transform(Xtr_c), y_sigma_tr, np.random.default_rng(500 + s), q=0.01, nperm=500)
                row = {
                    "target": "R_sigma",
                    "representation": rep_label,
                    "kind": kind_str,
                    "seed": s,
                    "alpha_cv": a_best,
                    "omega_ratio_design180": om_tr,
                    "R2_cv_design180": r2_cv,
                }
                for s_key, y_true in y_sets["R_sigma"].items():
                    row[f"R2_{s_key}"] = r2_score(y_true, preds[s_key])
                    row[f"RMSE_{s_key}"] = float(np.sqrt(np.mean((y_true - preds[s_key]) ** 2)))
                pf_rows.append(row)

                # R_path transfer
                idx_B_p = np.flatnonzero(mask_B_path[mask_B])
                idx_O_p = np.flatnonzero(mask_O_path[mask_O])
                r2_cv_p, a_best_p, preds_p = ridge_cv_and_predict(
                    Xtr_c, y_path_tr, cells_tr, {"Set_B": Xte_c["Set_B"][idx_B_p], "Set_O": Xte_c["Set_O"][idx_O_p]}
                )
                om_tr_p = omega_ratio(WhiteningTransform().fit_transform(Xtr_c), y_path_tr, np.random.default_rng(600 + s), q=0.01, nperm=500)
                row_p = {
                    "target": "R_path",
                    "representation": rep_label,
                    "kind": kind_str,
                    "seed": s,
                    "alpha_cv": a_best_p,
                    "omega_ratio_design180": om_tr_p,
                    "R2_cv_design180": r2_cv_p,
                    "R2_Set_B": r2_score(y_sets["R_path"]["Set_B"], preds_p["Set_B"]),
                    "RMSE_Set_B": float(np.sqrt(np.mean((y_sets["R_path"]["Set_B"] - preds_p["Set_B"]) ** 2))),
                    "R2_Set_O": r2_score(y_sets["R_path"]["Set_O"], preds_p["Set_O"]),
                    "RMSE_Set_O": float(np.sqrt(np.mean((y_sets["R_path"]["Set_O"] - preds_p["Set_O"]) ** 2))),
                    "R2_Set_C1": np.nan,
                    "RMSE_Set_C1": np.nan,
                    "R2_Set_C2": np.nan,
                    "RMSE_Set_C2": np.nan,
                }
                pf_rows.append(row_p)

    df_pf = pd.DataFrame(pf_rows)
    df_pf.to_csv(os.path.join(R, "pf_transfer_i19.csv"), index=False)

    # Summarize medians across seeds
    cond_med = (
        df_cond.groupby(["corpus", "test_code", "family"])[["ratio_max", "p_max", "holm_p_max", "ratio_mean", "p_mean", "holm_p_mean"]]
        .median()
        .reset_index()
    )
    pf_med = (
        df_pf.groupby(["target", "kind", "representation"])[
            ["omega_ratio_design180", "R2_cv_design180", "R2_Set_B", "R2_Set_O", "R2_Set_C1", "R2_Set_C2"]
        ]
        .median()
        .reset_index()
    )
    summary = {
        "conditional_tests_median": cond_med.to_dict(orient="records"),
        "pf_transfer_median": pf_med.to_dict(orient="records"),
    }
    json.dump(summary, open(os.path.join(R, "pf_transfer_summary.json"), "w"), indent=1)

    print("\n--- Summary of Conditional Tests (median across seeds) ---")
    print(cond_med.round(4).to_string(index=False))

    print("\n--- Summary of Phase-Field Transfer on R_sigma (median across seeds) ---")
    print(pf_med[pf_med["target"] == "R_sigma"].round(3).to_string(index=False))

    print("\n--- Summary of Phase-Field Transfer on R_path (median across seeds) ---")
    print(pf_med[pf_med["target"] == "R_path"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
