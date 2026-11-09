"""Phase 3 Ablations & Second Encoder Family (Draft v0.2, Section 15 Ablations 6, 8, 9 & Item I09):
1. Ablation 6 (Objective components on E-circ, d=8, 3 seeds):
   - MSE (existing: sup_RH_b20_gem2_d8)
   - SBR-in (existing: obj_RH_b20_sbrin1_d8)
   - SBR-CF without memory (abl_nomem_d8: mem_size=0, k_mine=128)
   - SBR-CF without hard mining (abl_nomine_d8: mem_size=1024, k_mine=0, k_b=0)
   - Full SBR-CF (existing: obj_RH_b20_sbrcf1_d8)
2. Ablation 8 (Pooling mismatch on E-circ, d=8, 3 seeds):
   - MSE + GAP (existing: sup_RH_b20_gap_d8)
   - SBR-CF + GAP (abl_gap_sbrcf1_d8: pool='gap', lam=1.0)
   - MSE + GeM (existing: sup_RH_b20_gem2_d8)
   - SBR-CF + GeM (existing: obj_RH_b20_sbrcf1_d8)
3. Ablation 9 & Item I09 (Second encoder family E-FNO and non-equivariant E-std, d=8, 3 seeds):
   - E-FNO + MSE (abl_efno_mse_d8: arch='efno', method='mse')
   - E-FNO + SBR-CF (abl_efno_sbrcf1_d8: arch='efno', method='sbr_cf', lam=1.0)
   - E-std + MSE (abl_estd_mse_d8: arch='estd', d2_sym=False, method='mse')
   - E-std + MSE + L_inv (abl_estd_linv_d8: arch='estd', d2_sym=False, method='mse', lam_inv=1.0)

Outputs:
  results/ablations_phase3_metrics.csv
  results/ablations_phase3_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import torch

from suffml.data.datasets import load_pilot
from suffml.analysis.evaluate import evaluate
from suffml.train.train_objectives import train_objective
from suffml.train.train_gate import Supervised, embed_all, DEV

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")
MD = os.path.join(R, "models")
SETS = ("train", "val", "test2", "ood_morphology")


def run_job(tag, method, **kw):
    marker = os.path.join(MD, f"{tag}.consol")
    if os.path.exists(marker):
        return
    train_objective(method, target="RH_b20", tag=tag, **kw)
    open(marker, "w").write("ok")


@torch.no_grad()
def measure_invariance_defect(tag: str, X_te: torch.Tensor) -> float:
    """Measure RMS embedding change under a circular shift (17, -23) + horizontal flip on X_te."""
    js_path = os.path.join(MD, f"{tag}.json")
    pt_path = os.path.join(MD, f"{tag}.pt")
    info = json.load(open(js_path))
    arch = info.get("arch", "ecirc")
    pool = info.get("pool", "gem")
    d2_sym = info.get("d2_sym", True)
    d = info.get("d", 8)
    Lam = info.get("Lambda", 16.0)
    model = Supervised(d, pool, Lam, arch=arch, d2_sym=d2_sym).to(DEV)
    model.load_state_dict(torch.load(pt_path, map_location=DEV, weights_only=True))
    model.eval()
    X_g = torch.roll(X_te, shifts=(17, -23), dims=(-2, -1)).flip(-1)
    Z0 = embed_all(model.embed, X_te)
    Zg = embed_all(model.embed, X_g)
    sd = Z0.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    diff = (Z0 - Zg) / sd
    return float(np.sqrt(np.mean(diff ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    E = args.epochs
    seeds = tuple(range(args.seeds))

    for s in seeds:
        # Ablation 6: No memory & No mining
        run_job(f"abl_nomem_d8_s{s}", "sbr_cf", lam=1.0, mem_size=0, k_mine=128, mine_every=5, epochs=E, seed=s)
        run_job(f"abl_nomine_d8_s{s}", "sbr_cf", lam=1.0, mem_size=1024, k_mine=0, k_b=0, epochs=E, seed=s)
        # Ablation 8: Pooling mismatch (GAP with SBR-CF)
        run_job(f"abl_gap_sbrcf1_d8_s{s}", "sbr_cf", lam=1.0, pool="gap", mine_every=5, epochs=E, seed=s)
        # Ablation 9 & Item I09: E-FNO (MSE and SBR-CF) and E-std (MSE and MSE + L_inv)
        run_job(f"abl_efno_mse_d8_s{s}", "mse", arch="efno", epochs=E, seed=s)
        run_job(f"abl_efno_sbrcf1_d8_s{s}", "sbr_cf", arch="efno", lam=1.0, mine_every=5, epochs=E, seed=s)
        run_job(f"abl_estd_mse_d8_s{s}", "mse", arch="estd", d2_sym=False, epochs=E, seed=s)
        run_job(f"abl_estd_linv_d8_s{s}", "mse", arch="estd", d2_sym=False, lam_inv=1.0, epochs=E, seed=s)

    # Evaluate all Phase 3 families alongside existing reference families
    P = load_pilot()
    D = P["table"]
    idx = {s: np.array(P["splits"][s]) for s in SETS}
    X_te_sub = torch.from_numpy(P["mask"][idx["test2"][:200], None].astype(np.float32))
    y = D["RH_b20"].values
    ymap = {
        "train": y[idx["train"]],
        "val": y[idx["val"]],
        "test2": y[idx["test2"]],
        "ood": y[idx["ood_morphology"]],
    }

    ALL_FAMS = {
        "E-circ + GeM + MSE": [f"sup_RH_b20_gem2_d8_s{s}" for s in seeds],
        "E-circ + GeM + SBR-in": [f"obj_RH_b20_sbrin1_d8_s{s}" for s in seeds],
        "E-circ + GeM + SBR-CF (no mem)": [f"abl_nomem_d8_s{s}" for s in seeds],
        "E-circ + GeM + SBR-CF (no mine)": [f"abl_nomine_d8_s{s}" for s in seeds],
        "E-circ + GeM + SBR-CF (full, lam=1)": [f"obj_RH_b20_sbrcf1_d8_s{s}" for s in seeds],
        "E-circ + GeM + SBR-CF (full, lam=3)": [f"obj_RH_b20_sbrcf3_d8_s{s}" for s in seeds],
        "E-circ + GAP + MSE": [f"sup_RH_b20_gap_d8_s{s}" for s in seeds],
        "E-circ + GAP + SBR-CF (lam=1)": [f"abl_gap_sbrcf1_d8_s{s}" for s in seeds],
        "E-FNO + GeM + MSE": [f"abl_efno_mse_d8_s{s}" for s in seeds],
        "E-FNO + GeM + SBR-CF (lam=1)": [f"abl_efno_sbrcf1_d8_s{s}" for s in seeds],
        "E-std (no sym) + GeM + MSE": [f"abl_estd_mse_d8_s{s}" for s in seeds],
        "E-std (no sym) + GeM + MSE + L_inv": [f"abl_estd_linv_d8_s{s}" for s in seeds],
    }

    rows = []
    for fam_label, tags in ALL_FAMS.items():
        for s, tag in enumerate(tags):
            emb_path = os.path.join(MD, f"{tag}_emb.npz")
            if not os.path.exists(emb_path):
                continue
            Z = np.load(emb_path)["Z"]
            Zs = {
                "train": Z[idx["train"]],
                "val": Z[idx["val"]],
                "test2": Z[idx["test2"]],
                "ood": Z[idx["ood_morphology"]],
            }
            r = evaluate(Zs, ymap, "RH_b20", np.random.default_rng(7000 + abs(hash(tag)) % 10000), metric="white",
                         eval_sets=("test2", "ood"))
            inv_def = measure_invariance_defect(tag, X_te_sub)
            js = json.load(open(os.path.join(MD, f"{tag}.json")))
            for sname in ("test2", "ood"):
                rows.append({
                    "family": fam_label,
                    "tag": tag,
                    "seed": s,
                    "set": sname,
                    "inv_defect": inv_def,
                    "p_final": js.get("p_final", np.nan),
                    **r[sname],
                })
        print(f"Evaluated {fam_label}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R, "ablations_phase3_metrics.csv"), index=False)

    med = (
        df.groupby(["set", "family"])[
            ["omega_ratio", "R2_mcshane", "R2_ridge", "cf_rmse_sd", "cf_max_sd", "inv_defect", "p_final"]
        ]
        .median()
        .reset_index()
    )
    json.dump(med.to_dict(orient="records"), open(os.path.join(R, "ablations_phase3_summary.json"), "w"), indent=1)
    print("\n--- Summary of Phase 3 Ablations (test2 median) ---")
    print(med[med["set"] == "test2"].round(3).to_string(index=False))
    print("\n--- Summary of Phase 3 Ablations (ood median) ---")
    print(med[med["set"] == "ood"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
