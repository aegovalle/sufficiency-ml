"""Phase 5B: Direct Phase-Field Weakened-Strip Translation Barrier Check.
Evaluates the exact translation-invariance floor:
    b_shift(x) = 0.5 * [max_T R(T x) - min_T R(T x)]
on 8 stratified inclusion cells from the AT1 phase-field design campaign across 4 periodic shifts
along e_1 (dx in {0, 32, 64, 95} pixels, i.e., 0, L/4, L/2, 3L/4), where the 2%-weakened vertical
strip remains fixed at x_1 = 0.95 L. Also combines with the 16-cell strip095 (5% strip) and strip099
(1% strip) sensitivity runs.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from suffml.data.datasets import load_design, ROOT
from suffml.data.frozen import DESCRIPTOR_STUDY
from suffml.physics.pf_fft import PhaseFieldRVE

R_DIR = os.path.join(ROOT, "results")
HALF_OFFS = [(1, 0), (0, 1), (1, 1), (-1, 1), (1, 2), (-1, 2), (2, 1), (-2, 1)]


def min_path(Gc: np.ndarray, allowed: np.ndarray) -> float:
    Nn = Gc.shape[0]
    h = 1.0 / Nn
    J = np.arange(Nn + 1)
    I = np.arange(Nn)
    II, JJ = np.meshgrid(I, J, indexing="ij")
    A = allowed[II, JJ % Nn]
    G = Gc[II, JJ % Nn]
    nid = JJ * Nn + II
    rows, cols, w = [], [], []
    for di, dj in HALF_OFFS:
        i2 = (II + di) % Nn
        j2 = JJ + dj
        ok = (j2 >= 0) & (j2 <= Nn)
        j2c = np.clip(j2, 0, Nn)
        ok &= A & A[i2, j2c]
        L = np.hypot(di, dj) * h
        rows.append(nid[ok]); cols.append((j2c * Nn + i2)[ok])
        w.append(L * 0.5 * (G[ok] + G[i2, j2c][ok]))
    Mx = sp.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(Nn * (Nn + 1),) * 2)
    src = np.flatnonzero(A[:, 0])
    if len(src) == 0:
        return float("nan")
    dist = dijkstra(Mx, directed=False, indices=src, return_predecessors=False)
    tgt = Nn * Nn + src
    vals = dist[np.arange(len(src)), tgt]
    k = int(np.argmin(vals))
    return float(vals[k]) if np.isfinite(vals[k]) else float("nan")


def solve_shifted_pf(args):
    cell_id, shift_px, mask_0, sigma_hom, Gp_hom = args
    cache_dir = os.path.join(R_DIR, "pf_shifts")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"rve_{cell_id:03d}_sh{shift_px:03d}.json")
    if os.path.exists(cache_file):
        return json.load(open(cache_file))

    t0 = time.time()
    mask_sh = np.roll(mask_0, shift=shift_px, axis=0)
    rve = PhaseFieldRVE(mask_sh)
    out, _, d_final = rve.run()
    gp = min_path(rve.Gc, d_final >= 0.9)
    R_sigma = float(np.log(out["sigma_peak"] / sigma_hom))
    R_path = float(np.log(gp / Gp_hom)) if np.isfinite(gp) else float("nan")
    res = {
        "cell_id": int(cell_id),
        "shift_px": int(shift_px),
        "sigma_peak": float(out["sigma_peak"]),
        "Gamma_path": float(gp),
        "R_sigma": R_sigma,
        "R_path": R_path,
        "separated": bool(out["separated"]),
        "runtime_s": float(time.time() - t0),
    }
    json.dump(res, open(cache_file, "w"), indent=2)
    return res


def main():
    t0 = time.time()
    des = load_design()
    D_des = des["table"]
    M_des = des["mask"]

    hom = json.load(open(os.path.join(DESCRIPTOR_STUDY, "pf_hom.json")))
    d_hom = np.load(os.path.join(DESCRIPTOR_STUDY, "pf_out", "hom_d.npy"))
    sigma_hom = float(hom["sigma_peak"])
    Gp_hom = min_path(PhaseFieldRVE(np.zeros((127, 127), bool)).Gc, d_hom >= 0.9)

    # 8 stratified cells covering laminates and inclusion cells (disks n=4,9 and ellipses n=4,9)
    cells_check = [13, 16, 28, 49, 96, 100, 190, 205]
    shifts = [32, 64, 95]

    # Also compute the adversarial shift T_on for each cell that places argmax(psi_plus) at the strip x_1 = 0.95 (idx 120)
    strip_idx = int(round(0.95 * 127 - 0.5))  # pixel 120
    adv_shifts = {}
    for cid in cells_check:
        rve0 = PhaseFieldRVE(M_des[cid])
        _, eps_u, _ = rve0.solve_elastic(np.zeros((127, 127)), 1.0, tol=1e-9)
        psi_u = rve0.psi_plus(eps_u)
        # Find x_1 index of maximum onset ratio psi_u / psic
        i_max, _ = np.unravel_index(np.argmax(psi_u / rve0.psic), psi_u.shape)
        sh_adv = int((strip_idx - i_max) % 127)
        adv_shifts[cid] = sh_adv

    tasks = []
    for cid in cells_check:
        for sh in shifts + [adv_shifts[cid]]:
            if sh != 0:
                tasks.append((cid, sh, M_des[cid], sigma_hom, Gp_hom))

    print(f"=== Phase 5B: Solving {len(tasks)} shifted AT1 phase-field runs across {len(cells_check)} cells ===", flush=True)
    with ProcessPoolExecutor(max_workers=min(24, len(tasks))) as ex:
        results = list(ex.map(solve_shifted_pf, tasks))

    # Combine with shift=0 from D_des (deduplicating if sh_adv in shifts)
    all_rows = []
    for cid in cells_check:
        r0 = D_des.loc[D_des["id"] == cid].iloc[0]
        seen_sh = {0}
        all_rows.append({
            "cell_id": int(cid),
            "family": str(r0["family"]),
            "n_inc": int(r0["n"]),
            "f": float(r0["f"]),
            "shift_px": 0,
            "is_adv": int(adv_shifts[cid] == 0),
            "R_sigma": float(r0["R_sigma"]),
            "R_path": float(r0["R_path"]),
        })
        for r in results:
            if r["cell_id"] == cid and int(r["shift_px"]) not in seen_sh:
                seen_sh.add(int(r["shift_px"]))
                all_rows.append({
                    "cell_id": int(cid),
                    "family": str(r0["family"]),
                    "n_inc": int(r0["n"]),
                    "f": float(r0["f"]),
                    "shift_px": int(r["shift_px"]),
                    "is_adv": int(int(r["shift_px"]) == adv_shifts[cid]),
                    "R_sigma": float(r["R_sigma"]),
                    "R_path": float(r["R_path"]),
                })

    df_sh = pd.DataFrame(all_rows).sort_values(["cell_id", "shift_px"]).reset_index(drop=True)
    df_sh.to_csv(os.path.join(R_DIR, "phase5_pf_strip_shifts.csv"), index=False)

    # Compute per-cell exact shift-invariance barrier b_shift(x) = 0.5 * (max_T R(Tx) - min_T R(Tx))
    cell_summaries = []
    for cid, g in df_sh.groupby("cell_id"):
        rs = g["R_sigma"].values
        rp = g["R_path"].dropna().values
        g_grid = g[g["shift_px"].isin([0, 32, 64, 95])]
        rs_grid = g_grid["R_sigma"].values
        b_sig_grid = float(0.5 * (np.max(rs_grid) - np.min(rs_grid)))
        b_sig = float(0.5 * (np.max(rs) - np.min(rs)))
        b_path = float(0.5 * (np.max(rp) - np.min(rp))) if len(rp) > 0 else float("nan")
        rel_sig_pct = float((np.exp(np.max(rs) - np.min(rs)) - 1.0) * 50.0)
        cell_summaries.append({
            "cell_id": int(cid),
            "family": str(g["family"].iloc[0]),
            "n_inc": int(g["n_inc"].iloc[0]),
            "f": float(g["f"].iloc[0]),
            "adv_shift_px": int(adv_shifts[cid]),
            "R_sigma_mean": float(np.mean(rs)),
            "R_sigma_span": float(np.max(rs) - np.min(rs)),
            "b_shift_grid_R_sigma": b_sig_grid,
            "b_shift_R_sigma": b_sig,
            "b_shift_rel_stress_pct": rel_sig_pct,
            "R_path_mean": float(np.mean(rp)),
            "R_path_span": float(np.max(rp) - np.min(rp)),
            "b_shift_R_path": b_path,
        })
    df_sum = pd.DataFrame(cell_summaries)
    df_sum.to_csv(os.path.join(R_DIR, "phase5_pf_strip_barrier_summary.csv"), index=False)

    summary = {
        "n_cells": int(len(df_sum)),
        "shifts_px": [0, 32, 64, 95, "adv_argmax_psi"],
        "b_shift_grid_R_sigma_max": float(df_sum["b_shift_grid_R_sigma"].max()),
        "b_shift_R_sigma_median": float(df_sum["b_shift_R_sigma"].median()),
        "b_shift_R_sigma_mean": float(df_sum["b_shift_R_sigma"].mean()),
        "b_shift_R_sigma_max": float(df_sum["b_shift_R_sigma"].max()),
        "b_shift_rel_stress_pct_median": float(df_sum["b_shift_rel_stress_pct"].median()),
        "b_shift_rel_stress_pct_max": float(df_sum["b_shift_rel_stress_pct"].max()),
        "b_shift_R_path_median": float(df_sum["b_shift_R_path"].median()),
        "b_shift_R_path_mean": float(df_sum["b_shift_R_path"].mean()),
        "b_shift_R_path_max": float(df_sum["b_shift_R_path"].max()),
        "runtime_s": float(time.time() - t0),
    }
    json.dump(summary, open(os.path.join(R_DIR, "phase5b_strip_summary.json"), "w"), indent=2)
    print(f"=== Phase 5B completed in {time.time() - t0:.1f}s ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
