"""Stage 0 (Draft v0.2, Section 9.1; checklist I01): exact reproduction of the 216-cell
elastic campaign of the descriptor study and registry of the frozen phase-field geometries.

Run from the repository root:  .venv\\Scripts\\python scripts\\reproduce_stage0.py [n_workers]
Outputs: data/stage0/design_cells.npz, data/stage0/design_cells.csv,
         data/stage0/stage0_report.json, data/frozen/frozen_hashes.json
"""
import os
import sys
import json
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd

from suffml.data.geometry import design_geometries, mask_of
from suffml.data.canonical_hash import canonical_hash
from suffml.data import frozen

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "stage0")
REF = frozen.DESCRIPTOR_STUDY


def work(args):
    k, mask = args
    from suffml.physics.fft_targets import solve_cell
    from suffml.physics.critical_distance import CriticalDistance
    global _CD
    if "_CD" not in globals():
        _CD = CriticalDistance(radii_ell=(0.0, 1.0, 1.5, 2.0))
    r = solve_cell(mask, keep_fields=True)
    r["scalars"].update(_CD(mask))
    return k, r


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rep = {}
    geoms = design_geometries()
    masks = np.array([mask_of(g) for g, _ in geoms])
    ref_masks = np.load(os.path.join(REF, "fields.npz"))["mask"]
    rep["masks_identical"] = bool(np.array_equal(masks, ref_masks))

    from multiprocessing import Pool
    with Pool(nw) as pool:
        res = dict(pool.map(work, list(enumerate(masks))))
    rows = []
    for k, (g, meta) in enumerate(geoms):
        s = dict(res[k]["scalars"]); s.update(meta); s["id"] = k
        rows.append(s)
    D = pd.DataFrame(rows)

    # 1. targets and descriptors against the stored campaign
    C = pd.read_csv(os.path.join(REF, "campaign_results.csv")).set_index("id").loc[D.id]
    cmp = {}
    for col in ("W_hom", "lnB", "f", "rhoI", "e11_m", "e22_m", "e12_m", "e11_i", "Jgap_m", "Jgap_i", "I", "lnVeff"):
        cmp[col] = float(np.max(np.abs(D[col].values - C[col].values)))
    rep["max_abs_diff_vs_campaign_results"] = cmp
    # 2. hazard family against the stored nonlocal stress fields (formula of v09_analysis.py)
    SB = np.load(os.path.join(REF, "fields.npz"))["sbar"]
    N2 = SB.shape[1] ** 2
    hz = {}
    for b in (2, 5, 10, 20, 40):
        ref = np.array([b * np.log(SB[i][~masks[i]].max())
                        + np.log(np.sum((SB[i][~masks[i]] / SB[i][~masks[i]].max()) ** b) / N2) for i in range(len(D))])
        hz[f"RH_b{b}"] = float(np.max(np.abs(D[f"RH_b{b}"].values - ref)))
    rep["max_abs_diff_hazard_family_vs_stored_fields"] = hz
    # 3. ligament amplification, two-point maps, critical-distance onset
    V6 = pd.read_csv(os.path.join(REF, "v06_desc.csv")).set_index("id").loc[D.id]
    rep["max_abs_diff_A_lig"] = float(np.max(np.abs(D.A_lig.values - V6.A_lig.values)))
    S2 = np.array([res[k]["S2"] for k in range(len(D))])
    rep["max_abs_diff_S2"] = float(np.max(np.abs(S2 - np.load(os.path.join(REF, "v09_S2.npy")))))
    T = pd.read_csv(os.path.join(REF, "v09_tcd.csv")).set_index("id").loc[D.id]
    rep["max_abs_diff_J"] = {"J_0": float(np.max(np.abs(D["J_0"].values - T.J_r0.values))),
                             "J_1": float(np.max(np.abs(D["J_1"].values - T.J_r_ell.values))),
                             "J_2": float(np.max(np.abs(D["J_2"].values - T.J_r_2ell.values)))}
    PF = pd.read_csv(os.path.join(REF, "pf_descriptors.csv")).set_index("id").loc[D.id]
    D["R_sigma"], D["R_path"], D["J_psi"] = PF.R_sigma.values, PF.R_Gpath.values, PF.J_psi.values
    rep["max_abs_diff_J0_vs_Jpsi"] = float(np.max(np.abs(D["J_0"].values - D["J_psi"].values)))

    # 4. frozen registry and exclusion of the design cells
    reg = frozen.build_registry()
    fz = set(reg["hashes"])
    dh = [canonical_hash(m) for m in masks]
    D["canonical_hash"] = dh
    rep["frozen_sources"] = reg["sources"]
    rep["design_cells_in_frozen_registry"] = int(sum(h in fz for h in dh))
    rep["design_unique_canonical"] = int(len(set(dh)))
    rep["runtime_s"] = round(time.time() - t0, 1)
    rep["max_iters"] = int(D.iters.max()); rep["max_eq_err"] = float(D.eq_err.max())

    D.to_csv(os.path.join(OUT, "design_cells.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "design_cells.npz"), mask=np.packbits(masks, axis=-1), S2=S2,
                        stress=np.array([res[k]["stress"] for k in range(len(D))]),
                        sbar=np.array([res[k]["sbar"] for k in range(len(D))]), N=masks.shape[1])
    json.dump(rep, open(os.path.join(OUT, "stage0_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
