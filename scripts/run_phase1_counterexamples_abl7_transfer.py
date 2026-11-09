"""Phase 1 Analysis (Draft v0.2):
1. Experiment F / H5 / I17: Systematic physical counterexample tracking across seeds & methods
   (MSE vs SBR-CF lambda=1.0, 3.0 vs RankSim gamma=1.0 vs RNC tau=1.0) on test2 and ood_morphology.
2. Ablation 7 (Section 15 & Section 19.7): Metric sensitivity (d_white vs d_coord vs d_N).
3. I16 Complete Cross-Horizon Transfer Matrix: transferring beta=20 encoders to R_E, RH_b2, RH_b20, RH_b40.

Outputs:
  results/counterexamples_h5.json
  results/counterexamples_top10.csv
  results/ablation7_metrics.csv
  results/transfer_matrix_i16.csv
  results/fig_counterexample_resolution.png
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from suffml.data.datasets import load_pilot
from suffml.analysis.evaluate import evaluate, noise_floor
from suffml.metrics import WhiteningTransform, Standardizer, crossfit_errors, select_lambda, mcshane_mid

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")
MD = os.path.join(R, "models")

FAMILIES = {
    "MSE": ["sup_RH_b20_gem2_d8_s0", "sup_RH_b20_gem2_d8_s1", "sup_RH_b20_gem2_d8_s2"],
    "SBR-CF (lam=1.0)": ["obj_RH_b20_sbrcf1_d8_s0", "obj_RH_b20_sbrcf1_d8_s1", "obj_RH_b20_sbrcf1_d8_s2"],
    "SBR-CF (lam=3.0)": ["obj_RH_b20_sbrcf3_d8_s0", "obj_RH_b20_sbrcf3_d8_s1", "obj_RH_b20_sbrcf3_d8_s2"],
    "SBR-in (lam=1.0)": ["obj_RH_b20_sbrin1_d8_s0", "obj_RH_b20_sbrin1_d8_s1", "obj_RH_b20_sbrin1_d8_s2"],
    "RankSim (gam=1.0)": ["obj_RH_b20_ranksimg1_d8_s0", "obj_RH_b20_ranksimg1_d8_s1", "obj_RH_b20_ranksimg1_d8_s2"],
    "RNC (tau=1.0)": ["obj_RH_b20_rnct1_d8_s0", "obj_RH_b20_rnct1_d8_s1", "obj_RH_b20_rnct1_d8_s2"],
}


def load_embeddings():
    embs = {}
    for fam, tags in FAMILIES.items():
        for s, tag in enumerate(tags):
            f = os.path.join(MD, f"{tag}_emb.npz")
            E = np.load(f)
            embs[(fam, s)] = {"tag": tag, "Z": E["Z"], "Zn": E["Zn"] / np.sqrt(E["Zn"].shape[1])}
    return embs


def run_counterexamples_h5(P, D, idx, embs):
    print("=== 1. Running Experiment F / H5 (Counterexample Tracking) ===", flush=True)
    y = D["RH_b20"].values
    tr, va = idx["train"], idx["val"]
    sd_tr = y[tr].std()
    grid = sd_tr * np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])

    # Precompute whitened & trace-normalized embeddings and McShane predictions on test2 and ood_morphology
    cache = {}
    for (fam, s), d in embs.items():
        Z = d["Z"]
        Zn = d["Zn"]
        T = WhiteningTransform().fit(Z[tr])
        Zw_tr, Zw_va = T.transform(Z[tr]), T.transform(Z[va])
        Lam_w = select_lambda(Zw_tr, y[tr], Zw_va, y[va], grid)
        Lam_n = select_lambda(Zn[tr], y[tr], Zn[va], y[va], grid)
        cache[(fam, s)] = {"T": T, "Lam_w": Lam_w, "Lam_n": Lam_n, "Zw": {}, "Zn": {}, "err_w": {}}
        for sname in ("test2", "ood_morphology"):
            ix = idx[sname]
            zw = T.transform(Z[ix])
            zn = Zn[ix]
            cache[(fam, s)]["Zw"][sname] = zw
            cache[(fam, s)]["Zn"][sname] = zn
            pred_w = mcshane_mid(Zw_tr, y[tr], zw, Lam_w)
            cache[(fam, s)]["err_w"][sname] = np.abs(pred_w - y[ix])

    h5_out = {}
    top_rows = []

    for sname in ("test2", "ood_morphology"):
        ix = idx[sname]
        ys = y[ix]
        n = len(ys)
        II, JJ = np.triu_indices(n, 1)
        dy = np.abs(ys[II] - ys[JJ])
        nq = int(np.ceil(0.01 * len(dy) - 1e-9))

        # Distance matrices and quantiles for every (fam, s)
        d_mat = {}
        q_mat = {}
        for (fam, s) in embs:
            dw = pdist(cache[(fam, s)]["Zw"][sname])
            order = np.argsort(dw)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(dw) + 1)
            d_mat[(fam, s)] = dw
            q_mat[(fam, s)] = ranks / len(dw)

        # (A) Track each MSE seed's exact Omega(q=0.01) pair across all methods (same seed and 3-seed median)
        seed_pairs = []
        for s in range(3):
            dw_mse = d_mat[("MSE", s)]
            dcut = np.partition(dw_mse, nq - 1)[nq - 1]
            sel = np.flatnonzero(dw_mse <= dcut + 1e-12)
            k_worst = sel[np.argmax(dy[sel])]
            i_loc, j_loc = int(II[k_worst]), int(JJ[k_worst])
            ci, cj = int(ix[i_loc]), int(ix[j_loc])
            pair_info = {
                "mse_seed": s,
                "cell_i": ci,
                "cell_j": cj,
                "dy": float(dy[k_worst]),
                "RH_i": float(ys[i_loc]),
                "RH_j": float(ys[j_loc]),
                "RE_i": float(D.R_E[ci]),
                "RE_j": float(D.R_E[cj]),
                "f_i": float(D.f[ci]),
                "f_j": float(D.f[cj]),
                "Jgap_m_i": float(D.Jgap_m[ci]),
                "Jgap_m_j": float(D.Jgap_m[cj]),
                "family_i": str(D.family[ci]),
                "family_j": str(D.family[cj]),
                "methods": {},
            }
            for fam in FAMILIES:
                ds = [float(d_mat[(fam, ss)][k_worst]) for ss in range(3)]
                qs = [float(q_mat[(fam, ss)][k_worst]) for ss in range(3)]
                slopes = [float(dy[k_worst] / max(d_mat[(fam, ss)][k_worst], 1e-9)) for ss in range(3)]
                errs = [float(max(cache[(fam, ss)]["err_w"][sname][i_loc], cache[(fam, ss)]["err_w"][sname][j_loc])) for ss in range(3)]
                pair_info["methods"][fam] = {
                    "d_same_seed": ds[s],
                    "q_same_seed": qs[s],
                    "slope_same_seed": slopes[s],
                    "max_err_same_seed": errs[s],
                    "d_median": float(np.median(ds)),
                    "q_median": float(np.median(qs)),
                    "slope_median": float(np.median(slopes)),
                    "max_err_median": float(np.median(errs)),
                    "in_q01_count": int(sum(q_ <= 0.01 for q_ in qs)),
                }
            seed_pairs.append(pair_info)

        # (B) Consensus top-10 worst MSE pairs (mean quantile across 3 MSE seeds <= 0.01, sorted by dy descending)
        mean_q_mse = np.mean([q_mat[("MSE", s)] for s in range(3)], axis=0)
        mean_d_mse = np.mean([d_mat[("MSE", s)] for s in range(3)], axis=0)
        cand = np.flatnonzero(mean_q_mse <= 0.01)
        top10_idx = cand[np.argsort(dy[cand])[::-1][:10]]
        for rank, k_idx in enumerate(top10_idx, 1):
            i_loc, j_loc = int(II[k_idx]), int(JJ[k_idx])
            ci, cj = int(ix[i_loc]), int(ix[j_loc])
            row = {
                "set": sname,
                "rank": rank,
                "cell_i": ci,
                "cell_j": cj,
                "dy": float(dy[k_idx]),
                "dE1": float(np.linalg.norm(
                    (D.loc[ci, ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]].values.astype(float) -
                     D.loc[cj, ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]].values.astype(float))
                )),
                "d_Jgap_m": float(abs(D.Jgap_m[ci] - D.Jgap_m[cj])),
            }
            for fam, short in [
                ("MSE", "mse"),
                ("SBR-CF (lam=1.0)", "sbrcf1"),
                ("SBR-CF (lam=3.0)", "sbrcf3"),
                ("RankSim (gam=1.0)", "ranksim"),
                ("RNC (tau=1.0)", "rnc"),
            ]:
                ds = [d_mat[(fam, s)][k_idx] for s in range(3)]
                qs = [q_mat[(fam, s)][k_idx] for s in range(3)]
                errs = [max(cache[(fam, s)]["err_w"][sname][i_loc], cache[(fam, s)]["err_w"][sname][j_loc]) for s in range(3)]
                row[f"d_{short}_med"] = float(np.median(ds))
                row[f"q_{short}_med"] = float(np.median(qs))
                row[f"slope_{short}_med"] = float(dy[k_idx] / max(np.median(ds), 1e-9))
                row[f"err_{short}_med"] = float(np.median(errs))
                row[f"in_q01_{short}"] = int(sum(q_ <= 0.01 for q_ in qs))
            top_rows.append(row)

        # Summary of how many of the top-25 MSE consensus collision pairs (dy > 1.0) are pushed out of q <= 1% by SBR-CF
        cand_severe = cand[dy[cand] >= np.quantile(dy[cand], 0.90)]  # top 10% severest collisions inside MSE's 1% ball
        res_summary = {"n_cand_q01": int(len(cand)), "n_severe_top10pct": int(len(cand_severe)), "by_method": {}}
        for fam in FAMILIES:
            med_q = np.median([q_mat[(fam, s)][cand_severe] for s in range(3)], axis=0)
            med_d = np.median([d_mat[(fam, s)][cand_severe] for s in range(3)], axis=0)
            med_slope = dy[cand_severe] / np.maximum(med_d, 1e-9)
            res_summary["by_method"][fam] = {
                "frac_pushed_out_of_q01": float(np.mean(med_q > 0.01)),
                "median_q_after": float(np.median(med_q)),
                "median_d_after": float(np.median(med_d)),
                "median_slope_after": float(np.median(med_slope)),
            }

        h5_out[sname] = {"seed_omega_pairs": seed_pairs, "severe_collision_resolution": res_summary}

    # Generate publication figure for the #1 consensus MSE counterexample on test2
    top1_test2 = [r for r in top_rows if r["set"] == "test2"][0]
    ci, cj = top1_test2["cell_i"], top1_test2["cell_j"]
    fig = plt.figure(figsize=(11.5, 4.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.35], wspace=0.28, hspace=0.32)

    for col, k in enumerate((ci, cj)):
        ax_m = fig.add_subplot(gs[0, col])
        ax_m.imshow(P["mask"][k].T, origin="lower", cmap="gray_r")
        ax_m.set_title(
            f"Cell {k} ({D.family[k]})\n$R_H={y[k]:.2f}$, $R_E={D.R_E[k]:.3f}$, $f={D.f[k]:.2f}$",
            fontsize=8.5,
        )
        ax_m.set_xticks([]); ax_m.set_yticks([])

        ax_s = fig.add_subplot(gs[1, col])
        s = P["sbar"][k].astype(float).copy()
        s[P["mask"][k]] = np.nan
        ax_s.imshow(s.T, origin="lower", cmap="inferno")
        mx = np.unravel_index(np.nanargmax(s), s.shape)
        ax_s.plot(*mx, "c+", ms=10, mew=1.5)
        ax_s.set_title(f"Nonlocal stress $s$ ($J_{{\\rm gap}}^m={D.Jgap_m[k]:.2f}$)", fontsize=8.5)
        ax_s.set_xticks([]); ax_s.set_yticks([])

    ax_bar = fig.add_subplot(gs[:, 2])
    fams_plot = ["MSE", "RankSim (gam=1.0)", "RNC (tau=1.0)", "SBR-CF (lam=1.0)", "SBR-CF (lam=3.0)"]
    short_keys = ["mse", "ranksim", "rnc", "sbrcf1", "sbrcf3"]
    labels_plot = ["MSE", "RankSim\n($\\gamma=1$)", "RNC\n($\\tau=1$)", "SBR-CF\n($\\lambda=1$)", "SBR-CF\n($\\lambda=3$)"]
    # Plot median slope |Delta R_H| / d_white across the top-10 consensus MSE counterexamples on test2
    t2_rows = [r for r in top_rows if r["set"] == "test2"]
    med_slopes = [np.median([r[f"slope_{sk}_med"] for r in t2_rows]) for sk in short_keys]
    min_slopes = [np.percentile([r[f"slope_{sk}_med"] for r in t2_rows], 25) for sk in short_keys]
    max_slopes = [np.percentile([r[f"slope_{sk}_med"] for r in t2_rows], 75) for sk in short_keys]
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c"]
    x_pos = np.arange(len(fams_plot))
    ax_bar.bar(x_pos, med_slopes, color=colors, alpha=0.85, width=0.58)
    ax_bar.errorbar(
        x_pos,
        med_slopes,
        yerr=[np.array(med_slopes) - np.array(min_slopes), np.array(max_slopes) - np.array(med_slopes)],
        fmt="none",
        ecolor="black",
        capsize=4,
        lw=1.2,
    )
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels_plot, fontsize=8)
    ax_bar.set_ylabel(r"Local Lipschitz slope $|\Delta R_H| / d_{\rm white}$", fontsize=9)
    ax_bar.set_title("Resolution of Top-10 MSE Counterexamples\n(median $\\pm$ IQR on test2)", fontsize=9)
    ax_bar.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Physical Counterexample Pair ({ci}, {cj}): $|\\Delta R_H|={top1_test2['dy']:.2f}$ "
        f"($d_{{\\rm white}}$: MSE={top1_test2['d_mse_med']:.2f} [$q={100*top1_test2['q_mse_med']:.2f}\\%$] "
        f"$\\rightarrow$ SBR-CF={top1_test2['d_sbrcf3_med']:.2f} [$q={100*top1_test2['q_sbrcf3_med']:.1f}\\%$])",
        fontsize=9.5,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(os.path.join(R, "fig_counterexample_resolution.png"), dpi=180)
    plt.close(fig)

    json.dump(h5_out, open(os.path.join(R, "counterexamples_h5.json"), "w"), indent=1)
    pd.DataFrame(top_rows).to_csv(os.path.join(R, "counterexamples_top10.csv"), index=False)
    print("Saved counterexamples_h5.json, counterexamples_top10.csv, fig_counterexample_resolution.png", flush=True)
    return h5_out


def evaluate_custom_metric(Z_dict, y_dict, target, rng, metric_mode="white"):
    """Evaluate under 'white' (d_white), 'std' (d_coord), or 'norm' (already trace-normalized d_N)."""
    if metric_mode == "white":
        T = WhiteningTransform().fit(Z_dict["train"])
        Zn = {k: T.transform(v) for k, v in Z_dict.items()}
    elif metric_mode == "std":
        T = Standardizer().fit(Z_dict["train"])
        Zn = {k: T.transform(v) for k, v in Z_dict.items()}
    elif metric_mode == "norm":
        Zn = Z_dict
    else:
        raise ValueError(metric_mode)

    # Delegate by passing identity-standardized or directly using evaluate's logic
    from suffml.metrics import omega_ratio, lambda_star, exact_barrier
    from suffml.analysis.evaluate import ridge_fit_predict, r2

    sd = y_dict["train"].std()
    grid = sd * np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])
    Lam = select_lambda(Zn["train"], y_dict["train"], Zn["val"], y_dict["val"], grid)
    eps = noise_floor(target)
    out = {}
    for s in ("test2", "ood"):
        zs, ys = Zn[s], y_dict[s]
        p = ridge_fit_predict(Zn["train"], y_dict["train"], zs)
        om = omega_ratio(zs, ys, rng, q=0.01, nperm=200, return_details=True)
        cf = crossfit_errors(Zn["train"], y_dict["train"], zs, ys, Lam)
        ls = lambda_star(zs, ys, eps)
        lsn = np.median([lambda_star(zs, rng.permutation(ys), eps) for _ in range(30)])
        sds = ys.std()
        out[s] = {
            "R2_ridge": r2(ys, p),
            "R2_mcshane": 1 - cf["rmse"] ** 2 / sds ** 2,
            "omega_ratio": om["ratio"],
            "cf_rmse_sd": cf["rmse"] / sds,
            "cf_max_sd": cf["max"] / sds,
            "lam_star_ratio": float(ls / lsn) if lsn > 0 else np.nan,
        }
    return out


def run_ablation7_metrics(D, idx, embs):
    print("=== 2. Running Ablation 7 (Metric Sensitivity: d_white vs d_coord vs d_N) ===", flush=True)
    y = D["RH_b20"].values
    ymap = {
        "train": y[idx["train"]],
        "val": y[idx["val"]],
        "test2": y[idx["test2"]],
        "ood": y[idx["ood_morphology"]],
    }
    rows = []
    for (fam, s), d in embs.items():
        for m_mode in ("white", "std", "norm"):
            src = d["Zn"] if m_mode == "norm" else d["Z"]
            Zs = {
                "train": src[idx["train"]],
                "val": src[idx["val"]],
                "test2": src[idx["test2"]],
                "ood": src[idx["ood_morphology"]],
            }
            res = evaluate_custom_metric(Zs, ymap, "RH_b20", np.random.default_rng(1000 + s), metric_mode=m_mode)
            for sname in ("test2", "ood"):
                rows.append({"family": fam, "seed": s, "metric": m_mode, "set": sname, **res[sname]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R, "ablation7_metrics.csv"), index=False)
    print("Saved ablation7_metrics.csv", flush=True)
    return df


def run_transfer_matrix_i16(D, idx, embs):
    print("=== 3. Running I16 Complete Cross-Horizon Transfer Matrix ===", flush=True)
    # Also load dedicated MSE encoders trained on R_E, RH_b2, RH_b40 for direct comparison
    extra_fams = {
        "MSE (trained on R_E)": [f"sup_R_E_gap_d8_s{s}" for s in range(3)],
        "MSE (trained on RH_b2)": [f"sup_RH_b2_gem2_d8_s{s}" for s in range(3)],
        "MSE (trained on RH_b40)": [f"sup_RH_b40_gem2_d8_s{s}" for s in range(3)],
    }
    all_embs = dict(embs)
    for fam, tags in extra_fams.items():
        for s, tag in enumerate(tags):
            f = os.path.join(MD, f"{tag}_emb.npz")
            if os.path.exists(f):
                E = np.load(f)
                all_embs[(fam, s)] = {"tag": tag, "Z": E["Z"], "Zn": E["Zn"] / np.sqrt(E["Zn"].shape[1])}

    rows = []
    for t in ("R_E", "RH_b2", "RH_b20", "RH_b40"):
        yt = D[t].values
        ymap = {
            "train": yt[idx["train"]],
            "val": yt[idx["val"]],
            "test2": yt[idx["test2"]],
            "ood": yt[idx["ood_morphology"]],
        }
        for (fam, s), d in all_embs.items():
            Z = d["Z"]
            Zs = {
                "train": Z[idx["train"]],
                "val": Z[idx["val"]],
                "test2": Z[idx["test2"]],
                "ood": Z[idx["ood_morphology"]],
            }
            res = evaluate_custom_metric(Zs, ymap, t, np.random.default_rng(2000 + abs(hash((fam, s, t))) % 10000), metric_mode="white")
            for sname in ("test2", "ood"):
                rows.append({"eval_target": t, "family": fam, "seed": s, "set": sname, **res[sname]})
        print(f"  Target {t} transfer evaluated", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R, "transfer_matrix_i16.csv"), index=False)
    print("Saved transfer_matrix_i16.csv", flush=True)
    return df


def main():
    P = load_pilot(fields=True)
    D = P["table"]
    SETS = ("train", "val", "test2", "ood_morphology")
    idx = {s: np.array(P["splits"][s]) for s in SETS}
    embs = load_embeddings()

    h5 = run_counterexamples_h5(P, D, idx, embs)
    abl7 = run_ablation7_metrics(D, idx, embs)
    i16 = run_transfer_matrix_i16(D, idx, embs)

    print("\n--- Summary of H5 Severe Collision Resolution (test2 & ood_morphology) ---")
    for sname in ("test2", "ood_morphology"):
        print(f"[{sname}]")
        for fam, stats in h5[sname]["severe_collision_resolution"]["by_method"].items():
            print(f"  {fam:20s}: frac_pushed_out={stats['frac_pushed_out_of_q01']:.2f}, "
                  f"med_q={100*stats['median_q_after']:.2f}%, med_d={stats['median_d_after']:.3f}, "
                  f"med_slope={stats['median_slope_after']:.2f}")

    print("\n--- Summary of Ablation 7 on test2 (median across 3 seeds) ---")
    piv7 = abl7[abl7["set"] == "test2"].groupby(["metric", "family"])[["omega_ratio", "R2_mcshane", "cf_max_sd"]].median()
    print(piv7.round(3).to_string())

    print("\n--- Summary of I16 Transfer Matrix on test2 (omega_ratio / R2_mcshane median) ---")
    piv16 = i16[i16["set"] == "test2"].groupby(["eval_target", "family"])[["omega_ratio", "R2_mcshane", "R2_ridge"]].median()
    print(piv16.round(3).to_string())


if __name__ == "__main__":
    main()
