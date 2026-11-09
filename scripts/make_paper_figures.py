"""Generate publication-quality figures for the Sufficiency-Aware Representation Learning manuscript:
1. fig_lambda_and_dim_sweep.png / .pdf:
   - Panel (a): Effect of lambda_cf in {0, 0.1, 0.3, 1.0, 3.0} on Omega/Omega_null (test2 n=1000 & OOD_morph n=300).
   - Panel (b): Latent dimension sweep d in {4, 8, 16} showing dimensional dilution in MSE vs. SBR-CF.
2. fig_ablation7_metrics.png / .pdf:
   - Evaluation metric sensitivity (d_white vs. d_std vs. d_N) across objective families on test2,
     highlighting SBR-CF invariance and RNC's 1D collapse under Mahalanobis whitening.
3. fig_counterexample_resolution.png / .pdf:
   - Polished layout of consensus rank #1 MSE collision pair (2751, 3270) and top-10 slope resolution.
4. fig_transfer_and_pf.png / .pdf:
   - Panel (a): Cross-horizon transfer matrix (beta=20 encoders evaluated on R_E, RH_b2, RH_b20, RH_b40).
   - Panel (b): Out-of-sample phase-field fracture transfer (R_sigma) across Design 180 CV, Set B, Set O, Set C1, Set C2.
5. fig_phase3_ablations.png / .pdf:
   - Panel (a): Objective component & pooling ablations on E-circ (test2 & ood).
   - Panel (b): Backbone & periodic/D2 invariance defect (E-FNO, E-circ, E-std, E-std + L_inv).
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

from suffml.data.datasets import load_pilot

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
})


def make_fig_lambda_and_dim():
    rob = json.load(open(os.path.join(R, "robust_summary.json")))
    t2 = rob["families"]["test2"]
    ood = rob["families"]["ood"]

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1))

    # Panel (a): lambda_cf curve at d=8 (full-set medians and 3-seed min/max)
    lams = [0.0, 0.1, 0.3, 1.0, 3.0]
    fams_l = [
        "sup_RH_b20_gem2_d8",
        "obj_RH_b20_sbrcf01_d8",
        "obj_RH_b20_sbrcf03_d8",
        "obj_RH_b20_sbrcf1_d8",
        "obj_RH_b20_sbrcf3_d8",
    ]
    x = np.arange(len(lams))
    om_t2 = [t2[f]["omega_ratio"][0] for f in fams_l]
    om_t2_lo = [t2[f]["omega_ratio"][0] - t2[f]["omega_ratio"][1] for f in fams_l]
    om_t2_hi = [t2[f]["omega_ratio"][2] - t2[f]["omega_ratio"][0] for f in fams_l]

    om_ood = [ood[f]["omega_ratio"][0] for f in fams_l]
    om_ood_lo = [ood[f]["omega_ratio"][0] - ood[f]["omega_ratio"][1] for f in fams_l]
    om_ood_hi = [ood[f]["omega_ratio"][2] - ood[f]["omega_ratio"][0] for f in fams_l]

    ax[0].errorbar(x - 0.06, om_t2, yerr=[om_t2_lo, om_t2_hi], fmt="-o", color="#1f77b4", capsize=4, lw=1.8,
                   label=r"IID $\mathrm{test}_2$ ($n=1000$)")
    ax[0].errorbar(x + 0.06, om_ood, yerr=[om_ood_lo, om_ood_hi], fmt="--s", color="#2ca02c", capsize=4, lw=1.8,
                   label=r"$\mathrm{OOD}_{\rm morph}$ ($n=300$)")
    ax[0].axhline(t2["ORACLE_10pct"]["omega_ratio"][0], color="gray", ls=":", lw=1.3, label=r"Oracle reference ($10\%$ noise)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["0\n(MSE)", "0.1", "0.3", "1.0", "3.0"])
    ax[0].set_xlabel(r"Cross-fitted barrier weight $\lambda_{\rm cf}$ ($d=8$)")
    ax[0].set_ylabel(r"Worst-case insufficiency $\Omega/\Omega^{\rm null}$ ($q=1\%$)")
    ax[0].set_title(r"(a) Barrier weight sweep across $\lambda_{\rm cf}$")
    ax[0].grid(alpha=0.3)
    ax[0].legend(loc="upper right")

    # Panel (b): Latent dimension sweep d in {4, 8, 16}
    dims = [4, 8, 16]
    mse_d = ["obj_RH_b20_mse_d4", "sup_RH_b20_gem2_d8", "obj_RH_b20_mse_d16"]
    sbr_d = ["obj_RH_b20_sbrcf1_d4", "obj_RH_b20_sbrcf1_d8", "obj_RH_b20_sbrcf1_d16"]
    xd = np.arange(len(dims))
    w = 0.34

    mse_vals = [t2[f]["omega_ratio"][0] for f in mse_d]
    mse_lo = [t2[f]["omega_ratio"][0] - t2[f]["omega_ratio"][1] for f in mse_d]
    mse_hi = [t2[f]["omega_ratio"][2] - t2[f]["omega_ratio"][0] for f in mse_d]

    sbr_vals = [t2[f]["omega_ratio"][0] for f in sbr_d]
    sbr_lo = [t2[f]["omega_ratio"][0] - t2[f]["omega_ratio"][1] for f in sbr_d]
    sbr_hi = [t2[f]["omega_ratio"][2] - t2[f]["omega_ratio"][0] for f in sbr_d]

    ax[1].bar(xd - w / 2, mse_vals, width=w, color="#d62728", alpha=0.85, label="MSE only")
    ax[1].errorbar(xd - w / 2, mse_vals, yerr=[mse_lo, mse_hi], fmt="none", ecolor="black", capsize=4, lw=1.2)
    ax[1].bar(xd + w / 2, sbr_vals, width=w, color="#1f77b4", alpha=0.85, label=r"MSE + SBR-CF ($\lambda_{\rm cf}=1$)")
    ax[1].errorbar(xd + w / 2, sbr_vals, yerr=[sbr_lo, sbr_hi], fmt="none", ecolor="black", capsize=4, lw=1.2)

    for i, (m_f, s_f) in enumerate(zip(mse_d, sbr_d)):
        r2m_mse = t2[m_f]["R2_mcshane"][0]
        r2m_sbr = t2[s_f]["R2_mcshane"][0]
        ax[1].text(xd[i] - w / 2, mse_vals[i] + mse_hi[i] + 0.012, f"$R^2_{{\\rm Mc}}={r2m_mse:.2f}$",
                   ha="center", va="bottom", fontsize=7.2, color="#8c1515")
        ax[1].text(xd[i] + w / 2, sbr_vals[i] + sbr_hi[i] + 0.012, f"$R^2_{{\\rm Mc}}={r2m_sbr:.2f}$",
                   ha="center", va="bottom", fontsize=7.2, color="#0d4a7a")

    ax[1].axhline(t2["ORACLE_10pct"]["omega_ratio"][0], color="gray", ls=":", lw=1.3, label="Oracle reference")
    ax[1].set_xticks(xd)
    ax[1].set_xticklabels([f"$d={d}$" for d in dims])
    ax[1].set_ylim(0, 0.57)
    ax[1].set_xlabel(r"Latent dimension $d$")
    ax[1].set_ylabel(r"$\Omega/\Omega^{\rm null}$ on $\mathrm{test}_2$ ($d_{\rm white}$)")
    ax[1].set_title(r"(b) Dimensional dilution in MSE vs. SBR-CF")
    ax[1].grid(axis="y", alpha=0.3)
    ax[1].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_lambda_and_dim_sweep.png"), dpi=180)
    fig.savefig(os.path.join(R, "fig_lambda_and_dim_sweep.pdf"))
    plt.close(fig)


def make_fig_ablation7_metrics():
    df5 = pd.read_csv(os.path.join(R, "phase5_whitening_and_tailmse.csv"))

    fams = [
        ("MSE", "MSE"),
        ("Tail-MSE (lam=3.0)", "Tail-MSE\n($\\lambda=3$)"),
        ("SBR-in (lam=1.0)", "SBR-in\n($\\lambda=1$)"),
        ("RankSim (gam=1.0)", "RankSim\n($\\gamma=1$)"),
        ("RNC (tau=1.0)", "RNC\n($\\tau=1$)"),
        ("SBR-CF (lam=1.0)", "SBR-CF\n($\\lambda=1$)"),
        ("SBR-CF (lam=3.0)", "SBR-CF\n($\\lambda=3$)"),
    ]
    metrics = [
        ("d_white (delta=1e-9)", r"$d_{\rm white}$ ($\delta=10^{-9}$)", "#1f77b4"),
        ("d_reg (val-selected by CF_rmse)", r"$d_{\rm reg}^\star$ (val-tuned $\alpha_{\rm reg}$)", "#9467bd"),
        ("d_std (coord)", r"$d_{\rm std}$ (coord. std.)", "#ff7f0e"),
        ("d_N (LatentNorm)", r"$d_{\mathcal{N}}$ (LatentNorm)", "#2ca02c"),
    ]

    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.35))
    x = np.arange(len(fams))
    w = 0.20

    for panel_idx, (col5, ylabel, title, ylim) in enumerate([
        ("test2_omega_ratio", r"Worst-case insufficiency $\Omega/\Omega^{\rm null}$ ($q=1\%$)", r"(a) Evaluation metric sensitivity ($\Omega/\Omega^{\rm null}$)", (0.0, 0.48)),
        ("test2_cf_max_norm", r"Cross-fitted worst-case error $\mathrm{CF}_{\max}/\sigma_R$", r"(b) Cross-fitted sup-norm error sensitivity", (0.0, 2.55)),
    ]):
        for m_idx, (m_key, m_lab, m_col) in enumerate(metrics):
            meds, lo, hi, xs = [], [], [], []
            for f_idx, (f_key, _) in enumerate(fams):
                vals = df5[(df5["family"] == f_key) & (df5["metric"] == m_key)][col5].values
                med = float(np.median(vals))
                meds.append(med)
                lo.append(med - float(np.min(vals)))
                hi.append(float(np.max(vals)) - med)
                xs.append(x[f_idx] + (m_idx - 1.5) * w)
            ax[panel_idx].bar(xs, meds, width=w, color=m_col, alpha=0.88, label=m_lab if panel_idx == 0 else None)
            ax[panel_idx].errorbar(xs, meds, yerr=[lo, hi], fmt="none", ecolor="black", capsize=2.5, lw=0.9)

        ax[panel_idx].set_xticks(x)
        ax[panel_idx].set_xticklabels([lab for _, lab in fams], fontsize=7.9)
        ax[panel_idx].set_ylim(*ylim)
        ax[panel_idx].set_ylabel(ylabel)
        ax[panel_idx].set_title(title)
        ax[panel_idx].grid(axis="y", alpha=0.3)

    # Annotate RNC sensitivity between d_white and d_reg^*/d_std/d_N
    ax[0].annotate(
        r"1D spectrum ($d_{\rm eff}=1.004$):" + "\n" + r"$d_{\rm reg}^\star: 0.085,\; d_{\rm std}/d_{\mathcal{N}}: 0.148 \rightarrow d_{\rm white}: 0.225$",
        xy=(4 - 1.5 * w, 0.229),
        xytext=(3.75, 0.370),
        ha="center",
        fontsize=7.0,
        color="#8c1515",
        arrowprops=dict(arrowstyle="->", color="#8c1515", lw=1.0),
    )
    ax[1].annotate(
        r"$d_{\rm std}/d_{\mathcal{N}}: 1.028,\; d_{\rm reg}^\star: 1.378 \rightarrow d_{\rm white}: 2.065$",
        xy=(4 - 1.5 * w, 2.08),
        xytext=(3.9, 2.34),
        ha="center",
        fontsize=7.1,
        color="#8c1515",
        arrowprops=dict(arrowstyle="->", color="#8c1515", lw=1.0),
    )
    ax[0].legend(loc="upper left", ncol=2, fontsize=7.3)

    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_ablation7_metrics.png"), dpi=180)
    fig.savefig(os.path.join(R, "fig_ablation7_metrics.pdf"))
    plt.close(fig)


def make_fig_counterexample_polish():
    P = load_pilot(fields=True)
    D = P["table"]
    y = D["RH_b20"].values
    top_df = pd.read_csv(os.path.join(R, "counterexamples_top10.csv"))
    t2_rows = top_df[top_df["set"] == "test2"].reset_index(drop=True)
    top1 = t2_rows.iloc[0]
    ci, cj = int(top1["cell_i"]), int(top1["cell_j"])

    fig = plt.figure(figsize=(11.6, 4.7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.38], wspace=0.28, hspace=0.38)

    for col, k in enumerate((ci, cj)):
        ax_m = fig.add_subplot(gs[0, col])
        ax_m.imshow(P["mask"][k].T, origin="lower", cmap="gray_r")
        ax_m.set_title(
            f"Cell {k} ({D.family[k]})\n$R_H={y[k]:.2f}$, $R_E={D.R_E[k]:.3f}$, $f={D.f[k]:.2f}$",
            fontsize=8.4,
            pad=4,
        )
        ax_m.set_xticks([]); ax_m.set_yticks([])

        ax_s = fig.add_subplot(gs[1, col])
        s = P["sbar"][k].astype(float).copy()
        s[P["mask"][k]] = np.nan
        ax_s.imshow(s.T, origin="lower", cmap="inferno")
        mx = np.unravel_index(np.nanargmax(s), s.shape)
        ax_s.plot(*mx, "c+", ms=10, mew=1.5)
        ax_s.set_title(f"Nonlocal stress $s$ ($J_{{\\rm gap}}^m={D.Jgap_m[k]:.2f}$)", fontsize=8.4, pad=4)
        ax_s.set_xticks([]); ax_s.set_yticks([])

    ax_bar = fig.add_subplot(gs[:, 2])
    short_keys = ["mse", "ranksim", "rnc", "sbrcf1", "sbrcf3"]
    labels_plot = ["MSE", "RankSim\n($\\gamma=1$)", "RNC\n($\\tau=1$)", "SBR-CF\n($\\lambda=1$)", "SBR-CF\n($\\lambda=3$)"]
    med_slopes = [float(np.median(t2_rows[f"slope_{sk}_med"].values)) for sk in short_keys]
    q25 = [float(np.percentile(t2_rows[f"slope_{sk}_med"].values, 25)) for sk in short_keys]
    q75 = [float(np.percentile(t2_rows[f"slope_{sk}_med"].values, 75)) for sk in short_keys]
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c"]
    x_pos = np.arange(len(short_keys))

    ax_bar.bar(x_pos, med_slopes, color=colors, alpha=0.86, width=0.56)
    ax_bar.errorbar(
        x_pos,
        med_slopes,
        yerr=[np.array(med_slopes) - np.array(q25), np.array(q75) - np.array(med_slopes)],
        fmt="none",
        ecolor="black",
        capsize=4,
        lw=1.2,
    )
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels_plot, fontsize=8.2)
    ax_bar.set_ylabel(r"Local Lipschitz slope $|\Delta R_H| / d_{\rm white}$", fontsize=9)
    ax_bar.set_title("Resolution of Top-10 MSE Counterexamples\n(median $\\pm$ IQR on $\\mathrm{test}_2$)", fontsize=9.0, pad=6)
    ax_bar.set_ylim(0, max(q75) * 1.12)
    ax_bar.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Consensus Top MSE Counterexample ({ci}, {cj}): $|\\Delta R_H|={top1['dy']:.2f}$ "
        f"($d_{{\\rm white}}$: MSE={top1['d_mse_med']:.2f} [$q={100*top1['q_mse_med']:.2f}\\%$] "
        f"$\\rightarrow$ SBR-CF($\\lambda=3$)={top1['d_sbrcf3_med']:.2f} [$q={100*top1['q_sbrcf3_med']:.1f}\\%$])",
        fontsize=9.5,
        y=0.97,
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.84, bottom=0.11)
    fig.savefig(os.path.join(R, "fig_counterexample_resolution.png"), dpi=180)
    fig.savefig(os.path.join(R, "fig_counterexample_resolution.pdf"))
    plt.close(fig)


def make_fig_transfer_and_pf():
    tr_df = pd.read_csv(os.path.join(R, "transfer_matrix_i16.csv"))
    pf_df = pd.read_csv(os.path.join(R, "pf_transfer_i19.csv"))

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.45))

    # Panel (a): Cross-horizon transfer on test2
    targets = ["R_E", "RH_b2", "RH_b20", "RH_b40"]
    t_labels = ["$R_E$\n(volume avg.)", r"$R_H(\beta=2)$" + "\n(mild tail)", r"$R_H(\beta=20)$" + "\n(trained target)", r"$R_H(\beta=40)$" + "\n(sharp tail)"]
    fams = ["MSE", "RankSim (gam=1.0)", "RNC (tau=1.0)", "SBR-CF (lam=1.0)", "SBR-CF (lam=3.0)"]
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c"]
    piv = tr_df[tr_df["set"] == "test2"].groupby(["eval_target", "family"])["omega_ratio"].median().unstack()

    x = np.arange(len(targets))
    w = 0.15
    for idx_f, (fam, col) in enumerate(zip(fams, colors)):
        vals = [piv.loc[t, fam] for t in targets]
        ax[0].bar(x + (idx_f - 2) * w, vals, width=w, color=col, alpha=0.88, label=fam)

    ax[0].set_xticks(x)
    ax[0].set_xticklabels(t_labels, fontsize=8.4)
    ax[0].set_ylim(0.0, 0.65)
    ax[0].set_ylabel(r"Transferred $\Omega/\Omega^{\rm null}$ on $\mathrm{test}_2$")
    ax[0].set_title(r"(a) Cross-horizon transfer of $\beta=20$ encoders")
    ax[0].grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=7.3, ncol=2, loc="upper right", frameon=True)

    # Panel (b): Phase-field fracture transfer (R_sigma) across Design 180 CV, Set B, Set O, Set C1, Set C2
    pf_sig = pf_df[pf_df["target"] == "R_sigma"].groupby("representation")[
        ["R2_cv_design180", "R2_Set_B", "R2_Set_O", "R2_Set_C1", "R2_Set_C2"]
    ].median()
    reps_show = [
        ("MSE (RH_b20) alone", "MSE alone", "#d62728"),
        ("SBR-CF (lam=1.0) alone", r"SBR-CF ($\lambda=1$) alone", "#1f77b4"),
        ("E1+J_1.5", r"$E_1 + J_{1.5\ell}$ (explicit)", "#7f7f7f"),
        ("E1+J_1.5 + MSE (RH_b20)", r"$E_1 + J_{1.5\ell}$ + MSE", "#ff7f0e"),
        ("E1+J_1.5 + SBR-CF (lam=1.0)", r"$E_1 + J_{1.5\ell}$ + SBR-CF ($\lambda=1$)", "#2ca02c"),
    ]
    cols_pf = ["R2_cv_design180", "R2_Set_B", "R2_Set_O", "R2_Set_C1", "R2_Set_C2"]
    labs_pf = [
        "Design 180\n(5-fold CV)",
        "Set B\n(blind, $n=45$)",
        "Set O\n($\\alpha=6$, $n=12$)",
        "Set C1\n(prosp., $n=30$)",
        "Set C2\n($n_{\\rm inc}=16$" + "\u201325)",
    ]
    xp = np.arange(len(cols_pf))
    wp = 0.155
    for idx_r, (r_key, r_lab, col) in enumerate(reps_show):
        vals = [pf_sig.loc[r_key, c] for c in cols_pf]
        ax[1].bar(xp + (idx_r - 2.0) * wp, vals, width=wp, color=col, alpha=0.88, label=r_lab)

    # Annotate Set C2 values above the bars
    v_e1j_c2 = pf_sig.loc["E1+J_1.5", "R2_Set_C2"]
    v_mse_c2 = pf_sig.loc["E1+J_1.5 + MSE (RH_b20)", "R2_Set_C2"]
    v_comb_c2 = pf_sig.loc["E1+J_1.5 + SBR-CF (lam=1.0)", "R2_Set_C2"]
    ax[1].text(xp[4] - 0.03, v_e1j_c2 + 0.015, f"{v_e1j_c2:.2f}", ha="center", va="bottom", fontsize=6.7, color="#444444")
    ax[1].text(xp[4] + 0.90 * wp, v_mse_c2 + 0.012, f"{v_mse_c2:.2f}", ha="right", va="bottom", fontsize=6.6, color="#994a00")
    ax[1].text(xp[4] + 2.05 * wp, v_comb_c2 + 0.018, f"{v_comb_c2:.2f}", ha="left", va="bottom", fontsize=7.0, color="#176117", fontweight="bold")

    ax[1].axhline(0.70, color="#444444", ls="--", lw=1.0, label=r"Benchmark target ($R^2=0.70$)")
    ax[1].set_xticks(xp)
    ax[1].set_xticklabels(labs_pf, fontsize=8.0)
    ax[1].set_ylim(0.0, 1.28)
    ax[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1].set_ylabel(r"Out-of-sample $R^2$ on peak strength $R_\sigma$")
    ax[1].set_title(r"(b) Frozen transfer to AT1 phase-field fracture ($R_\sigma$)")
    ax[1].grid(axis="y", alpha=0.3)
    ax[1].legend(fontsize=6.9, loc="upper center", ncol=2, frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_transfer_and_pf.png"), dpi=180)
    fig.savefig(os.path.join(R, "fig_transfer_and_pf.pdf"))
    plt.close(fig)


def make_fig_phase3_ablations():
    df = pd.read_csv(os.path.join(R, "ablations_phase3_metrics.csv"))

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Panel (a): Objective & pooling ablations on E-circ (test2 vs ood)
    fams_a = [
        ("E-circ + GAP + MSE", "GAP\nMSE"),
        ("E-circ + GAP + SBR-CF (lam=1)", "GAP\nSBR(1)"),
        ("E-circ + GeM + MSE", "GeM\nMSE"),
        ("E-circ + GeM + SBR-in", "GeM\nSBR-in"),
        ("E-circ + GeM + SBR-CF (no mem)", "GeM\nno mem"),
        ("E-circ + GeM + SBR-CF (no mine)", "GeM\nno mine"),
        ("E-circ + GeM + SBR-CF (full, lam=1)", "GeM\nfull(1)"),
        ("E-circ + GeM + SBR-CF (full, lam=3)", "GeM\nfull(3)"),
    ]
    xa = np.arange(len(fams_a))
    wa = 0.35
    for s_idx, (sname, slab, col) in enumerate([
        ("test2", r"IID $\mathrm{test}_2$ ($n=1000$)", "#1f77b4"),
        ("ood", r"$\mathrm{OOD}_{\rm morph}$ ($n=300$)", "#2ca02c"),
    ]):
        meds, lo, hi = [], [], []
        for f_key, _ in fams_a:
            vals = df[(df["family"] == f_key) & (df["set"] == sname)]["omega_ratio"].values
            m = float(np.median(vals))
            meds.append(m)
            lo.append(m - float(np.min(vals)))
            hi.append(float(np.max(vals)) - m)
        pos = xa + (s_idx - 0.5) * wa
        ax[0].bar(pos, meds, width=wa, color=col, alpha=0.86, label=slab)
        ax[0].errorbar(pos, meds, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3, lw=1.0)

    ax[0].set_xticks(xa)
    ax[0].set_xticklabels([lab for _, lab in fams_a], fontsize=7.6)
    ax[0].set_ylim(0.0, 0.48)
    ax[0].set_ylabel(r"Worst-case insufficiency $\Omega/\Omega^{\rm null}$ ($q=1\%$)")
    ax[0].set_title(r"(a) Objective & pooling ablations on E-circ ($d=8$)")
    ax[0].grid(axis="y", alpha=0.3)
    ax[0].legend(loc="upper right", fontsize=7.5)

    # Panel (b): Backbone & invariance ablations (with inv_defect annotation)
    fams_b = [
        ("E-FNO + GeM + MSE", "E-FNO\nMSE"),
        ("E-FNO + GeM + SBR-CF (lam=1)", "E-FNO\nSBR(1)"),
        ("E-circ + GeM + MSE", "E-circ\nMSE"),
        ("E-circ + GeM + SBR-CF (full, lam=1)", "E-circ\nSBR(1)"),
        ("E-std (no sym) + GeM + MSE", "E-std\nMSE"),
        ("E-std (no sym) + GeM + MSE + L_inv", r"E-std" + "\n" + r"$+\mathcal{L}_{\rm inv}$"),
    ]
    xb = np.arange(len(fams_b))
    wb = 0.35
    max_tops = np.zeros(len(fams_b))
    for s_idx, (sname, slab, col) in enumerate([
        ("test2", r"IID $\mathrm{test}_2$ ($n=1000$)", "#1f77b4"),
        ("ood", r"$\mathrm{OOD}_{\rm morph}$ ($n=300$)", "#2ca02c"),
    ]):
        meds, lo, hi = [], [], []
        for idx_f, (f_key, _) in enumerate(fams_b):
            vals = df[(df["family"] == f_key) & (df["set"] == sname)]["omega_ratio"].values
            m = float(np.median(vals))
            meds.append(m)
            lo.append(m - float(np.min(vals)))
            hi.append(float(np.max(vals)) - m)
            max_tops[idx_f] = max(max_tops[idx_f], float(np.max(vals)))
        pos = xb + (s_idx - 0.5) * wb
        ax[1].bar(pos, meds, width=wb, color=col, alpha=0.86, label=slab)
        ax[1].errorbar(pos, meds, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3, lw=1.0)

    for idx_f, (f_key, _) in enumerate(fams_b):
        dinv = float(df[df["family"] == f_key]["inv_defect"].median())
        txt = f"$\\delta_{{\\rm inv}}={dinv:.1e}$" if dinv < 1e-3 else f"$\\delta_{{\\rm inv}}={dinv:.3f}$"
        if dinv < 1e-3:
            txt = r"$\delta_{\rm inv}<10^{-4}$"
        col_txt = "#8c1515" if dinv > 0.05 else "#222222"
        ax[1].text(xb[idx_f], max_tops[idx_f] + 0.015, txt, ha="center", va="bottom", fontsize=7.0, color=col_txt)

    ax[1].set_xticks(xb)
    ax[1].set_xticklabels([lab for _, lab in fams_b], fontsize=7.9)
    ax[1].set_ylim(0.0, 0.69)
    ax[1].set_ylabel(r"Worst-case insufficiency $\Omega/\Omega^{\rm null}$ ($q=1\%$)")
    ax[1].set_title(r"(b) Backbone transfer (E-FNO) & invariance (E-std)")
    ax[1].grid(axis="y", alpha=0.3)
    ax[1].legend(loc="upper left", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_phase3_ablations.png"), dpi=180)
    fig.savefig(os.path.join(R, "fig_phase3_ablations.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    make_fig_lambda_and_dim()
    make_fig_ablation7_metrics()
    make_fig_counterexample_polish()
    make_fig_transfer_and_pf()
    make_fig_phase3_ablations()
    print(
        "Saved fig_lambda_and_dim_sweep, fig_ablation7_metrics, fig_counterexample_resolution, "
        "fig_transfer_and_pf, and fig_phase3_ablations (.png/.pdf)",
        flush=True,
    )
