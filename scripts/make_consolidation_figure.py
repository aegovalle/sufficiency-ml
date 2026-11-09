"""Consolidation figure: Omega/Omega_null (paired subsampling, 5-95 %, pooled over seeds) and
cross-fitted McShane R2 on test2 for R_E and R_H(beta=20), with the oracle floors.
Run:  .venv\\Scripts\\python scripts\\make_consolidation_figure.py -> results/fig_consolidation.png"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
M = pd.read_csv(os.path.join(R, "consolidation_metrics.csv"))
S = pd.read_csv(os.path.join(R, "consolidation_subsample.csv"))
FAM = [("ORACLE_noise", "oracle (noise floor)"), ("ORACLE_10pct", "oracle (10% noise)"), ("E0", "E0"), ("E1", "E1"),
       ("GL", "GL"), ("MKS10", "MKS10"), ("PCA10", "PCA10"), ("ae_d8", "AE d=8"), ("ae_d32", "AE d=32"),
       ("sup_R_E_gap_d8", "E-circ[R_E]"), ("sup_RH_b2_gem2_d8", r"E-circ[$R_H$,$\beta$=2]"),
       ("sup_RH_b20_gem2_d8", r"E-circ[$R_H$,$\beta$=20]"), ("sup_RH_b40_gem2_d8", r"E-circ[$R_H$,$\beta$=40]")]
COL = {"oracle": "0.6", "solved": "tab:green", "geometry": "tab:blue", "learned": "tab:red"}
fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharey="row")
for c, t in enumerate(("R_E", "RH_b20")):
    g = S[S.target == t]; m = M[(M.target == t) & (M.set == "test2")]
    for k, (f, lab) in enumerate(FAM):
        reps = [r for r in g.rep.unique() if r == f or r.rsplit("_s", 1)[0] == f]
        v = g[g.rep.isin(reps)].ratio.values
        kind = m[m.family == f].kind.iloc[0]
        ax[0, c].errorbar(k, np.median(v), yerr=[[np.median(v) - np.percentile(v, 5)], [np.percentile(v, 95) - np.median(v)]],
                          fmt="o", color=COL[kind], capsize=3)
        r2 = m[m.family == f].R2_mcshane
        ax[1, c].errorbar(k, r2.median(), yerr=[[r2.median() - r2.min()], [r2.max() - r2.median()]], fmt="s",
                          color=COL[kind], capsize=3)
    for a in ax[:, c]:
        a.set_xticks(range(len(FAM))); a.set_xticklabels([l for _, l in FAM], rotation=70, fontsize=7); a.grid(alpha=0.3)
    ax[0, c].set_title({"R_E": "$R_E$ (exact positive control: E0)", "RH_b20": r"$R_H$, $\beta=20$"}[t])
ax[0, 0].set_ylabel(r"$\Omega/\Omega^{\rm null}$, test2 (subsampling 5-95%)")
ax[1, 0].set_ylabel(r"cross-fitted McShane $R^2$ (seed range)")
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_consolidation.png"), dpi=160)
print("ok")
