"""Gate figures: (1) average versus worst case (R2 against Omega/Omega_null) on the test set for
R_E and R_H(beta=20); (2) the latent counterexample of the best learned encoder for R_H.
Run:  .venv\\Scripts\\python scripts\\make_gate_figures.py   ->  results/fig_gate_*.png"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from suffml.data.datasets import load_pilot
from suffml.metrics import WhiteningTransform, latent_counterexamples

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")
A = pd.read_csv(os.path.join(R, "gate_metrics.csv"))
COL = {"geometry": "tab:blue", "solved": "tab:green", "learned": "tab:red"}

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for a, t, lab in zip(ax, ("R_E", "RH_b20"), ("$R_E$", r"$R_H$ ($\beta=20$)")):
    g = A[(A.metric == "white") & (A.set == "test") & (A.target == t)]
    for _, r in g.iterrows():
        a.scatter(r.R2_mcshane, r.omega_ratio, c=COL[r.kind], s=30)
        a.annotate(r.rep.replace("Ecirc", "E-circ"), (r.R2_mcshane, r.omega_ratio), fontsize=6.5,
                   xytext=(3, 2), textcoords="offset points")
    a.set_xlabel(r"cross-fitted McShane $R^2$ (test)"); a.set_ylabel(r"$\Omega/\Omega^{\rm null}$ at $q=1\%$ (test)")
    a.set_title(lab); a.grid(alpha=0.3)
for k, c in COL.items():
    ax[0].scatter([], [], c=c, label=k)
ax[0].legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_gate_avg_vs_worst.png"), dpi=160)

# latent counterexample of E-circ[RH_b20, gap] on the test set
P = load_pilot(fields=True); D = P["table"]
te, tr = np.array(P["splits"]["test"]), np.array(P["splits"]["train"])
E = np.load(os.path.join(R, "models", "sup_RH_b20_gap_d8_s0_emb.npz"))["Z"]
T = WhiteningTransform().fit(E[tr])
y = D["RH_b20"].values
ce = latent_counterexamples(T.transform(E[te]), y[te], ids=te, q=0.01)
i, j = ce["omega_pair"]["ids"]
fig, ax = plt.subplots(2, 2, figsize=(7, 7))
for c, k in enumerate((i, j)):
    ax[0, c].imshow(P["mask"][k].T, origin="lower", cmap="gray_r")
    ax[0, c].set_title(f"cell {k}: $R_H$={y[k]:.2f}, $R_E$={D.R_E[k]:.3f}\nf={D.f[k]:.3f}, n={D.n[k]}, {D.family[k]}", fontsize=8)
    s = P["sbar"][k].astype(float); s[P["mask"][k]] = np.nan
    im = ax[1, c].imshow(s.T, origin="lower", cmap="inferno")
    mx = np.unravel_index(np.nanargmax(s), s.shape)
    ax[1, c].plot(*mx, "c+", ms=12)
    ax[1, c].set_title("normalized nonlocal matrix stress $s$", fontsize=8)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"Latent counterexample of E-circ[$R_H$, GAP]: d={ce['omega_pair']['d']:.3f}, "
             f"$\\Delta R_H$={ce['omega_pair']['dy']:.2f}", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_gate_counterexample.png"), dpi=160)
print(ce["omega_pair"])
