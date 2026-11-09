"""Experiment A (Draft v0.2, Section 17.1; checklist I07): physics and statistical baselines on
the pilot corpus, evaluated with all sufficiency metrics.

Representations (label: a priori geometry vs solved field):
  G1 = (f, rho_I)                       geometry
  GL = (f, rho_I, A_lig)                geometry
  MKS10, MKS20: PCA of the two-point autocovariance, fitted on train      geometry
  PCA10, PCA20: PCA of the flattened image, fitted on train               geometry (not shift-invariant)
  E0 = (f, phase mean strains)          solved field
  E1 = E0 + Jensen gaps                 solved field
Targets: R_E and R_H(beta) for beta in {2, 5, 10, 20, 40}.

Run:  .venv\\Scripts\\python scripts\\run_baselines.py
Output: results/baselines.csv, results/baselines.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd

from suffml.data.datasets import load_pilot
from suffml.analysis.evaluate import evaluate

ROOT = os.path.join(os.path.dirname(__file__), "..")
TARGETS = ["R_E", "RH_b2", "RH_b5", "RH_b10", "RH_b20", "RH_b40"]
E0 = ["f", "e11_m", "e22_m", "e12_m"]


def pca_fit(Xtr, k):
    mu = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    return lambda X: (X - mu) @ Vt[:k].T


def representations(P):
    D, sp = P["table"], P["splits"]
    idx = {s: np.array(sp[s]) for s in ("train", "val", "test", "ood_morphology")}
    reps = {}
    for name, cols, kind in (("G1", ["f", "rhoI"], "geometry"), ("GL", ["f", "rhoI", "A_lig"], "geometry"),
                             ("E0", E0, "solved"), ("E1", E0 + ["Jgap_m", "Jgap_i"], "solved")):
        X = D[cols].values.astype(float)
        reps[name] = (kind, {s: X[i] for s, i in idx.items()})
    S2 = P["S2"].reshape(len(D), -1).astype(float)
    img = P["mask"].reshape(len(D), -1).astype(float)
    for base, X in (("MKS", S2), ("PCA", img)):
        for k in (10, 20):
            f = pca_fit(X[idx["train"]], k)
            reps[f"{base}{k}"] = ("geometry", {s: f(X[i]) for s, i in idx.items()})
    return reps, idx


def main():
    P = load_pilot()
    D = P["table"]
    reps, idx = representations(P)
    rows, full = [], {}
    for t in TARGETS:
        y = {s: D[t].values[i] for s, i in idx.items()}
        y["ood"] = y.pop("ood_morphology")
        for name, (kind, Z) in reps.items():
            Z = dict(Z); Z["ood"] = Z.pop("ood_morphology")
            rng = np.random.default_rng(abs(hash((t, name))) % 2 ** 32)
            for metric in ("white", "std"):
                r = evaluate(Z, y, t, rng, metric=metric, eval_sets=("test", "ood"))
                full[f"{t}|{name}|{metric}"] = r
                for s in ("test", "ood"):
                    rows.append(dict(target=t, rep=name, kind=kind, metric=metric, set=s, d=r["d"],
                                     Lambda_cf=r["Lambda_cf"], **r[s]))
            print(t, name, f"R2={full[f'{t}|{name}|white']['test']['R2_ridge']:.3f}",
                  f"Om={full[f'{t}|{name}|white']['test']['omega_ratio']:.2f}",
                  f"CF={full[f'{t}|{name}|white']['test']['cf_rmse_sd']:.2f}", flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "baselines.csv"), index=False)
    json.dump(full, open(os.path.join(ROOT, "results", "baselines.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
