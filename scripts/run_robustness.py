"""Robustness sweep of Experiments D-E (R_H, beta = 20), 3 seeds unless stated:
  1. lambda_cf in {0.3, 3} (complements 0.1 and 1)                         d = 8
  2. latent dimension d in {4, 16}: MSE and MSE + SBR-CF (lambda_cf = 1)
  3. minimal tuning of the baselines on seed 0: RankSim gamma in {1, 100} (10 exists),
     RNC tau in {0.5, 1} (2 exists); the configuration with the lowest validation McShane rmse
     (head-free, Lambda chosen on validation, no permutation-based tuning) gets seeds 1 and 2.

Run:  .venv\\Scripts\\python scripts\\run_robustness.py [--epochs 60]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from suffml.train.train_objectives import train_objective
from suffml.data.datasets import load_pilot
from suffml.metrics import WhiteningTransform, select_lambda, crossfit_errors

ROOT = os.path.join(os.path.dirname(__file__), "..")
MD = os.path.join(ROOT, "results", "models")


def run(tag, method, **kw):
    if os.path.exists(os.path.join(MD, f"{tag}.consol")):
        return
    train_objective(method, target="RH_b20", tag=tag, **kw)
    open(os.path.join(MD, f"{tag}.consol"), "w").write("ok")


def val_rmse(tag, P):
    D = P["table"]; sp = P["splits"]
    Z = np.load(os.path.join(MD, f"{tag}_emb.npz"))["Z"]
    tr, va = np.array(sp["train"]), np.array(sp["val"])
    y = D["RH_b20"].values
    T = WhiteningTransform().fit(Z[tr])
    sd = y[tr].std()
    grid = sd * np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])
    L = select_lambda(T.transform(Z[tr]), y[tr], T.transform(Z[va]), y[va], grid)
    return crossfit_errors(T.transform(Z[tr]), y[tr], T.transform(Z[va]), y[va], L)["rmse"] / y[va].std()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=60); a = ap.parse_args()
    E = a.epochs
    for s in (0, 1, 2):                                            # 1. lambda_cf
        run(f"obj_RH_b20_sbrcf03_d8_s{s}", "sbr_cf", lam=0.3, epochs=E, seed=s)
        run(f"obj_RH_b20_sbrcf3_d8_s{s}", "sbr_cf", lam=3.0, epochs=E, seed=s)
    for d in (4, 16):                                              # 2. latent dimension
        for s in (0, 1, 2):
            run(f"obj_RH_b20_mse_d{d}_s{s}", "mse", d=d, epochs=E, seed=s)
            run(f"obj_RH_b20_sbrcf1_d{d}_s{s}", "sbr_cf", lam=1.0, d=d, epochs=E, seed=s)
    # 3. baseline tuning on seed 0
    run("obj_RH_b20_ranksimg1_d8_s0", "ranksim", lam=1.0, epochs=E, seed=0)
    run("obj_RH_b20_ranksimg100_d8_s0", "ranksim", lam=100.0, epochs=E, seed=0)
    run("obj_RH_b20_rnct05_d8_s0", "rnc", tau=0.5, epochs=E, seed=0)
    run("obj_RH_b20_rnct1_d8_s0", "rnc", tau=1.0, epochs=E, seed=0)
    P = load_pilot()
    choice = {}
    for base, cands in (("ranksim", {"ranksim": {"lam": 10.0}, "ranksimg1": {"lam": 1.0}, "ranksimg100": {"lam": 100.0}}),
                        ("rnc", {"rnc": {"tau": 2.0}, "rnct05": {"tau": 0.5}, "rnct1": {"tau": 1.0}})):
        crit = {n: val_rmse(f"obj_RH_b20_{n}_d8_s0", P) for n in cands}
        best = min(crit, key=crit.get)
        choice[base] = {"val_mcshane_rmse_sd": crit, "selected": best}
        for s in (1, 2):
            run(f"obj_RH_b20_{best}_d8_s{s}", base, epochs=E, seed=s, **cands[best])
    json.dump(choice, open(os.path.join(ROOT, "results", "robustness_tuning.json"), "w"), indent=1)
    print("robustness done", choice)


if __name__ == "__main__":
    main()
