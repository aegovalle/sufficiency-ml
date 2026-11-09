"""Experiments D-E (Draft v0.2, Sections 17.4-17.5) for R_H (beta = 20), d = 8, E-circ + GeM,
3 seeds, same schedule as the consolidated MSE models (sup_RH_b20_gem2_d8_s*):
  RNC (tau = 2), MSE + RankSim (gamma = 10, lambda = 2), MSE + SBR-CF (lambda_cf in {0.1, 1}),
  MSE + SBR-in (lambda = 1; ablation). Lambda_cf = 4 (validation choice of the certification step),
  eps_target = declared noise floor. lambda_cf is selected afterwards on validation by the
  cross-fitted maximum error (no permutation-based tuning).

Run:  .venv\\Scripts\\python scripts\\run_experiments_de.py [--epochs 60] [--smoke]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from suffml.train.train_objectives import train_objective

ROOT = os.path.join(os.path.dirname(__file__), "..")
MD = os.path.join(ROOT, "results", "models")
CONFIGS = [  # (short name, method, kwargs)
    ("sbrcf1", "sbr_cf", {"lam": 1.0}),
    ("rnc", "rnc", {}),
    ("ranksim", "ranksim", {"lam": 10.0}),
    ("sbrin1", "sbr_in", {"lam": 1.0}),
    ("sbrcf01", "sbr_cf", {"lam": 0.1}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        for name, method, kw in CONFIGS:
            train_objective(method, epochs=2, mine_from=1, seed=0, tag=f"smoke_{name}",
                            out_dir=os.path.join(ROOT, "results", "smoke"), **kw)
        print("smoke ok"); return
    for seed in (0, 1, 2):
        for name, method, kw in CONFIGS:
            tag = f"obj_RH_b20_{name}_d8_s{seed}"
            if os.path.exists(os.path.join(MD, f"{tag}.consol")):
                continue
            train_objective(method, target="RH_b20", epochs=a.epochs, seed=seed, tag=tag, **kw)
            open(os.path.join(MD, f"{tag}.consol"), "w").write("ok")
    print("experiments D-E done")


if __name__ == "__main__":
    main()
