"""Steps 4-5 of the recommended sequence (Draft v0.2, Section 25): train the smallest AE and the
MSE-supervised E-circ at d = 8 (GAP and GeM; targets R_E and R_H), evaluate every frozen
embedding on every target with the sufficiency protocol, and write the decision-gate report.

Run:  .venv\\Scripts\\python scripts\\run_gate.py [--epochs 120] [--skip-train]
Outputs: results/gate_metrics.csv, results/gate_report.json, results/models/*
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd

from suffml.data.datasets import load_pilot
from suffml.analysis.evaluate import evaluate
from suffml.metrics import exact_barrier, crossfit_errors

ROOT = os.path.join(os.path.dirname(__file__), "..")
MD = os.path.join(ROOT, "results", "models")
TARGETS = ["R_E", "RH_b2", "RH_b5", "RH_b10", "RH_b20", "RH_b40"]
SUP = [("R_E", "gap"), ("R_E", "gem"), ("RH_b20", "gap"), ("RH_b20", "gem"),
       ("RH_b2", "gem"), ("RH_b40", "gem")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--Lambda", type=float, default=16.0)
    a = ap.parse_args()
    if not a.skip_train:
        from suffml.train.train_gate import train_supervised, train_ae
        train_ae(d=a.d, epochs=a.epochs, tag=f"ae_d{a.d}_s0")
        for t, pool in SUP:
            train_supervised(target=t, pool=pool, d=a.d, Lambda=a.Lambda, epochs=a.epochs,
                             tag=f"sup_{t}_{pool}_d{a.d}_s0")

    P = load_pilot(); D = P["table"]
    idx = {s: np.array(P["splits"][s]) for s in ("train", "val", "test", "ood_morphology")}
    models = {f"AE": f"ae_d{a.d}_s0"} | {f"Ecirc[{t},{p}]": f"sup_{t}_{p}_d{a.d}_s0" for t, p in SUP}
    rows, report = [], {"models": {}}
    for name, tag in models.items():
        E = np.load(os.path.join(MD, f"{tag}_emb.npz")); info = json.load(open(os.path.join(MD, f"{tag}.json")))
        info.pop("log", None)
        report["models"][name] = info
        Z = {s: E["Z"][i] for s, i in idx.items()}; Z["ood"] = Z.pop("ood_morphology")
        for t in TARGETS:
            y = {s: D[t].values[i] for s, i in idx.items()}; y["ood"] = y.pop("ood_morphology")
            rng = np.random.default_rng(abs(hash((name, t))) % 2 ** 32)
            for metric in ("white", "std"):
                r = evaluate(Z, y, t, rng, metric=metric, eval_sets=("test", "ood"))
                for s in ("test", "ood"):
                    rows.append(dict(target=t, rep=name, kind="learned", metric=metric, set=s, d=r["d"],
                                     Lambda_cf=r["Lambda_cf"], **r[s]))
        # certified floor and head gap for the supervised models, on their own target (I13)
        if name.startswith("Ecirc"):
            t = info["target"]
            ys = (D[t].values - info["y_mu"]) / info["y_sd"]
            zn = E["Zn"] / math.sqrt(a.d)
            tr, te = idx["train"], idx["test"]
            Lc = info["Lambda_cert"]
            pred_s = (E["pred"] - info["y_mu"]) / info["y_sd"]
            einf_tr = float(np.abs(pred_s[tr] - ys[tr]).max())
            floor = exact_barrier(zn[tr], ys[tr], Lc)
            cf = crossfit_errors(zn[tr], ys[tr], zn[te], ys[te], Lc)
            report["models"][name]["certified"] = {
                "Lambda_cert": Lc, "train_e_inf_std": einf_tr, "train_floor_e_star_std": floor,
                "head_gap_std": einf_tr - floor, "test_max_err_std": float(np.abs(pred_s[te] - ys[te]).max()),
                "test_cf_max_std": cf["max"], "test_rmse_std": float(np.sqrt(np.mean((pred_s[te] - ys[te]) ** 2)))}
    G = pd.DataFrame(rows)
    B = pd.read_csv(os.path.join(ROOT, "results", "baselines.csv"))
    A = pd.concat([B, G], ignore_index=True)
    A.to_csv(os.path.join(ROOT, "results", "gate_metrics.csv"), index=False)
    json.dump(report, open(os.path.join(ROOT, "results", "gate_report.json"), "w"), indent=1)
    piv = A[(A.metric == "white") & (A.set == "test") & A.target.isin(["R_E", "RH_b20"])]
    print(piv.pivot_table(index="rep", columns="target", values=["R2_ridge", "R2_mcshane", "omega_ratio", "cf_rmse_sd",
                                                                 "lam_star_ratio"]).round(3).to_string())


if __name__ == "__main__":
    main()
