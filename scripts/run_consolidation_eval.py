"""Consolidation of the gate, evaluation part (Draft v0.2, Section 20): all representations
(baselines and every seed of every learned model) evaluated on the 1000-cell test2 set and on
the OOD-morphology set; paired subsampling intervals for Omega/Omega_null; frozen-embedding
certification with Lambda chosen on validation; GeM exponents.

Run:  .venv\\Scripts\\python scripts\\run_consolidation_eval.py
Outputs: results/consolidation_metrics.csv, results/consolidation_subsample.csv,
         results/consolidation_cert.csv, results/consolidation_summary.json
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd

from suffml.data.datasets import load_pilot
from suffml.analysis.evaluate import evaluate, noise_floor
from suffml.analysis.subsample import paired_omega_subsample, interval
from suffml.analysis.certify import certify
from suffml.metrics import WhiteningTransform

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results"); MD = os.path.join(R, "models")
TARGETS = ["R_E", "RH_b2", "RH_b20", "RH_b40"]
E0 = ["f", "e11_m", "e22_m", "e12_m"]
SETS = ("train", "val", "test2", "ood_morphology")


def pca_fit(Xtr, k):
    mu = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    return lambda X: (X - mu) @ Vt[:k].T


def main():
    P = load_pilot(); D = P["table"]
    idx = {s: np.array(P["splits"][s]) for s in SETS}
    reps = {}                                   # name -> (family, kind, seed, full N x d array)
    for name, cols in (("GL", ["f", "rhoI", "A_lig"]), ("E0", E0), ("E1", E0 + ["Jgap_m", "Jgap_i"])):
        reps[name] = (name, "geometry" if name == "GL" else "solved", 0, D[cols].values.astype(float))
    S2 = P["S2"].reshape(len(D), -1).astype(float); img = P["mask"].reshape(len(D), -1).astype(float)
    reps["MKS10"] = ("MKS10", "geometry", 0, pca_fit(S2[idx["train"]], 10)(S2))
    reps["PCA10"] = ("PCA10", "geometry", 0, pca_fit(img[idx["train"]], 10)(img))
    info = {}
    for f in sorted(glob.glob(os.path.join(MD, "*.consol"))):
        tag = os.path.basename(f)[:-7]
        js = json.load(open(os.path.join(MD, f"{tag}.json"))); js.pop("log", None)
        E = np.load(os.path.join(MD, f"{tag}_emb.npz"))
        if len(E["Z"]) != len(D):
            raise RuntimeError(f"{tag}: embeddings not on the extended corpus")
        fam = tag.rsplit("_s", 1)[0]
        reps[tag] = (fam, "learned", int(tag.rsplit("_s", 1)[1]), E["Z"])
        info[tag] = js | ({"Zn": E["Zn"]} if "Zn" in E.files else {})

    rows, subs, certs = [], [], []
    for t in TARGETS:
        y = {s: D[t].values[i] for s, i in idx.items()}
        ymap = {"train": y["train"], "val": y["val"], "test2": y["test2"], "ood": y["ood_morphology"]}
        Zm = {}
        # oracle controls (calibration of the attainable Omega floor at this sample size):
        # the target itself plus Gaussian noise at the declared numerical floor and at 10% of sd
        orng = np.random.default_rng(99 + TARGETS.index(t))
        yall = D[t].values
        tgt_reps = dict(reps)
        tgt_reps["ORACLE_noise"] = ("ORACLE_noise", "oracle", 0,
                                    (yall + orng.normal(0, noise_floor(t), len(yall)))[:, None])
        tgt_reps["ORACLE_10pct"] = ("ORACLE_10pct", "oracle", 0,
                                    (yall + orng.normal(0, 0.1 * yall[idx["train"]].std(), len(yall)))[:, None])
        for name, (fam, kind, seed, Z) in tgt_reps.items():
            Zs = {"train": Z[idx["train"]], "val": Z[idx["val"]], "test2": Z[idx["test2"]], "ood": Z[idx["ood_morphology"]]}
            rng = np.random.default_rng(abs(hash((name, t))) % 2 ** 32)
            r = evaluate(Zs, ymap, t, rng, metric="white", eval_sets=("test2", "ood"))
            for s in ("test2", "ood"):
                rows.append(dict(target=t, rep=name, family=fam, kind=kind, seed=seed, set=s, d=r["d"],
                                 Lambda_cf=r["Lambda_cf"], **r[s]))
            Zm[name] = WhiteningTransform().fit(Zs["train"]).transform(Zs["test2"])
        sub = paired_omega_subsample(Zm, y["test2"], np.random.default_rng(7 + TARGETS.index(t)), B=100, nperm=100)
        for name, arr in sub.items():
            for b, v in enumerate(arr):
                subs.append(dict(target=t, rep=name, b=b, ratio=v))
        print(t, "evaluated", flush=True)
    # certification on the own target of every supervised model
    for tag, js in info.items():
        if "Zn" not in js:
            continue
        t = js["target"]; ys = (D[t].values - js["y_mu"]) / js["y_sd"]
        Zn = {s: js["Zn"][idx[s]] for s in ("train", "val", "test2")}
        yy = {s: ys[idx[s]] for s in ("train", "val", "test2")}
        grid_rows, best = certify(Zn, yy, test="test2")
        for g in grid_rows:
            certs.append(dict(model=tag, target=t, selected=(g["Lambda"] == best["Lambda"]), **g))
        print("certified", tag, best["Lambda"], round(best["train_floor"], 3), flush=True)

    M = pd.DataFrame(rows); S = pd.DataFrame(subs); C = pd.DataFrame(certs)
    M.to_csv(os.path.join(R, "consolidation_metrics.csv"), index=False)
    S.to_csv(os.path.join(R, "consolidation_subsample.csv"), index=False)
    C.to_csv(os.path.join(R, "consolidation_cert.csv"), index=False)

    # summary: per family, median and range over seeds; paired differences against E1
    summ = {"families": {}, "paired_vs_E1": {}, "gem_p": {}, "cert_selected": {}}
    for (t, fam, s), g in M.groupby(["target", "family", "set"]):
        summ["families"].setdefault(t, {}).setdefault(s, {})[fam] = {
            k: [float(g[k].median()), float(g[k].min()), float(g[k].max())]
            for k in ("R2_ridge", "R2_mcshane", "omega_ratio", "cf_rmse_sd", "cf_max_sd", "lam_star_ratio")}
    for t, g in S.groupby("target"):
        piv = g.pivot(index="b", columns="rep", values="ratio")
        for rep in piv.columns:
            if rep == "E1":
                continue
            dlt = (piv[rep] - piv["E1"]).values
            summ["paired_vs_E1"].setdefault(t, {})[rep] = {"median": float(np.median(dlt)), "ci90": interval(dlt),
                                                         "frac_below": float(np.mean(dlt < 0))}
        summ.setdefault("subsample_ci", {})[t] = {rep: interval(piv[rep].values) for rep in piv.columns}
    for tag, js in info.items():
        if js.get("pool") == "gem":
            summ["gem_p"][tag] = js["p_final"]
    if len(C):
        summ["cert_selected"] = C[C.selected].drop(columns=["selected"]).to_dict(orient="records")
    json.dump(summ, open(os.path.join(R, "consolidation_summary.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
