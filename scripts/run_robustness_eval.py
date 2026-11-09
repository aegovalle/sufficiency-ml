"""Evaluation of the robustness sweep (R_H, beta = 20): every obj_RH_b20_* family and the MSE
references, on test2 and OOD, with paired subsampling of Omega/Omega_null against the MSE model of
the same latent dimension and the same seed.

Run:  .venv\\Scripts\\python scripts\\run_robustness_eval.py
Outputs: results/robust_metrics.csv, results/robust_subsample.csv, results/robust_summary.json
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd

from suffml.data.datasets import load_pilot
from suffml.analysis.evaluate import evaluate
from suffml.analysis.subsample import paired_omega_subsample, interval
from suffml.metrics import WhiteningTransform

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results"); MD = os.path.join(R, "models")
SETS = ("train", "val", "test2", "ood_morphology")
E0 = ["f", "e11_m", "e22_m", "e12_m"]


def family(tag):
    return tag.rsplit("_s", 1)[0]


def dim(fam):
    m = re.search(r"_d(\d+)", fam)
    return int(m.group(1)) if m else None


def mse_family(d):
    return "sup_RH_b20_gem2_d8" if d == 8 else f"obj_RH_b20_mse_d{d}"


def main():
    P = load_pilot(); D = P["table"]
    idx = {s: np.array(P["splits"][s]) for s in SETS}
    tags = sorted(os.path.basename(f)[:-7] for f in glob.glob(os.path.join(MD, "*.consol")))
    tags = [t for t in tags if t.startswith("obj_RH_b20_") or t.startswith("sup_RH_b20_gem2_")]
    t = "RH_b20"; y = D[t].values
    reps = {"E1": D[E0 + ["Jgap_m", "Jgap_i"]].values.astype(float)}
    for tag in tags:
        reps[tag] = np.load(os.path.join(MD, f"{tag}_emb.npz"))["Z"]
    reps["ORACLE_10pct"] = (y + np.random.default_rng(99).normal(0, 0.1 * y[idx["train"]].std(), len(y)))[:, None]
    ymap = {"train": y[idx["train"]], "val": y[idx["val"]], "test2": y[idx["test2"]], "ood": y[idx["ood_morphology"]]}
    rows, Zm = [], {}
    for name, Z in reps.items():
        Zs = {"train": Z[idx["train"]], "val": Z[idx["val"]], "test2": Z[idx["test2"]], "ood": Z[idx["ood_morphology"]]}
        r = evaluate(Zs, ymap, t, np.random.default_rng(abs(hash(name)) % 2 ** 32), metric="white", eval_sets=("test2", "ood"))
        for s in ("test2", "ood"):
            rows.append(dict(rep=name, family=family(name) if name.startswith(("obj", "sup")) else name, set=s, **r[s]))
        Zm[name] = WhiteningTransform().fit(Zs["train"]).transform(Zs["test2"])
    sub = paired_omega_subsample(Zm, y[idx["test2"]], np.random.default_rng(11), B=100, nperm=100)
    M = pd.DataFrame(rows)
    S = pd.DataFrame([dict(rep=k, b=b, ratio=v) for k, arr in sub.items() for b, v in enumerate(arr)])
    M.to_csv(os.path.join(R, "robust_metrics.csv"), index=False); S.to_csv(os.path.join(R, "robust_subsample.csv"), index=False)
    piv = S.pivot(index="b", columns="rep", values="ratio")
    summ = {"families": {}, "paired_vs_MSE_same_d": {}}
    for (fam, s), g in M.groupby(["family", "set"]):
        summ["families"].setdefault(s, {})[fam] = {k: [float(g[k].median()), float(g[k].min()), float(g[k].max())]
                                                   for k in ("R2_mcshane", "R2_ridge", "omega_ratio", "cf_rmse_sd",
                                                             "cf_max_sd", "lam_star_ratio")}
    fams = sorted({family(c) for c in piv.columns if c.startswith(("obj", "sup"))})
    for fam in fams:
        d = dim(fam); mf = mse_family(d)
        dif = []
        for c in [c for c in piv.columns if family(c) == fam]:
            seed = c.rsplit("_s", 1)[1]
            partner = f"{mf}_s{seed}"
            if partner in piv.columns and partner != c:
                dif.append((piv[c] - piv[partner]).values)
        vals = np.concatenate([piv[c].values for c in piv.columns if family(c) == fam])
        entry = {"d": d, "n_seeds": int(sum(family(c) == fam for c in piv.columns)), "ratio_median": float(np.median(vals))}
        if dif:
            dif = np.concatenate(dif)
            entry |= {"diff_median": float(np.median(dif)), "diff_ci90": interval(dif), "frac_below": float(np.mean(dif < 0))}
        summ["paired_vs_MSE_same_d"][fam] = entry
    json.dump(summ, open(os.path.join(R, "robust_summary.json"), "w"), indent=1)
    for fam, e in summ["paired_vs_MSE_same_d"].items():
        te = summ["families"]["test2"][fam]
        print(f"{fam:28s} n={e['n_seeds']} Om={e['ratio_median']:.3f} "
              + (f"diff={e['diff_median']:+.3f} [{e['diff_ci90'][0]:+.3f},{e['diff_ci90'][1]:+.3f}] below={e['frac_below']:.2f} " if "diff_median" in e else "")
              + f"R2mc={te['R2_mcshane'][0]:.3f} cfrmse={te['cf_rmse_sd'][0]:.3f} cfmax={te['cf_max_sd'][0]:.3f}")


if __name__ == "__main__":
    main()
