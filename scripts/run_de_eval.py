"""Evaluation of Experiments D-E (Draft v0.2, Section 17.5; hypothesis H4) on R_H (beta = 20):
MSE (sup_RH_b20_gem2) vs RNC vs RankSim vs MSE+SBR-CF (lambda_cf selected on validation by the
cross-fitted maximum error) vs MSE+SBR-in (ablation), with E1 and the oracle as references.
Metrics on test2 (1000 cells) and OOD; paired subsampling of Omega/Omega_null against MSE;
validation-selected Lipschitz heads (certification). Also the transfer of each encoder to beta = 40.

Run:  .venv\\Scripts\\python scripts\\run_de_eval.py
Outputs: results/de_metrics.csv, results/de_subsample.csv, results/de_cert.csv, results/de_summary.json
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
from suffml.metrics import WhiteningTransform, crossfit_errors, select_lambda

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results"); MD = os.path.join(R, "models")
SETS = ("train", "val", "test2", "ood_morphology")
E0 = ["f", "e11_m", "e22_m", "e12_m"]


def family(tag):
    return tag.rsplit("_s", 1)[0]


def main():
    P = load_pilot(); D = P["table"]
    idx = {s: np.array(P["splits"][s]) for s in SETS}
    tags = sorted(os.path.basename(f)[:-7] for f in glob.glob(os.path.join(MD, "*.consol")))
    tags = [t for t in tags if t.startswith("obj_RH_b20_") or t.startswith("sup_RH_b20_gem2_")]
    reps = {"E1": ("E1", D[E0 + ["Jgap_m", "Jgap_i"]].values.astype(float), None)}
    for tag in tags:
        E = np.load(os.path.join(MD, f"{tag}_emb.npz"))
        js = json.load(open(os.path.join(MD, f"{tag}.json"))); js.pop("log", None)
        reps[tag] = (family(tag), E["Z"], js | {"Zn": E["Zn"]})

    # validation selection of lambda_cf (head-free cross-fitted maximum error on val; R_H b20)
    y = D["RH_b20"].values
    val_crit = {}
    for tag, (fam, Z, js) in reps.items():
        if js is None:
            continue
        T = WhiteningTransform().fit(Z[idx["train"]])
        ztr, zva = T.transform(Z[idx["train"]]), T.transform(Z[idx["val"]])
        sd = y[idx["train"]].std()
        grid = sd * np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])
        L = select_lambda(ztr, y[idx["train"]], zva, y[idx["val"]], grid)
        val_crit[tag] = crossfit_errors(ztr, y[idx["train"]], zva, y[idx["val"]], L)["max"] / y[idx["val"]].std()
    fam_val = pd.Series(val_crit).groupby(lambda t: family(t)).median()
    cf_fams = [f for f in fam_val.index if "_sbrcf" in f]
    sel_cf = fam_val[cf_fams].idxmin() if cf_fams else None
    print("validation cf-max by family:", fam_val.round(3).to_dict(), "-> selected", sel_cf, flush=True)

    rows, subs, certs = [], [], []
    for t in ("RH_b20", "RH_b40"):
        yt = D[t].values
        ymap = {"train": yt[idx["train"]], "val": yt[idx["val"]], "test2": yt[idx["test2"]], "ood": yt[idx["ood_morphology"]]}
        orng = np.random.default_rng(99)
        allreps = dict(reps)
        allreps["ORACLE_10pct"] = ("ORACLE_10pct", (yt + orng.normal(0, 0.1 * yt[idx["train"]].std(), len(yt)))[:, None], None)
        Zm = {}
        for name, (fam, Z, js) in allreps.items():
            Zs = {"train": Z[idx["train"]], "val": Z[idx["val"]], "test2": Z[idx["test2"]], "ood": Z[idx["ood_morphology"]]}
            r = evaluate(Zs, ymap, t, np.random.default_rng(abs(hash((name, t))) % 2 ** 32), metric="white",
                         eval_sets=("test2", "ood"))
            for s in ("test2", "ood"):
                rows.append(dict(target=t, rep=name, family=fam, set=s, **r[s]))
            Zm[name] = WhiteningTransform().fit(Zs["train"]).transform(Zs["test2"])
        sub = paired_omega_subsample(Zm, yt[idx["test2"]], np.random.default_rng(11), B=100, nperm=100)
        for name, arr in sub.items():
            for b, v in enumerate(arr):
                subs.append(dict(target=t, rep=name, family=allreps[name][0], b=b, ratio=v))
        print(t, "evaluated", flush=True)
    for tag, (fam, Z, js) in reps.items():
        if js is None:
            continue
        ys = (y - js["y_mu"]) / js["y_sd"]
        Zn = {s: js["Zn"][idx[s]] for s in ("train", "val", "test2")}
        yy = {s: ys[idx[s]] for s in ("train", "val", "test2")}
        g, best = certify(Zn, yy, test="test2")
        for r_ in g:
            certs.append(dict(model=tag, family=fam, selected=(r_["Lambda"] == best["Lambda"]), **r_))
    M, S, C = pd.DataFrame(rows), pd.DataFrame(subs), pd.DataFrame(certs)
    M.to_csv(os.path.join(R, "de_metrics.csv"), index=False)
    S.to_csv(os.path.join(R, "de_subsample.csv"), index=False)
    C.to_csv(os.path.join(R, "de_cert.csv"), index=False)

    summ = {"val_cfmax_by_family": fam_val.to_dict(), "selected_sbrcf": sel_cf, "families": {}, "paired_vs_MSE": {}}
    for (t, fam, s), g in M.groupby(["target", "family", "set"]):
        summ["families"].setdefault(t, {}).setdefault(s, {})[fam] = {
            k: [float(g[k].median()), float(g[k].min()), float(g[k].max())]
            for k in ("R2_ridge", "R2_mcshane", "omega_ratio", "cf_rmse_sd", "cf_max_sd", "lam_star_ratio")}
    mse_fam = "sup_RH_b20_gem2_d8"
    for t, g in S.groupby("target"):
        piv = g.pivot(index="b", columns="rep", values="ratio")
        mse_cols = [c for c in piv.columns if family(c) == mse_fam]
        for fam in sorted(set(family(c) if c not in ("E1", "ORACLE_10pct") else c for c in piv.columns)):
            cols = [c for c in piv.columns if (family(c) == fam or c == fam)]
            # pair seed s of a method with seed s of MSE (same subsamples); references vs every MSE seed
            dif = []
            for c in cols:
                seed = c.rsplit("_s", 1)[1] if "_s" in c and c not in ("E1", "ORACLE_10pct") else None
                partners = [m for m in mse_cols if seed is not None and m.endswith(f"_s{seed}")] or mse_cols
                for m in partners:
                    dif.append((piv[c] - piv[m]).values)
            dif = np.concatenate(dif)
            summ["paired_vs_MSE"].setdefault(t, {})[fam] = {
                "ratio_median": float(np.median(np.concatenate([piv[c].values for c in cols]))),
                "diff_median": float(np.median(dif)), "diff_ci90": interval(dif), "frac_below": float(np.mean(dif < 0))}
    sel = C[C.selected]
    summ["cert"] = sel.groupby("family")[["Lambda", "train_e_inf", "train_floor", "head_gap", "test_rmse", "test_max",
                                           "test_cf_max"]].median().to_dict(orient="index")
    summ["gem_p"] = {t: reps[t][2]["p_final"] for t in reps if reps[t][2] is not None}
    json.dump(summ, open(os.path.join(R, "de_summary.json"), "w"), indent=1)
    print(json.dumps(summ["paired_vs_MSE"], indent=1))


if __name__ == "__main__":
    main()
