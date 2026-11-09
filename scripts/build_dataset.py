"""Stage 1 pilot corpus (Draft v0.2, Sections 9.2-9.3; checklist I05): ~2000 IID cells
(train/val/test) plus an OOD-morphology test set, with frozen-geometry exclusion by canonical
hash, and all elastic targets, descriptors and stress fields.

Run:  .venv\\Scripts\\python scripts\\build_dataset.py [n_workers]
Config: configs/data_elastic.yaml
Outputs (data/processed/): elastic_pilot.csv, elastic_pilot.npz, splits.json
"""
import os
import sys
import json
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import yaml

from suffml.data.geometry import sample_geometry, mask_of, DEFAULT_CORPUS, OOD_MORPHOLOGY
from suffml.data.canonical_hash import canonical_hash
from suffml.data import frozen

ROOT = os.path.join(os.path.dirname(__file__), "..")


def work(args):
    k, mask = args
    from suffml.physics.fft_targets import solve_cell
    return k, solve_cell(mask, keep_fields=True)


def draw(n, cfg, seed, exclude: set, tag):
    rng = np.random.default_rng(seed)
    geoms, metas, masks, hashes, rejected = [], [], [], [], {"frozen": 0, "duplicate": 0}
    while len(masks) < n:
        g, meta = sample_geometry(rng, cfg)
        m = mask_of(g)
        h = canonical_hash(m)
        if h in exclude:
            rejected["frozen" if h in FROZEN else "duplicate"] += 1
            continue
        exclude.add(h)
        meta["subset"] = tag
        geoms.append(g); metas.append(meta); masks.append(m); hashes.append(h)
    return metas, masks, hashes, rejected


def main():
    global FROZEN
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "data_elastic.yaml")))
    out = os.path.join(ROOT, cfg["output_dir"]); os.makedirs(out, exist_ok=True)
    FROZEN = frozen.frozen_hash_set()
    design = pd.read_csv(os.path.join(ROOT, "data", "stage0", "design_cells.csv")).canonical_hash
    exclude = set(FROZEN) | set(design)       # no frozen cell, and no copy of a design cell
    t0 = time.time()
    iid_cfg = dict(DEFAULT_CORPUS, **cfg.get("iid_overrides", {}))
    ood_cfg = dict(OOD_MORPHOLOGY, **cfg.get("ood_overrides", {}))
    m1, k1, h1, r1 = draw(cfg["n_iid"], iid_cfg, cfg["seed_iid"], exclude, "iid")
    m2, k2, h2, r2 = draw(cfg["n_ood_morphology"], ood_cfg, cfg["seed_ood"], exclude, "ood_morphology")
    metas, masks, hashes = m1 + m2, k1 + k2, h1 + h2
    print(f"sampled {len(masks)} cells in {time.time()-t0:.0f}s; rejected iid {r1}, ood {r2}", flush=True)

    from multiprocessing import Pool
    with Pool(nw) as pool:
        res = dict(pool.map(work, list(enumerate(masks)), chunksize=4))
    rows = []
    for k, meta in enumerate(metas):
        s = dict(res[k]["scalars"]); s.update(meta); s["id"] = k; s["canonical_hash"] = hashes[k]
        rows.append(s)
    D = pd.DataFrame(rows)

    # splits: IID cells shuffled into train/val/test; OOD cells are a separate test set
    rng = np.random.default_rng(cfg["seed_split"])
    iid = D.index[D.subset == "iid"].values.copy(); rng.shuffle(iid)
    ntr, nva = int(cfg["frac_train"] * len(iid)), int(cfg["frac_val"] * len(iid))
    split = np.full(len(D), "ood_morphology", dtype=object)
    split[iid[:ntr]], split[iid[ntr:ntr + nva]], split[iid[ntr + nva:]] = "train", "val", "test"
    D["split"] = split
    D.to_csv(os.path.join(out, "elastic_pilot.csv"), index=False)
    M = np.array(masks)
    np.savez_compressed(os.path.join(out, "elastic_pilot.npz"), mask=np.packbits(M, axis=-1), N=M.shape[1],
                        S2=np.array([res[k]["S2"] for k in range(len(D))]),
                        stress=np.array([res[k]["stress"] for k in range(len(D))]),
                        sbar=np.array([res[k]["sbar"] for k in range(len(D))]))
    sp = {s: D.index[D.split == s].tolist() for s in ("train", "val", "test", "ood_morphology")}
    sp["test_hashes"] = {s: frozen.hashlib.sha256("".join(D.canonical_hash[D.split == s]).encode()).hexdigest()
                         for s in ("test", "ood_morphology")}
    sp["config"] = cfg
    sp["frozen_registry_sha256"] = frozen.sha256_file(frozen.REGISTRY)
    sp["rejected"] = {"iid": r1, "ood": r2}
    json.dump(sp, open(os.path.join(out, "splits.json"), "w"), indent=1)
    print(D.groupby("split").size().to_dict(), f"max iters {D.iters.max()}, t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
