"""Extra IID test set ('test2', 1000 cells, new seed) appended to the pilot corpus, to tighten the
worst-case statistics of the consolidation step (Draft v0.2, Section 20). Same distribution and
exclusions as build_dataset.py; existing cell indices are unchanged.

Run:  .venv\\Scripts\\python scripts\\extend_test.py [n_workers]
"""
import os
import sys
import json

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd

import build_dataset as bd
from suffml.data.geometry import DEFAULT_CORPUS
from suffml.data import frozen

ROOT = os.path.join(os.path.dirname(__file__), "..")
N_EXTRA, SEED = 1000, 20261106


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    p = os.path.join(ROOT, "data", "processed")
    D = pd.read_csv(os.path.join(p, "elastic_pilot.csv"))
    if (D.split == "test2").any():
        print("test2 already present"); return
    Z = dict(np.load(os.path.join(p, "elastic_pilot.npz")))
    sp = json.load(open(os.path.join(p, "splits.json")))
    bd.FROZEN = frozen.frozen_hash_set()
    design = pd.read_csv(os.path.join(ROOT, "data", "stage0", "design_cells.csv")).canonical_hash
    exclude = set(bd.FROZEN) | set(design) | set(D.canonical_hash)
    metas, masks, hashes, rej = bd.draw(N_EXTRA, DEFAULT_CORPUS, SEED, exclude, "iid_test2")
    from multiprocessing import Pool
    with Pool(nw) as pool:
        res = dict(pool.map(bd.work, list(enumerate(masks)), chunksize=4))
    rows = []
    for k, meta in enumerate(metas):
        s = dict(res[k]["scalars"]); s.update(meta); s["id"] = len(D) + k; s["canonical_hash"] = hashes[k]
        s["split"] = "test2"; rows.append(s)
    D2 = pd.concat([D, pd.DataFrame(rows)], ignore_index=True)
    N = int(Z["N"])
    M = np.unpackbits(Z["mask"], axis=-1)[..., :N].astype(bool)
    M = np.concatenate([M, np.array(masks)])
    np.savez_compressed(os.path.join(p, "elastic_pilot.npz"), mask=np.packbits(M, axis=-1), N=N,
                        S2=np.concatenate([Z["S2"], np.array([res[k]["S2"] for k in range(len(masks))])]),
                        stress=np.concatenate([Z["stress"], np.array([res[k]["stress"] for k in range(len(masks))])]),
                        sbar=np.concatenate([Z["sbar"], np.array([res[k]["sbar"] for k in range(len(masks))])]))
    D2.to_csv(os.path.join(p, "elastic_pilot.csv"), index=False)
    sp["test2"] = D2.index[D2.split == "test2"].tolist()
    sp["test_hashes"]["test2"] = frozen.hashlib.sha256("".join(D2.canonical_hash[D2.split == "test2"]).encode()).hexdigest()
    sp["test2_seed"], sp["rejected"]["test2"] = SEED, rej
    json.dump(sp, open(os.path.join(p, "splits.json"), "w"), indent=1)
    print(D2.groupby("split").size().to_dict(), rej)


if __name__ == "__main__":
    main()
