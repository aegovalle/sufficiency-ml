"""Loading of the processed corpora."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")


def load_pilot(root=ROOT, fields=False):
    p = os.path.join(root, "data", "processed")
    D = pd.read_csv(os.path.join(p, "elastic_pilot.csv"))
    Z = np.load(os.path.join(p, "elastic_pilot.npz"))
    N = int(Z["N"])
    masks = np.unpackbits(Z["mask"], axis=-1)[..., :N].astype(bool)
    out = {"table": D, "mask": masks, "S2": Z["S2"], "splits": json.load(open(os.path.join(p, "splits.json")))}
    if fields:
        out["stress"], out["sbar"] = Z["stress"], Z["sbar"]
    return out


def load_design(root=ROOT):
    p = os.path.join(root, "data", "stage0")
    D = pd.read_csv(os.path.join(p, "design_cells.csv"))
    Z = np.load(os.path.join(p, "design_cells.npz"))
    N = int(Z["N"])
    return {"table": D, "mask": np.unpackbits(Z["mask"], axis=-1)[..., :N].astype(bool), "S2": Z["S2"]}
