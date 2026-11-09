"""Registry of frozen phase-field geometries (Sets B, O, C1, C2 of the descriptor study) and
exclusion check by canonical hash (Draft v0.2, Section 9.2)."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from .canonical_hash import canonical_hash

DESCRIPTOR_STUDY = os.environ.get(
    "SUFFML_DESCRIPTOR_STUDY", r"C:\Users\aegov\Downloads\Draft_v05_descriptor_discovery\campaign")
FROZEN_SOURCES = {
    # name: (relative path of the mask file, description)
    "blind_A_B_O": ("blind/blind_masks.npz", "blind validation Sets A/B and O (120 cells, all excluded)"),
    "tcd_C1_C2": ("tcd_frozen/tcd_masks.npz", "frozen prospective test of J_1.5ell, Sets C1 and C2 (42 cells)"),
}
REGISTRY = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "frozen", "frozen_hashes.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def build_registry(root=DESCRIPTOR_STUDY, out=REGISTRY):
    reg = {"sources": {}, "hashes": {}}
    for name, (rel, desc) in FROZEN_SOURCES.items():
        p = os.path.join(root, rel)
        masks = np.load(p)["mask"]
        hs = [canonical_hash(m) for m in masks]
        reg["sources"][name] = {"file": rel, "sha256": sha256_file(p), "n": int(len(masks)),
                                "n_unique_canonical": int(len(set(hs))), "description": desc}
        for k, h in enumerate(hs):
            reg["hashes"].setdefault(h, []).append(f"{name}:{k}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(reg, open(out, "w"), indent=1)
    return reg


def load_registry(path=REGISTRY):
    return json.load(open(path))


def frozen_hash_set(path=REGISTRY) -> set:
    return set(load_registry(path)["hashes"])


def is_frozen(mask, hashes: set) -> bool:
    return canonical_hash(mask) in hashes
