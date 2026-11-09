"""Canonical geometry hash invariant to periodic integer translations and to the reflections
x -> -x, y -> -y (Draft v0.2, Section 9.2, exclusion of frozen geometries).

Construction: for each of the four flipped images g(x), compute integer score maps by exact
circular correlations with fixed signed integer kernels (translation-equivariant). The
anchors are the pixels attaining the lexicographic maximum of (score_1, score_2, score_3);
every anchor is rolled to the origin and the packed bits are hashed. The canonical hash is the
minimum over flips and anchors. Translated or reflected copies of a cell give exactly the same
candidate set, hence the same hash. No anchor is ever discarded, so the hash is exact.
"""
from __future__ import annotations

import hashlib

import numpy as np

_RNG = np.random.default_rng(20261103)
_KERNELS = [(r, _RNG.integers(-2 ** 15, 2 ** 15, size=(2 * r + 1, 2 * r + 1))) for r in (3, 6, 10)]


def _score(m: np.ndarray, r: int, K: np.ndarray) -> np.ndarray:
    s = np.zeros(m.shape, np.int64)
    for a in range(-r, r + 1):
        for b in range(-r, r + 1):
            w = K[a + r, b + r]
            if w:
                s += w * np.roll(m, (-a, -b), axis=(0, 1))
    return s


def _anchors(g: np.ndarray) -> np.ndarray:
    m = g.astype(np.int64)
    cand = None
    for r, K in _KERNELS:
        s = _score(m, r, K)
        if cand is None:
            cand = np.argwhere(s == s.max())
        else:
            v = s[cand[:, 0], cand[:, 1]]
            cand = cand[v == v.max()]
        if len(cand) == 1:
            break
    return cand


def _flips(mask):
    return (mask, mask[::-1, :], mask[:, ::-1], mask[::-1, ::-1])


def canonical_hash(mask: np.ndarray) -> str:
    mask = np.asarray(mask, bool)
    best = None
    tag = bytes(str(mask.shape), "ascii")
    for g in _flips(mask):
        for i, j in _anchors(g):
            h = hashlib.sha256(np.packbits(np.roll(g, (-i, -j), axis=(0, 1))).tobytes() + tag).hexdigest()
            if best is None or h < best:
                best = h
    return best


def canonical_hashes(masks) -> list[str]:
    return [canonical_hash(m) for m in masks]
