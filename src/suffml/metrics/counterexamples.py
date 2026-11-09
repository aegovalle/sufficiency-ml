"""Latent counterexamples (Draft v0.2, Section 11.7): the pair attaining Omega(q), the
dominant pair of e*(Lambda), and the query with the largest cross-fitted residual together
with its nearest anchors."""
from __future__ import annotations

import numpy as np

from .lipschitz_barrier import exact_barrier, mcshane_mid, pairwise_euclidean
from .sufficiency import omega


def latent_counterexamples(z, y, ids=None, q=0.01, Lambdas=(), z_anchor=None, y_anchor=None,
                           Lambda_cf=None, k_nn=3):
    ids = np.arange(len(y)) if ids is None else np.asarray(ids)
    out = {}
    om, (i, j) = omega(z, y, q, return_pair=True)
    out["omega_pair"] = {"ids": (ids[i].item(), ids[j].item()), "dy": om,
                         "d": float(np.linalg.norm(z[i] - z[j]))}
    out["barrier_pairs"] = []
    for L in Lambdas:
        e, (a, b) = exact_barrier(z, y, L, return_pair=True)
        out["barrier_pairs"].append({"Lambda": float(L), "e_star": e, "ids": (ids[a].item(), ids[b].item()),
                                     "dy": float(abs(y[a] - y[b])), "d": float(np.linalg.norm(z[a] - z[b]))})
    if z_anchor is not None and Lambda_cf is not None:
        res = np.abs(mcshane_mid(z_anchor, y_anchor, z, Lambda_cf) - y)
        w = int(np.argmax(res))
        dn = pairwise_euclidean(z[w:w + 1], z_anchor)[0]
        nn = np.argsort(dn)[:k_nn]
        out["worst_query"] = {"id": ids[w].item(), "residual": float(res[w]), "y": float(y[w]),
                              "nearest_anchor_idx": nn.tolist(), "nearest_anchor_d": dn[nn].tolist(),
                              "nearest_anchor_y": y_anchor[nn].tolist()}
    return out
