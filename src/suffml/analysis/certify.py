"""Frozen-embedding certification (Draft v0.2, Section 7.2 and 11.5): for each Lambda of a grid,
fit a Lambda-Lipschitz head on the frozen normalized training embeddings, choose Lambda on
validation, and report Lambda_cert, the certified floor e*(Lambda_cert) in d_N, the head gap,
and the test maximum error against the cross-fitted barrier at Lambda_cert."""
from __future__ import annotations

import math

import numpy as np
import torch

from ..models.lipschitz_head import LipschitzHead, certified_lambda, finalize_spectral
from ..metrics import exact_barrier, crossfit_errors


def fit_head(zn_tr, y_tr, Lambda, steps=1500, lr=3e-3, seed=0, dev="cpu"):
    torch.manual_seed(seed)
    d = zn_tr.shape[1]
    head = LipschitzHead(d, Lambda).double().to(dev)
    X = torch.as_tensor(zn_tr, dtype=torch.float64, device=dev)
    Y = torch.as_tensor(y_tr, dtype=torch.float64, device=dev)
    opt = torch.optim.Adam(head.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad(); loss = ((head(X) - Y) ** 2).mean(); loss.backward(); opt.step(); sched.step()
    head.eval(); finalize_spectral(head)
    return head


@torch.no_grad()
def predict(head, zn):
    return head(torch.as_tensor(zn, dtype=torch.float64)).numpy()


def certify(Zn: dict, y: dict, grid=(1.0, 2.0, 4.0, 8.0, 16.0), test="test"):
    """Zn: split -> N(z) (frozen normalization), y: split -> standardized targets."""
    d = Zn["train"].shape[1]
    zm = {k: v / math.sqrt(d) for k, v in Zn.items()}           # metric d_N
    rows = []
    for L in grid:
        h = fit_head(Zn["train"], y["train"], L)
        Lc = certified_lambda(h)
        p = {k: predict(h, Zn[k]) for k in ("train", "val", test)}
        einf = float(np.abs(p["train"] - y["train"]).max())
        floor = exact_barrier(zm["train"], y["train"], Lc)
        cf = crossfit_errors(zm["train"], y["train"], zm[test], y[test], Lc)
        rows.append({"Lambda": L, "Lambda_cert": Lc, "val_rmse": float(np.sqrt(np.mean((p["val"] - y["val"]) ** 2))),
                     "train_e_inf": einf, "train_floor": floor, "head_gap": einf - floor,
                     "floor_holds": bool(einf >= floor - 1e-9),
                     "test_rmse": float(np.sqrt(np.mean((p[test] - y[test]) ** 2))),
                     "test_max": float(np.abs(p[test] - y[test]).max()), "test_cf_max": cf["max"],
                     "test_cf_rmse": cf["rmse"]})
    best = min(rows, key=lambda r: r["val_rmse"])
    return rows, best
