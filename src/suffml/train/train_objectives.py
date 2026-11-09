"""Experiments D and E (Draft v0.2, Sections 17.4-17.5): encoders trained with
  'rnc'     Rank-N-Contrast representation (no MSE); heads fitted later on the frozen embedding
  'ranksim' MSE + gamma * RankSim
  'sbr_cf'  MSE + lambda_cf * cross-fitted barrier (Eq. cf-loss) in d_N, with a cross-batch memory
            of detached anchors (same-cell anchors excluded) and global hard-query mining
  'sbr_in'  MSE + lambda * in-sample top-k barrier (v0.1 objective; ablation)
All share the E-circ encoder, latent normalization, Lambda-Lipschitz head, D2 symmetrization at
evaluation and the training schedule of the gate models.
"""
from __future__ import annotations

import gc
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..data.datasets import ROOT
from ..losses.crossfit_barrier import crossfit_barrier, insample_topk
from ..losses.continuous_label import rnc_loss, ranksim_loss
from ..models.normalization import cov_penalty
from ..models.lipschitz_head import certified_lambda, finalize_spectral
from .train_gate import Supervised, _data, embed_all, DEV


class Memory:
    def __init__(self, size, d):
        self.size, self.d = size, d
        self.z = torch.empty(0, d, device=DEV); self.y = torch.empty(0, device=DEV)
        self.ids = torch.empty(0, dtype=torch.long, device=DEV)

    def push(self, z, y, ids):
        self.z = torch.cat([self.z, z.detach()])[-self.size:]
        self.y = torch.cat([self.y, y.detach()])[-self.size:]
        self.ids = torch.cat([self.ids, ids])[-self.size:]

    def as_dict(self):
        return {"z": self.z, "y": self.y, "ids": self.ids} if len(self.y) else None


@torch.no_grad()
def mine_hard_queries(model, Xtr, ytr, Lambda, k_mine, bs=32):
    """Leave-one-out McShane residual of every training cell with the current encoder
    (exact D2 embedding, frozen normalization) -> indices of the k_mine largest residuals."""
    model.eval()
    zn = torch.cat([model.norm(model.embed(Xtr[a:a + bs])) for a in range(0, len(Xtr), bs)])
    zm = zn / math.sqrt(zn.shape[1])
    d = torch.cdist(zm, zm)
    d.fill_diagonal_(float("inf"))                           # leave one out
    up = (ytr[None, :] + Lambda * d).min(1).values
    lo = (ytr[None, :] - Lambda * d).max(1).values
    res = (0.5 * (up + lo) - ytr).abs()
    model.train()
    return torch.topk(res, k_mine).indices, res


def train_objective(method, target="RH_b20", d=8, Lambda=16.0, Lambda_cf=4.0, epochs=60, bs=64, lr=1e-3,
                    lr_q=0.05, lam_cov=0.1, lam=1.0, eps_target=None, k=16, mem_size=1024, k_mine=128,
                    k_b=16, mine_from=5, mine_every=1, tau=2.0, rank_lambda=2.0, seed=0, tag=None, out_dir=None,
                    arch="ecirc", pool="gem", d2_sym=True, lam_inv=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    P, X, idx = _data()
    D = P["table"]
    y = D[target].values.astype(np.float32)
    mu, sd = y[idx["train"]].mean(), y[idx["train"]].std()
    ys = torch.from_numpy((y - mu) / sd)
    if eps_target is None:                                   # declared noise floor, standardized
        eps_target = (0.004 * float(target[4:]) if target.startswith("RH_b") else 0.0025) / float(sd)
    model = Supervised(d, pool, Lambda, arch=arch, d2_sym=d2_sym).to(DEV)
    q = [model.enc.pool.q] if hasattr(model.enc.pool, "q") else []
    other = [p for n, p in model.named_parameters() if not n.endswith("pool.q")]
    groups = [{"params": other, "lr": lr, "weight_decay": 1e-4}]
    if q:
        groups.append({"params": q, "lr": lr_q, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    nb = len(idx["train"]) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * nb,
                                                pct_start=0.1)
    Xtr, ytr = X[idx["train"]].to(DEV), ys[idx["train"]].to(DEV)
    Xva, yva = X[idx["val"]].to(DEV), ys[idx["val"]].to(DEV)
    ids_tr = torch.arange(len(Xtr), device=DEV)
    mem = Memory(mem_size, d) if (method == "sbr_cf" and mem_size > 0) else None
    mined = None
    best, best_state, log = np.inf, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if method == "sbr_cf" and k_mine > 0 and ep >= mine_from and (ep - mine_from) % mine_every == 0:
            mined, _ = mine_hard_queries(model, Xtr, ytr, Lambda_cf, k_mine)
        perm = torch.randperm(len(Xtr), device=DEV)
        tl, aux_l = 0.0, 0.0
        for bi in range(nb):
            b = perm[bi * bs:(bi + 1) * bs]
            if mined is not None and k_b > 0:                # seed the batch with mined hard queries
                hard = mined[torch.randperm(len(mined), device=DEV)[:k_b]]
                b = torch.unique(torch.cat([hard, b[: bs - k_b]]))
            if lam_inv > 0.0:
                z = checkpoint(model.embed, Xtr[b], use_reentrant=False)
                zn = model.norm(z)
                pred = model.head(zn)
            else:
                pred, z, zn = model(Xtr[b])
            yb = ytr[b]
            if method == "rnc":
                aux = rnc_loss(z, yb, tau)
                loss = aux
            else:
                loss = F.mse_loss(pred, yb) + lam_cov * cov_penalty(zn)
                zm = zn / math.sqrt(d)
                if method == "ranksim":
                    aux = ranksim_loss(z, yb, rank_lambda)
                elif method == "sbr_cf":
                    aux = crossfit_barrier(zm, yb, ids_tr[b], Lambda_cf, eps_target, k=k,
                                           memory=mem.as_dict() if mem is not None else None)
                elif method == "sbr_in":
                    aux = insample_topk(zm, yb, Lambda_cf, eps_target, k=k)
                elif method == "tail_mse":
                    sq_err = (pred - yb).pow(2)
                    k_eff = min(k, sq_err.numel())
                    aux = torch.topk(sq_err, k_eff).values.mean()
                elif method == "mse":
                    aux = torch.zeros((), device=DEV)
                else:
                    raise ValueError(method)
                loss = loss + lam * aux
                if lam_inv > 0.0:
                    # Soft invariance penalty under a random shift + reflection (Ablation 9);
                    # checkpointed to stay comfortably within 8 GB GPU VRAM.
                    sh_y = int(torch.randint(1, Xtr.shape[-2], (1,)))
                    sh_x = int(torch.randint(1, Xtr.shape[-1], (1,)))
                    xb_g = torch.roll(Xtr[b], shifts=(sh_y, sh_x), dims=(-2, -1))
                    if int(torch.randint(2, (1,))):
                        xb_g = xb_g.flip(-1)
                    zn_g = model.norm(checkpoint(model.embed, xb_g, use_reentrant=False))
                    loss = loss + lam_inv * F.mse_loss(zn, zn_g)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            if mem is not None:
                mem.push(zm, yb, ids_tr[b])
            tl += loss.item(); aux_l += float(aux)
        if ep % 5 == 0 or ep >= epochs - 5:
            model.eval()
            with torch.no_grad():
                pv = torch.cat([model(Xva[a:a + 32])[0] for a in range(0, len(Xva), 32)])
                vl = F.mse_loss(pv, yva).item()
            crit = vl if method != "rnc" else -ep                # RNC: keep the final epoch
            if crit < best:
                best, best_state = crit, {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        log.append({"epoch": ep, "loss": tl / nb, "aux": aux_l / nb, "val_mse": float(vl), "p": float(model.enc.pool.p)})
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"[{tag}] ep {ep} loss {tl/nb:.4f} aux {aux_l/nb:.4f} val {vl:.4f} p {log[-1]['p']:.2f} "
                  f"t {time.time()-t0:.0f}s", flush=True)
    model.load_state_dict(best_state); model.eval()
    finalize_spectral(model.head)
    Z = embed_all(model.embed, X, bs=32)
    with torch.no_grad():
        zt = torch.from_numpy(Z).to(DEV)
        znt = model.norm(zt)
        Zn = znt.float().cpu().numpy()
        pred = model.head(znt).float().cpu().numpy() * sd + mu
    info = {"method": method, "target": target, "arch": arch, "pool": pool, "d2_sym": d2_sym, "lam_inv": lam_inv,
            "d": d, "Lambda": Lambda, "Lambda_cf": Lambda_cf,
            "Lambda_cert": certified_lambda(model.head), "p_final": float(model.enc.pool.p), "lam": lam,
            "eps_target_std": eps_target, "k": k, "mem_size": mem_size, "k_mine": k_mine, "k_b": k_b,
            "tau": tau, "rank_lambda": rank_lambda, "best_val_mse_std": float(min(l["val_mse"] for l in log)),
            "y_mu": float(mu), "y_sd": float(sd), "epochs": epochs, "seed": seed,
            "runtime_s": time.time() - t0, "log": log}
    out_dir = out_dir or os.path.join(ROOT, "results", "models")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}.pt"))
    np.savez_compressed(os.path.join(out_dir, f"{tag}_emb.npz"), Z=Z, Zn=Zn, pred=pred)
    json.dump(info, open(os.path.join(out_dir, f"{tag}.json"), "w"), indent=1)
    del model, opt, sched, Xtr, ytr, Xva, yva, ids_tr, mem, zt, znt, best_state
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return info
