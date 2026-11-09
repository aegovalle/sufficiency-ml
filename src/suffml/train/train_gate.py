"""Training of the gate models (Draft v0.2, Section 25, steps 4-5; Experiments B and C at d = 8):
  * supervised E-circ encoder (GAP or GeM pooling) + latent normalization + Lambda-Lipschitz
    head, trained with MSE (+ covariance penalty, support machinery), exact reflection
    symmetrization (d2_embed) and exact shift invariance by construction;
  * convolutional autoencoder (BCE reconstruction).
Embeddings of every split are written for frozen-representation evaluation.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.datasets import load_pilot, ROOT
from ..models.encoder_circ import ECirc, EFNO, EStd, d2_embed
from ..models.normalization import LatentNorm, cov_penalty
from ..models.lipschitz_head import LipschitzHead, certified_lambda, finalize_spectral
from ..models.autoencoder import ConvAE

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = False   # benchmark workspaces overflow the 8 GB laptop GPU
SPLITS = ("train", "val", "test", "ood_morphology")


def _data():
    P = load_pilot()
    X = torch.from_numpy(P["mask"].astype(np.float32))[:, None]          # N x 1 x 127 x 127
    idx = {s: np.array(P["splits"][s]) for s in SPLITS}
    return P, X, idx


@torch.no_grad()
def reembed_supervised(tag, out_dir=None):
    """Recompute the embeddings of every cell of the current corpus with a saved model."""
    out_dir = out_dir or os.path.join(ROOT, "results", "models")
    info = json.load(open(os.path.join(out_dir, f"{tag}.json")))
    P, X, _ = _data()
    model = Supervised(info["d"], info["pool"], info["Lambda"]).to(DEV)
    model.load_state_dict(torch.load(os.path.join(out_dir, f"{tag}.pt"), map_location=DEV))
    model.eval()
    Z = embed_all(model.embed, X)
    Zn = embed_all(lambda x: model.norm(model.embed(x)), X)
    pred = embed_all(lambda x: model(x)[0], X) * info["y_sd"] + info["y_mu"]
    np.savez_compressed(os.path.join(out_dir, f"{tag}_emb.npz"), Z=Z, Zn=Zn, pred=pred)


@torch.no_grad()
def reembed_ae(tag, out_dir=None):
    out_dir = out_dir or os.path.join(ROOT, "results", "models")
    info = json.load(open(os.path.join(out_dir, f"{tag}.json")))
    P, X, _ = _data()
    model = ConvAE(info["d"]).to(DEV)
    model.load_state_dict(torch.load(os.path.join(out_dir, f"{tag}.pt"), map_location=DEV))
    model.eval()
    np.savez_compressed(os.path.join(out_dir, f"{tag}_emb.npz"), Z=embed_all(model.encode, X))


class Supervised(nn.Module):
    def __init__(self, d, pool, Lambda, arch: str = "ecirc", d2_sym: bool = True):
        super().__init__()
        if arch == "ecirc":
            self.enc = ECirc(d, pool=pool)
        elif arch == "efno":
            self.enc = EFNO(d, pool=pool)
        elif arch == "estd":
            self.enc = EStd(d, pool=pool)
        else:
            raise ValueError(f"Unknown arch: {arch}")
        self.arch = arch
        self.d2_sym = d2_sym
        self.norm = LatentNorm(d)
        self.head = LipschitzHead(d, Lambda)

    def embed_single(self, v):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            f = self.enc.features(v)
        return self.enc.proj(self.enc.pool(f.float()))

    def embed(self, x):
        """Training: one random element of the reflection group per batch (flip augmentation;
        the exact D2 average would need four times the activation memory, more than the 8 GB
        GPU). Evaluation: exact D2 symmetrization (d2_embed) when d2_sym=True."""
        if not self.d2_sym:
            return self.embed_single(x)
        if self.training:
            g = int(torch.randint(4, (1,)))
            v = (x, x.flip(-1), x.flip(-2), x.flip((-2, -1)))[g]
            return self.embed_single(v)
        zs = [self.embed_single(v) for v in (x, x.flip(-1), x.flip(-2), x.flip((-2, -1)))]
        return torch.stack(zs).mean(0)

    def forward(self, x):
        z = self.embed(x)
        zn = self.norm(z)
        return self.head(zn), z, zn


@torch.no_grad()
def embed_all(model_embed, X, bs=32):
    out = []
    for a in range(0, len(X), bs):
        out.append(model_embed(X[a:a + bs].to(DEV)).float().cpu())
    return torch.cat(out).numpy()


def train_supervised(target="RH_b20", pool="gem", d=8, Lambda=16.0, epochs=120, bs=64, lr=1e-3,
                     lam_cov=0.1, seed=0, tag=None, out_dir=None, lr_q=0.05):
    torch.manual_seed(seed); np.random.seed(seed)
    P, X, idx = _data()
    D = P["table"]
    y = D[target].values.astype(np.float32)
    mu, sd = y[idx["train"]].mean(), y[idx["train"]].std()
    ys = torch.from_numpy((y - mu) / sd)
    model = Supervised(d, pool, Lambda).to(DEV)
    # the GeM exponent gets its own learning rate and no weight decay: with the shared
    # rate it barely moved in the first gate run (3.0 -> 3.2)
    q_params = [model.enc.pool.q] if pool == "gem" else []
    other = [p for n, p in model.named_parameters() if not n.endswith("pool.q")]
    groups = [{"params": other, "lr": lr, "weight_decay": 1e-4}]
    if q_params:
        groups.append({"params": q_params, "lr": lr_q, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    steps = epochs * math.ceil(len(idx["train"]) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps,
                                                pct_start=0.1)
    Xtr, ytr = X[idx["train"]].to(DEV), ys[idx["train"]].to(DEV)
    Xva, yva = X[idx["val"]].to(DEV), ys[idx["val"]].to(DEV)
    best, best_state, log = np.inf, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEV)
        tl = 0.0
        for a in range(0, len(perm) - bs + 1, bs):
            b = perm[a:a + bs]
            pred, z, zn = model(Xtr[b])
            loss = F.mse_loss(pred, ytr[b]) + lam_cov * cov_penalty(zn)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(Xva[a:a + 32])[0] for a in range(0, len(Xva), 32)])
            vl = F.mse_loss(pv, yva).item()
        log.append({"epoch": ep, "train": tl / len(perm), "val_mse": vl,
                    "p": float(model.enc.pool.p) if pool == "gem" else 1.0})
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"[{tag}] ep {ep} train {tl/len(perm):.4f} val {vl:.4f} p {log[-1]['p']:.2f} t {time.time()-t0:.0f}s",
                  flush=True)
    model.load_state_dict(best_state); model.eval()
    finalize_spectral(model.head)              # converge spectral-norm power iterations
    Z = embed_all(model.embed, X)
    with torch.no_grad():
        Zn = embed_all(lambda x: model.norm(model.embed(x)), X)
        pred = embed_all(lambda x: model(x)[0], X) * sd + mu
    info = {"target": target, "pool": pool, "d": d, "Lambda": Lambda, "Lambda_cert": certified_lambda(model.head),
            "p_final": float(model.enc.pool.p) if pool == "gem" else 1.0, "best_val_mse_std": best,
            "y_mu": float(mu), "y_sd": float(sd), "epochs": epochs, "seed": seed, "runtime_s": time.time() - t0,
            "log": log}
    tag = tag or f"sup_{target}_{pool}_d{d}_s{seed}"
    out_dir = out_dir or os.path.join(ROOT, "results", "models")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}.pt"))
    np.savez_compressed(os.path.join(out_dir, f"{tag}_emb.npz"), Z=Z, Zn=Zn, pred=pred)
    json.dump(info, open(os.path.join(out_dir, f"{tag}.json"), "w"), indent=1)
    return info


def train_ae(d=8, epochs=120, bs=64, lr=1e-3, seed=0, tag=None, out_dir=None):
    torch.manual_seed(seed)
    P, X, idx = _data()
    model = ConvAE(d).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(idx["train"]) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    Xtr, Xva = X[idx["train"]].to(DEV), X[idx["val"]].to(DEV)
    best, best_state, log = np.inf, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEV)
        for a in range(0, len(perm) - bs + 1, bs):
            xb = Xtr[perm[a:a + bs]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
                logits, _ = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits.float(), xb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            lv = np.mean([F.binary_cross_entropy_with_logits(model(Xva[a:a + 128])[0].float(), Xva[a:a + 128]).item()
                          for a in range(0, len(Xva), 128)])
            acc = np.mean([((model(Xva[a:a + 128])[0] > 0).float() == Xva[a:a + 128]).float().mean().item()
                           for a in range(0, len(Xva), 128)])
        log.append({"epoch": ep, "val_bce": float(lv), "val_pixel_acc": float(acc)})
        if lv < best:
            best, best_state = lv, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"[{tag}] ep {ep} val bce {lv:.4f} acc {acc:.4f} t {time.time()-t0:.0f}s", flush=True)
    model.load_state_dict(best_state); model.eval()
    Z = embed_all(model.encode, X)
    info = {"model": "ConvAE", "d": d, "best_val_bce": float(best), "val_pixel_acc": log[-1]["val_pixel_acc"],
            "epochs": epochs, "seed": seed, "runtime_s": time.time() - t0, "log": log}
    tag = tag or f"ae_d{d}_s{seed}"
    out_dir = out_dir or os.path.join(ROOT, "results", "models")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}.pt"))
    np.savez_compressed(os.path.join(out_dir, f"{tag}_emb.npz"), Z=Z)
    json.dump(info, open(os.path.join(out_dir, f"{tag}.json"), "w"), indent=1)
    return info
