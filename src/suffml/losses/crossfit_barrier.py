"""Cross-fitted barrier loss (Draft v0.2, Eq. cf-loss) and in-sample surrogates (ablation).
Implemented now so that the unit tests of I12 exist; training with these losses comes after
the decision gate (Section 25)."""
import torch
import torch.nn.functional as F


def mcshane_mid_torch(za, ya, zq, Lambda, same=None):
    # explicit distances avoid an undefined gradient of cdist at zero distance
    d = ((zq[:, None, :] - za[None, :, :]) ** 2).sum(-1).add(1e-12).sqrt()
    up = ya[None, :] + Lambda * d
    lo = ya[None, :] - Lambda * d
    if same is not None:                       # exclude anchors of the same cell
        up = up.masked_fill(same, float("inf"))
        lo = lo.masked_fill(same, float("-inf"))
    return 0.5 * (up.min(dim=1).values + lo.max(dim=1).values)


def crossfit_barrier(z, y, ids, Lambda, eps, k=16, memory=None):
    """z: B x d in the metric d_N (i.e. N(z)/sqrt(d)); ids: cell identifiers."""
    perm = torch.randperm(len(y), device=z.device)
    halves = (perm[: len(y) // 2], perm[len(y) // 2:])
    loss = z.new_zeros(())
    for anc, qry in (halves, halves[::-1]):
        za, ya, ia = z[anc], y[anc], ids[anc]
        if memory is not None:
            za = torch.cat([za, memory["z"]]); ya = torch.cat([ya, memory["y"]]); ia = torch.cat([ia, memory["ids"]])
        same = ids[qry][:, None] == ia[None, :]
        rho = (mcshane_mid_torch(za, ya, z[qry], Lambda, same) - y[qry]).abs()
        top = torch.topk(rho, min(k, len(rho))).values
        loss = loss + F.relu(top - eps).pow(2).mean()
    return 0.5 * loss


def insample_topk(z, y, Lambda, eps, k=16):
    d = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1).add(1e-12).sqrt()
    r = (y[:, None] - y[None, :]).abs() - Lambda * d
    iu = torch.triu_indices(len(y), len(y), 1, device=z.device)
    rr = r[iu[0], iu[1]]
    top = torch.topk(rr, min(k, len(rr))).values
    return F.relu(top - 2 * eps).pow(2).mean()


def insample_lse(z, y, Lambda, tau):
    d = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1).add(1e-12).sqrt()
    r = (y[:, None] - y[None, :]).abs() - Lambda * d
    iu = torch.triu_indices(len(y), len(y), 1, device=z.device)
    a = torch.cat([z.new_zeros(1), r[iu[0], iu[1]] / tau])
    return 0.5 * tau * torch.logsumexp(a, 0)
