"""Continuous-label representation baselines (Draft v0.2, Experiment D).

Rank-N-Contrast (Zha et al., NeurIPS 2023): for anchor i and positive j, the negatives are the
samples k whose label distance to i is at least that of j; similarity = -||z_i - z_j||_2 / tau.
RankSim (Gong et al., ICML 2022): MSE on the ranks of feature similarities (cosine) against
the ranks of label similarities (-|y_i - y_j|), with the blackbox-differentiation ranker of
Vlastelica et al. (interpolation strength lambda). Both follow the reference implementations,
adapted to a single view per sample (the encoders are exactly invariant to the symmetries).
"""
import torch
import torch.nn.functional as F


def rnc_loss(z, y, tau: float = 2.0):
    B = len(y)
    s = -torch.cdist(z, z) / tau                                  # B x B similarities
    dy = (y[:, None] - y[None, :]).abs()                          # label distances
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    # for anchor i, positive j: denominator over k != i with dy[i,k] >= dy[i,j]
    mask = (dy[:, None, :] >= dy[:, :, None]) & ~eye[:, None, :]  # i, j, k
    s_ik = s[:, None, :].expand(B, B, B)
    log_den = torch.logsumexp(s_ik.masked_fill(~mask, float("-inf")), dim=-1)   # i, j
    logp = s - log_den
    return -(logp.masked_fill(eye, 0.0).sum() / (B * (B - 1)))


def _rank_normalised(seq):
    return (torch.argsort(torch.argsort(seq, dim=-1).flip(-1), dim=-1) + 1).float() / seq.size(-1)


class TrueRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, lambda_val):
        rank = _rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, rank)
        return rank

    @staticmethod
    def backward(ctx, grad_output):
        sequence, rank = ctx.saved_tensors
        rank_prime = _rank_normalised(sequence + ctx.lambda_val * grad_output)
        return -(rank - rank_prime) / (ctx.lambda_val + 1e-8), None


def ranksim_loss(z, y, lambda_val: float = 2.0):
    zf = F.normalize(z, dim=1)
    sim = zf @ zf.T
    loss = z.new_zeros(())
    for i in range(len(y)):
        label_ranks = _rank_normalised(-(y[i] - y).abs()[None, :])
        feat_ranks = TrueRanker.apply(sim[i][None, :], lambda_val)
        loss = loss + F.mse_loss(feat_ranks, label_ranks)
    return loss / len(y)
