"""Normalization layer N and anti-collapse terms (Draft v0.2, Sections 5.4 and 7.1).
N standardizes each latent coordinate with batch statistics in training and running statistics
at evaluation (BatchNorm without affine parameters); d_N(i, j) = ||N(z_i) - N(z_j)|| / sqrt(d)."""
import torch
import torch.nn as nn


class LatentNorm(nn.Module):
    def __init__(self, d: int, momentum: float = 0.05):
        super().__init__()
        self.bn = nn.BatchNorm1d(d, affine=False, momentum=momentum)
        self.d = d

    def forward(self, z):
        return self.bn(z)


def cov_penalty(zn):
    """Sum of squared off-diagonal entries of the batch correlation of N(z), divided by d."""
    zc = zn - zn.mean(0)
    C = (zc.T @ zc) / (len(zn) - 1)
    off = C - torch.diag(torch.diag(C))
    return (off ** 2).sum() / zn.shape[1]


def var_penalty(z, gamma: float = 1.0, eps: float = 1e-4):
    """VICReg variance hinge on the raw embedding (prevents dimensional collapse before N)."""
    std = torch.sqrt(z.var(0) + eps)
    return torch.relu(gamma - std).mean()
