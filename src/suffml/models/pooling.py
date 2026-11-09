"""Pooling matched to the target (Draft v0.2, Section 12.2): global average pooling (volume
averages, R_E) and generalized-mean pooling with a learnable exponent p > 1 (tail targets)."""
import math

import torch
import torch.nn as nn


class GAP(nn.Module):
    def forward(self, Fm):
        return Fm.mean(dim=(-2, -1))

    @property
    def p(self):
        return torch.tensor(1.0)


class GeM(nn.Module):
    """GeM_p(F) = (mean F^p)^{1/p}, p = 1 + exp(q), evaluated in the log domain (stable to p ~ 40)."""

    def __init__(self, p0: float = 3.0):
        super().__init__()
        self.q = nn.Parameter(torch.tensor(math.log(p0 - 1.0)))

    @property
    def p(self):
        return 1.0 + self.q.exp()

    def forward(self, Fm):
        logF = Fm.clamp_min(1e-6).log().flatten(2)
        n = logF.shape[-1]
        p = self.p
        return torch.exp((torch.logsumexp(p * logF, dim=-1) - math.log(n)) / p)
