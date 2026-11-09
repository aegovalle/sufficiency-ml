"""Lambda-Lipschitz prediction head and certification (Draft v0.2, Section 7.2):
g(z) = Lambda f(N(z)/sqrt(d)), f 1-Lipschitz (spectral-normalized linear layers, GroupSort), so
|g(z_i) - g(z_j)| <= Lambda d_N(i, j). Lambda_cert uses exact spectral norms."""
import math

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class GroupSort(nn.Module):
    """MaxMin on pairs of units: 1-Lipschitz and gradient-norm preserving."""

    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return torch.cat([torch.maximum(a, b), torch.minimum(a, b)], dim=-1)


class LipschitzHead(nn.Module):
    def __init__(self, d: int, Lambda: float, widths=(64, 32), n_power_iterations: int = 3):
        super().__init__()
        dims = (d,) + tuple(widths) + (1,)
        self.layers = nn.ModuleList(spectral_norm(nn.Linear(i, o), n_power_iterations=n_power_iterations)
                                    for i, o in zip(dims[:-1], dims[1:]))
        self.act, self.Lambda, self.d = GroupSort(), float(Lambda), d

    def forward(self, zn):                     # zn = N(z)
        h = zn / math.sqrt(self.d)
        for layer in self.layers[:-1]:
            h = self.act(layer(h))
        return self.Lambda * self.layers[-1](h).squeeze(-1)


@torch.no_grad()
def finalize_spectral(head: LipschitzHead, n: int = 100):
    """Converge the power iterations of spectral normalization (they only run in training
    mode), so that the normalized weights have spectral norm 1 to high accuracy."""
    was = head.training
    head.train()
    x = torch.zeros(2, head.d, device=next(head.parameters()).device,
                    dtype=next(head.parameters()).dtype)
    for _ in range(n):
        head(x)
    head.train(was)


@torch.no_grad()
def certified_lambda(head: LipschitzHead) -> float:
    s = 1.0
    for layer in head.layers:
        s *= torch.linalg.matrix_norm(layer.weight.float(), ord=2).item()
    return head.Lambda * s


class RidgeHead(nn.Module):
    """w^T N(z) + b with ||w||_2 sqrt(d) <= Lambda (same Lipschitz constant in d_N)."""

    def __init__(self, d: int, Lambda: float):
        super().__init__()
        self.lin = nn.Linear(d, 1)
        self.Lambda, self.d = float(Lambda), d

    def forward(self, zn):
        w = self.lin.weight
        scale = torch.clamp(self.Lambda / (w.norm() * math.sqrt(self.d) + 1e-12), max=1.0)
        return (zn @ (w * scale).T).squeeze(-1) + self.lin.bias


class MLPHead(nn.Module):
    """Unconstrained head, for average-error comparisons only."""

    def __init__(self, d: int, widths=(64, 32)):
        super().__init__()
        dims = (d,) + tuple(widths)
        layers = []
        for i, o in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(i, o), nn.GELU()]
        self.net = nn.Sequential(*layers, nn.Linear(dims[-1], 1))

    def forward(self, zn):
        return self.net(zn).squeeze(-1)
