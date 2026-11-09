"""Metrics of the latent space (Draft v0.2, Section 5.4): training-set whitening divided by
sqrt(d) (d_white), per-coordinate standardization divided by sqrt(d) (d_N), and the
standardized Euclidean metric of the descriptor study."""
from __future__ import annotations

import numpy as np


class WhiteningTransform:
    """z -> (Sigma + delta I)^{-1/2} (z - mu) / sqrt(d), fitted on training embeddings only."""

    def __init__(self, delta: float = 1e-9, scale_sqrt_d: bool = True):
        self.delta, self.scale = delta, scale_sqrt_d

    def fit(self, z: np.ndarray):
        z = np.asarray(z, float)
        self.mu = z.mean(0)
        C = np.cov((z - self.mu).T).reshape(z.shape[1], z.shape[1]) + self.delta * np.eye(z.shape[1])
        w, V = np.linalg.eigh(C)
        self.W = V @ np.diag(w ** -0.5) @ V.T
        self.d = z.shape[1]
        return self

    def transform(self, z: np.ndarray) -> np.ndarray:
        out = (np.asarray(z, float) - self.mu) @ self.W
        return out / np.sqrt(self.d) if self.scale else out

    def fit_transform(self, z):
        return self.fit(z).transform(z)


class Standardizer:
    """Per-coordinate standardization with training statistics, divided by sqrt(d) (d_N)."""

    def __init__(self, scale_sqrt_d: bool = True, eps: float = 1e-12):
        self.scale, self.eps = scale_sqrt_d, eps

    def fit(self, z):
        z = np.asarray(z, float)
        self.mu, self.sd = z.mean(0), z.std(0)
        self.sd = np.where(self.sd > self.eps, self.sd, 1.0)
        self.d = z.shape[1]
        return self

    def transform(self, z):
        out = (np.asarray(z, float) - self.mu) / self.sd
        return out / np.sqrt(self.d) if self.scale else out

    def fit_transform(self, z):
        return self.fit(z).transform(z)


def whiten_pilot(z: np.ndarray) -> np.ndarray:
    """In-sample whitening used by the pilot script (sample covariance + 1e-9 I, / sqrt(d))."""
    return WhiteningTransform(1e-9).fit_transform(z)
