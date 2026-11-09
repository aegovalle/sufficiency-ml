"""Unit tests for the Experiment D/E losses and the miner (I11, I12, I14)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from suffml.losses.continuous_label import rnc_loss, ranksim_loss
from suffml.metrics import mcshane_mid

pytestmark = pytest.mark.torch


def rnc_reference(z, y, tau):
    """Loop implementation of the Rank-N-Contrast loss."""
    B = len(y); tot = 0.0
    for i in range(B):
        for j in range(B):
            if i == j:
                continue
            s = lambda a, b: -np.linalg.norm(z[a] - z[b]) / tau
            den = [np.exp(s(i, k)) for k in range(B) if k != i and abs(y[i] - y[k]) >= abs(y[i] - y[j])]
            tot += -(s(i, j) - np.log(np.sum(den)))
    return tot / (B * (B - 1))


def test_rnc_matches_reference():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((12, 3)); y = rng.standard_normal(12)
    got = rnc_loss(torch.tensor(z), torch.tensor(y), tau=2.0).item()
    assert got == pytest.approx(rnc_reference(z, y, 2.0), rel=1e-9)


def test_rnc_prefers_label_ordered_embedding():
    y = torch.linspace(-1, 1, 32, dtype=torch.float64)
    good = y[:, None] * 3.0
    bad = torch.randn(32, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) * 3.0
    assert rnc_loss(good, y) < rnc_loss(bad, y)


def test_ranksim_zero_for_perfect_ranking_and_gradients():
    # irregular labels: no ties in label distances (a regular grid has symmetric ties)
    y = torch.tensor(np.sort(np.random.default_rng(4).random(16)), dtype=torch.float64)
    ang = (y * 1.5)[:, None]
    z = torch.cat([torch.cos(ang), torch.sin(ang)], 1).requires_grad_(True)
    loss = ranksim_loss(z, y)
    assert loss.item() < 1e-12
    zr = torch.randn(16, 2, dtype=torch.float64, requires_grad=True)
    l2 = ranksim_loss(zr, y); l2.backward()
    assert l2.item() > 0 and torch.isfinite(zr.grad).all()


def test_loo_miner_matches_numpy():
    from suffml.train.train_objectives import mine_hard_queries

    class Dummy(torch.nn.Module):
        def __init__(self, Z):
            super().__init__(); self.Z = Z
            self.norm = torch.nn.Identity()
        def embed(self, x):
            return self.Z[x[:, 0, 0, 0].long()]

    rng = np.random.default_rng(3)
    Z = rng.standard_normal((50, 4)); y = rng.standard_normal(50)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = Dummy(torch.tensor(Z, dtype=torch.float32, device=dev))
    X = torch.arange(50, dtype=torch.float32, device=dev)[:, None, None, None].expand(50, 1, 1, 1).contiguous()
    top, res = mine_hard_queries(m, X, torch.tensor(y, dtype=torch.float32, device=dev), 2.0, 5)
    zm = Z / 2.0
    ref = np.array([abs(mcshane_mid(np.delete(zm, i, 0), np.delete(y, i), zm[i:i + 1], 2.0)[0] - y[i]) for i in range(50)])
    assert np.allclose(res.cpu().numpy(), ref, atol=1e-4)
    assert set(top.cpu().tolist()) == set(np.argsort(-ref)[:5].tolist())
