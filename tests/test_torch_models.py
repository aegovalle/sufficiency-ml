"""I09 (exact invariance), I10 (GeM), I12 (cross-fitted loss), I13 (Lipschitz head)."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from suffml.models.encoder_circ import ECirc, d2_embed
from suffml.models.pooling import GeM
from suffml.models.lipschitz_head import LipschitzHead, certified_lambda, finalize_spectral
from suffml.losses.crossfit_barrier import crossfit_barrier, mcshane_mid_torch
from suffml.metrics import mcshane_mid, exact_barrier

pytestmark = pytest.mark.torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_ecirc_exact_shift_and_reflection_invariance():
    torch.manual_seed(0)
    enc = ECirc(8, pool="gem").to(DEV).double().eval()
    x = (torch.rand(2, 1, 127, 127, device=DEV) < 0.3).double()
    with torch.no_grad():
        z = d2_embed(enc, x)
        zs = d2_embed(enc, torch.roll(x, shifts=(17, 53), dims=(-2, -1)))
        zf = d2_embed(enc, x.flip(-1))
        zff = d2_embed(enc, torch.roll(x.flip((-2, -1)), shifts=(5, 100), dims=(-2, -1)))
    for other in (zs, zf, zff):
        assert torch.allclose(z, other, atol=1e-9, rtol=1e-9)


def test_receptive_field_covers_cell():
    enc = ECirc(4, pool="gap").double()
    x = torch.zeros(1, 1, 127, 127, dtype=torch.float64, requires_grad=True)
    y = enc.trunk(x)[0, :, 0, 0].sum()
    y.backward()
    assert (x.grad[0, 0].abs() > 0).float().mean().item() == 1.0


def test_gem_matches_identity_and_limits():
    g = GeM(p0=20.0).double()
    F = torch.exp(0.3 * torch.randn(3, 4, 127, 127, dtype=torch.float64))
    p = g.p.item()
    ref = (F ** p).mean(dim=(-2, -1)) ** (1 / p)
    assert torch.allclose(g(F), ref, rtol=1e-10)
    g40 = GeM(p0=41.0).double()
    assert torch.isfinite(g40(F * 50)).all()


def test_lipschitz_head_certification_and_floor():
    torch.manual_seed(1)
    d, Lam = 8, 2.0
    head = LipschitzHead(d, Lam).double()
    opt = torch.optim.Adam(head.parameters(), 1e-2)
    zn = torch.randn(256, d, dtype=torch.float64)
    y = torch.sin(zn[:, 0]) + 0.1 * torch.randn(256, dtype=torch.float64)
    for _ in range(300):
        opt.zero_grad(); loss = ((head(zn) - y) ** 2).mean(); loss.backward(); opt.step()
    head.eval()
    finalize_spectral(head)                   # converge the power iteration
    Lc = certified_lambda(head)
    assert Lc <= (1 + 1e-3) * Lam
    with torch.no_grad():
        pred = head(zn).numpy()
    einf = np.abs(pred - y.numpy()).max()
    z_metric = zn.numpy() / math.sqrt(d)
    assert einf >= exact_barrier(z_metric, y.numpy(), Lc) - 1e-9
    # empirical Lipschitz check
    a, b = zn[:128], zn[128:]
    with torch.no_grad():
        dg = (head(a) - head(b)).abs()
    dz = (a - b).norm(dim=1) / math.sqrt(d)
    assert torch.all(dg <= Lc * dz + 1e-9)


def test_crossfit_loss_matches_numpy_and_gradients_finite():
    torch.manual_seed(2)
    z = torch.randn(64, 4, dtype=torch.float64, requires_grad=True)
    y = torch.randn(64, dtype=torch.float64)
    ids = torch.arange(64)
    m_t = mcshane_mid_torch(z[:32], y[:32], z[32:], 1.5).detach().numpy()
    m_n = mcshane_mid(z[:32].detach().numpy(), y[:32].numpy(), z[32:].detach().numpy(), 1.5)
    assert np.allclose(m_t, m_n, atol=1e-7)
    loss = crossfit_barrier(z, y, ids, Lambda=1.5, eps=0.0, k=8)
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_crossfit_loss_excludes_same_cell_anchors():
    z = torch.zeros(4, 2, dtype=torch.float64)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    ids = torch.tensor([0, 1, 0, 1])
    zq = z[:2]
    same = ids[:2][:, None] == ids[2:][None, :]
    m = mcshane_mid_torch(z[2:], y[2:], zq, 1.0, same)
    # query 0 may only use anchor 3 (y = 1), query 1 only anchor 2 (y = 0)
    assert torch.allclose(m, torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_scale_invariance_through_normalization():
    """Multiplying the raw embedding by a constant does not change d_N."""
    from suffml.models.normalization import LatentNorm
    n = LatentNorm(4).double().train()
    z = torch.randn(128, 4, dtype=torch.float64)
    assert torch.allclose(n(z), n(1000 * z), atol=1e-6)
