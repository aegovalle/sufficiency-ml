"""Unit tests for EFNO and EStd encoders (Item I09, Ablation 9)."""
import torch

from suffml.models.encoder_circ import EFNO, EStd, d2_embed


def test_efno_shift_and_reflection_invariance():
    torch.manual_seed(0)
    enc = EFNO(d=8, ch=(16, 24), modes=8, pool="gem").eval()
    x = torch.rand(2, 1, 64, 64)
    x_shift = torch.roll(x, shifts=(13, -7), dims=(-2, -1))
    x_flip = x.flip(-1)

    with torch.no_grad():
        z0 = d2_embed(enc, x)
        z_shift = d2_embed(enc, x_shift)
        z_flip = d2_embed(enc, x_flip)

    assert torch.allclose(z0, z_shift, atol=1e-5)
    assert torch.allclose(z0, z_flip, atol=1e-5)


def test_estd_breaks_shift_invariance():
    torch.manual_seed(0)
    enc = EStd(d=8, ch=(16, 24), dil=(1, 2), pool="gem").eval()
    x = torch.rand(2, 1, 64, 64)
    x_shift = torch.roll(x, shifts=(13, -7), dims=(-2, -1))

    with torch.no_grad():
        z0 = enc(x)
        z_shift = enc(x_shift)

    # Zero-padding breaks periodic shift invariance
    assert not torch.allclose(z0, z_shift, atol=1e-4)
