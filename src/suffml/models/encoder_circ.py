"""Encoders (Draft v0.2, Section 12.1, Item I09):

1. ECirc: circular padding, no striding, dilations 1-32, receptive field 129 px; every feature map
   is exactly equivariant to circular integer shifts, so the pooled embedding is exactly invariant.
2. EFNO: Fourier Neural Operator encoder combining 2D spectral convolutions on the low Fourier modes
   with local circular 3x3 convolutions, GroupNorm, and GELU; also exactly equivariant to circular
   integer shifts.
3. EStd: Standard CNN with zero padding (breaking exact periodic shift invariance at cell boundaries)
   and evaluated without D2 reflection averaging (Ablation 9).
4. d2_embed: exact invariance to the reflections x -> -x, y -> -y by averaging over the group.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pooling import GAP, GeM


class ECirc(nn.Module):
    def __init__(self, d: int, ch=(16, 32, 32, 64, 64, 64), dil=(1, 2, 4, 8, 16, 32), pool: str = "gem",
                 p0: float = 3.0):
        super().__init__()
        layers, c_in = [], 1
        for i, (c, r) in enumerate(zip(ch, dil)):
            k = 5 if i == 0 else 3
            layers += [nn.Conv2d(c_in, c, k, dilation=r, padding=r * (k // 2), padding_mode="circular"),
                       nn.GroupNorm(8, c), nn.GELU()]
            c_in = c
        self.trunk = nn.Sequential(*layers)
        self.pool = GeM(p0) if pool == "gem" else GAP()
        self.proj = nn.Linear(c_in, d)
        self.d = d

    def features(self, x):
        return F.softplus(self.trunk(x))

    def forward(self, x):                      # x: B x 1 x H x W
        return self.proj(self.pool(self.features(x)))


class EStd(nn.Module):
    """Standard CNN with zero-padding (no circular padding) for Ablation 9 (Item I09)."""

    def __init__(self, d: int, ch=(16, 32, 32, 64, 64, 64), dil=(1, 2, 4, 8, 16, 32), pool: str = "gem",
                 p0: float = 3.0):
        super().__init__()
        layers, c_in = [], 1
        for i, (c, r) in enumerate(zip(ch, dil)):
            k = 5 if i == 0 else 3
            layers += [nn.Conv2d(c_in, c, k, dilation=r, padding=r * (k // 2), padding_mode="zeros"),
                       nn.GroupNorm(8, c), nn.GELU()]
            c_in = c
        self.trunk = nn.Sequential(*layers)
        self.pool = GeM(p0) if pool == "gem" else GAP()
        self.proj = nn.Linear(c_in, d)
        self.d = d

    def features(self, x):
        return F.softplus(self.trunk(x))

    def forward(self, x):
        return self.proj(self.pool(self.features(x)))


class SpectralConv2d(nn.Module):
    """2D Fourier layer (Li et al. 2021) retaining the lowest (modes1, modes2) frequency modes."""

    def __init__(self, c_in: int, c_out: int, modes1: int = 12, modes2: int = 12):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (c_in * c_out) ** 0.5
        self.w1_real = nn.Parameter(scale * torch.randn(c_in, c_out, modes1, modes2))
        self.w1_imag = nn.Parameter(scale * torch.randn(c_in, c_out, modes1, modes2))
        self.w2_real = nn.Parameter(scale * torch.randn(c_in, c_out, modes1, modes2))
        self.w2_imag = nn.Parameter(scale * torch.randn(c_in, c_out, modes1, modes2))

    def _compl_mul2d(self, x_ft, wr, wi):
        w = torch.complex(wr, wi)
        return torch.einsum("bixy,ioxy->boxy", x_ft, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        B, _, H, W = xf.shape
        x_ft = torch.fft.rfft2(xf)
        out_ft = torch.zeros(B, self.c_out, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1, m2 = min(self.modes1, H // 2), min(self.modes2, W // 2 + 1)
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(
            x_ft[:, :, :m1, :m2], self.w1_real[:, :, :m1, :m2], self.w1_imag[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.w2_real[:, :, :m1, :m2], self.w2_imag[:, :, :m1, :m2]
        )
        out = torch.fft.irfft2(out_ft, s=(H, W))
        return out.to(orig_dtype)


class FNOBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, modes: int = 12):
        super().__init__()
        self.spec = SpectralConv2d(c_in, c_out, modes, modes)
        self.local = nn.Conv2d(c_in, c_out, 3, padding=1, padding_mode="circular")
        self.norm = nn.GroupNorm(8, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spec(x) + self.local(x)))


class EFNO(nn.Module):
    """Fourier Neural Operator encoder (Item I09): local circular stem + 4 FNO blocks + GeM/GAP pooling."""

    def __init__(self, d: int, ch=(24, 32, 48, 64), modes: int = 12, pool: str = "gem", p0: float = 3.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, ch[0], 5, padding=2, padding_mode="circular"),
            nn.GroupNorm(8, ch[0]),
            nn.GELU(),
        )
        blocks = []
        c_in = ch[0]
        for c_out in ch:
            blocks.append(FNOBlock(c_in, c_out, modes=modes))
            c_in = c_out
        self.trunk = nn.Sequential(*blocks)
        self.pool = GeM(p0) if pool == "gem" else GAP()
        self.proj = nn.Linear(c_in, d)
        self.d = d

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.trunk(self.stem(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.features(x)))


def d2_embed(encoder, x):
    """Average of the embedding over identity, the two reflections and their composition."""
    views = (x, x.flip(-1), x.flip(-2), x.flip((-2, -1)))
    return torch.stack([encoder(v) for v in views]).mean(0)
