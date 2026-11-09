"""Convolutional autoencoder baseline (Draft v0.2, Sections 10.3 and 12.4). The encoder is a
standard strided network (its latent is not translation invariant, as in the usual AE baseline);
the 127x127 image is padded circularly to 128x128 and the reconstruction cropped back."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAE(nn.Module):
    def __init__(self, d: int, ch=(16, 32, 64, 64)):
        super().__init__()
        enc, c_in = [], 1
        for c in ch:                                       # 128 -> 64 -> 32 -> 16 -> 8
            enc += [nn.Conv2d(c_in, c, 4, stride=2, padding=1), nn.GroupNorm(8, c), nn.GELU()]
            c_in = c
        self.enc = nn.Sequential(*enc)
        self.to_z = nn.Linear(c_in * 8 * 8, d)
        self.from_z = nn.Linear(d, c_in * 8 * 8)
        dec = []
        chs = list(ch[::-1]) + [16]                        # 8 -> 16 -> 32 -> 64 -> 128
        for a, b in zip(chs[:-1], chs[1:]):
            dec += [nn.ConvTranspose2d(a, b, 4, stride=2, padding=1), nn.GroupNorm(8, b), nn.GELU()]
        dec += [nn.Conv2d(chs[-1], 1, 3, padding=1)]
        self.dec = nn.Sequential(*dec)
        self.c_last, self.d = c_in, d

    @staticmethod
    def pad(x):
        return F.pad(x, (0, 1, 0, 1), mode="circular")     # 127 -> 128

    def encode(self, x):
        h = self.enc(self.pad(x))
        return self.to_z(h.flatten(1))

    def decode(self, z):
        h = self.from_z(z).view(-1, self.c_last, 8, 8)
        return self.dec(h)[..., :127, :127]                 # logits

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
