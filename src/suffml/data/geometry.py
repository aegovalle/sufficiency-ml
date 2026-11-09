"""Geometry generation: exact regeneration of the 216 design cells of the descriptor study
(Stage 0) and sampling of the scalable elastic corpus (Stage 1, Draft v0.2 Section 9.2)."""
from __future__ import annotations

import numpy as np

from ..physics.fft_rve import rasterize, gen_inclusions

N_DEFAULT = 127


def gen_laminate_rows(rows, nl, orient, Ngrid=N_DEFAULT):
    """Laminate with nl layers of `rows` grid rows in total (descriptor study, campaign_run.py)."""
    r = rows // nl
    layers = [(np.floor(k * Ngrid / nl) / Ngrid, r / Ngrid) for k in range(nl)]
    return {"type": "laminate", "orient": orient, "layers": layers}


def design_geometries():
    """The 216 design cells in the order and with the seed of campaign_run.py.
    Returns a list of (geometry, metadata) tuples."""
    rng = np.random.default_rng(20261008)
    out = []
    for nrow in (16, 24, 32, 40, 48, 64):
        for orient in ("h", "v"):
            for nl in (1, 2, 4):
                out.append((gen_laminate_rows(nrow, nl, orient),
                            dict(family="laminate", f_nom=round(nrow / N_DEFAULT, 4), n=nl, orient=orient,
                                 real=nl, cell=f"lam_{nrow}_{orient}")))
    for fam, ar, ns in (("disks", 1.0, (1, 4, 9, 16)), ("ellipses", 3.0, (4, 9))):
        for f in (0.1, 0.2, 0.3):
            for n in ns:
                for r in range(10):
                    out.append((gen_inclusions(f, n, rng, ar=ar),
                                dict(family=fam, f_nom=f, n=n, orient="-", real=r, cell=f"{fam}_{f}_{n}")))
    return out


# ------------------------------------------------------------------------------- Stage 1
DEFAULT_CORPUS = {
    "families": {"disks": 0.5, "ellipses": 0.5},
    "f_range": [0.05, 0.35],
    "n_values": [1, 2, 4, 6, 9, 12, 16, 20, 25],
    "ar_range": [1.5, 4.0],            # ellipses; orientation uniform in [0, pi)
}
OOD_MORPHOLOGY = {
    "families": {"ellipses": 0.5, "disks": 0.5},
    "f_range": [0.05, 0.35],
    "n_values": [30, 36],               # more inclusions than any training cell
    "ar_range": [5.0, 7.0],             # ellipses more slender than any training cell
}


def sample_geometry(rng, cfg=DEFAULT_CORPUS, max_tries: int = 20):
    """One random cell from the corpus distribution; placement failures are resampled."""
    fams = list(cfg["families"]); p = np.array([cfg["families"][k] for k in fams], float)
    for _ in range(max_tries):
        fam = fams[rng.choice(len(fams), p=p / p.sum())]
        f = float(rng.uniform(*cfg["f_range"]))
        n = int(rng.choice(cfg["n_values"]))
        ar = 1.0 if fam == "disks" else float(rng.uniform(*cfg["ar_range"]))
        try:
            g = gen_inclusions(f, n, rng, ar=ar, tries=4000)
        except RuntimeError:
            continue
        return g, dict(family=fam, f_nom=f, n=n, ar=ar)
    raise RuntimeError("could not place a cell after resampling")


def mask_of(geom, N=N_DEFAULT):
    return rasterize(geom, N)
