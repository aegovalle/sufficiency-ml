r"""
FFT-based periodic RVE solver and descriptor extraction for the
energy-informed descriptor-discovery campaign (Draft v0.5).

- 2D plane strain, small strain, isotropic linear-elastic phases.
- Moulinec--Suquet basic scheme with isotropic reference medium.
- Geometry is stored analytically and rasterized at any resolution N,
  so that resolution studies use the same microstructure.
"""
import numpy as np

# --------------------------------------------------------------------------
# Material
# --------------------------------------------------------------------------
def lame(E, nu):
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
class FFTElastic2D:
    def __init__(self, N):
        self.N = N
        k = np.fft.fftfreq(N)
        k1, k2 = np.meshgrid(k, k, indexing="ij")
        nrm = np.sqrt(k1 ** 2 + k2 ** 2)
        nrm[0, 0] = 1.0
        self.n1 = k1 / nrm
        self.n2 = k2 / nrm
        self.n1[0, 0] = 0.0
        self.n2[0, 0] = 0.0

    def solve(self, lam, mu, Ebar, tol=1e-9, maxit=5000):
        """Return strain (e11,e22,e12), stress (s11,s22,s12), iterations."""
        N = self.N
        lam0 = 0.5 * (lam.min() + lam.max())
        mu0 = 0.5 * (mu.min() + mu.max())
        c = (lam0 + mu0) / (mu0 * (lam0 + 2.0 * mu0))
        n1, n2 = self.n1, self.n2
        e11 = np.full((N, N), Ebar[0])
        e22 = np.full((N, N), Ebar[1])
        e12 = np.full((N, N), Ebar[2])
        for it in range(maxit):
            tr = e11 + e22
            s11 = lam * tr + 2.0 * mu * e11
            s22 = lam * tr + 2.0 * mu * e22
            s12 = 2.0 * mu * e12
            S11 = np.fft.fft2(s11)
            S22 = np.fft.fft2(s22)
            S12 = np.fft.fft2(s12)
            t1 = S11 * n1 + S12 * n2
            t2 = S12 * n1 + S22 * n2
            s0 = np.sqrt(abs(S11[0, 0]) ** 2 + abs(S22[0, 0]) ** 2
                         + 2 * abs(S12[0, 0]) ** 2)
            err = np.sqrt(np.sum(abs(t1) ** 2 + abs(t2) ** 2)) / max(s0, 1e-300)
            if err < tol:
                break
            nt = n1 * t1 + n2 * t2
            G11 = n1 * t1 / mu0 - c * nt * n1 * n1
            G22 = n2 * t2 / mu0 - c * nt * n2 * n2
            G12 = (n1 * t2 + n2 * t1) / (2.0 * mu0) - c * nt * n1 * n2
            E11 = np.fft.fft2(e11) - G11
            E22 = np.fft.fft2(e22) - G22
            E12 = np.fft.fft2(e12) - G12
            E11[0, 0] = Ebar[0] * N * N
            E22[0, 0] = Ebar[1] * N * N
            E12[0, 0] = Ebar[2] * N * N
            e11 = np.real(np.fft.ifft2(E11))
            e22 = np.real(np.fft.ifft2(E22))
            e12 = np.real(np.fft.ifft2(E12))
        tr = e11 + e22
        s11 = lam * tr + 2.0 * mu * e11
        s22 = lam * tr + 2.0 * mu * e22
        s12 = 2.0 * mu * e12
        return (e11, e22, e12), (s11, s22, s12), it, err


# --------------------------------------------------------------------------
# Geometry (analytic, unit periodic cell) and rasterization
# --------------------------------------------------------------------------
def _grid(N):
    x = (np.arange(N) + 0.5) / N
    return np.meshgrid(x, x, indexing="ij")


def _pdisp(X, c):
    return (X - c + 0.5) % 1.0 - 0.5


def rasterize(geom, N):
    X1, X2 = _grid(N)
    mask = np.zeros((N, N), dtype=bool)
    if geom["type"] == "laminate":
        coord = X2 if geom["orient"] == "h" else X1
        for (a, t) in geom["layers"]:
            d = (coord - a) % 1.0
            mask |= d < t
    elif geom["type"] in ("disks", "ellipses"):
        for (cx, cy, a, b, th) in geom["objs"]:
            d1 = _pdisp(X1, cx)
            d2 = _pdisp(X2, cy)
            u = np.cos(th) * d1 + np.sin(th) * d2
            v = -np.sin(th) * d1 + np.cos(th) * d2
            mask |= (u / a) ** 2 + (v / b) ** 2 <= 1.0
    return mask


def gen_laminate(f, nlayers, orient):
    t = f / nlayers
    layers = [(k / nlayers, t) for k in range(nlayers)]
    return {"type": "laminate", "orient": orient, "layers": layers}


def gen_inclusions(f, n, rng, ar=1.0, gap=2.0 / 128, Ncheck=256, tries=20000):
    """Random sequential addition of non-overlapping ellipses (ar=1: disks)."""
    r_eq = np.sqrt(f / (n * np.pi))
    a = r_eq * np.sqrt(ar)
    b = r_eq / np.sqrt(ar)
    for _restart in range(50):
        objs = []
        occ = np.zeros((Ncheck, Ncheck), dtype=bool)
        ok = True
        for _k in range(n):
            placed = False
            for _t in range(tries):
                cx, cy = rng.random(2)
                th = rng.random() * np.pi if ar > 1.0 else 0.0
                cand = {"type": "ellipses",
                        "objs": [(cx, cy, a + gap, b + gap, th)]}
                cm = rasterize(cand, Ncheck)
                if not np.any(cm & occ):
                    objs.append((cx, cy, a, b, th))
                    occ |= rasterize({"type": "ellipses",
                                      "objs": [(cx, cy, a, b, th)]}, Ncheck)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return {"type": "disks" if ar == 1.0 else "ellipses", "objs": objs}
    raise RuntimeError("placement failed")


# --------------------------------------------------------------------------
# Descriptors and targets
# --------------------------------------------------------------------------
def _disk_kernel_hat(N, ell):
    X1, X2 = _grid(N)
    d1 = _pdisp(X1, 0.5 / N)
    d2 = _pdisp(X2, 0.5 / N)
    K = ((d1 ** 2 + d2 ** 2) <= ell ** 2).astype(float)
    K /= K.sum()
    return np.fft.fft2(K)


def interface_density(mask):
    m = mask.astype(int)
    N = m.shape[0]
    edges = np.sum(m != np.roll(m, 1, 0)) + np.sum(m != np.roll(m, 1, 1))
    return edges / N  # edge length h=1/N, cell area 1


def analyze_state(mask, strain, stress, mats, Ebar, m=20, ell=2.0 / 128,
                  keep_field=False):
    lam_m, mu_m, lam_i, mu_i = mats
    e11, e22, e12 = strain
    s11, s22, s12 = stress
    N = mask.shape[0]
    h2 = 1.0 / N ** 2
    inc = mask
    mat = ~mask
    f = inc.mean()
    out = {"f": f, "rhoI": interface_density(mask)}

    w = 0.5 * (s11 * e11 + s22 * e22 + 2.0 * s12 * e12)
    M_m = lam_m + 2.0 * mu_m
    Wref = 0.5 * M_m * Ebar[0] ** 2
    out["W_hom"] = w.mean() / Wref

    def W_of(lam, mu, a11, a22, a12):
        tr = a11 + a22
        return 0.5 * (lam * tr ** 2 + 2 * mu * (a11 ** 2 + a22 ** 2 + 2 * a12 ** 2))

    for tag, ph, lam, mu in (("m", mat, lam_m, mu_m), ("i", inc, lam_i, mu_i)):
        if ph.sum() == 0:
            for k in ("e11", "e22", "e12", "Jgap"):
                out[f"{k}_{tag}"] = 0.0
            continue
        a11, a22, a12 = e11[ph].mean(), e22[ph].mean(), e12[ph].mean()
        out[f"e11_{tag}"] = a11 / Ebar[0]
        out[f"e22_{tag}"] = a22 / Ebar[0]
        out[f"e12_{tag}"] = a12 / Ebar[0]
        out[f"Jgap_{tag}"] = (w[ph].mean() - W_of(lam, mu, a11, a22, a12)) / Wref

    # Hill-Mandel reconstruction of W_hom from phase means
    sig11 = 0.0
    for tag, fa, lam, mu in (("m", 1 - f, lam_m, mu_m), ("i", f, lam_i, mu_i)):
        a11 = out[f"e11_{tag}"] * Ebar[0]
        a22 = out[f"e22_{tag}"] * Ebar[0]
        sig11 += fa * (lam * (a11 + a22) + 2 * mu * a11)
    out["W_HM"] = 0.5 * sig11 * Ebar[0] / Wref

    # global means (null-Lagrangian / Hill control: fixed by Ebar)
    out["e11_g"] = e11.mean() / Ebar[0]
    out["e22_g"] = e22.mean() / Ebar[0]
    out["e12_g"] = e12.mean() / Ebar[0]

    # Nonlocal Beremin-type hazard in the matrix
    s1 = 0.5 * (s11 + s22) + np.sqrt((0.5 * (s11 - s22)) ** 2 + s12 ** 2)
    s1p = np.maximum(s1, 0.0) * mat
    Kh = _disk_kernel_hat(N, ell)
    num = np.real(np.fft.ifft2(np.fft.fft2(s1p) * Kh))
    den = np.real(np.fft.ifft2(np.fft.fft2(mat.astype(float)) * Kh))
    sbar = np.where(mat & (den > 1e-12), num / np.maximum(den, 1e-12), 0.0)
    sref = M_m * Ebar[0]
    sm = sbar[mat] / sref
    sp = sm.max()
    I = m * np.log(sp)
    Veff = np.sum((sm / sp) ** m) * h2
    alpha = 0.01 ** (1.0 / m)
    Valpha = np.sum(sm >= alpha * sp) * h2
    out["I"] = I
    out["lnVeff"] = np.log(Veff)
    out["lnValpha"] = np.log(Valpha)
    out["lnB"] = I + np.log(Veff)
    out["e_alpha"] = np.log(Valpha / Veff)
    out["phi_alpha"] = I + np.log(Valpha)
    if keep_field:
        out["_sbar"] = (sbar / sref).astype(np.float32)
    return out
