r"""
FFT phase-field fracture (AT1) on periodic 2D RVEs, small strain, plane strain.

Model
-----
- AT1 functional:  E = int g(d) psi(eps) + (3 Gc / 8) (d/ell + ell |grad d|^2),
  g(d) = (1-d)^2 + k_res.
- Hybrid formulation (Ambati et al. 2015): stress degraded isotropically
  (linear elasticity solve), damage driven by the tensile part psi+ of the
  volumetric--deviatoric (Amor) split.
- Irreversibility: history field H = max_t psi+ and the bound d >= d_prev.
  The AT1 damage sub-problem is a bound-constrained QP (lo <= d <= 1) solved
  exactly by a primal-dual active-set method with sparse direct solves; this
  yields the compactly supported AT1 profile and the AT1 elastic phase
  (d = 0 while psi+ < psi_c = 3 Gc / (16 ell)).
- Elasticity: Galerkin FFT with conjugate gradients (compatibility
  projection, warm start).
- Alternate minimization (staggered) to convergence at each load step,
  macroscopic uniaxial strain Ebar = eps * e1 (x) e1.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.fft as sfft


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


class PhaseFieldRVE:
    def __init__(self, mask, Em=1.0, Ei=10.0, nu=0.3, Gm=1.0, Gi=3.0,
                 ell=0.025, kres=1e-3, strip_x=0.95, strip_w=2.0 / 127,
                 strip_fac=0.98, split=True):
        self.N = N = mask.shape[0]
        self.h = 1.0 / N
        self.ell = ell
        self.kres = kres
        self.split = split
        lm, mm = lame(Em, nu)
        li, mi = lame(Ei, nu)
        self.lam = np.where(mask, li, lm)
        self.mu = np.where(mask, mi, mm)
        self.K = self.lam + 2.0 * self.mu / 3.0
        Gc = np.where(mask, Gi, Gm).astype(float)
        x = (np.arange(N) + 0.5) / N
        dx = (x - strip_x + 0.5) % 1.0 - 0.5
        strip = (np.abs(dx) < 0.5 * strip_w + 1e-12)[:, None] * np.ones((1, N), bool)
        Gc = np.where(strip, strip_fac * Gc, Gc)
        self.Gc = Gc
        self.psic = 3.0 * Gc / (16.0 * ell)
        self.Gx = 0.5 * (Gc + np.roll(Gc, -1, 0))
        self.Gy = 0.5 * (Gc + np.roll(Gc, -1, 1))
        k1, k2 = np.meshgrid(np.fft.fftfreq(N), np.fft.rfftfreq(N), indexing="ij")
        nrm = np.sqrt(k1 ** 2 + k2 ** 2)
        nrm[0, 0] = 1.0
        self.n1, self.n2 = k1 / nrm, k2 / nrm
        self.n1[0, 0] = self.n2[0, 0] = 0.0
        self.stats = {"cg_el": 0, "pdas": 0, "stag": 0}
        self._build_laplacian()

    # ---------------- elasticity ----------------
    def _C(self, g, e):
        e11, e22, e12 = e
        tr = e11 + e22
        a = g * self.lam * tr
        return (a + 2 * g * self.mu * e11, a + 2 * g * self.mu * e22, 2 * g * self.mu * e12)

    def _P(self, s):
        S11, S22, S12 = sfft.rfft2(np.stack(s), workers=1)
        n1, n2 = self.n1, self.n2
        t1 = S11 * n1 + S12 * n2
        t2 = S12 * n1 + S22 * n2
        nt = n1 * t1 + n2 * t2
        P = np.stack((2 * n1 * t1 - nt * n1 * n1,
                      2 * n2 * t2 - nt * n2 * n2,
                      n1 * t2 + n2 * t1 - nt * n1 * n2))
        out = sfft.irfft2(P, s=(self.N, self.N), workers=1)
        return (out[0], out[1], out[2])

    @staticmethod
    def _dot(a, b):
        return float(np.sum(a[0] * b[0] + a[1] * b[1] + 2 * a[2] * b[2]))

    def solve_elastic(self, d, Ebar, x0=None, tol=1e-6, maxit=4000):
        g = (1 - d) ** 2 + self.kres
        E = (np.full((self.N,) * 2, Ebar), np.zeros((self.N,) * 2), np.zeros((self.N,) * 2))
        b = tuple(-c for c in self._P(self._C(g, E)))
        if x0 is None:
            x = tuple(np.zeros_like(c) for c in b)
            r = b
        else:
            x = x0
            Ax = self._P(self._C(g, x))
            r = tuple(bb - aa for bb, aa in zip(b, Ax))
        bn = np.sqrt(max(self._dot(b, b), 1e-300))
        rr = self._dot(r, r)
        p = r
        it = 0
        while np.sqrt(rr) > tol * bn and it < maxit:
            Ap = self._P(self._C(g, p))
            alpha = rr / self._dot(p, Ap)
            x = tuple(xx + alpha * pp for xx, pp in zip(x, p))
            r = tuple(rv - alpha * ap for rv, ap in zip(r, Ap))
            rr_new = self._dot(r, r)
            p = tuple(rv + (rr_new / rr) * pp for rv, pp in zip(r, p))
            rr = rr_new
            it += 1
        self.stats["cg_el"] += it
        eps = (E[0] + x[0], x[1], x[2])
        sig = self._C(g, eps)
        return x, eps, sig

    def psi_plus(self, eps):
        e11, e22, e12 = eps
        tr = e11 + e22
        if not self.split:
            return 0.5 * self.lam * tr ** 2 + self.mu * (e11 ** 2 + e22 ** 2 + 2 * e12 ** 2)
        if self.split == "miehe":
            rad = np.sqrt((0.5 * (e11 - e22)) ** 2 + e12 ** 2)
            ep1 = np.maximum(0.5 * tr + rad, 0.0)
            ep2 = np.maximum(0.5 * tr - rad, 0.0)
            return 0.5 * self.lam * np.maximum(tr, 0.0) ** 2 + self.mu * (ep1 ** 2 + ep2 ** 2)
        dev2 = e11 ** 2 + e22 ** 2 + 2 * e12 ** 2 - tr ** 2 / 3.0
        return 0.5 * self.K * np.maximum(tr, 0.0) ** 2 + self.mu * dev2

    # ---------------- damage ----------------
    def _build_laplacian(self):
        """Sparse matrix of (3/4) ell * [-div(Gc grad .)] on the periodic grid."""
        N = self.N
        idx = np.arange(N * N).reshape(N, N)
        rows, cols, vals = [], [], []
        for w, nb in ((self.Gx, np.roll(idx, -1, 0)), (self.Gy, np.roll(idx, -1, 1))):
            a = idx.ravel(); b = nb.ravel(); w = w.ravel() * N * N
            rows += [a, b, a, b]; cols += [a, b, b, a]; vals += [w, w, -w, -w]
        L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(N * N, N * N))
        self._L = ((0.75 * self.ell) * L).tocsr()

    def solve_damage(self, H, lo, d0, maxit=60):
        """AT1 damage sub-problem: min 1/2 d.A.d - b.d subject to lo <= d <= 1,
        A = diag(2H) + (3/4) ell L_Gc,  b = 2H - 3Gc/(8 ell);
        primal-dual active-set iteration (finite for M-matrices)."""
        n = self.N * self.N
        A = (self._L + sp.diags(2.0 * H.ravel())).tocsr()
        b = (2.0 * H - 3.0 * self.Gc / (8.0 * self.ell)).ravel()
        lo = lo.ravel()
        hi = np.ones(n)
        d = np.clip(d0.ravel().copy(), lo, hi)
        diag = A.diagonal()
        act_lo = act_hi = None
        for it in range(maxit):
            mu = b - A @ d
            t = d + mu / diag
            new_lo = t <= lo
            new_hi = t >= hi
            if act_lo is not None and np.array_equal(new_lo, act_lo) and np.array_equal(new_hi, act_hi):
                break
            act_lo, act_hi = new_lo, new_hi
            free = ~(act_lo | act_hi)
            d = np.where(act_lo, lo, np.where(act_hi, hi, d))
            if free.any():
                Ff = np.flatnonzero(free)
                dB = np.where(free, 0.0, d)
                rhs = b[Ff] - (A @ dB)[Ff]
                d[Ff] = spla.spsolve(A[Ff][:, Ff].tocsc(), rhs)
        self.stats["pdas"] += it + 1
        return np.clip(d, lo, hi).reshape(self.N, self.N)

    def fracture_energy(self, d):
        N = self.N
        gx = (np.roll(d, -1, 0) - d) * N
        gy = (np.roll(d, -1, 1) - d) * N
        dens = 0.375 * (self.Gc * d / self.ell + self.ell * (self.Gx * gx ** 2 + self.Gy * gy ** 2))
        return float(dens.sum() * self.h ** 2)

    # ---------------- load path ----------------
    def run(self, dE_rel=0.01, emax_rel=6.0, tol_d=1e-3, max_stag=400,
            stop_frac=0.10, max_steps=800, verbose=False):
        """Monotone uniaxial-strain loading from the AT1 onset strain e0 until
        the secant stiffness drops below stop_frac of its initial value
        (complete separation, residual stiffness set by k_res)."""
        N = self.N
        d = np.zeros((N, N))
        x_unit, eps_u, sig_u = self.solve_elastic(d, 1.0, tol=1e-9)
        psi_u = self.psi_plus(eps_u)
        M0 = float(sig_u[0].mean())                     # initial secant stiffness
        e0 = float(np.sqrt(np.min(self.psic / np.maximum(psi_u, 1e-300))))
        Hh = np.zeros((N, N))
        dh = d.copy()
        x = tuple(e0 * c for c in x_unit)
        hist = []
        Ebar = e0
        step = 0
        stag_max = 0
        separated = False
        while step < max_steps and Ebar < emax_rel * e0:
            step += 1
            Ebar = e0 * (1.0 + dE_rel * (step - 1))
            d = dh.copy()
            for it in range(max_stag):
                x, eps, sig = self.solve_elastic(d, Ebar, x0=x)
                H = self.psi_plus(eps)
                dn = self.solve_damage(H, dh, d)
                change = float(np.max(np.abs(dn - d)))
                d = dn
                if change < tol_d:
                    break
            x, eps, sig = self.solve_elastic(d, Ebar, x0=x)
            self.stats["stag"] += it + 1
            stag_max = max(stag_max, it + 1)
            Hh = np.maximum(Hh, self.psi_plus(eps))
            dh = d
            s11 = float(sig[0].mean())
            G = self.fracture_energy(d)
            hist.append((Ebar, s11, G, float(d.max()), it + 1))
            if verbose:
                print(f"step {step} E={Ebar/e0:.3f}e0 s={s11:.4f} G={G:.4f} dmax={d.max():.3f} stag={it+1}", flush=True)
            if s11 / Ebar < stop_frac * M0:
                separated = True
                break
        hist = np.array(hist)
        ipk = int(np.argmax(hist[:, 1]))
        return {
            "e0": e0,
            "M0": M0,
            "sigma_peak": float(hist[ipk, 1]),
            "eps_peak": float(hist[ipk, 0]),
            "Gamma": float(hist[-1, 2]),
            "Gamma_at_peak": float(hist[ipk, 2]),
            "eps_sep": float(hist[-1, 0]),
            "work": float(np.trapezoid(np.r_[0.0, hist[:, 1]], np.r_[0.0, hist[:, 0]])),
            "n_steps": int(len(hist)),
            "stag_max": int(stag_max),
            "separated": separated,
            "cg_el": self.stats["cg_el"], "pdas": self.stats["pdas"],
        }, hist, d
