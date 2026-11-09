"""Shared tools for the v0.6 analysis: geometric descriptors (ligament, minimum cut),
crack-path cost, insufficiency statistics with subsampling intervals, and
cross-validated surrogates (random K-fold and leave-one-nominal-cell-out)."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import pdist

# ---------------------------------------------------------------- geometry
def ligament(mask, ell=0.025):
    """Matrix ligaments along the loading direction e1 (axis 0 of the mask).
    For each line x -> mask[x, y] at fixed y, delta(y) is the shortest matrix run
    bounded by inclusion pixels (periodic); lines without inclusions or without
    matrix give delta = 1 (no ligament).  Returns
      lig_min = ln min_y delta(y)                      (narrowest gap)
      lig_ell = ln min_y <delta>_{|y'-y|<=ell}         (gap averaged over the
                                                         transverse width 2*ell)"""
    N = mask.shape[0]
    delta = np.ones(N)
    for j in range(N):
        col = mask[:, j]
        if col.all() or not col.any():
            continue
        k0 = int(np.flatnonzero(col)[0])
        c = np.roll(col, -k0)                 # starts with an inclusion pixel
        runs, cur = [], 0
        for v in c[1:]:
            if v:
                if cur > 0:
                    runs.append(cur)
                cur = 0
            else:
                cur += 1
        if cur > 0:
            runs.append(cur)
        if runs:
            delta[j] = min(runs) / N
    w = max(1, int(round(ell * N)))
    ker = np.ones(2 * w + 1) / (2 * w + 1)
    dpad = np.r_[delta[-w:], delta, delta[:w]]
    davg = np.convolve(dpad, ker, mode="valid")
    return float(np.log(delta.min())), float(np.log(davg.min()))


def _runs(col):
    """Periodic run-length encoding of a boolean line -> list of (value, length),
    starting at an inclusion run (col must contain both values)."""
    k0 = int(np.flatnonzero(col & ~np.roll(col, 1))[0])   # start of an inclusion run
    c = np.roll(col, -k0)
    edges = np.flatnonzero(np.diff(c.astype(int))) + 1
    starts = np.r_[0, edges]
    lens = np.diff(np.r_[starts, len(c)])
    return [(bool(c[s]), int(L)) for s, L in zip(starts, lens)]


def ligament_amp(mask, k=10.0, ell=0.025):
    """Series-model strain amplification of matrix ligaments loaded along e1.
    On each line y (axis 0 = load direction) every matrix gap delta bounded by
    inclusion chords c1, c2 is a soft spring in series with the two half chords:
        A = (cbar + delta) / (cbar/k + delta),   cbar = (c1 + c2)/2,
    k = E_i/E_m.  A_line(y) = max over gaps; lines without inclusions give 1.
    Returns ln max_y A_line and ln max_y <A_line>_{|y'-y|<=ell}.  Only the
    phase map and the contrast enter: no field solution is required."""
    N = mask.shape[0]
    A = np.ones(N)
    for j in range(N):
        col = mask[:, j]
        if col.all() or not col.any():
            continue
        r = _runs(col)
        m = len(r)
        best = 1.0
        for t, (v, L) in enumerate(r):
            if v:
                continue
            c1 = r[(t - 1) % m][1]
            c2 = r[(t + 1) % m][1]
            cb = 0.5 * (c1 + c2)
            best = max(best, (cb + L) / (cb / k + L))
        A[j] = best
    w = max(1, int(round(ell * N)))
    ker = np.ones(2 * w + 1) / (2 * w + 1)
    Apad = np.r_[A[-w:], A, A[:w]]
    Aavg = np.convolve(Apad, ker, mode="valid")
    return float(np.log(A.max())), float(np.log(Aavg.max()))


HALF_OFFS = [(1, 0), (0, 1), (1, 1), (-1, 1), (1, 2), (-1, 2), (2, 1), (-2, 1)]
HALF_OFFS8 = HALF_OFFS[:4]          # 8-neighbour stencil (sensitivity check only)


def min_path(Gc, allowed, offs=None):
    """min of int Gc ds over closed paths wrapping once in y, restricted to
    'allowed' cells (16-neighbour stencil by default, periodic in x). Returns value or nan."""
    offs = HALF_OFFS if offs is None else offs
    Nn = Gc.shape[0]
    h = 1.0 / Nn
    II, JJ = np.meshgrid(np.arange(Nn), np.arange(Nn + 1), indexing="ij")
    A = allowed[II, JJ % Nn]
    G = Gc[II, JJ % Nn]
    nid = JJ * Nn + II
    rows, cols, w = [], [], []
    for di, dj in offs:
        i2 = (II + di) % Nn
        j2 = JJ + dj
        ok = (j2 >= 0) & (j2 <= Nn)
        j2c = np.clip(j2, 0, Nn)
        ok &= A & A[i2, j2c]
        rows.append(nid[ok]); cols.append((j2c * Nn + i2)[ok])
        w.append(np.hypot(di, dj) * h * 0.5 * (G[ok] + G[i2, j2c][ok]))
    Mx = sp.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(Nn * (Nn + 1),) * 2)
    src = np.flatnonzero(A[:, 0])
    if len(src) == 0:
        return np.nan
    dist = dijkstra(Mx, directed=False, indices=src)
    vals = dist[np.arange(len(src)), Nn * Nn + src]
    v = float(np.min(vals))
    return v if np.isfinite(v) else np.nan


# ---------------------------------------------------------------- statistics
def standardize(A):
    A = np.asarray(A, float)
    s = A.std(0)
    keep = s > 1e-9
    Z = np.zeros_like(A)
    Z[:, keep] = (A[:, keep] - A[:, keep].mean(0)) / s[keep]
    return Z, keep


def omega(Xd, R, q, nperm, rng, return_pair=False):
    """Equal-coverage insufficiency: max |dR| over the ceil(q*npairs) closest
    pairs in the standardized descriptor metric (all ties kept), and the mean
    of the same statistic under nperm permutations of R."""
    n = len(R)
    II, JJ = np.triu_indices(n, 1)
    Z, _ = standardize(Xd)
    d = pdist(Z) if Z.shape[1] else np.zeros(len(II))
    nq = int(np.ceil(q * len(d)))
    dcut = np.partition(d, nq - 1)[nq - 1]
    sel = d <= dcut + 1e-12
    dR = np.abs(R[II] - R[JJ])
    om = dR[sel].max()
    Is, Js = II[sel], JJ[sel]
    nulls = np.empty(nperm)
    for k in range(nperm):
        p = rng.permutation(R)
        nulls[k] = np.abs(p[Is] - p[Js]).max()
    out = dict(Omega=float(om), Omega_null=float(nulls.mean()), ratio=float(om / nulls.mean()),
               npairs=int(sel.sum()), dcut=float(dcut),
               null_q05=float(np.quantile(nulls, 0.05) / nulls.mean()),
               null_q50=float(np.quantile(nulls, 0.50) / nulls.mean()),
               null_q95=float(np.quantile(nulls, 0.95) / nulls.mean()),
               p_perm=float((1 + np.sum(nulls <= om)) / (1 + nperm)))
    if return_pair:
        k = np.argmax(np.where(sel, dR, -1))
        out["pair"] = (int(II[k]), int(JJ[k]))
    return out


def omega_subsample_ci(Xd, R, q, rng, nsub=200, frac=0.8, nperm=60):
    n = len(R)
    m = int(round(frac * n))
    vals = np.empty(nsub)
    for b in range(nsub):
        idx = rng.choice(n, m, replace=False)
        vals[b] = omega(Xd[idx], R[idx], q, nperm, rng)["ratio"]
    return float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))


def _feats(Z, deg):
    cols = [np.ones(len(Z))] + [Z[:, a] for a in range(Z.shape[1])]
    if deg == 2:
        p = Z.shape[1]
        cols += [Z[:, a] * Z[:, b] for a in range(p) for b in range(a, p)]
    return np.column_stack(cols)


def surrogate_cv(Xd, R, groups=None, deg=1, k=5, rng=None, ridge=1e-3):
    """Cross-validated surrogate. groups=None: random k-fold; otherwise
    leave-one-group-out (e.g. nominal cell). Returns R2, RMSE, einf."""
    Z, keep = standardize(Xd)
    Phi = _feats(Z[:, keep], deg)
    n = len(R)
    pred = np.zeros(n)
    if groups is None:
        idx = rng.permutation(n)
        folds = np.array_split(idx, k)
    else:
        g = np.asarray(groups)
        folds = [np.flatnonzero(g == u) for u in np.unique(g)]
    for fo in folds:
        tr = np.setdiff1d(np.arange(n), fo)
        A = Phi[tr]
        c = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ R[tr])
        pred[fo] = Phi[fo] @ c
    ss = np.sum((R - R.mean()) ** 2)
    return dict(R2=float(1 - np.sum((R - pred) ** 2) / ss),
                RMSE=float(np.sqrt(np.mean((R - pred) ** 2))),
                einf=float(np.max(np.abs(R - pred)))), pred
