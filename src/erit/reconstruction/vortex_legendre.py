"""
Python port of Codes/VortexLegendre (MATLAB, R. Castaneda, EAFIT).

Faithful translation of vortexLegendre.m + functions_vortexLegendre.m:
  1. spatial filter of the +1 order (circular mask)
  2. sub-pixel carrier frequency via optical-vortex (Hilbert + residue theorem)
  3. tilt compensation with a digital reference wave
  4. PCA (rank-1 SVD) + Legendre fit of residual aberrations, removed with orders 2..6

MATLAB indices are 1-based; comments mark where the offset matters.

Differences vs MATLAB: skimage unwrap_phase replaces unwrap_phase.m; linear
interpolation/extrapolation via scipy replaces griddedInterpolant (same defaults).
Like main.m (NoPistonCompensation=true) the constant phase offset is NOT removed here;
use erit.phase.quantitative_phase for that.
"""
import numpy as np
from scipy.ndimage import median_filter
from scipy.interpolate import RegularGridInterpolator
from skimage.restoration import unwrap_phase


def _fts(x):   # fftshift(fft2(fftshift(x)))
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x)))


def _ifts(x):  # fftshift(ifft2(fftshift(x)))
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x)))


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def spatial_filter(holo, factor=3):
    N, M = holo.shape
    ft = _fts(holo)
    ft[:5, :5] = 0
    mask = np.ones((N, M))
    mask[N // 2 - 21:N // 2 + 20, M // 2 - 21:M // 2 + 20] = 0     # MATLAB N/2-20:N/2+20 (1-based)
    ft_I = ft * mask
    ft_I[0, 0] = 0
    roi = np.abs(ft_I[:, :M // 2])
    fy0, fx0 = np.unravel_index(np.argmax(roi), roi.shape)
    fy_max, fx_max = fy0 + 1, fx0 + 1                                # keep MATLAB 1-based convention
    dist = np.sqrt((fx_max - N / 2) ** 2 + (fy_max - M / 2) ** 2)
    r, p = np.mgrid[1:N + 1, 1:M + 1]
    cir = (np.sqrt((r - fy_max) ** 2 + (p - fx_max) ** 2) <= dist / factor).astype(float)
    holo_filtered = _ifts(ft * cir)
    return holo_filtered, fx_max, fy_max, cir


def hilbert2d(c):
    NR, NC = c.shape
    u, v = np.meshgrid(np.arange(1, NC + 1), np.arange(1, NR + 1))
    u0, v0 = NC // 2 + 1, NR // 2 + 1
    u, v = u - u0, v - v0
    with np.errstate(invalid="ignore", divide="ignore"):
        H = (u + 1j * v) / np.abs(u + 1j * v)
    H[v0 - 1, u0 - 1] = 0
    return np.conj(np.fft.ifft2(np.fft.fft2(c) * np.fft.ifftshift(H)))


def vortex_compensation(field, fx_max, fy_max, crop=5, over=55):
    # MATLAB: field(fy-crop : fy+crop-1, fx-crop : fx+crop-1), 1-based
    sd = field[fy_max - crop - 1:fy_max + crop - 1, fx_max - crop - 1:fx_max + crop - 1]
    sdc = hilbert2d(sd)
    sz = sdc.shape
    g = (np.arange(1, sz[0] + 1), np.arange(1, sz[1] + 1))
    q0 = np.arange(0, sz[0] - 1 / over + 1e-12, 1 / over)
    q1 = np.arange(0, sz[1] - 1 / over + 1e-12, 1 / over)
    Q0, Q1 = np.meshgrid(q0, q1, indexing="ij")
    pts = np.stack([Q0.ravel(), Q1.ravel()], -1)
    fr = RegularGridInterpolator(g, sdc.real, bounds_error=False, fill_value=None)(pts)
    fi = RegularGridInterpolator(g, sdc.imag, bounds_error=False, fill_value=None)(pts)
    psi = np.angle(fr + 1j * fi).reshape(Q0.shape)

    n1, m1 = psi.shape
    Ms = [np.zeros_like(psi) for _ in range(8)]
    Y1, Y2, Y3 = slice(0, n1 - 2), slice(1, n1 - 1), slice(2, n1)
    X1, X2, X3 = slice(0, m1 - 2), slice(1, m1 - 1), slice(2, m1)
    for Mk, (ys, xs) in zip(Ms, [(Y1, X1), (Y1, X2), (Y1, X3), (Y2, X3),
                                 (Y3, X3), (Y3, X2), (Y3, X1), (Y2, X1)]):
        Mk[Y2, X2] = psi[ys, xs]
    Ml = sum(_wrap(Ms[(i + 1) % 8] - Ms[i]) for i in range(8))
    Ml = np.fft.fftshift(Ml / (2 * np.pi))
    Ml[69:, 69:] = 0
    Ml = np.fft.ifftshift(Ml)
    yv, xv = np.unravel_index(np.argmin(Ml), Ml.shape)
    yv, xv = yv + 1, xv + 1
    return (xv / over + fx_max - crop - 2, yv / over + fy_max - crop - 2)


def reference_wave(M, N, m, n, lam, dxy, fx, fy, k, fx_0, fy_0):
    tx = np.arcsin((fx_0 - fx) * lam / (M * dxy))
    ty = np.arcsin((fy_0 - fy) * lam / (N * dxy))
    return np.exp(1j * k * (np.sin(tx) * n * dxy + np.sin(ty) * m * dxy))


def _legendre(order, x, y):
    P = [np.ones_like(x), x, y, (3 * x**2 - 1) / 2, x * y, (3 * y**2 - 1) / 2,
         x * (5 * x**2 - 3) / 2, y * (3 * x**2 - 1) / 2, x * (3 * y**2 - 1) / 2,
         y * (5 * y**2 - 3) / 2, (35 * x**4 - 30 * x**2 + 3) / 8,
         x * y * (5 * x**2 - 3) / 2, (3 * y**2 - 1) * (3 * x**2 - 1) / 4,
         x * y * (5 * y**2 - 3) / 2, (35 * y**4 - 30 * y**2 + 3) / 8]
    return np.stack([P[o - 1] for o in order], -1)


def _normalized_basis(order, g):
    t = np.arange(-1, 1 - 2 / g + 1e-12, 2 / g)
    X, Y = np.meshgrid(t, t)
    dA = (2 / g) ** 2
    P = _legendre(order, X, Y)
    L = P.reshape(-1, len(order))
    L = L / np.sqrt(np.diag(L.T @ L * dA))
    return L, (L**2).sum(0) * dA, dA, P.shape[:2]


def legendre_compensation(field_c, limit, no_piston=True, use_pca=True):
    F = _fts(field_c)  # MATLAB: fftshift(fft2(ifftshift(.))) — identical for even sizes
    A, B = F.shape
    ca, cb = round(A / 2), round(B / 2)
    F = F[ca - limit:ca + limit, cb - limit:cb + limit]
    square = _ifts(F)
    if use_pca:
        U, S, Vh = np.linalg.svd(square)
        dom = unwrap_phase(np.angle(S[0] * np.outer(U[:, 0], Vh[0])))
    else:
        dom = unwrap_phase(np.angle(square))
    order = list(range(1, 11))
    L, nc, dA, shp = _normalized_basis(order, dom.shape[0])
    coef = (L * dom.reshape(-1, 1)).sum(0) * dA
    if no_piston:
        W = (L[:, 1:] * (coef[1:] / np.sqrt(nc[1:]))).sum(1).reshape(shp)
        comp = np.exp(1j * np.angle(square)) / np.exp(1j * W)
    else:
        best, bv = 0, np.inf
        for pv in np.arange(-np.pi, np.pi + 1e-9, np.pi / 6):
            coef[0] = pv
            W = (L * (coef / np.sqrt(nc))).sum(1).reshape(shp)
            v = np.var(np.angle(np.exp(1j * np.angle(square)) / np.exp(1j * W)))
            if v < bv:
                best, bv = pv, v
        coef[0] = best
        W = (L * (coef / np.sqrt(nc))).sum(1).reshape(shp)
        comp = np.exp(1j * np.angle(square)) / np.exp(1j * W)
    return comp, coef


def vortex_legendre(hologram, wavelength, dxy, med_filt=5, no_piston=True, use_pca=True, limit=128):
    holo = np.asarray(hologram, dtype=float)
    if holo.ndim == 3:
        holo = holo[..., 0]
    if holo.shape[1] != holo.shape[0]:
        lim = round((holo.shape[1] - holo.shape[0]) / 2)
        holo = holo[:, lim - 1:holo.shape[1] - lim - 1]
    N, M = holo.shape
    n, m = np.meshgrid(np.arange(-M / 2, M / 2), np.arange(-N / 2, N / 2))
    k = 2 * np.pi / wavelength

    holo_f, fx, fy, _ = spatial_filter(holo, 3)
    ft = np.log(np.abs(_fts(holo)) ** 2)
    field = median_filter(ft, size=med_filt, mode="reflect")
    px, py = vortex_compensation(field, fx, fy)
    ref = reference_wave(M, N, m, n, wavelength, dxy, px, py, k, M / 2, N / 2)
    field_c = ref * holo_f

    _, coef = legendre_compensation(field_c, limit, no_piston, use_pca)

    order = list(range(2, 7))
    L, nc, _, shp = _normalized_basis(order, field_c.shape[0])
    W = (L * (coef[1:len(order) + 1] / np.sqrt(nc))).sum(1).reshape(shp)
    return np.abs(field_c) * np.exp(1j * np.angle(field_c)) / np.exp(1j * W)

