"""Spherical-wave machinery for the multiple-scattering path sum.

Conventions (all checked numerically in tests/test_abinitio_sphere.py):

* Y_lm are the complex Condon-Shortley spherical harmonics.
* h_l = h_l^(1) = j_l + i n_l, the outgoing spherical Hankel function.
* Two-centre expansion of an outgoing wave (|r| < |a|):

      h_l(k|r+a|) Y_L(r+a) = sum_L' G_{L'L}(a) j_l'(kr) Y_L'(r),
      G_{L'L}(a) = 4 pi sum_L'' i^(l'+l''-l) C(L, L', L'') h_l''(ka) Y_L''(a),
      C(L, L', L'') = int Y_L Y*_L' Y*_L'' dOmega.

* Along the z axis G is diagonal in m:
      g^m_{l'l}(x) = sum_l'' A^m_{l'l l''} h_l''(x),
      A^m_{l'l l''} = 4 pi i^(l'+l''-l) C(lm, l'm, l''0) sqrt((2l''+1)/(4 pi)).
* Rotations: D^l_{m'm}(alpha, beta, gamma) = e^{-i m' alpha} d^l_{m'm}(beta)
  e^{-i m gamma}, d^l(beta) = exp(-i beta J_y) (z-y-z, active), so that
  Y_lm(R^-1 r) = sum_m' D^l_{m'm}(R) Y_lm'(r), and
      G(R a_z) = D(R) g(|a|) D(R)^dagger.
* d^l(beta)_{mm'} = i^(m-m') sum_m'' Delta_{m''m} Delta_{m''m'} e^{-i m'' beta},
  Delta = d^l(pi/2): a rotation is two products with a constant matrix and a
  diagonal phase, which is how the CUDA kernel applies it.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


# ---------------------------------------------------------------------------
# Bessel and Hankel functions of complex argument
# ---------------------------------------------------------------------------

def hankel_all(lmax, x):
    """h_l(x), l = 0..lmax, for complex x (any shape) -> (..., lmax+1).
    Upward recurrence is stable for h."""
    x = np.asarray(x, complex)
    out = np.empty(x.shape + (lmax + 1,), complex)
    e = np.exp(1j * x)
    out[..., 0] = -1j * e / x
    if lmax >= 1:
        out[..., 1] = -e / x * (1.0 + 1j / x)
    for l in range(1, lmax):
        out[..., l + 1] = (2 * l + 1) / x * out[..., l] - out[..., l - 1]
    return out


def bessel_j_all(lmax, x):
    """j_l(x), l = 0..lmax, complex x, by Miller's downward recurrence."""
    x = np.asarray(x, complex)
    start = int(max(lmax, np.abs(x).max() if x.size else 0)) + 40
    out = np.zeros(x.shape + (lmax + 1,), complex)
    f1 = np.zeros(x.shape, complex)
    f0 = np.full(x.shape, 1e-30 + 0j)
    for l in range(start, 0, -1):
        fm = (2 * l + 1) / x * f0 - f1
        f1, f0 = f0, fm
        if l - 1 <= lmax:
            out[..., l - 1] = fm
        big = np.abs(f0) > 1e250
        if big.any():
            f0 = np.where(big, f0 * 1e-250, f0)
            f1 = np.where(big, f1 * 1e-250, f1)
            out = np.where(big[..., None], out * 1e-250, out)
    out *= (np.sin(x) / x / out[..., 0])[..., None]
    return out


# ---------------------------------------------------------------------------
# normalised associated Legendre functions and spherical harmonics
# ---------------------------------------------------------------------------

def theta_lm(lmax, x):
    """Theta_lm(x) for m = 0..lmax, l = m..lmax with Y_lm = Theta_lm e^{i m phi}
    (Condon-Shortley phase included). Returns (lmax+1, lmax+1, len(x)) indexed
    [l, m]; stable recursion."""
    x = np.atleast_1d(np.asarray(x, float))
    s = np.sqrt(np.maximum(1.0 - x * x, 0.0))
    T = np.zeros((lmax + 1, lmax + 1, x.size))
    T[0, 0] = 1.0 / math.sqrt(4.0 * math.pi)
    for m in range(1, lmax + 1):
        T[m, m] = -math.sqrt((2 * m + 1) / (2.0 * m)) * s * T[m - 1, m - 1]
    for m in range(0, lmax):
        T[m + 1, m] = math.sqrt(2 * m + 3) * x * T[m, m]
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            a = math.sqrt((4 * l * l - 1) / (l * l - m * m))
            b = math.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
            T[l, m] = a * (x * T[l - 1, m] - b * T[l - 2, m])
    return T


def ylm(lmax, theta, phi):
    """Y_lm(theta, phi) as a dict-free array (len, (lmax+1)^2), index l*l+l+m."""
    theta = np.atleast_1d(np.asarray(theta, float))
    phi = np.atleast_1d(np.asarray(phi, float))
    T = theta_lm(lmax, np.cos(theta))
    out = np.zeros((theta.size, (lmax + 1) ** 2), complex)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            v = T[l, am] * np.exp(1j * am * phi)
            if m < 0:
                v = (-1) ** am * np.conj(v)
            out[:, l * l + l + m] = v
    return out


def lidx(l, m):
    return l * l + l + m


# ---------------------------------------------------------------------------
# z-axis propagator coefficients
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def gaunt_z(lmax):
    """C(lm, l'm, l''0) for 0 <= m <= l, l' <= lmax, l'' = |l-l'|..l+l' (step 2),
    by Gauss-Legendre quadrature (exact). Returns dict
    (m, l', l) -> (array of l'', array of C). Symmetric in m -> -m."""
    n = 2 * lmax + 2
    xg, wg = np.polynomial.legendre.leggauss(n)
    T = theta_lm(2 * lmax, xg)                 # [l, m, x]
    out = {}
    for m in range(lmax + 1):
        for lp in range(m, lmax + 1):
            for l in range(m, lmax + 1):
                lpp = np.arange(abs(l - lp), l + lp + 1, 2)
                prod = T[l, m] * T[lp, m] * wg
                vals = 2.0 * math.pi * (T[lpp, 0] * prod[None, :]).sum(axis=1)
                out[(m, lp, l)] = (lpp, vals)
    return out


@lru_cache(maxsize=8)
def propagator_coeffs(lmax):
    """A^m_{l'l l''} = 4 pi i^(l'+l''-l) C sqrt((2l''+1)/(4pi)) packed for the
    kernel: for every (|m|, l', l) a run of (l'', A) pairs.

    Returns (offsets (lmax+1, lmax+1, lmax+1) int32 into the runs, counts,
    lpp int32 array, A complex128 array)."""
    G = gaunt_z(lmax)
    off = np.full((lmax + 1, lmax + 1, lmax + 1), -1, np.int32)
    cnt = np.zeros((lmax + 1, lmax + 1, lmax + 1), np.int32)
    L, A = [], []
    pos = 0
    for (m, lp, l), (lpp, c) in G.items():
        a = 4.0 * math.pi * (1j ** ((lp + lpp - l) % 4)) * c * np.sqrt((2 * lpp + 1) / (4 * math.pi))
        keep = np.abs(c) > 1e-15
        off[m, lp, l] = pos
        cnt[m, lp, l] = keep.sum()
        L.extend(lpp[keep].tolist())
        A.extend(a[keep].tolist())
        pos += int(keep.sum())
    return off, cnt, np.array(L, np.int32), np.array(A, complex)


def g_z(lmax, x):
    """g^m_{l'l}(x) for all m: array (2lmax+1, lmax+1, lmax+1) indexed
    [m+lmax, l', l] (zero where l or l' < |m|). x scalar complex."""
    h = hankel_all(2 * lmax, np.array([x]))[0]
    off, cnt, L, A = propagator_coeffs(lmax)
    g = np.zeros((2 * lmax + 1, lmax + 1, lmax + 1), complex)
    for m in range(lmax + 1):
        for lp in range(m, lmax + 1):
            for l in range(m, lmax + 1):
                o, c = off[m, lp, l], cnt[m, lp, l]
                v = (A[o:o + c] * h[L[o:o + c]]).sum()
                g[lmax + m, lp, l] = v
                g[lmax - m, lp, l] = v
    return g


# ---------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------

def jy_matrix(l):
    m = np.arange(-l, l + 1)
    Jp = np.zeros((2 * l + 1, 2 * l + 1))
    for i, mm in enumerate(m[:-1]):
        Jp[i + 1, i] = math.sqrt(l * (l + 1) - mm * (mm + 1))   # <m+1|J+|m>
    return (Jp - Jp.T) / 2j                                       # J_y = (J+ - J-)/(2i)


@lru_cache(maxsize=64)
def wigner_d_eig(l):
    """Eigen-decomposition of J_y for d^l(beta) = V e^{-i beta w} V^dagger."""
    w, V = np.linalg.eigh(jy_matrix(l))
    return w, V


def wigner_d(l, beta):
    w, V = wigner_d_eig(l)
    return ((V * np.exp(-1j * beta * w)) @ V.conj().T).real


@lru_cache(maxsize=8)
def delta_matrices(lmax):
    """Delta^l = d^l(pi/2) for l = 0..lmax, packed: offsets and a flat array
    with Delta^l[m''+l, m+l] at off[l] + (m''+l)*(2l+1) + (m+l)."""
    off = np.zeros(lmax + 2, np.int64)
    blocks = []
    for l in range(lmax + 1):
        blocks.append(wigner_d(l, math.pi / 2).ravel())
        off[l + 1] = off[l] + (2 * l + 1) ** 2
    return off, np.concatenate(blocks)


def rotation_D(l, alpha, beta, gamma):
    m = np.arange(-l, l + 1)
    return (np.exp(-1j * m * alpha)[:, None] * wigner_d(l, beta)
            * np.exp(-1j * m * gamma)[None, :])


def euler_zyz(Q):
    """(alpha, beta, gamma) with Q = Rz(alpha) Ry(beta) Rz(gamma)."""
    Q = np.asarray(Q, float)
    beta = math.acos(max(-1.0, min(1.0, Q[2, 2])))
    if abs(math.sin(beta)) > 1e-10:
        alpha = math.atan2(Q[1, 2], Q[0, 2])
        gamma = math.atan2(Q[2, 1], -Q[2, 0])
    else:
        # gimbal lock: only alpha + gamma (beta=0) or alpha - gamma (beta=pi)
        alpha = math.atan2(Q[1, 0], Q[0, 0]) if Q[2, 2] > 0 else math.atan2(-Q[1, 0], -Q[0, 0])
        gamma = 0.0
    return alpha, beta, gamma


def frame_of(v):
    """Rotation R (3x3) with R z = v/|v|: R = Rz(phi) Ry(theta)."""
    v = np.asarray(v, float)
    r = np.linalg.norm(v)
    th = math.acos(max(-1.0, min(1.0, v[2] / r)))
    ph = math.atan2(v[1], v[0])
    cz, sz = math.cos(ph), math.sin(ph)
    cy, sy = math.cos(th), math.sin(th)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
    Ry = np.array([[cy, 0, sy], [0, 1.0, 0], [-sy, 0, cy]])
    return Rz @ Ry
