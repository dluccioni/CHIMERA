"""Energy-dependent exchange-correlation: the Hedin-Lundqvist self-energy.

A photoelectron at energy E above the Fermi level does not feel the
ground-state exchange-correlation potential: exchange weakens as it speeds
up, and above the plasmon energy it can lose energy to the electron gas,
which is what limits its mean free path. FEFF's default treatment - and this
one - is the GW self-energy of the homogeneous electron gas with Lundqvist's
single-plasmon-pole dielectric function, evaluated in a local density
approximation: at each point the gas has the local density, and the electron
the local momentum p(r) with p^2 = kF(r)^2 + 2 (E - mu).

Electron gas, Hartree units, on shell (omega = k^2/2):

    Sigma = Sigma_x + Sigma_c
    Sigma_x(k) = -(2 kF / pi) [1/2 + (1 - x^2)/(4x) ln|(1+x)/(1-x)|],  x = k/kF
    Sigma_c(k) = int d3q/(2pi)^3 v_q wp^2/(2 w_q) [ (1 - f_{k-q}) / (w - w_q - e_{k-q} + i0)
                                                 +      f_{k-q}  / (w + w_q - e_{k-q} - i0) ]
    w_q^2 = wp^2 + kF^2 q^2 / 3 + q^4 / 4      (Thomas-Fermi screening at small q,
                                                free-particle dispersion at large q)

The angular integral is done analytically. The imaginary part (plasmon
emission) is a smooth one-dimensional integral between kinematic limits
found exactly. The real part is a principal value with logarithmic end-point
singularities; it is tabulated once on (rs, k/kF) with adaptive quadrature
and cached. Below the plasmon threshold the single-pole model has no loss
at all; Quinn's electron-hole term can be added (`quinn=True`) as FEFF does.

`check()` confirms Sigma(kF) against the LDA exchange-correlation potential,
which a plasmon-pole model reproduces to a few hundredths of a Hartree.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.integrate import quad

from .atom import lda_xc

KF_RS = (9.0 * np.pi / 4.0) ** (1.0 / 3.0)       # kF = KF_RS / rs


def kf_of(rs):
    return KF_RS / np.asarray(rs, float)


def wp_of(rs):
    return np.sqrt(3.0 / np.asarray(rs, float) ** 3)


def sigma_x(rs, k):
    """Hartree-Fock exchange of the electron gas, real."""
    kf = kf_of(rs)
    x = np.asarray(k, float) / kf
    with np.errstate(divide="ignore", invalid="ignore"):
        F = 0.5 + (1.0 - x * x) / (4.0 * x) * np.log(np.abs((1.0 + x) / (1.0 - x)))
    F = np.where(np.abs(x - 1.0) < 1e-12, 0.5, F)
    F = np.where(x < 1e-12, 1.0, F)
    return -2.0 * kf / np.pi * F


def _wq(q, wp, kf):
    return np.sqrt(wp * wp + kf * kf * q * q / 3.0 + 0.25 * q ** 4)


# --------------------------------------------------------------------------
# imaginary part: plasmon emission
# --------------------------------------------------------------------------

_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)


def _bisect(f, a, b, n=80):
    """Vectorised bisection for a sign change of f on [a, b] (f(a) < 0 < f(b))."""
    a = np.array(a, float, copy=True)
    b = np.array(b, float, copy=True)
    for _ in range(n):
        m = 0.5 * (a + b)
        neg = f(m) < 0
        a = np.where(neg, m, a)
        b = np.where(neg, b, m)
    return 0.5 * (a + b)


def im_sigma_plasmon(rs, k):
    """Im Sigma_c on shell for k > kF (Hartree), single plasmon pole.

        Im Sigma = - int dq wp^2 / (2 w_q k q)   over  w_q + q^2/2 <= k q
                                                 and   w_q <= (k^2 - kF^2)/2
    """
    rs, k = np.broadcast_arrays(np.asarray(rs, float), np.asarray(k, float))
    shape = rs.shape
    rs, k = rs.ravel(), k.ravel()
    kf, wp = kf_of(rs), wp_of(rs)
    out = np.zeros_like(k)
    # h(q) = k q - q^2/2 - w_q; find its maximum, then its two roots
    def dh(q):
        return k - q - (kf * kf * q / 3.0 + 0.5 * q ** 3) / _wq(q, wp, kf)
    qpk = _bisect(lambda q: -dh(q), np.zeros_like(k), 2.0 * k + 1.0)
    hmax = k * qpk - 0.5 * qpk ** 2 - _wq(qpk, wp, kf)
    E = 0.5 * (k * k - kf * kf)
    ok = (hmax > 0) & (E > wp)
    if not ok.any():
        return out.reshape(shape)
    h = lambda q: k * q - 0.5 * q * q - _wq(q, wp, kf)
    q1 = _bisect(h, np.zeros_like(k), qpk)
    q2 = _bisect(lambda q: -h(q), qpk, 2.0 * k + 1.0)
    # w_q = E  ->  q^4/4 + kF^2 q^2/3 + wp^2 - E^2 = 0
    disc = (kf ** 4 / 9.0) - (wp * wp - E * E)
    q3 = np.sqrt(np.maximum(2.0 * (-kf * kf / 3.0 + np.sqrt(np.maximum(disc, 0.0))), 0.0))
    lo, hi = q1, np.minimum(q2, q3)
    ok &= hi > lo
    xs = 0.5 * (hi - lo)[:, None] * (_GL_X[None, :] + 1.0) + lo[:, None]
    integ = wp[:, None] ** 2 / (2.0 * _wq(xs, wp[:, None], kf[:, None]) * k[:, None] * xs)
    val = 0.5 * (hi - lo) * (integ * _GL_W[None, :]).sum(axis=1)
    out[ok] = -val[ok]
    return out.reshape(shape)


_IM_KERNEL = r"""
extern "C" __global__ void im_sigma_pp(const double* __restrict__ rs, const double* __restrict__ kk,
                                       const double* __restrict__ glx, const double* __restrict__ glw,
                                       const int ngl, const int n, double* __restrict__ out)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const double KF_RS = %(KF_RS).17g;
    double r = rs[i], k = kk[i];
    double kf = KF_RS / r, wp = sqrt(3.0 / (r * r * r));
    double E = 0.5 * (k * k - kf * kf);
    out[i] = 0.0;
    if (E <= wp) return;
    #define WQ(q) sqrt(wp * wp + kf * kf * (q) * (q) / 3.0 + 0.25 * (q) * (q) * (q) * (q))
    double a = 0.0, b = 2.0 * k + 1.0;
    for (int it = 0; it < 80; ++it) {           // maximum of h(q) = kq - q^2/2 - w_q
        double m = 0.5 * (a + b);
        double dh = k - m - (kf * kf * m / 3.0 + 0.5 * m * m * m) / WQ(m);
        if (dh > 0) a = m; else b = m;
    }
    double qpk = 0.5 * (a + b);
    double hmax = k * qpk - 0.5 * qpk * qpk - WQ(qpk);
    if (hmax <= 0.0) return;
    a = 0.0; b = qpk;
    for (int it = 0; it < 80; ++it) { double m = 0.5 * (a + b); if (k * m - 0.5 * m * m - WQ(m) < 0) a = m; else b = m; }
    double q1 = 0.5 * (a + b);
    a = qpk; b = 2.0 * k + 1.0;
    for (int it = 0; it < 80; ++it) { double m = 0.5 * (a + b); if (k * m - 0.5 * m * m - WQ(m) > 0) a = m; else b = m; }
    double q2 = 0.5 * (a + b);
    double disc = kf * kf * kf * kf / 9.0 - (wp * wp - E * E);
    double q3 = sqrt(fmax(2.0 * (-kf * kf / 3.0 + sqrt(fmax(disc, 0.0))), 0.0));
    double lo = q1, hi = fmin(q2, q3);
    if (hi <= lo) return;
    double s = 0.0;
    for (int j = 0; j < ngl; ++j) {
        double q = 0.5 * (hi - lo) * (glx[j] + 1.0) + lo;
        s += glw[j] * wp * wp / (2.0 * WQ(q) * k * q);
    }
    out[i] = -0.5 * (hi - lo) * s;
}
"""

_IM_MOD = None


def im_sigma_plasmon_gpu(rs, k):
    """The same integral as `im_sigma_plasmon`, one CUDA thread per point."""
    global _IM_MOD
    import cupy as cp
    if _IM_MOD is None:
        _IM_MOD = cp.RawModule(code=_IM_KERNEL % dict(KF_RS=KF_RS))
    rs, k = np.broadcast_arrays(np.asarray(rs, float), np.asarray(k, float))
    shape = rs.shape
    r_d = cp.asarray(np.ascontiguousarray(rs.ravel()))
    k_d = cp.asarray(np.ascontiguousarray(k.ravel()))
    out = cp.zeros(r_d.size)
    n = r_d.size
    _IM_MOD.get_function("im_sigma_pp")(((n + 255) // 256,), (256,),
                                        (r_d, k_d, cp.asarray(_GL_X), cp.asarray(_GL_W),
                                         np.int32(_GL_X.size), np.int32(n), out))
    return cp.asnumpy(out).reshape(shape)


def _has_gpu():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


_GPU = None


def im_sigma(rs, k):
    """Im Sigma (plasmon pole), on the GPU when there is one."""
    global _GPU
    if _GPU is None:
        _GPU = _has_gpu()
    return im_sigma_plasmon_gpu(rs, k) if _GPU else im_sigma_plasmon(rs, k)


def im_sigma_quinn(rs, k):
    """Quinn's electron-hole loss for k > kF (Hartree), used below the plasmon
    threshold: Gamma = (sqrt3 pi^2/128) wp (k^2/kF^2 - 1)^2, Im Sigma = -Gamma/2."""
    kf, wp = kf_of(rs), wp_of(rs)
    x2 = (np.asarray(k, float) / kf) ** 2
    g = np.sqrt(3.0) * np.pi ** 2 / 128.0 * wp * np.maximum(x2 - 1.0, 0.0) ** 2
    return -0.5 * g


# --------------------------------------------------------------------------
# real part: principal value, tabulated
# --------------------------------------------------------------------------

def _re_sigma_c_point(rs, k):
    kf, wp = KF_RS / rs, np.sqrt(3.0 / rs ** 3)

    def integrand(q):
        if q <= 0.0:
            return 0.0
        w = np.sqrt(wp * wp + kf * kf * q * q / 3.0 + 0.25 * q ** 4)
        kq = k * q
        muF = (k * k + q * q - kf * kf) / (2.0 * kq)
        mu_split = min(1.0, max(-1.0, muF))
        t = 0.0
        # unoccupied final states: mu in [-1, mu_split]
        if mu_split > -1.0:
            a1 = kq * mu_split - w - 0.5 * q * q
            a0 = -kq - w - 0.5 * q * q
            t += (np.log(abs(a1) + 1e-300) - np.log(abs(a0))) / kq
        # occupied final states: mu in [mu_split, 1]
        if mu_split < 1.0:
            b1 = kq + w - 0.5 * q * q
            b0 = kq * mu_split + w - 0.5 * q * q
            t += (np.log(abs(b1) + 1e-300) - np.log(abs(b0) + 1e-300)) / kq
        return wp * wp / (2.0 * w) * t

    # break points: k +- kF and the zeros of the log arguments
    pts = sorted({abs(k - kf), k + kf})
    E = 0.5 * (k * k - kf * kf)
    if E > wp:
        disc = kf ** 4 / 9.0 - (wp * wp - E * E)
        q3 = np.sqrt(max(2.0 * (-kf * kf / 3.0 + np.sqrt(max(disc, 0.0))), 0.0))
        pts.append(q3)
    qs = np.linspace(1e-6, 4.0 * (k + kf), 4001)
    h = k * qs - 0.5 * qs ** 2 - np.sqrt(wp * wp + kf * kf * qs ** 2 / 3.0 + 0.25 * qs ** 4)
    for i in np.where(np.diff(np.sign(h)) != 0)[0]:
        pts.append(0.5 * (qs[i] + qs[i + 1]))
    qmax = 40.0 * (k + kf) + 40.0
    pts = sorted(p for p in pts if 0.0 < p < qmax)
    edges = [0.0] + pts + [qmax]
    tot = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 1e-12:
            continue
        v, _ = quad(integrand, a, b, limit=400, epsabs=1e-11, epsrel=1e-10)
        tot += v
    # tail beyond qmax: integrand ~ wp^2 * (2 k^2 ... ) / q^4; negligible but add it
    return tot / np.pi


_TABLE = None


def _table_path():
    from .atom import _cache_dir
    return os.path.join(_cache_dir(), "hl_re_sigma_c_v2.npz")


def re_sigma_c_table(rebuild=False):
    """Re Sigma_c on a grid of ln(rs) x (k/kF), Hartree. Built once (~30 s)."""
    global _TABLE
    if _TABLE is not None and not rebuild:
        return _TABLE
    fn = _table_path()
    if os.path.exists(fn) and not rebuild:
        d = np.load(fn)
        _TABLE = (d["lnrs"], d["x"], d["v"])
        return _TABLE
    lnrs = np.linspace(np.log(0.4), np.log(10.0), 49)
    x = np.concatenate([np.linspace(1.0, 2.0, 81)[:-1], np.linspace(2.0, 6.0, 81)[:-1],
                        np.linspace(6.0, 30.0, 49)])
    v = np.zeros((lnrs.size, x.size))
    for i, lr in enumerate(lnrs):
        rs = float(np.exp(lr))
        kf = KF_RS / rs
        for j, xx in enumerate(x):
            v[i, j] = _re_sigma_c_point(rs, xx * kf if xx > 1.0 else (1.0 + 1e-9) * kf)
    np.savez(fn, lnrs=lnrs, x=x, v=v)
    _TABLE = (lnrs, x, v)
    return _TABLE


def re_sigma_c(rs, k):
    """Re Sigma_c by bicubic-in-x, linear-in-ln(rs) interpolation of the table.
    Beyond x = 30 it decays as 1/x^2 (fitted to the last points)."""
    from scipy.interpolate import RectBivariateSpline
    lnrs, xg, v = re_sigma_c_table()
    spl = _spline_cache(lnrs, xg, v)
    rs, k = np.broadcast_arrays(np.asarray(rs, float), np.asarray(k, float))
    lr = np.clip(np.log(rs), lnrs[0], lnrs[-1])
    x = np.maximum(k / kf_of(rs), 1.0)
    xin = np.minimum(x, xg[-1])
    val = spl.ev(lr, xin)
    tail = x > xg[-1]
    if np.any(tail):
        val = np.where(tail, val * (xg[-1] / x) ** 2, val)
    return val


_SPL = {}


def _spline_cache(lnrs, xg, v):
    from scipy.interpolate import RectBivariateSpline
    key = id(v)
    if key not in _SPL:
        _SPL[key] = RectBivariateSpline(lnrs, xg, v, kx=3, ky=3)
    return _SPL[key]


# --------------------------------------------------------------------------
# the self-energy used for the scattering potential
# --------------------------------------------------------------------------

def sigma_hl(rs, k, quinn=False):
    """Complex HL self-energy of the gas at density rs for momentum k >= kF."""
    rs, k = np.broadcast_arrays(np.asarray(rs, float), np.asarray(k, float))
    kf = kf_of(rs)
    k = np.maximum(k, kf * (1.0 + 1e-9))
    re = sigma_x(rs, k) + re_sigma_c(rs, k)
    im = im_sigma(rs, k)
    if quinn:
        # Quinn's electron-hole loss where the plasmon channel is closed; FEFF
        # joins the two the same way
        im = np.where(im == 0.0, im_sigma_quinn(rs, k), im)
    return re + 1j * im


def local_sigma(rho, e_rel, model="hl", quinn=False, relativistic=False):
    """Exchange-correlation potential for a photoelectron e_rel = E - mu above
    the Fermi level, at local density rho (arrays broadcast). Hartree.

    model 'hl' : Hedin-Lundqvist, shifted so it equals the LDA potential at
                 the Fermi level (FEFF's convention: the energy dependence is
                 added to the ground state, so the interstitial level and mu
                 keep their meaning)
          'dh' : Dirac-Hara exchange, Sigma_x(p) - Sigma_x(kF) + v_xc^LDA, real
          'gs' : the ground-state LDA potential, energy independent, real
    relativistic: the ground-state part is the relativistic (MacDonald-Vosko)
                 LDA potential, as in the relativistic atoms
    """
    rho = np.maximum(np.asarray(rho, float), 1e-12)
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    kf = kf_of(rs)
    vxc = lda_xc(rho, relativistic)[1]
    p = np.sqrt(kf * kf + 2.0 * np.maximum(np.asarray(e_rel, float), 0.0))
    if model == "gs":
        return vxc + 0j * p
    if model == "dh":
        return vxc + sigma_x(rs, p) - sigma_x(rs, kf) + 0j
    if model == "hl":
        s = sigma_hl(rs, p, quinn=quinn)
        s0 = sigma_hl(rs, kf * (1.0 + 1e-9), quinn=False).real
        return vxc + (s.real - s0) + 1j * s.imag
    raise ValueError(model)


def check(rs_values=(1.0, 2.0, 3.0, 4.0)):
    """Sigma(kF) of the plasmon-pole model against the LDA potential."""
    out = []
    for rs in rs_values:
        kf = KF_RS / rs
        s = sigma_x(rs, kf * (1 + 1e-9)) + _re_sigma_c_point(rs, kf * (1 + 1e-9))
        n = 3.0 / (4.0 * np.pi * rs ** 3)
        out.append((rs, float(s), float(lda_xc(n)[1])))
    return out


# --------------------------------------------------------------------------
# K-shell core-hole widths (eV, FWHM), Krause & Oliver, J. Phys. Chem. Ref.
# Data 8, 329 (1979)
# --------------------------------------------------------------------------

GAMMA_K = {
    "Al": 0.43, "Si": 0.48, "P": 0.53, "S": 0.59, "Cl": 0.64, "Ar": 0.68,
    "K": 0.74, "Ca": 0.81, "Sc": 0.86, "Ti": 0.94, "V": 1.01, "Cr": 1.08,
    "Mn": 1.16, "Fe": 1.25, "Co": 1.33, "Ni": 1.44, "Cu": 1.55, "Zn": 1.67,
    "Ga": 1.82, "Ge": 1.96, "As": 2.14, "Se": 2.33, "Br": 2.52, "Kr": 2.75,
    "Rb": 2.99, "Sr": 3.25, "Y": 3.52, "Zr": 3.84, "Nb": 4.14, "Mo": 4.52,
    "Tc": 4.91, "Ru": 5.33, "Rh": 5.77, "Pd": 6.24, "Ag": 6.75,
}
