"""Complex scattering phase shifts of the muffin-tin potentials.

For every potential type, energy and angular momentum the radial equation

    y'' = [(l+1/2)^2 + 2 r^2 (V_C(r) + Sigma(r, E) - E - i Gamma/2)] y,
    u = r^(1/2) y,  x = ln r,

is integrated outward with Numerov's method from the nucleus to the
muffin-tin radius, where the logarithmic derivative is matched to the free
spherical waves of the interstitial at its complex momentum p:

    e^{2 i delta_l} = - (p h-'(pr) - L h-(pr)) / (p h+'(pr) - L h+(pr)),
    L = R'/R at r_mt,  h+- = j_l +- i n_l.

Sigma is the energy-dependent self-energy (`selfenergy.local_sigma`) at the
local density, Gamma the core-hole width (FEFF adds it the same way, as an
imaginary part of the energy everywhere).

Relativistic potentials (`potential.build(relativistic=True)`, read from the
potential's meta) are solved with the scalar-relativistic Koelling-Harmon
equation instead (`atom.py` has the form; the bracket above gains the
mass-velocity and Darwin terms with M = 1 + (E - V - Sigma)/2c^2), and the
interstitial momentum follows the relativistic dispersion
p^2 = 2T (1 + T/2c^2), T = E - V_int - Sigma_int (FEFF: ck = sqrt(2T + (alpha T)^2)).
At the sphere boundary P and Q = r g'/2M are continuous, so the log
derivative carries the factor M_out/M_in. Compared with the
non-relativistic solution the phase shifts of Cr-Ni rise by 0.02-0.04 rad
(l = 0, 1) almost independently of energy - the electron is fast near the
nucleus at every EXAFS energy - and the path phases by 0.04-0.08 rad.

`phase_shifts` runs one CUDA thread per (type, energy, l) in double
precision - a few milliseconds for a full table, so there is nothing to gain
from single precision here and the Bessel recurrences at high l want the
range. `phase_shifts_numpy` is the same calculation on the CPU, vectorised
over (energy, l), kept as the reference the kernel is tested against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .atom import C_LIGHT, sr_coefficients, sr_start_exponent
from .potential import BOHR, HARTREE, ClusterPotential
from .selfenergy import GAMMA_K, local_sigma

H_STEP = 0.005          # log-grid step for the phase-shift integration
R_START = 1e-6          # bohr
N_EXTRA = 2             # grid points past r_mt (for the 5-point derivative)
LMAX_DEFAULT = 45


@dataclass
class PhaseShifts:
    k: np.ndarray            # (nE,)  photoelectron wavenumber above mu (A^-1)
    energy: np.ndarray       # (nE,)  E - mu (Hartree)
    p: np.ndarray            # (nE,)  complex interstitial momentum (bohr^-1)
    S: np.ndarray            # (ntype, nE, lmax+1)  e^{2 i delta_l}, complex
    labels: list
    gamma_ch: float          # core-hole width (Hartree)
    r_mt: np.ndarray         # (ntype,) bohr
    meta: dict = field(default_factory=dict)

    @property
    def t(self):
        """Scattering amplitudes t_l = (e^{2 i delta} - 1)/2."""
        return 0.5 * (self.S - 1.0)

    @property
    def delta(self):
        """Complex phase shifts, real part unwrapped along energy."""
        d = np.log(self.S) / 2j
        re = np.unwrap(2.0 * d.real, axis=1) / 2.0
        return re + 1j * d.imag

    def lmax_needed(self, tol=1e-8):
        """Per energy, the highest l whose |t_l| exceeds tol for any type."""
        a = np.abs(self.t).max(axis=0)          # (nE, nl)
        out = np.zeros(a.shape[0], int)
        for i in range(a.shape[0]):
            big = np.where(a[i] > tol)[0]
            out[i] = big.max() if big.size else 0
        return out


def default_k_grid(kmax=20.0):
    """FEFF-like: fine at low k, 0.05 A^-1 everywhere (where tables are read)."""
    return np.round(np.arange(0.0, kmax + 1e-9, 0.05), 6)


def _type_grid(r_mt):
    """Log grid ending exactly at r_mt, with N_EXTRA points beyond."""
    n_in = int(np.ceil(np.log(r_mt / R_START) / H_STEP)) + 1
    x = np.log(r_mt) + H_STEP * (np.arange(n_in + N_EXTRA) - (n_in - 1))
    return np.exp(x), n_in - 1


def interstitial_momentum(T, relativistic=False):
    """Complex momentum of kinetic energy T (Ha, complex): p^2 = 2T, or
    2T (1 + T/2c^2) relativistically; Im p >= 0."""
    T = np.asarray(T, complex)
    p2 = 2.0 * T * (1.0 + T / (2.0 * C_LIGHT ** 2)) if relativistic else 2.0 * T
    p = np.sqrt(p2)
    return np.where(p.imag < 0, -p, p)


def _relativistic(cp, relativistic):
    return bool(cp.meta.get("relativistic", False)) if relativistic is None else bool(relativistic)


def _prepare(cp: ClusterPotential, k_A, gamma_ev, model, quinn, relativistic=False, n_sigma_r=240,
             mpse=None):
    """Each type's grid and the l-independent part U of the Numerov
    coefficient, G = (l+1/2)^2 + U(E, r); the interstitial momentum; and the
    boundary factors L_out = (L_in + ladd) * lmul (0 and 1 non-relativistically).
    mpse: `losses.Poles` of the material; its many-pole correction to the HL
    self-energy is added at every point (model must be 'hl')."""
    k_au = np.asarray(k_A, float) * BOHR
    e_rel = 0.5 * k_au ** 2
    gam = gamma_ev / HARTREE
    E = cp.mu + e_rel
    Ep = E + 0.5j * gam
    dsig = np.zeros(e_rel.size, complex)
    if mpse is not None:
        if model != "hl":
            raise ValueError("the many-pole self-energy corrects the HL one: model must be 'hl'")
        from .losses import mpse_correction
        dsig = mpse_correction(mpse, cp.rho_int, e_rel)
    s_int = local_sigma(np.full(1, cp.rho_int), e_rel, model, quinn, relativistic) + dsig
    T = Ep - cp.v_int_coul - s_int                       # interstitial kinetic energy
    p = interstitial_momentum(T, relativistic)
    grids, Us, imt, ladd, lmul = [], [], [], [], []
    for t in cp.types:
        r, m = _type_grid(t.r_mt)
        x_src = np.log(t.r)
        # r V_C is smooth through the nucleus (-> -Z); interpolate that
        rv = np.interp(np.log(r), x_src, t.r * t.v_coul)
        lnrho = np.interp(np.log(r), x_src, np.log(np.maximum(t.rho, 1e-300)))
        # self-energy on a coarse radial grid, then interpolated: it varies
        # through rho(r) only
        rc = np.exp(np.linspace(np.log(r[0]), np.log(r[-1]), n_sigma_r))
        rho_c = np.exp(np.interp(np.log(rc), np.log(r), lnrho))
        s_c = local_sigma(rho_c[None, :], e_rel[:, None], model, quinn, relativistic) + dsig[:, None]
        s = np.empty((k_au.size, r.size), complex)
        for i in range(k_au.size):
            s[i] = (np.interp(np.log(r), np.log(rc), s_c[i].real)
                    + 1j * np.interp(np.log(r), np.log(rc), s_c[i].imag))
        Vt = (rv / r)[None, :] + s                                         # (nE, nr)
        if relativistic:
            w = r[None, :] * Vt
            wx = np.gradient(w, H_STEP, axis=1, edge_order=2)
            wxx = np.gradient(wx, H_STEP, axis=1, edge_order=2)
            U, _, M = sr_coefficients(r[None, :], Vt, Ep[:, None], w, wx, wxx, 0.0)
            dM = -((wx[:, m] - w[:, m]) / r[m] ** 2) / (2.0 * C_LIGHT ** 2)   # M' just inside r_mt
            ladd.append(dM / (2.0 * M[:, m]))
            lmul.append((1.0 + T / (2.0 * C_LIGHT ** 2)) / M[:, m])
        else:
            U = 2.0 * r[None, :] ** 2 * (Vt - Ep[:, None])
            ladd.append(np.zeros(k_au.size, complex))
            lmul.append(np.ones(k_au.size, complex))
        grids.append(r)
        Us.append(U)
        imt.append(m)
    return k_au, e_rel, E, p, grids, Us, imt, gam, np.array(ladd), np.array(lmul)


def _sph_jn_yn(lmax, x):
    """j_l, n_l (l = 0..lmax+1) of complex x, arrays (..., lmax+2)."""
    x = np.asarray(x, complex)
    L = lmax + 1
    jn = np.zeros(x.shape + (L + 1,), complex)
    yn = np.zeros_like(jn)
    yn[..., 0] = -np.cos(x) / x
    yn[..., 1] = -np.cos(x) / x ** 2 - np.sin(x) / x
    for l in range(1, L):
        yn[..., l + 1] = (2 * l + 1) / x * yn[..., l] - yn[..., l - 1]
    # Miller: downward from well above max(l, |x|)
    start = int(max(L, np.abs(x).max()) + 40)
    f1 = np.zeros(x.shape, complex)
    f0 = np.full(x.shape, 1e-30 + 0j)
    for l in range(start, 0, -1):
        fm = (2 * l + 1) / x * f0 - f1
        f1, f0 = f0, fm
        if l - 1 <= L:
            jn[..., l - 1] = fm
        big = np.abs(f0) > 1e250
        if big.any():
            f0 = np.where(big, f0 * 1e-250, f0)
            f1 = np.where(big, f1 * 1e-250, f1)
            jn = np.where(big[..., None], jn * 1e-250, jn)
    j0 = np.sin(x) / x
    jn = jn * (j0 / jn[..., 0])[..., None]
    return jn, yn


def _match(lmax, x, p, LR):
    """e^{2 i delta} from the log derivative LR = R'/R at x = p r_mt."""
    jn, yn = _sph_jn_yn(lmax, x)
    l = np.arange(lmax + 1)
    j, y = jn[..., :lmax + 1], yn[..., :lmax + 1]
    # f_l' = l/x f_l - f_{l+1}
    jd = l / x[..., None] * j - jn[..., 1:lmax + 2]
    yd = l / x[..., None] * y - yn[..., 1:lmax + 2]
    hp, hm = j + 1j * y, j - 1j * y
    hpd, hmd = jd + 1j * yd, jd - 1j * yd
    pp = p[..., None]
    return -(pp * hmd - LR * hm) / (pp * hpd - LR * hp)


def _start_values(l, Z, xs_ab, r_ab, relativistic, G0=None):
    """Regular solution at the first two points (relative normalisation is
    free): r^(l+1/2) (1 - Z r/(l+1)), or r^sqrt(G) scalar-relativistically
    (G0 (nE,) the coefficient at the first point)."""
    if relativistic:
        s = sr_start_exponent(G0)
        return np.exp(s * xs_ab[0]), np.exp(s * xs_ab[1])
    corr = (1.0 - Z * r_ab / (l + 1.0)) if l < 3 else np.ones(2)
    return np.exp((l + 0.5) * xs_ab) * corr


def phase_shifts_numpy(cp: ClusterPotential, k_A=None, lmax=LMAX_DEFAULT, gamma_ev=None,
                       model="hl", quinn=False, relativistic=None, mpse=None):
    k_A = default_k_grid() if k_A is None else np.asarray(k_A, float)
    if gamma_ev is None:
        gamma_ev = GAMMA_K.get(cp.types[0].symbol, 1.0)
    rel = _relativistic(cp, relativistic)
    k_au, e_rel, E, p, grids, Us, imt, gam, ladd, lmul = _prepare(cp, k_A, gamma_ev, model, quinn, rel,
                                                                 mpse=mpse)
    nE, nl = k_au.size, lmax + 1
    S = np.zeros((len(cp.types), nE, nl), complex)
    h2 = H_STEP ** 2
    ls = np.arange(nl)
    for it, t in enumerate(cp.types):
        r, U, m = grids[it], Us[it], imt[it]
        G = ((ls[None, :] + 0.5) ** 2)[:, :, None] + U[:, None, :]
        G = np.broadcast_to(G, (nE, nl, r.size))
        f = 1.0 - h2 * G / 12.0
        y = np.zeros((nE, nl, r.size), complex)
        # start each l where (r/r_mt)^(l+1/2) ~ e^-35, near the origin for low l
        xs = np.log(r) - np.log(r[m])
        i0 = np.array([max(0, int(np.searchsorted(xs, -35.0 / (l + 0.5)))) for l in ls])
        for l in ls:
            a, b = i0[l], i0[l] + 1
            y[:, l, a], y[:, l, b] = _start_values(l, t.Z, xs[[a, b]], r[[a, b]], rel, G[:, l, a])
        for i in range(1, r.size - 1):
            act = i0 < i
            yn = ((12.0 - 10.0 * f[:, :, i]) * y[:, :, i] - f[:, :, i - 1] * y[:, :, i - 1]) / f[:, :, i + 1]
            y[:, :, i + 1] = np.where(act[None, :], yn, y[:, :, i + 1])
        dy = (y[:, :, m - 2] - 8 * y[:, :, m - 1] + 8 * y[:, :, m + 1] - y[:, :, m + 2]) / (12 * H_STEP)
        LR = ((dy / y[:, :, m] - 0.5) / r[m] + ladd[it][:, None]) * lmul[it][:, None]
        x = p * r[m]
        S[it] = _match(lmax, x, p, LR)
    return PhaseShifts(k=k_A, energy=e_rel, p=p, S=S, labels=[t.label for t in cp.types],
                       gamma_ch=gam, r_mt=np.array([t.r_mt for t in cp.types]),
                       meta=dict(model=model, quinn=quinn, mu=cp.mu, v_int_coul=cp.v_int_coul,
                                 rho_int=cp.rho_int, lmax=lmax, relativistic=rel,
                                 mpse=None if mpse is None else mpse.label))


# --------------------------------------------------------------------------
# CUDA
# --------------------------------------------------------------------------

_KERNEL = r"""
#include <cupy/complex.cuh>
typedef complex<double> cplx;

extern "C" __global__ void radial_phase(
    const double* __restrict__ rgrid,    // (ntype, nr)
    const cplx*   __restrict__ U,        // (ntype, nE, nr) G - (l+1/2)^2
    const int*    __restrict__ imt,      // (ntype,)  index of r_mt
    const int*    __restrict__ nrt,      // (ntype,)  valid points per type
    const double* __restrict__ zat,      // (ntype,)  nuclear charge
    const cplx*   __restrict__ pint,     // (nE,)  interstitial momentum
    const cplx*   __restrict__ ladd,     // (ntype, nE) boundary: L_out = (L_in + ladd) lmul
    const cplx*   __restrict__ lmul,     // (ntype, nE)
    const int rel, const double zc2,     // scalar-relativistic start (zc2 unused, kept for the ABI)
    const int ntype, const int nE, const int nr, const int nl, const double h,
    cplx* __restrict__ S)                // (ntype, nE, nl)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= ntype * nE * nl) return;
    int l = tid % nl;
    int ie = (tid / nl) % nE;
    int it = tid / (nl * nE);
    const double* r = rgrid + (size_t)it * nr;
    const cplx* u = U + ((size_t)it * nE + ie) * nr;
    const int m = imt[it];
    const int n = nrt[it];
    const double lam = (l + 0.5) * (l + 0.5);
    const double h2 = h * h / 12.0;
    const double xm = log(r[m]);
    // start where (r/r_mt)^(l+1/2) ~ e^-35
    int i0 = 0;
    {
        double xs0 = -35.0 / (l + 0.5);
        while (i0 < m - 8 && log(r[i0]) - xm < xs0) ++i0;
    }
    double Z = zat[it];
    cplx y0, y1;
    if (rel) {
        // local power law y ~ r^sqrt(G): sqrt(l(l+1)+1-(Z/c)^2) at a nucleus, l+1/2 far from one
        double s = sqrt(lam + u[i0].real());
        y0 = exp(s * (log(r[i0]) - xm));
        y1 = exp(s * (log(r[i0 + 1]) - xm));
    } else {
        y0 = exp((l + 0.5) * (log(r[i0]) - xm)) * ((l < 3) ? (1.0 - Z * r[i0] / (l + 1.0)) : 1.0);
        y1 = exp((l + 0.5) * (log(r[i0 + 1]) - xm)) * ((l < 3) ? (1.0 - Z * r[i0 + 1] / (l + 1.0)) : 1.0);
    }
    cplx f0 = 1.0 - h2 * (lam + u[i0]);
    cplx f1 = 1.0 - h2 * (lam + u[i0 + 1]);
    cplx ys[5];                 // y at m-2 .. m+2
    for (int q = 0; q < 5; ++q) ys[q] = cplx(0.0, 0.0);
    if (i0 >= m - 2) { if (i0 <= m + 2) ys[i0 - (m - 2)] = y0; }
    if (i0 + 1 >= m - 2 && i0 + 1 <= m + 2) ys[i0 + 1 - (m - 2)] = y1;
    for (int i = i0 + 1; i < n - 1; ++i) {
        cplx f2 = 1.0 - h2 * (lam + u[i + 1]);
        cplx y2 = ((12.0 - 10.0 * f1) * y1 - f0 * y0) / f2;
        y0 = y1; y1 = y2; f0 = f1; f1 = f2;
        int q = i + 1 - (m - 2);
        if (q >= 0 && q < 5) ys[q] = y2;
    }
    cplx dy = (ys[0] - 8.0 * ys[1] + 8.0 * ys[3] - ys[4]) / (12.0 * h);
    size_t ib = (size_t)it * nE + ie;
    cplx LR = ((dy / ys[2] - 0.5) / r[m] + ladd[ib]) * lmul[ib];
    // free spherical waves at x = p r_mt: j_l, n_l, j_{l+1}, n_{l+1}
    cplx p = pint[ie];
    cplx x = p * r[m];
    cplx sx = sin(x), cx = cos(x);
    // n_l upward
    cplx yA = -cx / x, yB = -cx / (x * x) - sx / x;      // n_0, n_1
    for (int q = 1; q <= l; ++q) { cplx yC = (2.0 * q + 1.0) / x * yB - yA; yA = yB; yB = yC; }
    cplx nl_ = yA, nl1 = yB;                              // n_l, n_{l+1}
    // j_l by Miller's downward recurrence, normalised to j_0
    double ax = abs(x);
    int start = (int)(fmax((double)(l + 1), ax)) + 40;
    cplx g1 = cplx(0.0, 0.0), g0 = cplx(1e-30, 0.0);
    cplx jl = 0.0, jl1 = 0.0, j0 = 0.0;
    for (int q = start; q > 0; --q) {
        cplx gm = (2.0 * q + 1.0) / x * g0 - g1;     // g_{q-1}
        g1 = g0; g0 = gm;
        if (q - 1 == l + 1) jl1 = gm;
        if (q - 1 == l) jl = gm;
        if (q - 1 == 0) j0 = gm;
        if (abs(g0) > 1e250) { g0 *= 1e-250; g1 *= 1e-250; jl *= 1e-250; jl1 *= 1e-250; j0 *= 1e-250; }
    }
    cplx norm = (sx / x) / j0;
    jl *= norm; jl1 *= norm;
    cplx jd = (double)l / x * jl - jl1;
    cplx yd = (double)l / x * nl_ - nl1;
    cplx I(0.0, 1.0);
    cplx hp = jl + I * nl_, hm = jl - I * nl_;
    cplx hpd = jd + I * yd, hmd = jd - I * yd;
    S[tid] = -(p * hmd - LR * hm) / (p * hpd - LR * hp);
}
"""

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        import cupy as cp
        _MOD = cp.RawModule(code=_KERNEL, options=("--std=c++14",))
    return _MOD


def phase_shifts(cp_: ClusterPotential, k_A=None, lmax=LMAX_DEFAULT, gamma_ev=None,
                 model="hl", quinn=False, relativistic=None, mpse=None) -> PhaseShifts:
    """All phase shifts of a cluster's potential types on the GPU.
    `relativistic` defaults to how the potential was built; `mpse` (a
    `losses.Poles`) adds the many-pole self-energy of the material."""
    import cupy as cp
    k_A = default_k_grid() if k_A is None else np.asarray(k_A, float)
    if gamma_ev is None:
        gamma_ev = GAMMA_K.get(cp_.types[0].symbol, 1.0)
    rel = _relativistic(cp_, relativistic)
    k_au, e_rel, E, p, grids, Us, imt, gam, ladd, lmul = _prepare(cp_, k_A, gamma_ev, model, quinn, rel,
                                                                 mpse=mpse)
    nt, nE, nl = len(cp_.types), k_au.size, lmax + 1
    nr = max(g.size for g in grids)
    R = np.ones((nt, nr)); Ua = np.zeros((nt, nE, nr), complex)
    nrt = np.zeros(nt, np.int32)
    for i in range(nt):
        n = grids[i].size
        R[i, :n], Ua[i, :, :n], nrt[i] = grids[i], Us[i], n
    out = cp.zeros((nt, nE, nl), cp.complex128)
    ker = _module().get_function("radial_phase")
    nthr = nt * nE * nl
    block = 128
    ker(((nthr + block - 1) // block,), (block,),
        (cp.asarray(R), cp.asarray(Ua), cp.asarray(np.array(imt, np.int32)),
         cp.asarray(nrt), cp.asarray(np.array([t.Z for t in cp_.types], float)),
         cp.asarray(p.astype(complex)), cp.asarray(np.ascontiguousarray(ladd, complex)),
         cp.asarray(np.ascontiguousarray(lmul, complex)), np.int32(rel), np.float64(1.0 / C_LIGHT ** 2),
         np.int32(nt), np.int32(nE), np.int32(nr), np.int32(nl), np.float64(H_STEP), out))
    return PhaseShifts(k=k_A, energy=e_rel, p=p, S=cp.asnumpy(out),
                       labels=[t.label for t in cp_.types], gamma_ch=gam,
                       r_mt=np.array([t.r_mt for t in cp_.types]),
                       meta=dict(model=model, quinn=quinn, mu=cp_.mu,
                                 v_int_coul=cp_.v_int_coul, rho_int=cp_.rho_int, lmax=lmax,
                                 relativistic=rel, mpse=None if mpse is None else mpse.label))
