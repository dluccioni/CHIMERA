"""Exact curved-wave multiple scattering, path by path.

For a K edge (l = 1 final state), orientation averaged, a scattering path
absorber -> R_1 -> ... -> R_N -> absorber contributes

    chi_path(E) = 1/3 Re[ e^{2 i delta_c} Tr_{l=1} G(a_N) t_N G(a_{N-1}) ... t_1 G(a_0) ]

with a_j the leg vectors, G(a) the free propagator between two sites in the
angular-momentum basis (`sphere.py`), t_l = (e^{2 i delta_l} - 1)/2 the
scatterers' amplitudes and delta_c the absorber's l = 1 phase shift. Nothing
is truncated but the angular momentum, chosen per energy from where |t_l|
falls below 1e-6 (1e-6 relative on chi): no plane-wave or small-atom approximation, and no
Rehr-Albers separable expansion (whose order-2 truncation is FEFF's default).
In the plane-wave limit the single-scattering term reduces to the textbook
-Im[f(pi) e^{2i(kR + delta_c)}]/(kR^2).

The product is evaluated in each leg's own frame (z along the leg), where G
is diagonal in m: a leg costs O(l^3) and the change of frame between legs is
a Wigner rotation, applied as two products with the constant matrix
Delta = d(pi/2) and three diagonal phases. `path_sum_numpy` does this on the
CPU as the reference; `path_sum` is the CUDA kernel, one thread block per
(path, energy), which is what the table builder and the synthetic-data
generator call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import sphere as S
from .phaseshift import PhaseShifts
from .potential import BOHR


@dataclass
class Path:
    """One scattering path. `sites` are the scatterer positions (bohr, the
    absorber at the origin is implicit at both ends); `types` their potential
    types (index into PhaseShifts.S)."""
    sites: np.ndarray
    types: tuple
    degeneracy: float = 1.0

    @property
    def nleg(self):
        return len(self.types) + 1

    def legs(self):
        pts = np.vstack([np.zeros(3), np.asarray(self.sites, float), np.zeros(3)])
        return np.diff(pts, axis=0)                    # a_0 .. a_N

    @property
    def reff(self):
        """Half path length (bohr)."""
        return 0.5 * np.linalg.norm(self.legs(), axis=1).sum()

    def frames(self):
        """Leg lengths and the Euler angles of the frame changes Q_1..Q_N and
        the closing Q_{N+1} = R_0^T R_N."""
        a = self.legs()
        R = [S.frame_of(v) for v in a]
        lens = np.linalg.norm(a, axis=1)
        eul = [S.euler_zyz(R[j].T @ R[j - 1]) for j in range(1, len(R))]
        eul.append(S.euler_zyz(R[0].T @ R[-1]))
        return lens, np.array(eul)


LMAX_TOL = 1e-6           # |t_l| cut; 1.4e-6 rms on chi, 2x faster than 1e-8


def lmax_per_energy(ph: PhaseShifts, types, tol=LMAX_TOL, cap=None):
    """Angular momentum needed at each energy for the given scatterer types."""
    t = np.abs(ph.t[list(sorted(set(types)))]).max(axis=0)      # (nE, nl)
    out = np.array([max(2, int(np.max(np.where(row > tol)[0], initial=1))) for row in t])
    if cap is not None:
        out = np.minimum(out, cap)
    return out


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------

def _apply_g(g, v, lmax):
    out = np.zeros_like(v)
    for m in range(-lmax, lmax + 1):
        ls = np.arange(abs(m), lmax + 1)
        idx = ls * ls + ls + m
        out[idx] = g[lmax + m][np.ix_(ls, ls)] @ v[idx]
    return out


def _apply_D(v, lmax, eul):
    out = np.empty_like(v)
    for l in range(lmax + 1):
        sl = slice(l * l, (l + 1) ** 2)
        out[sl] = S.rotation_D(l, *eul) @ v[sl]
    return out


def trace_numpy(path: Path, ph: PhaseShifts, ie: int, lmax: int):
    """Tr_{l=1} of the path product at one energy (complex)."""
    p = ph.p[ie]
    lens, eul = path.frames()
    n = (lmax + 1) ** 2
    v = np.zeros((n, 3), complex)
    for c, mu in enumerate((-1, 0, 1)):
        v[S.lidx(1, mu), c] = 1.0
    v = _apply_g(S.g_z(lmax, p * lens[0]), v, lmax)
    for j, typ in enumerate(path.types):
        t = ph.t[typ, ie, :lmax + 1]
        tl = np.repeat(t, 2 * np.arange(lmax + 1) + 1)
        v = tl[:, None] * v
        v = _apply_D(v, lmax, eul[j])
        v = _apply_g(S.g_z(lmax, p * lens[j + 1]), v, lmax)
    v = _apply_D(v, lmax, eul[-1])
    return sum(v[S.lidx(1, mu), c] for c, mu in enumerate((-1, 0, 1)))


def path_sum_numpy(path: Path, ph: PhaseShifts, lmax=None, energies=None):
    """chi_path on the phase-shift energy grid (or a subset), plus Tr P."""
    ie_all = np.arange(ph.k.size) if energies is None else np.asarray(energies)
    lm = lmax_per_energy(ph, path.types) if lmax is None else np.full(ph.k.size, lmax)
    tr = np.array([trace_numpy(path, ph, ie, int(lm[ie])) for ie in ie_all])
    Sc = ph.S[0, ie_all, 1]
    chi = (Sc * tr).real / 3.0
    return chi, tr


# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------

_KERNEL = r"""
#include <cupy/complex.cuh>
#ifndef REAL
#define REAL double
#endif
typedef complex<REAL> cplx;
#define LMAXK %(LMAXK)d
#define NL (LMAXK + 1)
#define NLM (NL * NL)
#define MAXOWN %(MAXOWN)d

__device__ __forceinline__ int lidx(int l, int m) { return l * l + l + m; }

// i^n for integer n
__device__ __forceinline__ cplx ipow(int n) {
    n = ((n %% 4) + 4) %% 4;
    return n == 0 ? cplx(1, 0) : n == 1 ? cplx(0, 1) : n == 2 ? cplx(-1, 0) : cplx(0, -1);
}

// One block per (path, energy). Path layout (all arrays padded to MAXS
// scatterers):  nsc[path], typ[path*MAXS + j], leg[path*(MAXS+1) + j] (bohr),
// eul[(path*(MAXS+1) + j)*3 + {0,1,2}]  j = 0..nsc  (j < nsc: Q_{j+1}; j = nsc: closing)
extern "C" __global__ void path_trace(
    const int*    __restrict__ nsc,
    const int*    __restrict__ typ,
    const double* __restrict__ leg,
    const double* __restrict__ eul,
    const int MAXS,
    const complex<double>* __restrict__ pmom,   // (nE,)
    const complex<double>* __restrict__ tl,     // (ntype, nE, NLT)
    const int NLT,
    const int*    __restrict__ lmaxE,           // (nE,)
    const int nE,
    const int*    __restrict__ goff,            // (NL, NL, NL)  [m][l'][l]
    const int*    __restrict__ gcnt,
    const int*    __restrict__ glpp,
    const cplx* __restrict__ gA,
    const long long* __restrict__ doff,         // (NL+1,) Delta^l offsets
    const REAL* __restrict__ dflat,
    complex<double>* __restrict__ out)          // (npath, nE)
{
    const int ie = blockIdx.x;
    const int ip = blockIdx.y;
    const int lmax = lmaxE[ie];
    const int nlm = (lmax + 1) * (lmax + 1);
    // dynamic shared memory: 3 state columns, then the Hankel functions
    extern __shared__ __align__(16) unsigned char smem_raw[];
    cplx (*v)[NLM] = reinterpret_cast<cplx (*)[NLM]>(smem_raw);
    cplx* hk = reinterpret_cast<cplx*>(smem_raw + sizeof(cplx) * 3 * NLM);
    const int tid = threadIdx.x, nth = blockDim.x;
    const cplx p = cplx((REAL)pmom[ie].real(), (REAL)pmom[ie].imag());
    const int ns = nsc[ip];

    for (int q = tid; q < 3 * NLM; q += nth) v[q / NLM][q %% NLM] = cplx(0, 0);
    __syncthreads();
    if (tid < 3) v[tid][lidx(1, tid - 1)] = cplx(1, 0);
    __syncthreads();

    for (int j = 0; j <= ns; ++j) {
        if (j > 0) {
            // --- scatterer j: multiply by t_l of its type
            const int ty = typ[ip * MAXS + j - 1];
            const complex<double>* t = tl + ((size_t)ty * nE + ie) * NLT;
            for (int q = tid; q < 3 * nlm; q += nth) {
                int c = q / nlm, L = q %% nlm;
                int l = (int)sqrtf((float)L + 0.5f);
                while (l * l > L) --l;
                while ((l + 1) * (l + 1) <= L) ++l;
                complex<double> tv = t[l];
                v[c][L] *= cplx((REAL)tv.real(), (REAL)tv.imag());
            }
            __syncthreads();
        }
        if (j > 0) {
            // --- rotate into this leg's frame: Q_j (Euler angles of j-1 entry)
            const double* e = eul + ((size_t)ip * (MAXS + 1) + (j - 1)) * 3;
            REAL al = (REAL)e[0], be = (REAL)e[1], ga = (REAL)e[2];
            // (1) v_m' *= i^-m' e^{-i m' ga}
            for (int q = tid; q < 3 * nlm; q += nth) {
                int c = q / nlm, L = q %% nlm;
                int l = (int)sqrtf((float)L + 0.5f); while (l * l > L) --l; while ((l + 1) * (l + 1) <= L) ++l;
                int m = L - l * l - l;
                REAL s, co; sincos(-(REAL)m * ga, &s, &co);
                v[c][L] *= ipow(-m) * cplx(co, s);
            }
            __syncthreads();
            // (2) u_m'' = sum_m' Delta_{m''m'} v_m'  then *= e^{-i m'' be}
            {
                cplx acc[3][MAXOWN];
                #pragma unroll
                for (int nown = 0; nown < MAXOWN; ++nown) {
                    int q = tid + nown * nth; if (q >= nlm) break;
                    int L = q; int l = (int)sqrtf((float)L + 0.5f); while (l * l > L) --l; while ((l + 1) * (l + 1) <= L) ++l;
                    int mpp = L - l * l - l;
                    const REAL* D = dflat + doff[l] + (size_t)(mpp + l) * (2 * l + 1);
                    cplx a0(0, 0), a1(0, 0), a2(0, 0);
                    for (int mp = -l; mp <= l; ++mp) {
                        REAL d = D[mp + l];
                        int Lp = l * l + l + mp;
                        a0 += d * v[0][Lp]; a1 += d * v[1][Lp]; a2 += d * v[2][Lp];
                    }
                    REAL s, co; sincos(-(REAL)mpp * be, &s, &co);
                    cplx ph(co, s);
                    acc[0][nown] = a0 * ph; acc[1][nown] = a1 * ph; acc[2][nown] = a2 * ph;
                }
                __syncthreads();
                #pragma unroll
                for (int nown = 0; nown < MAXOWN; ++nown) {
                    int q = tid + nown * nth; if (q >= nlm) break;
                    v[0][q] = acc[0][nown]; v[1][q] = acc[1][nown]; v[2][q] = acc[2][nown];
                }
                __syncthreads();
            }
            // (3) w_m = i^m e^{-i m al} sum_m'' Delta_{m''m} u_m''
            {
                cplx acc[3][MAXOWN];
                #pragma unroll
                for (int nown = 0; nown < MAXOWN; ++nown) {
                    int q = tid + nown * nth; if (q >= nlm) break;
                    int L = q; int l = (int)sqrtf((float)L + 0.5f); while (l * l > L) --l; while ((l + 1) * (l + 1) <= L) ++l;
                    int m = L - l * l - l;
                    const REAL* D = dflat + doff[l];
                    cplx a0(0, 0), a1(0, 0), a2(0, 0);
                    for (int mpp = -l; mpp <= l; ++mpp) {
                        REAL d = D[(size_t)(mpp + l) * (2 * l + 1) + (m + l)];
                        int Lp = l * l + l + mpp;
                        a0 += d * v[0][Lp]; a1 += d * v[1][Lp]; a2 += d * v[2][Lp];
                    }
                    REAL s, co; sincos(-(REAL)m * al, &s, &co);
                    cplx ph = ipow(m) * cplx(co, s);
                    acc[0][nown] = a0 * ph; acc[1][nown] = a1 * ph; acc[2][nown] = a2 * ph;
                }
                __syncthreads();
                #pragma unroll
                for (int nown = 0; nown < MAXOWN; ++nown) {
                    int q = tid + nown * nth; if (q >= nlm) break;
                    v[0][q] = acc[0][nown]; v[1][q] = acc[1][nown]; v[2][q] = acc[2][nown];
                }
                __syncthreads();
            }
        }
        // --- propagate along leg j: v_l'm = sum_l g^m_{l'l} v_lm,  g = sum A h_l''
        {
            const double R = leg[(size_t)ip * (MAXS + 1) + j];
            // Hankel functions h_0..h_{2lmax}(pR), upward recurrence
            if (tid == 0) {
                cplx x = p * (REAL)R;
                cplx ex = exp(cplx(0, 1) * x);
                hk[0] = cplx(0, -1) * ex / x;
                hk[1] = -ex / x * (cplx(1, 0) + cplx(0, 1) / x);
                for (int l = 1; l < 2 * lmax; ++l) hk[l + 1] = (REAL)(2 * l + 1) / x * hk[l] - hk[l - 1];
            }
            __syncthreads();
            cplx acc[3][MAXOWN];
            #pragma unroll
            for (int nown = 0; nown < MAXOWN; ++nown) {
                int q = tid + nown * nth; if (q >= nlm) break;
                int L = q; int lp = (int)sqrtf((float)L + 0.5f); while (lp * lp > L) --lp; while ((lp + 1) * (lp + 1) <= L) ++lp;
                int m = L - lp * lp - lp;
                int am = m < 0 ? -m : m;
                cplx a0(0, 0), a1(0, 0), a2(0, 0);
                for (int l = am; l <= lmax; ++l) {
                    int o = goff[(am * NL + lp) * NL + l];
                    int c = gcnt[(am * NL + lp) * NL + l];
                    cplx g(0, 0);
                    for (int u = 0; u < c; ++u) {
                        g += gA[o + u] * hk[glpp[o + u]];
                    }
                    int Ls = l * l + l + m;
                    a0 += g * v[0][Ls]; a1 += g * v[1][Ls]; a2 += g * v[2][Ls];
                }
                acc[0][nown] = a0; acc[1][nown] = a1; acc[2][nown] = a2;
            }
            __syncthreads();
            #pragma unroll
            for (int nown = 0; nown < MAXOWN; ++nown) {
                int q = tid + nown * nth; if (q >= nlm) break;
                v[0][q] = acc[0][nown]; v[1][q] = acc[1][nown]; v[2][q] = acc[2][nown];
            }
            __syncthreads();
        }
    }
    // --- closing rotation back to the first leg's frame, l = 1 block only
    if (tid == 0) {
        const double* e = eul + ((size_t)ip * (MAXS + 1) + ns) * 3;
        REAL al = (REAL)e[0], be = (REAL)e[1], ga = (REAL)e[2];
        const REAL* D = dflat + doff[1];
        cplx tr(0, 0);
        for (int c = 0; c < 3; ++c) {
            int mu = c - 1;          // column c started as e_{1,mu}; take row (1,mu)
            // row mu of D^1(al,be,ga) applied to the l=1 part of column c
            cplx w(0, 0);
            for (int mp = -1; mp <= 1; ++mp) {
                // d_{mu,mp}(be) = i^(mu-mp) sum_m'' D_{m''mu} D_{m''mp} e^{-i m'' be}
                cplx d(0, 0);
                for (int mpp = -1; mpp <= 1; ++mpp) {
                    REAL s, co; sincos(-(REAL)mpp * be, &s, &co);
                    d += (D[(mpp + 1) * 3 + (mu + 1)] * D[(mpp + 1) * 3 + (mp + 1)]) * cplx(co, s);
                }
                d *= ipow(mu - mp);
                REAL s1, c1; sincos(-(REAL)mu * al - (REAL)mp * ga, &s1, &c1);
                w += cplx(c1, s1) * d * v[c][lidx(1, mp)];
            }
            tr += w;
        }
        out[(size_t)ip * nE + ie] = complex<double>((double)tr.real(), (double)tr.imag());
    }
}
"""

_MODS = {}
MAX_OWN = 8          # outputs per thread held in registers (nlm <= 8 * threads)


_TABLES = {}


def _kernel(lmax_kernel, real="double"):
    """(kernel, device tables) for one angular-momentum size and precision,
    compiled and uploaded once per process."""
    key = (lmax_kernel, real)
    if key not in _MODS:
        import cupy as cp
        src = _KERNEL % dict(LMAXK=lmax_kernel, MAXOWN=MAX_OWN)
        mod = cp.RawModule(code=src, options=("--std=c++14", f"-DREAL={real}"))
        ker = mod.get_function("path_trace")
        off, cnt, L, A = S.propagator_coeffs(lmax_kernel)
        doff, dflat = S.delta_matrices(lmax_kernel)
        cdt = np.complex128 if real == "double" else np.complex64
        rdt = np.float64 if real == "double" else np.float32
        tabs = [cp.asarray(off), cp.asarray(cnt), cp.asarray(L), cp.asarray(A.astype(cdt)),
                cp.asarray(doff.astype(np.int64)), cp.asarray(dflat.astype(rdt))]
        _MODS[key] = (ker, tabs)
    return _MODS[key]


def _pack_paths(paths):
    maxs = max(len(p.types) for p in paths)
    n = len(paths)
    nsc = np.zeros(n, np.int32)
    typ = np.zeros((n, maxs), np.int32)
    leg = np.zeros((n, maxs + 1))
    eul = np.zeros((n, maxs + 1, 3))
    for i, p in enumerate(paths):
        ns = len(p.types)
        nsc[i] = ns
        typ[i, :ns] = p.types
        lens, e = p.frames()
        leg[i, :ns + 1] = lens
        eul[i, :ns + 1] = e
    return maxs, nsc, typ, leg, eul


def frames_batch(legs, nsc):
    """Vectorised Path.frames for many paths.

    legs : (P, MAXS+1, 3) leg vectors (bohr), zero-padded past nsc+1 legs
    nsc  : (P,) scatterers per path
    Returns lens (P, MAXS+1) and eul (P, MAXS+1, 3) in the kernel's layout:
    eul[:, j] = Euler(Q_{j+1}) for j < nsc, eul[:, nsc] = closing rotation.
    """
    legs = np.asarray(legs, float)
    P, M1, _ = legs.shape
    lens = np.linalg.norm(legs, axis=2)
    safe = np.where(lens > 0, lens, 1.0)
    th = np.arccos(np.clip(legs[..., 2] / safe, -1.0, 1.0))
    ph = np.arctan2(legs[..., 1], legs[..., 0])
    cz, sz, cy, sy = np.cos(ph), np.sin(ph), np.cos(th), np.sin(th)
    R = np.zeros((P, M1, 3, 3))
    # R = Rz(ph) Ry(th)
    R[..., 0, 0] = cz * cy; R[..., 0, 1] = -sz; R[..., 0, 2] = cz * sy
    R[..., 1, 0] = sz * cy; R[..., 1, 1] = cz;  R[..., 1, 2] = sz * sy
    R[..., 2, 0] = -sy;     R[..., 2, 1] = 0.0; R[..., 2, 2] = cy
    eul = np.zeros((P, M1, 3))
    rows = np.arange(P)
    for j in range(M1):
        nxt = np.minimum(j + 1, nsc)                     # leg after scatterer j+1
        cur = np.minimum(j, nsc)
        closing = j == nsc
        A = np.where(closing[:, None, None], R[:, 0], R[rows, nxt])
        B = np.where(closing[:, None, None], R[rows, nsc], R[rows, cur])
        Q = np.einsum("pji,pjk->pik", A, B)              # A^T B
        valid = j <= nsc
        beta = np.arccos(np.clip(Q[:, 2, 2], -1.0, 1.0))
        sb = np.sin(beta)
        gen = np.abs(sb) > 1e-10
        alpha = np.where(gen, np.arctan2(Q[:, 1, 2], Q[:, 0, 2]),
                         np.where(Q[:, 2, 2] > 0, np.arctan2(Q[:, 1, 0], Q[:, 0, 0]),
                                  np.arctan2(-Q[:, 1, 0], -Q[:, 0, 0])))
        gamma = np.where(gen, np.arctan2(Q[:, 2, 1], -Q[:, 2, 0]), 0.0)
        eul[:, j, 0] = np.where(valid, alpha, 0.0)
        eul[:, j, 1] = np.where(valid, beta, 0.0)
        eul[:, j, 2] = np.where(valid, gamma, 0.0)
    return lens, eul


def path_sum_packed(nsc, typ, leg, eul, ph: PhaseShifts, lmax, precision="double",
                    threads=128, energies=None, out_device=False):
    """Tr_{l=1} P from packed arrays (see _pack_paths / frames_batch).
    energies: optional index array into ph's grid; returns (npath, nE_sel)."""
    import cupy as cp
    lmax = np.asarray(lmax, np.int32)
    ie = np.arange(ph.k.size) if energies is None else np.asarray(energies)
    lsel = np.ascontiguousarray(lmax[ie])
    LK = int(lsel.max())
    if LK > 44:
        raise ValueError(f"lmax {LK} exceeds the kernel's shared-memory budget (44)")
    if (LK + 1) ** 2 > MAX_OWN * threads:
        threads = int(math.ceil((LK + 1) ** 2 / MAX_OWN / 32.0)) * 32
    ker, tabs = _kernel(LK, precision)
    csize = 16 if precision == "double" else 8
    smem = csize * (3 * (LK + 1) ** 2 + 2 * (LK + 1))
    if smem > 48 * 1024:
        ker.max_dynamic_shared_size_bytes = smem
    tl = cp.asarray(np.ascontiguousarray(ph.t[:, ie, :]))
    pm = cp.asarray(np.ascontiguousarray(ph.p[ie]).astype(complex))
    nE = ie.size
    npth = len(nsc)
    maxs = typ.shape[1]
    o = cp.zeros((npth, nE), cp.complex128)
    lsel_d = cp.asarray(lsel)
    CH = 60000
    for s in range(0, npth, CH):
        e = min(npth, s + CH)
        ker((nE, e - s), (threads,),
            (cp.asarray(np.ascontiguousarray(nsc[s:e], np.int32)),
             cp.asarray(np.ascontiguousarray(typ[s:e], np.int32).ravel()),
             cp.asarray(np.ascontiguousarray(leg[s:e], float).ravel()),
             cp.asarray(np.ascontiguousarray(eul[s:e], float).ravel()), np.int32(maxs),
             pm, tl, np.int32(ph.S.shape[2]), lsel_d, np.int32(nE), *tabs, o[s:e]),
            shared_mem=smem)
    return o if out_device else cp.asnumpy(o)


def path_sum(paths, ph: PhaseShifts, lmax=None, precision="double", threads=128,
             tol=LMAX_TOL):
    """Tr_{l=1} P for every path at every energy of `ph` (complex, (npath, nE)),
    on the GPU. lmax: per-energy array, or None for the |t_l| > tol rule."""
    paths = list(paths)
    if lmax is None:
        types = sorted({t for p in paths for t in p.types})
        lmax = lmax_per_energy(ph, types, tol=tol)
    maxs, nsc, typ, leg, eul = _pack_paths(paths)
    return path_sum_packed(nsc, typ, leg, eul, ph, lmax, precision, threads)


def chi_from_trace(tr, ph: PhaseShifts):
    """chi(E) = 1/3 Re[e^{2 i delta_c} Tr P]."""
    return (ph.S[0, :, 1][None, :] * np.atleast_2d(tr)).real / 3.0
