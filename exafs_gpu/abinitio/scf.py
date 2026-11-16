"""Self-consistent muffin-tin potentials from the cluster Green's function.

The overlapped-atom potentials of `potential.py` leave every electron where the
free atom has it. In a solid, charge moves between unlike neighbours and the
Fermi level settles where the cluster holds exactly its electrons. FEFF8 finds
that state self-consistently (its SCF card); this module does the same from the
multiple-scattering Green's function, on the GPU.

1. Valence density. For each potential type, a cluster of radius `r_scf` is
   centred on its representative atom (FEFF's choice for solids). With t_l the
   scattering amplitudes of the ground-state potentials (t = (e^{2i delta}-1)/2,
   the convention of `phaseshift`), G the free propagators between sites
   (`sphere`), and R_l, H_l the regular and irregular radial solutions
   (R -> j_l + t_l h_l and H -> h_l outside the muffin tin), the Green's
   function at the centre, spherically averaged, is

       G(r, r; z) = -2ip/(4 pi) sum_l R_l(r) [(2l+1) H_l(r) + R_l(r) tr_l T(z)],
       T = G_out t (1 - G t)^{-1} G_in,

   the single-site term plus every path that leaves the centre and returns.
   The single-site sum and the returning waves are carried to `lmax_out`; the
   scatterers to `lmax_fms`.
2. The density is -(2/pi) Im int f(E) G(E + i0) dE (two spins). The integral
   runs where G is smooth: up from E_b (between the core levels and the bottom
   of the valence band) to E_b + iY, then along Im z = Y = 2 N pi kT, where the
   Fermi function is real, plus the N Matsubara poles under that line.
3. The Fermi level makes the cluster neutral: the valence charge in the Norman
   spheres, weighted by how many atoms of each type the cluster holds, equals
   the valence electron count, the absorber's screening electron included.
4. Core electrons stay in their free-atom (core-hole atom) orbitals. New
   Coulomb potentials come from the Norman-sphere densities overlapped as in
   the Mattheiss construction, so a site that gained or lost charge shifts its
   neighbours' levels; exchange-correlation is the LDA of the new density. The
   input density is mixed (Anderson) until charges and Fermi level settle.

Muffin-tin and Norman radii stay those of the overlapped-atom start, as in
FEFF. The result is a `ClusterPotential` that `phaseshift.phase_shifts` takes
unchanged; the photoelectron then sees the Hedin-Lundqvist self-energy of the
self-consistent density.

Hartree atomic units inside; positions come in Angstrom.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np

from . import potential as PO
from .atom import C_LIGHT, LogGrid, atom_for, hartree, lda_xc, sr_coefficients, sr_derivatives, sr_start_exponent
from .phaseshift import _sph_jn_yn, interstitial_momentum
from .potential import BOHR, HARTREE, ClusterPotential
from .sphere import euler_zyz, frame_of, hankel_all, bessel_j_all, propagator_coeffs, rotation_D

H_RAD = 0.006          # log step of the radial grid for the Green's function
R_START = 1e-6         # bohr
CORE_CUT = -1.0        # Ha: free-atom orbitals below this are frozen core


# --------------------------------------------------------------------------
# energy contour
# --------------------------------------------------------------------------

def fermi(z, mu, kT):
    """Fermi function of complex energy, overflow-safe."""
    x = np.asarray((z - mu) / kT, complex)
    out = np.empty_like(x)
    big = x.real > 0
    e = np.exp(-x[big])
    out[big] = e / (1.0 + e)
    out[~big] = 1.0 / (np.exp(x[~big]) + 1.0)
    return out


def contour(e_bottom, mu, kT, n_pole=4, n_vert=8, n_main=24, n_fermi=32, width=15.0):
    """Points z and weights w with  int f(E) G(E + i0) dE = sum w G(z) - for any
    G analytic in the upper half plane and free of states below e_bottom.

    Vertical leg e_bottom -> e_bottom + iY, the line Im z = Y = 2 n_pole pi kT
    from e_bottom to mu + width kT (the Fermi window [mu -+ width kT] with its
    own Gauss-Legendre rule), and the n_pole Matsubara poles mu + i(2n+1) pi kT
    (residue of f: -kT; enclosed by the contour, so weight -2 pi i kT)."""
    Y = 2.0 * n_pole * math.pi * kT
    zs, ws = [], []
    x, w = np.polynomial.legendre.leggauss(n_vert)
    y = 0.5 * Y * (x + 1.0)
    z = e_bottom + 1j * y
    zs.append(z)
    ws.append(1j * 0.5 * Y * w * fermi(z, mu, kT))
    a, b = e_bottom, mu - width * kT
    if b > a:
        x, w = np.polynomial.legendre.leggauss(n_main)
        z = 0.5 * (b - a) * (x + 1.0) + a + 1j * Y
        zs.append(z)
        ws.append(0.5 * (b - a) * w * fermi(z, mu, kT))
    a, b = max(e_bottom, mu - width * kT), mu + width * kT
    x, w = np.polynomial.legendre.leggauss(n_fermi)
    z = 0.5 * (b - a) * (x + 1.0) + a + 1j * Y
    zs.append(z)
    ws.append(0.5 * (b - a) * w * fermi(z, mu, kT))
    zp = mu + 1j * (2 * np.arange(n_pole) + 1) * math.pi * kT
    zs.append(zp)
    ws.append(np.full(n_pole, -2j * math.pi * kT))
    return np.concatenate(zs), np.concatenate(ws)


# --------------------------------------------------------------------------
# free propagators
# --------------------------------------------------------------------------

def nlm(lmax):
    return (lmax + 1) ** 2


def l_of_index(lmax):
    return np.repeat(np.arange(lmax + 1), 2 * np.arange(lmax + 1) + 1)


def gz_batch(lmax, x):
    """z-axis propagator in the L basis, (..., nL, nL), for complex x = p d."""
    x = np.asarray(x, complex)
    h = hankel_all(2 * lmax, x)
    off, cnt, Lpp, A = propagator_coeffs(lmax)
    out = np.zeros(x.shape + (nlm(lmax), nlm(lmax)), complex)
    for m in range(lmax + 1):
        for lp in range(m, lmax + 1):
            for l in range(m, lmax + 1):
                o, c = off[m, lp, l], cnt[m, lp, l]
                if c == 0:
                    continue
                v = (h[..., Lpp[o:o + c]] * A[o:o + c]).sum(-1)
                out[..., lp * lp + lp + m, l * l + l + m] = v
                if m:
                    out[..., lp * lp + lp - m, l * l + l - m] = v
    return out


def wigner_blocks(lmax, vecs):
    """Block-diagonal D(R) with R z = v/|v|, one (nL, nL) matrix per vector."""
    nL = nlm(lmax)
    out = np.zeros((len(vecs), nL, nL), complex)
    for i, v in enumerate(vecs):
        al, be, ga = euler_zyz(frame_of(v))
        for l in range(lmax + 1):
            s = slice(l * l, (l + 1) ** 2)
            out[i, s, s] = rotation_D(l, al, be, ga)
    return out


def propagator(lmax, a, p):
    """G_{L'L}(a) for one vector a (bohr) and momenta p (nz,): (nz, nL, nL).
    h_l(p|r+a|) Y_L(r+a) = sum_L' G_{L'L}(a) j_l'(p r) Y_L'(r)."""
    D = wigner_blocks(lmax, [a])[0]
    g = gz_batch(lmax, np.asarray(p) * np.linalg.norm(a))
    return D[None] @ g @ D.conj().T[None]


# --------------------------------------------------------------------------
# radial solutions at complex energy
# --------------------------------------------------------------------------

def t_from_logderiv(lmax, x, p, LR):
    """t_l = -(p j' - L j)/(p h' - L h) at x = p r_mt, L = R'/R inside.

    The same t as (e^{2i delta} - 1)/2, but formed directly: at high l the
    physical t is far below the rounding of e^{2i delta} = 1 + 2t, and
    t h_l(p r_mt) - h_l is huge there - would swamp j_l in R = j_l + t_l h_l."""
    jn, yn = _sph_jn_yn(lmax, x)
    l = np.arange(lmax + 1)
    j, y = jn[..., :lmax + 1], yn[..., :lmax + 1]
    jd = l / x[..., None] * j - jn[..., 1:lmax + 2]
    yd = l / x[..., None] * y - yn[..., 1:lmax + 2]
    hp, hpd = j + 1j * y, jd + 1j * yd
    pp = p[..., None]
    return -(pp * jd - LR * j) / (pp * hpd - LR * hp)


def type_grid(r_mt, r_norman, h=H_RAD):
    """Log grid with r_mt exactly on it (index m) and two points past r_norman."""
    n_in = int(math.ceil(math.log(r_mt / R_START) / h)) + 1
    n_out = int(math.ceil(math.log(r_norman / r_mt) / h)) + 3
    x = math.log(r_mt) + h * (np.arange(n_in + n_out) - (n_in - 1))
    return np.exp(x), n_in - 1


def radial_solutions(r, m, V, Z, z, v_int, lmax, relativistic=False):
    """Regular and irregular solutions of the ground-state potential.

    r, m  : grid with r[m] = r_mt; V (nr,) the potential inside the muffin tin
            (values past m are ignored: the interstitial is flat at v_int)
    z     : (nz,) complex energies (Ha)
    Returns p (nz,), t (nz, nl), R (nz, nl, nr), H (nz, nl, nr), M_out (nz,),
    normalised R -> j_l(pr) + t_l h_l(pr), H -> h_l(pr) for r >= r_mt, so
    that r^2 (R H' - R' H) / M = i/(p M_out) everywhere and the Green's
    function is G_l(r, r') = -2 i p M_out R_l(r<) H_l(r>).

    relativistic: the scalar-relativistic equation inside the sphere (M =
    1 + (z - V)/2c^2, R = sqrt(M) r^-1/2 y), the relativistic dispersion
    outside (p^2 = 2 M_out (z - v_int)); non-relativistically M = 1."""
    z = np.asarray(z, complex)
    nz, nl, nr = z.size, lmax + 1, r.size
    h = math.log(r[1] / r[0])
    h2 = h * h
    T = z - v_int
    p = interstitial_momentum(T, relativistic)
    ls = np.arange(nl)
    inside = np.arange(nr) <= m
    if relativistic:
        Mo = 1.0 + T / (2.0 * C_LIGHT ** 2)
        w, wx, wxx = sr_derivatives(r, np.asarray(V, float), h)
        U, _, M = sr_coefficients(r[None, :], np.asarray(V, float)[None, :], z[:, None],
                                  w[None, :], wx[None, :], wxx[None, :], 0.0)
        # flat interstitial past r_mt: the free equation at the relativistic momentum
        U = np.where(inside[None, :], U, -(r * r)[None, :] * (p * p)[:, None])
        M = np.where(inside[None, :], M, Mo[:, None])
    else:
        Mo = np.ones(nz, complex)
        Vf = np.where(inside, V, v_int)
        U = 2.0 * (r * r)[None, :] * (Vf[None, :] - z[:, None])
        M = None
    G = ((ls + 0.5) ** 2)[None, :, None] + U[:, None, :]
    f = 1.0 - h2 * G / 12.0
    xs = np.log(r) - np.log(r[m])
    i0 = np.array([max(0, int(np.searchsorted(xs, -35.0 / (l + 0.5)))) for l in ls])
    i0 = np.minimum(i0, m - 8)
    # regular, outward to m+2
    y = np.zeros((nz, nl, nr), complex)
    for l in ls:
        a = i0[l]
        if relativistic:
            s = sr_start_exponent(G[:, l, a])
            y[:, l, a], y[:, l, a + 1] = np.exp(s * xs[a]), np.exp(s * xs[a + 1])
        else:
            corr = (1.0 - Z * r[[a, a + 1]] / (l + 1.0)) if l < 3 else np.ones(2)
            y[:, l, a] = np.exp((l + 0.5) * xs[a]) * corr[0]
            y[:, l, a + 1] = np.exp((l + 0.5) * xs[a + 1]) * corr[1]
    lo = int(i0.min())
    for i in range(lo + 1, m + 2):
        act = i0 < i
        yn = ((12.0 - 10.0 * f[:, :, i]) * y[:, :, i] - f[:, :, i - 1] * y[:, :, i - 1]) / f[:, :, i + 1]
        y[:, :, i + 1] = np.where(act[None, :], yn, y[:, :, i + 1])
    dy = (y[:, :, m - 2] - 8 * y[:, :, m - 1] + 8 * y[:, :, m + 1] - y[:, :, m + 2]) / (12 * h)
    LR = (dy / y[:, :, m] - 0.5) / r[m]
    if relativistic:
        # g = sqrt(M) r^-1/2 y; P and Q = r g'/2M continuous across r_mt
        Mi = 1.0 + (z - V[m]) / (2.0 * C_LIGHT ** 2)
        dM = -((wx[m] - w[m]) / r[m] ** 2) / (2.0 * C_LIGHT ** 2)
        LR = (LR + (dM / (2.0 * Mi))[:, None]) * (Mo / Mi)[:, None]
    t = t_from_logderiv(lmax, p * r[m], p, LR)
    # free waves from r_mt outwards
    xo = p[:, None] * r[None, m:]
    jo = np.moveaxis(bessel_j_all(lmax, xo), -1, 1)       # (nz, nl, nr-m)
    ho = np.moveaxis(hankel_all(lmax, xo), -1, 1)
    R = np.zeros((nz, nl, nr), complex)
    H = np.zeros((nz, nl, nr), complex)
    R[:, :, m:] = jo + t[:, :, None] * ho
    H[:, :, m:] = ho
    # interior: g = sqrt(M/M_out) r^-1/2 y (the M_out normalisation of y outside)
    rs = r ** -0.5
    sq = np.ones((nz, nr), complex) if M is None else np.sqrt(M / Mo[:, None])
    scale = R[:, :, m] / (y[:, :, m] * rs[m] * sq[:, None, m])
    R[:, :, :m] = y[:, :, :m] * (rs[None, :m] * sq[:, :m])[:, None, :] * scale[:, :, None]
    # irregular, inward from the free waves at m+2, m+1
    yi = np.zeros((nz, nl, nr), complex)
    yi[:, :, m + 2] = H[:, :, m + 2] * r[m + 2] ** 0.5
    yi[:, :, m + 1] = H[:, :, m + 1] * r[m + 1] ** 0.5
    for i in range(m + 1, lo, -1):
        yn = ((12.0 - 10.0 * f[:, :, i]) * yi[:, :, i] - f[:, :, i + 1] * yi[:, :, i + 1]) / f[:, :, i - 1]
        yi[:, :, i - 1] = yn
    H[:, :, :m] = yi[:, :, :m] * (rs[None, :m] * sq[:, :m])[:, None, :]
    # below each l's start the regular solution is negligible: no product there
    for l in ls:
        R[:, l, :i0[l]] = 0.0
        H[:, l, :i0[l]] = 0.0
    return p, t, R, H, Mo


# --------------------------------------------------------------------------
# the cluster Green's function at one site
# --------------------------------------------------------------------------

@dataclass
class SiteCluster:
    """The FMS cluster around one representative atom (local index 0)."""
    pos: np.ndarray            # (n, 3) bohr, centre at the origin
    types: np.ndarray          # (n,) potential type of each site
    pairs: np.ndarray          # (np, 2) ordered (k, m), k != m
    dist_idx: np.ndarray       # (np,) index into dists
    dists: np.ndarray          # unique distances
    D_f: np.ndarray            # (np, nLf, nLf) rotations, lmax_fms
    centre_out: np.ndarray     # (n-1,) sites k != 0
    D_o: np.ndarray            # (n-1, nLo, nLo) rotations for R_k - R_0, lmax_big
    D_i: np.ndarray            # (n-1, nLo, nLo) rotations for R_0 - R_k


def site_cluster(positions, site_type, centre, r_scf, lmax_fms, lmax_big):
    d = np.linalg.norm(positions - positions[centre], axis=1)
    idx = np.where(d <= r_scf + 1e-6)[0]
    idx = np.concatenate([[centre], idx[idx != centre]])
    pos = positions[idx] - positions[centre]
    n = len(idx)
    kk, mm = np.where(~np.eye(n, dtype=bool))
    vec = pos[kk] - pos[mm]
    dd = np.round(np.linalg.norm(vec, axis=1), 8)
    dists, dist_idx = np.unique(dd, return_inverse=True)
    others = np.arange(1, n)
    return SiteCluster(pos=pos, types=np.asarray(site_type)[idx], pairs=np.stack([kk, mm], 1),
                       dist_idx=dist_idx, dists=dists, D_f=wigner_blocks(lmax_fms, vec),
                       centre_out=others, D_o=wigner_blocks(lmax_big, pos[others]),
                       D_i=wigner_blocks(lmax_big, -pos[others]))


def _xp(gpu):
    if gpu:
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            return cp
        except Exception:
            pass
    return np


def site_trace(sc: SiteCluster, p, t_all, lmax_fms, lmax_out, gpu=True, chunk=24):
    """tr_l T^{00}(z), l = 0..lmax_out, at the centre of `sc`.

    p (nz,), t_all (ntype, nz, >= lmax_out+1) scattering amplitudes."""
    xp = _xp(gpu)
    nz, n = p.size, len(sc.pos)
    nLf, nLo = nlm(lmax_fms), nlm(lmax_out)
    lb = max(lmax_fms, lmax_out)
    lf, lo = l_of_index(lmax_fms), l_of_index(lmax_out)
    out = np.zeros((nz, lmax_out + 1), complex)
    kk, mm = sc.pairs[:, 0], sc.pairs[:, 1]
    D_f = xp.asarray(sc.D_f)
    D_o, D_i = xp.asarray(sc.D_o), xp.asarray(sc.D_i)
    rk = np.linalg.norm(sc.pos[sc.centre_out], axis=1)
    for s0 in range(0, nz, chunk):
        sl = slice(s0, min(nz, s0 + chunk))
        pz = p[sl]
        nc = pz.size
        # all-pair propagators at lmax_fms
        gz = xp.asarray(gz_batch(lmax_fms, pz[:, None] * sc.dists[None, :]))   # (nc, nu, nLf, nLf)
        Gp = D_f[None] @ gz[:, xp.asarray(sc.dist_idx)] @ D_f.conj().transpose(0, 2, 1)[None]
        tcol = xp.asarray(t_all[:, sl][:, :, lf])                                # (ntype, nc, nLf)
        tm = tcol[xp.asarray(sc.types[mm])].transpose(1, 0, 2)                   # (nc, np, nLf)
        A = xp.zeros((nc, n, n, nLf, nLf), complex)
        A[:, xp.asarray(kk), xp.asarray(mm)] = -Gp * tm[:, :, None, :]
        A = A.transpose(0, 1, 3, 2, 4).reshape(nc, n * nLf, n * nLf)
        A += xp.eye(n * nLf)[None]
        # propagators between the centre and every other site, to lmax_big
        gzc = xp.asarray(gz_batch(lb, pz[:, None] * rk[None, :]))               # (nc, n-1, nLb, nLb)
        G_k0 = D_o[None] @ gzc @ D_o.conj().transpose(0, 2, 1)[None]            # G(R_k - R_0)
        G_0k = D_i[None] @ gzc @ D_i.conj().transpose(0, 2, 1)[None]            # G(R_0 - R_k)
        Gin = xp.zeros((nc, n, nLf, nLo), complex)
        Gin[:, 1:] = G_k0[:, :, :nLf, :nLo]
        Gin = Gin.reshape(nc, n * nLf, nLo)
        Gout = xp.zeros((nc, nLo, n, nLf), complex)
        Gout[:, :, 1:] = G_0k[:, :, :nLo, :nLf].transpose(0, 2, 1, 3)
        Gout = Gout.reshape(nc, nLo, n * nLf)
        Y = xp.linalg.solve(A, Gin)                                              # (nc, n nLf, nLo)
        tvec = tcol[xp.asarray(sc.types)].transpose(1, 0, 2).reshape(nc, n * nLf)
        T = Gout @ (tvec[:, :, None] * Y)                                        # (nc, nLo, nLo)
        diag = xp.diagonal(T, axis1=1, axis2=2)                                  # (nc, nLo)
        dg = diag.get() if xp is not np else diag
        for l in range(lmax_out + 1):
            out[sl, l] = dg[:, lo == l].sum(1)
    return out


# --------------------------------------------------------------------------
# potentials from per-type densities
# --------------------------------------------------------------------------

@dataclass
class TypeState:
    label: str
    Z: int
    r: np.ndarray              # radial grid (r[m] = r_mt), to r_norman + 2 points
    m: int
    r_norman: float
    core: np.ndarray           # frozen core density on r
    n_val: float               # valence electrons this type holds when neutral
    atom_grid: LogGrid
    core_atom: np.ndarray      # frozen core density on the atom grid


def _core_split(atom):
    core = [o for o, e in atom.eps.items() if e < CORE_CUT]
    val = [o for o, e in atom.eps.items() if e >= CORE_CUT]
    if core and val and max(atom.eps[o] for o in core) > min(atom.eps[o] for o in val) - 0.3:
        raise ValueError(f"no clear core/valence gap for Z={atom.Z}: {atom.eps}")
    rho_c = sum(atom.occupations[o] * atom.orbital_density(o) for o in core) if core else 0 * atom.r
    n_core = sum(atom.occupations[o] for o in core)
    return rho_c, n_core


def _log_interp(r_new, r_old, f_old):
    return np.interp(np.log(r_new), np.log(r_old), f_old)


def potentials(cp0: ClusterPotential, states, drho_atom):
    """The overlapped-atom potentials plus the electrostatics of a change.

    drho_atom[c]: the change of type c's density (self-consistent minus
    overlapped atoms) inside its Norman sphere, on its atom grid, zero past
    r_norman. Every site carries the change of its type: its own spherical
    part, and at the other sites the spherical average about them (the field
    of its net charge included, so a site that gained or lost electrons
    shifts its neighbours' levels). With no change this returns the
    overlapped-atom potentials exactly, so the loop starts where the
    non-self-consistent engine stands. (Rebuilding from the Norman-truncated
    densities alone would move the mean electrostatic level by ~10 eV, which
    is a property of the truncation, not of the solid.)
    Returns (types with new rho, v_coul), v_int, v_int_coul, rho_int."""
    tabs = []
    phis = []
    for c, st in enumerate(states):
        g = st.atom_grid
        s = drho_atom[c]
        phi = hartree(g, s)
        phis.append(phi)
        tabs.append((g.r, g.cumulative(s * g.r), g.cumulative(phi * g.r)))
    pos = cp0.positions
    new_types = []
    for c, t in enumerate(cp0.types):
        rgrid = t.r
        g = states[c].atom_grid
        rho = t.rho + _log_interp(rgrid, g.r, drho_atom[c])
        vc = t.v_coul + _log_interp(rgrid, g.r, phis[c])
        for j in range(len(pos)):
            if j == t.rep:
                continue
            R = np.linalg.norm(pos[j] - pos[t.rep])
            sg, Frho, Fv = tabs[cp0.site_type[j]]
            rho = rho + PO._sphavg(rgrid, R, sg, Frho)
            vc = vc + PO._sphavg(rgrid, R, sg, Fv)
        new_types.append(replace(t, rho=np.maximum(rho, 1e-12), v_coul=vc))
    num_v = num_vc = num_rho = vol = 0.0
    for t in new_types:
        msk = (t.r >= t.r_mt) & (t.r <= t.r_norman)
        if msk.sum() < 3:
            continue
        rr = t.r[msk]
        w = 4.0 * np.pi * rr ** 2
        vgs = t.v_gs
        num_v += t.count * np.trapezoid(vgs[msk] * w, rr)
        num_vc += t.count * np.trapezoid(t.v_coul[msk] * w, rr)
        num_rho += t.count * np.trapezoid(t.rho[msk] * w, rr)
        vol += t.count * np.trapezoid(w, rr)
    return new_types, num_v / vol, num_vc / vol, num_rho / vol


# --------------------------------------------------------------------------
# the self-consistency loop
# --------------------------------------------------------------------------

@dataclass
class SCFResult:
    potential: ClusterPotential
    mu: float
    charges: dict              # label -> net charge of the Norman sphere (e, + = lost electrons)
    history: list = field(default_factory=list)
    converged: bool = False


def build(positions_A, species, absorber_core_hole="1s", folp=1.15, r_scf=4.5,
          lmax_fms=3, lmax_out=6, kT=0.005, n_pole=4, e_below=0.3, mix=0.05,
          history=6, tol_q=2e-4, tol_mu=5e-5, max_iter=60, neutrality="cluster",
          gpu=True, verbose=True, relativistic=False) -> SCFResult:
    """Self-consistent potentials for the cluster (atom 0 the absorber).

    r_scf      : radius of the FMS cluster around each representative atom (A)
    lmax_fms   : angular momentum carried by the scatterers
    lmax_out   : angular momentum of the single-site sum and returning waves
    kT         : electronic temperature of the Fermi smearing (Ha)
    e_below    : the contour starts this far below the interstitial level (Ha)
    neutrality : "cluster" (counts of every type in the absorber's r_scf
                 cluster, FEFF's convention) or "host" (the alloy at its
                 composition sets the Fermi level; the absorber screens as
                 it will)
    relativistic : scalar-relativistic atoms (frozen cores, start), valence
                 states and relativistic LDA exchange (`atom.py`)"""
    t_start = time.time()
    cp0 = PO.build(positions_A, species, absorber_core_hole, folp, relativistic=relativistic)
    pos = cp0.positions
    nt = len(cp0.types)
    states = []
    for c, t in enumerate(cp0.types):
        atom = atom_for(t.symbol, t.core_hole, relativistic=relativistic)
        rho_c, n_core = _core_split(atom)
        r, m = type_grid(t.r_mt, t.r_norman)
        states.append(TypeState(label=t.label, Z=t.Z, r=r, m=m, r_norman=t.r_norman,
                                core=_log_interp(r, atom.r, rho_c), n_val=t.Z - n_core,
                                atom_grid=atom.grid, core_atom=rho_c))
    clusters = [site_cluster(pos, cp0.site_type, t.rep, r_scf / BOHR, lmax_fms,
                             max(lmax_fms, lmax_out)) for t in cp0.types]
    d_abs = np.linalg.norm(pos, axis=1)
    in_scf = d_abs <= r_scf / BOHR + 1e-6
    if neutrality == "cluster":
        weight = np.array([np.sum(cp0.site_type[in_scf] == c) for c in range(nt)], float)
        weight[0] = 1.0
    elif neutrality == "host":
        weight = np.array([0.0] + [t.count for t in cp0.types[1:]], float)
    else:
        raise ValueError(neutrality)
    n_target = float(np.sum(weight * np.array([s.n_val for s in states])))

    # input valence densities on each type grid: the overlapped-atom start
    def val_from_types(types):
        out = []
        for c, t in enumerate(types):
            st = states[c]
            rho = np.exp(_log_interp(st.r, t.r, np.log(np.maximum(t.rho, 1e-300))))
            out.append(np.where(st.r <= st.r_norman, np.maximum(rho - st.core, 0.0), 0.0))
        return out

    def ground_potential(types):
        Vs = []
        for c, t in enumerate(types):
            st = states[c]
            rv = _log_interp(st.r, t.r, t.r * t.v_coul)
            rho = np.exp(_log_interp(st.r, t.r, np.log(np.maximum(t.rho, 1e-300))))
            Vs.append(rv / st.r + lda_xc(rho, relativistic)[1])
        return Vs

    def charge(c, rho):
        st = states[c]
        from scipy.integrate import cumulative_simpson
        q = cumulative_simpson(4.0 * np.pi * st.r ** 3 * rho, dx=math.log(st.r[1] / st.r[0]), initial=0.0)
        return float(np.interp(st.r_norman, st.r, q))

    ll = np.arange(lmax_out + 1)

    def green(Vs, v_int, z):
        """Spherically averaged G(r, r; z) at every type's centre: [(nz, nr)]."""
        sols = [radial_solutions(st.r, st.m, Vs[c], st.Z, z, v_int, lmax_out, relativistic)
                for c, st in enumerate(states)]
        p, Mo = sols[0][0], sols[0][4]
        t_all = np.stack([s[1] for s in sols])                     # (ntype, nz, lmax_out+1)
        out = []
        for c, st in enumerate(states):
            trT = site_trace(clusters[c], p, t_all, lmax_fms, lmax_out, gpu=gpu)
            _, _, R, H, _ = sols[c]
            out.append((-2j * p * Mo / (4.0 * np.pi))[:, None] * np.einsum(
                "zlr,zlr->zr", R, (2 * ll + 1)[None, :, None] * H + R * trT[:, :, None]))
        return out

    def solve_mu(Vs, v_int, mu0, half_width=0.2, n_window=96):
        """Fermi level of charge neutrality near mu0.

        G is computed once on fixed points: the vertical leg, the line
        Im z = Y up to the window, and a Gauss-Legendre rule across the window
        [mu0 -+ half_width] (fine enough to resolve f there). A trial Fermi
        level then only reweights those points by f and adds its own poles.
        A root outside the window is clamped to its edge; the next iteration
        re-centres."""
        Y = 2.0 * n_pole * math.pi * kT
        e_b = v_int - e_below
        lo, hi = mu0 - half_width, mu0 + half_width
        lo = max(lo, e_b + 0.05)
        zs, base = [], []
        x, w = np.polynomial.legendre.leggauss(8)
        zs.append(e_b + 1j * 0.5 * Y * (x + 1.0)); base.append(1j * 0.5 * Y * w)
        x, w = np.polynomial.legendre.leggauss(24)
        zs.append(0.5 * (lo - e_b) * (x + 1.0) + e_b + 1j * Y); base.append(0.5 * (lo - e_b) * w)
        x, w = np.polynomial.legendre.leggauss(n_window)
        zs.append(0.5 * (hi - lo) * (x + 1.0) + lo + 1j * Y); base.append(0.5 * (hi - lo) * w)
        z_fix, w_fix = np.concatenate(zs), np.concatenate(base)
        G_fix = green(Vs, v_int, z_fix)
        n_eval = [0]

        def resid(mu):
            zp = mu + 1j * (2 * np.arange(n_pole) + 1) * math.pi * kT
            G_p = green(Vs, v_int, zp)
            n_eval[0] += 1
            wf = w_fix * fermi(z_fix, mu, kT)
            rhos, qs = [], []
            for c, st in enumerate(states):
                I = np.einsum("z,zr->r", wf, G_fix[c]) + (-2j * math.pi * kT) * G_p[c].sum(0)
                rho = np.where(st.r <= st.r_norman, -(2.0 / np.pi) * I.imag, 0.0)
                rhos.append(rho)
                qs.append(charge(c, rho))
            q = np.array(qs)
            return float(np.sum(weight * q) - n_target), rhos, q

        a_lim, b_lim = lo + 15 * kT, hi - 15 * kT
        mu0 = min(max(mu0, a_lim), b_lim)
        f0, _, _ = resid(mu0)
        mu1 = min(max(mu0 - np.sign(f0) * 0.02, a_lim), b_lim)
        f1, r1, q1 = resid(mu1)
        while np.sign(f1) == np.sign(f0):
            slope = (f1 - f0) / (mu1 - mu0) if mu1 != mu0 else 0.0
            nxt = mu1 - f1 / slope if slope > 0 else mu1 - np.sign(f1) * 0.05
            nxt = min(max(nxt, a_lim), b_lim)
            if nxt == mu1:                                      # clamped at the window edge
                return mu1, r1, q1, n_eval[0], True
            mu0, f0 = mu1, f1
            mu1 = nxt
            f1, r1, q1 = resid(mu1)
        a, fa, b, fb, rb, qb = mu0, f0, mu1, f1, r1, q1
        for _ in range(30):
            if abs(fb) < 1e-6 * n_target or abs(b - a) < 1e-7:
                break
            c = b - fb * (b - a) / (fb - fa)                   # regula falsi (Illinois)
            fc, rc, qc = resid(c)
            if np.sign(fc) == np.sign(fb):
                fa *= 0.5
            else:
                a, fa = b, fb
            b, fb, rb, qb = c, fc, rc, qc
        return b, rb, qb, n_eval[0], False

    types = cp0.types
    v_int = cp0.v_int
    v_int_coul, rho_int = cp0.v_int_coul, cp0.rho_int
    mu = cp0.mu
    rho_in = val_from_types(types)
    hist_x, hist_f = [], []
    hist = []
    converged = False
    for it in range(1, max_iter + 1):
        Vs = ground_potential(types)
        mu_try = mu
        for _ in range(6):                      # re-centre the window until the root is inside
            mu_new, rho_out, q_out, n_eval, clamped = solve_mu(Vs, v_int, mu_try)
            if not clamped:
                break
            mu_try = mu_new
        q_in = np.array([charge(c, rho_in[c]) for c in range(nt)])
        net = np.array([s.n_val for s in states]) - q_out
        dq = np.max(np.abs(q_out - q_in))
        dmu = abs(mu_new - mu)
        hist.append(dict(it=it, mu_ev=mu_new * HARTREE, v_int_ev=v_int * HARTREE,
                         net=net.tolist(), dq=float(dq), n_eval=n_eval))
        if verbose:
            print(f"  scf {it:2d}: mu {mu_new * HARTREE:8.3f} eV  Vint {v_int * HARTREE:8.3f} eV  "
                  f"rs {float((3 / (4 * np.pi * rho_int)) ** (1 / 3)):.3f}  net "
                  + " ".join(f"{s.label}{x:+.3f}" for s, x in zip(states, net))
                  + f"  |dq| {dq:.1e}  ({n_eval} trial mu{', clamped' if clamped else ''}, "
                  f"{time.time() - t_start:.0f}s)", flush=True)
        mu = mu_new
        if it > 1 and dq < tol_q and dmu < tol_mu and not clamped:
            converged = True
            break
        # Anderson mixing of the valence densities (charge-weighted vectors)
        wvec = [4.0 * np.pi * s.r ** 3 for s in states]
        x = np.concatenate([rho_in[c] * wvec[c] for c in range(nt)])
        fres = np.concatenate([(rho_out[c] - rho_in[c]) * wvec[c] for c in range(nt)])
        hist_x.append(x)
        hist_f.append(fres)
        if len(hist_x) > history:
            hist_x.pop(0)
            hist_f.pop(0)
        k = len(hist_x)
        if k >= 2:
            dF = [hist_f[i] - hist_f[-1] for i in range(k - 1)]
            Amat = np.array([[dF[i] @ dF[j] for j in range(k - 1)] for i in range(k - 1)])
            bvec = np.array([dF[i] @ hist_f[-1] for i in range(k - 1)])
            try:
                cc = np.linalg.solve(Amat + 1e-12 * np.trace(Amat) * np.eye(k - 1), -bvec)
            except np.linalg.LinAlgError:
                cc = np.zeros(k - 1)
            xb = hist_x[-1] + sum(cc[i] * (hist_x[i] - hist_x[-1]) for i in range(k - 1))
            fb = hist_f[-1] + sum(cc[i] * dF[i] for i in range(k - 1))
            xn = xb + mix * fb
        else:
            xn = x + mix * fres
        off = 0
        for c in range(nt):
            nr = states[c].r.size
            rho_in[c] = np.maximum(xn[off:off + nr] / wvec[c], 0.0)
            off += nr
        # new potentials: the change of core + mixed valence from the
        # overlapped-atom density, inside each Norman sphere
        drho = []
        for c, st in enumerate(states):
            g = st.atom_grid
            val = _log_interp(g.r, st.r, rho_in[c])
            ref = np.exp(_log_interp(g.r, cp0.types[c].r, np.log(np.maximum(cp0.types[c].rho, 1e-300))))
            drho.append(np.where(g.r <= st.r_norman, st.core_atom + val - ref, 0.0))
        types, v_int, v_int_coul, rho_int = potentials(cp0, states, drho)
    cp = replace(cp0, types=types, v_int=v_int, v_int_coul=v_int_coul, rho_int=rho_int, mu=mu,
                 meta=dict(cp0.meta, scf=True, scf_iterations=len(hist), scf_converged=converged,
                           r_scf_A=r_scf, lmax_fms=lmax_fms, lmax_out=lmax_out, kT=kT,
                           neutrality=neutrality, relativistic=bool(relativistic)))
    charges = {s.label: float(x) for s, x in zip(states, hist[-1]["net"])}
    return SCFResult(potential=cp, mu=mu, charges=charges, history=hist, converged=converged)
