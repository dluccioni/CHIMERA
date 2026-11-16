"""Overlapped-atom muffin-tin potentials for a cluster (Mattheiss construction).

For every potential type - the absorber with its core hole, and each species -
a representative atom of the cluster is taken (the absorber itself, and the
atom of each species nearest to it, as FEFF does). Its charge density and
Coulomb potential are its own free-atom values plus the spherical average,
about its nucleus, of every other atom's:

    <f>(r) = 1/(2 r R) int_{|R-r|}^{R+r} f(s) s ds          (neighbour at R)

From those come
  * the Norman radius r_N: the sphere holding the atom's own charge Z;
  * the touching radius: where the atom's potential meets its nearest
    neighbour's along the bond;
  * the muffin-tin radius r_mt: the touching radius scaled by an overlap
    factor (FEFF's default is 15%), capped below the Norman radius;
  * the interstitial level: the volume average of the potential, and of the
    density, over the shells between r_mt and r_N, weighted by how many atoms
    of each type the cluster holds;
  * the Fermi level of a non-self-consistent calculation, mu = V_int + kF^2/2,
    the electron gas estimate (FEFF's SCF moves it; `mu` can be overridden).

Hartree atomic units; positions come in Angstrom.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .atom import Atom, LogGrid, atom_for, lda_xc, rs_of
from ..scattering import Z_TABLE

BOHR = 0.529177210903          # Angstrom
HARTREE = 27.211386245988      # eV


@dataclass
class PotentialType:
    label: str                 # "Cr*" for the absorber, else the symbol
    symbol: str
    Z: int
    core_hole: str | None
    rep: int                   # index of the representative atom in the cluster
    count: int                 # atoms of this type in the cluster
    r: np.ndarray = field(repr=False, default=None)        # radial grid (bohr)
    rho: np.ndarray = field(repr=False, default=None)      # overlapped density
    v_coul: np.ndarray = field(repr=False, default=None)   # overlapped Coulomb
    r_norman: float = 0.0
    r_touch: float = 0.0
    r_mt: float = 0.0
    rel_xc: bool = False       # MacDonald-Vosko relativistic exchange (relativistic potentials)

    @property
    def v_gs(self):
        """Ground-state potential: Coulomb + LDA exchange-correlation."""
        return self.v_coul + lda_xc(self.rho, self.rel_xc)[1]


@dataclass
class ClusterPotential:
    positions: np.ndarray      # (n, 3) bohr, absorber first at the origin
    species: list              # symbol per atom
    types: list                # [PotentialType], index 0 = absorber
    site_type: np.ndarray      # potential type of every atom
    v_int: float               # interstitial ground-state level (Ha)
    v_int_coul: float          # its Coulomb part
    rho_int: float             # interstitial density
    mu: float                  # Fermi level (Ha)
    folp: float
    meta: dict = field(default_factory=dict)

    @property
    def rs_int(self):
        return float(rs_of(self.rho_int))

    @property
    def kf_int(self):
        return float((3.0 * np.pi ** 2 * self.rho_int) ** (1.0 / 3.0))

    def summary(self):
        lines = []
        for t in self.types:
            lines.append(f"  {t.label:4s} Z={t.Z:3d}  Rmt={t.r_mt * BOHR:6.3f}  "
                         f"Rnm={t.r_norman * BOHR:6.3f}  Rtouch={t.r_touch * BOHR:6.3f} A"
                         f"  n={t.count}")
        lines.append(f"  Vint={self.v_int * HARTREE:8.3f} eV (Coulomb {self.v_int_coul * HARTREE:8.3f})"
                     f"  Rs_int={self.rs_int:6.3f}  kf={self.kf_int / BOHR:6.3f} A^-1"
                     f"  mu={self.mu * HARTREE:8.3f} eV")
        return "\n".join(lines)


def _sphavg_tables(atom: Atom):
    """F(s) = int_0^s f(t) t dt for the density and the Coulomb potential."""
    g = atom.grid
    r = g.r
    Frho = g.cumulative(atom.rho * r)
    Fv = g.cumulative(atom.v_coulomb * r)       # integrand -Z + r V_H: finite
    return r, Frho, Fv


def _sphavg(r_eval, R, s_grid, F):
    """Spherical average about the origin of f(|x - R|) at radii r_eval."""
    hi = np.interp(R + r_eval, s_grid, F)
    lo = np.interp(np.abs(R - r_eval), s_grid, F)
    return (hi - lo) / (2.0 * r_eval * R)


def build(positions_A, species, absorber_core_hole="1s", folp=1.15,
          mt_cap=0.97, mu=None, grid: LogGrid | None = None,
          r_mt_override=None, relativistic: bool = False) -> ClusterPotential:
    """Muffin-tin potentials for a cluster.

    positions_A : (n, 3) Angstrom; atom 0 is the absorber (moved to the origin)
    species     : symbol per atom (the absorber's included)
    folp        : overlap factor applied to the touching radius
    mt_cap      : r_mt is kept at or below mt_cap * r_Norman
    mu          : Fermi level override (Ha); default is the electron-gas value
    r_mt_override : {label: radius in Angstrom} to impose muffin-tin radii
    relativistic : scalar-relativistic atoms with relativistic (MacDonald-Vosko)
                  exchange; recorded in meta, where `phaseshift` and `scf`
                  read it to solve the scattering states the same way
    """
    pos = np.asarray(positions_A, float) / BOHR
    pos = pos - pos[0]
    species = list(species)
    n = len(pos)
    grid = grid or LogGrid()
    # atoms: ground state for every species, core-hole atom at the absorber
    gs = {s: atom_for(s, relativistic=relativistic) for s in sorted(set(species))}
    abs_atom = (atom_for(species[0], absorber_core_hole, relativistic=relativistic)
                if absorber_core_hole else gs[species[0]])
    tables = {s: _sphavg_tables(a) for s, a in gs.items()}
    tab_abs = _sphavg_tables(abs_atom)

    def atom_at(i):
        return abs_atom if i == 0 else gs[species[i]]

    def table_at(i):
        return tab_abs if i == 0 else tables[species[i]]

    # potential types
    d0 = np.linalg.norm(pos, axis=1)
    types = [PotentialType(label=species[0] + ("*" if absorber_core_hole else ""),
                           symbol=species[0], Z=Z_TABLE[species[0]],
                           core_hole=absorber_core_hole, rep=0, count=1, rel_xc=relativistic)]
    site_type = np.zeros(n, int)
    for s in sorted(set(species), key=lambda s: Z_TABLE[s]):
        idx = [i for i in range(1, n) if species[i] == s]
        if not idx:
            continue
        rep = min(idx, key=lambda i: d0[i])
        types.append(PotentialType(label=s, symbol=s, Z=Z_TABLE[s], core_hole=None,
                                   rep=rep, count=len(idx), rel_xc=relativistic))
        site_type[idx] = len(types) - 1

    # radial grid for the overlapped potentials: up to 2x the nearest spacing
    dmin = np.min([np.linalg.norm(pos[i] - pos[j]) for i in range(min(n, 13))
                   for j in range(n) if j != i])
    rgrid = grid.r[grid.r <= 1.2 * dmin]
    for t in types:
        i = t.rep
        a = atom_at(i)
        rho = np.interp(rgrid, a.r, a.rho)
        vc = np.interp(rgrid, a.r, a.v_coulomb)
        for j in range(n):
            if j == i:
                continue
            R = np.linalg.norm(pos[j] - pos[i])
            s_grid, Frho, Fv = table_at(j)
            rho = rho + _sphavg(rgrid, R, s_grid, Frho)
            vc = vc + _sphavg(rgrid, R, s_grid, Fv)
        t.r, t.rho, t.v_coul = rgrid, rho, vc
        q = np.cumsum(np.gradient(rgrid) * 4.0 * np.pi * rgrid ** 2 * rho)
        # accurate charge inside r: Simpson on the log grid
        from scipy.integrate import cumulative_simpson
        q = cumulative_simpson(4.0 * np.pi * rgrid ** 3 * rho, dx=grid.h, initial=0.0)
        t.r_norman = float(np.interp(t.Z, q, rgrid))

    # touching radii: equal ground-state potential along the nearest bond
    for t in types:
        i = t.rep
        dist = np.linalg.norm(pos - pos[i], axis=1)
        dist[i] = np.inf
        dnn = dist.min()
        best = np.inf
        for j in np.where(dist < 1.05 * dnn)[0]:
            tj = types[0] if j == 0 else types[site_type[j]]
            vi, vj = t.v_gs, tj.v_gs
            d = dist[j]
            xs = np.linspace(0.3 * d, 0.7 * d, 2001)
            diff = np.interp(xs, rgrid, vi) - np.interp(d - xs, rgrid, vj)
            k = np.where(np.diff(np.sign(diff)) != 0)[0]
            x = xs[k[0]] if k.size else 0.5 * d
            best = min(best, x)
        t.r_touch = float(best)
        t.r_mt = float(min(folp * t.r_touch, mt_cap * t.r_norman))
    if r_mt_override:
        for t in types:
            if t.label in r_mt_override:
                t.r_mt = r_mt_override[t.label] / BOHR

    # interstitial: volume average over r_mt < r < r_N, weighted by counts
    num_v = num_vc = num_rho = vol = 0.0
    for t in types:
        m = (t.r >= t.r_mt) & (t.r <= t.r_norman)
        if m.sum() < 3:
            continue
        rr = t.r[m]
        w = 4.0 * np.pi * rr ** 2
        def integ(f):
            return np.trapezoid(f * w, rr)
        dv = integ(np.ones_like(rr))
        num_v += t.count * integ(t.v_gs[m])
        num_vc += t.count * integ(t.v_coul[m])
        num_rho += t.count * integ(t.rho[m])
        vol += t.count * dv
    v_int, v_int_coul, rho_int = num_v / vol, num_vc / vol, num_rho / vol
    kf = (3.0 * np.pi ** 2 * rho_int) ** (1.0 / 3.0)
    # electron-gas Fermi level, written so that the interstitial momentum of an
    # electron at mu is exactly kF: Coulomb level + v_xc(rho_int) + kF^2/2
    mu_eg = v_int_coul + float(lda_xc(rho_int, relativistic)[1]) + 0.5 * kf ** 2
    cp = ClusterPotential(positions=pos, species=species, types=types,
                          site_type=site_type, v_int=v_int, v_int_coul=v_int_coul,
                          rho_int=rho_int, mu=mu if mu is not None else mu_eg,
                          folp=folp, meta=dict(mu_electron_gas=mu_eg, mt_cap=mt_cap,
                                               relativistic=bool(relativistic)))
    return cp
