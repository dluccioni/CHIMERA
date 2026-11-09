"""Realistic synthetic multi-edge EXAFS from atomistic snapshots.

What makes the data realistic, and different from the model CHIMERA fits:

* chemical order from Monte Carlo on effective pair interactions (two shells)
  at an equilibration temperature - the order is whatever that Hamiltonian
  produces, measured afterwards with the same Warren-Cowley definition the
  fit reports;
* static and thermal disorder from Langevin molecular dynamics with Morse
  nearest-neighbour bonds (pair-dependent bond lengths, anharmonic, so the
  first-shell distribution is skewed and expands with temperature) and
  harmonic second-neighbour springs;
* single scattering for every pair in every snapshot, exact curved wave,
  from distance histograms on a 0.002 A grid;
* multiple scattering computed path by path on the actual distorted
  geometry of sampled absorbers (3- and 4-leg paths, species and angles as
  they are), not from a frozen cluster with Debye-Waller factors.

The fit, by contrast, uses tabulated scattering, Gaussian (or cumulant)
disorder, pair counts on the ideal lattice and a composition-averaged
multiple-scattering term. Every one of those is an approximation the
synthetic data does not share.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..lattice import CrystalSupercell, warren_cowley
from ..scattering import Z_TABLE
from . import cluster as CL
from . import paths as PA
from . import pathsum as PSM
from . import phaseshift as PS
from . import potential as PO
from .potential import BOHR
from .selfenergy import GAMMA_K

AMU = {"Al": 26.98, "Ti": 47.87, "V": 50.94, "Cr": 52.00, "Mn": 54.94, "Fe": 55.85,
       "Co": 58.93, "Ni": 58.69, "Cu": 63.55, "Zn": 65.38, "Nb": 92.91, "Mo": 95.95,
       "Pd": 106.42, "Ag": 107.87, "W": 183.84, "Pt": 195.08, "Au": 196.97}
KB_EV = 8.617333262e-5
ACC = 0.0096485332          # (eV/A)/amu -> A/fs^2


@dataclass
class AlloyModel:
    elements: tuple
    composition: tuple
    lattice: str = "fcc"
    epi: np.ndarray = None          # (nshell, nsp, nsp) eV, symmetric
    t_order: float = 1000.0         # K: temperature the chemical order is equilibrated at
    bond_offset: np.ndarray = None  # (nsp, nsp) fractional NN bond length offsets
    morse_D: float = 0.45           # eV
    morse_a: float = 1.55           # A^-1
    k2: float = 0.35                # eV/A^2, second-neighbour springs
    meta: dict = field(default_factory=dict)

    @property
    def nsp(self):
        return len(self.elements)


# ---------------------------------------------------------------------------
# chemical order
# ---------------------------------------------------------------------------

def shell_neighbours(cell: CrystalSupercell, nshell):
    """Per shell s, an (natoms, z_s) array of neighbour indices."""
    radii, _ = cell.shell_radii(cell.a0 * 2.0)
    rmax = radii[nshell - 1] + 0.05
    i, j, sh, off = cell.neighbour_list(rmax)
    d = np.linalg.norm(cell.ideal[j] + sh - cell.ideal[i], axis=1)
    out = []
    for s in range(nshell):
        m = np.abs(d - radii[s]) < 0.05
        z = int(m.sum() // cell.natoms)
        out.append(j[m].reshape(cell.natoms, z))
    return out, radii[:nshell]


def order_monte_carlo(cell, species, epi, T, sweeps, rng, verbose=False):
    """Swap Metropolis on E = sum_shells sum_bonds V_s(a, b). Plain Python on
    scalars: ~10 microseconds a proposal."""
    nshell = epi.shape[0]
    nbrs, _ = shell_neighbours(cell, nshell)
    nb = [n.tolist() for n in nbrs]
    V = epi.tolist()
    t = species.astype(int).tolist()
    n = len(t)
    beta = 1.0 / (KB_EV * T)
    nprop = int(sweeps * n)
    I = rng.integers(0, n, nprop).tolist()
    J = rng.integers(0, n, nprop).tolist()
    U = np.log(rng.random(nprop)).tolist()
    acc = 0
    for q in range(nprop):
        i, j = I[q], J[q]
        a, b = t[i], t[j]
        if a == b:
            continue
        dE = 0.0
        for s in range(nshell):
            Vs = V[s]
            Va, Vb = Vs[a], Vs[b]
            for m in nb[s][i]:
                if m != j:
                    c = t[m]
                    dE += Vb[c] - Va[c]
            for m in nb[s][j]:
                if m != i:
                    c = t[m]
                    dE += Va[c] - Vb[c]
        if dE <= 0.0 or -beta * dE > U[q]:
            t[i], t[j] = b, a
            acc += 1
    if verbose:
        print(f"    MC: {nprop} proposals, acceptance {acc / max(1, nprop):.3f}")
    return np.array(t, np.int32)


def warren_cowley_shells(cell, species, nsp, nshell=3):
    radii, _ = cell.shell_radii(cell.a0 * 2.0)
    i, j, sh, off = cell.neighbour_list(radii[nshell - 1] + 0.05)
    comp = np.bincount(species, minlength=nsp) / species.size
    return np.array([warren_cowley(species, i, j, sh, cell, radii[s], tol=0.05, comp=comp,
                                   nsp=nsp) for s in range(nshell)])


# ---------------------------------------------------------------------------
# displacements: Morse NN + harmonic NNN, Langevin dynamics
# ---------------------------------------------------------------------------

def _xp():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return cp
    except Exception:
        return np


def molecular_dynamics(cell, species, model: AlloyModel, T, n_snap, rng, dt=2.0,
                       equil=3000, every=300, gamma=0.01, verbose=False):
    """Snapshots of positions (A) at temperature T. Returns (n_snap, N, 3) numpy."""
    xp = _xp()
    nbrs, radii = shell_neighbours(cell, 2)
    N = cell.natoms
    # bond vectors in the ideal lattice (fixed periodic shifts)
    ideal = cell.ideal
    def ideal_vec(nb):
        d = ideal[nb] - ideal[:, None, :]
        flat = cell.min_image(d.reshape(-1, 3)).reshape(d.shape)
        return flat
    v1, v2 = ideal_vec(nbrs[0]), ideal_vec(nbrs[1])
    d_nn = radii[0]
    off = model.bond_offset if model.bond_offset is not None else np.zeros((model.nsp, model.nsp))
    r0 = d_nn * (1.0 + off[species[:, None], species[nbrs[0]]])
    r02 = np.linalg.norm(v2, axis=2)
    mass = np.array([AMU[model.elements[s]] for s in species])
    u = xp.zeros((N, 3))
    vel = xp.asarray(rng.normal(size=(N, 3)) * np.sqrt(KB_EV * T / mass[:, None] * ACC))
    n1, n2 = xp.asarray(nbrs[0]), xp.asarray(nbrs[1])
    V1, V2 = xp.asarray(v1), xp.asarray(v2)
    R0, R02 = xp.asarray(r0), xp.asarray(r02)
    M = xp.asarray(mass)[:, None]
    D, a, k2 = model.morse_D, model.morse_a, model.k2
    gen = xp.random.default_rng(int(rng.integers(1 << 31))) if xp is not np else rng

    def force(u):
        d1 = V1 + u[n1] - u[:, None, :]
        r1 = xp.sqrt((d1 * d1).sum(-1))
        e = xp.exp(-a * (r1 - R0))
        f1 = (2.0 * D * a * (1.0 - e) * e / r1)[..., None] * d1
        d2 = V2 + u[n2] - u[:, None, :]
        r2 = xp.sqrt((d2 * d2).sum(-1))
        f2 = (k2 * (r2 - R02) / r2)[..., None] * d2
        return f1.sum(1) + f2.sum(1)

    # BAOAB Langevin
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1 - c1 ** 2) * KB_EV * T * ACC / mass)[:, None]
    C2 = xp.asarray(c2)
    F = force(u)
    snaps = []
    nsteps = equil + n_snap * every
    for step in range(1, nsteps + 1):
        vel = vel + 0.5 * dt * ACC * F / M
        u = u + 0.5 * dt * vel
        noise = gen.standard_normal((N, 3)) if xp is not np else rng.normal(size=(N, 3))
        vel = c1 * vel + C2 * noise
        u = u + 0.5 * dt * vel
        u = u - u.mean(0)
        F = force(u)
        vel = vel + 0.5 * dt * ACC * F / M
        if step > equil and (step - equil) % every == 0:
            snaps.append((xp.asarray(ideal) + u).get() if xp is not np else ideal + u)
    if verbose:
        ke = float((0.5 * M * vel * vel).sum() / ACC) if xp is np else float((0.5 * M * vel * vel).sum().get() / ACC)
        print(f"    MD: {nsteps} steps, kinetic T {2 * ke / (3 * N * KB_EV):.0f} K (target {T:.0f})")
    return np.array(snaps)


# ---------------------------------------------------------------------------
# scattering for each absorber species
# ---------------------------------------------------------------------------

def absorber_phases(elements, composition, lattice, d_nn, k_grid=None, model="hl", seed=0):
    """Phase shifts for every absorber species, from a random alloy cluster at
    the given spacing. Returns {element: (PhaseShifts, {symbol: type index})}."""
    out = {}
    for a in elements:
        pts = CL.lattice_cluster(d_nn, 6.5, lattice)
        sp = CL.alloy_species(pts, a, list(elements), d_nn, composition, seed)
        cp_ = PO.build(pts, sp)
        ph = PS.phase_shifts(cp_, k_grid, gamma_ev=GAMMA_K.get(a, 1.0), model=model)
        typ = {t.symbol: i for i, t in enumerate(cp_.types) if i > 0}
        out[a] = (ph, typ)
    return out


def ss_envelopes(ph, type_of, elements, r_grid_A, precision="float"):
    """Single-scattering chi~(R, k) for each scatterer species on an R grid,
    stored as the smooth envelope A = chi~ R^2 e^{-2ikR} (complex, (nsp, nR, nE))."""
    k = ph.k
    out = np.zeros((len(elements), r_grid_A.size, k.size), complex)
    lmax = PSM.lmax_per_energy(ph, list(type_of.values()))
    for b, el in enumerate(elements):
        P = r_grid_A.size
        nsc = np.ones(P, np.int32)
        typ = np.full((P, 1), type_of[el], np.int32)
        legs = np.zeros((P, 2, 3))
        legs[:, 0, 2] = r_grid_A / BOHR
        legs[:, 1, 2] = -r_grid_A / BOHR
        lens, eul = PSM.frames_batch(legs, nsc)
        tr = PSM.path_sum_packed(nsc, typ, lens, eul, ph, lmax, precision)
        chit = (1j / 3.0) * ph.S[0, :, 1][None, :] * tr
        out[b] = chit * r_grid_A[:, None] ** 2 * np.exp(-2j * k[None, :] * r_grid_A[:, None])
    return out


def pair_histograms(snaps, cell, species, nsp, rmax, dr=0.002, rmin=1.8):
    """Linear-weight histograms of pair distances: H[a, b, bin], per absorber
    of species a (averaged over absorbers and snapshots)."""
    i, j, sh, off = cell.neighbour_list(rmax + 0.6)
    edges = np.arange(rmin, rmax + dr, dr)
    H = np.zeros((nsp, nsp, edges.size))
    na = np.bincount(species, minlength=nsp).astype(float)
    a_s, b_s = species[i], species[j]
    for X in snaps:
        u = X - cell.ideal
        d = cell.ideal[j] + sh - cell.ideal[i] + u[j] - u[i]
        r = np.sqrt((d * d).sum(1))
        m = (r >= rmin) & (r < rmax - dr)
        x = (r[m] - rmin) / dr
        k0 = np.floor(x).astype(int)
        w1 = x - k0
        idx = (a_s[m] * nsp + b_s[m]) * edges.size
        flat = np.bincount(idx + k0, weights=1.0 - w1, minlength=nsp * nsp * edges.size)
        flat += np.bincount(idx + k0 + 1, weights=w1, minlength=nsp * nsp * edges.size)
        H += flat.reshape(nsp, nsp, edges.size)
    H /= (len(snaps) * na)[:, None, None]
    return edges, H


def chi_single_scattering(edges, H, env, r_grid, k):
    """chi~_SS(k) per absorber species (complex) from histograms and envelopes."""
    nsp = H.shape[0]
    out = np.zeros((nsp, k.size), complex)
    for a in range(nsp):
        for b in range(nsp):
            h = H[a, b]
            nz = h > 0
            if not nz.any():
                continue
            R = edges[nz]
            E = env[a][b]                                   # (nR, nE)
            Ai = np.empty((R.size, k.size), complex)
            for q in range(k.size):
                Ai[:, q] = np.interp(R, r_grid, E[:, q].real) + 1j * np.interp(R, r_grid, E[:, q].imag)
            out[a] += ((h[nz] / R ** 2)[:, None] * Ai * np.exp(2j * k[None, :] * R[:, None])).sum(0)
    return out


# ---------------------------------------------------------------------------
# multiple scattering, exact on the snapshot geometry
# ---------------------------------------------------------------------------

def ms_templates(lattice, a0, rmax_eff, nleg=4):
    """MS path templates on the ideal lattice, as lattice-offset sequences from
    the absorber in units of the conventional lattice constant a0 (exact
    multiples of 1/2 for fcc), so they transfer to any compression; time
    reverses merged, absorber-in-the-middle kept."""
    d_nn = a0 / CL.A0_PER_NN[lattice]
    pts = CL.lattice_cluster(d_nn, rmax_eff + 1.0, lattice)
    groups = PA.enumerate_paths(pts, np.zeros(len(pts), int), rmax_eff, nleg, group=False)
    tmpl = []
    for g in groups:
        if g.nleg < 3:
            continue
        seq = g.members[0]
        tmpl.append((pts[list(seq)] / a0, len(g.members), [s == 0 for s in seq]))
    return tmpl


def chi_multiple_scattering(snaps, cell, species, elements, phases, templates, absorbers_per_snap,
                            rng, energies_step=2, precision="float", verbose=False):
    """chi~_MS(k) per absorber species, averaged over sampled absorbers."""
    import cupy as cp
    nsp = len(elements)
    N = cell.natoms
    # lookup: ideal offset -> atom, per absorber, through the neighbour list
    a0 = cell.a0
    templates = [(o * a0, deg, ab) for o, deg, ab in templates]
    rmax = max(np.linalg.norm(t[0], axis=1).max() for t in templates) + 0.1
    i_n, j_n, sh, off = cell.neighbour_list(rmax)
    offs = cell.ideal[j_n] + sh - cell.ideal[i_n]
    # key on lattice coordinates (offset / a0), robust to rounding of a0
    keyof = lambda v: tuple(np.round(v / a0 * 1000).astype(np.int64))
    k = None
    out = {}
    for a, el in enumerate(elements):
        ph, type_of = phases[el]
        k = ph.k
        ie = np.arange(0, k.size, energies_step)
        if ie[-1] != k.size - 1:
            ie = np.append(ie, k.size - 1)
        tvec = np.array([type_of[e] for e in elements])
        lmax = PSM.lmax_per_energy(ph, list(type_of.values()) + [0])
        acc = np.zeros(k.size, complex)
        nabs = 0
        sites_a = np.where(species == a)[0]
        for X in snaps:
            u = X - cell.ideal
            pick = rng.choice(sites_a, size=min(absorbers_per_snap, sites_a.size), replace=False)
            rows_nsc, rows_typ, rows_legs, rows_w, rows_R = [], [], [], [], []
            for c in pick:
                lo, hi = off[c], off[c + 1]
                lut = {keyof(v): jj for v, jj in zip(offs[lo:hi], j_n[lo:hi])}
                for (opos, deg, is_abs) in templates:
                    idx = [c if ab else lut[keyof(v)] for v, ab in zip(opos, is_abs)]
                    rel = opos + u[idx] - u[c]
                    pts = np.vstack([np.zeros(3), rel, np.zeros(3)])
                    legs = np.diff(pts, axis=0)
                    ty = [0 if ab else tvec[species[ii]] for ii, ab in zip(idx, is_abs)]
                    rows_nsc.append(len(idx)); rows_typ.append(ty); rows_legs.append(legs)
                    rows_w.append(deg); rows_R.append(0.5 * np.linalg.norm(legs, axis=1).sum())
            nabs += pick.size
            P = len(rows_nsc)
            maxs = max(rows_nsc)
            nsc = np.array(rows_nsc, np.int32)
            typ = np.zeros((P, maxs), np.int32)
            legs = np.zeros((P, maxs + 1, 3))
            for q in range(P):
                typ[q, :nsc[q]] = rows_typ[q]
                legs[q, :nsc[q] + 1] = rows_legs[q]
            lens, eul = PSM.frames_batch(legs / BOHR, nsc)
            tr = PSM.path_sum_packed(nsc, typ, lens, eul, ph, lmax, precision, energies=ie,
                                     out_device=True)
            # per-path envelope on the coarse grid -> fine grid, then sum
            Reff = cp.asarray(np.array(rows_R))
            w = cp.asarray(np.array(rows_w, float))
            kc = cp.asarray(k[ie])
            kf = cp.asarray(k)
            env = tr * cp.exp(-2j * kc[None, :] * Reff[:, None])
            pos = cp.clip(cp.searchsorted(kc, kf) - 1, 0, kc.size - 2)
            t = (kf - kc[pos]) / (kc[pos + 1] - kc[pos])
            envf = env[:, pos] * (1 - t)[None, :] + env[:, pos + 1] * t[None, :]
            s = (w[:, None] * envf * cp.exp(2j * kf[None, :] * Reff[:, None])).sum(0)
            acc += cp.asnumpy(s)
        out[el] = (1j / 3.0) * ph.S[0, :, 1] * acc / max(1, nabs)
        if verbose:
            print(f"    MS {el}: {nabs} absorbers x {len(templates)} paths")
    return out
