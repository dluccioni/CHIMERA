"""FCC supercell, fixed-topology neighbour lists, and a harmonic thermal model.

Two design choices that matter:

1. The neighbour LIST is built once from the ideal lattice and never rebuilt.
   The structure is constrained to remain FCC and displacements stay well below
   half the NN spacing, so pair topology is invariant. Only distances change.
   This makes both the forward model and the MC delta-updates much cheaper.

2. Thermal displacements are drawn from a harmonic lattice-dynamics covariance,
   NOT independently per atom. EXAFS sigma^2 is the variance of the BOND length,
   which depends on the correlation between the two atoms' displacements. Drawing
   independent Gaussians gives sigma^2_bond = 2<u^2>, overestimating the first
   shell MSRD substantially. The harmonic model gets this right and needs no MD:
   u ~ N(0, kB*T * Phi^+) with Phi the force-constant matrix.
"""
from __future__ import annotations
import numpy as np
from .scattering import ELEMENTS, NSP

KB = 8.617333262e-5  # eV/K

# Metallic radii (Angstrom, CN=12) -> pair equilibrium NN distances for the
# static size-mismatch relaxation. Extend as needed for other systems.
METALLIC_RADIUS = {
    "Al": 1.432, "Si": 1.176, "Ti": 1.462, "V": 1.316, "Cr": 1.280,
    "Mn": 1.264, "Fe": 1.274, "Co": 1.251, "Ni": 1.246, "Cu": 1.278,
    "Zn": 1.394, "Zr": 1.603, "Nb": 1.468, "Mo": 1.400, "Pd": 1.376,
    "Hf": 1.580, "Ta": 1.467, "W": 1.408, "Pt": 1.387, "Au": 1.442,
}


class FCCSupercell:
    def __init__(self, ncell=(5, 5, 5), a0=3.56):
        self.ncell = tuple(ncell)
        self.a0 = float(a0)
        basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
        cells = np.array([(i, j, k) for i in range(ncell[0])
                          for j in range(ncell[1]) for k in range(ncell[2])], float)
        frac = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)
        frac[:, 0] /= ncell[0]
        frac[:, 1] /= ncell[1]
        frac[:, 2] /= ncell[2]
        self.frac = frac
        self.box = np.array([ncell[0] * a0, ncell[1] * a0, ncell[2] * a0])
        self.ideal = np.ascontiguousarray(frac * self.box)
        self.natoms = self.ideal.shape[0]

    def min_image(self, d):
        return d - self.box * np.round(d / self.box)

    def neighbour_list(self, rmax):
        """Ordered pair list (i, j) with |r_ij| <= rmax under minimum image.

        Returns i_idx, j_idx, shift, off where `shift` is the periodic offset to
        add to pos[j] so distances can be recomputed after displacement, and
        `off` is a CSR-style index so neighbours of atom i are [off[i], off[i+1]).
        """
        pos = self.ideal
        n = self.natoms
        I, J, SH = [], [], []
        for i in range(n):
            d = self.min_image(pos - pos[i])
            r = np.linalg.norm(d, axis=1)
            sel = np.where((r <= rmax) & (r > 1e-6))[0]
            I.append(np.full(sel.size, i))
            J.append(sel)
            SH.append(-np.round((pos[sel] - pos[i]) / self.box) * self.box)
        i_idx = np.concatenate(I).astype(np.int32)
        j_idx = np.concatenate(J).astype(np.int32)
        shift = np.ascontiguousarray(np.concatenate(SH), np.float64)
        counts = np.bincount(i_idx, minlength=n)
        off = np.zeros(n + 1, np.int32)
        off[1:] = np.cumsum(counts)
        return i_idx, j_idx, shift, off

    def shell_radii(self, rmax):
        d = self.min_image(self.ideal - self.ideal[0])
        r = np.linalg.norm(d, axis=1)
        r = r[(r > 1e-6) & (r <= rmax)]
        vals, cnt = np.unique(np.round(r, 4), return_counts=True)
        return vals, cnt


def pair_vectors(cell, pos, i_idx, j_idx, shift):
    return (pos[j_idx] + shift) - pos[i_idx]


def _nn_cut(cell, pad=1.05):
    """Nearest-neighbour cutoff for whatever lattice this is."""
    if hasattr(cell, "nn_distance"):
        return cell.nn_distance() * pad
    return cell.a0 / np.sqrt(2) * pad          # legacy FCC fallback


def force_constant_matrix(cell, i_idx, j_idx, shift,
                          k_bond=2.6, k_angle=0.35, nn_cut=None):
    """Harmonic force constants: NN bond-stretch + weaker transverse term (eV/A^2)."""
    n = cell.natoms
    nn = nn_cut if nn_cut is not None else _nn_cut(cell)
    Phi = np.zeros((3 * n, 3 * n))
    d = pair_vectors(cell, cell.ideal, i_idx, j_idx, shift)
    r = np.linalg.norm(d, axis=1)
    m = r <= nn
    for i, j, dv, rv in zip(i_idx[m], j_idx[m], d[m], r[m]):
        e = dv / rv
        blk = k_bond * np.outer(e, e) + k_angle * (np.eye(3) - np.outer(e, e))
        Phi[3 * i:3 * i + 3, 3 * j:3 * j + 3] -= blk
        Phi[3 * i:3 * i + 3, 3 * i:3 * i + 3] += blk
    return Phi


class ThermalSampler:
    """Draws correlated displacement fields from the harmonic covariance kB*T*Phi^+."""

    def __init__(self, Phi, temperature=300.0, rng=None):
        self.T = float(temperature)
        self.rng = rng or np.random.default_rng(0)
        w, V = np.linalg.eigh(Phi)
        keep = w > 1e-6 * max(1.0, float(w.max()))
        self.amp = V[:, keep] * np.sqrt(KB * self.T / w[keep])[None, :]
        self.nmode = int(keep.sum())
        self.natoms = Phi.shape[0] // 3

    def draw(self, nsample=1):
        xi = self.rng.standard_normal((self.nmode, nsample))
        u = self.amp @ xi
        return np.ascontiguousarray(u.T.reshape(nsample, self.natoms, 3))

    def msd(self):
        """Mean square displacement per atom per Cartesian axis."""
        return float((self.amp ** 2).sum() / self.natoms / 3.0)


def static_relaxation(cell, species, i_idx, j_idx, shift,
                      k_bond=2.6, n_iter=400, step=0.08, elements=None):
    """Relax atoms against species-dependent NN equilibrium bond lengths.

    This is the size-mismatch distortion: the part of the displacement field that
    carries chemical information. Kept separate from thermal motion so the two can
    be disentangled.
    """
    pos = cell.ideal.copy()
    d0 = pair_vectors(cell, cell.ideal, i_idx, j_idx, shift)
    r0 = np.linalg.norm(d0, axis=1)
    nn = _nn_cut(cell)
    m = r0 <= nn
    ii, jj, sh = i_idx[m], j_idx[m], shift[m]
    els = elements if elements is not None else ELEMENTS
    req = np.array([[METALLIC_RADIUS[a] + METALLIC_RADIUS[b] for b in els]
                    for a in els])
    target = req[species[ii], species[jj]]
    for _ in range(n_iter):
        d = (pos[jj] + sh) - pos[ii]
        r = np.linalg.norm(d, axis=1)
        fvec = ((k_bond * (r - target)) / r)[:, None] * d
        F = np.zeros_like(pos)
        np.add.at(F, ii, fvec)
        F -= F.mean(0)
        pos += step * F
    return pos - cell.ideal


def warren_cowley(species, i_idx, j_idx, shift, cell, shell_r, tol=0.15,
                  comp=None, nsp=None):
    """alpha_AB = 1 - P(B|A)/c_B for one shell. Zero = random, negative = AB-rich."""
    NSP = int(nsp) if nsp is not None else int(species.max()) + 1
    d = pair_vectors(cell, cell.ideal, i_idx, j_idx, shift)
    r = np.linalg.norm(d, axis=1)
    m = np.abs(r - shell_r) < tol
    a, b = species[i_idx[m]], species[j_idx[m]]
    c = comp if comp is not None else np.bincount(species, minlength=NSP) / species.size
    alpha = np.zeros((NSP, NSP))
    for A in range(NSP):
        sel = a == A
        tot = int(sel.sum())
        if tot == 0:
            continue
        for B in range(NSP):
            P = (b[sel] == B).sum() / tot
            alpha[A, B] = 1.0 - P / c[B]
    return alpha


def wc_vector(alpha):
    """Upper-triangular WC entries (the independent ones) as a flat vector."""
    return alpha[np.triu_indices(alpha.shape[0])]


def wc_labels(elements=None):
    els = elements if elements is not None else ELEMENTS
    return [f"a_{els[i]}{els[j]}" for i, j in zip(*np.triu_indices(len(els)))]


WC_LABELS = wc_labels()          # default CrCoNi labels, kept for existing code


# ===========================================================================
#  General lattices
# ===========================================================================
# Bravais lattice + basis, in fractional coordinates of the conventional cell.
# Every site listed must be crystallographically equivalent (same coordination),
# because the neighbour-list machinery and the MC proposal kernels assume a
# uniform coordination number.
LATTICES = {
    "sc":  dict(basis=[[0, 0, 0]], hex=False),
    "bcc": dict(basis=[[0, 0, 0], [.5, .5, .5]], hex=False),
    "fcc": dict(basis=[[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]],
                hex=False),
    "hcp": dict(basis=[[0, 0, 0], [1 / 3, 2 / 3, .5]], hex=True),
}


class CrystalSupercell:
    """Periodic supercell for an arbitrary Bravais lattice + equivalent basis.

    Handles non-orthogonal cells (hcp) by carrying an explicit cell matrix H
    whose ROWS are the supercell vectors. Minimum image is found by an explicit
    search over the 27 neighbouring images rather than by rounding fractional
    coordinates, which is only exact for near-orthogonal cells and silently
    wrong for the 120-degree hcp cell.
    """

    def __init__(self, lattice="fcc", ncell=(5, 5, 5), a0=3.56, c_over_a=None):
        if lattice not in LATTICES:
            raise ValueError(f"unknown lattice {lattice!r}; have {list(LATTICES)}")
        spec = LATTICES[lattice]
        self.lattice = lattice
        self.ncell = tuple(ncell)
        self.a0 = float(a0)
        if spec["hex"]:
            ca = c_over_a if c_over_a is not None else np.sqrt(8.0 / 3.0)
            A = np.array([[1.0, 0.0, 0.0],
                          [-0.5, np.sqrt(3) / 2, 0.0],
                          [0.0, 0.0, ca]]) * a0
        else:
            A = np.eye(3) * a0
        self.cell_matrix = A
        self.H = np.diag(ncell).astype(float) @ A     # supercell vectors (rows)

        basis = np.asarray(spec["basis"], float)
        cells = np.array([(i, j, k) for i in range(ncell[0])
                          for j in range(ncell[1]) for k in range(ncell[2])], float)
        frac_conv = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)
        self.ideal = np.ascontiguousarray(frac_conv @ A)
        self.frac = frac_conv / np.array(ncell, float)
        self.natoms = self.ideal.shape[0]
        # kept so code written against the cubic version keeps working
        self.box = np.diag(self.H).copy()
        self._images = np.array([(i, j, k) for i in (-1, 0, 1)
                                 for j in (-1, 0, 1) for k in (-1, 0, 1)], float) @ self.H

    def min_image(self, d):
        """Exact minimum image by explicit search over the 27 nearest images."""
        d = np.atleast_2d(d)
        cand = d[:, None, :] + self._images[None, :, :]
        n2 = (cand ** 2).sum(-1)
        return cand[np.arange(d.shape[0]), n2.argmin(1)]

    def neighbour_list(self, rmax):
        pos = self.ideal
        n = self.natoms
        I, J, SH = [], [], []
        for i in range(n):
            d0 = pos - pos[i]
            cand = d0[:, None, :] + self._images[None, :, :]
            n2 = (cand ** 2).sum(-1)
            best = n2.argmin(1)
            d = cand[np.arange(n), best]
            sh = self._images[best]
            r = np.linalg.norm(d, axis=1)
            sel = np.where((r <= rmax) & (r > 1e-6))[0]
            I.append(np.full(sel.size, i)); J.append(sel); SH.append(sh[sel])
        i_idx = np.concatenate(I).astype(np.int32)
        j_idx = np.concatenate(J).astype(np.int32)
        shift = np.ascontiguousarray(np.concatenate(SH), np.float64)
        counts = np.bincount(i_idx, minlength=n)
        off = np.zeros(n + 1, np.int32); off[1:] = np.cumsum(counts)
        return i_idx, j_idx, shift, off

    def shell_radii(self, rmax):
        d = self.min_image(self.ideal - self.ideal[0])
        r = np.linalg.norm(d, axis=1)
        r = r[(r > 1e-6) & (r <= rmax)]
        vals, cnt = np.unique(np.round(r, 3), return_counts=True)
        return vals, cnt

    def nn_distance(self):
        v, _ = self.shell_radii(self.a0 * 1.5)
        return float(v[0])

    def uniform_coordination(self, rmax):
        i, _, _, _ = self.neighbour_list(rmax)
        c = np.bincount(i, minlength=self.natoms)
        return int(c[0]) if c.min() == c.max() else None
