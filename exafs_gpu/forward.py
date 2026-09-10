"""GPU forward model: atomic configuration -> chi(k), plus its analytic adjoint.

Thermal disorder is handled two ways:

  ANALYTIC (default, used by the fits)
      For harmonic motion the average of sin(2kr+d) over thermal realisations is
      exactly sin(2k<r>+d) * exp(-2 k^2 sigma^2_bond). sigma^2_bond per pair comes
      in closed form from the force-constant matrix:
          sigma^2_ij = e^T (C_ii + C_jj - C_ij - C_ji) e,   C = kB*T*Phi^+
      So no snapshot averaging is needed at all, and the correlation between
      neighbouring atoms' motion is captured exactly.

  EXPLICIT (used to generate ground truth and to validate the above)
      Draw N correlated displacement fields, evaluate chi for each, average.

Comparing the two quantifies the harmonic approximation directly.
"""
from __future__ import annotations
import numpy as np
import cupy as cp
from . import kernels
from .lattice import KB
from .scattering import NSP

_TPB = 256  # power of two: the block reduction assumes it


def _grid(n, tpb=_TPB):
    return (int((n + tpb - 1) // tpb),)


def thermal_sigma2(Phi, cell, i_idx, j_idx, shift, temperature=300.0):
    """Per-pair mean-square relative displacement from the harmonic covariance."""
    w, V = np.linalg.eigh(Phi)
    keep = w > 1e-6 * max(1.0, float(w.max()))
    C = (V[:, keep] * (KB * temperature / w[keep])[None, :]) @ V[:, keep].T

    d = (cell.ideal[j_idx] + shift) - cell.ideal[i_idx]
    r = np.linalg.norm(d, axis=1)
    e = d / r[:, None]

    n3i = (3 * i_idx[:, None] + np.arange(3)[None, :])
    n3j = (3 * j_idx[:, None] + np.arange(3)[None, :])
    Cii = C[n3i[:, :, None], n3i[:, None, :]]
    Cjj = C[n3j[:, :, None], n3j[:, None, :]]
    Cij = C[n3i[:, :, None], n3j[:, None, :]]
    M = Cii + Cjj - Cij - np.transpose(Cij, (0, 2, 1))
    return np.einsum("pa,pab,pb->p", e, M, e)


class ForwardModel:
    """Holds GPU-resident tables and topology; evaluates chi and its gradients."""

    def __init__(self, cell, tables, i_idx, j_idx, shift, sigma2=None, shell=None):
        """`shell` (npairs, int) labels the coordination shell of each pair. It
        only matters when a bond-length offset (`set_dr`) or a third cumulant
        (`set_c3`) is used; both default to zero, so leaving it out reproduces
        the plain harmonic model exactly."""
        self.cell = cell
        self.natoms = cell.natoms
        self.npairs = int(i_idx.size)
        k, f, delta, lam = tables.as_float64()
        self.nk = int(k.size)
        self.nsp = int(f.shape[0])          # species count comes from the tables
        self.s02 = float(tables.s02)
        if shell is None:
            shell = np.zeros(self.npairs, np.int32)
        shell = np.ascontiguousarray(shell, np.int32)
        assert shell.shape == (self.npairs,), "shell must be one index per pair"
        self.nshell = int(shell.max()) + 1
        self.d_shell = cp.asarray(shell)
        self.dr = np.zeros((self.nshell, self.nsp, self.nsp))
        self.c3 = np.zeros(self.nshell)
        self.d_dr = cp.zeros(self.nshell * self.nsp * self.nsp, cp.float64)
        self.d_c3 = cp.zeros(self.nshell, cp.float64)

        self.d_k = cp.asarray(k)
        self.d_f = cp.asarray(np.ascontiguousarray(f))
        dflat = np.ascontiguousarray(delta.reshape(self.nsp ** 2, -1))
        self.d_delta = cp.asarray(dflat)
        # cos/sin of the phase, so the soft kernels can use the angle-addition
        # identity and call sincos once per (pair, k) rather than NSP*NSP times
        self.d_cosd = cp.asarray(np.ascontiguousarray(np.cos(dflat)))
        self.d_sind = cp.asarray(np.ascontiguousarray(np.sin(dflat)))
        self.d_lam = cp.asarray(lam)
        self.d_i = cp.asarray(np.ascontiguousarray(i_idx, np.int32))
        self.d_j = cp.asarray(np.ascontiguousarray(j_idx, np.int32))
        self.d_shift = cp.asarray(np.ascontiguousarray(shift, np.float64))
        if sigma2 is None:
            sigma2 = np.zeros(self.npairs)
        self.d_sig2 = cp.asarray(np.ascontiguousarray(sigma2, np.float64))

        n = self.nsp
        self._k_r = kernels.get("compute_r", n)
        self._k_pt = kernels.get("build_ptype", n)
        self._k_fw = kernels.get("chi_forward", n)
        self._k_adj = kernels.get("chi_adjoint_r", n)
        self._k_sc = kernels.get("scatter_force", n)
        self._k_fws = kernels.get("chi_forward_soft", n)
        self._k_adjs = kernels.get("chi_adjoint_soft", n)

    # ------------------------------------------------------ cumulant physics
    def set_dr(self, dr):
        """Bond-length offsets (nshell, nsp, nsp) in Angstrom, or (nsp, nsp)
        for the first shell only. r_eff = r + dr[shell][absorber, scatterer]."""
        dr = np.asarray(dr, np.float64)
        full = np.zeros((self.nshell, self.nsp, self.nsp))
        if dr.ndim == 2:
            full[0] = dr
        else:
            assert dr.shape == full.shape, f"dr must be {full.shape} or (nsp, nsp)"
            full[:] = dr
        self.dr = full
        self.d_dr = cp.asarray(np.ascontiguousarray(full.reshape(-1)))
        return self

    def set_c3(self, c3):
        """Third cumulant per shell (Angstrom^3); a scalar applies to shell 1."""
        c3 = np.atleast_1d(np.asarray(c3, np.float64))
        full = np.zeros(self.nshell)
        if c3.size == 1:
            full[0] = c3[0]
        else:
            assert c3.shape == (self.nshell,), f"c3 must have {self.nshell} entries"
            full[:] = c3
        self.c3 = full
        self.d_c3 = cp.asarray(np.ascontiguousarray(full))
        return self

    # ---------------------------------------------------------------- helpers
    def distances(self, pos):
        """pos: (nconf, natoms, 3) device array -> (r, evec)."""
        nconf = pos.shape[0]
        tot = self.npairs * nconf
        r = cp.empty((nconf, self.npairs), cp.float64)
        ev = cp.empty((nconf, self.npairs, 3), cp.float64)
        self._k_r(_grid(tot), (_TPB,),
                  (pos, self.d_i, self.d_j, self.d_shift, r, ev,
                   np.int32(self.npairs), np.int32(self.natoms), np.int32(nconf)))
        return r, ev

    def ptypes(self, species):
        nconf = species.shape[0]
        tot = self.npairs * nconf
        pt = cp.empty((nconf, self.npairs), cp.int32)
        self._k_pt(_grid(tot), (_TPB,),
                   (species, self.d_i, self.d_j, pt,
                    np.int32(self.npairs), np.int32(self.natoms), np.int32(nconf)))
        return pt

    # ---------------------------------------------------------------- forward
    def chi(self, pos, species, r=None, ptype=None, normalise=True):
        """Discrete-species chi. pos (nconf,natoms,3), species (nconf,natoms) int32."""
        nconf = pos.shape[0]
        if r is None:
            r, _ = self.distances(pos)
        if ptype is None:
            ptype = self.ptypes(species)
        chi = cp.empty((nconf, self.nsp, self.nk), cp.float64)
        self._k_fw((self.nk, nconf), (_TPB,),
                   (r, ptype, self.d_shell, self.d_sig2, self.d_f, self.d_delta,
                    self.d_lam, self.d_k, self.d_dr, self.d_c3, chi,
                    np.int32(self.npairs), np.int32(self.nk),
                    np.int32(nconf), np.float64(self.s02)),
                   shared_mem=self.nsp * _TPB * 8)
        if normalise:
            cnt = cp.zeros((nconf, self.nsp), cp.float64)
            for s in range(self.nsp):
                cnt[:, s] = (species == s).sum(axis=1)
            chi /= cp.maximum(cnt, 1.0)[:, :, None]
        return chi

    def chi_soft(self, pos, w, r=None, normalise=True):
        """Relaxed-occupancy chi. w (nconf, natoms, NSP) on the simplex."""
        nconf = pos.shape[0]
        if r is None:
            r, _ = self.distances(pos)
        chi = cp.empty((nconf, self.nsp, self.nk), cp.float64)
        self._k_fws((self.nk, nconf), (_TPB,),
                    (r, w, self.d_i, self.d_j, self.d_sig2, self.d_f,
                     self.d_cosd, self.d_sind, self.d_lam, self.d_k, chi,
                     np.int32(self.npairs), np.int32(self.nk), np.int32(nconf),
                     np.int32(self.natoms), np.float64(self.s02)),
                    shared_mem=self.nsp * _TPB * 8)
        if normalise:
            chi /= cp.maximum(w.sum(axis=1), 1.0)[:, :, None]
        return chi

    # ---------------------------------------------------------------- adjoint
    def grad_pos(self, pos, species, gchi, r=None, evec=None, ptype=None,
                 normalise=True):
        """dL/dpos given dL/dchi (nconf, NSP, nk)."""
        nconf = pos.shape[0]
        if r is None or evec is None:
            r, evec = self.distances(pos)
        if ptype is None:
            ptype = self.ptypes(species)
        g = gchi
        if normalise:
            cnt = cp.zeros((nconf, self.nsp), cp.float64)
            for s in range(self.nsp):
                cnt[:, s] = (species == s).sum(axis=1)
            g = gchi / cp.maximum(cnt, 1.0)[:, :, None]
        g = cp.ascontiguousarray(g)
        tot = self.npairs * nconf
        gr = cp.empty((nconf, self.npairs), cp.float64)
        self._k_adj(_grid(tot), (_TPB,),
                    (r, ptype, self.d_shell, self.d_sig2, self.d_f, self.d_delta,
                     self.d_lam, self.d_k, self.d_dr, self.d_c3, g, gr,
                     np.int32(self.npairs), np.int32(self.nk),
                     np.int32(nconf), np.float64(self.s02)))
        gpos = cp.zeros_like(pos)
        self._k_sc(_grid(tot), (_TPB,),
                   (gr, evec, self.d_i, self.d_j, gpos,
                    np.int32(self.npairs), np.int32(self.natoms), np.int32(nconf)))
        return gpos

    def grad_soft(self, pos, w, gchi, r=None, evec=None, normalise=True):
        """(dL/dpos, dL/dw) for the relaxed-occupancy model."""
        nconf = pos.shape[0]
        if r is None or evec is None:
            r, evec = self.distances(pos)
        g = gchi
        norm = cp.maximum(w.sum(axis=1), 1.0)
        if normalise:
            g = gchi / norm[:, :, None]
        g = cp.ascontiguousarray(g)
        tot = self.npairs * nconf
        gr = cp.empty((nconf, self.npairs), cp.float64)
        gw = cp.zeros_like(w)
        self._k_adjs(_grid(tot), (_TPB,),
                     (r, w, self.d_i, self.d_j, self.d_sig2, self.d_f,
                      self.d_cosd, self.d_sind, self.d_lam, self.d_k, g, gr, gw,
                      np.int32(self.npairs), np.int32(self.nk), np.int32(nconf),
                      np.int32(self.natoms), np.float64(self.s02)))
        if normalise:
            # d/dw of the 1/sum(w) normalisation
            chi_un = self.chi_soft(pos, w, r=r, normalise=False)
            corr = (gchi * chi_un).sum(axis=2) / norm ** 2
            gw -= corr[:, None, :]
        gpos = cp.zeros_like(pos)
        self._k_sc(_grid(tot), (_TPB,),
                   (gr, evec, self.d_i, self.d_j, gpos,
                    np.int32(self.npairs), np.int32(self.natoms), np.int32(nconf)))
        return gpos, gw

    # ------------------------------------------------------- reference (CPU)
    def chi_reference(self, pos, species, sigma2=None):
        """Straightforward NumPy implementation used only to validate the kernels."""
        pos = cp.asnumpy(pos) if isinstance(pos, cp.ndarray) else pos
        species = cp.asnumpy(species) if isinstance(species, cp.ndarray) else species
        k = cp.asnumpy(self.d_k)
        f = cp.asnumpy(self.d_f)
        delta = cp.asnumpy(self.d_delta)
        lam = cp.asnumpy(self.d_lam)
        ii = cp.asnumpy(self.d_i); jj = cp.asnumpy(self.d_j)
        sh = cp.asnumpy(self.d_shift)
        s2 = cp.asnumpy(self.d_sig2) if sigma2 is None else sigma2
        shell = cp.asnumpy(self.d_shell)
        out = np.zeros((pos.shape[0], self.nsp, k.size))
        for c in range(pos.shape[0]):
            d = (pos[c][jj] + sh) - pos[c][ii]
            sa = species[c][ii]; sb = species[c][jj]
            t = sa * self.nsp + sb
            r = np.linalg.norm(d, axis=1) + self.dr[shell, sa, sb]
            ph3 = (4.0 / 3.0) * k[None, :] ** 3 * self.c3[shell][:, None]
            amp = (self.s02 * f[sb] / (k[None, :] * r[:, None] ** 2)
                   * np.exp(-2 * r[:, None] / lam[None, :]
                            - 2 * k[None, :] ** 2 * s2[:, None]))
            term = amp * np.sin(2 * k[None, :] * r[:, None] + delta[t] - ph3)
            for s in range(self.nsp):
                m = sa == s          # selects PAIRS whose absorber is species s
                nabs = max(int((species[c] == s).sum()), 1)   # but normalise by ATOMS
                out[c, s] = term[m].sum(0) / nabs
        return out
