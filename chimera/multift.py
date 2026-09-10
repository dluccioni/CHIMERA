"""A Fourier transform that carries one k axis per edge.

The fitted E0 differs between the Cr, Co and Ni edges, so each edge's chi(k)
has to be transformed against its own experimental k axis. `RMCSampler` and
`HybridOptimiser` only ever call `ft.magnitude(chi)`, `ft.R`,
`ft.window_mask` and `ft.n_independent`, so a small adapter holding one
`FourierTransform` per edge is enough to drop in.

`grad_magnitude` is provided as well so Method B still works: the edges are
independent, so its adjoint is just the per-edge adjoint.
"""
from __future__ import annotations
import numpy as np
import cupy as cp

from exafs_gpu.fourier import FourierTransform


class MultiEdgeFT:
    """Per-edge Fourier transform, optionally with a fixed additive chi(k).

    `chi_extra` (nsp, nk) is added to chi before transforming. That is where
    the composition-averaged multiple-scattering term enters: it is the same
    for every configuration, so the samplers' incremental delta-chi machinery
    is untouched, while the fit and the chi^2 both see it.
    """

    def __init__(self, fts, chi_extra=None):
        self.fts = list(fts)
        self.chi_extra = None if chi_extra is None else cp.asarray(
            np.ascontiguousarray(chi_extra, np.float64))
        R = self.fts[0].R
        for f in self.fts[1:]:
            assert np.allclose(f.R, R), "edges must share the R grid"
        self.R = R
        self.kmin = min(f.kmin for f in self.fts)
        self.kmax = max(f.kmax for f in self.fts)
        self.kweight = self.fts[0].kweight
        self.xp = self.fts[0].xp

    @classmethod
    def from_spectrum(cls, S):
        return cls(S.ft, chi_extra=getattr(S, "chi_ms", None))

    def to_R(self, chi):
        """chi (..., nsp, nk) -> complex (..., nsp, nR), edge by edge."""
        if self.chi_extra is not None:
            chi = chi + self.chi_extra
        out = [f.to_R(chi[..., e, :]) for e, f in enumerate(self.fts)]
        return self.xp.stack(out, axis=-2)

    def magnitude(self, chi):
        return self.xp.abs(self.to_R(chi))

    def grad_magnitude(self, chi, gmag):
        if self.chi_extra is not None:
            chi = chi + self.chi_extra
        out = [f.grad_magnitude(chi[..., e, :], gmag[..., e, :])
               for e, f in enumerate(self.fts)]
        return self.xp.stack(out, axis=-2)

    def window_mask(self, rmin, rmax):
        return (self.R >= rmin) & (self.R <= rmax)

    def n_independent(self, rmin, rmax):
        return 2 * (self.kmax - self.kmin) * (rmax - rmin) / np.pi + 2
