"""A Fourier transform that carries one k axis per edge, and one channel per
(edge, k weight).

The fitted E0 differs between the Cr, Co and Ni edges, so each edge's chi(k)
has to be transformed against its own experimental k axis. When several k
weights are fitted at once each (edge, k weight) pair is a separate output
channel, ordered edge-major; `chan_edge` says which edge a channel belongs to
so the samplers can profile one amplitude per edge over all of its channels.

`RMCSampler` only ever calls `ft.to_R(chi)` / `ft.magnitude(chi)`, `ft.R`,
`ft.window_mask`, `ft.n_independent` and reads `ft.chan_edge`, so a small
adapter holding one `FourierTransform` per channel is enough to drop in.

`grad_magnitude` is provided as well so Method B still works: the channels are
independent, so its adjoint is the sum of the per-channel adjoints over the
channels of each edge.
"""
from __future__ import annotations
import numpy as np
import cupy as cp

from exafs_gpu.fourier import FourierTransform


class MultiEdgeFT:
    """Per-channel Fourier transform, optionally with a fixed additive chi(k).

    `chi_extra` (nsp, nk) is added to chi before transforming. That is where
    the composition-averaged multiple-scattering term enters: it is the same
    for every configuration, so the samplers' incremental delta-chi machinery
    is untouched, while the fit and the chi^2 both see it.
    """

    def __init__(self, fts, chi_extra=None, chan_edge=None):
        self.fts = list(fts)
        self.chi_extra = None if chi_extra is None else cp.asarray(
            np.ascontiguousarray(chi_extra, np.float64))
        R = self.fts[0].R
        for f in self.fts[1:]:
            assert np.allclose(f.R, R), "channels must share the R grid"
        self.R = R
        self.nchan = len(self.fts)
        self.chan_edge = (np.arange(self.nchan, dtype=np.int32) if chan_edge is None
                          else np.asarray(chan_edge, np.int32))
        assert self.chan_edge.shape == (self.nchan,)
        self.nsp = int(self.chan_edge.max()) + 1
        self.kmin = min(f.kmin for f in self.fts)
        self.kmax = max(f.kmax for f in self.fts)
        self.kweights = tuple(dict.fromkeys(f.kweight for f in self.fts))
        self.kweight = self.fts[0].kweight
        self.xp = self.fts[0].xp

    @classmethod
    def from_spectrum(cls, S):
        return cls(S.ft, chi_extra=getattr(S, "chi_ms", None),
                   chan_edge=getattr(S, "chan_edge", None))

    def to_R(self, chi):
        """chi (..., nsp, nk) -> complex (..., nchan, nR), channel by channel."""
        if self.chi_extra is not None:
            chi = chi + self.chi_extra
        out = [f.to_R(chi[..., e, :]) for f, e in zip(self.fts, self.chan_edge)]
        return self.xp.stack(out, axis=-2)

    def magnitude(self, chi):
        return self.xp.abs(self.to_R(chi))

    def observe(self, chi, complex_out=False):
        """What a target of the given kind compares against."""
        return self.to_R(chi) if complex_out else self.magnitude(chi)

    def grad_magnitude(self, chi, gmag):
        if self.chi_extra is not None:
            chi = chi + self.chi_extra
        out = self.xp.zeros(chi.shape, chi.dtype)
        for c, (f, e) in enumerate(zip(self.fts, self.chan_edge)):
            out[..., e, :] += f.grad_magnitude(chi[..., e, :], gmag[..., c, :])
        return out

    def window_mask(self, rmin, rmax):
        return (self.R >= rmin) & (self.R <= rmax)

    def n_independent(self, rmin, rmax):
        return 2 * (self.kmax - self.kmin) * (rmax - rmin) / np.pi + 2
