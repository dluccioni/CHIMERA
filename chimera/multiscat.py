"""Multiple-scattering contribution from FEFF8L paths, so the 4th shell can be fitted.

Why it is needed
----------------
In fcc, twice the nearest-neighbour distance equals the fourth-shell radius to
one part in 5000 (2 x 2.5221 = 5.0443 A against 5.0430 A). The collinear path
absorber -> nearest neighbour -> fourth-shell atom therefore lands exactly on
top of the fourth-shell single-scattering peak, and it is forward-focused: the
intervening atom scatters through zero degrees, where |f| is an order of
magnitude larger than at 180 degrees. A single-scattering model is not merely
imprecise there, it is missing most of the amplitude - measured as an
R-factor of 0.40 over 4.2-4.8 A while the data is still 25x the noise.

What is implemented
-------------------
Every path FEFF8L reports with 3 or more legs, out to a half-path length of
5.3 A, summed with FEFF's own degeneracies:

    chi_MS(p) = sum_paths  N S02 f(p) / (p reff^2)
                * exp(-2 reff / lambda(p)) * exp(-2 p^2 sigma^2_path)
                * sin(2 p reff + delta(p))

on the same internal-momentum axis as the pair sum, using the same
convention conversion as `scattering.read_feffdat`.

What is approximated, and why it is acceptable
----------------------------------------------
The sum is taken over the paths of ONE equiatomic random cluster, so chi_MS
is composition-averaged: it does not vary with the configuration being
fitted. The chemical information in the fourth shell is still there and still
configuration-dependent, because the fourth-shell SINGLE scattering stays in
the pair sum where it always was; what is frozen is only the focusing
correction on top of it. That is the standard treatment (fixed MS paths
sharing the single-scattering S02 and E0), and it is justified here because
the intervening atom enters through forward scattering, where Cr, Co and Ni
differ by a few percent - far less than they differ at 180 degrees.

Two parameters are fitted: an amplitude for the whole MS term, and the
Debye-Waller scale c_ms in sigma^2_path = c_ms sigma^2_NN reff / (2 r_NN).
"""
from __future__ import annotations
import os, glob
import numpy as np

from exafs_gpu.scattering import read_feffdat, ELEMENTS, NSP

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data")
D_NN_FEFF = 2.517          # the cluster the paths were computed for


def _read_path(fn):
    return read_feffdat(fn)


class MSPaths:
    """FEFF8L multiple-scattering paths for each absorber."""

    def __init__(self, name="crconi_fcc2517_ms", elements=ELEMENTS, min_leg=3,
                 rmax=None):
        self.elements = tuple(elements)
        self.paths = {}
        for a in self.elements:
            folder = os.path.join(_ROOT, name, a)
            got = []
            for fn in sorted(glob.glob(os.path.join(folder, "feff0*.dat"))):
                d = _read_path(fn)
                if d["nleg"] < min_leg:
                    continue
                if rmax is not None and d["reff"] > rmax:
                    continue
                got.append(d)
            self.paths[a] = got
        self.n = {a: len(v) for a, v in self.paths.items()}

    def summary(self):
        out = []
        for a in self.elements:
            reff = np.array([p["reff"] for p in self.paths[a]])
            deg = np.array([p["deg"] for p in self.paths[a]])
            nleg = np.array([p["nleg"] for p in self.paths[a]])
            out.append(dict(absorber=a, n_paths=len(reff),
                            total_degeneracy=float(deg.sum()),
                            reff_min=float(reff.min()) if reff.size else None,
                            reff_max=float(reff.max()) if reff.size else None,
                            by_nleg={int(l): int((nleg == l).sum())
                                     for l in np.unique(nleg)}))
        return out

    def prepare(self, k_phys):
        """Interpolate every path onto the model momentum grid, once."""
        p = np.asarray(k_phys, float)
        self._grid = p
        self._prep = {}
        for a in self.elements:
            reff = np.array([d["reff"] for d in self.paths[a]])
            deg = np.array([d["deg"] for d in self.paths[a]])
            F = np.array([np.interp(p, d["p"], d["f"]) for d in self.paths[a]])
            D = np.array([np.interp(p, d["p"], d["delta"]) for d in self.paths[a]])
            L = np.array([np.interp(p, d["p"], d["lam"]) for d in self.paths[a]])
            self._prep[a] = (reff, deg, F, D, L)
        return self

    def chi_fast(self, sigma2_nn, r_nn, c_ms=1.3, amp=1.0, s02=1.0, a_scale=1.0):
        """Vectorised chi_MS(p) on the grid passed to `prepare`."""
        p = self._grid
        out = np.zeros((len(self.elements), p.size))
        for e, a in enumerate(self.elements):
            reff, deg, F, D, L = self._prep[a]
            r = (reff * a_scale)[:, None]
            s2 = (c_ms * sigma2_nn * r / (2.0 * r_nn))
            term = (deg[:, None] * s02 * F / np.maximum(p[None, :] * r ** 2, 1e-12)
                    * np.exp(-2.0 * r / np.maximum(L, 1e-6) - 2.0 * p[None, :] ** 2 * s2)
                    * np.sin(2.0 * p[None, :] * r + D))
            out[e] = amp * term.sum(axis=0)
        return out

    def chi(self, k_phys, sigma2_nn, r_nn, c_ms=1.3, amp=1.0, s02=1.0,
            a_scale=1.0):
        """chi_MS(p), shape (nsp, nk), per absorbing atom.

        a_scale rescales every path length to the fitted lattice constant
        (the paths were generated at d_nn = 2.517 A); the tabulated f, delta
        and lambda are functions of momentum and are left alone.
        """
        p = np.asarray(k_phys, float)
        out = np.zeros((len(self.elements), p.size))
        for e, a in enumerate(self.elements):
            acc = np.zeros_like(p)
            for d in self.paths[a]:
                reff = d["reff"] * a_scale
                f = np.interp(p, d["p"], d["f"])
                dl = np.interp(p, d["p"], d["delta"])
                lam = np.interp(p, d["p"], d["lam"])
                s2 = c_ms * sigma2_nn * reff / (2.0 * r_nn)
                acc += (d["deg"] * s02 * f / np.maximum(p * reff ** 2, 1e-12)
                        * np.exp(-2.0 * reff / np.maximum(lam, 1e-6) - 2.0 * p ** 2 * s2)
                        * np.sin(2.0 * p * reff + dl))
            out[e] = amp * acc
        return out


_CACHE = {}


def get(name="crconi_fcc2517_ms", **kw):
    key = (name, tuple(sorted(kw.items())))
    if key not in _CACHE:
        _CACHE[key] = MSPaths(name, **kw)
    return _CACHE[key]
