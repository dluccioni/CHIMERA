"""chi(k) data: the representation that keeps the phase.

What the files hold decides what the fit can see. A |chi(R)| file is the
magnitude of a Fourier transform whose window, k weight and E0 were chosen by
the experimenters and are no longer recorded; the phase was discarded with
them. chi(k) is the signal before any of that happened: real-valued,
dimensionless, one sample every 0.05 A^-1 in the Athena / Larch default. From
it this module builds COMPLEX chi(R) with a window of our choosing, on the same
FFT grid the forward model uses, so data and model go through one transform
and the comparison is Re and Im point by point.

Why that matters for the fits
-----------------------------
* The phase carries r and E0; the magnitude carries the N sigma^2 S0^2
  product. With the magnitude alone, sigma^2 and the short-range order both
  just shrink the first-shell peak and the fit cannot tell them apart (the
  calibration variant sigma x0.85 moved alpha_CrCr from -0.38 to -0.15). With
  the phase, dE0 and the distances are pinned per edge and the amplitude
  channel is free to constrain sigma^2 on its own.
* The magnitude throws away roughly half the independent points
  (2 dk dR / pi per edge for the complex transform).
* The k weight stops being a nuisance to infer: it is a weighting we apply,
  and several can be applied at once. `Window.kweights` is a tuple; each
  (edge, k weight) pair is one CHANNEL of the target, ordered edge-major, and
  the per-edge amplitude is profiled over all the channels of that edge
  together. Fitting k^1, k^2 and k^3 simultaneously is the standard Artemis
  device for separating sigma^2 from the coordination numbers.

Conventions
-----------
* A complex target counts Re and Im separately: chi^2 divides by twice the
  number of points, the sampler's `nfit` does the same, and the noise sigma is
  the per-COMPONENT standard deviation.
* The R-factor is sum |fit - data|^2 / sum |data|^2 over the fit window,
  which for a real target is the usual definition.
* Absolute scale: chi(k) normalised to the edge step puts the per-edge
  amplitude near S0^2 (0.7-0.9 against tables with s02 = 1). It is still
  profiled out, but a value far from that is a normalisation problem worth
  reporting, not a result.
"""
from __future__ import annotations
from collections import namedtuple
import numpy as np

from exafs_gpu.fourier import FourierTransform, r_grid

Window = namedtuple("Window", "kmin kmax kweights dk_win")
Window.__doc__ = """Transform parameters shared by the data and the model.

kmin, kmax  Hanning window limits (A^-1);  kweights  tuple of integer k
weights, one channel each;  dk_win  taper width (A^-1)."""

NOISE_R = (15.0, 25.0)     # Larch's default band for the R-space noise estimate


# ------------------------------------------------------------------ k weights
def as_kweights(kw):
    """(int, ...) from an int, a sequence of ints, or a string like "1,2,3"."""
    if isinstance(kw, str):
        kw = [int(t) for t in kw.replace(",", " ").split()]
    elif np.isscalar(kw):
        kw = [int(kw)]
    out = tuple(int(w) for w in kw)
    assert out and all(w >= 0 for w in out), f"bad k weights {kw!r}"
    return out


def kweight_value(kweights):
    """The JSON form: the int itself for one weight, a list otherwise."""
    kws = as_kweights(kweights)
    return kws[0] if len(kws) == 1 else list(kws)


def kweight_label(kweights):
    kws = as_kweights(kweights)
    return "k^" + ",".join(str(w) for w in kws)


def channel_edges(nsp, nkw):
    """Edge index of every channel, channels ordered edge-major."""
    return np.repeat(np.arange(int(nsp)), int(nkw)).astype(np.int32)


def channel_labels(elements, kweights):
    kws = as_kweights(kweights)
    if len(kws) == 1:
        return [f"{e}" for e in elements]
    return [f"{e} k^{w}" for e in elements for w in kws]


def make_window(obj):
    """A Window from a Window, a (kmin, kmax, kweights[, dk_win]) tuple, or
    anything with a `.window` attribute (Spectrum, ShellModel)."""
    if isinstance(obj, Window):
        return obj
    if hasattr(obj, "window"):
        return obj.window
    kmin, kmax, kws = obj[:3]
    dk_win = obj[3] if len(obj) > 3 else 1.0
    return Window(float(kmin), float(kmax), as_kweights(kws), float(dk_win))


def _key(window):
    w = make_window(window)
    return (round(w.kmin, 6), round(w.kmax, 6), as_kweights(w.kweights),
            round(w.dk_win, 6))


# ------------------------------------------------------------------ transform
_FT_CACHE = {}


def transforms(k, window):
    """One FourierTransform per k weight for data on the grid `k` (cached)."""
    k = np.asarray(k, float)
    w = make_window(window)
    key = (k.tobytes(), _key(w))
    if key not in _FT_CACHE:
        if len(_FT_CACHE) > 64:
            _FT_CACHE.clear()
        _FT_CACHE[key] = [FourierTransform(k, w.kmin, w.kmax, kw, dk_win=w.dk_win, xp=np)
                          for kw in w.kweights]
    return _FT_CACHE[key]


def transform(k, chi, window):
    """chi(k) on grid k -> complex chi(R), (..., nkw, nR): the same resampling,
    window, weighting and normalisation the model's chi(k) goes through."""
    chi = np.asarray(chi, float)
    out = [ft.to_R(chi) for ft in transforms(k, window)]
    return np.stack(out, axis=-2)


class ChiK:
    """chi(k) for every edge of one spectrum (one map position, or a median).

    `k` is one array shared by all edges or a sequence of per-edge arrays;
    `chi` is (nsp, nk) or a sequence of per-edge arrays (lengths may differ
    between edges, since each edge has its own energy range). Rows are ordered
    like `exafs_gpu.scattering.ELEMENTS`, the same as the model's absorbers.
    """

    def __init__(self, k, chi, elements=None):
        if isinstance(chi, np.ndarray) and chi.ndim == 2:
            chi = list(chi)
        elif len(chi) and np.isscalar(chi[0]):
            chi = [chi]                               # a single edge
        chi = [np.ascontiguousarray(c, float).ravel() for c in chi]
        if isinstance(k, np.ndarray):
            k = [k] * len(chi) if k.ndim == 1 else list(k)
        elif len(k) and np.isscalar(k[0]):
            k = [np.asarray(k, float)] * len(chi)      # one grid for every edge
        k = [np.ascontiguousarray(x, float).ravel() for x in k]
        assert len(k) == len(chi), "one k grid per edge"
        for x, c in zip(k, chi):
            assert x.size == c.size, "k and chi differ in length"
            assert np.all(np.diff(x) > 0), "k must increase"
        self.k, self.chi = k, chi
        self.nsp = len(chi)
        self.elements = tuple(elements) if elements is not None else None
        self._cache = {}

    # ------------------------------------------------------------ grids
    @property
    def R(self):
        """The FFT R grid every transform lands on (shared with the model)."""
        return r_grid()

    def channel_edges(self, window):
        return channel_edges(self.nsp, len(make_window(window).kweights))

    # ------------------------------------------------------------ transform
    def to_R(self, window):
        """Complex chi(R), (nchan, nR), channels edge-major over k weights."""
        key = _key(window)
        if key not in self._cache:
            if len(self._cache) > 32:
                self._cache.clear()
            rows = [transform(self.k[e], self.chi[e], window) for e in range(self.nsp)]
            self._cache[key] = np.ascontiguousarray(np.concatenate(rows, axis=0))
        return self._cache[key]

    def mag_R(self, window):
        return np.abs(self.to_R(window))

    def noise(self, window, r_lo=NOISE_R[0], r_hi=NOISE_R[1]):
        """Per-channel, per-component noise sd from the structureless high-R
        band of the transform (Larch's epsilon_R convention)."""
        z = self.to_R(window)
        m = (self.R >= r_lo) & (self.R <= r_hi)
        tail = z[:, m]
        tail = tail - tail.mean(axis=1, keepdims=True)
        return np.sqrt(0.5 * np.mean(np.abs(tail) ** 2, axis=1))

    def n_independent(self, window, rmin, rmax):
        """Stern's count per edge, 2 dk dR / pi + 2: with the phase kept, all
        of it is available to the fit."""
        w = make_window(window)
        return 2 * (w.kmax - w.kmin) * (rmax - rmin) / np.pi + 2

    def weighted(self, e, kweight):
        """(k, k^w chi) of edge e, for plotting against the model."""
        return self.k[e], self.k[e] ** int(kweight) * self.chi[e]

    # ------------------------------------------------------------ combine
    @classmethod
    def median(cls, items, robust=True):
        """Point-wise median (or mean) over several ChiK on common grids."""
        items = list(items)
        assert items, "nothing to average"
        first = items[0]
        k, chi = [], []
        for e in range(first.nsp):
            for it in items[1:]:
                assert np.allclose(it.k[e], first.k[e]), "k grids differ between spectra"
            A = np.array([it.chi[e] for it in items])
            chi.append(np.median(A, axis=0) if robust else A.mean(axis=0))
            k.append(first.k[e])
        return cls(k, chi, elements=first.elements)

    def __repr__(self):
        spans = ", ".join(f"{x[0]:.2f}-{x[-1]:.2f}" for x in self.k)
        return f"ChiK(nsp={self.nsp}, k=[{spans}] A^-1)"
