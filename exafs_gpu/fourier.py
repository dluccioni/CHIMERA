"""k -> R Fourier transform using the standard EXAFS conventions.

Calibrated against the real data in ../../../Data/RDP MC: those files have
dR = 0.0306796 A, which is pi/(nfft*dk) for nfft=2048, dk=0.05 - the Athena /
Larch default. Matching it means synthetic data lands on the same grid as the
experiment, so nothing about the comparison is an artefact of resampling.

    chi(R) = 1/sqrt(2*pi) * Int k^n chi(k) W(k) exp(2ikR) dk

W(k) is a Hanning window with `dk_win` taper, which is what limits R-space
resolution to ~pi/(2*dkrange) and produces the sidelobes real data shows.
"""
from __future__ import annotations
import numpy as np
import cupy as cp

DK = 0.05
NFFT = 2048


def r_grid(nfft=NFFT, dk=DK):
    return np.arange(nfft // 2) * np.pi / (nfft * dk)


def hanning_window(k, kmin, kmax, dk_win=1.0):
    """Athena-style Hanning window with cosine tapers of width dk_win."""
    w = np.zeros_like(k)
    w[(k >= kmin + dk_win / 2) & (k <= kmax - dk_win / 2)] = 1.0
    lo = (k > kmin - dk_win / 2) & (k < kmin + dk_win / 2)
    w[lo] = 0.5 * (1 + np.sin(np.pi * (k[lo] - kmin) / dk_win))
    hi = (k > kmax - dk_win / 2) & (k < kmax + dk_win / 2)
    w[hi] = 0.5 * (1 - np.sin(np.pi * (k[hi] - kmax) / dk_win))
    return w


class FourierTransform:
    """Resamples chi(k) onto the uniform FFT grid and transforms to R space."""

    def __init__(self, k_model, kmin=3.0, kmax=14.0, kweight=2, dk_win=1.0,
                 nfft=NFFT, dk=DK, xp=cp):
        self.xp = xp
        self.nfft, self.dk, self.kweight = nfft, dk, kweight
        self.k_model = np.asarray(k_model)
        self.k_uniform = np.arange(nfft) * dk
        self.kmin, self.kmax = kmin, kmax

        win = hanning_window(self.k_uniform, kmin, kmax, dk_win)
        win[self.k_uniform > k_model.max()] = 0.0
        win[self.k_uniform < k_model.min()] = 0.0
        self.weight = win * self.k_uniform ** kweight
        self.R = r_grid(nfft, dk)

        # linear resampling from the model k-grid onto the uniform FFT grid,
        # precomputed as index + fraction so it is a single gather at run time
        idx = np.searchsorted(self.k_model, self.k_uniform) - 1
        idx = np.clip(idx, 0, self.k_model.size - 2)
        # a large E0 shift clamps the low end of k_model to one repeated
        # value; those intervals have zero width and lie outside the window,
        # so the fraction there is irrelevant - just keep it finite
        span = np.maximum(self.k_model[idx + 1] - self.k_model[idx], 1e-12)
        frac = np.clip((self.k_uniform - self.k_model[idx]) / span, 0.0, 1.0)
        inside = (self.k_uniform >= self.k_model[0]) & (self.k_uniform <= self.k_model[-1])
        self._idx = xp.asarray(idx)
        self._frac = xp.asarray(frac * inside)
        self._inside = xp.asarray(inside.astype(np.float64))
        self._w = xp.asarray(self.weight)
        self._norm = dk / np.sqrt(np.pi)

    def resample(self, chi):
        """chi (..., nk) on the model grid -> (..., nfft) on the uniform grid."""
        a = chi[..., self._idx]
        b = chi[..., self._idx + 1]
        return (a + (b - a) * self._frac) * self._inside

    def to_R(self, chi):
        """chi(k) -> complex chi(R). Returns (..., nfft//2)."""
        xp = self.xp
        cu = self.resample(chi) * self._w
        out = xp.fft.fft(cu, n=self.nfft, axis=-1)[..., : self.nfft // 2]
        return out * self._norm

    def magnitude(self, chi):
        return self.xp.abs(self.to_R(chi))

    # ------------------------------------------------------------- adjoints
    def grad_resample(self, g):
        """Adjoint of `resample`: (..., nfft) -> (..., nk) on the model grid."""
        xp = self.xp
        gi = g * self._inside
        out = xp.zeros(g.shape[:-1] + (self.k_model.size,), g.dtype)
        flat = out.reshape(-1, self.k_model.size)
        gf = gi.reshape(-1, gi.shape[-1])
        if xp is cp:
            import cupyx
            for b in range(flat.shape[0]):
                cupyx.scatter_add(flat[b], self._idx, gf[b] * (1.0 - self._frac))
                cupyx.scatter_add(flat[b], self._idx + 1, gf[b] * self._frac)
        else:
            for b in range(flat.shape[0]):
                np.add.at(flat[b], self._idx, gf[b] * (1.0 - self._frac))
                np.add.at(flat[b], self._idx + 1, gf[b] * self._frac)
        return out

    def grad_magnitude(self, chi, gmag):
        """Backpropagate dL/d|chi(R)| through to dL/dchi(k) on the model grid.

        |Z| with Z = F(x), F linear and x real:
            dL/dx = Re[ F^H ( gmag * Z/|Z| ) ]
        and F^H for a truncated unnormalised FFT is nfft * ifft on the
        zero-embedded gradient.
        """
        xp = self.xp
        Z = self.to_R(chi)
        A = xp.abs(Z)
        phase = Z / xp.maximum(A, 1e-300)
        v = gmag * phase * self._norm
        full = xp.zeros(v.shape[:-1] + (self.nfft,), v.dtype)
        full[..., : self.nfft // 2] = v
        y = xp.fft.ifft(full, axis=-1) * self.nfft
        gk = xp.real(y) * self._w
        return self.grad_resample(gk)

    def window_mask(self, rmin, rmax):
        """Boolean mask selecting the R range actually used in the fit."""
        return (self.R >= rmin) & (self.R <= rmax)

    def n_independent(self, rmin, rmax):
        """Nyquist/Stern estimate: 2*dk*dR/pi + 2."""
        return 2 * (self.kmax - self.kmin) * (rmax - rmin) / np.pi + 2
