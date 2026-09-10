"""Pair-count (shell) formulation of the swap-only forward model, on the CPU.

When every atom sits on its ideal site, chi depends on the species
configuration only through the number of ordered pairs of each type in each
coordination shell, N[a, b, s] (absorber a, scatterer b, shell s):

    chi_a(k) = 1/N_a  sum_{b,s} N[a,b,s] B[a,b,s](k)  +  chi_MS,a(k)

    B[a,b,s](k) = s02 f_b(k) / (k r^2) exp(-2 r/lambda(k) - 2 k^2 sigma_s^2)
                  sin(2 k r + delta_ab(k) - 4/3 k^3 C3_s),   r = r_s + dr_s[a,b]

This is exact - it agrees with the GPU pair sum to 1e-15 (tests/test_shellmodel.py)
- and it changes what is cheap: chi for any configuration is one small
contraction, the calibration objective needs no GPU, and the sensitivity of the
data to the short-range order is a small Jacobian rather than a Monte Carlo
experiment.

Three things are built on it:

  ShellModel        the forward model in pair-count space, carrying the same
                    calibration parameters as `model.Spectrum`, with
                    `from_spectrum` / `apply_to` to move between the two
  random_counts     the expectation of the pair counts for a random alloy:
                    what the calibration fits to, with no seed dependence
  crlb              Cramer-Rao bounds on the Warren-Cowley parameters - the
                    1-sigma uncertainty the data permit at a given noise
                    level, independent of any sampler. A bound above ~0.3 in
                    alpha units means the parameter is not measured, however
                    confidently a single best-fit configuration reports it.
"""
from __future__ import annotations
import numpy as np

from exafs_gpu.fourier import FourierTransform
from exafs_gpu.scattering import NSP
from . import multiscat
from .model import k_exp_axis, fit_scales, edge_scales, r_factor, target as _target
from .chik import Window, as_kweights, kweight_value, channel_edges

PARAM_NAMES = ("a0", "dE0", "kmin", "kmax", "kweight", "sigma_scale",
               "sigma_shell_scale", "dr", "c3", "ms_amp", "ms_c")


def dr_from_free(x_free, comp):
    """Symmetric (nsp, nsp) bond offsets from the free (a <= b) entries.

    The last diagonal entry is fixed by sum_ab c_a c_b dr_ab = 0, so the
    offsets describe the SPREAD of bond lengths about the mean and a0 keeps
    the mean. The number of free entries is nsp(nsp+1)/2 - 1.
    """
    comp = np.asarray(comp, float)
    nsp = comp.size
    iu = np.triu_indices(nsp)
    w = np.array([comp[a] * comp[b] * (1.0 if a == b else 2.0) for a, b in zip(*iu)])
    x = np.zeros(w.size)
    x[:-1] = np.asarray(x_free, float)
    x[-1] = -(w[:-1] @ x[:-1]) / w[-1]
    dr = np.zeros((nsp, nsp))
    dr[iu] = x
    return dr + dr.T - np.diag(np.diag(dr))


def sro_directions(nsp):
    """Independent short-range-order directions per shell.

    A symmetric change of the pair counts that conserves every absorber's
    coordination: +1 on (a,b) and (b,a), -1 on (a,a) and (b,b). There are
    nsp(nsp-1)/2 of them per shell, which is the number of independent
    Warren-Cowley parameters at fixed composition.
    """
    return [(a, b) for a in range(nsp) for b in range(a + 1, nsp)]


def dN_direction(nsp, nshell, s, a, b):
    d = np.zeros((nsp, nsp, nshell))
    d[a, b, s] += 1.0
    d[b, a, s] += 1.0
    d[a, a, s] -= 1.0
    d[b, b, s] -= 1.0
    return d


class ShellModel:
    """chi(k) and |chi(R)| from pair counts, with cheap parameter updates."""

    def __init__(self, k_phys, tables, shells_ref, shell_mult, sigma2_shell_ref,
                 natoms, a0_ref=3.566, ms_paths=None, s02=1.0, dk_win=1.0,
                 pairs=None):
        self.k = np.asarray(k_phys, float)
        _, f, delta, lam = tables.as_float64()
        self.f, self.delta, self.lam = f, delta, lam
        self.nsp = int(f.shape[0])
        self.shells_ref = np.asarray(shells_ref, float)
        self.nshell = int(self.shells_ref.size)
        self.z = np.asarray(shell_mult, float)
        self.sigma2_shell_ref = np.asarray(sigma2_shell_ref, float)
        self.natoms = int(natoms)
        self.a0_ref = float(a0_ref)
        self.ms_paths = ms_paths
        self.s02 = float(s02)
        self.dk_win = float(dk_win)
        self.pairs = pairs                      # (i_idx, j_idx, shell_idx)
        self.p = dict(a0=self.a0_ref, dE0=np.zeros(self.nsp), kmin=3.0, kmax=13.0,
                      kweight=(3,), sigma_scale=1.0,
                      sigma_shell_scale=np.ones(self.nshell),
                      dr=np.zeros((self.nsp, self.nsp)), c3=np.zeros(self.nshell),
                      ms_amp=1.0, ms_c=1.3)
        self._B = self._ms = self._ft = None
        self._B_key = self._ms_key = self._ft_key = None

    # ---------------------------------------------------------- construction
    @classmethod
    def from_spectrum(cls, S):
        m = cls(S.k_phys, S.tables, S.shells_ref, S.shell_mult, S.sigma2_shell_ref,
                S.cell.natoms, a0_ref=S.a0_ref, ms_paths=S.ms_paths, s02=S.s02,
                dk_win=S.dk_win, pairs=(S.i, S.j, S.shell_idx))
        m.set_params(a0=S.a0, dE0=S.dE0, kmin=S.kmin, kmax=S.kmax, kweight=S.kweights,
                     sigma_scale=S.sigma_scale, sigma_shell_scale=S.sigma_shell_scale,
                     dr=S.dr, c3=S.c3, ms_amp=getattr(S, "ms_amp", 1.0),
                     ms_c=getattr(S, "ms_c", 1.3))
        return m

    def apply_to(self, S):
        """Push the current parameters into a `model.Spectrum`."""
        p = self.p
        S.set_a0(p["a0"])
        S.set_sigma_scale(p["sigma_scale"], p["sigma_shell_scale"])
        S.set_dr(p["dr"])
        S.set_c3(p["c3"])
        S.set_ms(p["ms_amp"], p["ms_c"])
        S.set_window(p["kmin"], p["kmax"], p["kweight"], p["dE0"])
        return S

    def params(self):
        p = self.p
        return dict(a0=p["a0"], dE0=p["dE0"].tolist(), kmin=p["kmin"], kmax=p["kmax"],
                    kweight=kweight_value(p["kweight"]), sigma_scale=p["sigma_scale"],
                    sigma_shell_scale=p["sigma_shell_scale"].tolist(),
                    dr=p["dr"].tolist(), c3=p["c3"].tolist(),
                    ms_amp=p["ms_amp"], ms_c=p["ms_c"],
                    multiple_scattering=self.ms_paths is not None)

    def set_params(self, **kw):
        for key, v in kw.items():
            if key not in self.p:
                raise KeyError(f"unknown parameter {key!r}; have {PARAM_NAMES}")
            if v is None:
                continue
            if key == "dE0":
                v = np.asarray(v, float) * np.ones(self.nsp)
            elif key == "sigma_shell_scale":
                v = np.asarray(v, float).ravel()
                assert v.size == self.nshell, f"need {self.nshell} per-shell factors"
            elif key == "dr":
                v = np.asarray(v, float).reshape(self.nsp, self.nsp)
            elif key == "c3":
                c = np.atleast_1d(np.asarray(v, float))
                v = np.zeros(self.nshell)
                v[:min(c.size, self.nshell)] = c[:self.nshell]
            elif key == "kweight":
                v = as_kweights(v)
            else:
                v = float(v)
            self.p[key] = v
        return self

    @property
    def window(self):
        p = self.p
        return Window(p["kmin"], p["kmax"], p["kweight"], self.dk_win)

    @property
    def chan_edge(self):
        return channel_edges(self.nsp, len(self.p["kweight"]))

    @property
    def nchan(self):
        return self.nsp * len(self.p["kweight"])

    # ------------------------------------------------------------ geometry
    def radii(self):
        return self.shells_ref * (self.p["a0"] / self.a0_ref)

    def sigma2_shell(self):
        return self.sigma2_shell_ref * self.p["sigma_scale"] * self.p["sigma_shell_scale"]

    # -------------------------------------------------------------- pieces
    def basis(self):
        p = self.p
        key = (p["a0"], p["sigma_scale"], p["sigma_shell_scale"].tobytes(),
               p["dr"].tobytes(), p["c3"].tobytes())
        if self._B_key != key:
            k = self.k
            r_s, s2 = self.radii(), self.sigma2_shell()
            B = np.zeros((self.nsp, self.nsp, self.nshell, k.size))
            for s in range(self.nshell):
                ph3 = (4.0 / 3.0) * k ** 3 * p["c3"][s]
                for a in range(self.nsp):
                    for b in range(self.nsp):
                        r = r_s[s] + (p["dr"][a, b] if s == 0 else 0.0)
                        B[a, b, s] = (self.s02 * self.f[b] / (k * r * r)
                                      * np.exp(-2.0 * r / self.lam - 2.0 * k ** 2 * s2[s])
                                      * np.sin(2.0 * k * r + self.delta[a, b] - ph3))
            self._B, self._B_key = B, key
        return self._B

    def chi_ms(self):
        if self.ms_paths is None:
            return 0.0
        p = self.p
        r_nn = float(self.radii()[0])
        s2nn = float(self.sigma2_shell()[0])
        key = (r_nn, s2nn, p["ms_amp"], p["ms_c"])
        if self._ms_key != key:
            self._ms = self.ms_paths.chi_fast(
                sigma2_nn=s2nn, r_nn=r_nn, c_ms=p["ms_c"], amp=p["ms_amp"],
                s02=self.s02, a_scale=r_nn / multiscat.D_NN_FEFF)
            self._ms_key = key
        return self._ms

    def fts(self):
        """One transform per channel: (edge, k weight) pairs, edge-major."""
        p = self.p
        key = (p["dE0"].tobytes(), p["kmin"], p["kmax"], p["kweight"])
        if self._ft_key != key:
            self._ft = [FourierTransform(k_exp_axis(self.k, p["dE0"][e]), p["kmin"],
                                         p["kmax"], w, dk_win=self.dk_win, xp=np)
                        for e in range(self.nsp) for w in p["kweight"]]
            self._ft_key = key
        return self._ft

    @property
    def R(self):
        return self.fts()[0].R

    # ---------------------------------------------------------- pair counts
    def counts_of(self, species):
        """Ordered pair counts N[a, b, s] of a configuration."""
        assert self.pairs is not None, "ShellModel built without pair lists"
        i, j, sh = self.pairs
        sp = np.asarray(species, int)
        N = np.zeros((self.nsp, self.nsp, self.nshell))
        np.add.at(N, (sp[i], sp[j], sh), 1.0)
        return N

    def absorber_counts(self, N):
        """Number of absorbers of each species implied by the counts."""
        return N.sum(axis=(1, 2)) / self.z.sum()

    def random_counts(self, composition=None, counts=None):
        """Expected pair counts of a random alloy: N[a,b,s] = N_a z_s c_b."""
        if counts is None:
            comp = (np.ones(self.nsp) / self.nsp if composition is None
                    else np.asarray(composition, float))
            cnt = np.round(comp / comp.sum() * self.natoms).astype(int)
            cnt[-1] = self.natoms - cnt[:-1].sum()
        else:
            cnt = np.asarray(counts, int)
        c = cnt / cnt.sum()
        return cnt[:, None, None] * self.z[None, None, :] * c[None, :, None]

    def alpha(self, N, s=0):
        """Warren-Cowley matrix for shell s: 1 - P(b|a)/c_b."""
        cnt = self.absorber_counts(N)
        c = cnt / cnt.sum()
        return 1.0 - N[:, :, s] / np.maximum(cnt[:, None] * self.z[s] * c[None, :], 1e-30)

    # ------------------------------------------------------------- forward
    def chi_k(self, N):
        cnt = np.maximum(self.absorber_counts(N), 1e-30)
        return np.einsum("abs,absk->ak", N, self.basis()) / cnt[:, None] + self.chi_ms()

    def chi_R(self, N, idx=None):
        """Complex chi(R) per channel on the FFT grid, or on the data grid via idx."""
        chi = self.chi_k(N)
        out = np.array([ft.to_R(chi[e]) for ft, e in zip(self.fts(), self.chan_edge)])
        return out if idx is None else out[:, idx]

    def mag_R(self, N, idx=None):
        """|chi(R)| per channel on the FFT grid, or on the data grid via idx."""
        return np.abs(self.chi_R(N, idx))

    def target(self, data):
        """The comparison array for `data` (a |chi(R)| array or a ChiK), see
        `model.target`; a ChiK is transformed with this model's window."""
        return _target(data, self.window)

    def observe(self, N, tgt, idx=None):
        return self.chi_R(N, idx) if np.iscomplexobj(tgt) else self.mag_R(N, idx)

    def r_factor(self, N, data, mask, idx, return_scales=False):
        """R-factor of the pair counts N against `data`, with one amplitude
        per edge profiled out. Returns the per-edge amplitudes on request."""
        d = self.target(data)
        m = self.observe(N, d, idx)
        sc = fit_scales(m, d, mask, self.chan_edge)
        rf = r_factor(m, d, mask, sc)
        return (rf, edge_scales(sc, self.chan_edge)) if return_scales else rf


# ================================================================ Fisher
PRIOR_SD = dict(alpha=1.0, a0=0.02, sigma_scale=0.5, ms_amp=0.5, ms_c=0.5, c3=2e-4,
                dE0=5.0, dr=0.05)
NUISANCE_DEFAULT = ("a0", "sigma_scale", "ms_amp", "dE0", "dr", "c3")


def crlb(model, N, data, sigma, mask, idx, shells=None, alpha_shell=0,
         nuisance=NUISANCE_DEFAULT, profile_scale=True,
         threshold=0.3, steps=None, prior_sd=None):
    """Cramer-Rao bounds on the shell-`alpha_shell` Warren-Cowley parameters.

    Linearises the target - |chi(R)|, or complex chi(R) for chi(k) data, in
    which case Re and Im are separate residuals - around the pair counts N
    along the SRO directions of every shell in `shells` (default: all - which
    is what the sampler frees), plus the nuisance parameters, at noise level
    `sigma` (per channel, or per channel and R point, in DATA units; the
    per-component sd for a complex target). The per-edge amplitude is
    profiled out exactly as in the fit. `data` may be the array or a ChiK.

    The nuisances matter: the bond offsets dr and the per-edge E0 can mimic
    part of a short-range-order signal (a Cr-avoiding arrangement shortens
    the mean Cr-X bond just as a Cr-X offset would), and the closed-loop test
    shows a random-alloy calibration absorbing order that way. The default
    list therefore includes everything the calibration fits that the RMC
    could trade against: a0, the sigma^2 scale, the MS amplitude, dE0 per
    edge, the five free bond offsets and C3.

    Directions the data do not see at all (shells beyond the R window, say)
    would make the plain Fisher matrix singular, so a weak Gaussian prior is
    added: 1.0 in alpha units on every SRO direction, and generous widths on
    the nuisances (PRIOR_SD). The reported 1-sigma is then the posterior width
    under that prior: a value near 1 means "prior only", i.e. not measured.

    Returns the bounds in alpha units for the six (a <= b) entries of the
    chosen shell, marginalised over everything else, plus the eigen-spectrum
    of the data-only SRO block (in units of the finite-difference step).
    """
    nsp, nshell = model.nsp, model.nshell
    shells = list(range(nshell)) if shells is None else list(shells)
    data = model.target(data)
    cplx = np.iscomplexobj(data)
    nchan, chan_edge = model.nchan, model.chan_edge
    assert data.shape[0] == nchan, f"target has {data.shape[0]} channels, model {nchan}"
    sig = np.asarray(sigma, float)
    if sig.ndim == 1:
        sig = np.repeat(sig[:, None], data.shape[1], axis=1)
    cnt = model.absorber_counts(N)
    c = cnt / cnt.sum()
    iu = np.triu_indices(nsp)
    st = dict(a0=0.002, sigma_scale=0.05, ms_amp=0.05, ms_c=0.05, c3=2e-5,
              dE0=0.5, dr=0.005)
    if steps:
        st.update(steps)
    pr = dict(PRIOR_SD)
    if prior_sd:
        pr.update(prior_sd)

    def resid(Nx):
        m = model.observe(Nx, data, idx)
        sc = fit_scales(m, data, mask, chan_edge) if profile_scale else np.ones(nchan)
        r = (m * sc[:, None] - data)[:, mask] / sig[:, mask]
        return np.concatenate([r.real.ravel(), r.imag.ravel()]) if cplx else r.ravel()

    cols, A, names, prec = [], [], [], []
    for s in shells:
        h = 0.02 * cnt.min() * model.z[s] * c.min()     # ~0.02 in alpha units
        for a, b in sro_directions(nsp):
            d = dN_direction(nsp, nshell, s, a, b)
            cols.append((resid(N + h * d) - resid(N - h * d)) / (2 * h))
            names.append(f"sro_s{s + 1}_{a}{b}")
            dalpha = (model.alpha(N + h * d, s) - model.alpha(N, s)) / h
            A.append(dalpha[iu] if s == alpha_shell else np.zeros(len(iu[0])))
            # prior: alpha sd of pr["alpha"] on the entry this direction moves most
            prec.append((np.abs(dalpha).max() / pr["alpha"]) ** 2)
    nsro = len(cols)
    saved = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in model.p.items()}

    def column(name, hi_, lo_, step):
        model.set_params(**{name: hi_}); rp = resid(N)
        model.set_params(**{name: lo_}); rm = resid(N)
        model.set_params(**{name: saved[name]})
        cols.append((rp - rm) / (2 * step))
        prec.append(1.0 / pr[name] ** 2)

    ndr_free = nsp * (nsp + 1) // 2 - 1
    for name in nuisance or ():
        v0 = saved[name]
        if name == "a0":
            column(name, v0 + st[name], v0 - st[name], st[name]); names.append(name)
        elif name == "c3":
            hi_ = v0.copy(); hi_[0] += st[name]
            lo_ = v0.copy(); lo_[0] -= st[name]
            column(name, hi_, lo_, st[name]); names.append(name)
        elif name == "dE0":                      # one column per edge
            for e in range(nsp):
                hi_ = v0.copy(); hi_[e] += st[name]
                lo_ = v0.copy(); lo_[e] -= st[name]
                column(name, hi_, lo_, st[name]); names.append(f"dE0_{e}")
        elif name == "dr":                       # the free, mean-zero directions
            for q in range(ndr_free):
                e = np.zeros(ndr_free); e[q] = st[name]
                d = dr_from_free(e, c)
                column(name, v0 + d, v0 - d, st[name]); names.append(f"dr_{q}")
        else:                                    # multiplicative: log steps
            column(name, v0 * np.exp(st[name]), v0 * np.exp(-st[name]), st[name])
            names.append(name)
    J = np.array(cols).T
    F = J.T @ J
    cov = np.linalg.inv(F + np.diag(prec))
    A = np.array(A)                                      # (nsro, 6)
    cov_alpha = A.T @ cov[:nsro, :nsro] @ A
    sd = np.sqrt(np.maximum(np.diag(cov_alpha), 0.0))
    ev = np.linalg.eigvalsh(F[:nsro, :nsro])[::-1]
    with np.errstate(divide="ignore"):
        eig_sd = 1.0 / np.sqrt(np.maximum(ev, 1e-300))
    # marginal 1-sigma of the nuisances themselves: a0 and dr in Angstrom,
    # dE0 in eV, C3 in Angstrom^3, the multiplicative ones in log units
    csd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    nuis_sd = {}
    for j, name in enumerate(names[nsro:], start=nsro):
        key = name.split("_")[0] if name.startswith(("dE0_", "dr_")) else name
        nuis_sd.setdefault(key, []).append(float(csd[j]))
    dr_entry_sd = None
    if "dr" in nuis_sd:
        q = [j for j, n in enumerate(names) if n.startswith("dr_")]
        D = np.array([dr_from_free(np.eye(ndr_free)[i], c)[iu] for i in range(ndr_free)])
        dr_entry_sd = np.sqrt(np.maximum(np.einsum("ie,ij,je->e", D, cov[np.ix_(q, q)], D),
                                         0.0)).tolist()
    return dict(alpha_sd=sd.tolist(), determined=(sd < threshold).tolist(),
                threshold=threshold, prior_alpha_sd=pr["alpha"],
                shells=[int(s) + 1 for s in shells],
                nuisance=list(nuisance or ()), n_sro_directions=int(nsro),
                eig_sd=eig_sd.tolist(), columns=names,
                nuisance_sd={k: (v[0] if len(v) == 1 else v) for k, v in nuis_sd.items()},
                dr_entry_sd=dr_entry_sd)
