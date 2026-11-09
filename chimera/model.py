"""Forward model for the measured data: configuration -> |chi(R)| on the data grid.

What has to be calibrated, and why
----------------------------------
The files are |chi(R)| only. Everything the experimenters did between mu(E)
and that magnitude - where they put E0, which k window they transformed, what
k weight, how the edge step was normalised - is not recorded, and each of
those changes the R-space result. So they are inferred from the data itself,
once per pressure, by fitting an ideal FCC model to the location-averaged
spectra:

    dE0     an error in the threshold energy means the experimental k axis is
            not the physical one: k_exp^2 = k_phys^2 - ETOK * dE0. The model
            is computed on k_phys and transformed against k_exp, which is
            what happened to the data. One value per EDGE, because each edge
            had its E0 chosen separately. k_phys is the internal momentum,
            so with this constant gap dE0 also carries the inner potential,
            which really falls by ~9 eV over the fit range (the self-energy);
            the system's k_axis "table" maps through the scattering
            calculation's own k(p) instead and leaves dE0 the E0 offset
            alone (`k_exp_axis`).
    a0      lattice constant. Sets every path length, so it moves all the
            peaks together; strongly determined by the first shell.
    scale   one amplitude factor per edge, absorbing S0^2 and the edge-step
            normalisation. Standard EXAFS practice; it cannot manufacture
            short-range order, which lives in the R dependence.
    sig_s   multiplier on the harmonic Debye-Waller factor - one number
            controlling the whole shell-dependent sigma^2 set, since
            sigma^2 is linear in kB T / k_bond. `sigma_shell_scale` adds an
            optional per-shell multiplier on top, because a single pair of
            force constants does not get every shell's MSRD right.
    dr      nearest-neighbour bond-length offset per pair type (Angstrom,
            symmetric 3x3, composition-weighted mean zero so it does not
            duplicate a0). This is the size-mismatch term: in a random alloy
            the Cr-Cr, Cr-Ni and Ni-Ni bonds are not the same length, and a
            model that puts every first-shell pair at one distance has no way
            to express that except by faking it with the species arrangement.
    C3      third cumulant of the first-shell distance distribution
            (Angstrom^3): the anharmonic phase term -4/3 k^3 C3, standard in
            EXAFS at room temperature and above.
    window  (kmin, kmax) of the Hanning window, and the k weight. These set
            the R-space resolution and the sidelobes, so they are visible in
            the peak shapes.

The species configuration is NOT part of the calibration: it is fitted
afterwards, with these held fixed.

Two kinds of target
-------------------
The comparison happens in R space either way, but what is compared depends on
what was measured (see `chik.py`):

    |chi(R)| files   the model's chi(k) is transformed against each edge's
                     experimental k axis and its MAGNITUDE compared on the
                     data grid. The window and k weight are calibration
                     parameters, because they were the experimenters' choices
                     and are not recorded.
    chi(k) files     the data are transformed with the SAME window and k
                     weight(s) as the model and the COMPLEX chi(R) is compared,
                     Re and Im. The window is then our choice, and several k
                     weights can be fitted at once: each (edge, k weight) pair
                     is one CHANNEL of the target, ordered edge-major
                     (`chan_edge` gives the edge of every channel), with the
                     per-edge amplitude profiled over all channels of an edge.

`target(data, S)` produces the array to compare against for either kind and
`observe(S, species, target)` the matching model array; `fit_scales`,
`chi2` and `r_factor` accept both real and complex arrays.

Cheap parameter updates
-----------------------
The expensive pieces (neighbour list, force-constant eigendecomposition) do
not depend on a0, dE0 or the window: force constants are built from bond
DIRECTIONS, so sigma^2 is scale free, and the neighbour topology of a
uniformly scaled lattice is unchanged. `set_a0`, `set_dE0`, `set_sigma_scale`
and `set_window` therefore only touch small arrays, which is what makes a
grid search over the calibration affordable.
"""
from __future__ import annotations
import numpy as np
import cupy as cp

from exafs_gpu.scattering import (feff8l_tables, analytic_transition_metal_tables,
                                  ScatteringTables, ELEMENTS, NSP, is_native)
from exafs_gpu.lattice import (CrystalSupercell, force_constant_matrix,
                               warren_cowley, pair_vectors)
from . import system as SYS
from exafs_gpu.forward import ForwardModel, thermal_sigma2
from exafs_gpu.fourier import FourierTransform
from . import multiscat
from .chik import (ChiK, Window, as_kweights, kweight_value, channel_edges,
                   channel_labels, make_window)

ETOK = 0.2624682843        # k^2 [A^-2] per eV
NKMODEL = 320
K_LO, K_HI = 0.5, 18.0


def native_paths(system, name):
    """The native engine's path records for `system`, at its reference
    spacing, with the recipe the table-set `name` spells out ("native" or
    "native:seed=1,..."). Computed once per process and cached on disk."""
    from exafs_gpu.abinitio import provider
    return provider.path_records(system.elements, system.composition, system.lattice,
                                 system.nn_ref, system.c_over_a,
                                 recipe=provider.Recipe.parse(name))


def scattering_provenance(system):
    """Where a fit's scattering came from, for calibration and summary records:
    the table-set names, and for in-process native paths the engine version,
    recipe, reference spacing and muffin-tin summary."""
    out = dict(tables=system.tables_name, ms=system.ms_name, k_axis=system.k_axis)
    name = next((n for n in (system.tables_name, system.ms_name) if is_native(n)), None)
    if name is not None:
        out["native"] = native_paths(system, name).provenance()
    return out


def k_exp_axis(k_phys, dE0, k_table=None):
    """The k axis the experimenter used, given their E0 error.

    Without `k_table` the gap between the data's k and the internal momentum
    is one constant, dE0 (the system's k_axis "constant"). With it -
    `ScatteringTables.k_of_p` for this edge on the same grid - the
    calculation's own energy-dependent k(p) is used and dE0 is only the
    offset between the experimenter's E0 and the calculation's Fermi level
    (k_axis "table")."""
    k0 = np.asarray(k_phys) if k_table is None else np.asarray(k_table)
    return np.sqrt(np.maximum(k0 ** 2 - ETOK * dE0, 1e-4))


def k_table_for(system, tables, k_grid):
    """Per-edge k(p) on `k_grid` when the system maps k through the tables
    (k_axis "table"), else None. Tables tabulated on p alone (the analytic
    stand-in) have no Fermi-level reference: k = p."""
    if getattr(system, "k_axis", "constant") != "table":
        return None
    kp = getattr(tables, "k_of_p", None)
    if kp is None:
        return np.tile(np.asarray(k_grid, float), (tables.nsp, 1))
    return np.array([np.interp(k_grid, tables.k, row) for row in kp])


def dE0_centre(system):
    """Where a calibration starts its per-edge dE0 search: with the constant
    gap dE0 carries the whole inner potential (~12 eV for these tables);
    through the table only the E0 offset, around zero."""
    return 0.0 if getattr(system, "k_axis", "constant") == "table" else 12.0


class Spectrum:
    """Configuration -> |chi(R)|, with cheap calibration updates.

    chi(k) is always computed on a fixed PHYSICAL momentum grid, so changing
    dE0 rebuilds nothing but the (small) Fourier transforms: the E0 error is
    carried entirely by which k axis each edge is transformed against.
    """

    def __init__(self, a0=None, dE0=0.0, kmin=3.0, kmax=14.0,
                 kweight=2, sigma_scale=1.0, k_bond=2.6, k_angle=0.35,
                 temperature=300.0, ncell=(5, 5, 5), rmax=6.2, dk_win=1.0,
                 s02=1.0, tables_name=None, ms=True, ms_amp=1.0,
                 ms_c=1.3, ms_name=None, sigma_shell_scale=None,
                 dr=None, c3=None, system=None, elements=None, composition=None,
                 lattice=None, a0_ref=None, c_over_a=None, tables=None):
        """`system` names the alloy (an AlloySystem, a table-set name, a dict
        from a calibration record, or None for equiatomic fcc CrCoNi);
        `elements`, `composition`, `lattice`, `a0_ref`, `c_over_a`,
        `tables_name` and `ms_name` override its fields. `tables` is None
        for the set the system names (a FEFF8L or native_tables.py
        directory, or "native" for paths computed in-process with no
        files), "analytic" for the Z-parameterised stand-in (tests), or a
        ScatteringTables. `a0`
        defaults to the system's reference lattice constant, `dE0` (one
        value, or one per edge) to zero.
        """
        self.system = SYS.resolve(system, elements=elements, composition=composition,
                                  lattice=lattice, a0_ref=a0_ref, c_over_a=c_over_a,
                                  tables_name=tables_name, ms_name=ms_name)
        self.elements = self.system.elements
        self.nsp = self.system.nsp
        self.a0_ref = self.system.a0_ref
        self.cell = CrystalSupercell(self.system.lattice, ncell, a0=self.a0_ref,
                                     c_over_a=self.system.c_over_a)
        self.i, self.j, self.shift0, self.off = self.cell.neighbour_list(rmax)
        self.ideal0 = self.cell.ideal.copy()
        self.H0 = self.cell.H.copy()
        self.tables_name, self.s02, self.dk_win = self.system.tables_name, s02, dk_win
        self.rmax = rmax

        Phi = force_constant_matrix(self.cell, self.i, self.j, self.shift0,
                                    k_bond=k_bond, k_angle=k_angle)
        self.sigma2_ref = thermal_sigma2(Phi, self.cell, self.i, self.j,
                                         self.shift0, temperature)
        self.Phi = Phi
        # shell membership of every pair, and the exact (not rounded) radii
        d0 = np.linalg.norm((self.ideal0[self.j] + self.shift0) - self.ideal0[self.i],
                            axis=1)
        _, self.shell_idx = np.unique(np.round(d0, 3), return_inverse=True)
        self.shell_idx = self.shell_idx.astype(np.int32)
        self.nshell = int(self.shell_idx.max()) + 1
        self.shells_ref = np.array([d0[self.shell_idx == s].mean()
                                    for s in range(self.nshell)])
        self.shell_mult = np.array([(self.shell_idx == s).sum() / self.cell.natoms
                                    for s in range(self.nshell)])
        self.sigma2_shell_ref = np.array([self.sigma2_ref[self.shell_idx == s].mean()
                                          for s in range(self.nshell)])

        self.k_phys = np.linspace(K_LO, K_HI, NKMODEL)
        if isinstance(tables, ScatteringTables):
            self.tables = tables
        elif tables == "analytic" or (tables is None and self.tables_name == "analytic"):
            # the Z-parameterised stand-in, also reachable by naming the
            # table set "analytic" in the system (tests, dry runs)
            self.tables = analytic_transition_metal_tables(self.k_phys, s02=s02,
                                                           elements=self.elements)
        elif is_native(self.tables_name):
            # computed in-process by the native engine for this system; no files
            self.tables = native_paths(self.system, self.tables_name).ss_tables(self.k_phys, s02)
        else:
            if self.tables_name is None:
                raise ValueError(f"no FEFF tables named for {self.system.name}: pass "
                                 f"tables_name= (see feff8l_tables.py) or tables=")
            self.tables = feff8l_tables(self.k_phys, name=self.tables_name,
                                        elements=self.elements, s02=s02,
                                        composition=self.system.composition)
        assert self.tables.nsp == self.nsp, (
            f"tables hold {self.tables.nsp} species, the system {self.nsp}")
        self._k_table = k_table_for(self.system, self.tables, self.k_phys)
        self.fm = ForwardModel(self.cell, self.tables, self.i, self.j,
                               self.shift0, sigma2=self.sigma2_ref,
                               shell=self.shell_idx)
        self.R = None
        self.ms_paths = None
        if ms and is_native(self.system.ms_name):
            npth = native_paths(self.system, self.system.ms_name)
            self.ms_paths = multiscat.MSPaths.from_records(
                npth.ms, self.elements, npth.d_nn, name=self.system.ms_name).prepare(self.k_phys)
        elif ms and self.system.ms_name:
            self.ms_paths = (multiscat.get(self.system.ms_name, elements=self.elements)
                             .prepare(self.k_phys))
        self.chi_ms = None
        self.sigma_shell_scale = np.ones(self.nshell)
        self.set_a0(self.a0_ref if a0 is None else a0)
        self.set_sigma_scale(sigma_scale, sigma_shell_scale)
        self.set_dr(dr)
        self.set_c3(c3)
        self.set_ms(ms_amp, ms_c)
        self.set_window(kmin, kmax, kweight, dE0)

    # ------------------------------------------------------------- parameters
    def set_a0(self, a0):
        self.a0 = float(a0)
        s = self.a0 / self.a0_ref
        self.cell.ideal = self.ideal0 * s
        self.cell.H = self.H0 * s
        self.cell.a0 = self.a0
        self.shift = self.shift0 * s
        self.shells = self.shells_ref * s
        self.fm.d_shift = cp.asarray(np.ascontiguousarray(self.shift, np.float64))
        self._pos = cp.asarray(self.cell.ideal[None])
        if getattr(self, "ms_paths", None) is not None and self.chi_ms is not None:
            self.set_ms()
        return self

    def set_sigma_scale(self, s, per_shell=None):
        """Global sigma^2 multiplier, optionally times a per-shell factor."""
        self.sigma_scale = float(s)
        if per_shell is not None:
            ps = np.asarray(per_shell, float).ravel()
            assert ps.size == self.nshell, f"need {self.nshell} per-shell factors"
            self.sigma_shell_scale = ps.copy()
        self.sigma2 = (self.sigma2_ref * self.sigma_scale
                       * self.sigma_shell_scale[self.shell_idx])
        self.fm.d_sig2 = cp.asarray(np.ascontiguousarray(self.sigma2, np.float64))
        if getattr(self, "ms_paths", None) is not None and self.chi_ms is not None:
            self.set_ms(self.ms_amp, self.ms_c)
        return self

    def set_dr(self, dr=None):
        """Nearest-neighbour bond-length offsets by pair type, (nsp, nsp) A."""
        self.dr = (np.zeros((self.nsp, self.nsp)) if dr is None
                   else np.asarray(dr, float).reshape(self.nsp, self.nsp))
        self.fm.set_dr(self.dr)
        return self

    def set_c3(self, c3=None):
        """Third cumulant per shell (A^3); a scalar means the first shell."""
        c3 = np.zeros(self.nshell) if c3 is None else np.atleast_1d(np.asarray(c3, float))
        full = np.zeros(self.nshell)
        full[:c3.size] = c3[:self.nshell]
        self.c3 = full
        self.fm.set_c3(full)
        return self

    def set_ms(self, amp=None, c_ms=None):
        """Rebuild the composition-averaged multiple-scattering chi(k)."""
        if amp is not None:
            self.ms_amp = float(amp)
        if c_ms is not None:
            self.ms_c = float(c_ms)
        if self.ms_paths is None:
            self.chi_ms = None
            return self
        self.chi_ms = self.ms_paths.chi_fast(
            sigma2_nn=self.sigma2_nn(), r_nn=float(self.shells[0]),
            c_ms=self.ms_c, amp=self.ms_amp, s02=self.s02,
            a_scale=float(self.shells[0]) / self.ms_paths.d_nn)
        return self

    def sigma2_nn(self):
        d = np.linalg.norm((self.cell.ideal[self.j] + self.shift)
                           - self.cell.ideal[self.i], axis=1)
        return float(self.sigma2[np.abs(d - self.shells[0]) < 0.05].mean())

    def set_window(self, kmin=None, kmax=None, kweight=None, dE0=None):
        """Window, k weight(s) and per-edge E0. `kweight` may be one integer
        or several; every (edge, k weight) pair becomes one transform, and
        `chan_edge` records the edge of each."""
        if kmin is not None:
            self.kmin = float(kmin) if np.isscalar(kmin) else np.asarray(kmin, float)
        if kmax is not None:
            self.kmax = float(kmax) if np.isscalar(kmax) else np.asarray(kmax, float)
        if kweight is not None:
            self.kweights = as_kweights(kweight)
        if dE0 is not None:
            self.dE0 = np.atleast_1d(np.asarray(dE0, float)) * np.ones(self.nsp)
        self.ft = []
        lo = np.atleast_1d(self.kmin); hi = np.atleast_1d(self.kmax)
        for e in range(self.nsp):
            k_exp = self.k_exp(e)
            for w in self.kweights:
                self.ft.append(FourierTransform(k_exp, float(lo[e % lo.size]),
                                                float(hi[e % hi.size]), w,
                                                dk_win=self.dk_win, xp=cp))
        self.chan_edge = channel_edges(self.nsp, len(self.kweights))
        self.nchan = len(self.ft)
        self.R = self.ft[0].R
        return self

    @property
    def kweight(self):
        """The int for a single k weight, a list for several (JSON form)."""
        return kweight_value(self.kweights)

    @property
    def window(self):
        return Window(self.kmin, self.kmax, self.kweights, self.dk_win)

    def k_exp(self, e):
        """The experimental k axis of edge e (its E0 error applied)."""
        return k_exp_axis(self.k_phys, self.dE0[e],
                          None if self._k_table is None else self._k_table[e])

    def channel_labels(self):
        return channel_labels(self.elements, self.kweights)

    def params(self):
        return dict(system=self.system.to_dict(),
                    a0=self.a0, dE0=[float(x) for x in self.dE0],
                    kmin=self.kmin, kmax=self.kmax, kweight=self.kweight,
                    sigma_scale=self.sigma_scale,
                    sigma_shell_scale=self.sigma_shell_scale.tolist(),
                    dr=self.dr.tolist(), c3=self.c3.tolist(),
                    multiple_scattering=self.ms_paths is not None,
                    ms_amp=getattr(self, "ms_amp", None),
                    ms_c=getattr(self, "ms_c", None),
                    nn_distance=float(self.shells[0]))

    @classmethod
    def from_cal(cls, cal, **overrides):
        """Build a Spectrum from a calibration dict (as written to
        calibration.json / fit.json), with keyword overrides on top."""
        sysd = cal.get("system") or cal.get("shared", {}).get("system")
        kw = dict(system=SYS.AlloySystem.from_dict(sysd) if sysd else None,
                  a0=cal["a0"], dE0=cal["dE0"], kmin=cal["kmin"], kmax=cal["kmax"],
                  kweight=cal["kweight"], sigma_scale=cal["sigma_scale"],
                  ms=bool(cal.get("multiple_scattering", True)),
                  ms_amp=cal.get("ms_amp", 1.0), ms_c=cal.get("ms_c", 1.3),
                  sigma_shell_scale=cal.get("sigma_shell_scale"),
                  dr=cal.get("dr"), c3=cal.get("c3"))
        kw.update(overrides)
        return cls(**kw)

    # ------------------------------------------------------------- evaluation
    def _chi_device(self, species, pos=None):
        sp = cp.asarray(np.atleast_2d(np.asarray(species, np.int32)))
        if pos is None:
            p = self._pos
        else:
            p = cp.asarray(np.asarray(pos, float)[None] if np.ndim(pos) == 2 else pos)
        chi = self.fm.chi(p, sp)          # (1, nsp, nk), indexed by absorber
        if self.chi_ms is not None:
            chi = chi + cp.asarray(self.chi_ms)[None]
        return chi

    def chi_k(self, species, pos=None):
        """chi(k) (nsp, nk) on the PHYSICAL momentum grid `k_phys`, indexed by
        absorber, multiple scattering included. Plot edge e against
        `k_exp(e)` to put it on the experiment's axis."""
        return cp.asnumpy(self._chi_device(species, pos))[0]

    def chi_R(self, species, pos=None, complex_out=False):
        """chi(R) (nchan, nR) for one configuration: |chi(R)| by default, the
        complex transform with `complex_out`. Channels are (edge, k weight)
        pairs, edge-major - one per edge for a single k weight."""
        chi = self._chi_device(species, pos)
        out = []
        for f, e in zip(self.ft, self.chan_edge):
            z = f.to_R(chi[:, e:e + 1])[0, 0]
            out.append(cp.asnumpy(z if complex_out else cp.abs(z)))
        return np.array(out)

    def random_config(self, seed=0, composition=None):
        """A random arrangement at the system's composition (or `composition`),
        species counts rounded with the last species taking the remainder."""
        n = self.cell.natoms
        comp = self.system.composition if composition is None else composition
        cnt = SYS.integer_counts(comp, n)
        sp = np.repeat(np.arange(self.nsp), cnt).astype(np.int32)
        np.random.default_rng(seed).shuffle(sp)
        return sp

    def warren_cowley(self, species, shell=0):
        return warren_cowley(np.asarray(species, np.int32), self.i, self.j,
                             self.shift, self.cell, float(self.shells[shell]),
                             nsp=self.nsp)


# --------------------------------------------------------------- comparison
def match_grid(R_model, R_data):
    """Indices of the model R grid matching the data grid (they share dR)."""
    idx = np.searchsorted(R_model, R_data - 1e-9)
    assert np.allclose(R_model[idx], R_data, atol=1e-6), "R grids differ"
    return idx


def target(data, window):
    """The array a fit compares against.

    A |chi(R)| array is returned as given. A `ChiK` is transformed with
    `window` - a Spectrum, ShellModel or Window - into complex chi(R) on the
    model's own R grid, so `match_grid(S.R, data.R)` is the identity and the
    fit mask is taken on that grid.
    """
    if isinstance(data, ChiK):
        return data.to_R(make_window(window))
    return np.asarray(data)


def is_chik(data):
    return isinstance(data, ChiK)


def observe(S, species, target, pos=None):
    """The model array on the same footing as `target` (real or complex)."""
    return S.chi_R(species, pos=pos, complex_out=np.iscomplexobj(target))


def fit_scales(model, data, mask, edge=None):
    """Least-squares amplitude per channel (closed form), real or complex.

    With `edge` (the edge index of every channel) one amplitude is fitted per
    EDGE over all of its channels and broadcast back to the channels, so
    several k weights of the same edge share their S0^2.
    """
    m, d = model[:, mask], data[:, mask]
    num = np.real((np.conj(m) * d).sum(axis=1))
    den = np.real((np.conj(m) * m).sum(axis=1))
    if edge is not None:
        edge = np.asarray(edge, int)
        ne = int(edge.max()) + 1
        num = np.bincount(edge, num, ne)[edge]
        den = np.bincount(edge, den, ne)[edge]
    return num / np.maximum(den, 1e-30)


def edge_scales(scales, edge=None):
    """One value per edge from per-channel scales (the first channel of each)."""
    scales = np.asarray(scales, float)
    if edge is None:
        return scales
    edge = np.asarray(edge, int)
    first = [int(np.argmax(edge == e)) for e in range(int(edge.max()) + 1)]
    return scales[first]


def chi2(model, data, mask, sigma, scales=None):
    """Mean squared residual in units of sigma, counted per REAL component:
    a complex target contributes Re and Im separately and sigma is the
    per-component sd."""
    m = model if scales is None else model * np.asarray(scales)[:, None]
    d = (m[:, mask] - data[:, mask]) / np.asarray(sigma)[:, None]
    n = d.size * (2 if np.iscomplexobj(d) else 1)
    return float((np.abs(d) ** 2).sum() / n)


def r_factor(model, data, mask, scales=None):
    """EXAFS R factor: sum |data - fit|^2 / sum |data|^2 over the fit range."""
    m = model if scales is None else model * np.asarray(scales)[:, None]
    num = (np.abs(m[:, mask] - data[:, mask]) ** 2).sum()
    den = (np.abs(data[:, mask]) ** 2).sum()
    return float(num / max(den, 1e-30))


# --------------------------------------------------------------- structure out
def partial_rdf(species, cell, i_idx, j_idx, shift, r_edges, nsp=None):
    """Partial radial distribution functions g_ab(r) from a configuration.

    g_ab(r) = n_ab(r) / (N_a * rho * c_b * V_shell): unity for a random
    arrangement at large r, and the Warren-Cowley parameters are the shell
    integrals of 1 - g_ab/g_total.
    """
    sp = np.asarray(species, int)
    nsp = int(sp.max()) + 1 if nsp is None else int(nsp)
    d = pair_vectors(cell, cell.ideal, i_idx, j_idx, shift)
    r = np.linalg.norm(d, axis=1)
    a, b = sp[i_idx], sp[j_idx]
    counts = np.bincount(sp, minlength=nsp)
    c = counts / sp.size
    rho = sp.size / abs(np.linalg.det(cell.H))
    centres = 0.5 * (r_edges[1:] + r_edges[:-1])
    shell_vol = 4.0 / 3.0 * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    g = np.zeros((nsp, nsp, centres.size))
    for A in range(nsp):
        for B in range(nsp):
            m = (a == A) & (b == B)
            h, _ = np.histogram(r[m], bins=r_edges)
            g[A, B] = h / np.maximum(counts[A] * rho * c[B] * shell_vol, 1e-30)
    tot, _ = np.histogram(r, bins=r_edges)
    g_tot = tot / np.maximum(sp.size * rho * shell_vol, 1e-30)
    return centres, g, g_tot


def coordination(species, cell, i_idx, j_idx, shift, r_cut, nsp=None):
    """N_ab: mean number of B neighbours of an A atom within r_cut."""
    sp = np.asarray(species, int)
    nsp = int(sp.max()) + 1 if nsp is None else int(nsp)
    d = pair_vectors(cell, cell.ideal, i_idx, j_idx, shift)
    r = np.linalg.norm(d, axis=1)
    m = r <= r_cut
    a, b = sp[i_idx[m]], sp[j_idx[m]]
    counts = np.bincount(sp, minlength=nsp)
    N = np.zeros((nsp, nsp))
    for A in range(nsp):
        for B in range(nsp):
            N[A, B] = ((a == A) & (b == B)).sum() / max(counts[A], 1)
    return N
