"""Infer the unrecorded processing parameters, and the regime of validity.

Regime of validity (measured, not assumed)
------------------------------------------
Fitting an ideal random FCC model to the 0 GPa location-averaged spectra and
looking at the residual band by band, WITH the multiple-scattering term:

    R band (A)   R-factor        what is there
    0.6 - 1.2      0.13-0.80  background / atomic residual: not structure
    1.2 - 1.4      0.08-0.43  rising flank of shell 1, still contaminated
    1.4 - 1.8      0.04-0.13  shell 1 flank
    1.8 - 2.6      0.006-0.015  shell 1
    2.6 - 3.3      0.04-0.09  the trough between peaks
    3.3 - 4.1      0.02-0.05  shells 2 and 3
    4.1 - 4.8      0.001-0.03  shell 4 - only once MS is included
    4.8 - 5.0      0.002-0.07  tail of shell 4
    5.0 +          0.10-0.85  shell 5 and beyond: not modelled

    FIT RANGE  R = 1.4 - 5.0 A   (shells 1-4)

Without multiple scattering the model has to stop at 4.2 A: in fcc, twice the
nearest-neighbour distance equals the fourth-shell radius to one part in 5000,
so the forward-focused collinear path lands exactly on the fourth-shell peak
and carries about 11x the single-scattering amplitude there. A pair-only model
is not slightly wrong in that region, it is missing most of the intensity -
R-factor 0.40 while the data is 25x the noise. `multiscat.py` adds FEFF8L's
3- and 4-leg paths and the same band drops to 0.005.

How the calibration is done
---------------------------
The random-alloy model is linear in the pair counts, so the whole objective
runs on the CPU in a few milliseconds (`shellmodel.ShellModel`) and the search
can afford to be global: differential evolution over every continuous
processing and model parameter at once, repeated for each candidate k weight,
scored by the R-factor of the random-alloy expectation against the 0 GPa
location median with the per-edge amplitude profiled out.

    stage 1 (0 GPa, a0 anchored at the literature value)
        dE0 per edge, the k window (kmin, kmax), the sigma^2 scale and the
        per-shell sigma^2 factors, the pair-type bond-length offsets dr, the
        first-shell third cumulant C3, and the MS amplitude and Debye-Waller
        scale. Two passes: a "core" search without dr, C3 and the per-shell
        factors locates the E0 / window basin, then the full search starts
        from it (a modest budget without that continuation stalls with the
        offsets pinned at their bounds).
    stage 2 (every pressure)
        a0 and the global sigma^2 scale, everything else held; a0 stays at
        the anchor at the anchor pressure.
    stage 3 (joint, when more than one pressure is available)
        the shared parameters again, scored on the MEAN R-factor over every
        pressure with the per-pressure a0 and sigma^2 held, then stage 2
        once more. All pressures were measured on the same beamline with the
        same edges, so the processing choices are common to them; using all
        of them matters here because at 0 GPa the Co and Ni files hold the
        same spectrum, which no physical model can fit consistently on its own.

    The whole sequence runs once per k weight in `kweights` and the k weight
    with the lowest joint score is kept.

The k weight, k range and E0 are genuine processing choices the experimenters
made and did not record; if the Athena project files turn up, the values in
them replace stage 1 outright. The earlier manual scan that fixed k^3 and
3-13 A^-1 is what `calibrate_legacy` still does.

With chi(k) data (`chik.ChiK` inputs) two of those choices become ours:
the window is applied identically to data and model, so it is held at the
values given (`free_window` is switched off - searching it would only tune
a weighting to the model), and the k weight defaults to fitting k^1, k^2 and
k^3 at once (`KWEIGHTS_CHIK`) instead of scanning. E0 per edge stays a
fitted parameter - now determined by the phase rather than traded against
a0 - and `anchor_a0=False` lets the ambient lattice constant float with it,
which is only sensible once the phase is there to separate the two.

The a0 / E0 degeneracy
----------------------
The first-shell peak position depends on the lattice constant and on the
threshold energy in nearly the same way, so fitting both to one pressure is
ill-posed: free fits wander from a0 = 3.45 to 3.61 A. It is broken by
anchoring the AMBIENT lattice constant to the literature value for equiatomic
CrCoNi, a0(0 GPa) = 3.566 A, fitting dE0 per edge there, and then holding dE0
fixed while a0 is fitted at every other pressure - legitimate because all four
pressures were measured on the same beamline with the same edges and the same
normalisation. The bond-length offsets dr are constrained to a
composition-weighted mean of zero for the same reason: their mean IS a0.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

from exafs_gpu.scattering import ELEMENTS, NSP
from . import dataio as io
from . import model as M
from . import shellmodel as SM
from .chik import ChiK, as_kweights, kweight_label, kweight_value
from .system import pair_labels

A0_AMBIENT = 3.566        # equiatomic CrCoNi, literature (the default system)
A0_BOUNDS_REL = (0.953, 1.0236)   # search range for a0 relative to the reference
FIT_RMIN, FIT_RMAX = 1.4, 5.0
FIT_RMAX_NOMS = 4.2       # where a single-scattering model has to stop
KMIN, KMAX, KWEIGHT = 3.0, 13.0, 3
KWEIGHTS_MAG = (2, 3, 4)          # scanned, for |chi(R)| files
KWEIGHTS_CHIK = ((1, 2, 3),)      # fitted together, for chi(k) files
DR_LABELS = pair_labels(ELEMENTS)        # the default system's pair labels


def fit_mask(R, rmin=FIT_RMIN, rmax=FIT_RMAX):
    return (R > rmin) & (R < rmax)


def mean_zero(u):
    """A per-edge vector with the last entry fixed so the sum vanishes: the
    nsp-1 free numbers of a mean-zero per-edge shift."""
    u = np.atleast_1d(np.asarray(u, float))
    return np.append(u, -u.sum())


def _averages(spectra, pressures=None):
    """{P: data} from the dataio structure, or from a dict keyed by pressure.

    `data` is a (nsp, nR) |chi(R)| array or a ChiK, whichever the files hold
    (`dataio.fit_input`)."""
    if all(isinstance(k, (int, np.integer)) for k in spectra):
        return {int(P): (v if isinstance(v, ChiK) else np.asarray(v, float))
                for P, v in spectra.items() if pressures is None or P in pressures}
    P_all = sorted({P for (P, _el) in spectra})
    return {P: io.fit_input(spectra, P) for P in P_all
            if pressures is None or P in pressures}


dr_from_free = SM.dr_from_free


class Stage1:
    """Parameter vector <-> ShellModel parameters for the global search."""

    def __init__(self, sm, comp, ms=True, free_window=True, free_dr=True,
                 free_c3=True, free_shell_sigma=True, dE0_bounds=(-5.0, 35.0),
                 window_bounds=((2.5, 4.0), (11.0, 15.0)), with_sigma=True,
                 free_a0=False, a0_bounds=None,
                 free_c4=False, free_sigma_pair=False, free_ms_edge=False,
                 elements=None):
        self.sm, self.comp = sm, np.asarray(comp, float)
        self.nsp = int(self.comp.size)
        els = (tuple(elements) if elements is not None
               else tuple(sm.elements) if getattr(sm, "elements", None)
               else tuple(f"e{i}" for i in range(self.nsp)))
        self.elements = els
        labels = pair_labels(els)
        if a0_bounds is None:
            a0_bounds = (sm.a0_ref * A0_BOUNDS_REL[0], sm.a0_ref * A0_BOUNDS_REL[1])
        self.names, self.bounds = [], []

        def add(name, lo, hi):
            self.names.append(name); self.bounds.append((float(lo), float(hi)))

        for e in range(self.nsp):
            add(f"dE0_{els[e]}", *dE0_bounds)
        self.with_sigma = bool(with_sigma)
        if with_sigma:
            add("log_sigma", np.log(0.5), np.log(6.0))
        self.nsss = sm.nshell - 1 if free_shell_sigma else 0
        for s in range(1, sm.nshell) if free_shell_sigma else ():
            add(f"log_sigma_shell{s + 1}", np.log(0.4), np.log(2.5))
        self.ndr = len(labels) - 1 if free_dr else 0
        for lab in labels[:self.ndr]:
            add(f"dr_{lab}", -0.06, 0.06)
        self.c3 = bool(free_c3)
        if free_c3:
            add("c3", -6e-4, 6e-4)
        # fourth cumulant: lets the first-shell pair distribution be
        # non-Gaussian without over-driving C3
        self.c4 = bool(free_c4)
        if free_c4:
            add("c4", -3e-5, 3e-5)
        # per-pair-type width in the first shell, log scale, mean fixed
        self.nsp_pair = (len(labels) - 1) if free_sigma_pair else 0
        for lab in labels[:self.nsp_pair]:
            add(f"logs2_{lab}", -0.7, 0.7)
        self.ms = bool(ms)
        if ms:
            add("log_ms_amp", np.log(0.3), np.log(2.5))
            add("log_ms_c", np.log(0.4), np.log(3.0))
        # one MS amplitude per absorber on top of the global one
        self.ms_edge = bool(ms and free_ms_edge)
        for e in range(self.nsp - 1) if self.ms_edge else ():
            add(f"log_ms_{els[e]}", -0.5, 0.5)
        self.window = bool(free_window)
        if free_window:
            add("kmin", *window_bounds[0])
            add("kmax", *window_bounds[1])
        # a0 at the reference point, only when there is nothing to anchor it to
        self.free_a0 = bool(free_a0)
        if free_a0:
            add("a0", *a0_bounds)

    def unpack(self, x):
        x = np.asarray(x, float)
        i = 0
        p = dict(dE0=x[i:i + self.nsp].copy()); i += self.nsp
        if self.with_sigma:
            p["sigma_scale"] = float(np.exp(x[i])); i += 1
        if self.nsss:
            sss = np.ones(self.sm.nshell)
            sss[1:] = np.exp(x[i:i + self.nsss]); i += self.nsss
            p["sigma_shell_scale"] = sss
        if self.ndr:
            p["dr"] = dr_from_free(x[i:i + self.ndr], self.comp); i += self.ndr
        if self.c3:
            p["c3"] = [float(x[i])]; i += 1
        if self.c4:
            p["c4"] = [float(x[i])]; i += 1
        if self.nsp_pair:
            p["sigma_pair"] = SM.pair_log_from_free(x[i:i + self.nsp_pair], self.comp)
            i += self.nsp_pair
        if self.ms:
            p["ms_amp"] = float(np.exp(x[i])); p["ms_c"] = float(np.exp(x[i + 1])); i += 2
        if self.ms_edge:
            u = np.zeros(self.nsp)
            u[:self.nsp - 1] = x[i:i + self.nsp - 1]
            u[-1] = -u[:-1].sum()          # geometric mean 1, so ms_amp keeps it
            p["ms_amp_edge"] = np.exp(u); i += self.nsp - 1
        if self.window:
            p["kmin"] = float(x[i]); p["kmax"] = float(x[i + 1]); i += 2
        if self.free_a0:
            p["a0"] = float(x[i]); i += 1
        return p

    def x0(self, like=None, values=None):
        """A starting point: the defaults, overridden by `values` (a dict of
        name -> value) or by another Stage1's solution (`like` = (stage, x))."""
        defaults = dict(log_sigma=np.log(3.0), log_ms_amp=0.0, log_ms_c=np.log(1.3),
                        kmin=3.0, kmax=13.0, a0=self.sm.a0_ref)
        vals = {}
        if like is not None:
            other, xo = like
            vals.update(dict(zip(other.names, np.asarray(xo, float))))
        if values:
            vals.update(values)
        x = []
        for n in self.names:
            if n in vals:
                x.append(float(vals[n]))
            elif n.startswith("dE0"):
                x.append(12.0)
            else:
                x.append(defaults.get(n, 0.0))
        return np.array(x)


def _record(sm, N0, data, mask, idx, extra=None):
    rf, scales = sm.r_factor(N0, data, mask, idx, return_scales=True)
    p = sm.params()
    radii = sm.radii()
    out = dict(p)
    out.update(nn_distance=float(radii[0]), shells=radii.tolist(),
               sigma2_nn=float(sm.sigma2_shell()[0]),
               sigma2_shell=sm.sigma2_shell().tolist(),
               scales=scales.tolist(), r_factor_ideal=float(rf))
    if extra:
        out.update(extra)
    return out


def _win_str(kmin, kmax, elements=ELEMENTS):
    """A readable window, whether it is one number or one per edge."""
    lo, hi = np.atleast_1d(kmin), np.atleast_1d(kmax)
    if lo.size == 1 and hi.size == 1:
        return f"{float(lo[0]):.2f}-{float(hi[0]):.2f}"
    n = max(lo.size, hi.size)
    return ", ".join(f"{elements[e]} {float(lo[e % lo.size]):.2f}-"
                     f"{float(hi[e % hi.size]):.2f}" for e in range(n))


def calibrate(spectra, R_data, S=None, kmin=KMIN, kmax=KMAX, kweight=KWEIGHT,
              rmin=FIT_RMIN, rmax=FIT_RMAX, a0_ambient=None, ms=True,
              verbose=True, kweights=None, free_window=True, free_dr=True,
              free_c3=True, free_shell_sigma=True, free_dE0_per_pressure=False,
              free_dE0_edge_per_pressure=False,
              joint=True, de_maxiter=120, de_popsize=12, seed=0, pressures=None,
              anchor_a0=True, free_c4=False, free_sigma_pair=False,
              free_ms_edge=False, x0_values=None, system=None,
              joint_a0=True):
    """Global calibration in pair-count space. Returns ({P: dict}, S).

    See the module docstring for what is fitted where. `S` is returned set to
    the anchor-pressure calibration so it can be used for the band report.

    `kweights` lists the candidates the scan chooses between; an entry may be
    one integer or several fitted together, e.g. `(2, 3, (1, 2, 3))`. The
    default is KWEIGHTS_MAG for |chi(R)| inputs and KWEIGHTS_CHIK for chi(k).
    `anchor_a0=False` frees the ambient lattice constant in stage 2 (chi(k)
    data only; see the module docstring). `a0_ambient` defaults to the
    system's reference lattice constant. `system` names the alloy when `S`
    is not given (see chimera/system.py).

    `joint_a0` (the default) puts every pressure's a0 and sigma^2 scale -
    and, when freed, its mean-zero per-edge E0 shift - into the joint stage
    alongside the shared parameters, instead of holding them at their
    stage-2 values. That is what lets the COMMON E0 settle where all the
    pressures together put it: with a0 held, moving E0 is a step off the
    a0/E0 ridge at every pressure at once and the joint stage cannot move
    it, so it stayed wherever the first pressure's stage 1 left it, and the
    per-pressure a0 then each walked their own stretch of the ridge in
    stage 2 - which is a noisy equation of state. Off reproduces the
    2026-09 behaviour.
    """
    AV = _averages(spectra, pressures)
    chik = any(isinstance(v, ChiK) for v in AV.values())
    if chik:
        assert all(isinstance(v, ChiK) for v in AV.values()), \
            "cannot mix chi(k) and |chi(R)| inputs across pressures"
        if free_window and verbose:
            print("  chi(k) input: the k window is a choice, not a parameter - "
                  f"holding it at {_win_str(kmin, kmax)} A^-1", flush=True)
        free_window = False
    elif not anchor_a0:
        raise ValueError("anchor_a0=False needs chi(k) data: from |chi(R)| alone "
                         "a0 and dE0 are degenerate")
    if kweights is None:
        kweights = KWEIGHTS_CHIK if chik else KWEIGHTS_MAG
    kweights = [as_kweights(k) for k in kweights]
    S = S or M.Spectrum(system=system, kmin=kmin, kmax=kmax, kweight=kweight, ms=ms)
    nsp, els = S.nsp, S.elements
    labels = pair_labels(els)
    if a0_ambient is None:
        a0_ambient = float(S.a0_ref)
    a0_lo, a0_hi = S.a0_ref * A0_BOUNDS_REL[0], S.a0_ref * A0_BOUNDS_REL[1]
    idx = M.match_grid(S.R, R_data)
    mask = fit_mask(R_data, rmin, rmax)
    P0 = 0 if 0 in AV else min(AV)
    P_all = sorted(AV)
    sm = SM.ShellModel.from_spectrum(S)
    sm.set_params(kmin=kmin, kmax=kmax, kweight=kweight)
    N0 = sm.random_counts()
    comp = sm.absorber_counts(N0) / sm.natoms
    # Stage 1 holds a0 at `a0_ambient` while it fits dE0, which is what breaks
    # the a0/E0 degeneracy - but only if the reference point IS ambient. With
    # `anchor_a0=False` there is nothing to anchor to (the 2026-09-15 chi(k)
    # export starts at 3 GPa), and pinning a0 to the ambient literature value
    # there biases dE0: on that data it left the shared dE0 ~2.5 eV off and
    # every pressure 0.018 A high in a0, for a 24% worse R-factor. So a0 joins
    # the stage-1 search instead, which the phase makes identifiable.
    flags = dict(ms=ms, free_window=free_window, free_dr=free_dr, free_c3=free_c3,
                 free_shell_sigma=free_shell_sigma, free_a0=not anchor_a0,
                 free_c4=free_c4, free_sigma_pair=free_sigma_pair,
                 free_ms_edge=free_ms_edge, elements=els)
    stage = Stage1(sm, comp, **flags)
    core = Stage1(sm, comp, ms=ms, free_window=free_window, free_dr=False,
                  free_c3=False, free_shell_sigma=False, free_a0=not anchor_a0,
                  elements=els)
    # stage 3 refits only the parameters COMMON to every pressure, and a0 is
    # not one of them - it is held per pressure there and refit by stage 2
    shared = Stage1(sm, comp, with_sigma=False, **{**flags, "free_a0": False})
    data0 = AV[P0]
    de = dict(seed=seed, tol=1e-8, polish=True)

    def score1(x, kw, st):
        p = st.unpack(x)
        p.setdefault("a0", a0_ambient)
        sm.set_params(kweight=kw, **p)
        return sm.r_factor(N0, data0, mask, idx)

    # ---- stage 2: a0 (and the sigma^2 scale) per pressure, everything else
    # held. At the anchor pressure a0 stays at the literature value: it was
    # used to fix dE0, so freeing it again there would be circular.
    def score2(y, data, anchored, fixed):
        kw = dict(sigma_scale=np.exp(y[0]))
        kw["a0"] = a0_ambient if anchored else y[1]
        if free_dE0_per_pressure:
            # y is [log sigma_scale] + ([a0] unless anchored) + [common shift]
            # + ([u, v] for the per-edge shift), so the common shift sits right
            # after a0 - one slot earlier when this pressure is anchored.
            kw["dE0"] = np.asarray(fixed["dE0"]) + y[1 if anchored else 2]
        if free_dE0_edge_per_pressure:
            # A MEAN-ZERO shift across the edges, nsp-1 free numbers. What is
            # left is exactly the freedom a0 cannot absorb - one edge's
            # distance scale moving relative to another's, which is what the
            # measured per-edge compression anomaly looks like.
            #
            # The mean is excluded HERE because it is degenerate with a0, not
            # because it is unmeasurable: `free_dE0_per_pressure` frees it, and
            # on the 2026-09-16 export with per-edge windows the (a0, E0) plane
            # was mapped (scripts/test_common_dE0.py) and the ridge is tilted
            # but NOT flat - 3 of 169 cells lie within 2% of the minimum, and
            # the peak-alignment contour is nearly horizontal, so the first-
            # shell peak position is an E0 constraint rather than an a0 one.
            # Free the two together or neither; one at a time is worse than
            # neither. See testing/MODEL_CHANGES.md item 1.
            kw["dE0"] = (np.asarray(kw.get("dE0", fixed["dE0"]), float)
                         + mean_zero(y[-(nsp - 1):]))
        sm.set_params(**fixed); sm.set_params(**kw)
        return sm.r_factor(N0, data, mask, idx)

    # index of the common dE0 shift inside y: after sigma_scale and, unless the
    # pressure is anchored, a0. Anchored pressures shift it down by one, so it
    # is resolved per pressure inside stage2 rather than assumed here.
    def stage2(p_shared, extra, start=None):
        """a0, sigma^2 (and the per-pressure E0 shifts) per pressure with the
        shared parameters held. `start[P]` seeds the search from a previous
        per-pressure solution (the joint stage's)."""
        fixed = {k: v for k, v in p_shared.items() if k not in ("a0", "sigma_scale")}
        out = {}
        for P in P_all:
            data, anchored = AV[P], (P == P0 and anchor_a0)
            st = (start or {}).get(P, {})
            bounds2 = [(np.log(0.5), np.log(6.0))] + ([] if anchored else [(a0_lo, a0_hi)])
            y0 = ([np.log(st.get("sigma_scale", p_shared["sigma_scale"]))]
                  + ([] if anchored else [st.get("a0", p_shared["a0"])]))
            if free_dE0_per_pressure:
                bounds2.append((-5.0, 5.0)); y0.append(st.get("dE0_shift", 0.0))
            if free_dE0_edge_per_pressure:
                bounds2 += [(-4.0, 4.0)] * (nsp - 1)
                y0 += list(st.get("edge", np.zeros(nsp - 1)))
            r = differential_evolution(score2, bounds2, args=(data, anchored, fixed),
                                       maxiter=60, popsize=10, x0=np.clip(
                                           y0, [b[0] for b in bounds2],
                                           [b[1] for b in bounds2]), **de)
            score2(r.x, data, anchored, fixed)          # leave sm at the optimum
            out[P] = _record(sm, N0, data, mask, idx,
                             extra=dict(kmin=sm.p["kmin"], kmax=sm.p["kmax"],
                                        rmin=rmin, rmax=rmax, pressure=P,
                                        a0_anchored=anchored,
                                        dE0_shift=(
                                            float(r.x[1 if anchored else 2])
                                            if free_dE0_per_pressure else 0.0),
                                        dE0_edge_shift=(
                                            mean_zero(r.x[-(nsp - 1):]).tolist()
                                            if free_dE0_edge_per_pressure
                                            else [0.0] * nsp), **extra))
        return out

    # ---- stage 3: the shared parameters against every pressure at once,
    # together with each pressure's own a0, sigma^2 scale and (when freed)
    # mean-zero per-edge E0 shift. The per-pressure block of the vector is
    # [a0 (unless anchored), log sigma_scale, u_1..u_{nsp-1}] per pressure.
    nshared = len(shared.names)

    def _per_pressure(x, i, P):
        anchored = (P == P0 and anchor_a0)
        q = {}
        if anchored:
            q["a0"] = a0_ambient
        else:
            q["a0"] = float(x[i]); i += 1
        q["sigma_scale"] = float(np.exp(x[i])); i += 1
        u = None
        if free_dE0_edge_per_pressure:
            u = np.asarray(x[i:i + nsp - 1], float); i += nsp - 1
        return q, u, i

    def score3(x, kw):
        p = shared.unpack(x[:nshared])
        base = np.asarray(p["dE0"], float)
        tot, i = 0.0, nshared
        for P in P_all:
            q, u, i = _per_pressure(x, i, P)
            if u is not None:
                q["dE0"] = base + mean_zero(u)
            sm.set_params(kweight=kw, **{**p, **q})
            tot += sm.r_factor(N0, AV[P], mask, idx)
        return tot / len(P_all)

    def score3_held(x, kw, a0_P, sig_P):
        """The 2026-09 joint stage: a0 and sigma^2 held per pressure."""
        p = shared.unpack(x)
        tot = 0.0
        for P in P_all:
            sm.set_params(a0=a0_P[P], sigma_scale=sig_P[P], kweight=kw, **p)
            tot += sm.r_factor(N0, AV[P], mask, idx)
        return tot / len(P_all)

    scan, best = [], None
    for kw in kweights:
        kw = as_kweights(kw)
        t0 = time.time()
        rc = differential_evolution(score1, core.bounds, args=(kw, core),
                                    maxiter=de_maxiter, popsize=de_popsize,
                                    x0=core.x0(), **de)
        if stage.names != core.names:
            # Seed from the core solution, and from `x0_values` on top of it
            # where given. Seeding a LARGER model with a smaller one's answer
            # is what makes a nested comparison monotone: the extended model
            # starts where the simpler one finished, so it cannot end up
            # worse unless the search wanders, and a bigger parameter space
            # is exactly where a fixed budget starts to wander.
            r = differential_evolution(score1, stage.bounds, args=(kw, stage),
                                       maxiter=de_maxiter, popsize=de_popsize,
                                       x0=stage.x0(like=(core, rc.x),
                                                   values=x0_values), **de)
            nfev = int(rc.nfev + r.nfev)
        else:
            r, nfev = rc, int(rc.nfev)
        p1 = stage.unpack(r.x)
        p1.setdefault("a0", a0_ambient)
        p1.update(kweight=kw)
        row = dict(kweight=kweight_value(kw), r_factor_anchor=float(r.fun),
                   r_factor_core=float(rc.fun), n_eval=nfev)
        if verbose:
            print(f"  stage 1 {kweight_label(kw)}: R={r.fun:.5f} (core {rc.fun:.5f})  "
                  f"dE0={np.round(p1['dE0'], 2)} eV  sigma_scale={p1['sigma_scale']:.2f}"
                  + (f"  window {p1['kmin']:.2f}-{p1['kmax']:.2f}" if free_window else "")
                  + f"  [{nfev} evals, {time.time() - t0:.0f}s]", flush=True)
        out = stage2(p1, dict())
        if joint and len(P_all) > 1:
            a0_P = {P: out[P]["a0"] for P in P_all}
            sig_P = {P: out[P]["sigma_scale"] for P in P_all}
            if joint_a0:
                b3 = list(shared.bounds)
                x3 = list(shared.x0(like=(stage, r.x)))
                for P in P_all:
                    if not (P == P0 and anchor_a0):
                        b3.append((max(a0_P[P] - 0.03, a0_lo), min(a0_P[P] + 0.03, a0_hi)))
                        x3.append(a0_P[P])
                    b3.append((np.log(0.5), np.log(6.0))); x3.append(np.log(sig_P[P]))
                    if free_dE0_edge_per_pressure:
                        sh = np.asarray(out[P].get("dE0_edge_shift", [0.0] * nsp))
                        b3 += [(-4.0, 4.0)] * (nsp - 1); x3 += list(sh[:nsp - 1])
                x3 = np.clip(x3, [b[0] for b in b3], [b[1] for b in b3])
                r3 = differential_evolution(score3, b3, args=(kw,),
                                            maxiter=max(de_maxiter // 2, 20),
                                            popsize=de_popsize, x0=x3, **de)
                p1 = shared.unpack(r3.x[:nshared])
                start, i = {}, nshared
                for P in P_all:
                    q, u, i = _per_pressure(r3.x, i, P)
                    start[P] = dict(a0=q["a0"], sigma_scale=q["sigma_scale"],
                                    edge=(u if u is not None else np.zeros(nsp - 1)))
                p1.setdefault("a0", start[P0]["a0"])
                p1.update(kweight=kw, sigma_scale=start[P0]["sigma_scale"])
                if verbose:
                    a0s = ", ".join(f"{P}: {start[P]['a0']:.4f}" for P in P_all)
                    print(f"  stage 3 (joint, a0 and sigma^2 per pressure free): "
                          f"mean R={r3.fun:.5f}  dE0={np.round(p1['dE0'], 2)} eV  "
                          f"a0={{{a0s}}}", flush=True)
                out = stage2(p1, dict(), start=start)
            else:
                r3 = differential_evolution(score3_held, shared.bounds,
                                            args=(kw, a0_P, sig_P),
                                            maxiter=max(de_maxiter // 2, 20),
                                            popsize=de_popsize,
                                            x0=shared.x0(like=(stage, r.x)), **de)
                p1 = shared.unpack(r3.x)
                p1.setdefault("a0", a0_P[P0])
                p1.update(kweight=kw, sigma_scale=sig_P[P0])
                out = stage2(p1, dict())
            row["n_eval"] += int(r3.nfev)
        row["r_factor_joint"] = float(np.mean([out[P]["r_factor_ideal"] for P in P_all]))
        scan.append(row)
        if verbose:
            print(f"  {kweight_label(kw)}: mean R over {len(P_all)} pressure(s) = {row['r_factor_joint']:.5f}  "
                  f"dr={dict(zip(labels, np.round(np.array(p1.get('dr', np.zeros((nsp, nsp))))[np.triu_indices(nsp)], 4)))}"
                  f"  C3={p1.get('c3', [0.0])[0]:.2e}  ms_amp={p1.get('ms_amp', 0):.3f} "
                  f"ms_c={p1.get('ms_c', 0):.2f}  sigma_shell={np.round(p1.get('sigma_shell_scale', np.ones(sm.nshell)), 2)}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
        if best is None or row["r_factor_joint"] < best[0]:
            best = (row["r_factor_joint"], kw, dict(p1), out, r.x)
    _, kw_best, p_best, out, best_x1 = best
    shared_rec = dict(p_best, kweight_scan=scan, joint=bool(joint and len(P_all) > 1),
                      joint_a0=bool(joint_a0),
                      free=dict(window=free_window, dr=free_dr, c3=free_c3,
                                shell_sigma=free_shell_sigma,
                                dE0_edge_per_pressure=bool(free_dE0_edge_per_pressure),
                                dE0_common_per_pressure=bool(free_dE0_per_pressure)),
                      target_complex=bool(chik), a0_anchored=bool(anchor_a0),
                      a0_ambient=float(a0_ambient), system=S.system.to_dict())
    shared_rec["kweight"] = kweight_value(shared_rec["kweight"])
    # the raw stage-1 solution by parameter name, so a later call can start
    # from it (see `x0_values`)
    shared_rec["stage1_x"] = {n: float(v) for n, v in zip(stage.names, best_x1)}
    # the window is only in `names` when it was searched; record it either way
    shared_rec.setdefault("kmin", out[P0]["kmin"])
    shared_rec.setdefault("kmax", out[P0]["kmax"])
    shared_rec["dE0"] = np.asarray(shared_rec["dE0"]).tolist()
    for k in ("sigma_shell_scale", "dr", "c3", "c4", "sigma_pair", "ms_amp_edge"):
        if k in shared_rec:
            shared_rec[k] = np.asarray(shared_rec[k]).tolist()
    for P in P_all:
        out[P]["kweight_scan"] = scan
        out[P]["shared"] = shared_rec
        if verbose:
            print(f"  P={P:2d} GPa: a0={out[P]['a0']:.4f} A  r_NN={out[P]['nn_distance']:.4f} A  "
                  f"sigma2_NN={out[P]['sigma2_nn']:.5f} A^2  R_ideal={out[P]['r_factor_ideal']:.5f}",
                  flush=True)
    if verbose:
        print(f"  chosen {kweight_label(kw_best)}", flush=True)
    # hand the anchor-pressure calibration back through S
    sm.set_params(**{k: v for k, v in out[P0].items() if k in SM.PARAM_NAMES})
    sm.apply_to(S)
    return out, S


# ----------------------------------------------------------------- legacy
def _objective(S, cfg, idx, data, mask):
    tgt = M.target(data, S)

    def f(a0, dE0, log_sig):
        S.set_a0(a0)
        S.set_sigma_scale(np.exp(log_sig))
        S.set_window(dE0=dE0)
        m = M.observe(S, cfg, tgt)[:, idx]
        return M.r_factor(m, tgt, mask, M.fit_scales(m, tgt, mask, S.chan_edge)), m
    return f


def calibrate_legacy(spectra, R_data, S=None, kmin=KMIN, kmax=KMAX, kweight=KWEIGHT,
                     rmin=FIT_RMIN, rmax=FIT_RMAX, a0_ambient=None,
                     seeds=(0, 1, 2), ms=True, verbose=True, system=None):
    """The original Nelder-Mead calibration on the GPU pair sum (no dr, C3 or
    per-shell sigma^2; window and k weight fixed). Kept for comparison.

    Stage 1  a0 := a0_ambient at 0 GPa; fit dE0 (per edge) and the sigma^2
             scale to the 0 GPa location-averaged spectra.
    Stage 2  dE0 fixed; fit a0 and the sigma^2 scale at every pressure.

    The configuration used here is a random solid solution - the calibration
    must not depend on the short-range order it is meant to enable measuring.
    Averaging the objective over a few random configurations removes the
    residual sensitivity to which one was drawn.
    """
    S = S or M.Spectrum(system=system, kmin=kmin, kmax=kmax, kweight=kweight, ms=ms)
    nsp = S.nsp
    if a0_ambient is None:
        a0_ambient = float(S.a0_ref)
    S.set_window(kmin, kmax, kweight)
    idx = M.match_grid(S.R, R_data)
    mask = fit_mask(R_data, rmin, rmax)
    cfgs = [S.random_config(s) for s in seeds]
    # the window is fixed here, so each pressure's target is fixed too
    AV = {P: M.target(v, S) for P, v in _averages(spectra).items()}

    def score(a0, dE0, log_sig, data, log_ms_amp=None, log_ms_c=None):
        S.set_a0(a0); S.set_sigma_scale(np.exp(log_sig))
        if ms and log_ms_amp is not None:
            S.set_ms(np.exp(log_ms_amp), np.exp(log_ms_c))
        S.set_window(dE0=dE0)
        tot = 0.0
        for c in cfgs:
            m = M.observe(S, c, data)[:, idx]
            tot += M.r_factor(m, data, mask, M.fit_scales(m, data, mask, S.chan_edge))
        return tot / len(cfgs)

    # ---- stage 1: dE0 at the anchored ambient lattice constant
    data0 = AV[min(AV)]
    best = None
    for d in (4.0, 10.0, 16.0, 22.0):
        x0 = [d] + [d + 8] * (nsp - 1) + [0.4] + ([0.0, np.log(1.3)] if ms else [])
        r = minimize(lambda p: score(a0_ambient, p[:nsp], p[nsp], data0,
                                     *(p[nsp + 1:nsp + 3] if ms else ())),
                     x0, method="Nelder-Mead",
                     options=dict(maxiter=1500, xatol=1e-3, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    dE0 = np.array(best.x[:nsp], float)
    ms_amp = float(np.exp(best.x[nsp + 1])) if ms else 0.0
    ms_c = float(np.exp(best.x[nsp + 2])) if ms else 0.0
    if verbose:
        print(f"  stage 1 (a0 := {a0_ambient} A): R={best.fun:.4f}  "
              f"dE0={np.round(dE0, 2)} eV  sigma_scale={np.exp(best.x[nsp]):.2f}"
              + (f"  ms_amp={ms_amp:.3f} ms_c={ms_c:.2f}" if ms else ""))

    # ---- stage 2: a0 per pressure, dE0 fixed
    out = {}
    for P in sorted(AV):
        data = AV[P]
        b = None
        for a0 in a0_ambient * np.array([0.9815, 0.9913, 1.0, 1.0067]):
            r = minimize(lambda p: score(p[0], dE0, p[1], data,
                                         *((np.log(ms_amp), np.log(ms_c)) if ms else ())),
                         [a0, 0.4], method="Nelder-Mead",
                         options=dict(maxiter=500, xatol=1e-5, fatol=1e-10))
            if b is None or r.fun < b.fun:
                b = r
        a0, sig = float(b.x[0]), float(np.exp(b.x[1]))
        S.set_a0(a0); S.set_sigma_scale(sig)
        if ms:
            S.set_ms(ms_amp, ms_c)
        S.set_window(dE0=dE0)
        m = M.observe(S, cfgs[0], data)[:, idx]
        scales = M.edge_scales(M.fit_scales(m, data, mask, S.chan_edge), S.chan_edge)
        nn = np.abs(np.linalg.norm((S.cell.ideal[S.j] + S.shift) - S.cell.ideal[S.i],
                                   axis=1) - S.shells[0]) < 0.05
        out[P] = dict(system=S.system.to_dict(),
                      a0=a0, nn_distance=float(S.shells[0]), sigma_scale=sig,
                      multiple_scattering=bool(ms), ms_amp=ms_amp, ms_c=ms_c,
                      sigma2_nn=float(S.sigma2[nn].mean()), dE0=dE0.tolist(),
                      kmin=kmin, kmax=kmax, kweight=kweight,
                      rmin=rmin, rmax=rmax, scales=scales.tolist(),
                      r_factor_ideal=float(b.fun),
                      shells=[float(x) for x in S.shells])
        if verbose:
            print(f"  P={P:2d} GPa: a0={a0:.4f} A  r_NN={S.shells[0]:.4f} A  "
                  f"sigma2_NN={out[P]['sigma2_nn']:.5f} A^2  R_ideal={b.fun:.4f}")
    return out, S


def equation_of_state(cal):
    """V/V0 and bulk moduli from the fitted lattice constants.

    `B0_BM2_GPa` is a 2nd-order Birch-Murnaghan fit, which presumes V0 is the
    AMBIENT volume: the form P(V) = 1.5 B0 (x^7 - x^5) has P = 0 at V = V0. It
    is therefore only reported when the series actually contains 0 GPa. Fitting
    it to a series that starts at 3 GPa - pretending the 3 GPa volume is the
    zero-pressure one - inflated B0 from ~250 to 338 GPa on the 2026-09-15
    chi(k) run, which is an artefact of the missing anchor, not stiffness.

    `B_secant_GPa` needs no such assumption: between any two points it is
    dP / (dV/V), reported for every consecutive pair and, when ambient is
    present, from ambient to each point.
    """
    Ps = sorted(cal)
    P = np.array(Ps, float)
    a = np.array([cal[p]["a0"] for p in Ps])
    V = a ** 3
    out = dict(pressure=P.tolist(), a0=a.tolist(), V_over_V0=(V / V[0]).tolist(),
               V0_reference_GPa=float(P[0]))
    if P[0] == 0:
        hi = P > 0
        out["B_secant_GPa"] = [float(p / max(1 - v, 1e-9))
                               for p, v in zip(P[hi], (V / V[0])[hi])]
    else:
        out["B_secant_GPa"] = None
    out["B_secant_pairs_GPa"] = [
        dict(P_lo=float(P[i]), P_hi=float(P[i + 1]),
             B=float((P[i + 1] - P[i]) / max(1 - V[i + 1] / V[i], 1e-9)))
        for i in range(len(P) - 1)]
    out["B0_BM2_GPa"] = None
    if len(Ps) >= 2 and P[0] == 0:
        # Birch-Murnaghan 2nd order fit (B0' = 4)
        def bm(B0):
            x = (V[0] / V) ** (1 / 3)
            return np.sum((1.5 * B0 * (x ** 7 - x ** 5) - P) ** 2)
        from scipy.optimize import minimize_scalar
        out["B0_BM2_GPa"] = float(minimize_scalar(bm, bounds=(50, 500),
                                                  method="bounded").x)
    return out


def band_report(S, cfg, idx, data, R_data, scales=None):
    """R-factor and data/noise band by band - the evidence for the fit range.

    One entry per channel in each list; `data` may be a |chi(R)| array or a
    ChiK (then the bands are scored on the complex transform)."""
    tgt = M.target(data, S)
    m = M.observe(S, cfg, tgt)[:, idx]
    if scales is None:
        scales = M.fit_scales(m, tgt, fit_mask(R_data), S.chan_edge)
    elif np.size(scales) == S.nsp and S.nchan != S.nsp:
        scales = np.asarray(scales)[S.chan_edge]
    m = m * np.asarray(scales)[:, None]
    if M.is_chik(data):
        noise = data.noise(S.window)
    else:
        noise = np.array([io.noise_estimate(R_data, tgt[i]) for i in range(tgt.shape[0])])
    bands = ((0.6, 1.2), (1.2, 1.4), (1.4, 1.8), (1.8, 2.6), (2.6, 3.3),
             (3.3, 4.1), (4.1, 4.2), (4.2, 4.8), (4.8, 5.0), (5.0, 5.6),
             (5.6, 8.0))
    rows = []
    for lo, hi in bands:
        b = (R_data >= lo) & (R_data < hi)
        rf = [float((np.abs(m[i][b] - tgt[i][b]) ** 2).sum()
                    / max((np.abs(tgt[i][b]) ** 2).sum(), 1e-30))
              for i in range(tgt.shape[0])]
        snr = [float(np.abs(tgt[i][b]).mean() / noise[i]) for i in range(tgt.shape[0])]
        rows.append(dict(r_lo=lo, r_hi=hi, r_factor=rf, data_over_noise=snr,
                         in_fit=bool(lo >= FIT_RMIN and hi <= FIT_RMAX)))
    return rows
