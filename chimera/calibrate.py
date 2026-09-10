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

A0_AMBIENT = 3.566        # equiatomic CrCoNi, literature
FIT_RMIN, FIT_RMAX = 1.4, 5.0
FIT_RMAX_NOMS = 4.2       # where a single-scattering model has to stop
KMIN, KMAX, KWEIGHT = 3.0, 13.0, 3
DR_LABELS = [f"{ELEMENTS[a]}{ELEMENTS[b]}" for a, b in zip(*np.triu_indices(NSP))]


def fit_mask(R, rmin=FIT_RMIN, rmax=FIT_RMAX):
    return (R > rmin) & (R < rmax)


def _averages(spectra, pressures=None):
    """{P: (nsp, nR)} from the dataio structure, or from a dict of arrays."""
    if all(isinstance(k, (int, np.integer)) for k in spectra):
        return {int(P): np.asarray(v, float) for P, v in spectra.items()
                if pressures is None or P in pressures}
    P_all = sorted({P for (P, _el) in spectra})
    return {P: io.average(spectra, P) for P in P_all
            if pressures is None or P in pressures}


dr_from_free = SM.dr_from_free


class Stage1:
    """Parameter vector <-> ShellModel parameters for the global search."""

    def __init__(self, sm, comp, ms=True, free_window=True, free_dr=True,
                 free_c3=True, free_shell_sigma=True, dE0_bounds=(-5.0, 35.0),
                 window_bounds=((2.5, 4.0), (11.0, 15.0)), with_sigma=True):
        self.sm, self.comp = sm, np.asarray(comp, float)
        self.names, self.bounds = [], []

        def add(name, lo, hi):
            self.names.append(name); self.bounds.append((float(lo), float(hi)))

        for e in range(NSP):
            add(f"dE0_{ELEMENTS[e]}", *dE0_bounds)
        self.with_sigma = bool(with_sigma)
        if with_sigma:
            add("log_sigma", np.log(0.5), np.log(6.0))
        self.nsss = sm.nshell - 1 if free_shell_sigma else 0
        for s in range(1, sm.nshell) if free_shell_sigma else ():
            add(f"log_sigma_shell{s + 1}", np.log(0.4), np.log(2.5))
        self.ndr = len(DR_LABELS) - 1 if free_dr else 0
        for lab in DR_LABELS[:self.ndr]:
            add(f"dr_{lab}", -0.06, 0.06)
        self.c3 = bool(free_c3)
        if free_c3:
            add("c3", -6e-4, 6e-4)
        self.ms = bool(ms)
        if ms:
            add("log_ms_amp", np.log(0.3), np.log(2.5))
            add("log_ms_c", np.log(0.4), np.log(3.0))
        self.window = bool(free_window)
        if free_window:
            add("kmin", *window_bounds[0])
            add("kmax", *window_bounds[1])

    def unpack(self, x):
        x = np.asarray(x, float)
        i = 0
        p = dict(dE0=x[i:i + NSP].copy()); i += NSP
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
        if self.ms:
            p["ms_amp"] = float(np.exp(x[i])); p["ms_c"] = float(np.exp(x[i + 1])); i += 2
        if self.window:
            p["kmin"] = float(x[i]); p["kmax"] = float(x[i + 1]); i += 2
        return p

    def x0(self, like=None, values=None):
        """A starting point: the defaults, overridden by `values` (a dict of
        name -> value) or by another Stage1's solution (`like` = (stage, x))."""
        defaults = dict(log_sigma=np.log(3.0), log_ms_amp=0.0, log_ms_c=np.log(1.3),
                        kmin=3.0, kmax=13.0)
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


def calibrate(spectra, R_data, S=None, kmin=KMIN, kmax=KMAX, kweight=KWEIGHT,
              rmin=FIT_RMIN, rmax=FIT_RMAX, a0_ambient=A0_AMBIENT, ms=True,
              verbose=True, kweights=(2, 3, 4), free_window=True, free_dr=True,
              free_c3=True, free_shell_sigma=True, free_dE0_per_pressure=False,
              joint=True, de_maxiter=120, de_popsize=12, seed=0, pressures=None):
    """Global calibration in pair-count space. Returns ({P: dict}, S).

    See the module docstring for what is fitted where. `S` is returned set to
    the anchor-pressure calibration so it can be used for the band report.
    """
    S = S or M.Spectrum(kmin=kmin, kmax=kmax, kweight=kweight, ms=ms)
    idx = M.match_grid(S.R, R_data)
    mask = fit_mask(R_data, rmin, rmax)
    AV = _averages(spectra, pressures)
    P0 = 0 if 0 in AV else min(AV)
    P_all = sorted(AV)
    sm = SM.ShellModel.from_spectrum(S)
    sm.set_params(kmin=kmin, kmax=kmax, kweight=kweight)
    N0 = sm.random_counts()
    comp = sm.absorber_counts(N0) / sm.natoms
    flags = dict(ms=ms, free_window=free_window, free_dr=free_dr, free_c3=free_c3,
                 free_shell_sigma=free_shell_sigma)
    stage = Stage1(sm, comp, **flags)
    core = Stage1(sm, comp, ms=ms, free_window=free_window, free_dr=False,
                  free_c3=False, free_shell_sigma=False)
    shared = Stage1(sm, comp, with_sigma=False, **flags)
    data0 = AV[P0]
    de = dict(seed=seed, tol=1e-8, polish=True)

    # ---- stage 1: everything but a0, at the anchored ambient lattice constant
    def score1(x, kw, st):
        sm.set_params(a0=a0_ambient, kweight=int(kw), **st.unpack(x))
        return sm.r_factor(N0, data0, mask, idx)

    # ---- stage 2: a0 (and the sigma^2 scale) per pressure, everything else
    # held. At the anchor pressure a0 stays at the literature value: it was
    # used to fix dE0, so freeing it again there would be circular.
    def score2(y, data, anchored, fixed):
        kw = dict(sigma_scale=np.exp(y[0]))
        kw["a0"] = a0_ambient if anchored else y[1]
        if free_dE0_per_pressure:
            kw["dE0"] = np.asarray(fixed["dE0"]) + y[-1]
        sm.set_params(**fixed); sm.set_params(**kw)
        return sm.r_factor(N0, data, mask, idx)

    def stage2(p_shared, extra):
        fixed = {k: v for k, v in p_shared.items() if k not in ("a0", "sigma_scale")}
        out = {}
        for P in P_all:
            data, anchored = AV[P], (P == P0)
            bounds2 = [(np.log(0.5), np.log(6.0))] + ([] if anchored else [(3.40, 3.65)])
            y0 = [np.log(p_shared["sigma_scale"])] + ([] if anchored else [p_shared["a0"]])
            if free_dE0_per_pressure:
                bounds2.append((-5.0, 5.0)); y0.append(0.0)
            r = differential_evolution(score2, bounds2, args=(data, anchored, fixed),
                                       maxiter=60, popsize=10, x0=y0, **de)
            score2(r.x, data, anchored, fixed)          # leave sm at the optimum
            out[P] = _record(sm, N0, data, mask, idx,
                             extra=dict(kmin=sm.p["kmin"], kmax=sm.p["kmax"],
                                        rmin=rmin, rmax=rmax, pressure=P,
                                        a0_anchored=anchored,
                                        dE0_shift=(float(r.x[-1]) if free_dE0_per_pressure
                                                   else 0.0), **extra))
        return out

    # ---- stage 3: the shared parameters against every pressure at once
    def score3(x, kw, a0_P, sig_P):
        p = shared.unpack(x)
        tot = 0.0
        for P in P_all:
            sm.set_params(a0=a0_P[P], sigma_scale=sig_P[P], kweight=int(kw), **p)
            tot += sm.r_factor(N0, AV[P], mask, idx)
        return tot / len(P_all)

    scan, best = [], None
    for kw in kweights:
        kw = int(kw)
        t0 = time.time()
        rc = differential_evolution(score1, core.bounds, args=(kw, core),
                                    maxiter=de_maxiter, popsize=de_popsize,
                                    x0=core.x0(), **de)
        if stage.names != core.names:
            r = differential_evolution(score1, stage.bounds, args=(kw, stage),
                                       maxiter=de_maxiter, popsize=de_popsize,
                                       x0=stage.x0(like=(core, rc.x)), **de)
            nfev = int(rc.nfev + r.nfev)
        else:
            r, nfev = rc, int(rc.nfev)
        p1 = stage.unpack(r.x)
        p1.update(a0=a0_ambient, kweight=kw)
        row = dict(kweight=kw, r_factor_anchor=float(r.fun), r_factor_core=float(rc.fun),
                   n_eval=nfev)
        if verbose:
            print(f"  stage 1 k^{kw}: R={r.fun:.5f} (core {rc.fun:.5f})  "
                  f"dE0={np.round(p1['dE0'], 2)} eV  sigma_scale={p1['sigma_scale']:.2f}"
                  + (f"  window {p1['kmin']:.2f}-{p1['kmax']:.2f}" if free_window else "")
                  + f"  [{nfev} evals, {time.time() - t0:.0f}s]", flush=True)
        out = stage2(p1, dict())
        if joint and len(P_all) > 1:
            a0_P = {P: out[P]["a0"] for P in P_all}
            sig_P = {P: out[P]["sigma_scale"] for P in P_all}
            r3 = differential_evolution(score3, shared.bounds, args=(kw, a0_P, sig_P),
                                        maxiter=max(de_maxiter // 2, 20), popsize=de_popsize,
                                        x0=shared.x0(like=(stage, r.x)), **de)
            p1 = shared.unpack(r3.x)
            p1.update(a0=a0_ambient, kweight=kw, sigma_scale=sig_P[P0])
            out = stage2(p1, dict())
            row["n_eval"] += int(r3.nfev)
        row["r_factor_joint"] = float(np.mean([out[P]["r_factor_ideal"] for P in P_all]))
        scan.append(row)
        if verbose:
            print(f"  k^{kw}: mean R over {len(P_all)} pressure(s) = {row['r_factor_joint']:.5f}  "
                  f"dr={dict(zip(DR_LABELS, np.round(np.array(p1.get('dr', np.zeros((NSP, NSP))))[np.triu_indices(NSP)], 4)))}"
                  f"  C3={p1.get('c3', [0.0])[0]:.2e}  ms_amp={p1.get('ms_amp', 0):.3f} "
                  f"ms_c={p1.get('ms_c', 0):.2f}  sigma_shell={np.round(p1.get('sigma_shell_scale', np.ones(sm.nshell)), 2)}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
        if best is None or row["r_factor_joint"] < best[0]:
            best = (row["r_factor_joint"], kw, dict(p1), out)
    _, kw_best, p_best, out = best
    shared_rec = dict(p_best, kweight_scan=scan, joint=bool(joint and len(P_all) > 1),
                      free=dict(window=free_window, dr=free_dr, c3=free_c3,
                                shell_sigma=free_shell_sigma))
    shared_rec["dE0"] = np.asarray(shared_rec["dE0"]).tolist()
    for k in ("sigma_shell_scale", "dr", "c3"):
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
        print(f"  chosen k^{kw_best}", flush=True)
    # hand the anchor-pressure calibration back through S
    sm.set_params(**{k: v for k, v in out[P0].items() if k in SM.PARAM_NAMES})
    sm.apply_to(S)
    return out, S


# ----------------------------------------------------------------- legacy
def _objective(S, cfg, idx, data, mask):
    def f(a0, dE0, log_sig):
        S.set_a0(a0)
        S.set_sigma_scale(np.exp(log_sig))
        S.set_window(dE0=dE0)
        m = S.chi_R(cfg)[:, idx]
        return M.r_factor(m, data, mask, M.fit_scales(m, data, mask)), m
    return f


def calibrate_legacy(spectra, R_data, S=None, kmin=KMIN, kmax=KMAX, kweight=KWEIGHT,
                     rmin=FIT_RMIN, rmax=FIT_RMAX, a0_ambient=A0_AMBIENT,
                     seeds=(0, 1, 2), ms=True, verbose=True):
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
    S = S or M.Spectrum(kmin=kmin, kmax=kmax, kweight=kweight, ms=ms)
    S.set_window(kmin, kmax, kweight)
    idx = M.match_grid(S.R, R_data)
    mask = fit_mask(R_data, rmin, rmax)
    cfgs = [S.random_config(s) for s in seeds]
    AV = _averages(spectra)

    def score(a0, dE0, log_sig, data, log_ms_amp=None, log_ms_c=None):
        S.set_a0(a0); S.set_sigma_scale(np.exp(log_sig))
        if ms and log_ms_amp is not None:
            S.set_ms(np.exp(log_ms_amp), np.exp(log_ms_c))
        S.set_window(dE0=dE0)
        tot = 0.0
        for c in cfgs:
            m = S.chi_R(c)[:, idx]
            tot += M.r_factor(m, data, mask, M.fit_scales(m, data, mask))
        return tot / len(cfgs)

    # ---- stage 1: dE0 at the anchored ambient lattice constant
    data0 = AV[0]
    best = None
    for d in (4.0, 10.0, 16.0, 22.0):
        x0 = [d, d + 8, d + 8, 0.4] + ([0.0, np.log(1.3)] if ms else [])
        r = minimize(lambda p: score(a0_ambient, p[:3], p[3], data0,
                                     *(p[4:6] if ms else ())),
                     x0, method="Nelder-Mead",
                     options=dict(maxiter=1500, xatol=1e-3, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    dE0 = np.array(best.x[:3], float)
    ms_amp = float(np.exp(best.x[4])) if ms else 0.0
    ms_c = float(np.exp(best.x[5])) if ms else 0.0
    if verbose:
        print(f"  stage 1 (a0 := {a0_ambient} A): R={best.fun:.4f}  "
              f"dE0={np.round(dE0, 2)} eV  sigma_scale={np.exp(best.x[3]):.2f}"
              + (f"  ms_amp={ms_amp:.3f} ms_c={ms_c:.2f}" if ms else ""))

    # ---- stage 2: a0 per pressure, dE0 fixed
    out = {}
    for P in sorted(AV):
        data = AV[P]
        b = None
        for a0 in (3.50, 3.535, 3.566, 3.59):
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
        m = S.chi_R(cfgs[0])[:, idx]
        scales = M.fit_scales(m, data, mask)
        nn = np.abs(np.linalg.norm((S.cell.ideal[S.j] + S.shift) - S.cell.ideal[S.i],
                                   axis=1) - S.shells[0]) < 0.05
        out[P] = dict(a0=a0, nn_distance=float(S.shells[0]), sigma_scale=sig,
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
    """V/V0 and a secant bulk modulus from the fitted lattice constants."""
    Ps = sorted(cal)
    P = np.array(Ps, float)
    a = np.array([cal[p]["a0"] for p in Ps])
    V = a ** 3
    out = dict(pressure=P.tolist(), a0=a.tolist(), V_over_V0=(V / V[0]).tolist())
    hi = P > 0
    out["B_secant_GPa"] = [float(p / max(1 - v, 1e-9))
                           for p, v in zip(P[hi], (V / V[0])[hi])]
    if len(Ps) < 2:
        return out
    # Birch-Murnaghan 2nd order fit (B0' = 4)
    def bm(B0):
        x = (V[0] / V) ** (1 / 3)
        return np.sum((1.5 * B0 * (x ** 7 - x ** 5) * (1 + 0.75 * (4 - 4) * (x ** 2 - 1)) - P) ** 2)
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(bm, bounds=(50, 500), method="bounded")
    out["B0_BM2_GPa"] = float(r.x)
    return out


def band_report(S, cfg, idx, data, R_data, scales=None):
    """R-factor and data/noise band by band - the evidence for the fit range."""
    m = S.chi_R(cfg)[:, idx]
    if scales is None:
        scales = M.fit_scales(m, data, fit_mask(R_data))
    m = m * np.asarray(scales)[:, None]
    noise = np.array([io.noise_estimate(R_data, data[i]) for i in range(data.shape[0])])
    bands = ((0.6, 1.2), (1.2, 1.4), (1.4, 1.8), (1.8, 2.6), (2.6, 3.3),
             (3.3, 4.1), (4.1, 4.2), (4.2, 4.8), (4.8, 5.0), (5.0, 5.6),
             (5.6, 8.0))
    rows = []
    for lo, hi in bands:
        b = (R_data >= lo) & (R_data < hi)
        rf = [float(((m[i][b] - data[i][b]) ** 2).sum()
                    / max((data[i][b] ** 2).sum(), 1e-30)) for i in range(data.shape[0])]
        snr = [float(data[i][b].mean() / noise[i]) for i in range(data.shape[0])]
        rows.append(dict(r_lo=lo, r_hi=hi, r_factor=rf, data_over_noise=snr,
                         in_fit=bool(lo >= FIT_RMIN and hi <= FIT_RMAX)))
    return rows
