"""Fit one spectrum: refine the local lattice, sample configurations, write structure.

Per location the well-determined quantities and the poorly-determined ones are
kept strictly apart, because they differ by an order of magnitude in how much
the data actually says about them:

  determined      a0 (hence the nearest-neighbour distance), sigma^2, the
                  per-edge amplitude, and the fit quality. The first-shell
                  peak alone fixes these; the ideal-model R-factor over
                  1.8-2.6 A is 0.002.
  NOT determined  the Warren-Cowley parameters. Perturbing the calibration
                  within its own uncertainty (sigma^2 by 15%, k weight 2 vs 3,
                  the fit range by 0.2 A) moves a_CrCr over the whole range
                  -0.19 to +0.38 while the R-factor moves by under 8%. The
                  fitted values are therefore reported together with that
                  systematic band, which is what `systematic_band` measures.

Every fit also reports the Cramer-Rao bound on each alpha (`wc_crlb`): the
1-sigma uncertainty the data permit at the residual level the sampler sees,
computed from the pair-count model with all shells and the nuisance
parameters free (shellmodel.crlb). It is independent of the sampler and of
how many steps were run; a bound above `CRLB_THRESHOLD` marks the value as
not determined, however small the seed-to-seed scatter is.

Moves are species swaps only (`swap_frac=1.0`): the atoms stay on ideal FCC
sites and all displacement disorder is carried by the Debye-Waller factor.
That is what "keep the sample FCC" means operationally, and it also avoids
fitting 1500 coordinates to ~54 independent data points.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from exafs_gpu.lattice import warren_cowley
from exafs_gpu.scattering import ELEMENTS, NSP
from methods.method_a_rmc import RMCSampler
from . import dataio as io
from . import model as M
from . import shellmodel as SM
from .multift import MultiEdgeFT

CRLB_THRESHOLD = 0.3

WC_LABELS = [f"a_{ELEMENTS[i]}{ELEMENTS[j]}" for i, j in zip(*np.triu_indices(NSP))]
WC_LABELS_TEX = [rf"$\alpha_{{\rm {ELEMENTS[i]}{ELEMENTS[j]}}}$"
                 for i, j in zip(*np.triu_indices(NSP))]


def refine_local(S, cfg, idx, data, mask_d, a0_guess, sig_guess,
                 free_a0=True, free_sigma=True):
    """Refine a0 and/or the sigma^2 scale for one spectrum.

    Runs on the pair-count model with the exact random-alloy expectation of
    the pair counts (`cfg` only supplies the composition), so each evaluation
    costs milliseconds and the result has no dependence on which random
    configuration was drawn. The refined values are pushed back into S.

    A parameter being perturbed on purpose (the systematic scan) must be held
    fixed here, or the refinement simply undoes the perturbation.
    """
    sm = SM.ShellModel.from_spectrum(S)
    N0 = sm.random_counts(counts=np.bincount(np.asarray(cfg, int), minlength=NSP))
    sm.set_params(a0=a0_guess, sigma_scale=sig_guess)

    def obj(p):
        kw = {}
        if free_a0:
            kw["a0"] = float(p[0])
        if free_sigma:
            kw["sigma_scale"] = float(np.exp(p[-1]))
        sm.set_params(**kw)
        return sm.r_factor(N0, data, mask_d, idx)
    x0 = ([a0_guess] if free_a0 else []) + ([np.log(sig_guess)] if free_sigma else [])
    if not x0:
        S.set_a0(a0_guess); S.set_sigma_scale(sig_guess)
        return a0_guess, sig_guess, obj([])
    r = minimize(obj, x0, method="Nelder-Mead",
                 options=dict(maxiter=200, xatol=1e-5, fatol=1e-9))
    a0 = float(r.x[0]) if free_a0 else a0_guess
    ss = float(np.exp(r.x[-1])) if free_sigma else sig_guess
    S.set_a0(a0); S.set_sigma_scale(ss)
    return a0, ss, float(r.fun)


def fit_one(S, R_data, data, idx, mask_d, a0_guess, sig_guess, seeds=(0, 1),
            nsteps=6000, nrep=48, refine=True, collect=True,
            free_a0=True, free_sigma=True, swap_frac=1.0, max_disp=0.10,
            disp_sigma=0.012, eval_mask=None, crlb=True, profile_scale=True,
            collect_beta=0.5, noise=None):
    """Refine the lattice, then sample configurations. Returns a result dict.

    `crlb`          also compute the Cramer-Rao bounds on the shell-1 alphas
    `profile_scale` profile the per-edge amplitude inside the sampler's chi^2
    `collect_beta`  rung (inverse temperature) posterior samples come from;
                    0.5 is the likelihood temperature, None the coldest rungs
    `noise`         per-edge measurement noise for the noise-only bound
                    (default: estimated from the R > 7.5 A tail of the data)
    """
    cfg0 = S.random_config(0)
    if refine and (free_a0 or free_sigma):
        a0, ss, r_ideal = refine_local(S, cfg0, idx, data, mask_d, a0_guess,
                                       sig_guess, free_a0, free_sigma)
    else:
        S.set_a0(a0_guess); S.set_sigma_scale(sig_guess)
        a0, ss = a0_guess, sig_guess
        m = S.chi_R(cfg0)[:, idx]
        r_ideal = M.r_factor(m, data, mask_d, M.fit_scales(m, data, mask_d))

    m0 = S.chi_R(cfg0)[:, idx]
    scales = M.fit_scales(m0, data, mask_d)
    # sigma for the sampler: the residual the IDEAL model leaves, i.e. the
    # model error, since that dominates the measurement noise by ~5x here.
    sig_data = np.sqrt((((m0 * scales[:, None]) - data)[:, mask_d] ** 2).mean(1))
    sig = sig_data / scales
    target = np.zeros((NSP, S.R.size)); target[:, idx] = data / scales[:, None]
    mask = np.zeros(S.R.size, bool); mask[idx] = mask_d
    sigR = np.repeat(sig[:, None], S.R.size, axis=1)
    ft = MultiEdgeFT.from_spectrum(S)

    W, chi2s, best = [], [], None
    post, xrate = [], []
    for s in seeds:
        smp = RMCSampler(S.fm, ft, target, mask, sigR, nrep=nrep, seed=int(s),
                         swap_frac=swap_frac, max_disp=max_disp,
                         disp_sigma=disp_sigma,
                         collect_after=int(nsteps * 0.6) if collect else None,
                         collect_every=50, profile_scale=profile_scale,
                         collect_beta=collect_beta)
        smp.init_state(S.random_config(int(s)), S.cell.ideal)
        chi2_0 = float(smp.chi2.min())
        smp.run(nsteps, log_every=10 ** 9)
        xrate.append(float(np.nanmean(smp.exchange_rate)))
        spb, posb, c2 = smp.best()
        W.append(warren_cowley(spb.astype(np.int32), S.i, S.j, S.shift, S.cell,
                               float(S.shells[0]))[np.triu_indices(NSP)])
        chi2s.append(float(c2))
        if collect:
            for sample, _c2 in smp.samples:      # samples are (species, chi2)
                post.append(warren_cowley(np.asarray(sample, np.int32), S.i, S.j,
                                          S.shift, S.cell,
                                          float(S.shells[0]))[np.triu_indices(NSP)])
        if best is None or c2 < best[1]:
            best = (spb.copy(), c2, posb.copy())
    W = np.array(W)
    spb, posb = best[0], best[2]
    rms_disp = float(np.sqrt(((posb - S.cell.ideal) ** 2).sum(1).mean()))
    model_mag = S.chi_R(spb, pos=posb)[:, idx]
    sc_fit = M.fit_scales(model_mag, data, mask_d)
    # R-factor on a COMMON range, so variants fitted over different windows
    # can be compared: a narrower fit range always looks better on its own.
    if eval_mask is None:
        r_common = None
    else:
        sc_c = M.fit_scales(model_mag, data, eval_mask)
        r_common = float(M.r_factor(model_mag, data, eval_mask, sc_c))
    res = dict(
        a0=a0, nn_distance=float(S.shells[0]), sigma_scale=ss,
        sigma_shell_scale=S.sigma_shell_scale.tolist(), dr=S.dr.tolist(),
        c3=S.c3.tolist(),
        rms_displacement=rms_disp, swap_frac=float(swap_frac),
        multiple_scattering=bool(getattr(S, "chi_ms", None) is not None),
        ms_amp=getattr(S, "ms_amp", None), ms_c=getattr(S, "ms_c", None),
        sigma2_nn=float(_nn_sigma2(S)), shells=[float(x) for x in S.shells],
        scales=sc_fit.tolist(), chi2_start=chi2_0,
        chi2=float(np.mean(chi2s)), chi2_seeds=chi2s,
        r_factor_ideal=float(r_ideal),
        r_factor=float(M.r_factor(model_mag, data, mask_d, sc_fit)),
        r_factor_common=r_common,
        wc=W.mean(0).tolist(), wc_sd_seeds=W.std(0).tolist(),
        wc_posterior_sd=(np.std(post, axis=0).tolist() if post else None),
        wc_posterior_mean=(np.mean(post, axis=0).tolist() if post else None),
        posterior_beta=collect_beta, exchange_rate=xrate,
        sigma_model=sig_data.tolist(),
        species=spb.astype(np.int8).tolist(),
        positions=(posb - S.cell.ideal).astype(np.float32).round(4).tolist()
        if swap_frac < 1.0 else None,
        model_mag=(model_mag * sc_fit[:, None]).tolist(),
        ideal_mag=((m0 * scales[:, None]).tolist()),
    )
    if crlb:
        res.update(crlb_bounds(S, spb, R_data, data, idx, mask_d, sig_data, noise))
    return res


def crlb_bounds(S, species, R_data, data, idx, mask_d, sigma_model, noise=None,
                threshold=CRLB_THRESHOLD):
    """Cramer-Rao bounds on the shell-1 alphas at the fitted configuration.

    Three views: all shells and the nuisance parameters free at the residual
    level the fit actually has (the honest one, `wc_crlb`); the first shell
    alone (`wc_crlb_shell1_only`, the best case for this data); and all
    shells free at the measurement-noise level (`wc_crlb_noise`, what a
    perfect model would allow).
    """
    sm = SM.ShellModel.from_spectrum(S)
    N = sm.counts_of(species)
    if noise is None:
        noise = np.array([io.noise_estimate(R_data, data[e]) for e in range(NSP)])
    b_all = SM.crlb(sm, N, data, sigma_model, mask_d, idx, threshold=threshold)
    b_s1 = SM.crlb(sm, N, data, sigma_model, mask_d, idx, shells=[0], threshold=threshold)
    b_noise = SM.crlb(sm, N, data, noise, mask_d, idx, threshold=threshold)
    return dict(wc_crlb=b_all["alpha_sd"], wc_crlb_shell1_only=b_s1["alpha_sd"],
                wc_crlb_noise=b_noise["alpha_sd"], wc_determined=b_all["determined"],
                crlb_threshold=threshold, sigma_noise=np.asarray(noise).tolist(),
                crlb_detail=dict(nuisance=b_all["nuisance"], shells=b_all["shells"],
                                 prior_alpha_sd=b_all["prior_alpha_sd"],
                                 eig_sd=b_all["eig_sd"]))


def _nn_sigma2(S):
    d = np.linalg.norm((S.cell.ideal[S.j] + S.shift) - S.cell.ideal[S.i], axis=1)
    return S.sigma2[np.abs(d - S.shells[0]) < 0.05].mean()


# ------------------------------------------------------------------ systematics
CALIB_VARIANTS = [
    ("baseline", {}),
    ("a0 -0.005 A", dict(da0=-0.005)),
    ("a0 +0.005 A", dict(da0=+0.005)),
    ("dE0 -2 eV", dict(ddE0=-2.0)),
    ("dE0 +2 eV", dict(ddE0=+2.0)),
    ("sigma x0.85", dict(fsig=0.85)),
    ("sigma x1.15", dict(fsig=1.15)),
    ("k weight 2", dict(kweight=2)),
    ("k weight 4", dict(kweight=4)),
    ("k 3.5-13", dict(kmin=3.5)),
    ("k 3.0-12", dict(kmax=12.0)),
    ("R 1.5-5.0", dict(rmin=1.5)),
    ("R 1.4-4.6", dict(rmax=4.6)),
    ("R 1.4-5.2", dict(rmax=5.2)),
    ("ms_amp x0.85", dict(fms=0.85)),
    ("ms_amp x1.15", dict(fms=1.15)),
    ("no dr", dict(dr_off=True)),
    ("no C3", dict(c3_off=True)),
]


def systematic_band(R_data, data, cal, seeds=(0,), nsteps=6000, nrep=48,
                    variants=CALIB_VARIANTS, verbose=False, swap_frac=1.0):
    """Refit under each calibration variant; the spread IS the systematic error.

    Every variant is a choice the data cannot distinguish - they all fit to
    within a few percent in R-factor - so the range of Warren-Cowley values
    they produce is the honest uncertainty on those parameters.
    """
    rows = []
    common = (R_data > cal["rmin"]) & (R_data < cal["rmax"])
    for name, ch in variants:
        a0 = cal["a0"] + ch.get("da0", 0.0)
        dE0 = [d + ch.get("ddE0", 0.0) for d in cal["dE0"]]
        ss = cal["sigma_scale"] * ch.get("fsig", 1.0)
        ms_amp = cal.get("ms_amp", 1.0) * ch.get("fms", 1.0)
        kw = ch.get("kweight", cal["kweight"])
        kmin = ch.get("kmin", cal["kmin"]); kmax = ch.get("kmax", cal["kmax"])
        rmin = ch.get("rmin", cal["rmin"]); rmax = ch.get("rmax", cal["rmax"])
        ov = dict(a0=a0, dE0=dE0, kmin=kmin, kmax=kmax, kweight=kw,
                  sigma_scale=ss, ms_amp=ms_amp)
        if ch.get("dr_off"):
            ov["dr"] = None
        if ch.get("c3_off"):
            ov["c3"] = None
        S = M.Spectrum.from_cal(cal, **ov)
        idx = M.match_grid(S.R, R_data)
        md = (R_data > rmin) & (R_data < rmax)
        # do not refine a parameter this variant is deliberately perturbing
        res = fit_one(S, R_data, data, idx, md, a0, ss, seeds=seeds,
                      nsteps=nsteps, nrep=nrep, collect=False,
                      refine=True, free_a0=("da0" not in ch),
                      free_sigma=("fsig" not in ch), swap_frac=swap_frac,
                      eval_mask=common, crlb=False)
        rows.append(dict(variant=name, wc=res["wc"], r_factor=res["r_factor"],
                         r_factor_common=res["r_factor_common"],
                         a0=res["a0"], sigma2_nn=res["sigma2_nn"]))
        if verbose:
            print(f"    {name:16s} R={res['r_factor']:.4f} "
                  f"(common {res['r_factor_common']:.4f})  "
                  f"WC={np.round(res['wc'], 3)}")
        del S
    W = np.array([r["wc"] for r in rows])
    rf = np.array([r["r_factor_common"] if r["r_factor_common"] is not None
                   else r["r_factor"] for r in rows])
    # A variant only counts as an alternative the data cannot reject if it
    # fits comparably well ON THE COMMON RANGE - scoring each variant over its
    # own window would automatically favour the narrow ones.
    keep = rf <= 1.3 * rf.min()
    Wk = W[keep]
    return dict(variants=rows,
                wc_mean=W.mean(0).tolist(), wc_sd=W.std(0).tolist(),
                wc_min=Wk.min(0).tolist(), wc_max=Wk.max(0).tolist(),
                wc_min_all=W.min(0).tolist(), wc_max_all=W.max(0).tolist(),
                n_admissible=int(keep.sum()), n_variants=int(len(rows)),
                admissible=[r["variant"] for r, k in zip(rows, keep) if k],
                r_factor_range=[float(rf.min()), float(rf.max())])


# ------------------------------------------------------------------ structure
def broadened_partial_rdf(species, S, r_grid, extra_sigma2=0.0):
    """Partial g_ab(r) with each shell broadened by its fitted sigma^2.

    The fit puts atoms on ideal sites, so the bare partial RDF is a comb of
    delta functions. What EXAFS actually sees is that comb convolved with the
    pair-distance distribution, whose width is the sigma^2 the fit already
    determined - so the broadened form is both the physical object and the
    one the fit constrains.
    """
    sp = np.asarray(species, int)
    d = np.linalg.norm((S.cell.ideal[S.j] + S.shift) - S.cell.ideal[S.i], axis=1)
    a, b = sp[S.i], sp[S.j]
    counts = np.bincount(sp, minlength=NSP)
    c = counts / sp.size
    rho = sp.size / abs(np.linalg.det(S.cell.H))
    g = np.zeros((NSP, NSP, r_grid.size))
    s2 = S.sigma2 + extra_sigma2
    norm = 4.0 * np.pi * r_grid ** 2 * rho
    for A in range(NSP):
        for B in range(NSP):
            m = (a == A) & (b == B)
            if not m.any():
                continue
            sig = np.sqrt(np.maximum(s2[m], 1e-6))
            acc = np.zeros_like(r_grid)
            for rr, ss in zip(d[m], sig):
                acc += np.exp(-0.5 * ((r_grid - rr) / ss) ** 2) / (ss * np.sqrt(2 * np.pi))
            g[A, B] = acc / max(counts[A], 1) / np.maximum(norm * c[B], 1e-30)
    tot = np.zeros_like(r_grid)
    sig = np.sqrt(np.maximum(s2, 1e-6))
    for rr, ss in zip(d, sig):
        tot += np.exp(-0.5 * ((r_grid - rr) / ss) ** 2) / (ss * np.sqrt(2 * np.pi))
    tot = tot / sp.size / np.maximum(norm, 1e-30)
    return g, tot


def shell_table(species, S, nshell=3):
    """Per-shell coordination numbers and Warren-Cowley parameters."""
    sp = np.asarray(species, int)
    d = np.linalg.norm((S.cell.ideal[S.j] + S.shift) - S.cell.ideal[S.i], axis=1)
    counts = np.bincount(sp, minlength=NSP)
    out = []
    for s in range(min(nshell, len(S.shells))):
        rs = float(S.shells[s])
        m = np.abs(d - rs) < 0.05
        a, b = sp[S.i[m]], sp[S.j[m]]
        N = np.zeros((NSP, NSP)); alpha = np.zeros((NSP, NSP))
        c = counts / sp.size
        for A in range(NSP):
            selA = a == A
            tot = int(selA.sum())
            for B in range(NSP):
                nAB = int((b[selA] == B).sum())
                N[A, B] = nAB / max(counts[A], 1)
                alpha[A, B] = 1.0 - (nAB / max(tot, 1)) / c[B]
        out.append(dict(shell=s + 1, radius=rs, multiplicity=int(m.sum() / sp.size),
                        N=N.tolist(), alpha=alpha.tolist()))
    return out
