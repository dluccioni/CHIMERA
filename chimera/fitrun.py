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

`data` may be a |chi(R)| array on `R_data` or a `chik.ChiK` (measured
chi(k)). In the second case the target is the COMPLEX chi(R) obtained with
the Spectrum's own window and k weight(s), on the Spectrum's R grid (so
`R_data = data.R` and `idx` is the identity); every quantity below - the
sampler's sigma, the R-factor, the Cramer-Rao bounds - then counts Re and Im
as separate residuals. The result records `target_complex` and `kweights` so
the two kinds of fit are never confused downstream.
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from scipy.optimize import minimize, least_squares

from exafs_gpu.lattice import warren_cowley
from exafs_gpu.scattering import ELEMENTS, NSP
from methods.method_a_rmc import RMCSampler
from . import dataio as io
from . import model as M
from . import shellmodel as SM
from .multift import MultiEdgeFT
from .system import wc_labels
from .calibrate import mean_zero

CRLB_THRESHOLD = 0.3

# the default system's labels; `wc_labels(S.elements)` for any other
WC_LABELS = wc_labels(ELEMENTS)
WC_LABELS_TEX = wc_labels(ELEMENTS, tex=True)


def refine_local(S, cfg, idx, data, mask_d, a0_guess, sig_guess,
                 free_a0=True, free_sigma=True, free_dE0=False, N=None):
    """Refine a0, the sigma^2 scale and (optionally) the per-edge E0 for one
    spectrum.

    Runs on the pair-count model. By default the pair counts are the exact
    random-alloy expectation for the composition of `cfg` (so the result has
    no dependence on which random configuration was drawn); pass `N` to
    refine against the pair counts of an actual configuration instead, which
    is what the joint rounds of `fit_one` do. Each evaluation costs
    milliseconds. The refined values are pushed back into S.

    `free_dE0` adds two mean-zero per-edge E0 shifts (the common E0 stays at
    the calibration), the same freedom the least-squares harness has.

    A parameter being perturbed on purpose (the systematic scan) must be held
    fixed here, or the refinement simply undoes the perturbation.
    """
    sm = SM.ShellModel.from_spectrum(S)
    ne = sm.nsp - 1                       # free entries of a mean-zero E0 shift
    if N is None:
        N = sm.random_counts(counts=np.bincount(np.asarray(cfg, int), minlength=sm.nsp))
    sm.set_params(a0=a0_guess, sigma_scale=sig_guess)
    dE0b = np.asarray(sm.p["dE0"], float).copy()

    def unpack(p):
        kw, i = {}, 0
        if free_a0:
            kw["a0"] = float(p[i]); i += 1
        if free_sigma:
            kw["sigma_scale"] = float(np.exp(p[i])); i += 1
        if free_dE0:
            kw["dE0"] = dE0b + mean_zero(p[i:i + ne])
        return kw

    def obj(p):
        sm.set_params(**unpack(p))
        return sm.r_factor(N, data, mask_d, idx)
    x0 = (([a0_guess] if free_a0 else []) + ([np.log(sig_guess)] if free_sigma else [])
          + ([0.0] * ne if free_dE0 else []))
    if not x0:
        S.set_a0(a0_guess); S.set_sigma_scale(sig_guess)
        return a0_guess, sig_guess, obj([])
    opts = dict(maxiter=200, xatol=1e-5, fatol=1e-9)
    if free_dE0:
        # Nelder-Mead's default simplex is 0.00025 wide on a zero coordinate,
        # useless for an E0 in eV; give every coordinate a sensible first step
        step = (([0.004] if free_a0 else []) + ([0.08] if free_sigma else [])
                + [0.4] * ne)
        x0a = np.asarray(x0, float)
        opts.update(maxiter=200 * len(x0),
                    initial_simplex=np.array([x0a] + [x0a + np.eye(len(x0))[i] * step[i]
                                                     for i in range(len(x0))]))
    r = minimize(obj, x0, method="Nelder-Mead", options=opts)
    kw = unpack(r.x)
    a0 = kw.get("a0", a0_guess)
    ss = kw.get("sigma_scale", sig_guess)
    S.set_a0(a0); S.set_sigma_scale(ss)
    if free_dE0:
        S.set_window(dE0=kw["dE0"])
    return a0, ss, float(r.fun)


def joint_nuisance_fit(S, cfg, data, idx, mask_d, a0_guess, sig_guess,
                       free_a0=True, free_sigma=True, free_dE0=True,
                       nstart=3, seed=0, da0=0.02):
    """Nuisance parameters fitted JOINTLY with the first-shell order.

    Least squares on the pair-count model over the three symmetric
    short-range-order directions of shell 1 plus, when free, a0 (within
    `da0` of its guess), the sigma^2 scale and two mean-zero per-edge E0
    shifts; the per-edge amplitudes profiled; the residual weighted per
    channel by the residual the random-alloy model leaves, which is how the
    sampler weights its chi^2. Pushes a0, sigma^2 and dE0 into S and returns
    them with the continuous alpha and the weighted cost.

    Why: refining the nuisances on the random configuration and then holding
    them is a profile at alpha = 0. Along the (sigma^2, E0, alpha) valley the
    cost changes by under 1% while alpha(Cr-Cr) moves by 0.04-0.05, so where
    the nuisances are held decides alpha. Fitting them together is the
    defensible choice, and the sampler then finds the configuration at those
    values (measured to land within 0.003 of the continuous alpha).
    Coordinate descent - refine on the best configuration, re-sample - was
    tried first and stalls in that valley, and the unweighted R-factor it
    minimised is dominated by the k^3 channels besides.
    """
    sm = SM.ShellModel.from_spectrum(S)
    sm.set_params(a0=a0_guess, sigma_scale=sig_guess)
    ne = sm.nsp - 1
    cnt = np.bincount(np.asarray(cfg, int), minlength=sm.nsp)
    N0 = sm.random_counts(counts=cnt)
    d = sm.target(data)
    cplx = np.iscomplexobj(d)
    m0 = sm.observe(N0, d, idx)
    sc0 = M.fit_scales(m0, d, mask_d, sm.chan_edge)
    resid0 = (m0 * sc0[:, None] - d)[:, mask_d]
    sig = np.maximum(np.sqrt((np.abs(resid0) ** 2).mean(1) / (2.0 if cplx else 1.0)),
                     1e-12)
    dirs = [SM.dN_direction(sm.nsp, sm.nshell, 0, a, b)
            for a, b in SM.sro_directions(sm.nsp)]
    nsro = len(dirs)
    step = 0.02 * cnt.min() * sm.z[0] * (1.0 / sm.nsp)
    dE0b = np.asarray(sm.p["dE0"], float).copy()
    a0b, s0 = float(a0_guess), float(sig_guess)

    def unpack(x):
        N = N0 + step * sum(xi * dd for xi, dd in zip(x[:nsro], dirs))
        kw, i = {}, nsro
        if free_a0:
            kw["a0"] = float(x[i]); i += 1
        if free_sigma:
            kw["sigma_scale"] = float(np.exp(x[i])); i += 1
        if free_dE0:
            kw["dE0"] = dE0b + mean_zero(x[i:i + ne])
        return N, kw

    def model(x):
        N, kw = unpack(x)
        sm.set_params(**kw)
        m = sm.observe(N, d, idx)
        sc = M.fit_scales(m, d, mask_d, sm.chan_edge)
        return N, kw, m * sc[:, None]

    def res(x):
        r = (model(x)[2] - d)[:, mask_d] / sig[:, None]
        return (np.concatenate([r.real.ravel(), r.imag.ravel()]) if cplx
                else r.ravel())

    lo, hi, x0 = [-400.0] * nsro, [400.0] * nsro, [0.0] * nsro
    if free_a0:
        lo += [a0b - da0]; hi += [a0b + da0]; x0 += [a0b]
    if free_sigma:
        lo += [np.log(0.4)]; hi += [np.log(6.0)]; x0 += [np.log(s0)]
    if free_dE0:
        lo += [-4.0] * ne; hi += [4.0] * ne; x0 += [0.0] * ne
    x0 = np.asarray(x0, float)
    rng = np.random.default_rng(seed)
    best, bcost = None, np.inf
    for t in range(nstart):
        xs = x0.copy()
        if t:
            xs[:nsro] += rng.normal(0, 8, nsro)
            if free_dE0:
                xs[-ne:] += rng.normal(0, 0.5, ne)
        try:
            r = least_squares(res, np.clip(xs, lo, hi), bounds=(lo, hi),
                              xtol=1e-10, ftol=1e-10, max_nfev=600)
        except Exception:
            continue
        if r.cost < bcost:
            best, bcost = r.x, r.cost
    N, kw, _ = model(best)
    a0 = kw.get("a0", a0b)
    ss = kw.get("sigma_scale", s0)
    dE0 = np.asarray(kw.get("dE0", dE0b), float)
    S.set_a0(a0); S.set_sigma_scale(ss)
    if free_dE0:
        S.set_window(dE0=dE0)
    alpha = sm.alpha(N, 0)[np.triu_indices(sm.nsp)]
    return a0, ss, dE0, [float(v) for v in alpha], float(bcost)


def fit_one(S, R_data, data, idx, mask_d, a0_guess, sig_guess, seeds=(0, 1),
            nsteps=6000, nrep=48, refine=True, collect=True,
            free_a0=True, free_sigma=True, swap_frac=1.0, max_disp=0.10,
            disp_sigma=0.012, eval_mask=None, crlb=True, profile_scale=True,
            collect_beta=0.5, noise=None, joint=False, free_dE0=False,
            energy_prior=None, ms_library=None, n_alpha_shells=3):
    """Refine the lattice, then sample configurations. Returns a result dict.

    `energy_prior`  (V, T): weigh every move by the Boltzmann factor of the
                    alloy's pair interactions V (nshell, nsp, nsp; eV per
                    unordered pair, chimera/physics.py) at the temperature T
                    (K) the sample was last equilibrated at, on top of the
                    likelihood. The posterior samples then come from data x
                    prior; the cold rungs, which find the best-fit
                    configuration, are dominated by the data.
    `ms_library`    chimera.mslibrary.SupercellMS: multiple scattering from
                    the configuration's own species along every 3- and 4-leg
                    path, in place of the composition-averaged term
    `n_alpha_shells` shells whose Warren-Cowley parameters are reported
                    (`wc_shells*`; `wc` stays the first shell)

    `crlb`          also compute the Cramer-Rao bounds on the shell-1 alphas
    `profile_scale` profile the per-edge amplitude inside the sampler's chi^2
    `collect_beta`  rung (inverse temperature) posterior samples come from;
                    0.5 is the likelihood temperature, None the coldest rungs
    `noise`         per-channel measurement noise for the noise-only bound
                    (default: estimated from the structureless high-R band)
    `data`          a |chi(R)| array on R_data, or a ChiK (see module docstring)
    `joint`         fit the sigma^2 scale and, with `free_dE0`, two mean-zero
                    per-edge E0 shifts JOINTLY with the first-shell order on
                    the pair-count model (`joint_nuisance_fit`; a0 stays at
                    its random-configuration value) and sample at those
                    values. Off -
                    the default, and the behaviour before 2026-09-26 - the
                    nuisances stay where the random configuration put them,
                    which is a profile at alpha = 0; on this data that alone
                    moves alpha by 0.04-0.05 (testing/MODEL_CHANGES.md item 10).
    `free_dE0`      let the refinement move the per-edge E0 (mean-zero, so
                    the common E0 stays at the calibration)
    """
    cfg0 = S.random_config(0)
    if refine and (free_a0 or free_sigma or free_dE0):
        a0, ss, r_ideal = refine_local(S, cfg0, idx, data, mask_d, a0_guess,
                                       sig_guess, free_a0, free_sigma, free_dE0)
    else:
        S.set_a0(a0_guess); S.set_sigma_scale(sig_guess)
        a0, ss = a0_guess, sig_guess
        t0_ = M.target(data, S)
        m = M.observe(S, cfg0, t0_)[:, idx]
        r_ideal = M.r_factor(m, t0_, mask_d, M.fit_scales(m, t0_, mask_d, S.chan_edge))

    st = {}

    def prepare():
        """Target, weights and transforms at the CURRENT nuisance values.
        sigma for the sampler: the residual the IDEAL model leaves, i.e. the
        model error, since that dominates the measurement noise by ~5x here.
        For a complex target this is the per-COMPONENT sd (Re and Im)."""
        tgt = M.target(data, S)                 # |chi(R)| as given, or complex chi(R)
        cplx = np.iscomplexobj(tgt)
        assert tgt.shape[0] == S.nchan, (
            f"target has {tgt.shape[0]} channels but the model {S.nchan}: several "
            f"k weights need chi(k) data")
        m0 = M.observe(S, cfg0, tgt)[:, idx]
        scales = M.fit_scales(m0, tgt, mask_d, S.chan_edge)
        resid0 = (m0 * scales[:, None] - tgt)[:, mask_d]
        sig_data = np.sqrt((np.abs(resid0) ** 2).mean(1) / (2.0 if cplx else 1.0))
        sig = sig_data / scales
        target = np.zeros((S.nchan, S.R.size), complex if cplx else float)
        target[:, idx] = tgt / scales[:, None]
        mask = np.zeros(S.R.size, bool); mask[idx] = mask_d
        sigR = np.repeat(sig[:, None], S.R.size, axis=1)
        st.update(tgt=tgt, cplx=cplx, m0=m0, scales=scales, sig_data=sig_data,
                  target=target, mask=mask, sigR=sigR,
                  ft=MultiEdgeFT.from_spectrum(S))

    nsh_a = max(1, min(int(n_alpha_shells), len(S.shells)))

    def wc_all(sp):
        """(n_alpha_shells, npairs) Warren-Cowley parameters, triu order."""
        sp = np.asarray(sp, np.int32)
        return np.array([warren_cowley(sp, S.i, S.j, S.shift, S.cell, float(S.shells[s]),
                                       nsp=S.nsp)[np.triu_indices(S.nsp)]
                         for s in range(nsh_a)])

    def wc_of(sp):
        return wc_all(sp)[0]

    def sample(n, start=None, keep_samples=True):
        """One pass over the seeds at the current nuisances. `start` warm-starts
        every replica from (species, positions) instead of a random alloy."""
        W, chi2s, best, post, xrate, c20 = [], [], None, [], [], None
        for s in seeds:
            smp = RMCSampler(S.fm, st["ft"], st["target"], st["mask"], st["sigR"],
                             nrep=nrep, seed=int(s), swap_frac=swap_frac,
                             max_disp=max_disp, disp_sigma=disp_sigma,
                             collect_after=(int(n * 0.6) if (collect and keep_samples)
                                            else None),
                             collect_every=50, profile_scale=profile_scale,
                             collect_beta=collect_beta, energy_prior=energy_prior,
                             ms_library=(ms_library if ms_library is not None
                                         else getattr(S, "ms_library", None)))
            if start is None:
                smp.init_state(S.random_config(int(s)), S.cell.ideal)
            else:
                smp.init_state(np.asarray(start[0], np.int32),
                               np.asarray(start[1], float))
            if c20 is None:
                c20 = float(smp.chi2.min())
            smp.run(n, log_every=10 ** 9)
            xrate.append(float(np.nanmean(smp.exchange_rate)))
            spb, posb, c2 = smp.best()
            W.append(wc_all(spb))
            chi2s.append(float(c2))
            if collect and keep_samples:
                for sample_, _c2 in smp.samples:      # samples are (species, chi2)
                    post.append(wc_all(sample_))
            if best is None or c2 < best[1]:
                best = (spb.copy(), c2, posb.copy())
        return W, chi2s, best, post, xrate, c20

    joint_info = None
    if joint and refine and (free_sigma or free_dE0):
        # a0 stays where the random-configuration refinement put it. Along
        # the a0 / E0 / alpha ridge the cost changes by under 1% for a 2 mA
        # shift in a0 and 0.08 in alpha(Cr-Cr) (measured 2026-09-26: freeing
        # a0 here took alpha(Cr-Cr) from -0.03 to -0.11 for +0.002 A), so a
        # free a0 would make the answer a point on the ridge, not a fit.
        a0, ss, dE0j, wc_ls, cost = joint_nuisance_fit(
            S, cfg0, data, idx, mask_d, a0, ss, False, free_sigma, free_dE0)
        joint_info = dict(a0=float(a0), sigma_scale=float(ss),
                          dE0=[float(x) for x in dE0j], wc_ls=wc_ls, cost=cost)
    prepare()
    W, chi2s, best, post, xrate, chi2_0 = sample(nsteps)
    tgt, cplx, m0, scales, sig_data = (st["tgt"], st["cplx"], st["m0"], st["scales"],
                                       st["sig_data"])
    W_all = np.array(W)                         # (seeds, shells, pairs)
    post_all = np.array(post) if post else None
    W = W_all[:, 0]
    post = [p[0] for p in post] if post else []
    spb, posb = best[0], best[2]
    rms_disp = float(np.sqrt(((posb - S.cell.ideal) ** 2).sum(1).mean()))
    model_R = M.observe(S, spb, tgt, pos=posb)[:, idx]
    sc_fit = M.fit_scales(model_R, tgt, mask_d, S.chan_edge)
    # R-factor on a COMMON range, so variants fitted over different windows
    # can be compared: a narrower fit range always looks better on its own.
    if eval_mask is None:
        r_common = None
    else:
        sc_c = M.fit_scales(model_R, tgt, eval_mask, S.chan_edge)
        r_common = float(M.r_factor(model_R, tgt, eval_mask, sc_c))
    fit_R, ideal_R = model_R * sc_fit[:, None], m0 * scales[:, None]
    res = dict(
        a0=a0, nn_distance=float(S.shells[0]), sigma_scale=ss,
        dE0=[float(x) for x in S.dE0],
        joint=joint_info,
        sigma_shell_scale=S.sigma_shell_scale.tolist(), dr=S.dr.tolist(),
        c3=S.c3.tolist(),
        rms_displacement=rms_disp, swap_frac=float(swap_frac),
        multiple_scattering=bool(getattr(S, "chi_ms", None) is not None),
        ms_amp=getattr(S, "ms_amp", None), ms_c=getattr(S, "ms_c", None),
        sigma2_nn=float(_nn_sigma2(S)), shells=[float(x) for x in S.shells],
        scales=M.edge_scales(sc_fit, S.chan_edge).tolist(), chi2_start=chi2_0,
        chi2=float(np.mean(chi2s)), chi2_seeds=chi2s,
        r_factor_ideal=float(r_ideal),
        r_factor=float(M.r_factor(model_R, tgt, mask_d, sc_fit)),
        r_factor_common=r_common,
        target_complex=bool(cplx), kweights=list(S.kweights),
        channels=S.channel_labels(), n_fit=int(mask_d.sum() * S.nchan * (2 if cplx else 1)),
        # Stern's count per edge, SUMMED - with a per-edge k window each edge
        # brings its own dk and so its own information content, and a shared
        # window is just the case where the three terms happen to be equal.
        # Extra k-weight channels re-weight the same information, and a
        # magnitude carries about half of it.
        n_independent=float(np.sum(2 * (np.broadcast_to(np.atleast_1d(S.kmax), (S.nsp,))
                                        - np.broadcast_to(np.atleast_1d(S.kmin), (S.nsp,)))
                                   * (R_data[mask_d].max() - R_data[mask_d].min())
                                   / np.pi + 2) / (1.0 if cplx else 2.0)),
        wc=W.mean(0).tolist(), wc_sd_seeds=W.std(0).tolist(),
        wc_posterior_sd=(np.std(post, axis=0).tolist() if post else None),
        wc_posterior_mean=(np.mean(post, axis=0).tolist() if post else None),
        wc_shells=W_all.mean(0).tolist(),
        wc_shells_posterior_mean=(post_all.mean(0).tolist() if post_all is not None else None),
        wc_shells_posterior_sd=(post_all.std(0).tolist() if post_all is not None else None),
        energy_prior_T=(float(energy_prior[1]) if energy_prior is not None else None),
        ms_library=ms_library is not None,
        posterior_beta=collect_beta, exchange_rate=xrate,
        sigma_model=sig_data.tolist(),
        species=spb.astype(np.int8).tolist(),
        positions=(posb - S.cell.ideal).astype(np.float32).round(4).tolist()
        if swap_frac < 1.0 else None,
        model_mag=np.abs(fit_R).tolist(),
        ideal_mag=np.abs(ideal_R).tolist(),
    )
    if cplx:
        res.update(model_re=fit_R.real.tolist(), model_im=fit_R.imag.tolist(),
                   ideal_re=ideal_R.real.tolist(), ideal_im=ideal_R.imag.tolist())
    if crlb:
        res.update(crlb_bounds(S, spb, R_data, data, idx, mask_d, sig_data, noise))
    return res


MODEL_ARRAY_KEYS = ("model_mag", "ideal_mag", "model_re", "model_im", "ideal_re",
                    "ideal_im")


def model_arrays(res):
    """(fit, ideal) chi(R) arrays from a result dict: complex when the fit
    was against chi(k) data, |chi(R)| otherwise."""
    if "model_re" in res:
        fit = np.array(res["model_re"]) + 1j * np.array(res["model_im"])
        ideal = np.array(res["ideal_re"]) + 1j * np.array(res["ideal_im"])
        return fit, ideal
    return np.array(res["model_mag"]), np.array(res["ideal_mag"])


def crlb_bounds(S, species, R_data, data, idx, mask_d, sigma_model, noise=None,
                threshold=CRLB_THRESHOLD):
    """Cramer-Rao bounds on the shell-1 alphas at the fitted configuration.

    Three views: all shells and the nuisance parameters free at the residual
    level the fit actually has (the honest one, `wc_crlb`); the first shell
    alone (`wc_crlb_shell1_only`, the best case for this data); and all
    shells free at the measurement-noise level (`wc_crlb_noise`, what a
    perfect model would allow). The noise comes from the structureless
    high-R band of the data: 7.5 A up for a |chi(R)| file that stops at
    10 A, 15-25 A for the transform of chi(k) (Larch's convention).
    """
    sm = SM.ShellModel.from_spectrum(S)
    N = sm.counts_of(species)
    tgt = M.target(data, S)
    if noise is None:
        if M.is_chik(data):
            noise = data.noise(S.window)
        else:
            noise = np.array([io.noise_estimate(R_data, tgt[c]) for c in range(tgt.shape[0])])
    b_all = SM.crlb(sm, N, tgt, sigma_model, mask_d, idx, threshold=threshold)
    b_s1 = SM.crlb(sm, N, tgt, sigma_model, mask_d, idx, shells=[0], threshold=threshold)
    b_noise = SM.crlb(sm, N, tgt, noise, mask_d, idx, threshold=threshold)
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
    ("k min +0.5", dict(kmin=3.5)),
    # relative, so it probes the window whether production runs one k range or
    # one per edge. It used to name 12.0 absolutely, which on a 10.5 window
    # WIDENED past where Co's data stops - that variant scored 3-4x worse than
    # baseline at every pressure and was rejected as inadmissible, so it was
    # contributing nothing to the systematic band.
    ("k max -1.0", dict(dkmax=-1.0)),
    ("R 1.5-5.0", dict(rmin=1.5)),
    ("R 1.4-4.6", dict(rmax=4.6)),
    ("R 1.4-5.2", dict(rmax=5.2)),
    ("ms_amp x0.85", dict(fms=0.85)),
    ("ms_amp x1.15", dict(fms=1.15)),
    ("no dr", dict(dr_off=True)),
    ("no C3", dict(c3_off=True)),
]


def band_from_variants(rows, tol=1.3):
    """The Warren-Cowley spread over calibration variants that the data cannot
    reject, from the rows `systematic_band` produces (or reads back from
    `_variants_<P>.json`).

    Two things decide which variants count.

    A variant that changes the SET of k weights is not comparable at all: it
    fits a different number of channels, weighted differently, so its
    R-factor is on another scale (on the CrCoNi chi(k) data k^4 alone scored
    half the k^1,2,3 baseline). Those are reported but excluded.

    Among the comparable ones, "cannot reject" is measured against the
    BASELINE - the fit actually being reported - not against the best variant.
    Using the best made the band exclude the baseline itself as soon as some
    variant fitted much better, which happens with chi(k) data: the phase
    determines E0 well enough that the +-2 eV probe, sized for magnitude-only
    data, is now a several-sigma perturbation and moves the R-factor by tens
    of percent. A variant that fits BETTER than the baseline is always kept -
    the data does not reject it, and its presence is itself a signal that the
    calibration is not at its optimum.
    """
    W = np.array([r["wc"] for r in rows])
    rf = np.array([r["r_factor_common"] if r.get("r_factor_common") is not None
                   else r["r_factor"] for r in rows])
    nch = np.array([r.get("n_channels", 0) for r in rows])
    comparable = nch == nch[0]                    # baseline is rows[0]
    keep = comparable & (rf <= tol * rf[0])
    keep[0] = True                                # the reported fit is always in
    Wk = W[keep]
    better = [r["variant"] for r, k, c in zip(rows, rf < rf[0], comparable)
              if k and c]
    return dict(wc_mean=W[comparable].mean(0).tolist(),
                wc_sd=W[comparable].std(0).tolist(),
                wc_min=Wk.min(0).tolist(), wc_max=Wk.max(0).tolist(),
                wc_min_all=W[comparable].min(0).tolist(),
                wc_max_all=W[comparable].max(0).tolist(),
                n_admissible=int(keep.sum()), n_variants=int(len(rows)),
                n_comparable=int(comparable.sum()),
                admissible=[r["variant"] for r, k in zip(rows, keep) if k],
                excluded_incomparable=[r["variant"] for r, c in zip(rows, comparable)
                                       if not c],
                fit_better_than_baseline=better,
                r_factor_baseline=float(rf[0]),
                r_factor_range=[float(rf[comparable].min()),
                                float(rf[comparable].max())])


# variants that come in a +/- pair around the baseline, with the size of the
# probe. The band rescales each to the range the DATA supports for that
# parameter rather than to the arbitrary size the probe happened to use.
PAIRED_VARIANTS = {
    "a0":      (("a0 -0.005 A", -0.005), ("a0 +0.005 A", +0.005)),
    "dE0":     (("dE0 -2 eV", -2.0), ("dE0 +2 eV", +2.0)),
    "sigma":   (("sigma x0.85", np.log(0.85)), ("sigma x1.15", np.log(1.15))),
    "ms_amp":  (("ms_amp x0.85", np.log(0.85)), ("ms_amp x1.15", np.log(1.15))),
}


def hamilton_ratio(n_ind, n_var, p=1, alpha=0.05, floor=1.05):
    """The R-factor ratio above which a variant is rejected by the data.

    Hamilton's R-ratio test for `p` constrained parameters on Stern's count.
    `floor` keeps the threshold above the fitting procedure's own
    reproducibility (the RMC search returns R to about 0.5%), so a variant is
    never rejected on a difference the sampler could have produced.
    """
    n = max(float(n_ind) - float(n_var), 1.0)
    crit = float(np.sqrt(1.0 + (p / n) * stats.f.ppf(1.0 - alpha, p, n)))
    return max(crit, float(floor))


def parameter_1sigma(rows, key, conf=0.317, n_ind=68.0, n_var=11.0):
    """The 1-sigma half-width of one nuisance parameter, from the curvature of
    R against it.

    The +/- probes give three points (minus, baseline, plus). A parabola
    through them has a minimum and a curvature, and the R-ratio test converts
    that curvature into a confidence interval. This replaces the probe size -
    which is arbitrary, and on the 2026-09-16 data was 4x wider than the data
    supports for a0 - with the interval the measurement actually allows.
    """
    by = {r["variant"]: r for r in rows}
    (nm_lo, x_lo), (nm_hi, x_hi) = PAIRED_VARIANTS[key]
    if nm_lo not in by or nm_hi not in by or not rows:
        return None
    rf = lambda r: (r["r_factor_common"] if r.get("r_factor_common") is not None
                    else r["r_factor"])
    x = np.array([x_lo, 0.0, x_hi], float)
    R = np.array([rf(by[nm_lo]), rf(rows[0]), rf(by[nm_hi])], float)
    c = np.polyfit(x, R, 2)
    if c[0] <= 0:                       # not convex: the probe says nothing
        return None
    xmin = -c[1] / (2 * c[0])
    Rmin = float(np.polyval(c, xmin))
    n = max(float(n_ind) - float(n_var), 1.0)
    ratio = np.sqrt(1.0 + (1.0 / n) * stats.f.ppf(1.0 - conf, 1, n))
    disc = c[1] ** 2 - 4 * c[0] * (c[2] - Rmin * ratio)
    if disc <= 0:
        return None
    half = float(np.sqrt(disc) / (2 * c[0]))
    return dict(half_width=half, centre=float(xmin), r_min=Rmin,
                probe=float(x_hi), scale=half / abs(x_hi))


def band_calibrated(rows, n_ind=68.0, n_var=11.0, alpha=0.05, conf=0.317,
                    floor=1.05, max_scale=1.0):
    """Systematic band with both of the fixes described in the module notes.

    1. A paired variant is rescaled to the 1-sigma range of its own parameter.
       The +-0.005 A probe on a0 produced HALF the whole band on the 2026-09-16
       data while the data pins a0 to +-0.0012 A, so the probe was reporting a
       lattice constant the measurement had already excluded.
    2. Unpaired variants - the ones that are genuine alternative choices rather
       than a perturbation, like dropping dr or moving the R range - are
       admitted by `hamilton_ratio` instead of an arbitrary R-factor ratio.

    Rescaling only ever shrinks a probe (`max_scale` caps it at 1), so the band
    cannot be widened by a parameter the data happens to constrain poorly.
    """
    base = np.asarray(rows[0]["wc"], float)
    nch = np.array([r.get("n_channels", 0) for r in rows])
    comparable = nch == nch[0]
    rf = np.array([(r["r_factor_common"] if r.get("r_factor_common") is not None
                    else r["r_factor"]) for r in rows], float)
    thr = hamilton_ratio(n_ind, n_var, alpha=alpha, floor=floor)

    paired = {nm: key for key, pr in PAIRED_VARIANTS.items() for nm, _ in pr}
    scales, dev, used, dropped = {}, [], [], []
    for key in PAIRED_VARIANTS:
        got = parameter_1sigma(rows, key, conf=conf, n_ind=n_ind, n_var=n_var)
        scales[key] = min(got["scale"], max_scale) if got else max_scale
    for i, r in enumerate(rows):
        if i == 0 or not comparable[i]:
            continue
        nm = r["variant"]
        d = np.asarray(r["wc"], float) - base
        if nm in paired:
            f = scales[paired[nm]]
            dev.append(d * f)
            used.append(f"{nm} x{f:.2f}")
        elif rf[i] <= thr * rf[0]:
            dev.append(d)
            used.append(nm)
        else:
            dropped.append(f"{nm} (R ratio {rf[i] / rf[0]:.2f} > {thr:.2f})")
    D = np.array(dev) if dev else np.zeros((1, base.size))
    lo = base + np.minimum(D.min(0), 0.0)
    hi = base + np.maximum(D.max(0), 0.0)
    return dict(wc_min=lo.tolist(), wc_max=hi.tolist(),
                wc_half_width=(0.5 * (hi - lo)).tolist(),
                hamilton_threshold=float(thr),
                parameter_scales={k: float(v) for k, v in scales.items()},
                parameter_1sigma={k: (parameter_1sigma(rows, k, conf=conf,
                                                       n_ind=n_ind, n_var=n_var)
                                      or {}) for k in PAIRED_VARIANTS},
                admissible=used, rejected=dropped,
                n_used=len(dev), n_variants=len(rows))


def systematic_band(R_data, data, cal, seeds=(0,), nsteps=6000, nrep=48,
                    variants=CALIB_VARIANTS, verbose=False, swap_frac=1.0,
                    tol=1.3, joint=False, free_dE0=False):
    """Refit under each calibration variant; the spread IS the systematic error.

    Every variant is a choice the data cannot distinguish - they all fit to
    within a few percent in R-factor - so the range of Warren-Cowley values
    they produce is the honest uncertainty on those parameters.

    With chi(k) data the k-weight and k-window variants re-transform the data
    as well as the model (the window is then a weighting we choose, not a
    property of the files), so they probe the model's robustness rather than
    an unknown processing choice.
    """
    rows = []
    common = (R_data > cal["rmin"]) & (R_data < cal["rmax"])
    for name, ch in variants:
        a0 = cal["a0"] + ch.get("da0", 0.0)
        dE0 = [d + ch.get("ddE0", 0.0) for d in cal["dE0"]]
        ss = cal["sigma_scale"] * ch.get("fsig", 1.0)
        ms_amp = cal.get("ms_amp", 1.0) * ch.get("fms", 1.0)
        kw = ch.get("kweight", cal["kweight"])
        kmin = ch.get("kmin", cal["kmin"]); kmax = cal["kmax"]
        if "kmax" in ch:
            # A variant names one number, but production may run a per-edge
            # window sitting at each edge's own data limit. Taking the number
            # literally would push the short edge PAST its data while pulling
            # the long ones in, so cap instead of replace.
            kmax = (min(kmax, ch["kmax"]) if np.isscalar(kmax)
                    else np.minimum(np.asarray(kmax, float), ch["kmax"]))
        if "dkmax" in ch:
            # relative, so it is the same perturbation whether the window is
            # one number or one per edge, and always lands inside the data
            kmax = (kmax + ch["dkmax"] if np.isscalar(kmax)
                    else np.asarray(kmax, float) + ch["dkmax"])
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
                      eval_mask=common, crlb=False, joint=joint,
                      free_dE0=(free_dE0 and "ddE0" not in ch))
        rows.append(dict(variant=name, wc=res["wc"], r_factor=res["r_factor"],
                         r_factor_common=res["r_factor_common"],
                         a0=res["a0"], sigma2_nn=res["sigma2_nn"],
                         n_channels=len(res["channels"]),
                         kweights=list(res["kweights"])))
        if verbose:
            print(f"    {name:16s} R={res['r_factor']:.4f} "
                  f"(common {res['r_factor_common']:.4f})  "
                  f"WC={np.round(res['wc'], 3)}")
        del S
    return dict(variants=rows, **band_from_variants(rows, tol=tol))


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
    nsp = S.nsp
    d = np.linalg.norm((S.cell.ideal[S.j] + S.shift) - S.cell.ideal[S.i], axis=1)
    a, b = sp[S.i], sp[S.j]
    counts = np.bincount(sp, minlength=nsp)
    c = counts / sp.size
    rho = sp.size / abs(np.linalg.det(S.cell.H))
    g = np.zeros((nsp, nsp, r_grid.size))
    s2 = S.sigma2 + extra_sigma2
    norm = 4.0 * np.pi * r_grid ** 2 * rho
    for A in range(nsp):
        for B in range(nsp):
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
    nsp = S.nsp
    d = np.linalg.norm((S.cell.ideal[S.j] + S.shift) - S.cell.ideal[S.i], axis=1)
    counts = np.bincount(sp, minlength=nsp)
    out = []
    for s in range(min(nshell, len(S.shells))):
        rs = float(S.shells[s])
        m = np.abs(d - rs) < 0.05
        a, b = sp[S.i[m]], sp[S.j[m]]
        N = np.zeros((nsp, nsp)); alpha = np.zeros((nsp, nsp))
        c = counts / sp.size
        for A in range(nsp):
            selA = a == A
            tot = int(selA.sum())
            for B in range(nsp):
                nAB = int((b[selA] == B).sum())
                N[A, B] = nAB / max(counts[A], 1)
                alpha[A, B] = 1.0 - (nAB / max(tot, 1)) / c[B]
        out.append(dict(shell=s + 1, radius=rs, multiplicity=int(m.sum() / sp.size),
                        N=N.tolist(), alpha=alpha.tolist()))
    return out
