"""Closed loop: synthesise |chi(R)| from a known configuration with known
processing choices, run the calibration and the fit blind, and compare what
comes back with the truth and with the reported Cramer-Rao bound.

Two truths, because they test different things:

  A. a RANDOM alloy with bond-length offsets, a third cumulant, per-shell
     sigma^2 factors, per-edge amplitudes and 1.5% noise. The calibration
     assumes a random alloy, so here it must recover the k weight, the
     window and C3, reach the noise floor, and return dE0 and the bond
     offsets within the uncertainty the same Fisher analysis assigns them.
     (From |chi(R)| alone the offsets trade against dE0 and against each
     other: they are only weakly identifiable, and this test documents that
     rather than pretending otherwise.)
  B. the same physics with strong short-range order (alpha ~ +0.4, Cr and
     Ni avoiding themselves). A random-alloy calibration can absorb part of
     that order into dE0, dr and a0 - which is exactly why those parameters
     are nuisances in the bound. What must hold is that the fit's alpha
     errors stay within the bound it reports, and that the "determined"
     entries are recovered.

Needs the GPU; about ten minutes.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exafs_gpu.lattice import warren_cowley
from exafs_gpu.scattering import NSP
from chimera import model as M, calibrate as C, fitrun as F, shellmodel as SM

COMP = np.array([167, 167, 166]) / 500.0
TRUE = dict(a0=3.552, dE0=(12.5, 19.0, 21.5), kmin=3.0, kmax=13.0, kweight=3,
            sigma_scale=3.0, sigma_shell_scale=[1.0, 1.25, 0.85, 1.1, 1.0, 1.0],
            dr=SM.dr_from_free([0.030, 0.008, -0.012, 0.0, -0.010], COMP),
            c3=[1.2e-4], ms_amp=0.9, ms_c=1.15)
SCALES = np.array([0.85, 1.10, 0.95])
NOISE = 0.015
DE = dict(de_maxiter=80, de_popsize=10)
t_all = time.time()

S_true = M.Spectrum(**TRUE)
R_data = S_true.R[:326]                      # the measured grid: 326 points to 10 A
idx = M.match_grid(S_true.R, R_data)
mask_d = C.fit_mask(R_data)
iu = np.triu_indices(NSP)
labels = ["CrCr", "CrCo", "CrNi", "CoCo", "CoNi", "NiNi"]


def sro_truth(seed, nsweeps=60, T=0.7):
    """Ising-like swaps favouring unlike nearest neighbours: alpha_AA > 0."""
    rng = np.random.default_rng(seed)
    sp = S_true.random_config(seed)
    n = sp.size
    nn = S_true.shell_idx == 0
    nbr = [S_true.j[nn][S_true.i[nn] == a] for a in range(n)]
    for _ in range(nsweeps * n):
        a, b = rng.integers(n, size=2)
        if sp[a] == sp[b]:
            continue
        sa, sb = sp[a], sp[b]
        dE = ((sp[nbr[a]] == sb).sum() + (sp[nbr[b]] == sa).sum()
              - (sp[nbr[a]] == sa).sum() - (sp[nbr[b]] == sb).sum())
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            sp[a], sp[b] = sb, sa
    return sp


def alpha_of(cfg):
    return warren_cowley(cfg, S_true.i, S_true.j, S_true.shift, S_true.cell,
                         float(S_true.shells[0]))[iu]


def synthesise(cfg, seed):
    clean = S_true.chi_R(cfg)[:, idx] * SCALES[:, None]
    return clean + np.random.default_rng(seed).standard_normal(clean.shape) * NOISE * clean[:, mask_d].max()


def report_cal(c, S, data, tag):
    """Print the recovered calibration next to the truth, and return the
    parameter errors together with the 1-sigma each carries at the noise
    level (from the same Fisher analysis the fit reports)."""
    err_dE0 = np.abs(np.array(c["dE0"]) - np.array(TRUE["dE0"]))
    err_dr = np.abs(np.array(c["dr"])[iu] - TRUE["dr"][iu])
    sm = SM.ShellModel.from_spectrum(S)
    noise = NOISE * data[:, mask_d].max() * np.ones(NSP)
    b = SM.crlb(sm, sm.random_counts(), data, noise, mask_d, idx)
    sd_dE0 = np.array(b["nuisance_sd"]["dE0"]); sd_dr = np.array(b["dr_entry_sd"])
    print(f"[{tag}] kweight {c['kweight']} (true 3); window {c['kmin']:.2f}-{c['kmax']:.2f} "
          f"(true 3-13); a0 {c['a0']:.4f} (true {TRUE['a0']}); "
          f"C3 {c['c3'][0]:.2e} (true {TRUE['c3'][0]:.1e}); R_ideal {c['r_factor_ideal']:.5f}")
    print(f"    dE0 err {np.round(err_dE0, 2)} eV   1-sigma at this noise {np.round(sd_dE0, 2)}")
    print(f"    dr fitted {np.round(np.array(c['dr'])[iu], 4)}  true {np.round(TRUE['dr'][iu], 4)}")
    print(f"    dr err    {np.round(err_dr, 4)}  1-sigma {np.round(sd_dr, 4)}")
    print(f"    sigma_shell fitted {np.round(c['sigma_shell_scale'], 2)}  true {TRUE['sigma_shell_scale']}")
    return err_dE0, sd_dE0, err_dr, sd_dr


# ============================================================ A. random alloy
cfg_A = S_true.random_config(3)
data_A = synthesise(cfg_A, 11)
print(f"A: random truth, alpha = {np.round(alpha_of(cfg_A), 3)}; noise {NOISE:.1%} of peak")
t0 = time.time()
cal, S = C.calibrate({0: data_A}, R_data, verbose=True, kweights=(2, 3, 4), a0_ambient=TRUE["a0"], **DE)
c = cal[0]
print(f"    calibration took {time.time() - t0:.0f}s")
err_dE0, sd_dE0, err_dr, sd_dr = report_cal(c, S, data_A, "A")
assert c["kweight"] == 3, "k weight not recovered"
assert abs(c["kmin"] - 3.0) < 0.3 and abs(c["kmax"] - 13.0) < 0.5, "window not recovered"
assert abs(c["c3"][0] - TRUE["c3"][0]) < 6e-5, "C3 not recovered"
assert c["r_factor_ideal"] < 0.003, "did not reach the noise floor"
assert np.all(err_dE0 < 3 * sd_dE0 + 0.3), "dE0 error beyond its own bound"
assert np.all(err_dr < 3 * sd_dr + 0.005), "bond-offset error beyond its own bound"
# the fit on a random truth must come back random, within its bound
res = F.fit_one(S, R_data, data_A, idx, mask_d, c["a0"], c["sigma_scale"], seeds=(0,),
                nsteps=3000, nrep=48)
wc, crlb = np.array(res["wc"]), np.array(res["wc_crlb"])
err = np.abs(wc - alpha_of(cfg_A))
print(f"    fit: alpha {np.round(wc, 3)}  |err| {np.round(err, 3)}  CRLB {np.round(crlb, 3)}  "
      f"determined {res['wc_determined']}")
assert np.all(err < 3.0 * crlb + 0.05)

# ======================================================= B. ordered alloy
cfg_B = sro_truth(1)
a_B = alpha_of(cfg_B)
data_B = synthesise(cfg_B, 12)
print(f"\nB: ordered truth, alpha = {np.round(a_B, 3)}")
t0 = time.time()
cal, S = C.calibrate({0: data_B}, R_data, verbose=False, kweights=(3,), a0_ambient=TRUE["a0"], **DE)
c = cal[0]
print(f"    calibration took {time.time() - t0:.0f}s")
report_cal(c, S, data_B, "B")
t0 = time.time()
res = F.fit_one(S, R_data, data_B, idx, mask_d, c["a0"], c["sigma_scale"], seeds=(0,),
                nsteps=4000, nrep=48)
wc, crlb, crn = np.array(res["wc"]), np.array(res["wc_crlb"]), np.array(res["wc_crlb_noise"])
err = np.abs(wc - a_B)
print(f"    fit took {time.time() - t0:.0f}s: chi2 {res['chi2_start']:.3f} -> {res['chi2']:.3f}, "
      f"R {res['r_factor_ideal']:.5f} -> {res['r_factor']:.5f}, exchange rate {res['exchange_rate'][0]:.2f}")
print("    " + "  ".join(f"{l:>6s}" for l in labels))
for name, v in (("alpha fit", wc), ("alpha true", a_B), ("|error|", err), ("CRLB", crlb),
                ("CRLB noise", crn), ("posterior sd", np.array(res["wc_posterior_sd"]))):
    print(f"    {name:13s}" + "  ".join(f"{x:6.3f}" for x in v))
print(f"    determined: {res['wc_determined']}")
det = np.array(res["wc_determined"])
assert np.all(err < 3.0 * crlb + 0.05), "fit error exceeds its own reported bound"
assert np.all(np.array(res["wc_posterior_sd"]) < 2.0 * crlb + 0.05), \
    "posterior spread claims more precision than the bound"
if det.any():
    assert np.all(err[det] < 3.0 * crlb[det] + 0.05)
print(f"\nOK  [{time.time() - t_all:.0f}s]")
