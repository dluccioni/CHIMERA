"""Closed loop on chi(k): an ordered truth fitted from its complex chi(R),
against the same truth fitted from |chi(R)| alone.

The truth carries alpha ~ +0.4 in the first shell (Cr and Ni avoiding
themselves), bond offsets, a third cumulant, per-edge amplitudes and white
noise in k. The calibration is taken as KNOWN here - the point is the
target, not the search (tests/test_closed_loop.py covers that) - and both
fits get the same sampler budget. What must hold:

  * the fit to complex chi(R) returns shell-1 alphas within its own
    reported Cramer-Rao bound, and marks more of them determined;
  * that bound is tighter than the magnitude fit's, and the alpha error is
    not worse;
  * a fit with three k weights at once runs and does at least as well.

Needs the GPU; a few minutes.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exafs_gpu.lattice import warren_cowley
from exafs_gpu.scattering import NSP, ELEMENTS
from chimera import model as M, calibrate as C, fitrun as F, shellmodel as SM, chik as CK

COMP = np.array([167, 167, 166]) / 500.0
TRUE = dict(a0=3.552, dE0=(12.5, 19.0, 21.5), kmin=3.0, kmax=13.0, kweight=3,
            sigma_scale=3.0, sigma_shell_scale=[1.0, 1.25, 0.85, 1.1, 1.0, 1.0],
            dr=SM.dr_from_free([0.030, 0.008, -0.012, 0.0, -0.010], COMP),
            c3=[1.2e-4], ms_amp=0.9, ms_c=1.15)
SCALES = np.array([0.85, 1.10, 0.95])
NOISE_K = 0.0025             # white noise in chi(k); ~1.5% of the k^3 first-shell peak
NSTEPS, NREP = 3000, 48
K_DATA = np.arange(0.0, 15.0 + 1e-9, 0.05)
t_all = time.time()

S_true = M.Spectrum(**TRUE)
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


def measure(cfg, seed):
    """chi(k) as an experiment delivers it: each edge on the uniform grid
    against its own k axis, scaled per edge, with white noise in k."""
    chi = S_true.chi_k(cfg)
    r = np.random.default_rng(seed)
    out = []
    for e in range(NSP):
        ke = S_true.k_exp(e)
        ok = S_true.k_phys ** 2 > M.ETOK * S_true.dE0[e] + 1e-9
        y = np.interp(K_DATA, ke[ok], chi[e][ok], left=0.0, right=0.0) * SCALES[e]
        out.append(y + NOISE_K * r.standard_normal(y.size))
    return CK.ChiK(K_DATA, out, elements=ELEMENTS)


def show(tag, res, a_true, dt):
    wc, crlb = np.array(res["wc"]), np.array(res["wc_crlb"])
    err = np.abs(wc - a_true)
    print(f"  [{tag}] {dt:.0f}s  chi2 {res['chi2_start']:.3f} -> {res['chi2']:.3f}, "
          f"R {res['r_factor_ideal']:.5f} -> {res['r_factor']:.5f}, scales "
          f"{np.round(res['scales'], 3)}, channels {len(res['channels'])}, "
          f"n_fit {res['n_fit']}")
    print("        " + "  ".join(f"{l:>6s}" for l in labels))
    for name, v in (("alpha fit", wc), ("alpha true", a_true), ("|error|", err),
                    ("CRLB", crlb), ("CRLB noise", np.array(res["wc_crlb_noise"]))):
        print(f"    {name:11s}" + "  ".join(f"{x:6.3f}" for x in v))
    print(f"    determined: {res['wc_determined']}")
    return err, crlb


cfg = sro_truth(1)
a_true = alpha_of(cfg)
ck = measure(cfg, 12)
print(f"ordered truth, alpha = {np.round(a_true, 3)}")
noise_R = ck.noise(S_true.window)
peak = np.abs(ck.to_R(S_true.window)[:, (ck.R > 1.4) & (ck.R < 5.0)]).max()
print(f"R-space noise per component {np.round(noise_R, 4)} = {noise_R.mean() / peak:.1%} of the "
      f"first-shell peak")

# --------------------------------------------------- magnitude only, as before
S = M.Spectrum(**TRUE)
R326 = S.R[:326]
mag = np.abs(ck.to_R(S.window))[:, :326]
idx_m = M.match_grid(S.R, R326)
mask_m = C.fit_mask(R326)
t0 = time.time()
res_m = F.fit_one(S, R326, mag, idx_m, mask_m, TRUE["a0"], TRUE["sigma_scale"],
                  seeds=(0,), nsteps=NSTEPS, nrep=NREP)
err_m, crlb_m = show("|chi(R)|, k^3", res_m, a_true, time.time() - t0)
assert not res_m["target_complex"] and res_m["n_fit"] == int(mask_m.sum()) * NSP

# ------------------------------------------------------ complex chi(R), k^3
S = M.Spectrum(**TRUE)
idx_c = M.match_grid(S.R, ck.R)
mask_c = C.fit_mask(ck.R)
t0 = time.time()
res_c = F.fit_one(S, ck.R, ck, idx_c, mask_c, TRUE["a0"], TRUE["sigma_scale"],
                  seeds=(0,), nsteps=NSTEPS, nrep=NREP)
err_c, crlb_c = show("chi(k) -> complex chi(R), k^3", res_c, a_true, time.time() - t0)
assert res_c["target_complex"] and res_c["n_fit"] == int(mask_c.sum()) * NSP * 2
assert np.all(err_c < 3.0 * crlb_c + 0.05), "complex fit error exceeds its own bound"
assert crlb_c.mean() < crlb_m.mean(), "the phase should tighten the bound"
assert np.sum(res_c["wc_determined"]) >= np.sum(res_m["wc_determined"])
assert err_c.mean() <= err_m.mean() + 0.05, "the complex fit should not be worse"
assert abs(res_c["a0"] - TRUE["a0"]) < 0.004, "a0 from the phase"
assert np.all(np.abs(np.array(res_c["scales"]) - SCALES) < 0.08), "per-edge amplitudes"

# --------------------------------------------------- complex, k^1,2,3 at once
S = M.Spectrum(**dict(TRUE, kweight=(1, 2, 3)))
t0 = time.time()
res_3 = F.fit_one(S, ck.R, ck, idx_c, mask_c, TRUE["a0"], TRUE["sigma_scale"],
                  seeds=(0,), nsteps=NSTEPS, nrep=NREP)
err_3, crlb_3 = show("chi(k) -> complex chi(R), k^1,2,3", res_3, a_true, time.time() - t0)
assert res_3["kweights"] == [1, 2, 3] and len(res_3["channels"]) == 9
assert res_3["n_fit"] == int(mask_c.sum()) * 9 * 2
assert np.all(err_3 < 3.0 * crlb_3 + 0.05)
assert err_3.mean() <= err_m.mean() + 0.05

print(f"\nmean |alpha error|: magnitude {err_m.mean():.3f}   complex k^3 {err_c.mean():.3f}   "
      f"complex k^1,2,3 {err_3.mean():.3f}")
print(f"mean CRLB:          magnitude {crlb_m.mean():.3f}   complex k^3 {crlb_c.mean():.3f}   "
      f"complex k^1,2,3 {crlb_3.mean():.3f}")
print(f"\nOK  [{time.time() - t_all:.0f}s]")
