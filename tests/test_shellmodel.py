"""The pair-count model must be the GPU pair sum, exactly, and its bounds sane.

  1. ShellModel |chi(R)| from the pair counts of a configuration equals
     Spectrum.chi_R for that configuration - with bond offsets, a third
     cumulant and per-shell sigma^2 all switched on.
  2. random_counts is the expectation over random configurations.
  3. Cramer-Rao bounds scale linearly with the noise, grow when higher shells
     are freed, and grow (or stay) when nuisance parameters are marginalised.

Needs the GPU for (1) only.
"""
import sys, os, warnings, time
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chimera import model as M
from chimera import shellmodel as SM

S = M.Spectrum(a0=3.55, dE0=(11.0, 20.0, 21.0), kmin=3.0, kmax=13.0, kweight=3,
               sigma_scale=2.5, ms=True, ms_amp=0.9, ms_c=1.2,
               sigma_shell_scale=[1.0, 1.3, 0.8, 1.1, 1.0, 0.9],
               dr=[[0.03, 0.005, -0.01], [0.005, 0.0, -0.015], [-0.01, -0.015, -0.02]],
               c3=[1.2e-4, 0.5e-4, 0, 0, 0, 0])
sm = SM.ShellModel.from_spectrum(S)
print(f"shells: {np.round(S.shells, 4)}  multiplicities {S.shell_mult.astype(int)}")

print("[1] pair-count model == GPU pair sum")
worst = 0.0
for seed in range(3):
    cfg = S.random_config(seed)
    N = sm.counts_of(cfg)
    a = sm.mag_R(N)
    b = S.chi_R(cfg)
    worst = max(worst, float(np.abs(a - b).max() / np.abs(b).max()))
print(f"    max rel diff over 3 configs: {worst:.1e}")
assert worst < 1e-12
t0 = time.perf_counter()
for _ in range(20):
    sm.set_params(a0=3.55 + 1e-4 * (_ % 2), sigma_scale=2.5)
    sm.mag_R(N)
dt = (time.perf_counter() - t0) / 20 * 1e3
print(f"    one full evaluation with a changed a0: {dt:.2f} ms")
sm.set_params(a0=3.55)

print("\n[2] random_counts is the expectation over random configurations")
acc = np.zeros_like(N)
nsamp = 60
for seed in range(nsamp):
    acc += sm.counts_of(S.random_config(100 + seed))
acc /= nsamp
Nr = sm.random_counts()
rel = np.abs(acc - Nr).max() / Nr.max()
print(f"    max |mean(counts) - expectation| / max = {rel:.3f} over {nsamp} draws "
      f"(alpha of the expectation: {np.abs(sm.alpha(Nr, 0)).max():.1e})")
assert rel < 0.05 and np.abs(sm.alpha(Nr, 0)).max() < 1e-12
assert np.allclose(sm.absorber_counts(Nr), np.bincount(S.random_config(0), minlength=3))

print("\n[3] Cramer-Rao bounds")
data = sm.mag_R(sm.counts_of(S.random_config(5)))
R = sm.R
idx = np.arange(R.size)
mask = (R > 1.4) & (R < 5.0)
sig1 = 0.02 * data[:, mask].max() * np.ones(3)
flat = dict(alpha=1e9)                      # no prior: the pure Cramer-Rao bound
b1 = SM.crlb(sm, Nr, data, sig1, mask, idx, shells=[0], nuisance=(), prior_sd=flat)
b2 = SM.crlb(sm, Nr, data, 2 * sig1, mask, idx, shells=[0], nuisance=(), prior_sd=flat)
ratio = np.array(b2["alpha_sd"]) / np.array(b1["alpha_sd"])
print(f"    shell-1 bounds at 2% noise, no prior: {np.round(b1['alpha_sd'], 3)}")
print(f"    doubling the noise scales them by {np.round(ratio, 3)}")
assert np.allclose(ratio, 2.0, rtol=1e-6)
b1p = SM.crlb(sm, Nr, data, sig1, mask, idx, shells=[0], nuisance=())
print(f"    same with the default alpha prior (sd 1.0): {np.round(b1p['alpha_sd'], 3)}")
assert np.all(np.array(b1p["alpha_sd"]) <= np.array(b1["alpha_sd"]) + 1e-9)
b1 = b1p
b3 = SM.crlb(sm, Nr, data, sig1, mask, idx, shells=None, nuisance=())
b4 = SM.crlb(sm, Nr, data, sig1, mask, idx, shells=None,
             nuisance=("a0", "sigma_scale", "ms_amp"))
print(f"    all shells free:              {np.round(b3['alpha_sd'], 3)}")
print(f"    ... plus nuisance marginalised: {np.round(b4['alpha_sd'], 3)}")
print(f"    data-only eigen-sd of the 18 SRO directions (step units): "
      f"{np.array2string(np.array(b3['eig_sd']), precision=1, max_line_width=120)}")
assert np.all(np.array(b3["alpha_sd"]) >= np.array(b1["alpha_sd"]) - 1e-9)
assert np.all(np.array(b4["alpha_sd"]) >= np.array(b3["alpha_sd"]) - 1e-9)
assert b3["n_sro_directions"] == 3 * S.nshell and len(b4["columns"]) == 3 * S.nshell + 3
b5 = SM.crlb(sm, Nr, data, sig1, mask, idx)          # default: + dE0 per edge, dr, C3
print(f"    ... plus dE0, dr and C3 marginalised: {np.round(b5['alpha_sd'], 3)}")
assert len(b5["columns"]) == 3 * S.nshell + 3 + 3 + 5 + 1
assert np.all(np.array(b5["alpha_sd"]) >= np.array(b4["alpha_sd"]) - 1e-9)
# the model must be left untouched by the nuisance finite differences
assert np.allclose(sm.mag_R(Nr), sm.mag_R(Nr)) and sm.p["a0"] == 3.55

print("\nOK")
