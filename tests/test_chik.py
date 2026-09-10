"""chi(k) input: the loader, the shared transform, and what the phase buys.

  1. File kinds are told apart and read back exactly: `R, chi_mag` CSV,
     `k, chi` CSV, and Athena's whitespace `.chi` with a `#` header.
  2. A ChiK built from the model's own chi(k) transforms to the model's
     complex chi(R) to machine precision, for one k weight and for three at
     once (channels edge-major); the pair-count model agrees.
  3. fit_scales / chi2 / r_factor reduce to the old formulas on real input,
     recover a known amplitude on complex input, and share one amplitude per
     edge across k-weight channels.
  4. RMCSampler's chi^2 on a complex target equals an independent NumPy
     evaluation, profiled and un-profiled, with nfit counting Re and Im.
  5. At equal noise the Cramer-Rao bound on the shell-1 alphas is tighter
     with the phase than without, for the same truth - the reason to ask
     for chi(k) in the first place.
  6. dataio.screen on a chi(k) tree applies the same QC cuts, keeps chi(k),
     fit_input hands back a ChiK, and the duplicate-edge check still fires.

Needs the GPU (Spectrum, sampler); about a minute.
"""
import sys, os, warnings, tempfile
import numpy as np
import cupy as cp
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exafs_gpu.scattering import NSP, ELEMENTS
from chimera import model as M, shellmodel as SM, dataio as io, chik as CK
from chimera.multift import MultiEdgeFT
from methods.method_a_rmc import RMCSampler

rng = np.random.default_rng(0)
S = M.Spectrum(a0=3.55, dE0=(11.0, 20.0, 21.0), kmin=3.0, kmax=13.0, kweight=3,
               sigma_scale=2.5, ms=True, ms_amp=0.9, ms_c=1.2,
               dr=[[0.03, 0.005, -0.01], [0.005, 0.0, -0.015], [-0.01, -0.015, -0.02]],
               c3=[1.2e-4])
cfg = S.random_config(0)
K_DATA = np.arange(0.0, 15.0 + 1e-9, 0.05)          # the Athena grid


def chik_of(species, noise=0.0, seed=0, scales=None):
    """chi(k) of a configuration as an experiment would deliver it: on the
    uniform 0.05 A^-1 grid, against each edge's own k axis, optionally with
    white noise in k."""
    chi = S.chi_k(species)
    r = np.random.default_rng(seed)
    out = []
    for e in range(NSP):
        ke = S.k_exp(e)
        ok = S.k_phys ** 2 > M.ETOK * S.dE0[e] + 1e-9
        y = np.interp(K_DATA, ke[ok], chi[e][ok], left=0.0, right=0.0)
        y *= 1.0 if scales is None else scales[e]
        out.append(y + noise * r.standard_normal(y.size))
    return CK.ChiK(K_DATA, out, elements=ELEMENTS)


# ------------------------------------------------------------ 1. the loader
print("[1] file kinds")
tmp = tempfile.mkdtemp(prefix="chimera_chik_")
ck0 = chik_of(cfg)
mag0 = np.abs(S.chi_R(cfg))[:, :326]
R326 = S.R[:326]
p_mag = os.path.join(tmp, "a_sample1.csv")
np.savetxt(p_mag, np.column_stack([R326, mag0[0]]), delimiter=",", header="R,chi_mag", comments="")
p_k = os.path.join(tmp, "b_sample1.csv")
np.savetxt(p_k, np.column_stack([K_DATA, ck0.chi[1]]), delimiter=",", header="k,chi", comments="")
p_ath = os.path.join(tmp, "c_sample1.chi")
with open(p_ath, "w") as f:
    f.write("# Athena data file -- Athena version 0.9.26\n# Saving c as chi(k)\n"
            "# .  Element=Ni   Edge=K\n#------------------------\n#  k chi chik chik2 chik3\n")
    for k, c in zip(K_DATA, ck0.chi[2]):
        f.write(f" {k:8.3f} {c: .10e} {k * c: .10e} {k ** 2 * c: .10e} {k ** 3 * c: .10e}\n")
for path, kind_exp, x_exp, y_exp in ((p_mag, "mag_R", R326, mag0[0]),
                                     (p_k, "chi_k", K_DATA, ck0.chi[1]),
                                     (p_ath, "chi_k", K_DATA, ck0.chi[2])):
    x, y, kind = io.load_one(path)
    assert kind == kind_exp, (path, kind)
    assert np.allclose(x, x_exp, atol=1e-3) and np.allclose(y, y_exp, rtol=1e-9, atol=1e-12), path
    print(f"    {os.path.basename(path):14s} -> {kind}  ({x.size} points)")
# and without any header the numbers decide
p_bare = os.path.join(tmp, "d_sample1.csv")
np.savetxt(p_bare, np.column_stack([K_DATA, ck0.chi[0]]), delimiter=",")
assert io.load_one(p_bare)[2] == "chi_k"
p_bare2 = os.path.join(tmp, "e_sample1.csv")
np.savetxt(p_bare2, np.column_stack([R326, mag0[1]]), delimiter=",")
assert io.load_one(p_bare2)[2] == "mag_R"
print("    headerless files classified by grid and sign")

# ------------------------------------------------------- 2. one transform
print("\n[2] data and model share the transform")
z1 = S.chi_R(cfg, complex_out=True)
d1 = ck0.to_R(S.window)
rel = np.abs(d1 - z1).max() / np.abs(z1).max()
print(f"    k^3: max rel diff {rel:.1e}, shape {d1.shape}")
assert rel < 1e-9 and d1.shape == (NSP, S.R.size)
assert np.allclose(np.abs(z1), S.chi_R(cfg)), "magnitude path changed"
S.set_window(kweight=(1, 2, 3))
z3 = S.chi_R(cfg, complex_out=True)
d3 = ck0.to_R(S.window)
rel3 = np.abs(d3 - z3).max() / np.abs(z3).max()
print(f"    k^1,2,3: max rel diff {rel3:.1e}, shape {d3.shape}, chan_edge {S.chan_edge.tolist()}")
assert rel3 < 1e-9 and d3.shape == (3 * NSP, S.R.size)
assert S.channel_labels() == ["Cr k^1", "Cr k^2", "Cr k^3", "Co k^1", "Co k^2", "Co k^3",
                              "Ni k^1", "Ni k^2", "Ni k^3"]
sm = SM.ShellModel.from_spectrum(S)
N = sm.counts_of(cfg)
relsm = np.abs(sm.chi_R(N) - z3).max() / np.abs(z3).max()
print(f"    pair-count model, 9 channels: max rel diff {relsm:.1e}")
assert relsm < 1e-12 and sm.nchan == 9
idx = np.arange(S.R.size)
mask = (S.R > 1.4) & (S.R < 5.0)
rf = sm.r_factor(N, ck0, mask, idx)
print(f"    R-factor of the truth against its own chi(k): {rf:.1e}")
assert rf < 1e-20

# --------------------------------------------------- 3. the comparison maths
print("\n[3] fit_scales / chi2 / r_factor")
m_re, d_re = np.abs(z1), np.abs(d1) * 1.3
old_sc = (m_re[:, mask] * d_re[:, mask]).sum(1) / (m_re[:, mask] ** 2).sum(1)
assert np.allclose(M.fit_scales(m_re, d_re, mask), old_sc) and np.allclose(old_sc, 1.3)
old_rf = (((m_re * old_sc[:, None] - d_re)[:, mask]) ** 2).sum() / (d_re[:, mask] ** 2).sum()
assert np.isclose(M.r_factor(m_re, d_re, mask, old_sc), old_rf)
sig = 0.1 * np.ones(NSP)
old_c2 = ((((m_re * old_sc[:, None] - d_re)[:, mask]) / sig[:, None]) ** 2).mean()
assert np.isclose(M.chi2(m_re, d_re, mask, sig, old_sc), old_c2)
print("    real input: identical to the previous formulas")
ck_s = chik_of(cfg, scales=[0.8, 1.1, 0.95])
t3 = ck_s.to_R(S.window)
sc = M.fit_scales(z3, t3, mask, S.chan_edge)
print(f"    complex, 3 k weights: per-channel scales {np.round(sc, 4)}")
assert np.allclose(sc, np.repeat([0.8, 1.1, 0.95], 3))
assert np.allclose(M.edge_scales(sc, S.chan_edge), [0.8, 1.1, 0.95])
assert M.r_factor(z3, t3, mask, sc) < 1e-24
sig9 = 0.01 * np.ones(9)
d = (z3 * sc[:, None] - t3)[:, mask] / sig9[:, None]
assert np.isclose(M.chi2(z3, t3, mask, sig9, sc), (np.abs(d) ** 2).sum() / (2 * d.size))
print("    known amplitudes recovered per edge; Re and Im counted separately")

# ------------------------------------------------------------ 4. the sampler
print("\n[4] sampler chi^2 on a complex target")
S.set_window(kweight=(2, 3))
truth = S.random_config(5)
ck_t = chik_of(truth, noise=0.002, seed=3)
tgt = ck_t.to_R(S.window)
sigR = np.full(tgt.shape, 0.02 * np.abs(tgt[:, mask]).max())
ft = MultiEdgeFT.from_spectrum(S)
smp = RMCSampler(S.fm, ft, tgt, mask, sigR, nrep=4, seed=0, swap_frac=1.0)
start = S.random_config(6)
smp.init_state(start, S.cell.ideal)
assert smp.complex and smp.nchan == 6 and smp.nkw == 2
assert smp.nfit == int(mask.sum()) * 6 * 2
for profile in (True, False):
    smp.profile_scale = profile
    got = float(cp.asnumpy(smp._chi2(smp.chi))[0])
    m = M.observe(S, start, tgt)
    sc = M.fit_scales(m, tgt, mask, S.chan_edge) if profile else np.ones(6)
    ref = M.chi2(m, tgt, mask, sigR[:, 0], sc)
    print(f"    profile_scale={profile!s:5s}: sampler {got:.10f}  numpy {ref:.10f}")
    assert np.isclose(got, ref, rtol=1e-10, atol=1e-13)
smp.profile_scale = True
s_smp = smp.scales()[0]
assert np.allclose(s_smp, M.fit_scales(M.observe(S, start, tgt), tgt, mask, S.chan_edge))
assert np.allclose(s_smp[:2], s_smp[0]) and np.allclose(s_smp[2:4], s_smp[2])
print("    profiled scale is one value per edge across its channels")
smp.run(400, log_every=10 ** 9)
tracked = cp.asnumpy(smp.chi2)
fresh = cp.asnumpy(smp._chi2(S.fm.chi(smp.pos, smp.species, normalise=False)))
print(f"    after 400 steps: max |tracked - recomputed| = {np.abs(tracked - fresh).max():.1e}")
assert np.abs(tracked - fresh).max() < 1e-9

# ------------------------------------------------- 5. the bound with the phase
print("\n[5] Cramer-Rao bound: phase vs magnitude at the same noise")
S.set_window(kweight=3)
sm = SM.ShellModel.from_spectrum(S)
Nr = sm.random_counts()
ck_r = chik_of(S.random_config(7))
tc = ck_r.to_R(S.window)
tm = np.abs(tc)
noise = 0.01 * np.abs(tc[:, mask]).max() * np.ones(NSP)
b_c = SM.crlb(sm, Nr, tc, noise, mask, idx)
b_m = SM.crlb(sm, Nr, tm, noise, mask, idx)
sd_c, sd_m = np.array(b_c["alpha_sd"]), np.array(b_m["alpha_sd"])
print(f"    shell-1 alpha 1-sigma, all nuisances free, complex:   {np.round(sd_c, 3)}")
print(f"    shell-1 alpha 1-sigma, all nuisances free, magnitude: {np.round(sd_m, 3)}")
print(f"    dE0 1-sigma (eV): complex {np.round(b_c['nuisance_sd']['dE0'], 2)}  "
      f"magnitude {np.round(b_m['nuisance_sd']['dE0'], 2)}")
assert np.all(sd_c <= sd_m * 1.02), "the phase must not loosen any bound"
assert sd_c.mean() < 0.7 * sd_m.mean(), "the phase should tighten the bounds substantially"
assert np.all(np.array(b_c["nuisance_sd"]["dE0"]) < np.array(b_m["nuisance_sd"]["dE0"]))
S.set_window(kweight=(1, 2, 3))
sm3 = SM.ShellModel.from_spectrum(S)
b_3 = SM.crlb(sm3, Nr, ck_r, ck_r.noise(S.window) + noise.mean() * 0.0 + 0.01 * np.abs(tc[:, mask]).max(),
              mask, idx)
print(f"    with k^1,2,3 at the k^3 channel noise level:            {np.round(b_3['alpha_sd'], 3)}")

# ------------------------------------------------------------ 6. the screen
print("\n[6] screening a chi(k) tree")
root = os.path.join(tmp, "Data")
truth_cfg = S.random_config(8)
for P in io.PRESSURES:
    for e, el in enumerate(ELEMENTS):
        folder = os.path.join(root, f"{el}_{P}GPa")
        os.makedirs(folder)
        for s in range(4):
            c = chik_of(truth_cfg, noise=0.001, seed=100 * P + 10 * e + s)
            y = c.chi[e]
            if s == 3 and P == 3:
                y = y * 0.05                        # an off-sample position
            if P == 7 and el == "Ni":
                y = chik_of(truth_cfg, noise=0.001, seed=100 * P + 10 + s).chi[1]  # Co copied
            np.savetxt(os.path.join(folder, f"run_sample{s}.csv"),
                       np.column_stack([K_DATA, y]), delimiter=",", header="k,chi", comments="")
spectra, report = io.screen(root)
assert all(spectra[(P, el)]["kind"] == "chi_k" for P in io.PRESSURES for el in ELEMENTS)
assert spectra[(3, "Cr")]["rejected"] and 3 in spectra[(3, "Cr")]["rejected"]
assert 3 not in spectra[(3, "Cr")]["chi_k"] and len(spectra[(0, "Cr")]["chi_k"]) == 4
print(f"    3 GPa Cr rejected: {spectra[(3, 'Cr')]['rejected']}")
dups = io.duplicate_edges(spectra)
print(f"    duplicate edges: {list(dups)}")
assert list(dups) == [(7, "Co", "Ni")]
fi = io.fit_input(spectra, 0)
assert isinstance(fi, CK.ChiK) and fi.nsp == NSP
one = io.fit_input(spectra, 0, sample=2)
assert isinstance(one, CK.ChiK)
assert np.allclose(one.chi[0], spectra[(0, "Cr")]["chi_k"][2])
assert io.locations(spectra, 3) == [0, 1, 2]
avg_mag = io.average(spectra, 0)
assert avg_mag.shape == (NSP, S.R.size) and np.all(avg_mag >= 0)
print(f"    fit_input -> {fi};  |chi(R)| medians kept for the figures: {avg_mag.shape}")
r_ideal = SM.ShellModel.from_spectrum(S).r_factor(Nr, fi, mask, idx)
print(f"    random-alloy R-factor against the screened median, k^1,2,3: {r_ideal:.4f}")
assert r_ideal < 0.05

print("\nOK")
