"""The incremental delta-chi the sampler relies on must not drift.

Every accepted move updates chi by adding the change from the handful of
pairs the move touched, rather than recomputing the whole spectrum. If that
bookkeeping is even slightly wrong the error compounds silently over tens of
thousands of steps and the reported chi^2 stops describing the configuration
the sampler is actually holding.

This checks it the only way that settles it: run the sampler, then recompute
chi^2 from scratch for the final configuration and compare. It also confirms
the composition is conserved (swaps must not create or destroy atoms).

An earlier version of the affected-pair list contained duplicates - every
pair twice on displacement moves, and the shared pair when two swapped atoms
were neighbours - which showed up here as a drift of 4.5e-4 per 200 steps.

Self-contained: builds its own target from a known configuration, so it needs
nothing but the CHIMERA modules themselves.
"""
import sys, os, warnings
import numpy as np
import cupy as cp
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exafs_gpu.scattering import default_tables, NSP
from exafs_gpu.lattice import (CrystalSupercell, force_constant_matrix,
                               warren_cowley)
from exafs_gpu.forward import ForwardModel, thermal_sigma2
from exafs_gpu.fourier import FourierTransform
from methods.method_a_rmc import RMCSampler

NCELL, RMAX, NSTEPS = (4, 4, 4), 6.2, 4000

k = np.linspace(1.5, 16.0, 300)
tab = default_tables(k, "analytic")
cell = CrystalSupercell("fcc", NCELL, a0=3.56)
i_idx, j_idx, shift, _ = cell.neighbour_list(RMAX)
Phi = force_constant_matrix(cell, i_idx, j_idx, shift)
sig2 = thermal_sigma2(Phi, cell, i_idx, j_idx, shift, 300.0)
fm = ForwardModel(cell, tab, i_idx, j_idx, shift, sigma2=sig2)
ft = FourierTransform(k, 3.0, 14.0, 2, xp=cp)

n = cell.natoms
rng = np.random.default_rng(0)
counts = np.full(NSP, n // NSP); counts[-1] = n - counts[:-1].sum()
truth = np.repeat(np.arange(NSP), counts).astype(np.int32); rng.shuffle(truth)
target = cp.asnumpy(ft.magnitude(
    fm.chi(cp.asarray(cell.ideal[None]), cp.asarray(truth[None]))))[0]
mask = ft.window_mask(1.2, 5.5)
sigR = np.full_like(target, 0.02 * target[:, mask].max())

start = truth.copy(); rng.shuffle(start)
smp = RMCSampler(fm, ft, target, mask, sigR, nrep=8, seed=0, swap_frac=1.0)
smp.init_state(start, cell.ideal)
c0 = float(smp.chi2.min())
smp.run(NSTEPS, log_every=10 ** 9)

# tracked value vs an independent recomputation of the same configuration:
# a full forward pass through the same chi^2 (which profiles the per-edge
# scale, so the formula is not repeated here) against the incremental sum
tracked = cp.asnumpy(smp.chi2)
fresh = cp.asnumpy(smp._chi2(fm.chi(smp.pos, smp.species, normalise=False)))
drift = float(np.max(np.abs(tracked - fresh)))
# and the un-profiled form must agree with the plain formula
smp.profile_scale = False
chi = fm.chi(smp.pos, smp.species)
d = (ft.magnitude(chi) - cp.asarray(target)[None]) / cp.asarray(sigR)[None]
plain = cp.asnumpy((d[:, :, cp.asarray(mask)] ** 2).sum(axis=(1, 2)) / smp.nfit)
assert np.allclose(cp.asnumpy(smp._chi2(fm.chi(smp.pos, smp.species, normalise=False))),
                   plain, rtol=1e-12, atol=1e-14)
smp.profile_scale = True

print(f"[1] incremental delta-chi over {NSTEPS} steps x {smp.nrep} replicas")
print(f"    chi2 {c0:.4f} -> {tracked.min():.4f}")
print(f"    max |tracked - recomputed| = {drift:.2e}")
assert drift < 1e-9, f"delta-chi drifted by {drift:.2e}"

print("\n[2] composition conserved")
sp = cp.asnumpy(smp.species)
for r in range(smp.nrep):
    got = np.bincount(sp[r], minlength=NSP)
    assert np.array_equal(got, counts), f"replica {r}: {got} != {counts}"
print(f"    every replica still holds {counts.tolist()}")

print("\n[3] atoms stay on the lattice to within a small tolerance")
# swap_frac=1.0 always ATTEMPTS a swap, but if all 8 candidate partners drawn
# happen to share the atom's species - probability (1/3)^8 = 1.5e-4 at
# equiatomic - the proposal falls through to a small displacement instead.
# Over a full run that leaves a sub-milliangstrom drift, ~1% of sigma.
disp = cp.asnumpy(smp.pos) - cell.ideal[None]
moved = float(np.abs(disp).max())
rms = float(np.sqrt((disp ** 2).sum(-1).mean()))
print(f"    max |displacement| = {moved:.2e} A,  rms = {rms:.2e} A")
print(f"    (first-shell sigma is {np.sqrt(sig2.mean()):.3f} A for comparison)")
assert rms < 0.02, "displacement fallback should stay far below sigma"

print("\n[4] the sampler moved toward the truth")
a_start = warren_cowley(start, i_idx, j_idx, shift, cell, float(cell.shell_radii(RMAX)[0][0]))
best_sp, _, _ = smp.best()
a_fit = warren_cowley(best_sp.astype(np.int32), i_idx, j_idx, shift, cell,
                      float(cell.shell_radii(RMAX)[0][0]))
a_true = warren_cowley(truth, i_idx, j_idx, shift, cell,
                       float(cell.shell_radii(RMAX)[0][0]))
iu = np.triu_indices(NSP)
e0 = np.abs(a_start[iu] - a_true[iu]).mean()
e1 = np.abs(a_fit[iu] - a_true[iu]).mean()
print(f"    WC mean abs error: start {e0:.3f} -> fit {e1:.3f}")
assert e1 < e0, "the fit should not be further from the truth than the start"

print("\nOK")
