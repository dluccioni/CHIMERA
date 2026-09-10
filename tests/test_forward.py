"""Validate the CUDA forward model and its analytic adjoint.

Four checks:
  1. chi_forward           vs an independent NumPy implementation
  2. chi_forward_soft      with one-hot weights == chi_forward
  3. grad_pos / grad_soft  vs central finite differences
  4. analytic Debye-Waller vs explicit correlated-snapshot averaging
"""
import sys, os
import numpy as np
import cupy as cp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exafs_gpu.scattering import analytic_transition_metal_tables, NSP
from exafs_gpu.lattice import (FCCSupercell, force_constant_matrix, ThermalSampler,
                               static_relaxation)
from exafs_gpu.forward import ForwardModel, thermal_sigma2

rng = np.random.default_rng(7)
cell = FCCSupercell((4, 4, 4))
i, j, sh, off = cell.neighbour_list(5.6)
k = np.linspace(2.0, 15.0, 260)
tab = analytic_transition_metal_tables(k)
Phi = force_constant_matrix(cell, i, j, sh, k_bond=2.0, k_angle=0.27)
sig2 = thermal_sigma2(Phi, cell, i, j, sh, 300.0)
print(f"natoms={cell.natoms} npairs={i.size} nk={k.size}")
print(f"analytic sigma^2: NN={sig2[:12].mean():.5f}  all={sig2.mean():.5f} A^2")

fm = ForwardModel(cell, tab, i, j, sh, sigma2=sig2)

nconf = 3
sp = rng.integers(0, 3, (nconf, cell.natoms)).astype(np.int32)
u = 0.03 * rng.standard_normal((nconf, cell.natoms, 3))
pos = cell.ideal[None] + u
d_pos = cp.asarray(pos); d_sp = cp.asarray(sp)

# ---- 1. forward vs NumPy reference
g = cp.asnumpy(fm.chi(d_pos, d_sp))
ref = fm.chi_reference(pos, sp)
rel = np.abs(g - ref).max() / np.abs(ref).max()
print(f"\n[1] chi_forward vs NumPy reference: max rel err = {rel:.3e}  "
      f"{'PASS' if rel < 1e-11 else 'FAIL'}")

# ---- 2. soft with one-hot == discrete
w = np.zeros((nconf, cell.natoms, NSP))
for c in range(nconf):
    w[c, np.arange(cell.natoms), sp[c]] = 1.0
d_w = cp.asarray(w)
gs = cp.asnumpy(fm.chi_soft(d_pos, d_w))
rel2 = np.abs(gs - g).max() / np.abs(g).max()
print(f"[2] chi_forward_soft(one-hot) vs chi_forward: max rel err = {rel2:.3e}  "
      f"{'PASS' if rel2 < 1e-11 else 'FAIL'}")

# ---- 3. gradient checks
gchi = cp.asarray(rng.standard_normal((nconf, NSP, k.size)))


def loss_pos(p):
    return float((cp.asarray(gchi) * fm.chi(cp.asarray(p), d_sp)).sum())


gpos = cp.asnumpy(fm.grad_pos(d_pos, d_sp, gchi))
eps = 1e-6
err, mx = 0.0, 0.0
for _ in range(12):
    c = rng.integers(nconf); a = rng.integers(cell.natoms); x = rng.integers(3)
    pp = pos.copy(); pp[c, a, x] += eps
    pm = pos.copy(); pm[c, a, x] -= eps
    fd = (loss_pos(pp) - loss_pos(pm)) / (2 * eps)
    err = max(err, abs(fd - gpos[c, a, x])); mx = max(mx, abs(fd))
print(f"[3a] grad_pos vs finite diff: max abs err = {err:.3e} (scale {mx:.3e})  "
      f"{'PASS' if err < 1e-4 * max(mx, 1) else 'FAIL'}")

wr = rng.random((nconf, cell.natoms, NSP)) + 0.1
wr /= wr.sum(-1, keepdims=True)
d_wr = cp.asarray(wr)


def loss_w(ww):
    return float((cp.asarray(gchi) * fm.chi_soft(d_pos, cp.asarray(ww))).sum())


gp_s, gw_s = fm.grad_soft(d_pos, d_wr, gchi)
gw_s = cp.asnumpy(gw_s)
errw, mxw = 0.0, 0.0
for _ in range(12):
    c = rng.integers(nconf); a = rng.integers(cell.natoms); s = rng.integers(NSP)
    wp = wr.copy(); wp[c, a, s] += eps
    wm = wr.copy(); wm[c, a, s] -= eps
    fd = (loss_w(wp) - loss_w(wm)) / (2 * eps)
    errw = max(errw, abs(fd - gw_s[c, a, s])); mxw = max(mxw, abs(fd))
print(f"[3b] grad_soft(w) vs finite diff: max abs err = {errw:.3e} (scale {mxw:.3e})  "
      f"{'PASS' if errw < 1e-4 * max(mxw, 1) else 'FAIL'}")


def loss_ps(p):
    return float((cp.asarray(gchi) * fm.chi_soft(cp.asarray(p), d_wr)).sum())


gp_s = cp.asnumpy(gp_s)
errp, mxp = 0.0, 0.0
for _ in range(12):
    c = rng.integers(nconf); a = rng.integers(cell.natoms); x = rng.integers(3)
    pp = pos.copy(); pp[c, a, x] += eps
    pm = pos.copy(); pm[c, a, x] -= eps
    fd = (loss_ps(pp) - loss_ps(pm)) / (2 * eps)
    errp = max(errp, abs(fd - gp_s[c, a, x])); mxp = max(mxp, abs(fd))
print(f"[3c] grad_soft(pos) vs finite diff: max abs err = {errp:.3e} (scale {mxp:.3e})  "
      f"{'PASS' if errp < 1e-4 * max(mxp, 1) else 'FAIL'}")

# ---- 4. analytic Debye-Waller vs explicit correlated snapshot averaging
print("\n[4] Analytic Debye-Waller vs explicit thermal snapshot averaging")
sp1 = rng.integers(0, 3, cell.natoms).astype(np.int32)
ustat = static_relaxation(cell, sp1, i, j, sh)
base = cell.ideal + ustat

fm_nodw = ForwardModel(cell, tab, i, j, sh, sigma2=np.zeros(i.size))
chi_dw = cp.asnumpy(fm.chi(cp.asarray(base[None]), cp.asarray(sp1[None])))[0]

ts = ThermalSampler(Phi, 300.0, rng=np.random.default_rng(3))
for nsnap in [16, 64, 256, 1024]:
    uu = ts.draw(nsnap)
    pp = cp.asarray(base[None] + uu)
    ss = cp.asarray(np.repeat(sp1[None], nsnap, 0))
    ch = cp.asnumpy(fm_nodw.chi(pp, ss)).mean(0)
    num = np.sqrt(np.mean((ch - chi_dw) ** 2))
    den = np.sqrt(np.mean(chi_dw ** 2))
    print(f"   {nsnap:5d} snapshots: RMS difference = {num/den:6.2%} of signal")
print("   (converging to a small residual = harmonic DW is the right closed form;")
print("    the floor is the anharmonic/cumulant term the Gaussian average drops)")

# ---- 5. bond-length offsets and third cumulant: kernel vs reference, and adjoint
print("\n[5] per-shell pair-type bond offsets dr and third cumulant C3")
d0 = np.linalg.norm((cell.ideal[j] + sh) - cell.ideal[i], axis=1)
_, shell = np.unique(np.round(d0, 3), return_inverse=True)
fm_c = ForwardModel(cell, tab, i, j, sh, sigma2=sig2, shell=shell)
dr = rng.normal(0.0, 0.02, (fm_c.nshell, NSP, NSP))
dr = 0.5 * (dr + np.transpose(dr, (0, 2, 1)))          # symmetric in the pair
c3 = rng.normal(0.0, 1e-4, fm_c.nshell)
fm_c.set_dr(dr); fm_c.set_c3(c3)
g5 = cp.asnumpy(fm_c.chi(d_pos, d_sp))
ref5 = fm_c.chi_reference(pos, sp)
rel5 = np.abs(g5 - ref5).max() / np.abs(ref5).max()
chg = np.abs(g5 - g).max() / np.abs(g).max()
print(f"    chi_forward(dr, C3) vs NumPy reference: max rel err = {rel5:.3e}  "
      f"{'PASS' if rel5 < 1e-11 else 'FAIL'}   (changes chi by {chg:.1%})")
assert rel5 < 1e-11 and chg > 1e-3


def loss_pos5(p):
    return float((cp.asarray(gchi) * fm_c.chi(cp.asarray(p), d_sp)).sum())


gpos5 = cp.asnumpy(fm_c.grad_pos(d_pos, d_sp, gchi))
err5, mx5 = 0.0, 0.0
for _ in range(12):
    c = rng.integers(nconf); a = rng.integers(cell.natoms); x = rng.integers(3)
    pp = pos.copy(); pp[c, a, x] += eps
    pm = pos.copy(); pm[c, a, x] -= eps
    fd = (loss_pos5(pp) - loss_pos5(pm)) / (2 * eps)
    err5 = max(err5, abs(fd - gpos5[c, a, x])); mx5 = max(mx5, abs(fd))
print(f"    grad_pos(dr, C3) vs finite diff: max abs err = {err5:.3e} (scale {mx5:.3e})  "
      f"{'PASS' if err5 < 1e-4 * max(mx5, 1) else 'FAIL'}")
assert err5 < 1e-4 * max(mx5, 1)
# zero tables must reproduce the plain model bit for bit
fm_c.set_dr(np.zeros_like(dr)); fm_c.set_c3(np.zeros_like(c3))
assert np.array_equal(cp.asnumpy(fm_c.chi(d_pos, d_sp)), g)
print("    zero dr / C3 reproduce the harmonic model exactly")
