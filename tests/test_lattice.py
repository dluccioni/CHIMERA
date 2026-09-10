"""Validate the lattice geometry and the correlated-thermal-motion physics."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exafs_gpu.lattice import (FCCSupercell, force_constant_matrix, ThermalSampler,
                               static_relaxation, warren_cowley, pair_vectors)
from exafs_gpu.scattering import ELEMENTS

c = FCCSupercell((5, 5, 5))
print(f"atoms={c.natoms}  box={c.box}  a0={c.a0}")

r, n = c.shell_radii(5.6)
print("FCC shells (r, multiplicity) -- expect 2.518x12, 3.560x6, 4.361x24, 5.036x12:")
for a, b in zip(r, n):
    print(f"   {a:.3f}  x{b}")

i, j, sh, off = c.neighbour_list(5.6)
print(f"pairs={i.size}  per atom={i.size / c.natoms:.1f}")

# distances must be reproducible from the stored shifts
d = pair_vectors(c, c.ideal, i, j, sh)
rr = np.linalg.norm(d, axis=1)
assert rr.max() <= 5.6 + 1e-9 and rr.min() > 1e-6, "shift bookkeeping broken"
print(f"pair distance range: {rr.min():.3f} - {rr.max():.3f}  OK")

# ---- thermal physics ----
Phi = force_constant_matrix(c, i, j, sh)
ts = ThermalSampler(Phi, 300.0, rng=np.random.default_rng(1))
u = ts.draw(2000)
print(f"\nmodes={ts.nmode} (expect 3N-3 = {3 * c.natoms - 3})")
print(f"<u^2> per atom per axis = {ts.msd():.5f} A^2   (typical metal @300K: 0.004-0.008)")

nnm = rr < 2.6
e = d[nnm] / rr[nnm][:, None]
du = u[:, j[nnm], :] - u[:, i[nnm], :]
proj = (du * e[None]).sum(-1)
s2_bond = float(proj.var())
s2_uncorr = 2 * ts.msd()
print(f"\nNN sigma^2_bond (correlated)   = {s2_bond:.5f} A^2   <- physical, expect ~0.003-0.006")
print(f"NN sigma^2_bond if uncorrelated = {s2_uncorr:.5f} A^2   <- what naive RMC gives")
print(f"ratio = {s2_bond / s2_uncorr:.2f}  (correlation removes {(1 - s2_bond / s2_uncorr):.0%} of the MSRD)")

# 4th shell should be markedly less correlated than the 1st
s4 = rr > 4.9
e4 = d[s4] / rr[s4][:, None]
du4 = u[:, j[s4], :] - u[:, i[s4], :]
p4 = float((du4 * e4[None]).sum(-1).var())
print(f"4th shell sigma^2_bond = {p4:.5f} A^2  ratio to uncorrelated = {p4 / s2_uncorr:.2f}"
      f"   (should approach 1 as correlation decays)")

# ---- static relaxation carries chemistry ----
rng = np.random.default_rng(0)
sp = rng.integers(0, 3, c.natoms).astype(np.int32)
us = static_relaxation(c, sp, i, j, sh)
print(f"\nstatic relaxation rms displacement = {np.linalg.norm(us, axis=1).std():.4f} A"
      f"  max = {np.linalg.norm(us, axis=1).max():.4f} A")

a1 = warren_cowley(sp, i, j, sh, c, r[0])
print(f"random config WC alpha (should be ~0):\n{np.round(a1, 3)}")
print(f"max |alpha| = {np.abs(a1).max():.3f}")
print("\nOK")
