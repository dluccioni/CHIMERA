"""Replica exchange must satisfy detailed balance.

For two rungs holding energies E_cold and E_hot, the swap is accepted with
probability min(1, exp[(beta_cold - beta_hot)(E_cold - E_hot)]): a cold rung
holding the WORSE state always swaps, a cold rung holding the better state
swaps with the Boltzmann factor. An earlier version had the energies the
other way round and sent good states up the ladder to be melted; this checks
both directions of the rule on a tiny sampler with the chi^2 set by hand.

Needs the GPU (the sampler allocates device state), but runs in seconds.
"""
import sys, os, warnings
import numpy as np
import cupy as cp
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exafs_gpu.scattering import default_tables, NSP
from exafs_gpu.lattice import CrystalSupercell, force_constant_matrix
from exafs_gpu.forward import ForwardModel, thermal_sigma2
from exafs_gpu.fourier import FourierTransform
from methods.method_a_rmc import RMCSampler

k = np.linspace(1.5, 16.0, 200)
tab = default_tables(k, "analytic")
cell = CrystalSupercell("fcc", (3, 3, 3), a0=3.56)
i_idx, j_idx, shift, _ = cell.neighbour_list(5.2)
Phi = force_constant_matrix(cell, i_idx, j_idx, shift)
fm = ForwardModel(cell, tab, i_idx, j_idx, shift,
                  sigma2=thermal_sigma2(Phi, cell, i_idx, j_idx, shift))
ft = FourierTransform(k, 3.0, 14.0, 2, xp=cp)
n = cell.natoms
sp = np.repeat(np.arange(NSP), [n // 3, n // 3, n - 2 * (n // 3)]).astype(np.int32)
target = cp.asnumpy(ft.magnitude(fm.chi(cp.asarray(cell.ideal[None]), cp.asarray(sp[None]))))[0]
mask = ft.window_mask(1.2, 5.5)
sigR = np.full_like(target, 0.02 * target.max())

smp = RMCSampler(fm, ft, target, mask, sigR, nrep=2, tmin=0.5, tmax=2.0, seed=3)
smp.init_state(sp, cell.ideal)
beta = cp.asnumpy(smp.beta)
assert beta[0] > beta[1], "rung 0 must be the cold one"
tag = cp.arange(2)                     # follows the states through swaps


def trial(e_cold, e_hot, nattempt):
    """Acceptance over `nattempt` ATTEMPTED exchanges (the alternating scheme
    attempts nothing on every other call when there is only one pair)."""
    smp.exch_attempts[:] = 0; smp.exch_accepted[:] = 0
    while smp.exch_attempts[0] < nattempt:
        smp.chi2 = cp.asarray([e_cold, e_hot])
        smp.replica_exchange()
    return float(smp.exch_accepted[0] / smp.exch_attempts[0])


print("[1] cold rung holds the worse state -> always swap")
p = trial(1.00, 0.90, 50)
print(f"    acceptance {p:.2f}")
assert p == 1.0

print("[2] cold rung holds the better state -> Boltzmann factor")
d = 0.30 / ((beta[0] - beta[1]) * smp.nfit)      # exp(-0.30) = 0.741 expected
p = trial(0.90, 0.90 + d, 4000)
print(f"    acceptance {p:.3f}, expected {np.exp(-0.30):.3f}")
assert abs(p - np.exp(-0.30)) < 0.03

print("[3] a large gap the wrong way -> essentially never")
p = trial(0.90, 0.90 + 20 * d, 200)
print(f"    acceptance {p:.3f}")
assert p < 0.02
print("\nOK")
