"""The FEFF8L alloy tables: complete, continuous, correctly dispatched.

CPU only, a few seconds.
"""
import sys, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exafs_gpu.scattering import (feff8l_tables, read_feffdat, default_tables,
                                  contrast_report, ELEMENTS)

k = np.linspace(1.5, 16.0, 300)
F = feff8l_tables(k)
print(f"route {F.meta['route']}, e0 offset {F.e0_offset_ev:.2f} eV")

print("[1] every (absorber, scatterer) pair, all at the first-shell distance")
for (a, s), m in sorted(F.meta["pairs"].items()):
    assert abs(m["reff"] - 2.517) < 1e-3 and m["deg"] == 4, (a, s, m)
assert len(F.meta["pairs"]) == 9
print("  9 pairs, reff 2.517 A, degeneracy 4")

print("\n[2] tables are smooth in k (no branch jumps in the unwrapped phase)")
dk = k[1] - k[0]
fit = k >= 3.0
slope = np.abs(np.diff(F.delta, axis=-1)) / dk
kmid = 0.5 * (k[1:] + k[:-1])
j = np.unravel_index(np.argmax(slope * (kmid >= 3.0)), slope.shape)
print(f"  max |d delta/dk| over k >= 3: {slope[j]:.2f} rad/A^-1 at k = {kmid[j[-1]]:.2f}")
# FEFF's p(E) has a kink at the plasmon threshold (Re Sigma dips), which on
# the p axis makes the phase steep around p ~ 3; above that it must be smooth.
hi = kmid >= 4.0
print(f"  max |d delta/dk| over k >= 4: {slope[..., hi].max():.2f} rad/A^-1")
assert slope[..., hi].max() < 2.5 and slope[j] < 8.0
assert np.all(F.f[:, fit] > 0) and np.all(F.lam > 3.0)

print("\n[3] contrast")
print("  " + contrast_report(F).replace("\n", "\n  "))
def con(t, a, b):
    num = np.sqrt(np.mean((t.f[a][fit] - t.f[b][fit]) ** 2))
    den = np.sqrt(np.mean(0.5 * (t.f[a][fit] ** 2 + t.f[b][fit] ** 2)))
    return num / den
assert con(F, 1, 2) < 0.12 and con(F, 0, 1) > 2 * con(F, 1, 2)

print("\n[4] default_tables dispatch")
assert default_tables(k, "feff8l").meta["route"] == "FEFF8L"
assert default_tables(k, "analytic").f.shape == (3, k.size)
print("\nOK")
