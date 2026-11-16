"""Embedded-atom-method energies and forces on the GPU, for alloy supercells.

An EAM potential (setfl / LAMMPS `pair_style eam/alloy` file) gives

    E = sum_i F_{s_i}(rho_i) + 1/2 sum_{i != j} phi_{s_i s_j}(r_ij),
    rho_i = sum_{j != i} f_{s_j}(r_ij),

with tabulated embedding functions F, densities f and pair terms r*phi. The
tables are turned into cubic splines once (scipy) and evaluated per pair on
the GPU (CuPy elementwise kernels). Atoms vibrate about lattice sites, so the
neighbour list is built once from the reference lattice with a skin.

What CHIMERA uses this for (`chimera/physics.py`): molecular dynamics at the
measured density and temperature, from which the bond-length offsets, sigma^2
and third cumulants of every pair type and shell follow; relaxed energies of
many arrangements, from which effective pair interactions follow.

Units: Angstrom, eV, amu, fs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

try:
    import cupy as cp
except Exception:                      # pragma: no cover
    cp = None

KB_EV = 8.617333262e-5                 # eV/K
AMU_EV_FS2_A2 = 103.642697             # 1 amu = 103.64 eV fs^2 / A^2


def _read_numbers(lines):
    for ln in lines:
        for tok in ln.split():
            yield float(tok)


@dataclass
class Spline:
    """Cubic spline on a uniform grid x0 + i dx, as coefficients per interval."""
    x0: float
    dx: float
    coef: np.ndarray            # (n-1, 4): c0 + c1 t + c2 t^2 + c3 t^3, t = x - x_i

    @classmethod
    def fit(cls, x0, dx, y):
        x = x0 + dx * np.arange(len(y))
        cs = CubicSpline(x, y, bc_type="not-a-knot")
        c = cs.c                                  # (4, n-1), highest power first
        return cls(x0=float(x0), dx=float(dx), coef=np.stack([c[3], c[2], c[1], c[0]], axis=1))

    def __call__(self, x, deriv=False):
        x = np.asarray(x, float)
        n = self.coef.shape[0]
        i = np.clip(((x - self.x0) / self.dx).astype(int), 0, n - 1)
        t = x - (self.x0 + i * self.dx)
        c = self.coef[i]
        v = ((c[..., 3] * t + c[..., 2]) * t + c[..., 1]) * t + c[..., 0]
        if not deriv:
            return v
        return v, (3 * c[..., 3] * t + 2 * c[..., 2]) * t + c[..., 1]


class EAMPotential:
    """A setfl (eam/alloy) potential."""

    def __init__(self, path):
        lines = open(path).read().split("\n")
        hdr = lines[3].split()
        self.nel = int(hdr[0])
        self.elements = hdr[1:1 + self.nel]
        nrho, drho, nr, dr, cut = lines[4].split()[:5]
        self.nrho, self.drho, self.nr, self.dr = int(nrho), float(drho), int(nr), float(dr)
        self.cutoff = float(cut)
        rest = lines[5:]
        self.mass, self.Z = [], []
        F, f = [], []
        pos = 0

        def table(n):
            """n numbers read line by line; what is left on the last line is
            dropped, as LAMMPS does (some writers add one point per table)."""
            nonlocal pos
            vals = []
            while len(vals) < n:
                vals += [float(t) for t in rest[pos].split()]
                pos += 1
            return np.array(vals[:n])

        for e in range(self.nel):
            h = rest[pos].split()
            self.Z.append(int(h[0]))
            self.mass.append(float(h[1]))
            pos += 1
            F.append(table(self.nrho))
            f.append(table(self.nr))
        rphi = {}
        for i in range(self.nel):
            for j in range(i + 1):
                rphi[(i, j)] = rphi[(j, i)] = table(self.nr)
        self._raw = dict(F=F, f=f, rphi={k: v for k, v in rphi.items() if k[0] >= k[1]})
        self.F = [Spline.fit(0.0, self.drho, y) for y in F]
        self.f = [Spline.fit(0.0, self.dr, y) for y in f]
        self.rphi = {key: Spline.fit(0.0, self.dr, y) for key, y in rphi.items()}
        self.path = path

    def index(self, symbols):
        return np.array([self.elements.index(s) for s in symbols], np.int32)

    def write_clean(self, path):
        """The same tables with exactly nrho/nr values each (for readers that
        take the numbers as one flat list, such as ASE's)."""
        with open(path, "w") as fh:
            fh.write("# clean copy\n#\n#\n")
            fh.write(f"{self.nel} " + " ".join(self.elements) + "\n")
            fh.write(f"{self.nrho} {self.drho:.16e} {self.nr} {self.dr:.16e} {self.cutoff:.16e}\n")

            def arr(a):
                for i in range(0, len(a), 5):
                    fh.write(" ".join(f"{x:.16e}" for x in a[i:i + 5]) + "\n")
            for e in range(self.nel):
                fh.write(f"{self.Z[e]} {self.mass[e]} 0 dummy\n")
                arr(self._raw["F"][e])
                arr(self._raw["f"][e])
            for i in range(self.nel):
                for j in range(i + 1):
                    arr(self._raw["rphi"][(i, j)])

    # stacked coefficient tables for the GPU
    def device_tables(self):
        n = self.nel
        Fc = np.stack([s.coef for s in self.F])                       # (nel, nseg, 4)
        fc = np.stack([s.coef for s in self.f])                       # (nel, nseg, 4)
        pc = np.stack([self.rphi[(a, b)].coef for a in range(n) for b in range(n)])   # (nel^2, nseg, 4)
        return cp.asarray(Fc), cp.asarray(fc), cp.asarray(pc)


_EVAL = r"""
extern "C" __device__ void spl(const double* c, int nseg, double x0, double dx, double x,
                               double* v, double* d) {
    int i = (int)floor((x - x0) / dx);
    if (i < 0) i = 0;
    if (i > nseg - 1) i = nseg - 1;
    double t = x - (x0 + i * dx);
    const double* q = c + 4 * (long)i;
    *v = ((q[3] * t + q[2]) * t + q[1]) * t + q[0];
    *d = (3.0 * q[3] * t + 2.0 * q[2]) * t + q[1];
}

// pass 1: per directed pair, distance, unit vector and density contribution
extern "C" __global__ void eam_rho(
    const double* __restrict__ pos, const int* __restrict__ sp, const int* __restrict__ pi,
    const int* __restrict__ pj, const double* __restrict__ shift, const double* __restrict__ fc,
    int nseg, double dr, double cut, int npair, int natoms, int nconf,
    double* __restrict__ rr, double* __restrict__ uu, double* __restrict__ rho)
{
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)npair * nconf) return;
    int c = (int)(t / npair), p = (int)(t % npair);
    int i = pi[p], j = pj[p];
    const double* P = pos + (long)c * natoms * 3;
    double dx = P[3*j] + shift[3*p] - P[3*i];
    double dy = P[3*j+1] + shift[3*p+1] - P[3*i+1];
    double dz = P[3*j+2] + shift[3*p+2] - P[3*i+2];
    double r = sqrt(dx*dx + dy*dy + dz*dz);
    rr[t] = r;
    uu[3*t] = dx / r; uu[3*t+1] = dy / r; uu[3*t+2] = dz / r;
    if (r >= cut) return;
    int sj = sp[(long)c * natoms + j];
    double v, d;
    spl(fc + (long)sj * nseg * 4, nseg, 0.0, dr, r, &v, &d);
    atomicAdd(&rho[(long)c * natoms + i], v);
}

// pass 2: per directed pair, the scalar dE/dr and the pair energy; forces on i
extern "C" __global__ void eam_force(
    const int* __restrict__ sp, const int* __restrict__ pi, const int* __restrict__ pj,
    const double* __restrict__ fc, const double* __restrict__ pc, const double* __restrict__ dF,
    const double* __restrict__ rr, const double* __restrict__ uu,
    int nseg, double dr, double cut, int nel, int npair, int natoms, int nconf,
    double* __restrict__ frc, double* __restrict__ epair)
{
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)npair * nconf) return;
    int c = (int)(t / npair), p = (int)(t % npair);
    double r = rr[t];
    if (r >= cut) return;
    int i = pi[p], j = pj[p];
    int si = sp[(long)c * natoms + i], sj = sp[(long)c * natoms + j];
    double fi, dfi, fj, dfj, rp, drp;
    spl(fc + (long)si * nseg * 4, nseg, 0.0, dr, r, &fi, &dfi);
    spl(fc + (long)sj * nseg * 4, nseg, 0.0, dr, r, &fj, &dfj);
    spl(pc + (long)(si * nel + sj) * nseg * 4, nseg, 0.0, dr, r, &rp, &drp);
    double phi = rp / r, dphi = (drp - phi) / r;
    double g = dphi + dF[(long)c * natoms + i] * dfj + dF[(long)c * natoms + j] * dfi;
    double* Fi = frc + ((long)c * natoms + i) * 3;
    atomicAdd(&Fi[0], g * uu[3*t]);
    atomicAdd(&Fi[1], g * uu[3*t+1]);
    atomicAdd(&Fi[2], g * uu[3*t+2]);
    atomicAdd(&epair[c], 0.5 * phi);
}
"""
_MOD = None


def _mod():
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_EVAL, options=("--std=c++14",))
    return _MOD


class EAMCell:
    """A periodic supercell of sites with species; energies and forces of
    `nconf` configurations at once (same sites and neighbour list)."""

    def __init__(self, pot: EAMPotential, cell_matrix, sites, skin=0.5):
        self.pot = pot
        self.H = np.asarray(cell_matrix, float)            # rows are lattice vectors (A)
        self.sites = np.asarray(sites, float)
        self.natoms = len(self.sites)
        cut = pot.cutoff + skin
        Hinv = np.linalg.inv(self.H)
        # images needed per direction
        nimg = [int(math.ceil(cut / (1.0 / np.linalg.norm(Hinv[:, d])))) for d in range(3)]
        rng = [range(-n, n + 1) for n in nimg]
        pi, pj, sh = [], [], []
        for a in rng[0]:
            for b in rng[1]:
                for c in rng[2]:
                    s = a * self.H[0] + b * self.H[1] + c * self.H[2]
                    d = self.sites[None, :, :] + s[None, None, :] - self.sites[:, None, :]
                    r = np.linalg.norm(d, axis=-1)
                    ii, jj = np.where((r < cut) & (r > 1e-8))
                    pi.append(ii); pj.append(jj); sh.append(np.repeat(s[None], ii.size, 0))
        self.pi = np.concatenate(pi).astype(np.int32)
        self.pj = np.concatenate(pj).astype(np.int32)
        self.shift = np.concatenate(sh)
        self.npair = self.pi.size
        self.r0 = np.linalg.norm(self.sites[self.pj] + self.shift - self.sites[self.pi], axis=1)
        self.d_pi, self.d_pj = cp.asarray(self.pi), cp.asarray(self.pj)
        self.d_shift = cp.asarray(self.shift)
        self.Fc, self.fc, self.pc = pot.device_tables()
        self.nseg_r = self.fc.shape[1]
        self.nseg_rho = self.Fc.shape[1]
        self.Fc_list = [cp.asarray(s.coef) for s in pot.F]

    def shells(self, tol=0.05):
        """Shell index (0-based) of every pair from the reference distances."""
        u = []
        for r in np.sort(np.unique(np.round(self.r0, 4))):
            if not u or r - u[-1] > tol:
                u.append(r)
        u = np.array(u)
        return np.argmin(np.abs(self.r0[:, None] - u[None, :]), axis=1), u

    def energy_forces(self, pos, species):
        """pos (nconf, natoms, 3) device float64, species (nconf, natoms) device int32.
        Returns (E per configuration (nconf,), forces (nconf, natoms, 3))."""
        nconf = pos.shape[0]
        n, npair = self.natoms, self.npair
        tot = npair * nconf
        rr = cp.empty(tot, cp.float64)
        uu = cp.empty(3 * tot, cp.float64)
        rho = cp.zeros(nconf * n, cp.float64)
        grid = ((tot + 255) // 256,)
        m = _mod()
        m.get_function("eam_rho")(grid, (256,), (
            pos, species, self.d_pi, self.d_pj, self.d_shift, self.fc,
            np.int32(self.nseg_r), np.float64(self.pot.dr), np.float64(self.pot.cutoff),
            np.int32(npair), np.int32(n), np.int32(nconf), rr, uu, rho))
        # embedding energy and its derivative per atom
        sp = species.ravel()
        Fv = cp.zeros_like(rho)
        dF = cp.zeros_like(rho)
        for e, coef in enumerate(self.Fc_list):
            msk = sp == e
            x = rho[msk]
            i = cp.clip((x / self.pot.drho).astype(cp.int32), 0, coef.shape[0] - 1)
            t = x - i * self.pot.drho
            c = coef[i]
            Fv[msk] = ((c[:, 3] * t + c[:, 2]) * t + c[:, 1]) * t + c[:, 0]
            dF[msk] = (3 * c[:, 3] * t + 2 * c[:, 2]) * t + c[:, 1]
        frc = cp.zeros(nconf * n * 3, cp.float64)
        ep = cp.zeros(nconf, cp.float64)
        m.get_function("eam_force")(grid, (256,), (
            species, self.d_pi, self.d_pj, self.fc, self.pc, dF, rr, uu,
            np.int32(self.nseg_r), np.float64(self.pot.dr), np.float64(self.pot.cutoff),
            np.int32(self.pot.nel), np.int32(npair), np.int32(n), np.int32(nconf), frc, ep))
        E = Fv.reshape(nconf, n).sum(1) + ep
        return E, frc.reshape(nconf, n, 3)

    def pair_distances(self, pos):
        """(nconf, npair) distances for the current positions."""
        P = pos
        d = P[:, self.d_pj] + self.d_shift[None] - P[:, self.d_pi]
        return cp.sqrt((d * d).sum(-1))


# ----------------------------------------------------------------- dynamics
def relax_fire(cell: EAMCell, pos, species, fmax=1e-3, nmax=3000, dt0=1.0, dtmax=5.0):
    """FIRE minimisation of every configuration at fixed cell; returns (pos, E)."""
    nconf = pos.shape[0]
    masses = cp.asarray(np.array(cell.pot.mass))[species]                  # (nconf, natoms)
    v = cp.zeros_like(pos)
    dt = cp.full(nconf, dt0)
    alpha = cp.full(nconf, 0.1)
    npos = cp.zeros(nconf, cp.int32)
    E, F = cell.energy_forces(pos, species)
    for it in range(nmax):
        fm = cp.sqrt((F * F).sum(-1)).max(axis=1)
        if bool((fm < fmax).all()):
            break
        P = (F * v).sum(axis=(1, 2))
        vn = cp.sqrt((v * v).sum(axis=(1, 2)))
        fn = cp.sqrt((F * F).sum(axis=(1, 2)))
        mix = (alpha * vn / cp.maximum(fn, 1e-30))[:, None, None]
        v = (1 - alpha)[:, None, None] * v + mix * F
        up = P > 0
        npos = cp.where(up, npos + 1, 0)
        grow = up & (npos > 5)
        dt = cp.where(grow, cp.minimum(dt * 1.1, dtmax), cp.where(up, dt, dt * 0.5))
        alpha = cp.where(grow, alpha * 0.99, cp.where(up, alpha, 0.1))
        v = cp.where(up[:, None, None], v, 0.0)
        v = v + dt[:, None, None] * F / (masses[..., None] * AMU_EV_FS2_A2)
        pos = pos + dt[:, None, None] * v
        E, F = cell.energy_forces(pos, species)
    return pos, E


def langevin(cell: EAMCell, pos, species, T, nsteps, dt=2.0, gamma=0.01, seed=0,
             every=10, callback=None):
    """BAOAB Langevin dynamics at temperature T (K); calls callback(step, pos)
    every `every` steps. Returns the final positions and velocities."""
    rs = cp.random.default_rng(seed)
    m = cp.asarray(np.array(cell.pot.mass))[species][..., None] * AMU_EV_FS2_A2   # eV fs^2/A^2
    kT = KB_EV * T
    v = rs.standard_normal(pos.shape) * cp.sqrt(kT / m)
    c1 = math.exp(-gamma * dt)
    c2 = math.sqrt(1 - c1 * c1)
    _, F = cell.energy_forces(pos, species)
    for s in range(nsteps):
        v = v + 0.5 * dt * F / m
        pos = pos + 0.5 * dt * v
        v = c1 * v + c2 * cp.sqrt(kT / m) * rs.standard_normal(pos.shape)
        pos = pos + 0.5 * dt * v
        _, F = cell.energy_forces(pos, species)
        v = v + 0.5 * dt * F / m
        if callback is not None and (s + 1) % every == 0:
            callback(s + 1, pos)
    return pos, v
