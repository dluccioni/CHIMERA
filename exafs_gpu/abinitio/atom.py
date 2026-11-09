"""Self-consistent spherical atom in the local density approximation.

This is the first step of the scattering calculation: the free-atom charge
density and Coulomb potential that are superposed into the muffin-tin
potentials of a cluster (see `potential.py`). FEFF uses a relativistic
Dirac-Fock atom for the same purpose; this module solves the
non-relativistic Kohn-Sham equations with the Vosko-Wilk-Nusair (VWN5)
correlation, which reproduces the NIST LDA reference atoms to the
micro-Hartree (`tests/test_abinitio_atom.py`). For the 3d metals the missing
relativistic contraction of the 1s/2s shells changes the potential only
inside ~0.05 bohr, where it has no measurable effect on the scattering phase
shifts at EXAFS energies; for 5d elements and heavier it would matter, and a
warning is printed.

Units are Hartree atomic units throughout (bohr, Hartree).

Method
------
Radial equation  -u''/2 + [l(l+1)/(2r^2) + V(r)] u = eps u  on a logarithmic
grid r = exp(x). With u = r^(1/2) y it becomes y'' = [(l+1/2)^2 +
2 r^2 (V - eps)] y, which has no first derivative and is integrated with
Numerov's method: outward from the origin to the classical turning point and
inward from the decaying tail, matched there. The eigenvalue is bracketed by
node counting and refined with the matching correction
    d eps = - y_c K / (2 h^2 sum r^2 y^2),
K being the Numerov residual of the kink at the matching point. The loops
are plain Python on scalars, which for ~2300 grid points is faster than
numpy; a full self-consistent 3d-metal atom takes a few seconds and is
cached on disk.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_simpson, simpson

from ..scattering import SYMBOLS, Z_TABLE

# ---------------------------------------------------------------------------
# electron configurations
# ---------------------------------------------------------------------------

_MADELUNG = ["1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p", "5s", "4d", "5p",
             "6s", "4f", "5d", "6p", "7s", "5f", "6d", "7p"]
_CAP = {"s": 2, "p": 6, "d": 10, "f": 14}
_LQN = {"s": 0, "p": 1, "d": 2, "f": 3}
_LETTER = "spdf"

# neutral ground states that break the Madelung order (NIST)
_EXCEPTIONS = {
    24: {"3d": 5, "4s": 1}, 29: {"3d": 10, "4s": 1},
    41: {"4d": 4, "5s": 1}, 42: {"4d": 5, "5s": 1}, 44: {"4d": 7, "5s": 1},
    45: {"4d": 8, "5s": 1}, 46: {"4d": 10, "5s": 0}, 47: {"4d": 10, "5s": 1},
    57: {"4f": 0, "5d": 1, "6s": 2}, 58: {"4f": 1, "5d": 1, "6s": 2},
    64: {"4f": 7, "5d": 1, "6s": 2}, 78: {"4f": 14, "5d": 9, "6s": 1},
    79: {"4f": 14, "5d": 10, "6s": 1},
}


def ground_configuration(Z: int) -> dict[str, float]:
    """Occupations {'1s': 2, ...} of the neutral ground state."""
    if not 1 <= Z <= 86:
        raise ValueError(f"Z={Z} outside the supported range 1..86")
    occ, left = {}, Z
    for orb in _MADELUNG:
        if left <= 0:
            break
        n = min(_CAP[orb[1]], left)
        occ[orb] = float(n)
        left -= n
    if Z in _EXCEPTIONS:
        occ.update({k: float(v) for k, v in _EXCEPTIONS[Z].items()})
        # re-balance the total by the shell just before the changed ones
        tot = sum(occ.values())
        assert abs(tot - Z) < 1e-9, (Z, occ)
    return {k: v for k, v in occ.items() if v > 0}


def core_hole_configuration(Z: int, hole: str = "1s",
                            screening: str | None = None) -> dict[str, float]:
    """Final-state configuration of the absorber: one electron out of `hole`,
    with the core hole fully screened (the final state rule) so the atom stays
    neutral. The screening electron goes into the open d (or f) shell when
    there is one - in a transition metal the d states at the Fermi level do
    the screening - else into the open valence shell, else the next empty
    one: Cr* = 1s1 ... 3d6 4s1, Ni* = 1s1 ... 3d9 4s2, Cu* = 1s1 ... 3d10 4s2.
    The alternative Z+1 rule (valence of the next element) swings with the
    s-d configuration of that element (Mn 4s2, Cu 3d10 4s1) and changes the
    charge near the absorber by -0.2 to +0.3 electrons; this rule gives a
    uniform +0.06-0.09, like FEFF's self-consistent absorbers. `screening`
    names the shell explicitly."""
    occ = ground_configuration(Z)
    if occ.get(hole, 0.0) < 1.0:
        raise ValueError(f"no electron in {hole} for Z={Z}")
    occ[hole] -= 1.0
    shell = screening or _screening_shell(occ, skip=hole)
    occ[shell] = occ.get(shell, 0.0) + 1.0
    if occ[shell] > _CAP[shell[-1]]:
        raise ValueError(f"{shell} is full")
    return {k: v for k, v in occ.items() if v > 0}


def _screening_shell(occ, skip="1s"):
    """Open d/f shell, else the outermost open shell, else the lowest empty
    one in Madelung order (Zn -> 4p, Pd 4d10 -> 5s)."""
    occupied = [o for o in _MADELUNG if occ.get(o, 0.0) > 0]
    last = max(_MADELUNG.index(o) for o in occupied)
    for o in reversed(_MADELUNG[:last + 1]):
        if o[1] in "df" and 0.0 < occ.get(o, 0.0) < _CAP[o[1]]:
            return o
    for o in reversed(_MADELUNG[:last + 1]):
        if o == skip:
            continue
        if 0.0 < occ.get(o, 0.0) < _CAP[o[1]]:
            return o
    for o in _MADELUNG:
        if o != skip and occ.get(o, 0.0) < _CAP[o[1]]:
            return o
    raise ValueError("no shell for the screening electron")


def parse_orbital(name):
    return int(name[:-1]), _LQN[name[-1]]


# ---------------------------------------------------------------------------
# exchange-correlation (spin-unpolarised LDA)
# ---------------------------------------------------------------------------

_VWN_A, _VWN_X0, _VWN_B, _VWN_C = 0.0310907, -0.10498, 3.72744, 12.9352
_CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)        # eps_x = -CX n^(1/3)


def rs_of(n):
    n = np.maximum(np.asarray(n, float), 1e-30)
    return (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)


def lda_xc(n):
    """(eps_xc, v_xc) per electron, Hartree: Slater exchange + VWN5 correlation."""
    n = np.maximum(np.asarray(n, float), 1e-30)
    ex = -_CX * n ** (1.0 / 3.0)
    vx = 4.0 / 3.0 * ex
    rs = rs_of(n)
    x = np.sqrt(rs)
    A, x0, b, c = _VWN_A, _VWN_X0, _VWN_B, _VWN_C
    X = x * x + b * x + c
    X0 = x0 * x0 + b * x0 + c
    Q = math.sqrt(4.0 * c - b * b)
    at = np.arctan(Q / (2.0 * x + b))
    ec = A * (np.log(x * x / X) + 2.0 * b / Q * at
              - b * x0 / X0 * (np.log((x - x0) ** 2 / X) + 2.0 * (b + 2.0 * x0) / Q * at))
    den = Q * Q + (2.0 * x + b) ** 2
    dec = A * (2.0 / x - (2.0 * x + b) / X - 4.0 * b / den
               - b * x0 / X0 * (2.0 / (x - x0) - (2.0 * x + b) / X
                                 - 4.0 * (b + 2.0 * x0) / den))
    vc = ec - x / 6.0 * dec
    return ex + ec, vx + vc


# ---------------------------------------------------------------------------
# radial grid and Numerov integration
# ---------------------------------------------------------------------------

@dataclass
class LogGrid:
    """r_i = exp(x0 + i h)."""
    rmin: float = 1e-6
    rmax: float = 100.0
    h: float = 0.008

    def __post_init__(self):
        n = int(math.ceil(math.log(self.rmax / self.rmin) / self.h)) + 1
        self.x = math.log(self.rmin) + self.h * np.arange(n)
        self.r = np.exp(self.x)
        self.n = n

    def integrate(self, f):
        """int f dr over the grid (f sampled on r)."""
        return simpson(f * self.r, dx=self.h)

    def cumulative(self, f):
        """int_0^r f dr' at every grid point."""
        out = cumulative_simpson(f * self.r, dx=self.h, initial=0.0)
        return out


def _numerov_out(G, h2, y0, y1, iend):
    """Outward Numerov for y'' = G y up to index iend (inclusive). Returns the
    list y[0..iend] and the number of sign changes."""
    f = [1.0 - h2 * g / 12.0 for g in G[:iend + 1]]
    y = [0.0] * (iend + 1)
    y[0], y[1] = y0, y1
    nodes = 0
    for i in range(1, iend):
        yn = ((12.0 - 10.0 * f[i]) * y[i] - f[i - 1] * y[i - 1]) / f[i + 1]
        y[i + 1] = yn
        if (yn < 0.0) != (y[i] < 0.0) and y[i] != 0.0:
            nodes += 1
    return y, nodes


def _numerov_in(G, h2, istart, iend):
    """Inward Numerov from istart (y=0 beyond) down to iend (inclusive).
    Returns a dict-like list indexed from iend to istart."""
    f = [1.0 - h2 * g / 12.0 for g in G[iend:istart + 1]]
    m = istart - iend
    y = [0.0] * (m + 1)
    y[m] = 0.0
    y[m - 1] = 1e-200
    for j in range(m - 1, 0, -1):
        y[j - 1] = ((12.0 - 10.0 * f[j]) * y[j] - f[j + 1] * y[j + 1]) / f[j - 1]
        if abs(y[j - 1]) > 1e200:          # rescale the tail to stay finite
            s = 1e-200
            for q in range(j - 1, m + 1):
                y[q] *= s
    return y


def solve_bound(V, grid: LogGrid, l: int, nodes: int, Z: float,
                eps_guess: float | None = None, tol: float = 1e-11):
    """One bound eigenstate of the radial equation in potential V(r).

    Returns (eps, y) with y normalised so that int u^2 dr = 1 (u = r^0.5 y).
    """
    r = grid.r
    h2 = grid.h ** 2
    lam = (l + 0.5) ** 2
    Vr2 = 2.0 * r * r
    Vl = V.tolist() if isinstance(V, np.ndarray) else list(V)
    rl = r.tolist()
    n = grid.n
    lo = -0.6 * Z * Z - 10.0
    hi = 0.0
    eps = eps_guess if eps_guess is not None and lo < eps_guess < hi else -0.5 * (Z / (nodes + l + 1)) ** 2
    y0 = rl[0] ** (l + 0.5) * (1.0 - Z * rl[0] / (l + 1.0))
    y1 = rl[1] ** (l + 0.5) * (1.0 - Z * rl[1] / (l + 1.0))
    for _ in range(200):
        G = [lam + 2.0 * rr * rr * (v - eps) for rr, v in zip(rl, Vl)]
        # classical turning point: last index with G < 0
        ic = None
        for i in range(n - 1, 0, -1):
            if G[i] < 0.0:
                ic = i
                break
        if ic is None or ic < 20:
            ic = max(20, n // 3)
        ic = min(ic, n - 30)
        yout, nd = _numerov_out(G, h2, y0, y1, ic + 1)
        if nd > nodes:
            hi = eps
            eps = 0.5 * (lo + hi)
            continue
        if nd < nodes:
            lo = eps
            eps = 0.5 * (lo + hi)
            continue
        kappa = math.sqrt(max(-2.0 * eps, 1e-8))
        rinf = rl[ic] + 60.0 / kappa
        istart = min(n - 1, int(np.searchsorted(r, rinf)))
        istart = max(istart, ic + 10)
        yin = _numerov_in(G, h2, istart, ic - 1)       # indices ic-1 .. istart
        sc = yout[ic] / yin[1]
        # matched solution
        ym = yout[:ic + 1] + [v * sc for v in yin[2:]]
        f = lambda i: 1.0 - h2 * G[i] / 12.0
        K = f(ic - 1) * yout[ic - 1] + f(ic + 1) * yin[2] * sc - (12.0 - 10.0 * f(ic)) * yout[ic]
        norm = 0.0
        for i in range(len(ym)):
            norm += rl[i] * rl[i] * ym[i] * ym[i]
        deps = -yout[ic] * K / (2.0 * h2 * norm)
        if deps > 0:
            lo = max(lo, eps)
        else:
            hi = min(hi, eps)
        new = eps + deps
        if not (lo < new < hi):
            new = 0.5 * (lo + hi)
        if abs(new - eps) < tol * max(1.0, abs(eps)):
            eps = new
            break
        eps = new
    else:
        raise RuntimeError(f"bound state l={l} nodes={nodes} did not converge")
    y = np.zeros(n)
    y[:len(ym)] = ym
    nrm = grid.integrate(r * y * y)          # int u^2 dr = int r y^2 dr
    y /= math.sqrt(nrm)
    if y[ic] < 0:
        y = -y
    return eps, y


# ---------------------------------------------------------------------------
# the self-consistent atom
# ---------------------------------------------------------------------------

def thomas_fermi_potential(Z, r):
    """Starting potential: Thomas-Fermi screening (Latter's fit), with the
    -1/r tail a neutral atom shows an electron."""
    b = 0.8853 * Z ** (-1.0 / 3.0)
    x = r / b
    phi = 1.0 / (1.0 + 0.02747 * np.sqrt(x) + 1.243 * x - 0.1486 * x ** 1.5
                 + 0.2302 * x ** 2 + 0.007298 * x ** 2.5 + 0.006944 * x ** 3)
    return np.minimum(-Z * phi / r, -1.0 / r)


@dataclass
class Atom:
    Z: int
    occupations: dict
    grid: LogGrid
    eps: dict = field(default_factory=dict)       # orbital -> eigenvalue (Ha)
    orbitals: dict = field(default_factory=dict)  # orbital -> y(r)
    rho: np.ndarray | None = None                 # electron density n(r)
    v_coulomb: np.ndarray | None = None           # -Z/r + V_Hartree
    v_xc: np.ndarray | None = None
    energy: float = 0.0
    iterations: int = 0

    @property
    def r(self):
        return self.grid.r

    def orbital_density(self, name):
        y = self.orbitals[name]
        return y * y / (4.0 * np.pi * self.r)

    def charge_within(self, R):
        q = self.grid.cumulative(4.0 * np.pi * self.r ** 2 * self.rho)
        return float(np.interp(R, self.r, q))


def hartree(grid, rho):
    r = grid.r
    q = grid.cumulative(4.0 * np.pi * r * r * rho)            # charge inside r
    outer = grid.cumulative(4.0 * np.pi * r * rho)
    outer = outer[-1] - outer                                  # int_r^inf 4 pi r' rho
    return q / r + outer


def solve_atom(Z: int, occupations: dict | None = None, grid: LogGrid | None = None,
               tol: float = 1e-9, max_iter: int = 300, mix: float = 0.4,
               history: int = 6, verbose: bool = False) -> Atom:
    """Self-consistent LDA atom. `occupations` defaults to the ground state;
    fractional occupations are fine."""
    if Z > 54:
        print(f"warning: Z={Z}: non-relativistic atom, core densities of heavy "
              "elements will be off")
    occ = dict(occupations) if occupations is not None else ground_configuration(Z)
    grid = grid or LogGrid()
    r = grid.r
    nel = sum(occ.values())
    V = thomas_fermi_potential(Z, r)
    eps = {}
    hist_in, hist_res = [], []
    rho = None
    for it in range(1, max_iter + 1):
        orbs = {}
        for name, f in sorted(occ.items(), key=lambda t: parse_orbital(t[0])):
            nq, l = parse_orbital(name)
            e, y = solve_bound(V, grid, l, nq - l - 1, Z, eps.get(name))
            eps[name] = e
            orbs[name] = y
        rho = np.zeros_like(r)
        for name, f in occ.items():
            rho += f * orbs[name] ** 2 / (4.0 * np.pi * r)
        vh = hartree(grid, rho)
        exc, vxc = lda_xc(rho)
        Vout = -Z / r + vh + vxc
        res = Vout - V
        # convergence in r*V (the Coulomb tail would otherwise dominate)
        err = float(np.max(np.abs(res * r) / np.maximum(Z, 1)))
        if verbose:
            print(f"  it {it:3d}  max|r dV|/Z = {err:.2e}  " +
                  " ".join(f"{k}={v:.6f}" for k, v in eps.items()))
        if err < tol:
            break
        # Anderson (Pulay) mixing of the potential
        hist_in.append(V.copy())
        hist_res.append(res.copy())
        if len(hist_in) > history:
            hist_in.pop(0)
            hist_res.pop(0)
        m = len(hist_in)
        if m >= 2:
            w = r ** 2           # weight the residual where the electrons are
            dR = [hist_res[i] - hist_res[-1] for i in range(m - 1)]
            A = np.array([[np.sum(w * dR[i] * dR[j]) for j in range(m - 1)] for i in range(m - 1)])
            bvec = np.array([np.sum(w * dR[i] * hist_res[-1]) for i in range(m - 1)])
            try:
                c = np.linalg.solve(A + 1e-12 * np.trace(A) * np.eye(m - 1), -bvec)
            except np.linalg.LinAlgError:
                c = np.zeros(m - 1)
            Vbar = hist_in[-1] + sum(c[i] * (hist_in[i] - hist_in[-1]) for i in range(m - 1))
            Rbar = hist_res[-1] + sum(c[i] * dR[i] for i in range(m - 1))
            V = Vbar + mix * Rbar
        else:
            V = V + mix * res
    else:
        raise RuntimeError(f"atom Z={Z} not converged after {max_iter} iterations (err {err:.1e})")
    atom = Atom(Z=Z, occupations=occ, grid=grid, eps=eps, orbitals=orbs, rho=rho,
                v_coulomb=-Z / r + vh, v_xc=vxc, iterations=it)
    # total energy: sum f eps - 1/2 int rho V_H + int rho (eps_xc - v_xc)
    band = sum(occ[k] * eps[k] for k in occ)
    eh = 0.5 * grid.integrate(4.0 * np.pi * r * r * rho * vh)
    exc_term = grid.integrate(4.0 * np.pi * r * r * rho * (exc - vxc))
    atom.energy = band - eh + exc_term
    q = grid.integrate(4.0 * np.pi * r * r * rho)
    if abs(q - nel) > 1e-6 * max(1.0, nel):
        raise RuntimeError(f"charge {q} != {nel}")
    return atom


# ---------------------------------------------------------------------------
# disk cache: atoms are deterministic in (Z, occupations, grid)
# ---------------------------------------------------------------------------

def _cache_dir():
    d = os.environ.get("CHIMERA_CACHE",
                       os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                           os.path.abspath(__file__)))), "data", "_cache"))
    os.makedirs(d, exist_ok=True)
    return d


def cached_atom(Z: int, occupations: dict | None = None, grid: LogGrid | None = None) -> Atom:
    occ = dict(occupations) if occupations is not None else ground_configuration(Z)
    grid = grid or LogGrid()
    key = json.dumps(dict(Z=Z, occ=sorted(occ.items()), g=[grid.rmin, grid.rmax, grid.h],
                          v=3), sort_keys=True)
    tag = hashlib.sha1(key.encode()).hexdigest()[:16]
    fn = os.path.join(_cache_dir(), f"atom_{SYMBOLS[Z - 1]}_{tag}.npz")
    if os.path.exists(fn):
        d = np.load(fn, allow_pickle=False)
        names = [str(s) for s in d["names"]]
        return Atom(Z=Z, occupations=occ, grid=grid,
                    eps={n: float(e) for n, e in zip(names, d["eps"])},
                    orbitals={n: d["orb"][i] for i, n in enumerate(names)},
                    rho=d["rho"], v_coulomb=d["vc"], v_xc=d["vxc"],
                    energy=float(d["energy"]), iterations=int(d["it"]))
    a = solve_atom(Z, occ, grid)
    names = sorted(a.orbitals)
    np.savez(fn, names=np.array(names), eps=np.array([a.eps[n] for n in names]),
             orb=np.array([a.orbitals[n] for n in names]), rho=a.rho,
             vc=a.v_coulomb, vxc=a.v_xc, energy=a.energy, it=a.iterations)
    return a


def atom_for(symbol: str, core_hole: str | None = None) -> Atom:
    """Ground-state atom, or the final-state absorber with `core_hole` ('1s')."""
    Z = Z_TABLE[symbol]
    occ = ground_configuration(Z) if core_hole is None else core_hole_configuration(Z, core_hole)
    return cached_atom(Z, occ)
