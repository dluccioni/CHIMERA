"""Measured energy-loss functions and the many-pole self-energy.

The Hedin-Lundqvist self-energy (`selfenergy.py`) lets the photoelectron lose
energy to one plasmon of the free-electron gas: no loss at all below that
plasmon (about 20 eV in these metals), and nothing of the 3p/3s semicore or
the interband excitations of a transition metal. The many-pole self-energy
(Kas et al., Phys. Rev. B 76, 195116 (2007), FEFF9's MPSE) keeps the GW form
but takes the screening from the material's measured loss function,

    -Im 1/eps(q = 0, w) = (pi/2) sum_j g_j w_j delta(w - w_j),

each pole dispersing like the plasmon, w_j(q)^2 = w_j^2 + kF^2 q^2/3 + q^4/4.
The poles come from binning the loss function so that every bin keeps both
sum rules: g_j = (2/pi) int_bin L/w dw (perfect screening: sum g_j = 1) and
g_j w_j^2 = (2/pi) int_bin w L dw (f-sum: sum g_j w_j^2 = 4 pi n_e).

Loss functions, eps = (n + i k)^2 and L = Im(-1/eps):
  * Werner, Glantschnig & Ambrosch-Draxl, J. Phys. Chem. Ref. Data 38, 1013
    (2009): n, k from reflection electron energy-loss spectra, 0.5-70 eV
    (Co, Ni, Cu, Fe, ...; refractiveindex.info compilation, CC0);
  * CXRO / Henke atomic scattering factors f1, f2 to 30 keV (Henke et al.,
    At. Data Nucl. Data Tables 54, 181 (1993); Cr: Delmotte et al., J. Appl.
    Phys. 124, 035107 (2018), from 0.4 eV): n = 1 - delta + i beta,
    delta, beta = r_e lambda^2 n_at f1,f2 / 2 pi.
Werner below 55 eV, CXRO above 70 eV, a linear blend of eps between. An
alloy mixes susceptibilities: eps - 1 = sum_s x_s (n_at / n_s)(eps_s - 1).

The files are downloaded once into data/_external.

The self-energy enters as a correction to the local HL one: at the
interstitial density, Delta Sigma(E) = [Sigma_MP(E) - Sigma_MP(E_F)] -
[Sigma_HL(E) - Sigma_HL(E_F)], added at every point (FEFF9 adds its
many-pole correction the same way). Hartree atomic units inside.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .atom import _cache_dir
from .selfenergy import KF_RS, _GL_W, _GL_X, _bisect

HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903
R_E_CM = 2.8179403262e-13              # classical electron radius (cm)
AVOGADRO = 6.02214076e23
HC_EV_UM = 1.239841984                 # E[eV] * lambda[um]

# solid density (g/cm^3) and atomic mass of the elements with loss data
ELEMENTS = {
    "Ti": (4.506, 47.867, 22), "V": (6.11, 50.942, 23), "Cr": (7.19, 51.996, 24),
    "Mn": (7.21, 54.938, 25), "Fe": (7.874, 55.845, 26), "Co": (8.90, 58.933, 27),
    "Ni": (8.908, 58.693, 28), "Cu": (8.96, 63.546, 29), "Zn": (7.14, 65.38, 30),
    "Mo": (10.28, 95.95, 42), "Pd": (12.02, 106.42, 46), "Ag": (10.49, 107.87, 47),
    "Pt": (21.45, 195.08, 78), "Au": (19.30, 196.97, 79),
}
CXRO_URL = "https://henke.lbl.gov/optical_constants/sf/{}.nff"
WERNER_URL = ("https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/"
              "main/database/data/main/{}/nk/Werner.yml")
BLEND = (55.0, 70.0)                   # eV: Werner below, CXRO above, linear in between
VALENCE_EV = 40.0                      # the valence (plasmon + interband) part of the loss function


def _external_dir():
    d = os.path.join(os.path.dirname(_cache_dir()), "_external", "optical")
    os.makedirs(d, exist_ok=True)
    return d


def _fetch(url, fn):
    path = os.path.join(_external_dir(), fn)
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "chimera-exafs"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
    return path


def atomic_density_cm3(symbol):
    rho, mass, _ = ELEMENTS[symbol]
    return rho * AVOGADRO / mass


def cxro(symbol):
    """(E eV, f1, f2) from the CXRO table; f1 = -9999 marks 'no value'."""
    path = _fetch(CXRO_URL.format(symbol.lower()), f"cxro_{symbol}.nff")
    rows = []
    for ln in open(path, encoding="latin-1"):
        s = ln.split()
        if len(s) < 3 or ln.lstrip().startswith(("#", "E")):
            continue
        try:
            rows.append([float(x) for x in s[:3]])
        except ValueError:
            continue
    a = np.array(rows)
    a = a[np.argsort(a[:, 0], kind="stable")]
    keep = np.concatenate([[True], np.diff(a[:, 0]) > 0])        # duplicate edge energies
    return a[keep, 0], a[keep, 1], a[keep, 2]


def werner(symbol):
    """(E eV, n, k) from Werner et al. 2009 (REELS), or None when not tabulated."""
    try:
        path = _fetch(WERNER_URL.format(symbol), f"werner_{symbol}.yml")
    except Exception:
        return None
    rows, inside = [], False
    for ln in open(path, encoding="utf-8"):
        if ln.strip().startswith("data:"):
            inside = True
            continue
        if inside:
            s = ln.split()
            if len(s) == 3:
                try:
                    rows.append([float(x) for x in s])
                except ValueError:
                    inside = False
            elif s:
                inside = False
    if not rows:
        return None
    a = np.array(rows)
    E = HC_EV_UM / a[:, 0]
    o = np.argsort(E)
    return E[o], a[o, 1], a[o, 2]


def eps_element(symbol, E_ev):
    """Complex dielectric function of the pure element at its solid density."""
    E_ev = np.asarray(E_ev, float)
    n_at = atomic_density_cm3(symbol)
    Ec, f1, f2 = cxro(symbol)
    ok = f1 > -9000
    lam_cm = HC_EV_UM / E_ev * 1e-4
    pref = R_E_CM * lam_cm ** 2 * n_at / (2.0 * np.pi)
    f1i = np.interp(E_ev, Ec[ok], f1[ok], left=np.nan, right=np.nan)
    f2i = np.interp(E_ev, Ec, f2, left=np.nan, right=np.nan)
    eps_x = (1.0 - pref * f1i + 1j * pref * f2i) ** 2
    w = werner(symbol)
    if w is None:
        return eps_x
    Ew, n, k = w
    inr = (E_ev >= Ew[0]) & (E_ev <= Ew[-1])
    eps_w = np.full(E_ev.shape, np.nan + 0j)
    eps_w[inr] = (np.interp(E_ev[inr], Ew, n) + 1j * np.interp(E_ev[inr], Ew, k)) ** 2
    a, b = BLEND
    t = np.clip((E_ev - a) / (b - a), 0.0, 1.0)
    out = np.where(E_ev < a, eps_w, np.where(E_ev > b, eps_x, (1 - t) * eps_w + t * eps_x))
    return out


@dataclass
class LossFunction:
    E_ev: np.ndarray            # photon / loss energy grid (eV)
    eps: np.ndarray             # complex dielectric function
    n_at: float                 # atoms per bohr^3
    n_el: float                 # electrons per atom (composition-weighted Z)
    label: str
    valence_scale: float = 1.0  # factor on L below VALENCE_EV that restores perfect screening

    @property
    def L(self):
        L = np.imag(-1.0 / self.eps)
        if self.valence_scale != 1.0:
            L = np.where(self.E_ev < VALENCE_EV, self.valence_scale * L, L)
        return L

    def sum_rules(self):
        """Electrons per atom counted by the f-sum up to each energy, and the
        perfect-screening integral (2/pi) int L/w dw (1 for a metal)."""
        w = self.E_ev / HARTREE_EV
        L = self.L
        neff = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] * L[1:] + w[:-1] * L[:-1]) * np.diff(w))])
        neff *= 2.0 / (np.pi * 4.0 * np.pi * self.n_at)
        ps = (2.0 / np.pi) * np.trapezoid(L / w, w)
        return neff, ps


def atomic_density_A3(lattice, d_nn, c_over_a=None):
    """Atoms per A^3 of a lattice with nearest-neighbour spacing d_nn (A)."""
    if lattice == "fcc":
        return 4.0 / (d_nn * math.sqrt(2.0)) ** 3
    if lattice == "bcc":
        return 2.0 / (2.0 * d_nn / math.sqrt(3.0)) ** 3
    if lattice == "sc":
        return 1.0 / d_nn ** 3
    if lattice == "hcp":
        ca = math.sqrt(8.0 / 3.0) if c_over_a is None else float(c_over_a)
        return 2.0 / (0.5 * math.sqrt(3.0) * d_nn ** 3 * ca)
    raise ValueError(lattice)


def material_poles(elements, composition, lattice, d_nn, c_over_a=None, n_poles=80):
    """Many-pole representation of the measured loss function of the alloy
    (or element) at the density of its lattice."""
    n_at = atomic_density_A3(lattice, d_nn, c_over_a)
    return poles(loss_function(elements, composition, n_at_A3=n_at), n_poles)


def loss_function(elements, composition=None, n_at_A3=None, n_grid=6000, label=None,
                  perfect_screening=True):
    """Loss function of a pure element or an alloy (susceptibilities mixed).
    n_at_A3: atomic density of the material (atoms/A^3); default the
    composition-weighted pure-element value.

    perfect_screening: a metal screens a static charge completely, so
    (2/pi) int L/w dw = 1 exactly. The measured valence loss functions miss
    that by -6% (Ni) to +11% (Cr, the sputtered-film data of Delmotte et al.);
    the part below VALENCE_EV, which carries ~80% of the integral, is scaled
    to restore it. The f-sum (the electron count) moves by under 0.5 electron."""
    elements = list(elements)
    comp = np.ones(len(elements)) if composition is None else np.asarray(composition, float)
    comp = comp / comp.sum()
    lo = max(0.5, max(min(werner(s)[0][0] if werner(s) else cxro(s)[0][0] for s in elements), 0.5))
    E = np.geomspace(lo, 29000.0, n_grid)
    ns = np.array([atomic_density_cm3(s) for s in elements])          # per cm^3
    n_mat = (1e24 * n_at_A3) if n_at_A3 is not None else float(comp @ ns)
    chi = np.zeros(E.size, complex)
    for s, x, n_s in zip(elements, comp, ns):
        e = eps_element(s, E)
        if np.any(~np.isfinite(e)):
            bad = E[~np.isfinite(e)]
            raise ValueError(f"no optical data for {s} at {bad.min():.2f}-{bad.max():.2f} eV")
        chi += x * (n_mat / n_s) * (e - 1.0)
    n_au = n_mat * (BOHR_A * 1e-8) ** 3
    n_el = float(comp @ np.array([ELEMENTS[s][2] for s in elements]))
    lf = LossFunction(E_ev=E, eps=1.0 + chi, n_at=n_au, n_el=n_el,
                      label=(label or "-".join(elements)) + ("" if perfect_screening else "-raw"))
    if perfect_screening:
        w = E / HARTREE_EV
        L = lf.L
        lo = E < VALENCE_EV
        ps_lo = (2.0 / np.pi) * np.trapezoid(np.where(lo, L / w, 0.0), w)
        ps_hi = (2.0 / np.pi) * np.trapezoid(np.where(lo, 0.0, L / w), w)
        lf.valence_scale = float((1.0 - ps_hi) / ps_lo)
    return lf


@dataclass
class Poles:
    w: np.ndarray        # pole frequencies (Ha)
    g: np.ndarray        # weights: sum g = perfect screening, sum g w^2 = 4 pi n_e
    label: str

    def key(self):
        return hashlib.sha1(np.concatenate([self.w, self.g]).tobytes()).hexdigest()[:12]


def poles(lf: LossFunction, n_poles=80, w_min_ev=None, w_max_ev=None):
    """Bins equally spaced in ln(w); each keeps the f-sum and the
    perfect-screening moment of its part of the loss function."""
    E, L = lf.E_ev, lf.L
    w = E / HARTREE_EV
    edges = np.geomspace(w_min_ev or E[0], w_max_ev or E[-1], n_poles + 1) / HARTREE_EV

    def cum(f):
        return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(w))])
    C1, Cm1 = cum(w * L), cum(L / w)                  # the two moments, cumulative
    m1 = np.diff(np.interp(edges, w, C1))
    m_1 = np.diff(np.interp(edges, w, Cm1))
    ok = (m1 > 0) & (m_1 > 0)
    return Poles(w=np.sqrt(m1[ok] / m_1[ok]), g=2.0 / np.pi * m_1[ok], label=lf.label)


# --------------------------------------------------------------------------
# one dispersing pole of the electron gas (kF, frequency wj), weight 1;
# the pole-strength factor wj^2 of the plasmon-pole formula is included
# --------------------------------------------------------------------------

def _wq(q, w, kf):
    return np.sqrt(w * w + kf * kf * q * q / 3.0 + 0.25 * q ** 4)


def im_sigma_pole(kf, w, k):
    """Im Sigma_c on shell (k > kF) for one pole of frequency w (Hartree);
    zero below the threshold E = (k^2 - kF^2)/2 = w."""
    kf, w, k = (np.asarray(x, float) for x in np.broadcast_arrays(kf, w, k))
    shape = k.shape
    kf, w, k = kf.ravel(), w.ravel(), k.ravel()
    out = np.zeros_like(k)

    def dh(q):
        return k - q - (kf * kf * q / 3.0 + 0.5 * q ** 3) / _wq(q, w, kf)
    qpk = _bisect(lambda q: -dh(q), np.zeros_like(k), 2.0 * k + 1.0)
    hmax = k * qpk - 0.5 * qpk ** 2 - _wq(qpk, w, kf)
    E = 0.5 * (k * k - kf * kf)
    ok = (hmax > 0) & (E > w)
    if not ok.any():
        return out.reshape(shape)
    h = lambda q: k * q - 0.5 * q * q - _wq(q, w, kf)
    q1 = _bisect(h, np.zeros_like(k), qpk)
    q2 = _bisect(lambda q: -h(q), qpk, 2.0 * k + 1.0)
    disc = (kf ** 4 / 9.0) - (w * w - E * E)
    q3 = np.sqrt(np.maximum(2.0 * (-kf * kf / 3.0 + np.sqrt(np.maximum(disc, 0.0))), 0.0))
    lo, hi = q1, np.minimum(q2, q3)
    ok &= hi > lo
    xs = 0.5 * (hi - lo)[:, None] * (_GL_X[None, :] + 1.0) + lo[:, None]
    integ = w[:, None] ** 2 / (2.0 * _wq(xs, w[:, None], kf[:, None]) * k[:, None] * xs)
    val = 0.5 * (hi - lo) * (integ * _GL_W[None, :]).sum(axis=1)
    out[ok] = -val[ok]
    return out.reshape(shape)


def re_sigma_pole(kf, w, k, epsrel=1e-7):
    """Re Sigma_c on shell for one pole (principal value; Hartree)."""
    def integrand(q):
        if q <= 0.0:
            return 0.0
        wq = math.sqrt(w * w + kf * kf * q * q / 3.0 + 0.25 * q ** 4)
        kq = k * q
        muF = (k * k + q * q - kf * kf) / (2.0 * kq)
        ms = min(1.0, max(-1.0, muF))
        t = 0.0
        if ms > -1.0:
            a1 = kq * ms - wq - 0.5 * q * q
            a0 = -kq - wq - 0.5 * q * q
            t += (math.log(abs(a1) + 1e-300) - math.log(abs(a0))) / kq
        if ms < 1.0:
            b1 = kq + wq - 0.5 * q * q
            b0 = kq * ms + wq - 0.5 * q * q
            t += (math.log(abs(b1) + 1e-300) - math.log(abs(b0) + 1e-300)) / kq
        return w * w / (2.0 * wq) * t

    pts = sorted({abs(k - kf), k + kf})
    E = 0.5 * (k * k - kf * kf)
    if E > w:
        disc = kf ** 4 / 9.0 - (w * w - E * E)
        pts.append(math.sqrt(max(2.0 * (-kf * kf / 3.0 + math.sqrt(max(disc, 0.0))), 0.0)))
    qs = np.linspace(1e-6, 4.0 * (k + kf), 2001)
    hh = k * qs - 0.5 * qs ** 2 - np.sqrt(w * w + kf * kf * qs ** 2 / 3.0 + 0.25 * qs ** 4)
    for i in np.where(np.diff(np.sign(hh)) != 0)[0]:
        pts.append(0.5 * (qs[i] + qs[i + 1]))
    qmax = 40.0 * (k + kf) + 40.0 + 4.0 * math.sqrt(w)
    pts = sorted(p for p in pts if 0.0 < p < qmax)
    edges = [0.0] + pts + [qmax]
    tot = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 1e-12:
            continue
        v, _ = quad(integrand, a, b, limit=300, epsabs=1e-10, epsrel=epsrel)
        tot += v
    return tot / np.pi


def sigma_c_poles(P: Poles, kf, k_grid):
    """Sum over poles of g_j Sigma_c^pole(k), complex, on the momentum grid."""
    k_grid = np.asarray(k_grid, float)
    im = np.zeros_like(k_grid)
    re = np.zeros_like(k_grid)
    for wj, gj in zip(P.w, P.g):
        im += gj * im_sigma_pole(kf, wj, k_grid)
        re += gj * np.array([re_sigma_pole(kf, wj, kk) for kk in k_grid])
    return re + 1j * im


def mpse_correction(P: Poles, rho_int, e_rel, n_coarse=90):
    """Delta Sigma(E - mu) = [Sigma_MP - Sigma_MP(E_F)] - [Sigma_HL - Sigma_HL(E_F)]
    at the interstitial density, on the energies e_rel (Ha), cached on disk."""
    e_rel = np.asarray(e_rel, float)
    rs = round(float((3.0 / (4.0 * np.pi * rho_int)) ** (1.0 / 3.0)), 3)    # smooth in rs; cache-friendly
    kf = KF_RS / rs
    wp = math.sqrt(3.0 / rs ** 3)
    emax = max(float(e_rel.max()), 1.0)
    # coarse grid: dense near the Fermi level, where the poles switch on
    ec = np.unique(np.concatenate([[0.0], np.geomspace(1e-3, emax, n_coarse - 1)]))
    tag = hashlib.sha1(json.dumps(dict(p=P.key(), rs=round(rs, 8), e=[round(float(x), 10) for x in ec]))
                       .encode()).hexdigest()[:16]
    fn = os.path.join(_cache_dir(), f"mpse_{P.label}_{tag}.npz")
    if os.path.exists(fn):
        d = np.load(fn)
        dre, dim = d["dre"], d["dim"]
    else:
        kc = np.sqrt(kf * kf + 2.0 * ec) * (1.0 + 1e-9)
        smp = sigma_c_poles(P, kf, kc)
        shl = sigma_c_poles(Poles(w=np.array([wp]), g=np.array([1.0]), label="hl"), kf, kc)
        dre = (smp.real - smp.real[0]) - (shl.real - shl.real[0])
        dim = smp.imag - shl.imag
        np.savez(fn, dre=dre, dim=dim, ec=ec)
    return np.interp(e_rel, ec, dre) + 1j * np.interp(e_rel, ec, dim)


def imfp_angstrom(P: Poles | None, rho_int, e_rel_ev):
    """Inelastic mean free path (intensity, A) of an electron e_rel above
    the Fermi level in the gas at rho_int with the poles' losses; HL when P is
    None. 1/(2 Im p), p^2 = kF^2 + 2 (E - Sigma)."""
    rs = (3.0 / (4.0 * np.pi * rho_int)) ** (1.0 / 3.0)
    kf = KF_RS / rs
    e = np.asarray(e_rel_ev, float) / HARTREE_EV
    k = np.sqrt(kf * kf + 2.0 * e)
    if P is None:
        P = Poles(w=np.array([math.sqrt(3.0 / rs ** 3)]), g=np.array([1.0]), label="hl")
    im = np.zeros_like(k)
    for wj, gj in zip(P.w, P.g):
        im += gj * im_sigma_pole(kf, wj, k)
    p = np.sqrt(k * k - 2j * im)
    return BOHR_A / (2.0 * p.imag)
