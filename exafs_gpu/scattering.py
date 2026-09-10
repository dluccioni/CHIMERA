"""
Scattering tables: f(k), delta(k), lambda(k) for the EXAFS path expansion.

The physics interface here is deliberately FEFF-shaped so that real
feffNNNN.dat tables can be dropped in later (see `ScatteringTables.from_feff`).

The bundled `analytic_transition_metal_tables` is a STAND-IN, not FEFF output.
It reproduces the qualitative features that matter for a method comparison:
  * |f(k)| rises from zero, peaks near k ~ 3-5 A^-1, decays at high k
  * amplitude scales with Z, so Cr(24) is separable but Co(27)/Ni(28) are not
  * total phase delta(k) = delta_absorber(k) + delta_scatterer(k), smooth in k
  * lambda(k) follows the usual "universal curve" shape

Because the SAME tables generate the synthetic data and drive the fits, the
A-vs-B comparison is self-consistent. Absolute recovered values are only as
good as the tables, so real analysis must swap in FEFF.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

# Element registry for the CrCoNi system
ELEMENTS = ("Cr", "Co", "Ni")
Z_OF = {"Cr": 24, "Co": 27, "Ni": 28}
NSP = len(ELEMENTS)


@dataclass
class ScatteringTables:
    """Tabulated scattering functions on a shared k-grid.

    f      : (NSP, Nk)        backscattering amplitude, indexed by scatterer
    delta  : (NSP, NSP, Nk)   total phase, indexed [absorber, scatterer]
    lam    : (Nk,)            photoelectron mean free path (Angstrom)
    k      : (Nk,)            wavenumber grid (A^-1)
    s02    : float            amplitude reduction factor

    Optional, filled by the first-principles generator:
    red          : (NSP, Nk)  central-atom reduction factor |exp(2i delta_1)|
                   by absorber. Its composition average is already folded
                   into f (the kernel carries one amplitude per scatterer).
    e0_offset_ev : where k = 0 of this table sits relative to the Fermi
                   level. The generator tabulates on the INTERNAL momentum
                   (relative to the interstitial potential), so k = 0 here is
                   e0_offset_ev BELOW the Fermi level. For real data whose k
                   is measured from the edge, expect a fitted E0 shift of
                   about -e0_offset_ev.
    meta         : provenance and the muffin-tin / interstitial parameters
    """
    k: np.ndarray
    f: np.ndarray
    delta: np.ndarray
    lam: np.ndarray
    s02: float = 0.85
    red: np.ndarray | None = None
    e0_offset_ev: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = self.f.shape[0]
        assert self.f.shape == (n, self.k.size), "f must be (nsp, nk)"
        assert self.delta.shape == (n, n, self.k.size), "delta must be (nsp, nsp, nk)"
        assert self.lam.shape == self.k.shape
        if self.red is not None:
            assert self.red.shape == (n, self.k.size), "red must be (nsp, nk)"

    @property
    def nsp(self):
        """Number of species. NOT the module-level NSP: that is only the default."""
        return self.f.shape[0]

    def as_float64(self):
        return (np.ascontiguousarray(self.k, np.float64),
                np.ascontiguousarray(self.f, np.float64),
                np.ascontiguousarray(self.delta, np.float64),
                np.ascontiguousarray(self.lam, np.float64))

    @staticmethod
    def from_feff(paths: dict, k: np.ndarray, elements=ELEMENTS, s02: float = 0.85,
                  composition=None) -> "ScatteringTables":
        """Build tables from real FEFF feffNNNN.dat single-scattering files.

        `paths` maps (absorber, scatterer) -> filename. Every pair must be
        present. No larch needed; see `read_feffdat` for the convention
        conversion (FEFF tabulates on k relative to the Fermi level and folds
        2(p-k)R into the phase; this pipeline needs the internal momentum p).
        """
        n = len(elements)
        comp = np.ones(n) / n if composition is None else np.asarray(composition, float)
        comp = comp / comp.sum()
        f_pair = np.zeros((n, n, k.size)); delta = np.zeros((n, n, k.size))
        lam_abs = np.zeros((n, k.size)); e0 = np.zeros(n)
        meta = {}
        for (a, s), fn in paths.items():
            ia, isc = elements.index(a), elements.index(s)
            d = read_feffdat(fn)
            f_pair[ia, isc] = np.interp(k, d["p"], d["f"])
            delta[ia, isc] = np.interp(k, d["p"], d["delta"])
            lam_abs[ia] = np.interp(k, d["p"], d["lam"])
            e0[ia] = d["e0_offset_ev"]
            meta[(a, s)] = dict(reff=d["reff"], deg=d["deg"], header=d["header"])
        # the kernel carries one amplitude per scatterer: the absorber
        # dependence (FEFF's reduction factor, a few %) is composition-averaged
        f = np.einsum("a,ask->sk", comp, f_pair)
        red = np.array([f_pair[ia, :, :].mean(axis=0) for ia in range(n)])
        red = red / np.maximum(f[None].mean(axis=1), 1e-12)
        lam = comp @ lam_abs
        return ScatteringTables(k=k, f=f, delta=delta, lam=lam, s02=s02,
                                red=red, e0_offset_ev=float(comp @ e0),
                                meta=dict(route="FEFF8L", pairs=meta,
                                          f_pair=f_pair, elements=tuple(elements)))


def read_feffdat(fn):
    """Parse one feffNNNN.dat and convert to this pipeline's convention.

    FEFF: chi = amp/(k R^2) exp(-2R/lam) sin(2 k R + pha), with k relative to
    the Fermi level, amp = mag_feff * red_fact, pha = real[2*phc] + pha_feff,
    and real[p] the internal momentum. On the p axis:
        f(p) = amp * p/k,   delta(p) = pha - 2 (p - k) R_eff,   lam(p) = lam.
    """
    lines = open(fn).read().split("\n")
    i_hdr = next(i for i, ln in enumerate(lines) if ln.strip().startswith("k ") and "real[2*phc]" in ln)
    i_geo = next(i for i, ln in enumerate(lines) if "nleg, deg, reff" in ln)
    nleg, deg, reff = (float(x) for x in lines[i_geo].split()[:3])
    atoms = [ln.split() for ln in lines[i_geo + 2:i_hdr]]
    scatterer = atoms[1][5] if len(atoms) > 1 else None
    data = np.array([[float(x) for x in ln.split()] for ln in lines[i_hdr + 1:] if ln.strip()])
    k, phc, mag, pha, red, lam, p = data.T
    sel = k > 0.3
    k, phc, mag, pha, red, lam, p = (x[sel] for x in (k, phc, mag, pha, red, lam, p))
    f = mag * red * p / k
    delta = np.unwrap(phc + pha) - 2.0 * (p - k) * reff
    e0 = float(np.interp(0.0, k, p) ** 2 / (2.0 / (27.211386245988 * 0.529177210903 ** 2)))
    return dict(p=p, k=k, f=f, delta=delta, lam=lam, reff=reff, deg=deg,
                nleg=int(nleg), scatterer=scatterer, e0_offset_ev=e0,
                header="\n".join(lines[:i_geo]))


def feff8l_tables(k, name="crconi_fcc2517", elements=ELEMENTS, s02=0.85):
    """Tables from the FEFF8L alloy runs in data/<name>/.

    One run per absorber, produced by feff8l_tables.py; each holds the
    nearest-neighbour single-scattering path to every species.
    """
    import os
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", name)
    paths = {}
    for a in elements:
        d = os.path.join(root, a)
        for fn in sorted(os.listdir(d)):
            if fn.startswith("feff0") and fn.endswith(".dat"):
                s = read_feffdat(os.path.join(d, fn))["scatterer"]
                paths[(a, s)] = os.path.join(d, fn)
    missing = [(a, s) for a in elements for s in elements if (a, s) not in paths]
    assert not missing, f"FEFF8L runs lack pairs {missing}"
    return ScatteringTables.from_feff(paths, np.asarray(k, float), elements, s02)


def default_tables(k, which=None, s02=0.85):
    """The tables the fitting scripts use.

    which: "feff8l" (real FEFF8L output - what produced the published results)
           or "analytic" (a hand-tuned 3d-transition-metal stand-in, used only
           by the kernel validation tests). Defaults to the EXAFS_TABLES
           environment variable, then "feff8l".
    """
    import os
    which = which or os.environ.get("EXAFS_TABLES", "feff8l")
    k = np.asarray(k, float)
    if which == "feff8l":
        return feff8l_tables(k, s02=s02)
    if which == "analytic":
        return analytic_transition_metal_tables(k, s02=s02)
    raise ValueError(f"unknown tables {which!r}; CHIMERA ships 'feff8l' and 'analytic'")


def analytic_transition_metal_tables(k: np.ndarray, s02: float = 0.85) -> ScatteringTables:
    """Physically-shaped stand-in tables for Cr / Co / Ni. See module docstring."""
    k = np.asarray(k, np.float64)
    kk = np.maximum(k, 1e-6)

    f = np.zeros((NSP, k.size))
    for i, el in enumerate(ELEMENTS):
        Z = Z_OF[el]
        # log-normal envelope in k: peak position and amplitude drift with Z
        amp = 0.084 * Z ** 0.78
        kpeak = 3.55 + 0.021 * (Z - 24)
        width = 0.86 - 0.004 * (Z - 24)
        f[i] = amp * np.exp(-0.5 * (np.log(kk / kpeak) / width) ** 2)
        # low-k suppression: amplitude must vanish as k -> 0
        f[i] *= (1.0 - np.exp(-(kk / 1.1) ** 2))

    # Phases. Central-atom phase dominates and is absorber-specific; the
    # scatterer contributes a weaker, Z-dependent term.
    delta = np.zeros((NSP, NSP, k.size))
    for ia, ea in enumerate(ELEMENTS):
        Za = Z_OF[ea]
        d_cen = (np.pi - 0.905 * kk + 0.0143 * kk ** 2
                 - 0.00031 * kk ** 3 - 0.0032 * (Za - 24))
        for isc, es in enumerate(ELEMENTS):
            Zs = Z_OF[es]
            d_sc = (-1.42 + 0.271 * kk - 0.00686 * kk ** 2
                    + 0.0121 * (Zs - 24) - 0.00042 * (Zs - 24) * kk)
            delta[ia, isc] = d_cen + d_sc

    # Universal-curve mean free path (Angstrom): shallow minimum near k~6,
    # rising at both ends. Calibrated to lam(3)~9, lam(6)~6.4, lam(14)~11.4,
    # which is the right regime for 3d metals and lets shells 3-4 survive.
    lam = 59.57 / np.maximum(kk, 1.0) ** 2 + 0.794 * kk
    lam = np.clip(lam, 4.0, 30.0)

    return ScatteringTables(k=k, f=f, delta=delta, lam=lam, s02=s02)


def contrast_report(tab: ScatteringTables, labels=None) -> str:
    """Quantify how separable the scatterers are - the crux of the problem."""
    n = tab.nsp
    labels = labels or (ELEMENTS if n == len(ELEMENTS)
                        else tuple(f"s{i}" for i in range(n)))
    lines = ["Backscattering amplitude contrast (RMS relative difference over k):"]
    for i in range(n):
        for j in range(i + 1, n):
            num = np.sqrt(np.mean((tab.f[i] - tab.f[j]) ** 2))
            den = np.sqrt(np.mean(0.5 * (tab.f[i] ** 2 + tab.f[j] ** 2)))
            lines.append(f"  |f_{labels[i]} - f_{labels[j]}| / <f> = {num/den:6.2%}")
    return "\n".join(lines)
