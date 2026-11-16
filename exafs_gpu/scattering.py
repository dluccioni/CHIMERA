"""
Scattering tables: f(k), delta(k), lambda(k) for the EXAFS path expansion.

The physics interface here is deliberately FEFF-shaped so that real
feffNNNN.dat tables can be dropped in later (see `ScatteringTables.from_feff`).

The bundled `analytic_transition_metal_tables` is a STAND-IN, not FEFF output.
It reproduces the qualitative features that matter for a method comparison:
  * |f(k)| rises from zero, peaks near k ~ 3-5 A^-1, decays at high k
  * amplitude scales with Z, so neighbours in the periodic table (Co/Ni) are
    barely separable while species a few Z apart (Cr/Ni) are
  * total phase delta(k) = delta_absorber(k) + delta_scatterer(k), smooth in k
  * lambda(k) follows the usual "universal curve" shape

Because the SAME tables generate the synthetic data and drive the fits, the
A-vs-B comparison is self-consistent. Absolute recovered values are only as
good as the tables, so real analysis must swap in FEFF.
"""
from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass, field

# Atomic numbers by symbol, H to U. The species of a fit come from the
# AlloySystem (chimera/system.py); these are only the lookup behind it.
SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
           "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb "
           "Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re "
           "Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U").split()
Z_TABLE = {s: i + 1 for i, s in enumerate(SYMBOLS)}

# Table-set name for paths the native engine computes in-process
# (exafs_gpu.abinitio.provider): "native", or "native:<recipe options>".
NATIVE = "native"


def is_native(name):
    """True when a table-set name asks for in-process native paths (no files)."""
    return isinstance(name, str) and (name == NATIVE or name.startswith(NATIVE + ":"))


def _meta(root):
    import json
    p = os.path.join(root, "meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


@dataclass
class ScatteringTables:
    """Tabulated scattering functions on a shared k-grid.

    f      : (nsp, Nk)        backscattering amplitude, indexed by scatterer
    delta  : (nsp, nsp, Nk)   total phase, indexed [absorber, scatterer]
    lam    : (Nk,)            photoelectron mean free path (Angstrom)
    k      : (Nk,)            wavenumber grid (A^-1)
    s02    : float            amplitude reduction factor

    Optional, filled by the first-principles generator:
    red          : (nsp, Nk)  central-atom reduction factor |exp(2i delta_1)|
                   by absorber. Its composition average is already folded
                   into f (the kernel carries one amplitude per scatterer).
    e0_offset_ev : where k = 0 of this table sits relative to the Fermi
                   level. The generator tabulates on the INTERNAL momentum
                   (relative to the interstitial potential), so k = 0 here is
                   e0_offset_ev BELOW the Fermi level. For real data whose k
                   is measured from the edge, expect a fitted E0 shift of
                   about -e0_offset_ev.
    meta         : provenance and the muffin-tin / interstitial parameters
    k_of_p       : (nsp, Nk) for each absorber edge, the k measured from the
                   calculation's Fermi level that goes with each internal
                   momentum on the grid: p^2 - k^2 is the energy-dependent
                   inner potential (self-energy included), which typically
                   falls by several eV over k = 3-13 for 3d metals. None for
                   tables tabulated on p alone.
    """
    k: np.ndarray
    f: np.ndarray
    delta: np.ndarray
    lam: np.ndarray
    s02: float = 0.85
    red: np.ndarray | None = None
    e0_offset_ev: float = 0.0
    meta: dict = field(default_factory=dict)
    k_of_p: np.ndarray | None = None

    def __post_init__(self):
        n = self.f.shape[0]
        assert self.f.shape == (n, self.k.size), "f must be (nsp, nk)"
        assert self.delta.shape == (n, n, self.k.size), "delta must be (nsp, nsp, nk)"
        assert self.lam.shape == self.k.shape
        if self.red is not None:
            assert self.red.shape == (n, self.k.size), "red must be (nsp, nk)"
        if self.k_of_p is not None:
            assert self.k_of_p.shape == (n, self.k.size), "k_of_p must be (nsp, nk)"

    @property
    def nsp(self):
        """Number of species."""
        return self.f.shape[0]

    def as_float64(self):
        return (np.ascontiguousarray(self.k, np.float64),
                np.ascontiguousarray(self.f, np.float64),
                np.ascontiguousarray(self.delta, np.float64),
                np.ascontiguousarray(self.lam, np.float64))

    @staticmethod
    def from_feff(paths: dict, k: np.ndarray, elements, s02: float = 0.85,
                  composition=None) -> "ScatteringTables":
        """Build tables from real FEFF feffNNNN.dat single-scattering files.

        `paths` maps (absorber, scatterer) -> filename. Every pair must be
        present. No larch needed; see `read_feffdat` for the convention
        conversion (FEFF tabulates on k relative to the Fermi level and folds
        2(p-k)R into the phase; this pipeline needs the internal momentum p).
        """
        return ScatteringTables.from_records(
            {pair: read_feffdat(fn) for pair, fn in paths.items()}, k, elements, s02,
            composition, route="FEFF8L")

    @staticmethod
    def from_records(records: dict, k: np.ndarray, elements, s02: float = 0.85,
                     composition=None, route="FEFF8L", extra_meta=None) -> "ScatteringTables":
        """Build tables from path records, (absorber, scatterer) -> the dict
        `columns_to_record` makes: from files (`from_feff`) or handed over in
        memory by the native engine (`exafs_gpu.abinitio.provider`)."""
        k = np.asarray(k, float)
        elements = tuple(elements)
        n = len(elements)
        comp = np.ones(n) / n if composition is None else np.asarray(composition, float)
        comp = comp / comp.sum()
        f_pair = np.zeros((n, n, k.size)); delta = np.zeros((n, n, k.size))
        lam_abs = np.zeros((n, k.size)); e0 = np.zeros(n)
        k_of_p = np.zeros((n, k.size))
        meta = {}
        for (a, s), d in records.items():
            ia, isc = elements.index(a), elements.index(s)
            f_pair[ia, isc] = np.interp(k, d["p"], d["f"])
            delta[ia, isc] = np.interp(k, d["p"], d["delta"])
            lam_abs[ia] = np.interp(k, d["p"], d["lam"])
            # one relation per absorber (its potential), the same in each of its records
            k_of_p[ia] = np.interp(k, d["p"], d["k"])
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
                                meta=dict(route=route, pairs=meta, f_pair=f_pair,
                                          elements=elements, **(extra_meta or {})),
                                k_of_p=k_of_p)


def columns_to_record(cols, reff, deg=1.0, nleg=2, scatterer=None, header=""):
    """One path's FEFF columns -> this pipeline's convention.

    `cols` is (nk, 7): k, real[2*phc], mag[feff], phase[feff], red factor,
    lambda, real[p], as in a feffNNNN.dat. FEFF: chi = amp/(k R^2)
    exp(-2R/lam) sin(2 k R + pha), with k relative to the Fermi level,
    amp = mag_feff * red_fact, pha = real[2*phc] + pha_feff, and real[p] the
    internal momentum. On the p axis:
        f(p) = amp * p/k,   delta(p) = pha - 2 (p - k) R_eff,   lam(p) = lam.
    The file reader and the in-process native engine both go through here, so
    both routes hand the fit the same numbers.
    """
    k, phc, mag, pha, red, lam, p = np.asarray(cols, float).T
    sel = k > 0.3
    k, phc, mag, pha, red, lam, p = (x[sel] for x in (k, phc, mag, pha, red, lam, p))
    f = mag * red * p / k
    delta = np.unwrap(phc + pha) - 2.0 * (p - k) * reff
    e0 = float(np.interp(0.0, k, p) ** 2 / (2.0 / (27.211386245988 * 0.529177210903 ** 2)))
    return dict(p=p, k=k, f=f, delta=delta, lam=lam, reff=float(reff), deg=float(deg),
                nleg=int(nleg), scatterer=scatterer, e0_offset_ev=e0, header=header)


def read_feffdat(fn):
    """Parse one feffNNNN.dat and convert it with `columns_to_record`."""
    lines = open(fn).read().split("\n")
    i_hdr = next(i for i, ln in enumerate(lines) if ln.strip().startswith("k ") and "real[2*phc]" in ln)
    i_geo = next(i for i, ln in enumerate(lines) if "nleg, deg, reff" in ln)
    nleg, deg, reff = (float(x) for x in lines[i_geo].split()[:3])
    atoms = [ln.split() for ln in lines[i_geo + 2:i_hdr]]
    scatterer = atoms[1][5] if len(atoms) > 1 else None
    data = np.array([[float(x) for x in ln.split()] for ln in lines[i_hdr + 1:] if ln.strip()])
    return columns_to_record(data, reff, deg, nleg, scatterer, header="\n".join(lines[:i_geo]))


def tables_root(name):
    """`data/<name>` under the package, or `name` when it is a directory."""
    if os.path.isdir(name):
        return name
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", name)


def feff8l_tables(k, name, elements=None, s02=0.85, composition=None):
    """Tables from the FEFF8L alloy runs in data/<name>/.

    One run per absorber, produced by feff8l_tables.py; each holds the
    nearest-neighbour single-scattering path to every species. The element
    list comes from the set's meta.json unless given.
    """
    root = tables_root(name)
    if elements is None:
        elements = _meta(root).get("elements")
        if not elements:
            raise FileNotFoundError(f"{root} has no meta.json naming the elements; "
                                    f"pass elements=")
    elements = tuple(elements)
    paths = {}
    for a in elements:
        d = os.path.join(root, a)
        for fn in sorted(os.listdir(d)):
            if fn.startswith("feff0") and fn.endswith(".dat"):
                s = read_feffdat(os.path.join(d, fn))["scatterer"]
                paths[(a, s)] = os.path.join(d, fn)
    missing = [(a, s) for a in elements for s in elements if (a, s) not in paths]
    assert not missing, f"FEFF8L runs lack pairs {missing}"
    return ScatteringTables.from_feff(paths, np.asarray(k, float), elements, s02,
                                      composition=composition)


def default_tables(k, which, s02=0.85, elements=None):
    """Tables by kind: "analytic" (the Z-parameterised stand-in below, for
    kernel tests) or the name of a FEFF8L set under data/."""
    k = np.asarray(k, float)
    if which == "analytic":
        if elements is None:
            raise ValueError("analytic tables need elements=")
        return analytic_transition_metal_tables(k, elements, s02=s02)
    return feff8l_tables(k, which, s02=s02, elements=elements)


def analytic_transition_metal_tables(k: np.ndarray, elements, s02: float = 0.85
                                     ) -> ScatteringTables:
    """Physically-shaped stand-in tables, parameterised by Z (tuned on the
    3d transition metals; any elements are accepted). See module docstring."""
    elements = tuple(elements)
    nsp = len(elements)
    k = np.asarray(k, np.float64)
    kk = np.maximum(k, 1e-6)

    f = np.zeros((nsp, k.size))
    for i, el in enumerate(elements):
        Z = Z_TABLE[el]
        # log-normal envelope in k: peak position and amplitude drift with Z
        amp = 0.084 * Z ** 0.78
        kpeak = 3.55 + 0.021 * (Z - 24)
        width = 0.86 - 0.004 * (Z - 24)
        f[i] = amp * np.exp(-0.5 * (np.log(kk / kpeak) / width) ** 2)
        # low-k suppression: amplitude must vanish as k -> 0
        f[i] *= (1.0 - np.exp(-(kk / 1.1) ** 2))

    # Phases. Central-atom phase dominates and is absorber-specific; the
    # scatterer contributes a weaker, Z-dependent term.
    delta = np.zeros((nsp, nsp, k.size))
    for ia, ea in enumerate(elements):
        Za = Z_TABLE[ea]
        d_cen = (np.pi - 0.905 * kk + 0.0143 * kk ** 2
                 - 0.00031 * kk ** 3 - 0.0032 * (Za - 24))
        for isc, es in enumerate(elements):
            Zs = Z_TABLE[es]
            d_sc = (-1.42 + 0.271 * kk - 0.00686 * kk ** 2
                    + 0.0121 * (Zs - 24) - 0.00042 * (Zs - 24) * kk)
            delta[ia, isc] = d_cen + d_sc

    # Universal-curve mean free path (Angstrom): shallow minimum near k~6,
    # rising at both ends. Calibrated to lam(3)~9, lam(6)~6.4, lam(14)~11.4,
    # which is the right regime for 3d metals and lets shells 3-4 survive.
    lam = 59.57 / np.maximum(kk, 1.0) ** 2 + 0.794 * kk
    lam = np.clip(lam, 4.0, 30.0)

    return ScatteringTables(k=k, f=f, delta=delta, lam=lam, s02=s02,
                            meta=dict(route="analytic", elements=elements))


def contrast_report(tab: ScatteringTables, labels=None) -> str:
    """Quantify how separable the scatterers are - the crux of the problem."""
    n = tab.nsp
    labels = labels or tuple(tab.meta.get("elements") or (f"s{i}" for i in range(n)))
    lines = ["Backscattering amplitude contrast (RMS relative difference over k):"]
    for i in range(n):
        for j in range(i + 1, n):
            num = np.sqrt(np.mean((tab.f[i] - tab.f[j]) ** 2))
            den = np.sqrt(np.mean(0.5 * (tab.f[i] ** 2 + tab.f[j] ** 2)))
            lines.append(f"  |f_{labels[i]} - f_{labels[j]}| / <f> = {num/den:6.2%}")
    return "\n".join(lines)
