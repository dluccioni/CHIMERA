"""The alloy being fitted: elements, composition, lattice, reference cell, tables.

One value object holds everything the model needs to know about the sample:
the absorbing elements (in model order), their atomic fractions, the
lattice, the reference lattice constant the supercell is built at, and the
names of the scattering-table sets. The model, the pair-count model, the
calibration and the fit driver all take an `AlloySystem`, derive the
species count from it, and write it into every calibration and fit record
so a results tree says what it was fitted as.

There is no built-in alloy. Name one explicitly:

    sys = AlloySystem(elements=("Fe", "Ni"), lattice="fcc", a0_ref=3.59,
                      tables_name="native", ms_name="native")
    sys = AlloySystem.from_tables("feni_fcc2538", ms_name="feni_fcc2538_ms")
    S = Spectrum(system=sys)

or set it once for a session, after which calls that are not given a
system use it:

    set_default(sys)

`from_tables` reads `data/<name>/meta.json`, which `feff8l_tables.py` writes:
the elements, the nearest-neighbour spacing of the FEFF cluster and (new) the
lattice it was built on. The reference lattice constant follows from the
spacing and the lattice unless given.

The table-set name "native" (or "native:seed=1,rpath=5.8", ...) needs no
directory: the native engine computes the paths in-process for the system's
own elements, composition, lattice and reference spacing
(`exafs_gpu.abinitio.provider`), so any alloy can be fitted without FEFF:

    sys = resolve("native", elements=("Fe", "Ni"), lattice="fcc", a0_ref=3.59)

What is NOT general, still: every basis site must be crystallographically
equivalent (uniform coordination) - the neighbour-list machinery and the MC
proposal kernels assume it - which `CrystalSupercell` enforces for sc, bcc,
fcc and hcp.
"""
from __future__ import annotations
import os, json
from dataclasses import dataclass, replace
import numpy as np

from exafs_gpu.scattering import Z_TABLE, is_native

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")

# conventional lattice constant per nearest-neighbour distance
A0_PER_NN = {"fcc": np.sqrt(2.0), "bcc": 2.0 / np.sqrt(3.0), "sc": 1.0, "hcp": 1.0}

# how the data's k maps onto the model's momentum (AlloySystem.k_axis)
K_AXES = ("constant", "table")


def tables_dir(name):
    """`data/<name>`, or `name` itself when it is already a directory."""
    if os.path.isdir(name):
        return name
    return os.path.join(DATA_DIR, name)


def read_meta(name):
    """The `meta.json` of a table set, or {} when there is none."""
    p = os.path.join(tables_dir(name), "meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def pair_labels(elements, sep=""):
    """Unordered pair labels in the (a <= b) order the code uses everywhere:
    for elements (A, B, C): AA, AB, AC, BB, BC, CC."""
    els = tuple(elements)
    return [f"{els[a]}{sep}{els[b]}" for a, b in zip(*np.triu_indices(len(els)))]


def wc_labels(elements, tex=False):
    if tex:
        return [rf"$\alpha_{{\rm {lab}}}$" for lab in pair_labels(elements)]
    return [f"a_{lab}" for lab in pair_labels(elements)]


def integer_counts(composition, natoms):
    """Species counts summing to `natoms`, rounded the way `random_config`
    always has (the last species takes the remainder)."""
    comp = np.asarray(composition, float)
    comp = comp / comp.sum()
    cnt = np.round(comp * natoms).astype(int)
    cnt[-1] = natoms - cnt[:-1].sum()
    return cnt


@dataclass(frozen=True)
class AlloySystem:
    elements: tuple
    composition: tuple = None
    lattice: str = "fcc"
    a0_ref: float = None
    tables_name: str = None
    ms_name: str = None
    c_over_a: float = None
    name: str = None
    # How the data's k (measured from the edge) maps onto the model's internal
    # momentum p. "constant": p^2 = k^2 + ETOK dE0 with one dE0 per edge, which
    # then has to carry the whole inner potential. "table": through the
    # scattering calculation's own energy-dependent k(p); dE0 is then only the
    # offset between the experimenter's E0 and the calculation's Fermi level.
    k_axis: str = "constant"

    def __post_init__(self):
        els = tuple(str(e) for e in self.elements)
        object.__setattr__(self, "elements", els)
        n = len(els)
        if n < 2:
            raise ValueError("an alloy needs at least two species")
        for e in els:
            if e not in Z_TABLE:
                raise ValueError(f"unknown element symbol {e!r}")
        comp = (np.ones(n) / n if self.composition is None
                else np.asarray(self.composition, float))
        if comp.shape != (n,) or comp.min() <= 0:
            raise ValueError(f"composition must be {n} positive fractions")
        object.__setattr__(self, "composition", tuple(float(x) for x in comp / comp.sum()))
        if self.lattice not in A0_PER_NN:
            raise ValueError(f"unknown lattice {self.lattice!r}; have {list(A0_PER_NN)}")
        if self.a0_ref is None:
            raise ValueError("a0_ref is required (use from_tables to derive it)")
        object.__setattr__(self, "a0_ref", float(self.a0_ref))
        if self.name is None:
            object.__setattr__(self, "name", "".join(els))
        if self.k_axis not in K_AXES:
            raise ValueError(f"k_axis must be one of {K_AXES}, not {self.k_axis!r}")

    # ----------------------------------------------------------- derived
    @property
    def nsp(self):
        return len(self.elements)

    @property
    def z(self):
        return tuple(Z_TABLE[e] for e in self.elements)

    @property
    def nn_ref(self):
        """Nearest-neighbour distance of the reference cell."""
        return self.a0_ref / A0_PER_NN[self.lattice]

    def wc_labels(self, tex=False):
        return wc_labels(self.elements, tex)

    def pair_labels(self, sep=""):
        return pair_labels(self.elements, sep)

    def counts(self, natoms):
        return integer_counts(self.composition, natoms)

    def replace(self, **changes):
        return replace(self, **changes)

    # ------------------------------------------------------- persistence
    def to_dict(self):
        return dict(elements=list(self.elements), composition=list(self.composition),
                    lattice=self.lattice, a0_ref=self.a0_ref,
                    tables_name=self.tables_name, ms_name=self.ms_name,
                    c_over_a=self.c_over_a, name=self.name, k_axis=self.k_axis)

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return _require_default("no system recorded")
        return cls(elements=tuple(d["elements"]), composition=d.get("composition"),
                   lattice=d.get("lattice", "fcc"), a0_ref=d.get("a0_ref"),
                   tables_name=d.get("tables_name"), ms_name=d.get("ms_name"),
                   c_over_a=d.get("c_over_a"), name=d.get("name"),
                   k_axis=d.get("k_axis", "constant"))

    @classmethod
    def from_tables(cls, name, ms_name=None, composition=None, a0_ref=None,
                    lattice=None, c_over_a=None, elements=None, system_name=None):
        """Describe an alloy from the FEFF table set `data/<name>` and its
        meta.json (elements, d_nn, lattice). `ms_name` is the matching
        multiple-scattering set, if one was generated. A native name has no
        directory and is handed to `resolve`."""
        if is_native(name):
            return resolve(name, ms_name=ms_name, composition=composition, a0_ref=a0_ref,
                           lattice=lattice, c_over_a=c_over_a, elements=elements,
                           name=system_name)
        meta = read_meta(name)
        els = tuple(elements or meta.get("elements") or ())
        if not els:
            raise FileNotFoundError(f"no meta.json with elements under {tables_dir(name)}; "
                                    f"pass elements=")
        lat = lattice or meta.get("lattice", "fcc")
        if a0_ref is None:
            d_nn = meta.get("d_nn")
            if d_nn is None:
                raise ValueError("meta.json records no d_nn; pass a0_ref=")
            a0_ref = float(d_nn) * A0_PER_NN[lat]
        ca = c_over_a if c_over_a is not None else meta.get("c_over_a")
        comp = composition if composition is not None else meta.get("composition")
        return cls(elements=els, composition=comp, lattice=lat, a0_ref=a0_ref,
                   tables_name=name, ms_name=ms_name, c_over_a=ca, name=system_name)


_default = None


def set_default(system):
    """Make `system` (anything `resolve` accepts, or None to clear it) the
    alloy used by every call that is not given one. Returns it."""
    global _default
    _default = None if system is None else resolve(system)
    return _default


def get_default():
    """The session default set with `set_default`, or None."""
    return _default


def _require_default(why):
    if _default is None:
        raise ValueError(
            f"{why}, and no default alloy is set: pass system= (an AlloySystem, "
            f"a table-set name or a calibration record), or elements= with "
            f"a0_ref=, or call chimera.system.set_default(...) once")
    return _default


def _from_overrides(kind, ch):
    """An AlloySystem from keyword fields alone (no base system)."""
    if not ch.get("elements"):
        return _require_default("no elements given")
    if ch.get("a0_ref") is None:
        raise ValueError(f"{kind}: the reference lattice constant is needed "
                         f"(a0_ref=, or --a0-ref on the command line)")
    return AlloySystem(**ch)


def resolve(system=None, **overrides):
    """An AlloySystem from `system` (an AlloySystem, a dict, a table-set name,
    "analytic", "native[:recipe]", or None) with any non-None keyword
    overrides applied. With None the session default (`set_default`) is the
    base; without one, the overrides must describe the alloy."""
    ch = {k: v for k, v in overrides.items() if v is not None}
    if "elements" in ch:
        ch["elements"] = tuple(ch["elements"])
    if isinstance(system, str) and (system == "analytic" or is_native(system)):
        # the stand-in tables, or paths computed in-process: no directory and
        # no meta.json, so the alloy comes from the overrides (or the default)
        fields = dict(tables_name=system,
                      ms_name=None if system == "analytic" else system)
        base = _default
        if base is not None and ch.get("elements", base.elements) == base.elements:
            return base.replace(**{**fields, **ch})
        return _from_overrides(system, {**fields, **ch})
    if system is None:
        if _default is None:
            return _from_overrides("no system named", ch)
        base = _default
        if "elements" in ch and ch["elements"] != base.elements:
            # a different element list: nothing sensible to inherit from the
            # default's table names, name or composition
            for k in ("tables_name", "ms_name", "name", "composition"):
                ch.setdefault(k, None)
    elif isinstance(system, AlloySystem):
        base = system
    elif isinstance(system, dict):
        base = AlloySystem.from_dict(system)
    elif isinstance(system, str):
        base = AlloySystem.from_tables(system)
    else:
        raise TypeError(f"cannot make an AlloySystem from {type(system).__name__}")
    return base.replace(**ch) if ch else base


def system_of_tree(out):
    """The AlloySystem a results tree was fitted as, from its calibration.json
    (top level, or inside the shared record). A tree that records none falls
    back to the session default."""
    p = os.path.join(out, "calibration.json")
    if not os.path.exists(p):
        return _require_default(f"{out} has no calibration.json")
    c = json.load(open(p))
    d = c.get("system")
    if d is None:
        cal = c.get("calibration", {})
        for k in sorted(cal, key=lambda s: float(s)):
            d = cal[k].get("system") or cal[k].get("shared", {}).get("system")
            if d:
                break
    return AlloySystem.from_dict(d) if d else _require_default(f"{p} records no system")
