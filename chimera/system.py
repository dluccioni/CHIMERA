"""The alloy being fitted: elements, composition, lattice, reference cell, tables.

Everything that used to be a CrCoNi constant scattered over the package -
`ELEMENTS`, `Z_OF`, the 3.566 A reference cell, the `crconi_fcc2517` table
names, the 2.517 A spacing the multiple-scattering paths were computed for -
lives here as one value object. The model, the pair-count model, the
calibration and the fit driver all take an `AlloySystem` (or build the
default one), derive the species count from it, and write it into every
calibration and fit record so a results tree says what it was fitted as.

The default remains equiatomic fcc CrCoNi, so code that never mentions a
system behaves exactly as before.

Retargeting to another alloy:

    sys = AlloySystem.from_tables("feni_bcc2482", ms_name=None,
                                  composition=(0.5, 0.5))
    S = Spectrum(system=sys)

`from_tables` reads `data/<name>/meta.json`, which `feff8l_tables.py` writes:
the elements, the nearest-neighbour spacing of the FEFF cluster and (new) the
lattice it was built on. The reference lattice constant follows from the
spacing and the lattice unless given.

What is NOT general, still: every basis site must be crystallographically
equivalent (uniform coordination) - the neighbour-list machinery and the MC
proposal kernels assume it - which `CrystalSupercell` enforces for sc, bcc,
fcc and hcp.
"""
from __future__ import annotations
import os, json
from dataclasses import dataclass, replace
import numpy as np

from exafs_gpu.scattering import Z_TABLE, ELEMENTS as _DEFAULT_ELEMENTS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")

# conventional lattice constant per nearest-neighbour distance
A0_PER_NN = {"fcc": np.sqrt(2.0), "bcc": 2.0 / np.sqrt(3.0), "sc": 1.0, "hcp": 1.0}


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
    CrCr, CrCo, CrNi, CoCo, CoNi, NiNi for CrCoNi."""
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
                    c_over_a=self.c_over_a, name=self.name)

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return DEFAULT
        return cls(elements=tuple(d["elements"]), composition=d.get("composition"),
                   lattice=d.get("lattice", "fcc"), a0_ref=d.get("a0_ref"),
                   tables_name=d.get("tables_name"), ms_name=d.get("ms_name"),
                   c_over_a=d.get("c_over_a"), name=d.get("name"))

    @classmethod
    def from_tables(cls, name, ms_name=None, composition=None, a0_ref=None,
                    lattice=None, c_over_a=None, elements=None, system_name=None):
        """Describe an alloy from the FEFF table set `data/<name>` and its
        meta.json (elements, d_nn, lattice). `ms_name` is the matching
        multiple-scattering set, if one was generated."""
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


DEFAULT = AlloySystem(elements=_DEFAULT_ELEMENTS, lattice="fcc", a0_ref=3.566,
                      tables_name="crconi_fcc2517", ms_name="crconi_fcc2517_ms",
                      name="CrCoNi")


def resolve(system=None, **overrides):
    """An AlloySystem from `system` (an AlloySystem, a dict, a table-set name
    or None for the default) with any non-None keyword overrides applied."""
    if system is None:
        base = DEFAULT
    elif isinstance(system, AlloySystem):
        base = system
    elif isinstance(system, dict):
        base = AlloySystem.from_dict(system)
    elif isinstance(system, str) and system == "analytic":
        els = overrides.get("elements") or DEFAULT.elements
        base = AlloySystem(elements=els, composition=overrides.get("composition"),
                           lattice=overrides.get("lattice") or "fcc",
                           a0_ref=overrides.get("a0_ref") or (DEFAULT.a0_ref if tuple(els) == DEFAULT.elements else None),
                           tables_name="analytic", ms_name=None,
                           c_over_a=overrides.get("c_over_a"), name=overrides.get("name"))
    elif isinstance(system, str):
        base = AlloySystem.from_tables(system)
    else:
        raise TypeError(f"cannot make an AlloySystem from {type(system).__name__}")
    ch = {k: v for k, v in overrides.items() if v is not None}
    if "elements" in ch and base is DEFAULT and tuple(ch["elements"]) != DEFAULT.elements:
        # a different element list with no tables named: there is nothing
        # sensible to inherit from the CrCoNi default
        ch.setdefault("tables_name", None); ch.setdefault("ms_name", None)
        ch.setdefault("name", None)
        if "composition" not in ch:
            ch["composition"] = None
    return base.replace(**ch) if ch else base


def system_of_tree(out):
    """The AlloySystem a results tree was fitted as, from its calibration.json
    (top level, or inside the shared record); the default for older trees."""
    p = os.path.join(out, "calibration.json")
    if not os.path.exists(p):
        return DEFAULT
    c = json.load(open(p))
    d = c.get("system")
    if d is None:
        cal = c.get("calibration", {})
        for k in sorted(cal, key=lambda s: float(s)):
            d = cal[k].get("system") or cal[k].get("shared", {}).get("system")
            if d:
                break
    return AlloySystem.from_dict(d) if d else DEFAULT
