"""Scattering paths computed in-process by the native engine: no table files.

A fit takes two things from a scattering calculation:

* one nearest-neighbour single-scattering path per (absorber, scatterer)
  pair, whose amplitude, phase and mean free path drive every pair of the
  supercell (`exafs_gpu.forward`);
* the 3- and 4-leg paths of one random alloy cluster around each absorber,
  summed into the composition-averaged multiple-scattering term
  (`chimera.multiscat`).

`path_records` computes both with the native engine and returns them as the
same path records `scattering.read_feffdat` makes from feffNNNN.dat files,
through the same column conversion (`scattering.columns_to_record`). A fit on
in-process paths and a fit on files written from the same calculation
therefore get the same numbers, and no FEFF or FEFF-format file is involved.

The calculation takes ~15 s for CrCoNi with 4-leg paths to 5.3 A, so
results are cached in memory and in data/_cache (or $CHIMERA_CACHE), keyed
on the alloy, the reference spacing, the recipe and ENGINE_VERSION. Bump
ENGINE_VERSION whenever a change to the engine moves the numbers.

    recs = path_records(("Cr", "Co", "Ni"), (1/3, 1/3, 1/3), "fcc", 2.5215)
    tables = recs.ss_tables(k_grid, s02=1.0)
    ms = recs.ms["Cr"]                      # path records, 3+ legs

The physics is chosen in the recipe; the defaults are the original engine
(overlapped non-relativistic atoms, Hedin-Lundqvist self-energy). The most
complete set is "native:rel=1,scf=1,sigma=mpse": scalar-relativistic atoms
and scattering states, self-consistent potentials, and the many-pole
self-energy of the alloy's measured loss function (losses.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, fields

import numpy as np

from ..scattering import NATIVE, ScatteringTables, columns_to_record, is_native
from . import cluster as CL
from . import paths as PA
from . import pathsum as PSM
from . import phaseshift as PS
from . import potential as PO
from . import scf as SCF
from .atom import _cache_dir
from .potential import BOHR, HARTREE
from .selfenergy import GAMMA_K
from .tables import feff_columns

ENGINE_VERSION = "native-1"
_FIELDS = ("p", "k", "f", "delta", "lam")


@dataclass(frozen=True)
class Recipe:
    """What the in-process calculation does. Part of the cache key; written
    into a fit's records by naming it in the table-set name, e.g.
    "native" (all defaults) or "native:seed=1,rpath=5.8"."""
    radius: float = 6.5      # cluster radius for the overlapped-atom potentials (A)
    rpath: float = 5.3       # longest half path length of the multiple-scattering set (A)
    nleg: int = 4            # most legs per multiple-scattering path
    seed: int = 0            # which random alloy cluster sets the species around each absorber
    exchange: str = "hl"     # self-energy: Hedin-Lundqvist ("dh", "gs" also built)
    kmax: float = 20.0       # top of the tabulated momentum grid (A^-1)
    scf: int = 0             # 1: self-consistent potentials (scf.py); 0: overlapped atoms
    rel: int = 0             # 1: scalar-relativistic atoms, potentials and scattering states
    sigma: str = "hl"        # "mpse": many-pole self-energy from the measured loss function (losses.py)
    quinn: int = 0           # 1: Quinn's electron-hole loss below the plasmon (HL only)

    @classmethod
    def parse(cls, name):
        """Recipe from a table-set name: "native" or "native:key=value,...";
        None when the name is not a native one."""
        if not is_native(name):
            return None
        _, _, opts = str(name).partition(":")
        kw = {}
        types = {f.name: f.type for f in fields(cls)}
        for item in filter(None, (s.strip() for s in opts.split(","))):
            key, _, val = item.partition("=")
            key = key.strip().replace("-", "_")
            if key not in types:
                raise ValueError(f"unknown native recipe option {key!r}; have {sorted(types)}")
            conv = {"float": float, "int": int, "str": str}[str(types[key])]
            kw[key] = conv(val.strip())
        return cls(**kw)

    def name(self):
        """The table-set name that reproduces this recipe."""
        changed = [f"{f.name}={getattr(self, f.name)}" for f in fields(self)
                   if getattr(self, f.name) != f.default]
        return NATIVE + (":" + ",".join(changed) if changed else "")


@dataclass
class NativePaths:
    """Path records of one alloy at one reference spacing."""
    elements: tuple
    composition: tuple
    lattice: str
    d_nn: float
    c_over_a: float | None
    recipe: Recipe
    key: str
    ss: dict          # (absorber, scatterer) -> nearest-neighbour single-scattering record
    ms: dict          # absorber -> [records with 3+ legs]
    potential: dict   # absorber -> muffin-tin and interstitial summary

    def ss_tables(self, k, s02=0.85):
        """The ScatteringTables a fit model takes, on momentum grid `k`."""
        return ScatteringTables.from_records(
            self.ss, k, self.elements, s02, self.composition, route="native",
            extra_meta=dict(provenance=self.provenance()))

    def provenance(self):
        return dict(engine="chimera-native", version=ENGINE_VERSION, table_set=self.recipe.name(),
                    recipe=asdict(self.recipe), d_nn=self.d_nn, cache_key=self.key,
                    n_ms_paths={a: len(v) for a, v in self.ms.items()},
                    potential=self.potential)


_MEM: dict = {}


def path_records(elements, composition, lattice, d_nn, c_over_a=None, recipe=None,
                 cache=True, verbose=True) -> NativePaths:
    """Native path records for every absorber of the alloy, at nearest-neighbour
    spacing `d_nn` (A). Cached in memory and, unless cache=False, on disk."""
    recipe = recipe or Recipe()
    elements = tuple(str(e) for e in elements)
    comp = np.ones(len(elements)) if composition is None else np.asarray(composition, float)
    comp = tuple(float(x) for x in comp / comp.sum())
    d_nn = float(d_nn)
    key = _key(elements, comp, lattice, d_nn, c_over_a, recipe)
    if key in _MEM:
        return _MEM[key]
    fn = os.path.join(_cache_dir(), f"native_paths_{key}.npz")
    out = None
    if cache and os.path.exists(fn):
        try:
            out = _load(fn, elements, d_nn)
        except (OSError, KeyError, ValueError):
            out = None                                      # a damaged file is recomputed
    if out is None:
        t0 = time.time()
        out = _compute(elements, comp, lattice, d_nn, c_over_a, recipe, verbose)
        if verbose:
            print(f"  native scattering paths computed in {time.time() - t0:.1f}s "
                  f"({recipe.name()}, d_nn {d_nn:.4f} A)", flush=True)
        if cache:
            _save(fn, out, elements)
    res = NativePaths(elements=elements, composition=comp, lattice=lattice, d_nn=d_nn,
                      c_over_a=c_over_a, recipe=recipe, key=key, **out)
    _MEM[key] = res
    return res


def _key(elements, comp, lattice, d_nn, c_over_a, recipe):
    blob = json.dumps(dict(v=ENGINE_VERSION, elements=list(elements),
                           composition=[round(x, 6) for x in comp], lattice=lattice,
                           d_nn=round(d_nn, 6),
                           c_over_a=None if c_over_a is None else round(float(c_over_a), 6),
                           recipe=asdict(recipe)), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _compute(elements, comp, lattice, d_nn, c_over_a, recipe, verbose):
    k_grid = PS.default_k_grid(recipe.kmax)
    ss, ms, pot = {}, {}, {}
    rel = bool(recipe.rel)
    mp = None
    if recipe.sigma == "mpse":
        from . import losses as LS
        mp = LS.material_poles(elements, comp, lattice, d_nn, c_over_a)
    elif recipe.sigma != "hl":
        raise ValueError(f"unknown self-energy {recipe.sigma!r}: 'hl' or 'mpse'")
    if mp is not None and recipe.quinn:
        raise ValueError("quinn=1 and sigma=mpse both add the low-energy loss; choose one")
    for a in elements:
        t0 = time.time()
        pts = CL.lattice_cluster(d_nn, recipe.radius, lattice, c_over_a)
        species = CL.alloy_species(pts, a, list(elements), d_nn, comp, recipe.seed)
        scf_info = None
        if recipe.scf:
            res = SCF.build(pts, species, verbose=False, relativistic=rel)
            cpot = res.potential
            scf_info = dict(converged=bool(res.converged), iterations=len(res.history),
                            net_charge=res.charges)
        else:
            cpot = PO.build(pts, species, relativistic=rel)
        ph = PS.phase_shifts(cpot, k_grid, gamma_ev=GAMMA_K.get(a, 1.0), model=recipe.exchange,
                             quinn=bool(recipe.quinn), mpse=mp)
        type_of = {t.symbol: i for i, t in enumerate(cpot.types) if i > 0}
        missing = [s for s in elements if s not in type_of]
        if missing:
            raise ValueError(f"the {recipe.radius} A cluster around {a} holds no {missing}")
        # nearest-neighbour single scattering, one path per scatterer species
        # (a 2-leg path depends only on its length and the two potentials)
        nn = [PSM.Path(np.array([[0.0, 0.0, d_nn / BOHR]]), (type_of[s],), 1.0) for s in elements]
        # multiple scattering: every 3- and 4-leg path of the cluster to rpath
        site_types = np.array([0] + [type_of[s] for s in species[1:]])
        groups = [g for g in PA.enumerate_paths(pts, site_types, recipe.rpath, recipe.nleg)
                  if g.nleg >= 3]
        tr = PSM.path_sum(nn + [g.path for g in groups], ph, precision="double")
        for i, s in enumerate(elements):
            ss[(a, s)] = columns_to_record(feff_columns(tr[i], ph, d_nn), d_nn, 1.0, 2, s)
        ms[a] = [columns_to_record(feff_columns(tr[len(nn) + i], ph, g.reff_A), g.reff_A,
                                   g.degeneracy, g.nleg, cpot.types[g.path.types[0]].symbol)
                 for i, g in enumerate(groups)]
        pot[a] = dict(mu_ev=float(cpot.mu * HARTREE), v_int_ev=float(cpot.v_int * HARTREE),
                      rs_int=float(cpot.rs_int),
                      r_mt_A={t.symbol if i else f"{t.symbol}*": float(t.r_mt * BOHR)
                              for i, t in enumerate(cpot.types)},
                      gamma_ev=float(GAMMA_K.get(a, 1.0)), cluster_atoms=int(len(pts)),
                      scf=scf_info)
        if verbose:
            print(f"    {a}: {len(pts)} atoms, {len(groups)} multiple-scattering paths, "
                  f"{time.time() - t0:.1f}s", flush=True)
    return dict(ss=ss, ms=ms, potential=pot)


# ------------------------------------------------------------------ disk cache
def _save(fn, out, elements):
    arrs, meta = {}, {}
    for a in elements:
        ss = [out["ss"][(a, s)] for s in elements]
        ms = out["ms"][a]
        for fld in _FIELDS:
            arrs[f"{a}.ss.{fld}"] = np.stack([r[fld] for r in ss])
            if ms:
                arrs[f"{a}.ms.{fld}"] = np.stack([r[fld] for r in ms])
        arrs[f"{a}.ms.reff"] = np.array([r["reff"] for r in ms], float)
        arrs[f"{a}.ms.deg"] = np.array([r["deg"] for r in ms], float)
        arrs[f"{a}.ms.nleg"] = np.array([r["nleg"] for r in ms], int)
        meta[a] = dict(potential=out["potential"][a],
                       ss_e0=[r["e0_offset_ev"] for r in ss],
                       ms_e0=[r["e0_offset_ev"] for r in ms],
                       ms_scatterer=[r["scatterer"] for r in ms])
    arrs["meta"] = np.array(json.dumps(meta))
    tmp = f"{fn}.{os.getpid()}.tmp.npz"                       # parallel fits may race here
    np.savez(tmp, **arrs)
    os.replace(tmp, fn)


def _load(fn, elements, d_nn):
    z = np.load(fn, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    ss, ms, pot = {}, {}, {}
    for a in elements:
        m = meta[a]
        for i, s in enumerate(elements):
            ss[(a, s)] = dict({fld: z[f"{a}.ss.{fld}"][i] for fld in _FIELDS},
                              reff=d_nn, deg=1.0, nleg=2, scatterer=s,
                              e0_offset_ev=m["ss_e0"][i], header="")
        reff, deg, nleg = z[f"{a}.ms.reff"], z[f"{a}.ms.deg"], z[f"{a}.ms.nleg"]
        ms[a] = [dict({fld: z[f"{a}.ms.{fld}"][i] for fld in _FIELDS},
                      reff=float(reff[i]), deg=float(deg[i]), nleg=int(nleg[i]),
                      scatterer=m["ms_scatterer"][i], e0_offset_ev=m["ms_e0"][i], header="")
                 for i in range(reff.size)]
        pot[a] = m["potential"]
    return dict(ss=ss, ms=ms, potential=pot)
