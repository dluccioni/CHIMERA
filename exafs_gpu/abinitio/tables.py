"""Build a scattering-table set natively and write it in feffNNNN.dat format.

The output is read by exactly the code that reads FEFF8L sets
(`scattering.read_feffdat`, `scattering.feff8l_tables`, `chimera.multiscat`,
`AlloySystem.from_tables`), so switching a fit to native tables is a change
of table name, nothing else.

Each feffNNNN.dat carries, per path, the columns FEFF writes, defined so that
FEFF's formula

    chi(k) = mag * red / (k R^2) * sin(2 k R + 2phc + phase) * exp(-2 R / lambda)

reproduces the native chi exactly:  red = |e^{2 i delta_c}|,
2phc = arg e^{2 i delta_c} - pi (FEFF's -l pi convention for a K edge),
lambda = 1/Im p, real[p] = Re p, and mag e^{i phase} the rest of the path
amplitude. k is measured from the Fermi level of the calculation.

    python native_tables.py --elements Cr Co Ni --d-nn 2.517 --name crconi_fcc2517_native
    python native_tables.py --elements Cr Co Ni --d-nn 2.517 --name crconi_fcc2517_native_ms \\
        --rpath 5.3 --nleg 4
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ..scattering import Z_TABLE
from . import cluster as CL
from . import paths as PA
from . import pathsum as PSM
from . import phaseshift as PS
from . import potential as PO
from . import scf as SCF
from .potential import BOHR, HARTREE
from .selfenergy import GAMMA_K

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")


def feff_columns(tr, ph: PS.PhaseShifts, reff_A):
    """FEFF-format columns of one path from its trace Tr_{l=1} P."""
    k = ph.k
    Sc = ph.S[0, :, 1]
    p = ph.p / BOHR                                   # complex, A^-1
    red = np.abs(Sc)
    two_phc = np.unwrap(np.angle(Sc)) - np.pi
    lam = 1.0 / p.imag
    z = (-1j / 3.0) * tr * k * reff_A ** 2 * np.exp(2.0 * reff_A * p.imag) * np.exp(-2j * k * reff_A)
    mag = np.abs(z)
    pha = np.unwrap(np.angle(z))
    return np.column_stack([k, two_phc, mag, pha, red, lam, p.real])


def chi_of_columns(cols, reff_A):
    k, phc, mag, pha, red, lam, p = cols.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(k > 0, mag * red / (k * reff_A ** 2) * np.sin(2 * k * reff_A + phc + pha)
                        * np.exp(-2 * reff_A / lam), 0.0)


def _write_feffdat(fn, title, cpot, group: PA.PathGroup, cols, symbols, gamma_ev, model):
    ty = cpot.types
    kind = (f"SCF {cpot.meta.get('r_scf_A', 0):.2f} A (cluster Green's function)"
            if cpot.meta.get("scf") else "non-SCF overlapped atoms")
    if cpot.meta.get("relativistic"):
        kind += ", scalar-relativistic"
    lines = [f" {title}    CHIMERA native (EXAFS)",
             f" POT  {kind}, core-hole, folp {cpot.folp:.3f}, exchange {model}"]
    lines.append(f" Abs   Z={ty[0].Z:2d} Rmt={ty[0].r_mt * BOHR:6.3f} Rnm={ty[0].r_norman * BOHR:6.3f} K  shell")
    for i, t in enumerate(ty[1:], 1):
        lines.append(f" Pot {i} Z={t.Z:2d} Rmt={t.r_mt * BOHR:6.3f} Rnm={t.r_norman * BOHR:6.3f}")
    lines.append(f" Gam_ch={gamma_ev:.3E} H-L exch")
    lines.append(f" Mu={cpot.mu * HARTREE:.3E} kf={cpot.kf_int / BOHR:.3E} "
                 f"Vint={cpot.v_int * HARTREE:.3E} Rs_int={cpot.rs_int:6.3f}")
    lines.append(" -----------------------------------------------------------------------")
    rnrm = np.mean([t.r_norman for t in ty])
    lines.append(f"  {group.nleg:2d}  {group.degeneracy:6.3f}  {group.reff_A:7.4f}   "
                 f"{rnrm:7.4f}  {cpot.mu * HARTREE:9.5f} nleg, deg, reff, rnrmav(bohr), edge")
    lines.append("        x         y         z   pot at#")
    abs_sym = symbols[0]
    lines.append(f"   {0.0:8.4f}  {0.0:8.4f}  {0.0:8.4f}  0  {Z_TABLE[abs_sym]:2d} {abs_sym:<4s}  absorbing atom")
    for s, typ in zip(group.path.sites * BOHR, group.path.types):
        sym = ty[typ].symbol
        lines.append(f"   {s[0]:8.4f}  {s[1]:8.4f}  {s[2]:8.4f}  {typ}  {Z_TABLE[sym]:2d} {sym}")
    lines.append("    k   real[2*phc]   mag[feff]  phase[feff] red factor   lambda     real[p]@#")
    for row in cols:
        lines.append(f" {row[0]:6.3f} {row[1]:11.4E} {row[2]:11.4E} {row[3]:11.4E} {row[4]:10.3E} "
                     f"{row[5]:11.4E} {row[6]:11.4E}")
    with open(fn, "w") as f:
        f.write("\n".join(lines) + "\n")


def build_absorber(absorber, elements, d_nn, lattice="fcc", composition=None, radius=6.5,
                   rpath=None, nleg=2, seed=0, k_grid=None, gamma_ev=None, model="hl",
                   c_over_a=None, precision="double", verbose=True, scf=False, rel=False,
                   mpse=None, quinn=False):
    """Potentials, phase shifts, paths and their amplitudes for one absorber.
    rel: scalar-relativistic; mpse: `losses.Poles` of the material (many-pole
    self-energy); quinn: Quinn's low-energy loss with the HL self-energy."""
    t0 = time.time()
    pts = CL.lattice_cluster(d_nn, radius, lattice, c_over_a)
    species = CL.alloy_species(pts, absorber, list(elements), d_nn, composition, seed)
    cpot = (SCF.build(pts, species, verbose=verbose, relativistic=rel).potential if scf
            else PO.build(pts, species, relativistic=rel))
    gam = GAMMA_K.get(absorber, 1.0) if gamma_ev is None else gamma_ev
    ph = PS.phase_shifts(cpot, k_grid, gamma_ev=gam, model=model, quinn=quinn, mpse=mpse)
    sym_type = {t.symbol: i for i, t in enumerate(cpot.types) if i > 0}
    site_types = np.array([0] + [sym_type[s] for s in species[1:]])
    rp = rpath if rpath is not None else d_nn + 0.2
    groups = PA.enumerate_paths(pts, site_types, rp, nleg)
    tr = PSM.path_sum([g.path for g in groups], ph, precision=precision)
    if verbose:
        print(f"  {absorber}: {len(pts)} atoms, {len(groups)} paths, "
              f"{time.time() - t0:.1f}s; Rmt {cpot.types[0].r_mt * BOHR:.3f} A, "
              f"mu {cpot.mu * HARTREE:.2f} eV, rs_int {cpot.rs_int:.3f}")
    return dict(pts=pts, species=species, cpot=cpot, ph=ph, groups=groups, trace=tr,
                gamma_ev=gam, model=model)


def build_set(name, elements, d_nn, lattice="fcc", composition=None, radius=6.5,
              rpath=None, nleg=2, seed=0, k_grid=None, gamma_ev=None, model="hl",
              c_over_a=None, out_root=None, library=False, scf=False, rel=False,
              sigma="hl", quinn=False):
    """Write data/<name>/<absorber>/feffNNNN.dat, paths.dat and meta.json."""
    elements = list(elements)
    comp = (np.ones(len(elements)) / len(elements) if composition is None
            else np.asarray(composition, float))
    comp = comp / comp.sum()
    root = out_root or os.path.join(DATA_DIR, name)
    os.makedirs(root, exist_ok=True)
    mp = None
    if sigma == "mpse":
        from . import losses as LS
        mp = LS.material_poles(elements, comp, lattice, d_nn, c_over_a)
    for a in elements:
        res = build_absorber(a, elements, d_nn, lattice, comp, radius, rpath, nleg, seed,
                             k_grid, None if gamma_ev is None else gamma_ev.get(a), model, c_over_a,
                             scf=scf, rel=rel, mpse=mp, quinn=quinn)
        d = os.path.join(root, a)
        os.makedirs(d, exist_ok=True)
        for fn in os.listdir(d):
            if fn.startswith("feff0") and fn.endswith(".dat"):
                os.remove(os.path.join(d, fn))
        title = f"{lattice} {'-'.join(elements)} alloy, {a} absorber, d_nn={d_nn}"
        plines = [f"PATH  Rmax= {rpath if rpath else d_nn + 0.2:.3f},  native, all paths kept"]
        for i, g in enumerate(res["groups"]):
            cols = feff_columns(res["trace"][i], res["ph"], g.reff_A)
            _write_feffdat(os.path.join(d, f"feff{i + 1:04d}.dat"), title, res["cpot"], g, cols,
                           res["species"], res["gamma_ev"],
                           model + ("+many-pole" if mp is not None else "") + ("+Quinn" if quinn else ""))
            plines.append(f"  {i + 1:4d}  {g.nleg:2d}  {g.degeneracy:7.3f}  index, nleg, degeneracy, r= {g.reff_A:8.4f}")
        with open(os.path.join(d, "paths.dat"), "w") as f:
            f.write("\n".join(plines) + "\n")
        with open(os.path.join(d, "cluster.xyz"), "w") as f:
            f.write(f"{len(res['pts'])}\n{title}\n")
            for s, p in zip(res["species"], res["pts"]):
                f.write(f"{s} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
    meta = dict(elements=elements, composition=comp.tolist(), lattice=lattice,
                c_over_a=c_over_a, d_nn=d_nn, radius=radius, rpath=rpath, nleg=nleg,
                engine="chimera-native", exchange=model, seed=seed, scf=bool(scf),
                relativistic=bool(rel), sigma=sigma, quinn=bool(quinn))
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return root


def main(argv=None):
    ap = argparse.ArgumentParser(description="Native scattering tables (no FEFF needed)")
    ap.add_argument("--elements", nargs="+", default=["Cr", "Co", "Ni"])
    ap.add_argument("--composition", type=float, nargs="+", default=None)
    ap.add_argument("--lattice", default="fcc", choices=["sc", "bcc", "fcc", "hcp"])
    ap.add_argument("--c-over-a", type=float, default=None)
    ap.add_argument("--d-nn", type=float, default=2.517)
    ap.add_argument("--radius", type=float, default=6.5)
    ap.add_argument("--name", default="crconi_fcc2517_native")
    ap.add_argument("--rpath", type=float, default=None,
                    help="max half-path length (A); default = first shell only")
    ap.add_argument("--nleg", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exchange", default="hl", choices=["hl", "dh", "gs"])
    ap.add_argument("--kmax", type=float, default=20.0)
    ap.add_argument("--scf", action="store_true",
                    help="self-consistent potentials (cluster Green's function) instead of "
                         "overlapped free atoms")
    ap.add_argument("--rel", action="store_true",
                    help="scalar-relativistic atoms, potentials and scattering states")
    ap.add_argument("--sigma", default="hl", choices=["hl", "mpse"],
                    help="self-energy: Hedin-Lundqvist, or the many-pole model of the measured "
                         "loss function")
    ap.add_argument("--quinn", action="store_true",
                    help="Quinn's electron-hole loss below the plasmon (HL only)")
    args = ap.parse_args(argv)
    t = time.time()
    root = build_set(args.name, args.elements, args.d_nn, args.lattice, args.composition,
                     args.radius, args.rpath, args.nleg, args.seed,
                     PS.default_k_grid(args.kmax), None, args.exchange, args.c_over_a,
                     scf=args.scf, rel=args.rel, sigma=args.sigma, quinn=args.quinn)
    print(f"wrote {root} in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
