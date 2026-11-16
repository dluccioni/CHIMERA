"""Generate FEFF8L single-scattering tables for an alloy, one run per absorber.

Needs a Python with xraylarch installed (it bundles the FEFF8L binaries);
pass it as --larch-python. The main environment never imports larch: the
outputs (feffNNNN.dat) are copied into data/<name>/ and
read back by `exafs_gpu.scattering.feff8l_tables`, which needs nothing but
numpy.

Cluster: the chosen lattice (sc, bcc, fcc or hcp) at nearest-neighbour
distance d, radius ~6.5 A (~87 atoms for fcc). The absorber sits at the
origin as potential 0; every other atom carries one of the alloy species as
potentials 1..n. The first shell holds every species (so every (absorber,
scatterer) single-scattering path exists at R = d and the SCF potentials see
the alloy environment), filled at the composition with at least one of each;
the rest of the cluster is a fixed pseudo-random arrangement at the
composition. With RPATH just past d only the nearest-neighbour paths are
generated, one per scatterer.

    python feff8l_tables.py --larch-python <venv>/Scripts/python.exe \\
        --elements Fe Ni --d-nn 2.538 --name feni_fcc2538
    python feff8l_tables.py --larch-python <venv>/Scripts/python.exe \\
        --elements Fe Ni --d-nn 2.538 --name feni_fcc2538_ms --rpath 5.3 --nleg 4
    python feff8l_tables.py --larch-python <venv>/Scripts/python.exe \\
        --elements Fe Cr --lattice bcc --d-nn 2.485 --name fecr_bcc2485

`--d-nn` is the nearest-neighbour distance of the reference cell (for fcc,
a0 / sqrt(2)); the fit rescales path lengths from it to the fitted a0.

`meta.json` next to the tables records the elements, composition, lattice
and spacing, which `chimera.system.AlloySystem.from_tables` reads back.
"""
from __future__ import annotations
import os, sys, json, argparse, subprocess, shutil, textwrap
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exafs_gpu.scattering import Z_TABLE as Z
from exafs_gpu.lattice import LATTICES

A0_PER_NN = {"fcc": np.sqrt(2.0), "bcc": 2.0 / np.sqrt(3.0), "sc": 1.0, "hcp": 1.0}

RUNNER = textwrap.dedent("""
    import sys
    from larch.xafs.feffrunner import feff8l
    feff8l(folder=sys.argv[1], feffinp="feff.inp", verbose=False)
""")


def cluster(d_nn, radius, lattice="fcc", c_over_a=None):
    """Atom positions of the lattice within `radius` of an origin site,
    sorted by distance; the first entry is the origin."""
    spec = LATTICES[lattice]
    a = d_nn * A0_PER_NN[lattice]
    if spec["hex"]:
        ca = c_over_a if c_over_a is not None else np.sqrt(8.0 / 3.0)
        A = np.array([[1.0, 0.0, 0.0],
                      [-0.5, np.sqrt(3) / 2, 0.0],
                      [0.0, 0.0, ca]]) * a
    else:
        A = np.eye(3) * a
    basis = np.asarray(spec["basis"], float)
    n = int(np.ceil(radius / a)) + 2
    pts = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            for k in range(-n, n + 1):
                for b in basis:
                    p = (np.array([i, j, k]) + b) @ A
                    if np.linalg.norm(p) < radius:
                        pts.append(p)
    pts = np.array(sorted(pts, key=lambda p: (round(np.linalg.norm(p), 4), tuple(np.round(p, 4)))))
    assert np.linalg.norm(pts[0]) < 1e-9, "the origin must be a lattice site"
    return pts


def fcc_cluster(d_nn, radius):
    """Kept for callers of the original name."""
    return cluster(d_nn, radius, "fcc")


def _fill(n_sites, elements, composition, rng, at_least_one=False):
    """Species indices (1-based potentials) for n sites at the composition."""
    n = len(elements)
    comp = np.asarray(composition, float); comp = comp / comp.sum()
    cnt = np.floor(comp * n_sites).astype(int)
    if at_least_one:
        cnt = np.maximum(cnt, 1)
    while cnt.sum() < n_sites:
        cnt[np.argmax(comp * n_sites - cnt)] += 1
    while cnt.sum() > n_sites:
        cnt[np.argmax(cnt - comp * n_sites)] -= 1
    pots = np.repeat(np.arange(1, n + 1), cnt)
    rng.shuffle(pots)
    return pots


def write_input(folder, absorber, elements, pts, d_nn, seed=0, rpath=None, nleg=2,
                composition=None, lattice="fcc"):
    n = len(elements)
    comp = np.ones(n) / n if composition is None else np.asarray(composition, float)
    r = np.linalg.norm(pts, axis=1)
    rng = np.random.default_rng(seed)
    ipot = np.zeros(len(pts), int)
    # the first shell, with a 3% tolerance so that an hcp cell off the ideal
    # c/a - whose 12 neighbours split into 6 + 6 at slightly different
    # distances - still counts all twelve
    shell1 = np.where((r > 1e-6) & (np.abs(r - d_nn) < 0.03 * d_nn))[0]
    ipot[shell1] = _fill(len(shell1), elements, comp, rng, at_least_one=True)
    rest = np.where((r > d_nn * 1.03))[0]
    ipot[rest] = _fill(len(rest), elements, comp, rng)
    rp = rpath if rpath is not None else d_nn + 0.2
    lines = [f"TITLE {lattice} {'-'.join(elements)} alloy, {absorber} absorber, d_nn={d_nn}",
             "EDGE K", "S02 1.0", "CONTROL 1 1 1 1 1 1", "PRINT 1 0 0 0 0 3",
             "EXCHANGE 0 0.0 0.0", "SCF 4.5", f"RPATH {rp:.3f}", f"NLEG {nleg}",
             "CRITERIA 0 0", "", "POTENTIALS", f"0 {Z[absorber]} {absorber}"]
    for i, e in enumerate(elements):
        lines.append(f"{i + 1} {Z[e]} {e}")
    lines += ["", "ATOMS"]
    for p, ip in zip(pts, ipot):
        e = absorber if ip == 0 else elements[ip - 1]
        lines.append(f"{p[0]:10.5f} {p[1]:10.5f} {p[2]:10.5f} {ip} {e}")
    lines.append("END")
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, "feff.inp"), "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--larch-python", required=True)
    ap.add_argument("--elements", nargs="+", required=True,
                    help="absorbing elements, in the order the fit will use")
    ap.add_argument("--composition", type=float, nargs="+", default=None,
                    help="atomic fractions in element order (default equiatomic)")
    ap.add_argument("--lattice", default="fcc", choices=list(LATTICES))
    ap.add_argument("--c-over-a", type=float, default=None, help="hcp only")
    ap.add_argument("--d-nn", type=float, required=True,
                    help="nearest-neighbour distance of the reference cell (A)")
    ap.add_argument("--radius", type=float, default=6.5)
    ap.add_argument("--name", required=True, help="the set is written to data/<name>/")
    ap.add_argument("--rpath", type=float, default=None,
                    help="max half-path length (A); default = first shell only")
    ap.add_argument("--nleg", type=int, default=2,
                    help="max legs per path; >2 turns on multiple scattering")
    ap.add_argument("--workdir", default=os.path.join(HERE, "results", "feff8l_runs"))
    args = ap.parse_args()
    for e in args.elements:
        if e not in Z:
            sys.exit(f"unknown element {e!r}")
    comp = (np.ones(len(args.elements)) / len(args.elements) if args.composition is None
            else np.asarray(args.composition, float))
    assert comp.size == len(args.elements), "one fraction per element"
    comp = comp / comp.sum()

    out_root = os.path.join(HERE, "data", args.name)
    pts = cluster(args.d_nn, args.radius, args.lattice, args.c_over_a)
    print(f"{len(pts)} atoms in the {args.lattice} cluster")
    runner = os.path.join(args.workdir, "_run_feff8l.py")
    os.makedirs(args.workdir, exist_ok=True)
    open(runner, "w").write(RUNNER)
    for absorber in args.elements:
        folder = os.path.join(args.workdir, absorber)
        write_input(folder, absorber, args.elements, pts, args.d_nn,
                    rpath=args.rpath, nleg=args.nleg, composition=comp,
                    lattice=args.lattice)
        print(f"running FEFF8L for absorber {absorber} ...", flush=True)
        # the FEFF8L binaries fail on an absolute Windows path with spaces;
        # hand larch a relative path instead
        subprocess.run([args.larch_python, os.path.relpath(runner, HERE),
                        os.path.relpath(folder, HERE)], check=True, cwd=HERE)
        dst = os.path.join(out_root, absorber)
        os.makedirs(dst, exist_ok=True)
        for fn in os.listdir(folder):
            if fn.startswith("feff0") and fn.endswith(".dat") or fn in ("feff.inp", "paths.dat"):
                shutil.copy(os.path.join(folder, fn), os.path.join(dst, fn))
        got = sorted(f for f in os.listdir(dst) if f.startswith("feff0"))
        print(f"  -> {dst}: {got}")
    json.dump(dict(elements=args.elements, composition=comp.tolist(),
                   lattice=args.lattice, c_over_a=args.c_over_a,
                   d_nn=args.d_nn, radius=args.radius,
                   natoms=len(pts), rpath=args.rpath, nleg=args.nleg),
              open(os.path.join(out_root, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
