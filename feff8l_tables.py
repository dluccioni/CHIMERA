"""Generate FEFF8L single-scattering tables for an fcc alloy, one run per absorber.

Needs a Python with xraylarch installed (it bundles the FEFF8L binaries);
pass it as --larch-python. The main environment never imports larch: the
outputs (feffNNNN.dat) are copied into data/<name>/ and
read back by `exafs_gpu.scattering.feff8l_tables`, which needs nothing but
numpy.

Cluster: fcc, nearest-neighbour distance d, radius ~6.5 A (~87 atoms). The
absorber sits at the origin as potential 0; every other atom carries one of
the alloy species as potentials 1..n. The first shell is filled with equal
numbers of each species (so every (absorber, scatterer) single-scattering path
exists at R = d and the SCF potentials see the alloy environment); the rest of
the cluster is a fixed pseudo-random equiatomic arrangement. With RPATH just
past d only the nearest-neighbour paths are generated, one per scatterer.

    python feff8l_tables.py --larch-python <venv>/Scripts/python.exe
"""
from __future__ import annotations
import os, sys, json, argparse, subprocess, shutil, textwrap
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Z = {"Cr": 24, "Co": 27, "Ni": 28, "Fe": 26, "Mn": 25, "Cu": 29, "V": 23, "Ti": 22}

RUNNER = textwrap.dedent("""
    import sys
    from larch.xafs.feffrunner import feff8l
    feff8l(folder=sys.argv[1], feffinp="feff.inp", verbose=False)
""")


def fcc_cluster(d_nn, radius):
    a = d_nn * np.sqrt(2.0)
    basis = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                for b in basis:
                    p = (np.array([i, j, k]) + b) * a
                    if np.linalg.norm(p) < radius:
                        pts.append(p)
    pts = np.array(sorted(pts, key=lambda p: (round(np.linalg.norm(p), 4), tuple(np.round(p, 4)))))
    return pts


def write_input(folder, absorber, elements, pts, d_nn, seed=0, rpath=None, nleg=2):
    n = len(elements)
    r = np.linalg.norm(pts, axis=1)
    rng = np.random.default_rng(seed)
    ipot = np.zeros(len(pts), int)
    shell1 = np.where(np.abs(r - d_nn) < 1e-3)[0]
    shell1_pots = np.tile(np.arange(1, n + 1), len(shell1) // n + 1)[:len(shell1)]
    rng.shuffle(shell1_pots)
    ipot[shell1] = shell1_pots
    rest = np.where((r > d_nn + 1e-3))[0]
    rest_pots = np.tile(np.arange(1, n + 1), len(rest) // n + 1)[:len(rest)]
    rng.shuffle(rest_pots)
    ipot[rest] = rest_pots
    rp = rpath if rpath is not None else d_nn + 0.2
    lines = [f"TITLE fcc {'-'.join(elements)} alloy, {absorber} absorber, d_nn={d_nn}",
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
    ap.add_argument("--elements", nargs="+", default=["Cr", "Co", "Ni"])
    ap.add_argument("--d-nn", type=float, default=2.517)
    ap.add_argument("--radius", type=float, default=6.5)
    ap.add_argument("--name", default="crconi_fcc2517")
    ap.add_argument("--rpath", type=float, default=None,
                    help="max half-path length (A); default = first shell only")
    ap.add_argument("--nleg", type=int, default=2,
                    help="max legs per path; >2 turns on multiple scattering")
    ap.add_argument("--workdir", default=os.path.join(HERE, "results", "feff8l_runs"))
    args = ap.parse_args()

    out_root = os.path.join(HERE, "data", args.name)
    pts = fcc_cluster(args.d_nn, args.radius)
    print(f"{len(pts)} atoms in the cluster")
    runner = os.path.join(args.workdir, "_run_feff8l.py")
    os.makedirs(args.workdir, exist_ok=True)
    open(runner, "w").write(RUNNER)
    for absorber in args.elements:
        folder = os.path.join(args.workdir, absorber)
        write_input(folder, absorber, args.elements, pts, args.d_nn,
                    rpath=args.rpath, nleg=args.nleg)
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
    json.dump(dict(elements=args.elements, d_nn=args.d_nn, radius=args.radius,
                   natoms=len(pts), rpath=args.rpath, nleg=args.nleg),
              open(os.path.join(out_root, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
