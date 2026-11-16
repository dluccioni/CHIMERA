"""Export chi(k) and the processing parameters from Athena project files.

    python export_athena.py --larch-python <venv>/Scripts/python.exe \
        --element Fe [--point 300 --unit K] --out <data root> project.prj [more.prj ...]

Run it once per edge (and series point). Writes, in the layout
`chimera.dataio.screen` reads:

    <out>/<Element>_<P><unit>/<group>_sample<N>.csv  columns k, chi
    <out>/<Element>_<P><unit>/processing.json        per group: e0, edge step,
                                                     rbkg, the pre-edge /
                                                     normalisation / autobk /
                                                     FFT parameters Athena
                                                     stored, and the source
                                                     project

Without `--point` the folder is `<out>/<Element>/`, a single measurement.
`N` is taken from trailing digits of the group label (`sample12`, `pos_12`,
`12`), else from the group's position in the project. Groups that hold no
mu(E) (merged references, say) are skipped and listed.

The main environment never imports larch. Reading a `.prj` needs it, so the
read runs in the interpreter given by --larch-python (the same one
feff8l_tables.py uses for FEFF8L), which dumps one JSON per project into a
work directory; this script then writes the CSVs. Athena's own background
removal and normalisation are applied by `read_athena(do_bkg=True)`, so the
chi(k) written here is what Athena would export from that project.

Athena projects differ in what they store, so the runner reads attributes
defensively and dumps whatever it finds under `athena_params`: a project
that names things differently still yields its chi(k) plus a parameter dump
to read by eye. Check the first export against Athena's own chi(k).
"""
from __future__ import annotations
import os, sys, re, json, argparse, subprocess, textwrap
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

RUNNER = textwrap.dedent('''
    """Runs inside the larch interpreter: .prj -> JSON with k, chi per group."""
    import sys, json
    import numpy as np
    from larch.io import read_athena

    def plain(obj, depth=0):
        """JSON-able copy of a larch Group / array / scalar tree."""
        if depth > 4:
            return None
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, np.ndarray):
            return obj.tolist() if obj.size <= 16 else f"array{obj.shape}"
        if isinstance(obj, (list, tuple)):
            return [plain(x, depth + 1) for x in obj]
        if isinstance(obj, dict):
            return {str(k): plain(v, depth + 1) for k, v in obj.items()}
        if hasattr(obj, "__dict__"):
            return {k: plain(v, depth + 1) for k, v in vars(obj).items()
                    if not k.startswith("_") and not callable(v)}
        return str(obj)

    prj, out = sys.argv[1], sys.argv[2]
    p = read_athena(prj, do_preedge=True, do_bkg=True, do_fft=False)
    groups, skipped = [], []
    for name, g in vars(p).items():
        if name.startswith("_"):
            continue
        if not (hasattr(g, "energy") and hasattr(g, "mu")):
            skipped.append(name)
            continue
        k, chi = getattr(g, "k", None), getattr(g, "chi", None)
        if k is None or chi is None:
            skipped.append(name)
            continue
        rec = dict(name=name, label=str(getattr(g, "label", name)),
                   k=np.asarray(k, float).tolist(), chi=np.asarray(chi, float).tolist(),
                   e0=plain(getattr(g, "e0", None)),
                   edge_step=plain(getattr(g, "edge_step", None)),
                   rbkg=plain(getattr(getattr(g, "autobk_details", None), "rbkg", None)),
                   athena_params=plain(getattr(g, "athena_params", None)),
                   pre_edge_details=plain(getattr(g, "pre_edge_details", None)),
                   autobk_details=plain(getattr(g, "autobk_details", None)))
        groups.append(rec)
    json.dump(dict(project=prj, groups=groups, skipped=skipped), open(out, "w"))
    print(f"{prj}: {len(groups)} groups with chi(k), {len(skipped)} skipped")
''')


def sample_index(label, fallback):
    m = re.search(r"(\d+)\s*$", label)
    return int(m.group(1)) if m else int(fallback)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("projects", nargs="+", help="Athena .prj files")
    ap.add_argument("--larch-python", required=True)
    ap.add_argument("--element", required=True, help="the absorbing element of this edge")
    ap.add_argument("--point", type=int, default=None,
                    help="series index (a pressure, temperature, ...); omit for a "
                         "single measurement")
    ap.add_argument("--unit", default="", help="series unit for the folder name, e.g. GPa or K")
    ap.add_argument("--out", required=True, help="the data root")
    ap.add_argument("--workdir", default=os.path.join(HERE, "results", "athena_export"))
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    runner = os.path.join(args.workdir, "_run_read_athena.py")
    open(runner, "w").write(RUNNER)
    name = args.element if args.point is None else f"{args.element}_{args.point}{args.unit}"
    folder = os.path.join(args.out, name)
    os.makedirs(folder, exist_ok=True)

    processing, n_written, used = {}, 0, set()
    for i, prj in enumerate(args.projects):
        dump = os.path.join(args.workdir, f"_prj{i}.json")
        subprocess.run([args.larch_python, os.path.relpath(runner, HERE),
                        os.path.abspath(prj), os.path.abspath(dump)],
                       check=True, cwd=HERE)
        d = json.load(open(dump))
        if d["skipped"]:
            print(f"  skipped (no chi(k)): {d['skipped']}")
        for j, g in enumerate(d["groups"]):
            n = sample_index(g["label"], n_written)
            while n in used:            # two groups claiming one index
                n += 1000
            used.add(n)
            base = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(os.path.basename(prj))[0])
            fn = os.path.join(folder, f"{base}_{g['name']}_sample{n}.csv")
            k, chi = np.asarray(g["k"]), np.asarray(g["chi"])
            np.savetxt(fn, np.column_stack([k, chi]), delimiter=",", header="k,chi",
                       comments="", fmt="%.18e")
            processing[f"sample{n}"] = {kk: v for kk, v in g.items() if kk not in ("k", "chi")}
            processing[f"sample{n}"].update(project=prj, file=os.path.basename(fn),
                                            k_range=[float(k.min()), float(k.max())],
                                            k_step=float(np.median(np.diff(k))))
            n_written += 1
    json.dump(processing, open(os.path.join(folder, "processing.json"), "w"), indent=1)
    print(f"{n_written} spectra -> {folder}")


if __name__ == "__main__":
    main()
