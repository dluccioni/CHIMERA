"""FEFF10 versions of the FEFF8L alloy table sets.

Each absorber's run reuses the cluster of the FEFF8L set it mirrors - the
same atoms, potentials, RPATH and NLEG, read from that set's feff.inp - and
changes only the physics cards, so FEFF8L and FEFF10 tables differ in the
theory and in nothing else:

    python feff10_tables.py --from crconi_fcc2517    --tag rpa_mpse
    python feff10_tables.py --from crconi_fcc2517_ms --tag rpa_mpse

writes data/crconi_fcc2517_feff10_rpa_mpse[_ms]/<absorber>/feffNNNN.dat with a
meta.json, readable wherever a FEFF8L set is (`--tables` / `--ms-tables`).

Tags (exafs_gpu.feff10.Settings): fsr (FEFF8L's physics), rpa, fsr_mpse,
rpa_mpse, none. The binaries come from $FEFF10_BIN or the default install.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exafs_gpu import feff10  # noqa: E402

DATA = os.path.join(HERE, "data")
TAGS = {
    "fsr": feff10.Settings(corehole="FSR"),
    "rpa": feff10.Settings(corehole="RPA"),
    "none": feff10.Settings(corehole="NONE"),
    "fsr_mpse": feff10.Settings(corehole="FSR", mpse="opcons"),
    "rpa_mpse": feff10.Settings(corehole="RPA", mpse="opcons"),
}
_PHYSICS = ("COREHOLE", "EXCHANGE", "SCF", "MPSE", "OPCONS", "S02", "CONTROL", "PRINT", "SFCONV")


def convert_input(src, dst_folder, settings: feff10.Settings):
    """The FEFF8L feff.inp with its physics cards replaced by `settings`."""
    lines = open(src).read().split("\n")
    head, rest, in_head = [], [], True
    for ln in lines:
        key = ln.strip().split()[0].upper() if ln.strip() else ""
        if key in ("POTENTIALS", "ATOMS"):
            in_head = False
        if in_head:
            if key not in _PHYSICS:
                head.append(ln)
        else:
            rest.append(ln)
    s = settings
    cards = ["S02 1.0", "CONTROL 1 1 1 1 1 1", "PRINT 1 0 0 0 0 3",
             f"COREHOLE {s.corehole}", f"EXCHANGE {s.exchange} 0.0 0.0 0"]
    if s.scf:
        cards.append(f"SCF {s.scf}")
    if s.mpse is not None:
        if s.mpse == "opcons":
            cards.append("OPCONS")
        else:
            shutil.copy(s.mpse, os.path.join(dst_folder, "loss.dat"))
        cards.append("MPSE 2")
    os.makedirs(dst_folder, exist_ok=True)
    with open(os.path.join(dst_folder, "feff.inp"), "w") as f:
        f.write("\n".join(head + cards + [""] + rest) + "\n")


def build(src_name, tag, elements=None, workers=3):
    src = os.path.join(DATA, src_name)
    meta = json.load(open(os.path.join(src, "meta.json")))
    elements = elements or meta["elements"]
    ms = src_name.endswith("_ms")
    base = src_name[:-3] if ms else src_name
    name = f"{base}_feff10_{tag}" + ("_ms" if ms else "")
    dst = os.path.join(DATA, name)
    settings = TAGS[tag]

    def one(a):
        t0 = time.time()
        fd = os.path.join(dst, a)
        if os.path.isdir(fd):
            for fn in os.listdir(fd):
                if fn.startswith("feff0") and fn.endswith(".dat"):
                    os.remove(os.path.join(fd, fn))
        convert_input(os.path.join(src, a, "feff.inp"), fd, settings)
        feff10.run(fd)
        return a, len(feff10.path_files(fd)), time.time() - t0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for a, n, dt in ex.map(one, elements):
            print(f"  {name}/{a}: {n} paths, {dt:.0f}s", flush=True)
    meta.update(engine="FEFF10", feff10_settings=dict(tag=tag, corehole=settings.corehole,
                                                      scf=settings.scf, mpse=settings.mpse,
                                                      exchange=settings.exchange),
                mirrors=src_name)
    with open(os.path.join(dst, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src", nargs="+", default=["crconi_fcc2517", "crconi_fcc2517_ms"])
    ap.add_argument("--tag", nargs="+", default=["rpa_mpse"], choices=sorted(TAGS))
    a = ap.parse_args()
    for tag in a.tag:
        for s in a.src:
            print(f"{s} -> FEFF10 [{tag}]", flush=True)
            print("  wrote", build(s, tag), flush=True)


if __name__ == "__main__":
    main()
