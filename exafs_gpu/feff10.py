"""Run FEFF10 (the University of Washington / TIMES distribution) from Python.

FEFF10 ships one executable per module; a calculation is those modules run in
order inside a folder holding `feff.inp` (what the distribution's feff.bat
does, minus its `cd` to the home folder). `run` does that and checks each
module's exit; `write_input` writes an EXAFS input for a cluster with the
physics chosen by `Settings`.

    from exafs_gpu import feff10
    s = feff10.Settings(corehole="RPA", mpse="opcons")
    feff10.write_input(folder, positions, species, s, rpath=6.0, nleg=4)
    feff10.run(folder)

The binaries are found in $FEFF10_BIN, else N:/Applications/JFEFF_FEFF10/feff10/bin.
Results are the usual feffNNNN.dat path files, read by
`exafs_gpu.scattering.read_feffdat`.

Paths of one's own choosing (FEFF10 users guide 4.3.1): run once with
CONTROL ... so pot and xsph are done, write `paths.dat` with `write_paths`,
then `run(folder, modules=("genfmt", "ff2x"))`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field

import numpy as np

from .scattering import Z_TABLE

BIN_DEFAULT = r"N:\Applications\JFEFF_FEFF10\feff10\bin"
# the order of feff.bat; modules a card set does not use exit at once
MODULES = ("rdinp", "dmdw", "atomic", "pot", "ldos", "screen", "crpa", "opconsat",
           "xsph", "fms", "mkgtr", "path", "genfmt", "ff2x", "sfconv", "compton",
           "eels", "rhorrp")


def bin_dir():
    d = os.environ.get("FEFF10_BIN", BIN_DEFAULT)
    if not os.path.isfile(os.path.join(d, "rdinp.exe")) and not os.path.isfile(os.path.join(d, "rdinp")):
        raise FileNotFoundError(f"FEFF10 binaries not found in {d} (set FEFF10_BIN)")
    return d


@dataclass
class Settings:
    """The physics of an EXAFS run.

    corehole : "FSR" (final-state rule, screened core hole: FEFF8's default),
               "RPA" (linear-response screened core hole), "NONE"
    scf      : self-consistency radius (A), None for overlapped atoms
    mpse     : None (Hedin-Lundqvist plasmon pole), "opcons" (many-pole
               self-energy from FEFF's atomic loss-function estimate), or a
               path to a loss.dat (many-pole from that loss function)
    exchange : FEFF EXCHANGE index (0 Hedin-Lundqvist)
    s02      : S02 card (1.0: leave the amplitude to the fit)
    extra    : further cards, verbatim
    """
    corehole: str = "FSR"
    scf: float | None = 4.5
    mpse: str | None = None
    exchange: int = 0
    s02: float = 1.0
    extra: list = field(default_factory=list)

    def tag(self):
        m = "hl" if self.mpse is None else ("mpse-opcons" if self.mpse == "opcons" else "mpse-loss")
        return f"{self.corehole.lower()}-{m}" + ("" if self.scf else "-noscf")


def write_input(folder, positions, species, settings: Settings, rpath=6.0, nleg=4,
                title="CHIMERA FEFF10 run", criteria=(0.0, 0.0), print_level=(1, 0, 0, 0, 0, 3),
                potentials=None):
    """feff.inp for an absorber at positions[0]. `potentials` maps each
    species to its unique-potential index (default: by order of first
    appearance after the absorber, 1, 2, ...)."""
    os.makedirs(folder, exist_ok=True)
    pos = np.asarray(positions, float) - np.asarray(positions, float)[0]
    species = list(species)
    if potentials is None:
        potentials = {}
        for s in species[1:]:
            potentials.setdefault(s, len(potentials) + 1)
    s = settings
    L = [f"TITLE {title}", "EDGE K", f"S02 {s.s02}", "CONTROL 1 1 1 1 1 1",
         "PRINT " + " ".join(str(x) for x in print_level),
         f"COREHOLE {s.corehole}", f"EXCHANGE {s.exchange} 0.0 0.0 0",
         f"RPATH {rpath}", f"NLEG {nleg}", f"CRITERIA {criteria[0]} {criteria[1]}"]
    if s.scf:
        L.append(f"SCF {s.scf}")
    if s.mpse is not None:
        if s.mpse == "opcons":
            L.append("OPCONS")
        else:
            shutil.copy(s.mpse, os.path.join(folder, "loss.dat"))
        L.append("MPSE 2")
    L += list(s.extra)
    L += ["", "POTENTIALS", f"0 {Z_TABLE[species[0]]} {species[0]}"]
    for sym, ip in sorted(potentials.items(), key=lambda t: t[1]):
        L.append(f"{ip} {Z_TABLE[sym]} {sym}")
    L += ["", "ATOMS"]
    for i, (p, sym) in enumerate(zip(pos, species)):
        L.append(f"  {p[0]:11.6f} {p[1]:11.6f} {p[2]:11.6f} {0 if i == 0 else potentials[sym]} {sym}")
    L.append("END")
    with open(os.path.join(folder, "feff.inp"), "w") as f:
        f.write("\n".join(L) + "\n")
    return potentials


def run(folder, modules=MODULES, timeout=3600, log=True):
    """Run the FEFF10 modules in `folder`; raises if one fails."""
    d = bin_dir()
    out = []
    for m in modules:
        exe = os.path.join(d, m + (".exe" if os.name == "nt" else ""))
        r = subprocess.run([exe], cwd=folder, capture_output=True, text=True, timeout=timeout)
        out.append(f"== {m} (exit {r.returncode})\n{r.stdout[-3000:]}{r.stderr[-2000:]}")
        if r.returncode != 0:
            open(os.path.join(folder, "chimera_feff10.log"), "w").write("\n".join(out))
            raise RuntimeError(f"FEFF10 {m} failed in {folder}: {r.stderr[-500:] or r.stdout[-500:]}")
    if log:
        with open(os.path.join(folder, "chimera_feff10.log"), "w") as f:
            f.write("\n".join(out))
    return folder


def path_files(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.startswith("feff") and f.endswith(".dat") and f[4:8].isdigit())
