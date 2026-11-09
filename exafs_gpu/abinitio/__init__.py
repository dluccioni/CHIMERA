"""Native ab initio EXAFS scattering: CHIMERA's own replacement for FEFF8L.

The chain is the one FEFF follows, written from the physics rather than
ported:

    atom.py        self-consistent LDA atoms (ground state and core-hole)
    potential.py   overlapped-atom muffin-tin potentials for a cluster,
                   Norman and muffin-tin radii, interstitial level
    selfenergy.py  Hedin-Lundqvist plasmon-pole self-energy of the electron
                   gas (energy-dependent exchange-correlation, mean free path)
    phaseshift.py  complex scattering phase shifts (CUDA, one thread per
                   potential x energy x l)
    paths.py       path enumeration with degeneracies and species
    pathsum.py     exact curved-wave multiple scattering in the angular-
                   momentum basis, no separable or small-atom approximation
                   (CUDA)
    tables.py      builds a table set and writes feffNNNN.dat-format files, so
                   everything downstream reads them exactly as it reads FEFF's

Hartree atomic units inside; Angstrom and eV at the interfaces.
"""
