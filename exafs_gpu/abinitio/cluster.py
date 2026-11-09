"""Alloy clusters for table generation: the lattice around an absorber, filled
with species at a composition.

The construction is the one `feff8l_tables.py` uses (same ordering, same
random fill for a given seed), so a native table set and a FEFF8L set built
with the same arguments describe the same cluster and can be compared path
by path.
"""
from __future__ import annotations

import numpy as np

from ..lattice import LATTICES

A0_PER_NN = {"fcc": np.sqrt(2.0), "bcc": 2.0 / np.sqrt(3.0), "sc": 1.0, "hcp": 1.0}


def lattice_cluster(d_nn, radius, lattice="fcc", c_over_a=None):
    """Positions (Angstrom) of the lattice within `radius` of an origin site,
    sorted by distance; the first entry is the origin."""
    spec = LATTICES[lattice]
    a = d_nn * A0_PER_NN[lattice]
    if spec["hex"]:
        ca = c_over_a if c_over_a is not None else np.sqrt(8.0 / 3.0)
        A = np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, ca]]) * a
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


def _fill(n_sites, n_species, composition, rng, at_least_one=False):
    comp = np.asarray(composition, float)
    comp = comp / comp.sum()
    cnt = np.floor(comp * n_sites).astype(int)
    if at_least_one:
        cnt = np.maximum(cnt, 1)
    while cnt.sum() < n_sites:
        cnt[np.argmax(comp * n_sites - cnt)] += 1
    while cnt.sum() > n_sites:
        cnt[np.argmax(cnt - comp * n_sites)] -= 1
    pots = np.repeat(np.arange(1, n_species + 1), cnt)
    rng.shuffle(pots)
    return pots


def alloy_species(pts, absorber, elements, d_nn, composition=None, seed=0):
    """Species per site: the absorber at the origin, every species present in
    the first shell (so each absorber-scatterer pair has a nearest-neighbour
    path), the rest a fixed random fill at the composition."""
    n = len(elements)
    comp = np.ones(n) / n if composition is None else np.asarray(composition, float)
    r = np.linalg.norm(pts, axis=1)
    rng = np.random.default_rng(seed)
    ipot = np.zeros(len(pts), int)
    shell1 = np.where((r > 1e-6) & (np.abs(r - d_nn) < 0.03 * d_nn))[0]
    ipot[shell1] = _fill(len(shell1), n, comp, rng, at_least_one=True)
    rest = np.where(r > d_nn * 1.03)[0]
    ipot[rest] = _fill(len(rest), n, comp, rng)
    species = [absorber] + [elements[i - 1] for i in ipot[1:]]
    return species
