"""Scattering-path enumeration with degeneracies.

Every sequence absorber -> s_1 -> ... -> s_N -> absorber through the atoms of
a cluster, with no atom repeated consecutively and a half path length up to
`rmax`, is generated (the absorber may appear in the middle of a path, as in
FEFF). Two sequences are degenerate when they have the same potential types
in the same order and the same shape: every distance between the points of
the path, in path order, equal to `tol`. A sequence and its time reverse
always contribute the same chi and are grouped too. Shapes are compared up to
rotation and reflection, which leave the orientation-averaged chi unchanged.

On an 87-atom fcc alloy cluster this reproduces FEFF8L's own path list to
5.3 A with up to four legs, with the same degeneracies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pathsum import Path
from .potential import BOHR


@dataclass
class PathGroup:
    path: Path                 # representative (bohr, potential types)
    degeneracy: int
    reff_A: float
    nleg: int
    atoms: tuple               # cluster indices of the representative's scatterers
    members: list = field(default_factory=list, repr=False)


def _key(points, types, tol):
    """Shape-and-species key of one path (points include the absorber at both ends)."""
    P = np.asarray(points)
    n = len(P) - 1                      # distinct points: P[0..n-1], P[n] == P[0]
    d = np.linalg.norm(P[:n, None, :] - P[None, :n, :], axis=-1)
    iu = np.triu_indices(n, 1)
    return tuple(types), tuple(np.round(d[iu] / tol).astype(np.int64).tolist())


def enumerate_paths(positions_A, site_types, rmax_A, nleg_max=4, tol=1e-3,
                    group=True, absorber=0, max_paths=2_000_000):
    """All paths to half-length rmax_A with up to nleg_max legs.

    positions_A : (n, 3) Angstrom
    site_types  : potential type per atom (the absorber's own is used if it
                  appears mid-path)
    Returns a list of PathGroup sorted by (reff, nleg), or, with group=False,
    one PathGroup of degeneracy 1 per sequence (time reverses still merged).
    """
    pos = np.asarray(positions_A, float)
    pos = pos - pos[absorber]
    types = np.asarray(site_types)
    n = len(pos)
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    r0 = dist[absorber]
    L = 2.0 * rmax_A + 1e-9
    others = np.array([i for i in range(n) if i != absorber])
    near = others[r0[others] <= rmax_A + 1e-9]
    seqs = []
    # two legs
    for i in near:
        if 2 * r0[i] <= L:
            seqs.append((i,))
    # three and four legs: depth-first with the remaining-length bound
    def extend(seq, used):
        last = seq[-1]
        if len(seq) + 1 > nleg_max - 1 + 1:
            return
        for j in range(n):
            if j == last:
                continue
            # an extra stop must still allow the way home
            nu = used + dist[last, j]
            if nu + r0[j] > L:
                continue
            new = seq + (j,)
            if j != absorber:
                seqs.append(new)
            if len(new) < nleg_max - 1:
                extend(new, nu)
            if len(seqs) > max_paths:
                raise RuntimeError("too many paths; lower rmax or nleg_max")
    if nleg_max >= 3:
        for i in near:
            extend((i,), r0[i])
    # only paths that end on a scatterer can close; drop those visiting the
    # absorber last (they are an SS path times an absorber return)
    out = {}
    order = []
    for s in seqs:
        if len(s) + 1 > nleg_max or s[-1] == absorber or s[0] == absorber:
            continue
        pts = np.vstack([pos[absorber], pos[list(s)], pos[absorber]])
        ty = tuple(int(types[i]) for i in s)
        k_f = _key(pts, ty, tol)
        rev = s[::-1]
        pts_r = np.vstack([pos[absorber], pos[list(rev)], pos[absorber]])
        k_r = _key(pts_r, ty[::-1], tol)
        key = min(k_f, k_r) if group else (min(k_f, k_r), min(s, rev))
        if key not in out:
            out[key] = [s]
            order.append(key)
        else:
            out[key].append(s)
    groups = []
    for key in order:
        mem = out[key]
        s = mem[0]
        sites = pos[list(s)] / BOHR
        ty = tuple(int(types[i]) for i in s)
        pth = Path(sites=sites, types=ty, degeneracy=float(len(mem)))
        groups.append(PathGroup(path=pth, degeneracy=len(mem), reff_A=float(pth.reff * BOHR),
                                nleg=len(s) + 1, atoms=tuple(s), members=mem))
    groups.sort(key=lambda g: (round(g.reff_A, 4), g.nleg))
    return groups
