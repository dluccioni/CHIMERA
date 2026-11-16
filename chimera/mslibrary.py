"""Exact multiple scattering from a library of species-resolved paths.

The composition-averaged term in `chimera.multiscat` sums the 3- and 4-leg
paths of ONE random cluster: one draw of the species along every path,
frozen, whatever configuration is being fitted. This module replaces it with
the exact sum for the configuration itself.

On a lattice with swap-only moves the GEOMETRY of every path is fixed; only
the species at its sites change. Every multiple-scattering path within the
path length and leg limits falls into one of a few geometric classes (25 for
fcc to a half-path length of 5.3 A with up to 4 legs), and a path's amplitude
and phase depend only on its class, the absorber's species and the species
at its 1-3 other sites. So:

  1. `build`: for every class and every assignment of species to its sites
     (327 paths per absorber for fcc), one FEFF path with degeneracy 1,
     written into a custom paths.dat and computed by FEFF10's genfmt and ff2x
     on the potentials of an existing FEFF10 run. pot and xsph are not rerun,
     so the scattering physics is exactly that run's.
  2. `PathLibrary`: the library on the model's momentum grid, and the
     random-alloy EXPECTATION per absorber (every site an independent draw
     from the composition) with the `multiscat.MSPaths` interface, so it
     drops into calibration in place of the one-cluster average.
  3. `SupercellMS`: the sampler's state - how often every library entry
     occurs in each replica's configuration, the multiple-scattering
     analogue of the pair counts - and the change a swap makes to it, for
     `methods.method_a_rmc.RMCSampler(ms_library=...)`.

Classes are identified under the cubic group O_h and time reversal: a path
and its reverse have the same amplitude, and FEFF counts both in one
degeneracy. The images enumerated here reproduce FEFF's degeneracies class by
class, and the library summed over a FEFF cluster reproduces FEFF's own
multiple scattering for that cluster (`validate_cluster`).
"""
from __future__ import annotations

import itertools
import json
import os
import shutil
import time

import numpy as np

from exafs_gpu.scattering import read_feffdat, tables_root, _meta

LIB_FILE = "library.json"


# ------------------------------------------------------------------ geometry
def _cubic_ops():
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            A = np.zeros((3, 3), int)
            for r, c in enumerate(perm):
                A[r, c] = sg[r]
            out.append(A)
    return out


OPS = _cubic_ops()


def read_paths_dat(fn):
    """FEFF paths.dat -> [dict(index, nleg, deg, sites=[(xyz, ipot), ...])];
    the last site of every path is the absorber."""
    L = open(fn).read().splitlines()
    i = next(k for k, ln in enumerate(L) if ln.strip().startswith("---")) + 1
    out = []
    while i < len(L):
        h = L[i].split()
        if len(h) < 3 or not h[0].isdigit():
            i += 1
            continue
        nleg = int(h[1])
        sites = []
        for m in range(nleg):
            t = L[i + 2 + m].split()
            sites.append((np.array([float(t[0]), float(t[1]), float(t[2])]), int(t[3])))
        out.append(dict(index=int(h[0]), nleg=nleg, deg=float(h[2]), sites=sites))
        i += 2 + nleg
    return out


def canonical(seq):
    """Canonical form of a path given as the integer positions of its sites
    after the absorber (the absorber closes it implicitly): the smallest
    image under O_h and reversal."""
    best = None
    S0 = np.asarray(seq, int)
    for s in (S0, S0[::-1]):
        for A in OPS:
            t = tuple(map(tuple, s @ A.T))
            if best is None or t < best:
                best = t
    return best


class PathClass:
    """One geometric class: its representative, slots and directed images."""

    def __init__(self, rep):
        self.rep = np.asarray(rep, int)                      # (nleg-1, 3)
        self.nleg = self.rep.shape[0] + 1
        slot_of, seen = [], {}
        for q in map(tuple, self.rep):
            if q == (0, 0, 0):
                slot_of.append(-1)
            else:
                seen.setdefault(q, len(seen))
                slot_of.append(seen[q])
        self.slot_of = np.array(slot_of, int)
        self.nslot = len(seen)
        imgs = {}
        for rev in (False, True):
            s = self.rep[::-1] if rev else self.rep
            so = self.slot_of[::-1] if rev else self.slot_of
            for A in OPS:
                t = tuple(map(tuple, s @ A.T))
                if t not in imgs:
                    imgs[t] = so.copy()
        self.images = [(np.array(k, int), v) for k, v in imgs.items()]
        self.deg = len(self.images)

    def reff_units(self):
        pts = np.vstack([np.zeros((1, 3)), self.rep, np.zeros((1, 3))])
        return 0.5 * float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    def to_dict(self):
        return dict(rep=self.rep.tolist(), nslot=self.nslot, deg=self.deg,
                    reff_units=self.reff_units())


def classes_from_feff(folder, unit, min_leg=3):
    """The geometric classes of the multiple-scattering paths FEFF found in
    one run, with each class's FEFF degeneracy summed (for checking)."""
    got = {}
    for p in read_paths_dat(os.path.join(folder, "paths.dat")):
        if p["nleg"] < min_leg:
            continue
        x = np.array([s[0] for s in p["sites"][:-1]]) / unit
        q = np.rint(x).astype(int)
        assert np.allclose(x, q, atol=2e-3), "path sites are not lattice points"
        key = canonical(q)
        got.setdefault(key, 0.0)
        got[key] += p["deg"]
    keys = sorted(got, key=lambda k: (len(k), PathClass(k).reff_units(), k))
    return [PathClass(k) for k in keys], [got[k] for k in keys]


# ------------------------------------------------------------------ build
def _potentials(feff_inp):
    """ipot -> element symbol from a feff.inp POTENTIALS block."""
    L = open(feff_inp).read().splitlines()
    i = next(k for k, ln in enumerate(L) if ln.strip().upper().startswith("POTENTIALS"))
    out = {}
    for ln in L[i + 1:]:
        t = ln.split()
        if not t or not t[0].lstrip("-").isdigit():
            break
        out[int(t[0])] = t[2] if len(t) > 2 else t[1]
    return out


def _write_paths(fn, entries, unit, ipot_of, symbols, absorber):
    """A paths.dat FEFF's genfmt reads: one path per (class, species) entry."""
    with open(fn, "w") as f:
        f.write(" PATH  Rmax= 9.000,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 0.00%\n")
        f.write(" -----------------------------------------------------------------------\n")
        for n, (cl, sp) in enumerate(entries, 1):
            pts = list(cl.rep) + [np.zeros(3, int)]
            reff = cl.reff_units() * unit
            f.write(f"  {n:4d}    {cl.nleg}   1.000  index, nleg, degeneracy, r= {reff:8.4f}\n")
            f.write("      x           y           z     ipot  label      rleg      beta        eta\n")
            prev = np.zeros(3)
            for m, q in enumerate(pts):
                x = np.asarray(q, float) * unit
                if m == len(pts) - 1:
                    ip, lab = 0, absorber
                else:
                    s = cl.slot_of[m]
                    ip, lab = (0, absorber) if s < 0 else (ipot_of[symbols[sp[s]]], symbols[sp[s]])
                rleg = float(np.linalg.norm(x - prev))
                prev = x
                f.write(f"  {x[0]:10.6f}  {x[1]:10.6f}  {x[2]:10.6f}  {ip:2d} '{lab:<6s}'  "
                        f"{rleg:9.4f}    0.0000    0.0000\n")


def entry_list(classes, nsp):
    """[(class, species tuple)] in library order: class-major, then the slot
    species with slot 0 fastest (entry = offset + sum_q s_q nsp^q)."""
    out = []
    for cl in classes:
        for code in range(nsp ** cl.nslot):
            sp = [(code // nsp ** q) % nsp for q in range(cl.nslot)]
            out.append((cl, tuple(sp)))
    return out


def build(src, out=None, verbose=True):
    """Build the library from the FEFF10 multiple-scattering run `src`
    (data/<src>/<El>/ with pot.bin, phase.bin and paths.dat) into
    data/<out>/ (default <src>lib)."""
    from exafs_gpu import feff10
    root = tables_root(src)
    meta = _meta(root)
    els = tuple(meta["elements"])
    nsp = len(els)
    d_nn = float(meta.get("d_nn", 2.517))
    unit = d_nn / np.sqrt(2.0)                      # a0 / 2 for fcc
    out = out or (src + "lib")
    dst = tables_root(out) if os.path.isabs(out) else os.path.join(os.path.dirname(root), out)
    os.makedirs(dst, exist_ok=True)
    classes, feff_deg = None, None
    for el in els:
        c, dg = classes_from_feff(os.path.join(root, el), unit)
        if classes is None:
            classes, feff_deg = c, dg
        else:
            assert [x.rep.tolist() for x in c] == [x.rep.tolist() for x in classes], \
                "absorbers disagree on the path classes"
    for cl, dg in zip(classes, feff_deg):
        assert abs(cl.deg - dg) < 1e-6, f"class {cl.rep.tolist()}: {cl.deg} images, FEFF {dg}"
    entries = entry_list(classes, nsp)
    for el in els:
        t0 = time.time()
        work = os.path.join(dst, "_work", el)
        if os.path.isdir(work):
            shutil.rmtree(work)
        shutil.copytree(os.path.join(root, el), work,
                        ignore=shutil.ignore_patterns("feff0*.dat", "chip0*.dat"))
        pots = _potentials(os.path.join(work, "feff.inp"))
        ipot_of = {}
        for ip, sym in sorted(pots.items()):
            if ip >= 1:
                ipot_of.setdefault(sym, ip)
        _write_paths(os.path.join(work, "paths.dat"), entries, unit, ipot_of, els, el)
        feff10.run(work, modules=("genfmt", "ff2x"))
        files = sorted(f for f in os.listdir(work) if f.startswith("feff0") and f.endswith(".dat"))
        assert len(files) == len(entries), (
            f"{el}: genfmt wrote {len(files)} paths for {len(entries)} entries "
            f"(a filter dropped some: check CRITERIA)")
        od = os.path.join(dst, el)
        os.makedirs(od, exist_ok=True)
        for f_ in files:
            shutil.copy(os.path.join(work, f_), os.path.join(od, f_))
        if verbose:
            print(f"  {out}/{el}: {len(files)} library paths [{time.time() - t0:.0f}s]", flush=True)
    lib = dict(elements=list(els), d_nn=d_nn, unit=unit, lattice="fcc", source=src,
               classes=[c.to_dict() for c in classes], n_entries_per_absorber=len(entries),
               feff10_settings=meta.get("feff10_settings"), engine=meta.get("engine"))
    json.dump(lib, open(os.path.join(dst, LIB_FILE), "w"), indent=1)
    json.dump(dict(meta, library_of=src, d_nn=d_nn), open(os.path.join(dst, "meta.json"), "w"),
              indent=1)
    shutil.rmtree(os.path.join(dst, "_work"), ignore_errors=True)
    return dst


# ------------------------------------------------------------------ library
class PathLibrary:
    """The species-resolved paths of every class, per absorber."""

    def __init__(self, name, composition=None):
        root = tables_root(name)
        js = json.load(open(os.path.join(root, LIB_FILE)))
        self.name = name
        self.elements = tuple(js["elements"])
        self.nsp = len(self.elements)
        self.d_nn = float(js["d_nn"])
        self.unit = float(js["unit"])
        self.classes = [PathClass(c["rep"]) for c in js["classes"]]
        self.entries = entry_list(self.classes, self.nsp)
        self.nblk = len(self.entries)
        self.cls_off = np.cumsum([0] + [self.nsp ** c.nslot for c in self.classes])[:-1]
        comp = (np.ones(self.nsp) / self.nsp if composition is None
                else np.asarray(composition, float) / np.sum(composition))
        self.composition = comp
        self.paths = {}
        for el in self.elements:
            d = os.path.join(root, el)
            fs = sorted(f for f in os.listdir(d) if f.startswith("feff0") and f.endswith(".dat"))
            assert len(fs) == self.nblk, f"{el}: {len(fs)} files for {self.nblk} entries"
            self.paths[el] = [read_feffdat(os.path.join(d, f)) for f in fs]
        self.n = {a: self.nblk for a in self.elements}
        self._grid = None

    # the MSPaths interface -------------------------------------------------
    def summary(self):
        return [dict(absorber=a, n_classes=len(self.classes), n_entries=self.nblk,
                     total_degeneracy=int(sum(c.deg for c in self.classes)))
                for a in self.elements]

    def prepare(self, k_phys):
        p = np.asarray(k_phys, float)
        self._grid = p
        recs = [d for a in self.elements for d in self.paths[a]]
        self._F = np.array([np.interp(p, d["p"], d["f"]) for d in recs])
        self._D = np.array([np.interp(p, d["p"], d["delta"]) for d in recs])
        self._L = np.array([np.interp(p, d["p"], d["lam"]) for d in recs])
        self._reff = np.array([d["reff"] for d in recs])
        return self

    def table(self, sigma2_nn, r_nn, c_ms=1.3, amp=1.0, s02=1.0, a_scale=1.0):
        """(nsp * nblk, nk): chi of ONE occurrence of every entry (degeneracy
        1), absorber-major, with the same Debye-Waller and length convention
        as `multiscat.MSPaths.chi_fast`."""
        p = self._grid
        r = (self._reff * a_scale)[:, None]
        s2 = c_ms * sigma2_nn * r / (2.0 * r_nn)
        return amp * (s02 * self._F / np.maximum(p[None, :] * r ** 2, 1e-12)
                      * np.exp(-2.0 * r / np.maximum(self._L, 1e-6) - 2.0 * p[None, :] ** 2 * s2)
                      * np.sin(2.0 * p[None, :] * r + self._D))

    def expected_counts(self, composition=None):
        """(nsp * nblk,): occurrences of every entry around one absorber of
        each species in a RANDOM alloy - the class's image count times the
        probability of the slot species, the slots being distinct sites."""
        c = self.composition if composition is None else np.asarray(composition, float)
        blk = np.array([cl.deg * np.prod([c[s] for s in sp]) for cl, sp in self.entries])
        return np.tile(blk, self.nsp)

    def chi_counts(self, C, table):
        """Unnormalised chi_MS (..., nsp, nk) from entry counts (..., nsp * nblk)."""
        C = np.asarray(C, float)
        return np.stack([C[..., a * self.nblk:(a + 1) * self.nblk] @ table[a * self.nblk:(a + 1) * self.nblk]
                         for a in range(self.nsp)], axis=-2)

    def chi_fast(self, sigma2_nn, r_nn, c_ms=1.3, amp=1.0, s02=1.0, a_scale=1.0):
        """Random-alloy chi_MS per absorbing atom (nsp, nk): drop-in for
        `MSPaths.chi_fast`."""
        T = self.table(sigma2_nn, r_nn, c_ms, amp, s02, a_scale)
        return self.chi_counts(self.expected_counts(), T)


def get(name, composition=None):
    return PathLibrary(name, composition)


# ------------------------------------------------------------------ supercell
class SupercellMS:
    """Per-replica entry counts on a periodic supercell, for the sampler.

    `init(species)` counts every replica's paths; `delta` returns the change
    in unnormalised chi_MS of each replica's proposed swap (and keeps the
    count change); `commit(acc)` applies it where accepted; `permute(ix)`
    follows replica exchange; `full(species)` recounts from scratch.
    """

    def __init__(self, lib: PathLibrary, S):
        import cupy as cp
        self.cp = cp
        self.lib = lib
        self.nsp = lib.nsp
        self.nblk = lib.nblk
        self.nent = self.nsp * self.nblk
        unit = S.a0_ref / 2.0
        X = np.asarray(S.ideal0, float) / unit
        Xi = np.rint(X).astype(int)
        assert np.allclose(X, Xi, atol=1e-6), "supercell sites are not fcc lattice points"
        H = np.asarray(S.H0, float) / unit
        assert np.allclose(H, np.diag(np.diag(H))), "SupercellMS needs an orthogonal box"
        box = np.rint(np.diag(H)).astype(int)
        Xi %= box
        look = -np.ones(tuple(box), int)
        look[Xi[:, 0], Xi[:, 1], Xi[:, 2]] = np.arange(len(Xi))
        n = self.natoms = len(Xi)
        P_abs, P_cls, P_slot = [], [], []
        for g, cl in enumerate(lib.classes):
            assert 2 * np.abs(cl.rep).max() < box.min(), "supercell too small for the paths"
            for img, so in cl.images:
                pos = (Xi[:, None, :] + img[None]) % box
                sites = look[pos[..., 0], pos[..., 1], pos[..., 2]]
                assert (sites >= 0).all()
                slot = -np.ones((n, 3), int)
                for q in range(cl.nslot):
                    m = int(np.argmax(so == q))
                    slot[:, q] = sites[:, m]
                P_abs.append(np.arange(n)); P_cls.append(np.full(n, g)); P_slot.append(slot)
        P_abs = np.concatenate(P_abs); P_cls = np.concatenate(P_cls)
        P_slot = np.concatenate(P_slot)
        self.npath = P_abs.size
        # incidence: every path a site takes part in, as absorber or slot
        site = np.concatenate([P_abs] + [P_slot[:, q] for q in range(3)])
        pid = np.tile(np.arange(self.npath), 4)
        keep = site >= 0
        site, pid = site[keep], pid[keep]
        order = np.argsort(site, kind="stable")
        site, pid = site[order], pid[order]
        cnt = np.bincount(site, minlength=n)
        self.M = int(cnt.max())
        inc = np.full((n, self.M), self.npath, int)            # pad -> dummy path
        start = np.r_[0, np.cumsum(cnt)[:-1]]
        col = np.arange(site.size) - start[site]
        inc[site, col] = pid
        # the dummy path (id npath) has no sites and weight 0
        P_abs = np.r_[P_abs, 0]; P_cls = np.r_[P_cls, 0]
        P_slot = np.vstack([P_slot, -np.ones((1, 3), int)])
        self.d_abs = cp.asarray(P_abs.astype(np.int32))
        self.d_cls = cp.asarray(P_cls.astype(np.int32))
        self.d_slot = cp.asarray(P_slot.astype(np.int32))
        self.d_inc = cp.asarray(inc.astype(np.int32))
        self.d_cls_off = cp.asarray(lib.cls_off.astype(np.int32))
        self.d_pow = cp.asarray(np.array([1, self.nsp, self.nsp ** 2], np.int32))
        self.T = None
        self.C = None
        self._dC = None

    # ------------------------------------------------------------ parameters
    def set_params(self, sigma2_nn, r_nn, c_ms=1.3, amp=1.0, s02=1.0, a_scale=1.0):
        self.T = self.cp.asarray(self.lib.table(sigma2_nn, r_nn, c_ms, amp, s02, a_scale))
        return self

    def expected_chi(self, composition=None):
        """Random-alloy chi_MS per absorbing atom (nsp, nk), numpy."""
        T = self.cp.asnumpy(self.T)
        return self.lib.chi_counts(self.lib.expected_counts(composition), T)

    # ------------------------------------------------------------ bookkeeping
    def _entries(self, spf, rows, pid, a=None, b=None, nsa=None, nsb=None):
        cp = self.cp
        ab = self.d_abs[pid]
        cl = self.d_cls[pid]
        sl = self.d_slot[pid]
        valid = sl >= 0
        s_ab = spf[rows + ab]
        s_sl = spf[rows[..., None] + cp.maximum(sl, 0)]
        if a is not None:
            a_, b_ = a[:, None], b[:, None]
            s_ab = cp.where(ab == a_, nsa[:, None], cp.where(ab == b_, nsb[:, None], s_ab))
            s_sl = cp.where(sl == a_[..., None], nsa[:, None, None],
                            cp.where(sl == b_[..., None], nsb[:, None, None], s_sl))
        return (s_ab * self.nblk + self.d_cls_off[cl]
                + (s_sl * self.d_pow * valid).sum(-1))

    def counts(self, species, chunk=4):
        """(nrep, nent) entry counts of each configuration, from scratch."""
        cp = self.cp
        species = cp.asarray(species, cp.int32)
        nrep = species.shape[0]
        out = cp.zeros((nrep, self.nent), cp.float64)
        pid_all = cp.arange(self.npath, dtype=cp.int32)
        for r0 in range(0, nrep, chunk):
            sp = species[r0:r0 + chunk]
            m = sp.shape[0]
            rows = (cp.arange(m, dtype=cp.int32) * self.natoms)[:, None]
            e = self._entries(sp.ravel(), rows, cp.broadcast_to(pid_all, (m, self.npath)))
            off = (cp.arange(m) * self.nent)[:, None]
            out[r0:r0 + m] = cp.bincount((off + e).ravel(),
                                         minlength=m * self.nent).reshape(m, self.nent)
        return out

    def chi_of(self, C):
        """Unnormalised chi_MS (nrep, nsp, nk) from entry counts."""
        cp = self.cp
        return cp.stack([C[:, a * self.nblk:(a + 1) * self.nblk] @ self.T[a * self.nblk:(a + 1) * self.nblk]
                         for a in range(self.nsp)], axis=1)

    def init(self, species):
        self.C = self.counts(species)
        return self.chi_of(self.C)

    def full(self, species):
        return self.chi_of(self.counts(species))

    def delta(self, species, a, b, nsa, nsb):
        """Change in unnormalised chi_MS for every replica's proposal (swap
        a <-> b to species nsa, nsb; a == b and unchanged species for a
        displacement, which gives zero)."""
        cp = self.cp
        nrep = species.shape[0]
        spf = species.ravel()
        rows = (cp.arange(nrep, dtype=cp.int32) * self.natoms)[:, None]
        pa, pb = self.d_inc[a], self.d_inc[b]
        pid = cp.concatenate([pa, pb], axis=1)
        # a path through both sites is in both lists: count it once
        has_a = (self.d_abs[pb] == a[:, None]) | (self.d_slot[pb] == a[:, None, None]).any(-1)
        w = cp.concatenate([pa < self.npath, (pb < self.npath) & ~has_a], axis=1).astype(cp.float64)
        e_old = self._entries(spf, rows, pid)
        e_new = self._entries(spf, rows, pid, a, b, nsa, nsb)
        off = (cp.arange(nrep) * self.nent)[:, None]
        nb = nrep * self.nent
        self._dC = (cp.bincount((off + e_new).ravel(), weights=w.ravel(), minlength=nb)
                    - cp.bincount((off + e_old).ravel(), weights=w.ravel(), minlength=nb)
                    ).reshape(nrep, self.nent)
        return self.chi_of(self._dC)

    def commit(self, acc):
        self.C = self.C + self._dC * acc[:, None].astype(self.C.dtype)

    def permute(self, ix):
        self.C = self.C[ix]


# ------------------------------------------------------------------ checks
def validate_cluster(lib: PathLibrary, src, absorber, sigma2_nn=0.006, c_ms=1.0):
    """The library summed over the FEFF cluster of `src`/<absorber> against
    FEFF's own multiple-scattering paths for that cluster (feffNNNN.dat with
    3+ legs, FEFF's degeneracies). Returns (chi_library, chi_feff) on the
    prepared grid; they agree when the library is exact."""
    root = tables_root(src)
    folder = os.path.join(root, absorber)
    unit = lib.unit
    # the cluster: integer position -> species index
    L = open(os.path.join(folder, "feff.inp")).read().splitlines()
    i = next(k for k, ln in enumerate(L) if ln.strip().upper().startswith("ATOMS"))
    pots = _potentials(os.path.join(folder, "feff.inp"))
    site = {}
    for ln in L[i + 1:]:
        t = ln.split()
        if len(t) < 4:
            continue
        try:
            x = np.array([float(t[0]), float(t[1]), float(t[2])]) / unit
        except ValueError:
            break
        site[tuple(np.rint(x).astype(int))] = lib.elements.index(pots[int(t[3])])
    a = lib.elements.index(absorber)
    T = lib.table(sigma2_nn, lib.d_nn, c_ms, 1.0, 1.0, 1.0)
    C = np.zeros(lib.nsp * lib.nblk)
    for g, cl in enumerate(lib.classes):
        for img, so in cl.images:
            sp = []
            for q in range(cl.nslot):
                m = int(np.argmax(so == q))
                sp.append(site[tuple(img[m])])
            e = a * lib.nblk + lib.cls_off[g] + sum(s * lib.nsp ** q for q, s in enumerate(sp))
            C[e] += 1
    chi_lib = lib.chi_counts(C, T)[a]
    # FEFF's own paths for the same cluster
    p = lib._grid
    chi_ref = np.zeros_like(p)
    for fn in sorted(os.listdir(folder)):
        if not (fn.startswith("feff0") and fn.endswith(".dat")):
            continue
        d = read_feffdat(os.path.join(folder, fn))
        if d["nleg"] < 3:
            continue
        r = d["reff"]
        f = np.interp(p, d["p"], d["f"]); dl = np.interp(p, d["p"], d["delta"])
        lam = np.interp(p, d["p"], d["lam"])
        s2 = c_ms * sigma2_nn * r / (2.0 * lib.d_nn)
        chi_ref += (d["deg"] * f / np.maximum(p * r ** 2, 1e-12)
                    * np.exp(-2.0 * r / np.maximum(lam, 1e-6) - 2.0 * p ** 2 * s2)
                    * np.sin(2.0 * p * r + dl))
    return chi_lib, chi_ref
