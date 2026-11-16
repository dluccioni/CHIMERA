"""Disorder and energetics of the alloy from an interatomic potential.

Two things the EXAFS fit otherwise has to guess, taken instead from an
embedded-atom potential for the alloy (`exafs_gpu.eam`; for CrCoNi the
Li, Sheng & Ma 2019 NiCoCr potential, data/_external/potentials):

1. `pair_statistics` - molecular dynamics of random supercells at the
   measured lattice constant and temperature. For every coordination shell
   and ordered pair type it returns the mean distance, the variance sigma^2
   and the third cumulant C3 of the pair distance, and the same for the
   relaxed (0 K) structure. The static part (size mismatch) is classical;
   the thermal part is corrected for zero-point motion with a correlated
   Einstein model fitted to it (sigma^2_q = hbar/(2 mu w) coth(hbar w / 2kT),
   w from sigma^2_classical = kT/(mu w^2)). These replace the fit's free
   per-shell sigma^2 multipliers, per-pair bond-length offsets and C3.

2. `pair_interactions` - relaxed energies of many arrangements fitted to an
   effective pair interaction (EPI) model on the lattice,
       E = E_0 + sum_s sum_{a<=b} V_ab(s) N_ab(s),
   N_ab(s) the number of a-b pairs in shell s. An arrangement's energy is
   then a function of the same pair counts the EXAFS model uses, which is what
   lets the sampler weigh each move by exp(-dE/kT) at the temperature the
   sample was last annealed at (`methods.method_a_rmc`, `energy_prior`).

Species are in CHIMERA's order (Cr, Co, Ni by default); the potential's own
element order is mapped by symbol.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

HBAR_EVFS = 0.6582119569          # eV fs
KB_EV = 8.617333262e-5
AMU = 103.642697                  # eV fs^2 / A^2
POTENTIAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "_external", "potentials", "NiCoCr.lammps.eam")


def fcc_supercell(n, a):
    b = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    g = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    return ((g[:, None, :] + b[None]) * a).reshape(-1, 3), np.eye(3) * n * a


def random_species(natoms, composition, nconf, seed=0):
    comp = np.asarray(composition, float)
    comp = comp / comp.sum()
    cnt = np.round(comp * natoms).astype(int)
    cnt[-1] = natoms - cnt[:-1].sum()
    base = np.repeat(np.arange(len(comp)), cnt).astype(np.int32)
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(base) for _ in range(nconf)])


@dataclass
class PairStats:
    """Moments of the pair-distance distribution, (nshell, nsp, nsp) each."""
    elements: tuple
    a0: float
    T: float
    shells: np.ndarray            # reference shell radii (A)
    mean: np.ndarray              # <r>
    sigma2: np.ndarray            # variance, zero-point corrected
    sigma2_classical: np.ndarray
    c3: np.ndarray                # third cumulant (A^3)
    static_mean: np.ndarray
    static_sigma2: np.ndarray
    counts: np.ndarray            # pairs sampled per entry
    meta: dict = field(default_factory=dict)

    def dr(self):
        """Pair-type bond-length offsets: <r_ab> - <r> (shell average), per shell."""
        w = self.counts
        avg = (self.mean * w).sum((1, 2)) / w.sum((1, 2))
        return self.mean - avg[:, None, None]

    def shell_sigma2(self):
        w = self.counts
        return (self.sigma2 * w).sum((1, 2)) / w.sum((1, 2))

    def shell_c3(self):
        w = self.counts
        return (self.c3 * w).sum((1, 2)) / w.sum((1, 2))

    def to_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


def _cell(elements, a0, ncell, potential):
    import cupy as cp
    from exafs_gpu import eam as EAM
    pot = EAM.EAMPotential(potential)
    sites, H = fcc_supercell(ncell, a0)
    cell = EAM.EAMCell(pot, H, sites)
    pmap = np.array([pot.elements.index(e) for e in elements], np.int32)
    return pot, cell, pmap, cp


def _moments(cell, rr, sp, shell_of, nshell, nsp, acc):
    """Accumulate sum r^0..3 over directed pairs into acc (nshell, nsp, nsp, 4)."""
    import cupy as cp
    pi, pj = cell.d_pi, cell.d_pj
    sh = cp.asarray(shell_of)
    keep = sh < nshell
    for c in range(rr.shape[0]):
        r = rr[c][keep]
        t = (sp[c][pi] * nsp + sp[c][pj])[keep]
        idx = sh[keep] * nsp * nsp + t
        n = nshell * nsp * nsp
        acc[..., 0] += cp.asnumpy(cp.bincount(idx, minlength=n)).reshape(nshell, nsp, nsp)
        for p in (1, 2, 3):
            acc[..., p] += cp.asnumpy(cp.bincount(idx, weights=r ** p, minlength=n)).reshape(nshell, nsp, nsp)


def _finish(acc):
    n = np.maximum(acc[..., 0], 1)
    m1 = acc[..., 1] / n
    m2 = acc[..., 2] / n - m1 ** 2
    m3 = acc[..., 3] / n - 3 * m1 * (acc[..., 2] / n) + 2 * m1 ** 3
    return m1, m2, m3, acc[..., 0]


def pair_statistics(a0, T, elements=("Cr", "Co", "Ni"), composition=(1, 1, 1), ncell=6,
                    nconf=8, n_equil=3000, n_sample=15000, every=20, dt=2.0, nshell=6,
                    seed=0, species=None, potential=POTENTIAL, verbose=True) -> PairStats:
    """Molecular dynamics of `nconf` random supercells (or the given `species`,
    (nconf, natoms) in CHIMERA's order) at lattice constant a0 and temperature
    T; pair-distance moments by shell and pair type, plus the relaxed ones."""
    from exafs_gpu import eam as EAM
    t0 = time.time()
    pot, cell, pmap, cp = _cell(elements, a0, ncell, potential)
    nsp = len(elements)
    if species is None:
        species = random_species(cell.natoms, composition, nconf, seed)
    species = np.asarray(species, np.int32)
    nconf = species.shape[0]
    sp_dev = cp.asarray(species)
    sp_pot = cp.asarray(pmap[species])
    shell_of, shell_r = cell.shells()
    pos0 = cp.asarray(np.repeat(cell.sites[None], nconf, 0))
    # static: relaxed at fixed cell
    pos_rel, E_rel = EAM.relax_fire(cell, pos0.copy(), sp_pot)
    acc_s = np.zeros((nshell, nsp, nsp, 4))
    _moments(cell, cell.pair_distances(pos_rel), sp_dev, shell_of, nshell, nsp, acc_s)
    # thermal: Langevin MD from the relaxed structure
    acc = np.zeros((nshell, nsp, nsp, 4))
    pos, _ = EAM.langevin(cell, pos_rel, sp_pot, T, n_equil, dt=dt, seed=seed + 1)

    def cb(step, p):
        _moments(cell, cell.pair_distances(p), sp_dev, shell_of, nshell, nsp, acc)
    EAM.langevin(cell, pos, sp_pot, T, n_sample, dt=dt, seed=seed + 2, every=every, callback=cb)
    m1, m2, m3, n = _finish(acc)
    s1, s2, _, _ = _finish(acc_s)
    # zero-point correction of the thermal part, Einstein model per entry
    masses = np.array(pot.mass)[pmap]
    mu = (masses[:, None] * masses[None, :] / (masses[:, None] + masses[None, :])) * AMU
    th = np.maximum(m2 - s2, 1e-8)
    kT = KB_EV * T
    w = np.sqrt(kT / (mu[None] * th))
    x = HBAR_EVFS * w / (2 * kT)
    th_q = th * x / np.tanh(x)
    sig2 = s2 + th_q
    if verbose:
        print(f"  pair statistics a0={a0:.4f} T={T:.0f} K: {nconf} x {cell.natoms} atoms, "
              f"{n_sample} steps, {time.time() - t0:.0f}s", flush=True)
    return PairStats(elements=tuple(elements), a0=float(a0), T=float(T), shells=shell_r[:nshell],
                     mean=m1, sigma2=sig2, sigma2_classical=m2, c3=m3, static_mean=s1,
                     static_sigma2=s2, counts=n,
                     meta=dict(potential=os.path.basename(potential), ncell=ncell, nconf=nconf,
                               n_sample=n_sample, dt_fs=dt, zero_point="Einstein per entry",
                               E_relaxed_eV_per_atom=float(cp.asnumpy(E_rel).mean() / cell.natoms)))


# ----------------------------------------------------------------- energetics
@dataclass
class PairInteractions:
    """E = E0 + sum_s sum_{a<=b} V[s, a, b] N_ab(s) (eV), N_ab counting unordered pairs."""
    elements: tuple
    a0: float
    V: np.ndarray                 # (nshell, nsp, nsp), symmetric; gauge: see `fit`
    e0: float
    rms_train_meV: float
    rms_cv_meV: float
    meta: dict = field(default_factory=dict)

    def directed_table(self):
        """V/2 per ORDERED pair type (shell, a*nsp+b): each unordered pair is
        listed twice in a directed neighbour list."""
        n = len(self.elements)
        return 0.5 * self.V.reshape(self.V.shape[0], n * n)

    def energy(self, counts_unordered):
        return self.e0 + float((self.V * counts_unordered).sum())


def pair_counts(species, pi, pj, shell_of, nshell, nsp):
    """Unordered pair counts N[s, a, b] (symmetric) from a directed list.

    In a directed list an a-b pair (a != b) appears once as a->b and once as
    b->a, and an a-a pair twice as a->a. With D the directed counts that makes
    N_ab = D_ab = D_ba and N_aa = D_aa / 2."""
    t = species[pi] * nsp + species[pj]
    keep = shell_of < nshell
    idx = shell_of[keep] * nsp * nsp + t[keep]
    D = np.bincount(idx, minlength=nshell * nsp * nsp).reshape(nshell, nsp, nsp).astype(float)
    return 0.5 * (D + D.transpose(0, 2, 1)) * np.where(np.eye(nsp, dtype=bool), 0.5, 1.0)


def neighbour_table(pi, pj, shell_of, natoms, nshell):
    """nb[s] (natoms, z_s): the shell-s neighbours of every site (directed list)."""
    out = []
    for s in range(nshell):
        m = shell_of == s
        i, j = pi[m], pj[m]
        order = np.argsort(i, kind="stable")
        z = np.bincount(i, minlength=natoms)
        assert z.min() == z.max(), "non-uniform coordination"
        out.append(np.ascontiguousarray(j[order].reshape(natoms, z[0]), np.int64))
    return out


def _mc_kernel():
    import numba

    @numba.njit(cache=True)
    def run(sp, nb_stack, zs, V, beta, nsweep, seed):
        """Swap-move Metropolis on lattice pair interactions V[s, a, b].
        nb_stack (nshell, natoms, zmax) padded with -1."""
        np.random.seed(seed)
        n = sp.size
        nsh = V.shape[0]
        for it in range(nsweep * n):
            a = np.random.randint(n)
            b = np.random.randint(n)
            sa, sb = sp[a], sp[b]
            if sa == sb:
                continue
            dE = 0.0
            for s in range(nsh):
                for q in range(zs[s]):
                    j = nb_stack[s, a, q]
                    if j != b:
                        dE += V[s, sb, sp[j]] - V[s, sa, sp[j]]
                    j = nb_stack[s, b, q]
                    if j != a:
                        dE += V[s, sa, sp[j]] - V[s, sb, sp[j]]
            if dE <= 0.0 or np.random.random() < np.exp(-beta * dE):
                sp[a] = sb
                sp[b] = sa
        return sp
    return run


_MC = None


def swap_mc(sp, nb, V, T, nsweep, seed=0):
    """Equilibrate an arrangement under pair interactions V (nshell, nsp, nsp)
    at temperature T (K) with swap moves; nb from `neighbour_table`."""
    global _MC
    if _MC is None:
        _MC = _mc_kernel()
    nsh = V.shape[0]
    zmax = max(t.shape[1] for t in nb[:nsh])
    stack = -np.ones((nsh, nb[0].shape[0], zmax), np.int64)
    zs = np.zeros(nsh, np.int64)
    for s in range(nsh):
        stack[s, :, :nb[s].shape[1]] = nb[s]
        zs[s] = nb[s].shape[1]
    beta = 1.0 / (KB_EV * T) if T > 0 else 1e12
    return _MC(np.array(sp, np.int64), stack, zs, np.ascontiguousarray(V, np.float64), beta,
               int(nsweep), int(seed)).astype(np.int32)


def _ordered_configs(natoms, composition, n, cell, shell_of, rng, sweeps=40):
    """Arrangements spanning order and clustering: lattice MC with random
    first/second-shell pair interactions at random temperatures."""
    nsp = len(composition)
    nb = neighbour_table(cell.pi, cell.pj, shell_of, natoms, 2)
    out = []
    for k in range(n):
        sp = random_species(natoms, composition, 1, int(rng.integers(1 << 30)))[0]
        if k < n // 5:
            out.append(sp)
            continue
        J = rng.normal(0, 0.03, (2, nsp, nsp))
        J = 0.5 * (J + J.transpose(0, 2, 1))
        out.append(swap_mc(sp, nb, J, rng.uniform(400, 3000), sweeps, int(rng.integers(1 << 30))))
    return np.array(out, np.int32)


def _relaxed_energies(cell, pmap, configs, batch, cp):
    from exafs_gpu import eam as EAM
    E = np.zeros(len(configs))
    for s in range(0, len(configs), batch):
        sp = configs[s:s + batch]
        pos0 = cp.asarray(np.repeat(cell.sites[None], len(sp), 0))
        _, e = EAM.relax_fire(cell, pos0, cp.asarray(pmap[sp]))
        E[s:s + batch] = cp.asnumpy(e)
    return E


def _epi_features(configs, cell, shell_of, nshell, nsp):
    """Unlike-pair counts per shell (like pairs follow from the composition)."""
    pairs = [(a, b) for a in range(nsp) for b in range(a + 1, nsp)]
    return np.array([[pair_counts(sp, cell.pi, cell.pj, shell_of, nshell, nsp)[s, a, b]
                      for s in range(nshell) for (a, b) in pairs] for sp in configs])


def _epi_fit(X, E, ridge):
    """Least squares with an intercept; returns (coef, rms, rms_loo) in the
    units of E (total energy per cell)."""
    X1 = np.hstack([np.ones((len(X), 1)), X])
    A = X1.T @ X1 + ridge * np.eye(X1.shape[1])
    coef = np.linalg.solve(A, X1.T @ E)
    pred = X1 @ coef
    Hm = X1 @ np.linalg.solve(A, X1.T)            # leave-one-out via the hat matrix
    loo = (E - pred) / (1 - np.diag(Hm))
    return coef, float(np.sqrt(np.mean((pred - E) ** 2))), float(np.sqrt(np.mean(loo ** 2)))


def pair_interactions(a0, elements=("Cr", "Co", "Ni"), composition=(1, 1, 1), ncell=4,
                      nconfig=240, nshell=4, rounds=2, n_round=96, T_round=(400.0, 2500.0),
                      batch=24, seed=0, potential=POTENTIAL, ridge=1e-6,
                      verbose=True) -> PairInteractions:
    """Relaxed energies of arrangements at fixed lattice constant, fitted to
    pair interactions in the first `nshell` shells.

    The first training set is `nconfig` arrangements: a fifth random, the rest
    partly ordered or clustered by lattice MC under random interactions. Then,
    `rounds` times, `n_round` arrangements are equilibrated under the CURRENT
    fit at temperatures spread over `T_round`, relaxed, predicted BEFORE they
    are added (the out-of-sample error, recorded per round in
    meta['validation']), and added to the training set for the next fit. The
    arrangements the fitted model favours are exactly where a fit to generic
    arrangements is least trustworthy, so they are both the test and the
    correction.

    Gauge: with the composition fixed only the unlike-pair combinations are
    determined; V is reported in the gauge V_aa = 0, so V_ab(s) is the energy
    of an a-b pair relative to the average of an a-a and a b-b pair, half the
    usual ordering energy w = 2V_ab - V_aa - V_bb."""
    t0 = time.time()
    pot, cell, pmap, cp = _cell(elements, a0, ncell, potential)
    nsp = len(elements)
    n = cell.natoms
    shell_of, _ = cell.shells()
    rng = np.random.default_rng(seed)
    configs = _ordered_configs(n, composition, nconfig, cell, shell_of, rng)
    E = _relaxed_energies(cell, pmap, configs, batch, cp)
    X = _epi_features(configs, cell, shell_of, nshell, nsp)
    pairs = [(a, b) for a in range(nsp) for b in range(a + 1, nsp)]
    nb = neighbour_table(cell.pi, cell.pj, shell_of, n, nshell)

    def to_V(coef):
        V = np.zeros((nshell, nsp, nsp))
        k = 1
        for s in range(nshell):
            for (a, b) in pairs:
                V[s, a, b] = V[s, b, a] = coef[k]
                k += 1
        return V

    coef, rms, loo = _epi_fit(X, E, ridge)
    E_random = float(E[:nconfig // 5].mean())     # the random arrangements come first
    validation = []
    for r in range(rounds):
        V = to_V(coef)
        Ts = np.exp(rng.uniform(np.log(T_round[0]), np.log(T_round[1]), n_round))
        new = np.array([swap_mc(random_species(n, composition, 1, int(rng.integers(1 << 30)))[0],
                                nb, V, T, 200, int(rng.integers(1 << 30))) for T in Ts], np.int32)
        En = _relaxed_energies(cell, pmap, new, batch, cp)
        Xn = _epi_features(new, cell, shell_of, nshell, nsp)
        pred = coef[0] + Xn @ coef[1:]
        err = (pred - En) / n * 1e3
        validation.append(dict(round=r + 1, n=len(new), T=Ts.tolist(),
                               rms_meV=float(np.sqrt(np.mean(err ** 2))),
                               bias_meV=float(err.mean()),
                               E_minus_random_meV=((En - E_random) / n * 1e3).tolist(),
                               pred_minus_random_meV=((pred - E_random) / n * 1e3).tolist()))
        if verbose:
            print(f"    round {r + 1}: {len(new)} arrangements equilibrated under the fit "
                  f"(T {T_round[0]:.0f}-{T_round[1]:.0f} K): out-of-sample rms "
                  f"{validation[-1]['rms_meV']:.2f} meV/atom, bias {validation[-1]['bias_meV']:+.2f}",
                  flush=True)
        configs = np.concatenate([configs, new])
        E = np.concatenate([E, En])
        X = np.vstack([X, Xn])
        coef, rms, loo = _epi_fit(X, E, ridge)
    V = to_V(coef)
    if verbose:
        print(f"  pair interactions a0={a0:.4f}, {nshell} shells: {len(configs)} relaxed "
              f"arrangements of {n} atoms, fit rms {rms / n * 1e3:.3f} meV/atom "
              f"(LOO {loo / n * 1e3:.3f}), {time.time() - t0:.0f}s", flush=True)
    return PairInteractions(elements=tuple(elements), a0=float(a0), V=V, e0=float(coef[0]),
                            rms_train_meV=rms / n * 1e3, rms_cv_meV=loo / n * 1e3,
                            meta=dict(potential=os.path.basename(potential), ncell=ncell,
                                      nshell=nshell, nconfig=len(configs), rounds=rounds,
                                      validation=validation, E_random=E_random,
                                      energies=E.tolist(),
                                      features=X.tolist(), natoms=n))


def fcc_neighbours(ncell, nshell=3, a0=3.56):
    """Shell neighbour tables of an fcc supercell (periodic)."""
    sites, H = fcc_supercell(ncell, a0)
    n = len(sites)
    L = H[0, 0]
    d = sites[None] - sites[:, None]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=-1)
    radii = a0 * np.sqrt(np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))[:nshell]
    out = []
    for rs in radii:
        ii, jj = np.where(np.abs(r - rs) < 0.05)
        order = np.lexsort((jj, ii))
        z = np.bincount(ii, minlength=n)[0]
        out.append(np.ascontiguousarray(jj[order].reshape(n, z), np.int64))
    return out


def warren_cowley_from_nb(sp, nb, nsp):
    """alpha[s, a, b] = 1 - P(b | a, shell s) / c_b."""
    c = np.bincount(sp, minlength=nsp) / sp.size
    out = []
    for t in nb:
        P = np.zeros((nsp, nsp))
        np.add.at(P, (np.repeat(sp, t.shape[1]), sp[t.ravel()]), 1.0)
        P /= P.sum(1, keepdims=True)
        out.append(1.0 - P / c[None, :])
    return np.array(out)


def lattice_mc_alpha(V, T, composition=(1, 1, 1), ncell=8, sweeps=600, nshell_alpha=3,
                     seed=0, a0=3.56, anneal=300, T_start=3000.0, nsnap=10):
    """Equilibrium Warren-Cowley parameters of the pair-interaction model at
    temperature T by lattice Monte Carlo (swap moves, numba).

    The arrangement is cooled from `T_start` over `anneal` sweeps first, so a
    low temperature does not freeze in whatever domains form in the first few
    sweeps, then run for `sweeps` more; alpha is averaged over the last half
    of `nsnap` snapshots. Returns (alpha (nshell_alpha, nsp, nsp), last arrangement)."""
    nsp = V.shape[1]
    nb = fcc_neighbours(ncell, max(V.shape[0], nshell_alpha), a0)
    sp = random_species(nb[0].shape[0], composition, 1, seed)[0]
    if T < T_start and anneal > 0:
        for k, Tk in enumerate(np.geomspace(T_start, T, 10)):
            sp = swap_mc(sp, nb, V, Tk, max(anneal // 10, 1), seed + 1 + k)
    al = []
    per = max(sweeps // nsnap, 1)
    for k in range(nsnap):
        sp = swap_mc(sp, nb, V, T, per, seed + 100 + k)
        if k >= nsnap // 2:
            al.append(warren_cowley_from_nb(sp, nb[:nshell_alpha], nsp))
    return np.mean(al, 0), sp
