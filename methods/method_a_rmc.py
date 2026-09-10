"""Method A: parallel-tempering Reverse Monte Carlo in configuration space.

The classical (EvAX / RMCProfile-lineage) approach, done properly on GPU:

  * discrete species, composition-conserving swap moves
  * continuous single-atom displacement moves
  * incremental delta-chi: a move touches ~200 of ~27000 pairs, so the cost per
    step is ~125x below a full recomputation
  * replica exchange over a temperature ladder to escape the local minima that
    Joress et al. show are a real hazard for this problem
  * the entire proposal is GPU-resident: no host round-trip inside the loop,
    which was 58% of the step time in the first cut

Every replica proposes one move per step. Acceptance uses the R-space chi^2
over the fit window against whatever the experiment gives: |chi(R)| for
magnitude-only files, or the COMPLEX chi(R) when chi(k) was measured (a
complex target is detected from its dtype; Re and Im then count as separate
residuals and `nfit` is doubled). The target may carry several channels per
edge - one per k weight - and `ft.chan_edge` says which edge each belongs to.

Two statistical details that matter for what comes out:

  * The per-edge amplitude is a nuisance parameter with a closed-form
    least-squares value, so by default it is profiled out inside chi^2
    (`profile_scale=True`), jointly over all channels of an edge. The sampler
    then optimises exactly the quantity the R-factor reports; with a frozen
    scale the two disagree and the "best" configuration under one is not the
    best under the other.
  * Posterior samples are drawn from the rung whose temperature matches the
    likelihood. chi^2 here is sum(r^2)/nfit and acceptance is
    exp(-beta * dchi2 * nfit) = exp(-beta * dsum(r^2)); a Gaussian likelihood
    is exp(-sum(r^2)/2), so beta = 1/2 IS the posterior. The coldest rungs are
    near-greedy and their spread only measures how flat the bottom of the
    basin is, which is ~10x too small (`collect_beta`).
"""
from __future__ import annotations
import numpy as np
import cupy as cp
import time

from exafs_gpu import kernels
from exafs_gpu.scattering import NSP

_TPB = 256
_NCAND = 4  # swap-partner candidates drawn per replica per step


def _grid(n, tpb=_TPB):
    return (int((n + tpb - 1) // tpb),)


def build_reverse_index(i_idx, j_idx, natoms):
    """rev[p] = index of the pair (j,i) given pair p = (i,j)."""
    key = i_idx.astype(np.int64) * natoms + j_idx
    rkey = j_idx.astype(np.int64) * natoms + i_idx
    order = np.argsort(key)
    pos = np.searchsorted(key[order], rkey)
    rev = order[np.clip(pos, 0, key.size - 1)]
    assert np.array_equal(i_idx[rev], j_idx) and np.array_equal(j_idx[rev], i_idx), \
        "neighbour list is not symmetric"
    return rev.astype(np.int32)


class RMCSampler:
    def __init__(self, fm, ft, target, mask, sigma_R,
                 nrep=48, tmin=0.02, tmax=5.0, seed=0,
                 swap_frac=0.75, disp_sigma=0.015, max_disp=0.30,
                 nsp=None, collect_after=None, collect_every=25,
                 adapt_every=0, target_accept=0.35, profile_scale=True,
                 collect_beta=0.5, n_collect=4):
        # NOTE on the ladder: acceptance is exp(-beta*dChi2*nfit) with nfit ~ 400,
        # so the effective scale is beta*nfit. The original tmin=0.35 left the
        # coldest replica accepting moves that worsened chi^2 by ~1e-3, which in
        # ~1500 dimensions is an entropic uphill random walk: chi^2 rose from 3.4
        # to 5.6 over 20k steps. tmin=0.02 makes the cold rung near-greedy.
        self.fm, self.ft = fm, ft
        self.nsp = int(nsp) if nsp is not None else getattr(fm, "nsp", NSP)
        self.nrep = nrep
        # posterior collection and step-size adaptation (see run())
        self.collect_after = collect_after
        self.collect_every = int(collect_every)
        self.collect_beta = None if collect_beta is None else float(collect_beta)
        self.n_collect = int(n_collect)
        self.adapt_every = int(adapt_every)
        self.target_accept = float(target_accept)
        self.profile_scale = bool(profile_scale)
        self.samples = []
        self.rng = np.random.default_rng(seed)
        self.cp_rng = cp.random.default_rng(seed)
        self.swap_frac = swap_frac
        self.disp_sigma = disp_sigma
        self.max_disp = max_disp

        self.natoms = fm.natoms
        self.npairs = fm.npairs
        self.nk = fm.nk

        i_np = cp.asnumpy(fm.d_i)
        j_np = cp.asnumpy(fm.d_j)
        self.rev = build_reverse_index(i_np, j_np, self.natoms)
        counts = np.bincount(i_np, minlength=self.natoms)
        assert counts.min() == counts.max(), "expected uniform coordination"
        self.ncoord = int(counts[0])
        self.nbr_pairs = np.arange(i_np.size, dtype=np.int32).reshape(
            self.natoms, self.ncoord)
        self.d_nbr = cp.asarray(self.nbr_pairs)
        self.d_rev = cp.asarray(self.rev)

        self.maxaff = 4 * self.ncoord
        self.beta = cp.asarray(1.0 / np.geomspace(tmin, tmax, nrep))
        self._ar = cp.asarray(np.arange(nrep))

        self.complex = bool(np.iscomplexobj(target))
        self.target = cp.asarray(target)
        self.mask = cp.asarray(mask)
        self.sigR = cp.asarray(sigma_R)
        # channels: (edge, k weight) pairs, edge-major; one per edge when a
        # single k weight is fitted (then chan_edge is just arange)
        self.nchan = int(self.target.shape[0])
        self.nkw = self.nchan // self.nsp
        assert self.nkw * self.nsp == self.nchan, \
            f"{self.nchan} channels do not divide into {self.nsp} edges"
        chan_edge = getattr(ft, "chan_edge", None)
        if chan_edge is not None:
            assert np.array_equal(np.asarray(chan_edge),
                                  np.repeat(np.arange(self.nsp), self.nkw)), \
                "channels must be ordered edge-major"
        # Re and Im are separate residuals for a complex target
        self.nfit = int(mask.sum()) * self.nchan * (2 if self.complex else 1)
        # the fit window only, gathered once: (nchan, nwin)
        self._tm = cp.ascontiguousarray(self.target[:, self.mask])
        self._sm = cp.ascontiguousarray(self.sigR[:, self.mask])
        # exchange bookkeeping (per adjacent pair), reported by run()
        self.exch_attempts = np.zeros(max(nrep - 1, 1))
        self.exch_accepted = np.zeros(max(nrep - 1, 1))

        self.seed = int(seed)
        n = self.nsp
        self._k_delta = kernels.get("chi_delta", n)
        self._k_commit = kernels.get("commit_move", n)
        self._k_sel = kernels.get("propose_select", n)
        self._k_build = kernels.get("propose_build", n)
        self._k_cstate = kernels.get("commit_state", n)

    # ------------------------------------------------------------------ state
    def init_state(self, species0, pos0):
        n = self.nrep
        self.species = cp.asarray(np.repeat(species0[None], n, 0).astype(np.int32))
        self.pos = cp.asarray(np.repeat(pos0[None], n, 0).astype(np.float64))
        self.ideal = cp.asarray(np.repeat(self.fm.cell.ideal[None], n, 0))
        self.r, self.evec = self.fm.distances(self.pos)
        self.ptype = self.fm.ptypes(self.species)
        self.chi = self.fm.chi(self.pos, self.species, r=self.r, ptype=self.ptype,
                               normalise=False)
        self.count = cp.zeros((n, self.nsp), cp.float64)
        for s in range(self.nsp):
            self.count[:, s] = (self.species == s).sum(axis=1)
        self.chi2 = self._chi2(self.chi)
        self.best_chi2 = self.chi2.copy()
        self.best_sp = self.species.copy()
        self.best_pos = self.pos.copy()
        self._alloc()

    def _observe(self, chi):
        """The model on the target's footing: complex chi(R) or |chi(R)|."""
        return self.ft.to_R(chi) if self.complex else self.ft.magnitude(chi)

    def _profiled_scale(self, m):
        """Least-squares amplitude per EDGE, broadcast to (nrep, nchan).

        s_e = Re<m.t> / <m.m> summed over the fit window and over every
        channel of the edge - the same closed form `model.fit_scales` uses
        for the reported R-factor with `edge` given.
        """
        if self.complex:
            num = (m.real * self._tm.real[None] + m.imag * self._tm.imag[None]).sum(-1)
            den = (m.real ** 2 + m.imag ** 2).sum(-1)
        else:
            num = (m * self._tm[None]).sum(-1)
            den = (m * m).sum(-1)
        if self.nkw > 1:
            n = m.shape[0]
            num = num.reshape(n, self.nsp, self.nkw).sum(-1, keepdims=True)
            den = den.reshape(n, self.nsp, self.nkw).sum(-1, keepdims=True)
            num = cp.broadcast_to(num, (n, self.nsp, self.nkw)).reshape(n, self.nchan)
            den = cp.broadcast_to(den, (n, self.nsp, self.nkw)).reshape(n, self.nchan)
        return num / cp.maximum(den, 1e-30)

    def _chi2(self, chi_un, count=None):
        """Reduced chi^2 per replica from the UNNORMALISED pair sum.

        With `profile_scale` the per-edge amplitude takes its least-squares
        value for every replica at every step (`_profiled_scale`), so the
        chain and the report score the same thing. A complex target counts
        Re and Im separately, matching `nfit`.
        """
        cnt = self.count if count is None else count
        chi = chi_un / cp.maximum(cnt, 1.0)[:, :, None]
        aR = self._observe(chi)
        m = aR[:, :, self.mask]                     # (nrep, nchan, nwin)
        if self.profile_scale:
            m = m * self._profiled_scale(m)[:, :, None]
        d = (m - self._tm[None]) / self._sm[None]
        sq = (d.real ** 2 + d.imag ** 2) if self.complex else d * d
        return sq.sum(axis=(1, 2)) / self.nfit

    def scales(self, chi_un=None):
        """The profiled amplitudes, (nrep, nchan), for the current (or given) chi."""
        chi_un = self.chi if chi_un is None else chi_un
        chi = chi_un / cp.maximum(self.count, 1.0)[:, :, None]
        m = self._observe(chi)[:, :, self.mask]
        return cp.asnumpy(self._profiled_scale(m))

    # ------------------------------------------------------------------ moves
    def _alloc(self):
        n, m = self.nrep, self.maxaff
        self._a = cp.empty(n, cp.int32); self._b = cp.empty(n, cp.int32)
        self._nsa = cp.empty(n, cp.int32); self._nsb = cp.empty(n, cp.int32)
        self._npa = cp.empty((n, 3), cp.float64)
        self._isswap = cp.empty(n, cp.uint8)
        self._aff = cp.empty((n, m), cp.int32)
        self._rnew = cp.empty((n, m), cp.float64)
        self._ptnew = cp.empty((n, m), cp.int32)
        self._dchi = cp.empty((n, self.nsp, self.nk), cp.float64)
        self._stepc = 0

    # ------------------------------------------------------------------- step
    def step(self):
        n, m = self.nrep, self.maxaff
        self._stepc += 1
        self._k_sel(_grid(n), (_TPB,),
                    (self.species, self.pos, self.ideal, self._a, self._b,
                     self._nsa, self._nsb, self._npa, self._isswap,
                     np.int32(self.natoms), np.int32(n),
                     np.uint64(self.seed), np.uint64(self._stepc),
                     np.float64(self.swap_frac), np.float64(self.disp_sigma),
                     np.float64(self.max_disp)))
        self._k_build(_grid(n * m), (_TPB,),
                      (self.species, self.pos, self.d_nbr, self.d_rev,
                       self.fm.d_i, self.fm.d_j, self.fm.d_shift,
                       self._a, self._b, self._nsa, self._nsb, self._npa,
                       self._aff, self._rnew, self._ptnew,
                       np.int32(self.natoms), np.int32(self.ncoord),
                       np.int32(self.npairs), np.int32(n), np.int32(m)))
        self._k_delta((self.nk, n), (_TPB,),
                      (self.r, self.ptype, self.fm.d_shell, self._aff,
                       self._rnew, self._ptnew,
                       self.fm.d_sig2, self.fm.d_f, self.fm.d_delta,
                       self.fm.d_lam, self.fm.d_k, self.fm.d_dr, self.fm.d_c3,
                       self._dchi,
                       np.int32(self.npairs), np.int32(self.nk),
                       np.int32(n), np.int32(m), np.float64(self.fm.s02)),
                      shared_mem=self.nsp * _TPB * 8)
        trial = self.chi + self._dchi
        c2 = self._chi2(trial)
        dE = c2 - self.chi2
        u = self.cp_rng.random(n)
        acc = (dE <= 0) | (u < cp.exp(-cp.minimum(self.beta * dE * self.nfit, 60.0)))
        accb = acc.astype(cp.uint8)

        self._k_commit(_grid(m * n), (_TPB,),
                       (self._aff, self._rnew, self._ptnew, accb,
                        self.r, self.ptype,
                        np.int32(self.npairs), np.int32(n), np.int32(m)))
        self._k_cstate(_grid(n), (_TPB,),
                       (accb, self._a, self._b, self._nsa, self._nsb, self._npa,
                        self._isswap, self.species, self.pos,
                        np.int32(self.natoms), np.int32(n)))
        self.chi = cp.where(acc[:, None, None], trial, self.chi)
        self.chi2 = cp.where(acc, c2, self.chi2)

        # Retain the best state ever visited. Without this the answer is
        # whatever the chain happened to be holding when the budget ran out,
        # which is not the same thing as the best fit found.
        imp = self.chi2 < self.best_chi2
        self.best_chi2 = cp.where(imp, self.chi2, self.best_chi2)
        self.best_sp = cp.where(imp[:, None], self.species, self.best_sp)
        self.best_pos = cp.where(imp[:, None, None], self.pos, self.best_pos)
        return acc

    def _refresh_counts(self):
        for s in range(self.nsp):
            self.count[:, s] = (self.species == s).sum(axis=1)

    def replica_exchange(self, return_rates=False):
        """Swap adjacent rungs of the temperature ladder.

        Detailed balance for the joint distribution prod_i exp(-beta_i E_i)
        gives the acceptance min(1, exp[(beta_i - beta_j)(E_i - E_j)]) with E
        the energy CURRENTLY held at each rung: a cold rung (large beta)
        holding a worse state than its hotter neighbour always swaps. An
        earlier version had the energies the other way round, which sent
        good states up the ladder to be melted.
        """
        c2 = cp.asnumpy(self.chi2)
        beta = cp.asnumpy(self.beta)
        order = np.arange(self.nrep)
        start = self.rng.integers(0, 2)
        nsw = 0
        rates = np.full(self.nrep - 1, np.nan)
        for i in range(start, self.nrep - 1, 2):
            d = (beta[i] - beta[i + 1]) * (c2[order[i]] - c2[order[i + 1]]) * self.nfit
            ok = d >= 0 or self.rng.random() < np.exp(max(d, -60.0))
            rates[i] = 1.0 if ok else 0.0
            self.exch_attempts[i] += 1
            if ok:
                order[i], order[i + 1] = order[i + 1], order[i]
                nsw += 1
                self.exch_accepted[i] += 1
        if nsw:
            ix = cp.asarray(order)
            self.chi = self.chi[ix]; self.chi2 = self.chi2[ix]
            self.pos = self.pos[ix]; self.species = self.species[ix]
            self.r = self.r[ix]; self.ptype = self.ptype[ix]
            self.count = self.count[ix]
            self.best_chi2 = self.best_chi2[ix]
            self.best_sp = self.best_sp[ix]; self.best_pos = self.best_pos[ix]
        if return_rates:
            # unattempted pairs (alternating scheme) carry the running mean
            filled = np.where(np.isnan(rates),
                              np.nanmean(rates) if np.any(~np.isnan(rates)) else 0.3,
                              rates)
            return nsw, filled
        return nsw

    # -------------------------------------------------- ladder / step tuning
    def adapt(self, window_acc, window_exch):
        """Retune the displacement step and the temperature ladder.

        Displacement step: multiplicative adaptation toward `target_accept`.
        Ladder: geometric spacing is only optimal if the exchange rate is
        roughly uniform between rungs. Where a neighbouring pair exchanges too
        rarely the rungs are too far apart in beta, so we contract that gap and
        expand the ones that exchange too freely, keeping the endpoints fixed.
        """
        if window_acc is not None and window_acc > 0:
            self.disp_sigma *= float(np.clip(
                (window_acc / self.target_accept) ** 0.35, 0.7, 1.4))
            self.disp_sigma = float(np.clip(self.disp_sigma, 1e-4, 0.15))
        if window_exch is None or self.nrep < 4:
            return
        b = np.log(cp.asnumpy(self.beta))
        gaps = np.diff(b)
        rate = np.clip(window_exch, 0.02, 0.98)
        # widen where exchange is easy, narrow where it is hard
        scale = (rate / max(rate.mean(), 1e-6)) ** 0.5
        gaps = gaps * scale
        gaps *= (b[-1] - b[0]) / gaps.sum()
        self.beta = cp.asarray(np.exp(np.concatenate([[b[0]], b[0] + np.cumsum(gaps)])))

    # ------------------------------------------------------------------- loop
    def run(self, nsteps, exchange_every=25, log_every=500, callback=None):
        """Run the chain.

        If `collect_after` is set, configurations from the coldest replicas are
        stored every `collect_every` steps after that point. Those samples are
        the posterior: given a genuinely flat direction in parameter space, the
        single best-chi2 configuration is not the answer, the spread is.
        """
        hist = []
        acc_run = cp.zeros(1)
        t0 = time.time()
        win_acc, win_exch = [], []
        for it in range(nsteps):
            acc = self.step()
            acc_run = acc_run + acc.mean()
            if self.adapt_every:
                win_acc.append(acc.mean())
            if exchange_every and (it + 1) % exchange_every == 0:
                nsw, per_pair = self.replica_exchange(return_rates=True)
                if self.adapt_every:
                    win_exch.append(per_pair)
            if self.adapt_every and (it + 1) % self.adapt_every == 0:
                wa = float(cp.asnumpy(cp.stack(win_acc)).mean()) if win_acc else None
                we = np.mean(win_exch, axis=0) if win_exch else None
                self.adapt(wa, we)
                win_acc, win_exch = [], []
            if (self.collect_after is not None and it >= self.collect_after
                    and (it + 1) % self.collect_every == 0):
                self._collect()
            if (it + 1) % log_every == 0 or it == 0:
                best = float(self.best_chi2.min())
                hist.append((it + 1, best, float(acc_run) / (it + 1),
                             time.time() - t0))
                if callback:
                    callback(it + 1, best, hist[-1][2])
        self._refresh_counts()
        self.chi2 = self._chi2(self.chi)
        cp.cuda.Stream.null.synchronize()
        # verify the retained best really is what it claims
        bsp, bpos, bc2 = self.best()
        chk = self.fm.chi(cp.asarray(bpos[None]),
                          cp.asarray(bsp[None].astype(np.int32)), normalise=False)
        cnt = cp.zeros((1, self.nsp), cp.float64)
        for s_ in range(self.nsp):
            cnt[:, s_] = (cp.asarray(bsp[None].astype(np.int32)) == s_).sum(axis=1)
        self.best_verified = float(self._chi2(chk, cnt)[0])
        self.walltime = time.time() - t0
        self.history = hist
        with np.errstate(invalid="ignore", divide="ignore"):
            self.exchange_rate = np.where(self.exch_attempts > 0,
                                          self.exch_accepted / np.maximum(self.exch_attempts, 1),
                                          np.nan)
        return hist

    def collect_rungs(self):
        """Indices of the rungs posterior samples are drawn from.

        The `n_collect` rungs whose beta is nearest `collect_beta` (in log
        spacing); the coldest rungs if `collect_beta` is None.
        """
        beta = cp.asnumpy(self.beta)
        k = min(self.n_collect, self.nrep)
        if self.collect_beta is None:
            return np.argsort(beta)[::-1][:k]                 # largest beta = coldest
        return np.argsort(np.abs(np.log(beta) - np.log(self.collect_beta)))[:k]

    def _collect(self):
        """Store configurations from the likelihood-temperature rungs."""
        idx = self.collect_rungs()
        sp = cp.asnumpy(self.species[cp.asarray(idx)])
        c2 = cp.asnumpy(self.chi2[cp.asarray(idx)])
        for a, b in zip(sp, c2):
            self.samples.append((a.astype(np.int32), float(b)))

    def posterior_wc(self, shell_r, i_idx, j_idx, shift, cell, nsp=None, tol=0.15):
        """Warren-Cowley parameters over the collected posterior samples.

        Returns (mean, std, n). The spread is the honest uncertainty: for a
        direction the data does not constrain it will be large, which is exactly
        the information a single best-fit configuration hides.
        """
        from exafs_gpu.lattice import warren_cowley
        if not self.samples:
            raise RuntimeError("no samples collected; set collect_after in the constructor")
        n = nsp if nsp is not None else self.nsp
        arr = np.array([warren_cowley(sp, i_idx, j_idx, shift, cell, shell_r,
                                      tol=tol, nsp=n)[np.triu_indices(n)]
                        for sp, _ in self.samples])
        return arr.mean(0), arr.std(0), len(arr)

    def best(self):
        b = int(cp.asnumpy(self.best_chi2).argmin())
        return (cp.asnumpy(self.best_sp[b]).astype(np.int32),
                cp.asnumpy(self.best_pos[b]), float(self.best_chi2[b]))
