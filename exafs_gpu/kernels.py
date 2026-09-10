"""Compiled CUDA kernels (NVRTC via CuPy RawModule) for the EXAFS forward model
and its analytic adjoint.

Layout conventions
------------------
  nconf   batch dimension: MC replicas, or ensemble snapshots, or both
  npairs  ordered neighbour pairs, fixed topology, shared across the batch
  nk      wavenumber grid points
  NSP=3   species (Cr, Co, Ni)

  r      (nconf, npairs)      pair distances
  ptype  (nconf, npairs)      absorber*NSP + scatterer
  shell  (npairs,)            coordination-shell index of each pair (fixed topology)
  sig2   (npairs,)            thermal MSRD per pair (analytic Debye-Waller)
  drtab  (nshell*NSP*NSP,)    bond-length offset by (shell, pair type): the
                              size-mismatch term, r_eff = r + dr[shell][type]
  c3tab  (nshell,)            third cumulant by shell: phase -(4/3) k^3 C3
  chi    (nconf, NSP, nk)     unnormalised per-absorber EXAFS

The pair term evaluated everywhere below is the standard cumulant expansion
to third order:

    s02 f_b(k) / (k r_eff^2) exp(-2 r_eff/lambda(k) - 2 k^2 sigma^2)
        * sin(2 k r_eff + delta_ab(k) - 4/3 k^3 C3)

with r_eff = r + dr[shell][ab]. Both dr and C3 default to zero, in which case
this is exactly the harmonic single-scattering model.

Performance notes
-----------------
* Distances are computed once per configuration, then reused for all nk. Folding
  the distance calculation into the chi kernel would repeat it nk times.
* blockIdx.x indexes k, so the 13 k-dependent table values (3 amplitudes,
  9 phases, lambda) are loaded into shared memory once per block and reused
  across every pair. The inner loop is then pure arithmetic on one global read.
* Each block owns one (k, conf) pair, so the per-absorber reduction is entirely
  within-block: no global atomics in the forward pass.
"""
from __future__ import annotations
import cupy as cp

NSP = 3

_SRC = r"""
// NSP is injected at compile time by module(nsp) via -DNSP=<n>, so the same
// source serves any number of species while keeping every per-species array a
// compile-time size (and therefore in registers, not local memory).
#ifndef NSP
#define NSP 3
#endif

// ---------------------------------------------------------------- distances
extern "C" __global__ void compute_r(
    const double* __restrict__ pos,     // (nconf, natoms, 3)
    const int*    __restrict__ iidx,    // (npairs)
    const int*    __restrict__ jidx,    // (npairs)
    const double* __restrict__ shift,   // (npairs, 3)
    double*       __restrict__ r,       // (nconf, npairs)
    double*       __restrict__ evec,    // (nconf, npairs, 3) unit vectors
    int npairs, int natoms, int nconf)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long tot = (long)npairs * nconf;
    if (tid >= tot) return;
    int ic = (int)(tid / npairs);
    int p  = (int)(tid % npairs);

    int i = iidx[p], j = jidx[p];
    const double* pi = pos + ((long)ic * natoms + i) * 3;
    const double* pj = pos + ((long)ic * natoms + j) * 3;
    double dx = pj[0] + shift[3*p+0] - pi[0];
    double dy = pj[1] + shift[3*p+1] - pi[1];
    double dz = pj[2] + shift[3*p+2] - pi[2];
    double rr = sqrt(dx*dx + dy*dy + dz*dz);
    r[tid] = rr;
    double inv = 1.0 / rr;
    evec[3*tid+0] = dx*inv; evec[3*tid+1] = dy*inv; evec[3*tid+2] = dz*inv;
}

// ------------------------------------------------------------- pair types
extern "C" __global__ void build_ptype(
    const int* __restrict__ species,   // (nconf, natoms)
    const int* __restrict__ iidx,
    const int* __restrict__ jidx,
    int*       __restrict__ ptype,     // (nconf, npairs)
    int npairs, int natoms, int nconf)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long tot = (long)npairs * nconf;
    if (tid >= tot) return;
    int ic = (int)(tid / npairs);
    int p  = (int)(tid % npairs);
    int sa = species[(long)ic*natoms + iidx[p]];
    int sb = species[(long)ic*natoms + jidx[p]];
    ptype[tid] = sa * NSP + sb;
}

// ------------------------------------------------------------ forward chi
extern "C" __global__ void chi_forward(
    const double* __restrict__ r,       // (nconf, npairs)
    const int*    __restrict__ ptype,   // (nconf, npairs)
    const int*    __restrict__ shell,   // (npairs)
    const double* __restrict__ sig2,    // (npairs)
    const double* __restrict__ ftab,    // (NSP, nk)
    const double* __restrict__ dtab,    // (NSP*NSP, nk)
    const double* __restrict__ lamtab,  // (nk)
    const double* __restrict__ ktab,    // (nk)
    const double* __restrict__ drtab,   // (nshell*NSP*NSP)
    const double* __restrict__ c3tab,   // (nshell)
    double*       __restrict__ chi,     // (nconf, NSP, nk)
    int npairs, int nk, int nconf, double s02)
{
    int ik = blockIdx.x;
    int ic = blockIdx.y;
    if (ik >= nk || ic >= nconf) return;

    __shared__ double sf[NSP], sd[NSP*NSP], sk, slam;
    if (threadIdx.x < NSP)     sf[threadIdx.x] = ftab[threadIdx.x*nk + ik];
    if (threadIdx.x < NSP*NSP) sd[threadIdx.x] = dtab[threadIdx.x*nk + ik];
    if (threadIdx.x == 0) { sk = ktab[ik]; slam = lamtab[ik]; }
    __syncthreads();

    double acc[NSP];
    #pragma unroll
    for (int s = 0; s < NSP; ++s) acc[s] = 0.0;
    const double* rc = r + (long)ic * npairs;
    const int*    tc = ptype + (long)ic * npairs;
    double dw_k = -2.0 * sk * sk;
    double c3k  = (4.0 / 3.0) * sk * sk * sk;

    for (int p = threadIdx.x; p < npairs; p += blockDim.x) {
        int t = tc[p];
        int sh = shell[p];
        double rr = rc[p] + drtab[sh*NSP*NSP + t];
        int sa = t / NSP, sb = t - sa*NSP;
        double amp = s02 * sf[sb] / (sk * rr * rr)
                   * exp(-2.0*rr/slam + dw_k*sig2[p]);
        double v = amp * sin(2.0*sk*rr + sd[t] - c3k*c3tab[sh]);
        // unrolled predicated add keeps acc[] in registers despite the
        // runtime index sa
        #pragma unroll
        for (int s = 0; s < NSP; ++s) if (sa == s) acc[s] += v;
    }

    extern __shared__ double sm[];
    int T = blockDim.x, tx = threadIdx.x;
    #pragma unroll
    for (int s = 0; s < NSP; ++s) sm[s*T + tx] = acc[s];
    __syncthreads();
    for (int s = T/2; s > 0; s >>= 1) {
        if (tx < s) {
            #pragma unroll
            for (int a = 0; a < NSP; ++a) sm[a*T+tx] += sm[a*T+tx+s];
        }
        __syncthreads();
    }
    if (tx == 0) {
        long base = ((long)ic * NSP) * nk + ik;
        #pragma unroll
        for (int a = 0; a < NSP; ++a) chi[base + a*nk] = sm[a*T];
    }
}

// ------------------------------------------------- adjoint: dL/dchi -> dL/dr
extern "C" __global__ void chi_adjoint_r(
    const double* __restrict__ r,
    const int*    __restrict__ ptype,
    const int*    __restrict__ shell,
    const double* __restrict__ sig2,
    const double* __restrict__ ftab,
    const double* __restrict__ dtab,
    const double* __restrict__ lamtab,
    const double* __restrict__ ktab,
    const double* __restrict__ drtab,
    const double* __restrict__ c3tab,
    const double* __restrict__ gchi,    // (nconf, NSP, nk)
    double*       __restrict__ gr,      // (nconf, npairs)
    int npairs, int nk, int nconf, double s02)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long tot = (long)npairs * nconf;
    if (tid >= tot) return;
    int ic = (int)(tid / npairs);
    int p  = (int)(tid % npairs);

    int t = ptype[tid];
    int sh = shell[p];
    double rr = r[tid] + drtab[sh*NSP*NSP + t];
    double c3 = c3tab[sh];
    int sa = t / NSP, sb = t - sa*NSP;
    double s2 = sig2[p];
    const double* g = gchi + ((long)ic * NSP + sa) * nk;

    double acc = 0.0;
    for (int ik = 0; ik < nk; ++ik) {
        double gk = g[ik];
        if (gk == 0.0) continue;
        double kv = ktab[ik], lam = lamtab[ik];
        double amp = s02 * ftab[sb*nk + ik] / (kv*rr*rr)
                   * exp(-2.0*rr/lam - 2.0*kv*kv*s2);
        double th = 2.0*kv*rr + dtab[t*nk + ik] - (4.0/3.0)*kv*kv*kv*c3;
        // d/dr [ amp(r) * sin(th(r)) ]
        double dTdr = amp * ((-2.0/rr - 2.0/lam) * sin(th) + 2.0*kv*cos(th));
        acc += gk * dTdr;
    }
    gr[tid] = acc;
}

// ------------------------------------------- scatter dL/dr -> dL/dpositions
extern "C" __global__ void scatter_force(
    const double* __restrict__ gr,     // (nconf, npairs)
    const double* __restrict__ evec,   // (nconf, npairs, 3)
    const int*    __restrict__ iidx,
    const int*    __restrict__ jidx,
    double*       __restrict__ gpos,   // (nconf, natoms, 3)
    int npairs, int natoms, int nconf)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long tot = (long)npairs * nconf;
    if (tid >= tot) return;
    int ic = (int)(tid / npairs);
    int p  = (int)(tid % npairs);

    double g = gr[tid];
    if (g == 0.0) return;
    int i = iidx[p], j = jidx[p];
    double* gi = gpos + ((long)ic*natoms + i)*3;
    double* gj = gpos + ((long)ic*natoms + j)*3;
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        double ge = g * evec[3*tid + a];
        atomicAdd(gj + a,  ge);
        atomicAdd(gi + a, -ge);
    }
}

// ------------------------------------------------- incremental delta-chi (MC)
// A swap or single-atom move touches only the pairs adjacent to the moved
// atoms: ~200 of ~27000. Recomputing the whole spectrum per MC step would be
// ~125x more work, so this is what makes configuration-space MC competitive.
extern "C" __global__ void chi_delta(
    const double* __restrict__ r_old,    // (nrep, npairs)
    const int*    __restrict__ pt_old,   // (nrep, npairs)
    const int*    __restrict__ shell,    // (npairs)
    const int*    __restrict__ aff,      // (nrep, maxaff) pair ids, -1 = padding
    const double* __restrict__ r_new,    // (nrep, maxaff)
    const int*    __restrict__ pt_new,   // (nrep, maxaff)
    const double* __restrict__ sig2,
    const double* __restrict__ ftab,
    const double* __restrict__ dtab,
    const double* __restrict__ lamtab,
    const double* __restrict__ ktab,
    const double* __restrict__ drtab,
    const double* __restrict__ c3tab,
    double*       __restrict__ dchi,     // (nrep, NSP, nk)
    int npairs, int nk, int nrep, int maxaff, double s02)
{
    int ik = blockIdx.x;
    int ic = blockIdx.y;
    if (ik >= nk || ic >= nrep) return;

    __shared__ double sf[NSP], sd[NSP*NSP], sk, slam;
    if (threadIdx.x < NSP)     sf[threadIdx.x] = ftab[threadIdx.x*nk + ik];
    if (threadIdx.x < NSP*NSP) sd[threadIdx.x] = dtab[threadIdx.x*nk + ik];
    if (threadIdx.x == 0) { sk = ktab[ik]; slam = lamtab[ik]; }
    __syncthreads();

    double acc[NSP];
    #pragma unroll
    for (int s = 0; s < NSP; ++s) acc[s] = 0.0;
    const int*    ac = aff    + (long)ic * maxaff;
    const double* rn = r_new  + (long)ic * maxaff;
    const int*    tn = pt_new + (long)ic * maxaff;
    const double* ro = r_old  + (long)ic * npairs;
    const int*    to = pt_old + (long)ic * npairs;
    double dw_k = -2.0 * sk * sk;
    double c3k  = (4.0 / 3.0) * sk * sk * sk;

    for (int q = threadIdx.x; q < maxaff; q += blockDim.x) {
        int p = ac[q];
        if (p < 0) continue;
        double s2 = sig2[p];
        int sh = shell[p];
        double ph3 = c3k * c3tab[sh];
        const double* drs = drtab + sh*NSP*NSP;
        // new contribution
        int t = tn[q];
        double rr = rn[q] + drs[t];
        int sa = t / NSP, sb = t - sa*NSP;
        double v = s02 * sf[sb] / (sk*rr*rr) * exp(-2.0*rr/slam + dw_k*s2)
                 * sin(2.0*sk*rr + sd[t] - ph3);
        #pragma unroll
        for (int s = 0; s < NSP; ++s) if (sa == s) acc[s] += v;
        // minus old contribution
        t = to[p];
        rr = ro[p] + drs[t];
        sa = t / NSP; sb = t - sa*NSP;
        v = s02 * sf[sb] / (sk*rr*rr) * exp(-2.0*rr/slam + dw_k*s2)
          * sin(2.0*sk*rr + sd[t] - ph3);
        #pragma unroll
        for (int s = 0; s < NSP; ++s) if (sa == s) acc[s] -= v;
    }

    extern __shared__ double sm[];
    int T = blockDim.x, tx = threadIdx.x;
    #pragma unroll
    for (int s = 0; s < NSP; ++s) sm[s*T + tx] = acc[s];
    __syncthreads();
    for (int s = T/2; s > 0; s >>= 1) {
        if (tx < s) {
            #pragma unroll
            for (int a = 0; a < NSP; ++a) sm[a*T+tx] += sm[a*T+tx+s];
        }
        __syncthreads();
    }
    if (tx == 0) {
        long b = ((long)ic * NSP) * nk + ik;
        #pragma unroll
        for (int a = 0; a < NSP; ++a) dchi[b + a*nk] = sm[a*T];
    }
}

// commit an accepted move: write the new r / ptype into the full arrays
extern "C" __global__ void commit_move(
    const int*    __restrict__ aff,
    const double* __restrict__ r_new,
    const int*    __restrict__ pt_new,
    const unsigned char* __restrict__ accept,   // (nrep)
    double* __restrict__ r_old,
    int*    __restrict__ pt_old,
    int npairs, int nrep, int maxaff)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long)maxaff * nrep) return;
    int ic = (int)(tid / maxaff);
    if (!accept[ic]) return;
    int q = (int)(tid % maxaff);
    int p = aff[(long)ic*maxaff + q];
    if (p < 0) return;
    r_old[(long)ic*npairs + p]  = r_new[(long)ic*maxaff + q];
    pt_old[(long)ic*npairs + p] = pt_new[(long)ic*maxaff + q];
}

// ---------------------------------------------------- GPU-resident proposals
// The first implementation built proposals from ~30 small CuPy ops and was
// launch-overhead bound (57% of step time). These two kernels replace all of it.

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
__device__ __forceinline__ double u01(unsigned long long h) {
    return (double)(h >> 11) * (1.0 / 9007199254740992.0);
}

extern "C" __global__ void propose_select(
    const int*    __restrict__ species,   // (nrep, natoms)
    const double* __restrict__ pos,       // (nrep, natoms, 3)
    const double* __restrict__ ideal,
    int*    __restrict__ a_out,
    int*    __restrict__ b_out,
    int*    __restrict__ newsa,
    int*    __restrict__ newsb,
    double* __restrict__ newpa,           // (nrep, 3)
    unsigned char* __restrict__ isswap,
    int natoms, int nrep,
    unsigned long long seed, unsigned long long stepc,
    double swap_frac, double disp_sigma, double max_disp)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nrep) return;
    unsigned long long h = splitmix64(seed ^ (stepc * 0x9E3779B97F4A7C15ULL)
                                           ^ ((unsigned long long)c << 32));
    h = splitmix64(h);
    int sw = (u01(h) < swap_frac) ? 1 : 0;
    h = splitmix64(h);
    int a = (int)(u01(h) * natoms); if (a >= natoms) a = natoms - 1;
    const int* S = species + (long)c * natoms;
    int sa = S[a], b = a, sb = sa;

    if (sw) {
        for (int t = 0; t < 8; ++t) {
            h = splitmix64(h);
            int cand = (int)(u01(h) * natoms); if (cand >= natoms) cand = natoms - 1;
            if (S[cand] != sa) { b = cand; sb = S[cand]; break; }
        }
    }
    int real_swap = (sw && b != a) ? 1 : 0;
    isswap[c] = (unsigned char)real_swap;
    a_out[c] = a; b_out[c] = b;
    newsa[c] = real_swap ? sb : sa;
    newsb[c] = real_swap ? sa : sb;

    const double* P  = pos   + ((long)c * natoms + a) * 3;
    const double* Id = ideal + ((long)c * natoms + a) * 3;
    double np3[3];
    if (real_swap) {
        np3[0] = P[0]; np3[1] = P[1]; np3[2] = P[2];
    } else {
        double g[3];
        for (int t = 0; t < 3; t += 2) {
            h = splitmix64(h); double u1 = u01(h);
            h = splitmix64(h); double u2 = u01(h);
            u1 = fmax(u1, 1e-300);
            double rad = sqrt(-2.0 * log(u1));
            g[t] = rad * cos(6.283185307179586 * u2);
            if (t + 1 < 3) g[t+1] = rad * sin(6.283185307179586 * u2);
        }
        double ox = P[0] + g[0]*disp_sigma - Id[0];
        double oy = P[1] + g[1]*disp_sigma - Id[1];
        double oz = P[2] + g[2]*disp_sigma - Id[2];
        double m = sqrt(ox*ox + oy*oy + oz*oz);
        double sc = (m > max_disp) ? (max_disp / m) : 1.0;
        np3[0] = Id[0] + ox*sc; np3[1] = Id[1] + oy*sc; np3[2] = Id[2] + oz*sc;
    }
    newpa[3*c] = np3[0]; newpa[3*c+1] = np3[1]; newpa[3*c+2] = np3[2];
}

extern "C" __global__ void propose_build(
    const int*    __restrict__ species,
    const double* __restrict__ pos,
    const int*    __restrict__ nbr,      // (natoms, ncoord)
    const int*    __restrict__ rev,      // (npairs)
    const int*    __restrict__ iidx,
    const int*    __restrict__ jidx,
    const double* __restrict__ shift,
    const int*    __restrict__ a_in,
    const int*    __restrict__ b_in,
    const int*    __restrict__ newsa,
    const int*    __restrict__ newsb,
    const double* __restrict__ newpa,
    int*    __restrict__ aff,
    double* __restrict__ r_new,
    int*    __restrict__ pt_new,
    int natoms, int ncoord, int npairs, int nrep, int maxaff)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long)maxaff * nrep) return;
    int c = (int)(tid / maxaff), q = (int)(tid % maxaff);
    int a = a_in[c], b = b_in[c];

    int blk = q / ncoord, o = q % ncoord;
    int p;
    if      (blk == 0) p = nbr[(long)a*ncoord + o];
    else if (blk == 1) p = rev[nbr[(long)a*ncoord + o]];
    else if (blk == 2) p = nbr[(long)b*ncoord + o];
    else               p = rev[nbr[(long)b*ncoord + o]];

    // Deduplicate. Blocks 0/1 cover every edge touching atom a in both
    // directions and are internally disjoint. Blocks 2/3 must therefore drop
    // (i) everything when b == a (a displacement move, where they are an exact
    // copy of blocks 0/1) and (ii) any pair whose other endpoint is a, which is
    // already covered above when a and b are neighbours. Without this the delta
    // kernel double-counts those pairs and the tracked chi drifts.
    if (blk >= 2) {
        if (b == a) p = -1;
        else {
            int other = (blk == 2) ? jidx[p] : iidx[p];
            if (other == a) p = -1;
        }
    }
    aff[tid] = p;
    if (p < 0) { r_new[tid] = 0.0; pt_new[tid] = 0; return; }

    int i = iidx[p], j = jidx[p];
    const double* P = pos + (long)c * natoms * 3;
    const double* NP = newpa + 3*c;
    double pix, piy, piz, pjx, pjy, pjz;
    if (i == a) { pix = NP[0]; piy = NP[1]; piz = NP[2]; }
    else        { pix = P[3*i]; piy = P[3*i+1]; piz = P[3*i+2]; }
    if (j == a) { pjx = NP[0]; pjy = NP[1]; pjz = NP[2]; }
    else        { pjx = P[3*j]; pjy = P[3*j+1]; pjz = P[3*j+2]; }
    double dx = pjx + shift[3*p+0] - pix;
    double dy = pjy + shift[3*p+1] - piy;
    double dz = pjz + shift[3*p+2] - piz;
    r_new[tid] = sqrt(dx*dx + dy*dy + dz*dz);

    const int* S = species + (long)c * natoms;
    int si = (i == a) ? newsa[c] : ((i == b) ? newsb[c] : S[i]);
    int sj = (j == a) ? newsa[c] : ((j == b) ? newsb[c] : S[j]);
    pt_new[tid] = si * NSP + sj;
}

extern "C" __global__ void commit_state(
    const unsigned char* __restrict__ accept,
    const int*    __restrict__ a_in,
    const int*    __restrict__ b_in,
    const int*    __restrict__ newsa,
    const int*    __restrict__ newsb,
    const double* __restrict__ newpa,
    const unsigned char* __restrict__ isswap,
    int*    __restrict__ species,
    double* __restrict__ pos,
    int natoms, int nrep)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nrep || !accept[c]) return;
    int a = a_in[c], b = b_in[c];
    species[(long)c*natoms + a] = newsa[c];
    species[(long)c*natoms + b] = newsb[c];
    if (!isswap[c]) {
        double* P = pos + ((long)c*natoms + a)*3;
        P[0] = newpa[3*c]; P[1] = newpa[3*c+1]; P[2] = newpa[3*c+2];
    }
}

// --------------------------------------- soft-occupancy forward (Method B)
// chi is bilinear in the per-site species weights w. Exploited by the
// relaxed-chemistry optimiser: exact gradients, no discrete search.
extern "C" __global__ void chi_forward_soft(
    const double* __restrict__ r,       // (nconf, npairs)
    const double* __restrict__ w,       // (nconf, natoms, NSP) simplex weights
    const int*    __restrict__ iidx,
    const int*    __restrict__ jidx,
    const double* __restrict__ sig2,
    const double* __restrict__ ftab,
    const double* __restrict__ cosd,    // cos(delta), precomputed
    const double* __restrict__ sind,    // sin(delta), precomputed
    const double* __restrict__ lamtab,
    const double* __restrict__ ktab,
    double*       __restrict__ chi,     // (nconf, NSP, nk)
    int npairs, int nk, int nconf, int natoms, double s02)
{
    int ik = blockIdx.x;
    int ic = blockIdx.y;
    if (ik >= nk || ic >= nconf) return;

    // fc[t] = f_sb*cos(delta_t), fs[t] = f_sb*sin(delta_t). Folding amplitude
    // into the phase tables makes the inner loop pure multiply-add and lets us
    // call sincos once per (pair, k) instead of NSP*NSP times.
    __shared__ double fc[NSP*NSP], fs[NSP*NSP], sk, slam;
    if (threadIdx.x < NSP*NSP) {
        int t = threadIdx.x, sb = t % NSP;
        double fv = ftab[sb*nk + ik];
        fc[t] = fv * cosd[t*nk + ik];
        fs[t] = fv * sind[t*nk + ik];
    }
    if (threadIdx.x == 0) { sk = ktab[ik]; slam = lamtab[ik]; }
    __syncthreads();

    double acc[NSP] = {0.0};
    const double* rc = r + (long)ic * npairs;
    const double* wc = w + (long)ic * natoms * NSP;
    double dw_k = -2.0 * sk * sk;

    for (int p = threadIdx.x; p < npairs; p += blockDim.x) {
        double rr = rc[p];
        double base = s02 / (sk*rr*rr) * exp(-2.0*rr/slam + dw_k*sig2[p]);
        double sp_, cp_;
        sincos(2.0*sk*rr, &sp_, &cp_);
        const double* wi = wc + (long)iidx[p]*NSP;
        const double* wj = wc + (long)jidx[p]*NSP;
        #pragma unroll
        for (int sa = 0; sa < NSP; ++sa) {
            double wia = wi[sa];
            if (wia == 0.0) continue;
            double s = 0.0;
            #pragma unroll
            for (int sb = 0; sb < NSP; ++sb) {
                int t = sa*NSP + sb;
                s += wj[sb] * (sp_*fc[t] + cp_*fs[t]);
            }
            acc[sa] += wia * base * s;
        }
    }

    extern __shared__ double sm[];
    int T = blockDim.x, tx = threadIdx.x;
    #pragma unroll
    for (int s = 0; s < NSP; ++s) sm[s*T + tx] = acc[s];
    __syncthreads();
    for (int s = T/2; s > 0; s >>= 1) {
        if (tx < s) {
            #pragma unroll
            for (int a = 0; a < NSP; ++a) sm[a*T+tx] += sm[a*T+tx+s];
        }
        __syncthreads();
    }
    if (tx == 0) {
        long b = ((long)ic * NSP) * nk + ik;
        #pragma unroll
        for (int a = 0; a < NSP; ++a) chi[b + a*nk] = sm[a*T];
    }
}

// ---------------------- soft-occupancy adjoint: dL/dchi -> dL/dw and dL/dr
extern "C" __global__ void chi_adjoint_soft(
    const double* __restrict__ r,
    const double* __restrict__ w,
    const int*    __restrict__ iidx,
    const int*    __restrict__ jidx,
    const double* __restrict__ sig2,
    const double* __restrict__ ftab,
    const double* __restrict__ cosd,
    const double* __restrict__ sind,
    const double* __restrict__ lamtab,
    const double* __restrict__ ktab,
    const double* __restrict__ gchi,
    double*       __restrict__ gr,     // (nconf, npairs)
    double*       __restrict__ gw,     // (nconf, natoms, NSP)
    int npairs, int nk, int nconf, int natoms, double s02)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long tot = (long)npairs * nconf;
    if (tid >= tot) return;
    int ic = (int)(tid / npairs);
    int p  = (int)(tid % npairs);

    double rr = r[tid];
    double s2 = sig2[p];
    const double* wc = w + (long)ic * natoms * NSP;
    const double* wi = wc + (long)iidx[p]*NSP;
    const double* wj = wc + (long)jidx[p]*NSP;
    const double* gc = gchi + (long)ic * NSP * nk;

    double gwi[NSP] = {0.0};
    double gwj[NSP] = {0.0};
    double gracc = 0.0;

    for (int ik = 0; ik < nk; ++ik) {
        double kv = ktab[ik], lam = lamtab[ik];
        double base = s02 / (kv*rr*rr) * exp(-2.0*rr/lam - 2.0*kv*kv*s2);
        double dbase = base * (-2.0/rr - 2.0/lam);
        double sp_, cp_;
        sincos(2.0*kv*rr, &sp_, &cp_);
        #pragma unroll
        for (int sa = 0; sa < NSP; ++sa) {
            double gk = gc[sa*nk + ik];
            if (gk == 0.0) continue;
            double sterm = 0.0, dsterm = 0.0;
            #pragma unroll
            for (int sb = 0; sb < NSP; ++sb) {
                int t = sa*NSP + sb;
                double fv = ftab[sb*nk + ik];
                double fcv = fv * cosd[t*nk + ik], fsv = fv * sind[t*nk + ik];
                double sn = sp_*fcv + cp_*fsv;   // f*sin(2kr+delta)
                double cs = cp_*fcv - sp_*fsv;   // f*cos(2kr+delta)
                sterm  += wj[sb] * sn;
                dsterm += wj[sb] * cs * 2.0*kv;
                gwj[sb] += gk * wi[sa] * base * sn;
            }
            gwi[sa] += gk * base * sterm;
            gracc   += gk * wi[sa] * (dbase * sterm + base * dsterm);
        }
    }
    gr[tid] = gracc;
    double* gwi_out = gw + ((long)ic*natoms + iidx[p])*NSP;
    double* gwj_out = gw + ((long)ic*natoms + jidx[p])*NSP;
    #pragma unroll
    for (int s = 0; s < NSP; ++s) {
        atomicAdd(gwi_out + s, gwi[s]);
        atomicAdd(gwj_out + s, gwj[s]);
    }
}
"""

_MODULES = {}


def module(nsp=NSP):
    """Compile (and cache) the kernels for a given number of species.

    NVRTC compiles at run time, so the species count is injected as -DNSP=<n>.
    That keeps every per-species array a compile-time size while still letting
    the same source serve binaries, ternaries, quinaries and beyond.
    """
    nsp = int(nsp)
    if nsp < 2:
        raise ValueError("need at least two species")
    if nsp not in _MODULES:
        _MODULES[nsp] = cp.RawModule(
            code=_SRC,
            # no fast-math: fp64 accuracy is the point
            options=("-std=c++17", f"-DNSP={nsp}"),
            backend="nvrtc")
    return _MODULES[nsp]


def get(name, nsp=NSP):
    return module(nsp).get_function(name)
