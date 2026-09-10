"""Load, quality-screen and group the measured |chi(R)| files.

Layout of the source tree
-------------------------
    Data/<Element>_<P>GPa/<run>.h5_..._sample<N>.csv,  columns R, chi_mag

`sample<N>` is a position on the sample map. The R grid is the same in every
file (dR = 0.0306796 A, 326 points), and the same one the forward model uses.

Two provenance facts that the caller must be told about, because they change
what the fits mean:

1. Cr was measured in a DIFFERENT diamond anvil cell (DAC4) from Co and Ni
   (DAC3). A given `sample<N>` therefore does NOT refer to the same physical
   spot across all three edges - only within Co/Ni. Per-location fits pair
   them by index for want of anything better; the pressure-averaged fit is
   the one that is unambiguously about a single material state.
2. At 0 GPa the Co and Ni files are the SAME spectrum to machine precision
   (max relative difference 4e-16 over all 72 positions). One of the two
   folders has been filled with the other edge's data. At 0 GPa there are
   therefore only two independent edges, which the ablation in REPORT.md
   shows is not enough for the Co/Ni Warren-Cowley block.

Quality screening
-----------------
Some positions are off-sample or failed normalisation. A file is rejected if
its first-shell peak (1.5-3.0 A) is more than 3x or less than 1/3 of the
folder median, or if its peak-to-baseline ratio (baseline = mean over
R > 8 A) is below 5. Both cuts are recorded in the QC report.
"""
from __future__ import annotations
import os, re, glob, json
import numpy as np

ELEMENTS = ("Cr", "Co", "Ni")
PRESSURES = (0, 3, 7, 10)
PEAK_LO, PEAK_HI = 1.5, 3.0
BASE_LO = 8.0
AMP_TOL = 3.0
SNR_MIN = 5.0


def _folder(data_root, el, P):
    for d in os.listdir(data_root):
        m = re.fullmatch(rf"{el}_(\d+)GPa", d, re.I)
        if m and int(m.group(1)) == P:
            return os.path.join(data_root, d)
    raise FileNotFoundError(f"no folder for {el} {P} GPa in {data_root}")


def _index(folder):
    out = {}
    for f in glob.glob(os.path.join(folder, "*.csv")):
        m = re.search(r"sample(\d+)\.csv$", os.path.basename(f))
        if m:
            out[int(m.group(1))] = f
    return out


def load_one(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1]


def scan(data_root):
    """{(P, element): {sample_index: path}} for the whole tree."""
    return {(P, el): _index(_folder(data_root, el, P))
            for P in PRESSURES for el in ELEMENTS}


def quality(R, chi):
    pk = float(chi[(R >= PEAK_LO) & (R <= PEAK_HI)].max())
    base = float(chi[R >= BASE_LO].mean())
    return pk, base, pk / max(base, 1e-12)


def screen(data_root, verbose=False):
    """Load everything, apply the QC cuts, return (spectra, report).

    spectra[(P, el)] = dict(R=..., chi={sample: array}, rejected={sample: why})
    """
    idx = scan(data_root)
    spectra, report = {}, []
    for (P, el), files in sorted(idx.items()):
        R = None
        raw, qual = {}, {}
        for s, f in sorted(files.items()):
            r, c = load_one(f)
            R = r if R is None else R
            assert np.allclose(r, R), f"{f}: R grid differs"
            raw[s] = c
            qual[s] = quality(r, c)
        med = np.median([q[0] for q in qual.values()])
        keep, drop = {}, {}
        for s, c in raw.items():
            pk, base, snr = qual[s]
            if pk > AMP_TOL * med:
                drop[s] = f"peak {pk:.3g} > {AMP_TOL}x folder median {med:.3g}"
            elif pk < med / AMP_TOL:
                drop[s] = f"peak {pk:.3g} < 1/{AMP_TOL:g} of folder median {med:.3g}"
            elif snr < SNR_MIN:
                drop[s] = f"peak/baseline {snr:.1f} < {SNR_MIN}"
            else:
                keep[s] = c
        spectra[(P, el)] = dict(R=R, chi=keep, rejected=drop, median_peak=float(med))
        report.append(dict(pressure=P, element=el, n_files=len(raw),
                           n_kept=len(keep), median_peak=float(med),
                           rejected={str(k): v for k, v in drop.items()}))
        if verbose:
            print(f"  {el} {P:2d} GPa: {len(keep):3d}/{len(raw):3d} kept"
                  + (f"   dropped {sorted(drop)}" if drop else ""))
    return spectra, report


def duplicate_edges(spectra, tol=1e-8):
    """Pressures where two edges hold the same data (a provenance error)."""
    out = {}
    for P in PRESSURES:
        for a, b in (("Co", "Ni"), ("Cr", "Co"), ("Cr", "Ni")):
            A, B = spectra[(P, a)], spectra[(P, b)]
            common = sorted(set(A["chi"]) & set(B["chi"]))
            if not common:
                continue
            rel = []
            for s in common:
                x, y = A["chi"][s], B["chi"][s]
                m = x > 0.05 * x.max()
                rel.append(np.max(np.abs(x[m] - y[m]) / x[m]))
            if np.median(rel) < tol:
                out[(P, a, b)] = dict(n=len(common), max_rel=float(np.max(rel)))
    return out


def locations(spectra, P, require_all=True):
    """Sample indices usable at pressure P."""
    sets = [set(spectra[(P, el)]["chi"]) for el in ELEMENTS]
    return sorted(set.intersection(*sets) if require_all else set.union(*sets))


def stack(spectra, P, sample):
    """(3, nR) array of |chi(R)| for one location, ordered as ELEMENTS."""
    return np.array([spectra[(P, el)]["chi"][sample] for el in ELEMENTS])


def average(spectra, P, samples=None, robust=True):
    """Location-averaged |chi(R)| per edge, (3, nR).

    Uses the median over positions by default: the maps contain occasional
    hot positions that survive the QC cut but still skew a mean.
    """
    out = []
    for el in ELEMENTS:
        d = spectra[(P, el)]["chi"]
        keys = sorted(d) if samples is None else [s for s in samples if s in d]
        A = np.array([d[s] for s in keys])
        out.append(np.median(A, axis=0) if robust else A.mean(axis=0))
    return np.array(out)


def spread(spectra, P, samples=None):
    """Robust per-point scatter over locations (1.4826 * MAD), (3, nR)."""
    out = []
    for el in ELEMENTS:
        d = spectra[(P, el)]["chi"]
        keys = sorted(d) if samples is None else [s for s in samples if s in d]
        A = np.array([d[s] for s in keys])
        med = np.median(A, axis=0)
        out.append(1.4826 * np.median(np.abs(A - med), axis=0))
    return np.array(out)


def noise_estimate(R, chi, r_lo=7.5):
    """Baseline scatter from the structureless high-R region."""
    tail = chi[..., R >= r_lo]
    return float(np.sqrt(np.mean((tail - tail.mean(axis=-1, keepdims=True)) ** 2)))


def write_qc(report, dups, path):
    with open(path, "w") as f:
        json.dump(dict(quality=report,
                       duplicate_edges=[dict(pressure=k[0], a=k[1], b=k[2], **v)
                                        for k, v in dups.items()]), f, indent=2)
