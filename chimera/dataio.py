"""Load, quality-screen and group the measured files: |chi(R)| or chi(k).

Layout of the source tree
-------------------------
    Data/<Element>_<P>GPa/<run>.h5_..._sample<N>.csv

`sample<N>` is a position on the sample map. Two file kinds are understood,
told apart by the header (or, failing that, by the numbers):

    R, chi_mag      |chi(R)| on the experimenters' grid (dR = 0.0306796 A,
                    326 points to 10 A, the Athena / Larch default). What the
                    2026 CrCoNi export holds. Its phase, window, k weight and
                    E0 are gone, and the fit works on the magnitude.
    k, chi          chi(k) as Athena / Larch export it: real, dimensionless,
                    on a 0.05 A^-1 grid. Also accepted: Athena's whitespace
                    `.chi` files with a `#` header (columns k chi chik chik2
                    chik3; the `chi` column is used). The fit then compares
                    the COMPLEX chi(R) of data and model under one window,
                    and every k weight is available (`chik.py`).

Every file of one folder must be of one kind and on one grid. For chi(k)
folders the quality screen and the duplicate check run on |chi(R)| computed
with a fixed window (QC_WINDOW), so the same cuts apply to both kinds, and
the chi(k) itself is kept alongside (`spectra[...]["chi_k"]`).

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

from .chik import ChiK, Window, transform, make_window
from exafs_gpu.fourier import r_grid

ELEMENTS = ("Cr", "Co", "Ni")
PRESSURES = (0, 3, 7, 10)
PEAK_LO, PEAK_HI = 1.5, 3.0
BASE_LO = 8.0
AMP_TOL = 3.0
SNR_MIN = 5.0
SPAN_TOL = 1.0     # A^-1 a scan may fall short of the folder median k range
KINDS = ("mag_R", "chi_k")
QC_WINDOW = Window(3.0, 13.0, (3,), 1.0)     # for screening chi(k) folders only


def _folder(data_root, el, P):
    """The folder for one (element, series point).

    Accepts `Cr_0GPa` and the export variants seen in practice: an arbitrary
    prefix, with or without the separator (`chik_Cr0GPa`, `rdf_Cr0GPa`), any
    case.
    """
    for d in os.listdir(data_root):
        m = re.fullmatch(rf"(?:.*?[_-])?{el}[_-]?(\d+)\s*GPa", d, re.I)
        if m and int(m.group(1)) == P:
            return os.path.join(data_root, d)
    raise FileNotFoundError(f"no folder for {el} {P} GPa in {data_root}")


def _index(folder):
    """{sample index: path}. The index is the `sample<N>` group where there is
    one, else the LAST integer in the name (`0007_12.1.csv` -> 12), which is
    how the Athena batch exports label map positions."""
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))
                    + glob.glob(os.path.join(folder, "*.chi"))):
        base = os.path.basename(f)
        m = re.search(r"sample(\d+)\.(csv|chi)$", base)
        if m is None:
            nums = re.findall(r"(\d+)", os.path.splitext(base)[0])
            if not nums:
                continue
            # trailing ".1" style suffixes are a scan counter, not a position
            stem = re.sub(r"\.\d+$", "", os.path.splitext(base)[0])
            nums = re.findall(r"(\d+)", stem) or nums
            out[int(nums[-1])] = f
        else:
            out[int(m.group(1))] = f
    return out


def _common_grid(grids):
    """The largest shared prefix of a set of k (or R) grids.

    Athena writes one file per scan and each scan has its own energy range, so
    a folder's chi(k) files share dk and the low end but stop at different
    kmax. Everything is truncated to the shortest, and how much was lost is
    reported; a file whose grid does not lie on the common one at all is
    flagged for rejection rather than silently resampled.
    """
    n = min(g.size for g in grids)
    ref = min(grids, key=lambda g: g.size)
    bad = [i for i, g in enumerate(grids) if not np.allclose(g[:n], ref, atol=1e-6)]
    return ref, bad


# ------------------------------------------------------------------ reading
def _is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def read_columns(path):
    """(names, array) from a CSV or a whitespace table.

    Column names come from a header line, or from the last `#` comment that
    names them (Athena's convention); `names` is None if there is neither.
    """
    names, rows = None, []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            if s[0] in "#;%":
                toks = s.lstrip("#;% ").replace(",", " ").split()
                if toks and not _is_number(toks[0]):
                    names = [t.lower() for t in toks]
                continue
            toks = s.replace(",", " ").split()
            if not _is_number(toks[0]):
                names = [t.lower() for t in toks]
                continue
            rows.append([float(t) for t in toks])
    d = np.array(rows, float)
    assert d.ndim == 2 and d.shape[1] >= 2, f"{path}: need at least two columns"
    return names, d


def classify(names, d):
    """'mag_R' or 'chi_k' from the column names, else from the numbers."""
    if names:
        n0 = names[0]
        if n0 == "r" or any(n in ("chi_mag", "chir_mag", "|chi(r)|") for n in names):
            return "mag_R"
        if n0 == "k" or "chi" in names:
            return "chi_k"
    x, y = d[:, 0], d[:, 1]
    if (y < 0).any():                       # a magnitude is never negative
        return "chi_k"
    dx = float(np.median(np.diff(x)))
    return "mag_R" if abs(dx - np.pi / (2048 * 0.05)) < 1e-4 else "chi_k"


def load_one(path, kind=None):
    """(x, y, kind): (R, |chi(R)|) for 'mag_R', (k, chi(k)) for 'chi_k'."""
    names, d = read_columns(path)
    kind = kind or classify(names, d)
    assert kind in KINDS, f"unknown file kind {kind!r}"
    col = 1
    if kind == "chi_k" and names and "chi" in names:
        col = names.index("chi")
    elif kind == "mag_R" and names:
        for n in ("chi_mag", "chir_mag"):
            if n in names:
                col = names.index(n)
    return d[:, 0], d[:, col], kind


def scan(data_root):
    """{(P, element): {sample_index: path}} for the whole tree."""
    return {(P, el): _index(_folder(data_root, el, P))
            for P in PRESSURES for el in ELEMENTS}


def quality(R, chi):
    pk = float(chi[(R >= PEAK_LO) & (R <= PEAK_HI)].max())
    base = float(chi[R >= BASE_LO].mean())
    return pk, base, pk / max(base, 1e-12)


def screen(data_root, verbose=False, kind=None, qc_window=QC_WINDOW):
    """Load everything, apply the QC cuts, return (spectra, report).

    spectra[(P, el)] = dict(R=..., chi={sample: |chi(R)|}, rejected={sample: why},
                            kind=..., and for chi(k) files k=..., chi_k={sample: chi(k)})

    `kind` forces the file kind; by default it is read off each file.
    """
    idx = scan(data_root)
    spectra, report = {}, []
    for (P, el), files in sorted(idx.items()):
        raw, raw_k, qual, kinds = {}, {}, {}, set()
        xs, ys, keys = [], [], []
        for s, f in sorted(files.items()):
            x, y, kd = load_one(f, kind)
            kinds.add(kd)
            xs.append(x); ys.append(y); keys.append(s)
        assert len(kinds) <= 1, f"{el} {P} GPa: mixed file kinds {sorted(kinds)}"
        kd = kinds.pop() if kinds else "mag_R"
        # One grid for the folder. A scan that stops far short of the rest
        # would otherwise truncate everything to its own range, so those are
        # dropped first; the survivors are then cut to their shared prefix.
        ends = np.array([x[-1] for x in xs])
        cut = np.median(ends) - SPAN_TOL
        grid_drop = {s: f"k range ends at {ends[i]:.2f}, folder median "
                        f"{np.median(ends):.2f} A^-1"
                     for i, s in enumerate(keys) if ends[i] < cut}
        ok = [i for i, s in enumerate(keys) if s not in grid_drop]
        assert ok, f"{el} {P} GPa: no file reaches the folder's median k range"
        X, off_grid = _common_grid([xs[i] for i in ok])
        n = X.size
        for j, i in enumerate(ok):
            s = keys[i]
            if j in off_grid:
                grid_drop[s] = (f"grid does not match the folder's "
                                f"({xs[i][0]:.3g}..{xs[i][-1]:.3g}, {xs[i].size} points)")
                continue
            (raw_k if kd == "chi_k" else raw)[s] = ys[i][:n]
        if verbose and (grid_drop or X[-1] < ends.max() - 1e-9):
            unit = "A^-1" if kd == "chi_k" else "A"
            print(f"  {el} {P:2d} GPa: grid {X[0]:.2f}-{X[-1]:.2f} {unit}"
                  f" (longest file {ends.max():.2f})"
                  + (f", {len(grid_drop)} dropped on range" if grid_drop else ""))
        if kd == "chi_k":
            # |chi(R)| under a fixed window, for the screen and the figures.
            # The window has to fit inside the folder's k range, or every
            # spectrum is scored on a window that is mostly zeros.
            R = r_grid()
            qw = make_window(qc_window)
            if qw.kmax > X[-1]:
                qw = qw._replace(kmax=float(X[-1]))
            kept = sorted(raw_k)
            mags = np.abs(transform(X, np.array([raw_k[s] for s in kept]), qw))[:, 0]
            raw = {s: m for s, m in zip(kept, mags)}
        else:
            R = X
        for s, c in raw.items():
            qual[s] = quality(R, c)
        med = np.median([q[0] for q in qual.values()]) if qual else 0.0
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
        drop.update(grid_drop)               # range/grid cuts join the QC report
        entry = dict(R=R, chi=keep, rejected=drop, median_peak=float(med), kind=kd)
        if kd == "chi_k":
            entry.update(k=X, chi_k={s: raw_k[s] for s in keep})
        spectra[(P, el)] = entry
        report.append(dict(pressure=P, element=el, kind=kd, n_files=len(files),
                           n_kept=len(keep), median_peak=float(med),
                           k_range=[float(X[0]), float(X[-1])] if kd == "chi_k" else None,
                           rejected={str(k): v for k, v in drop.items()}))
        if verbose:
            print(f"  {el} {P:2d} GPa: {len(keep):3d}/{len(files):3d} kept"
                  + ("  [chi(k)]" if kd == "chi_k" else "")
                  + (f"   dropped {sorted(drop)}" if drop else ""))
    return spectra, report


def has_chi_k(spectra, P):
    """True when every edge at pressure P holds chi(k)."""
    return all(spectra[(P, el)].get("kind") == "chi_k" for el in ELEMENTS)


def _same_data(A, B, tol):
    """(is_duplicate, n_common, max_rel) for two screened entries."""
    key = "chi_k" if ("chi_k" in A and "chi_k" in B) else "chi"
    common = sorted(set(A[key]) & set(B[key]))
    if not common:
        return False, 0, None
    rel = []
    for s in common:
        x, y = A[key][s], B[key][s]
        if x.shape != y.shape:
            rel.append(np.inf)
            continue
        ax = np.abs(x)
        m = ax > 0.05 * ax.max()
        rel.append(np.max(np.abs(x[m] - y[m]) / ax[m]) if m.any() else np.inf)
    return bool(np.median(rel) < tol), len(common), float(np.max(rel))


def duplicate_edges(spectra, tol=1e-8):
    """Folders holding the same data - a provenance error in the export.

    Checked BOTH ways round, because both have been seen in this dataset:
    two edges of one series point (the 0 GPa Co/Ni copy in the 2026-09-04
    export, the 0 GPa Cr/Co copy in the chi(k) one), and two series points of
    ONE edge (Cr 3 == Cr 7 GPa, Co 7 == Co 10 GPa). The second kind is the
    more dangerous: nothing downstream compares pressures file by file, so it
    silently turns one measurement into two data points on an equation of
    state. Keys are (P, a, b) for an edge pair and (el, P, Q) for a pressure
    pair.
    """
    out = {}
    for P in PRESSURES:
        for a, b in (("Co", "Ni"), ("Cr", "Co"), ("Cr", "Ni")):
            if (P, a) not in spectra or (P, b) not in spectra:
                continue
            dup, n, mx = _same_data(spectra[(P, a)], spectra[(P, b)], tol)
            if dup:
                out[(P, a, b)] = dict(n=n, max_rel=mx, kind="edges")
    for el in ELEMENTS:
        for i, P in enumerate(PRESSURES):
            for Q in PRESSURES[i + 1:]:
                if (P, el) not in spectra or (Q, el) not in spectra:
                    continue
                dup, n, mx = _same_data(spectra[(P, el)], spectra[(Q, el)], tol)
                if dup:
                    out[(el, P, Q)] = dict(n=n, max_rel=mx, kind="pressures")
    return out


def describe_duplicates(dups):
    """One human-readable line per duplicate, for the log and the report."""
    out = []
    for k, v in sorted(dups.items(), key=lambda kv: str(kv[0])):
        if v.get("kind") == "pressures":
            el, P, Q = k
            out.append(f"{el}: the {P} GPa and {Q} GPa files are the SAME data "
                       f"(max rel diff {v['max_rel']:.1e} over {v['n']} positions) "
                       f"- one of the two pressures is mislabelled")
        else:
            P, a, b = k
            out.append(f"{P} GPa: the {a} and {b} files are the SAME data "
                       f"(max rel diff {v['max_rel']:.1e} over {v['n']} positions) "
                       f"- only two independent edges there")
    return out


def locations(spectra, P, require_all=True):
    """Sample indices usable at pressure P."""
    sets = [set(spectra[(P, el)]["chi"]) for el in ELEMENTS]
    return sorted(set.intersection(*sets) if require_all else set.union(*sets))


# ------------------------------------------------------------------ |chi(R)|
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
    """Robust per-point scatter of |chi(R)| over locations (1.4826 * MAD), (3, nR)."""
    out = []
    for el in ELEMENTS:
        d = spectra[(P, el)]["chi"]
        keys = sorted(d) if samples is None else [s for s in samples if s in d]
        A = np.array([d[s] for s in keys])
        med = np.median(A, axis=0)
        out.append(1.4826 * np.median(np.abs(A - med), axis=0))
    return np.array(out)


# ------------------------------------------------------------------ chi(k)
def stack_k(spectra, P, sample):
    """ChiK for one location (needs chi(k) files at every edge)."""
    assert has_chi_k(spectra, P), f"{P} GPa: not every edge holds chi(k)"
    return ChiK([spectra[(P, el)]["k"] for el in ELEMENTS],
                [spectra[(P, el)]["chi_k"][sample] for el in ELEMENTS],
                elements=ELEMENTS)


def average_k(spectra, P, samples=None, robust=True):
    """Location-median chi(k) per edge as a ChiK (point-wise in k; the
    transform is linear, so this is the median spectrum's transform up to
    the median-vs-mean difference)."""
    assert has_chi_k(spectra, P), f"{P} GPa: not every edge holds chi(k)"
    k, chi = [], []
    for el in ELEMENTS:
        d = spectra[(P, el)]["chi_k"]
        keys = sorted(d) if samples is None else [s for s in samples if s in d]
        A = np.array([d[s] for s in keys])
        chi.append(np.median(A, axis=0) if robust else A.mean(axis=0))
        k.append(spectra[(P, el)]["k"])
    return ChiK(k, chi, elements=ELEMENTS)


def fit_input(spectra, P, sample=None, robust=True):
    """What the fit should be given at pressure P: a ChiK when chi(k) was
    measured, the |chi(R)| array otherwise. One location with `sample`, the
    location median without."""
    if has_chi_k(spectra, P):
        return stack_k(spectra, P, sample) if sample is not None \
            else average_k(spectra, P, robust=robust)
    return stack(spectra, P, sample) if sample is not None \
        else average(spectra, P, robust=robust)


# ------------------------------------------------------------------ noise
def noise_estimate(R, chi, r_lo=None, r_hi=None):
    """Baseline scatter from the structureless high-R region.

    Real or complex input; for complex the per-COMPONENT sd is returned. The
    band defaults to R > 7.5 A on a grid that stops at 10 A (the |chi(R)|
    files) and to 15-25 A when the grid reaches that far (a chi(k) transform).
    """
    R = np.asarray(R)
    long_grid = R.max() >= 25.0
    if r_lo is None:
        r_lo = 15.0 if long_grid else 7.5
    if r_hi is None:
        r_hi = 25.0 if long_grid else float(R.max()) + 1.0
    m = (R >= r_lo) & (R <= r_hi)
    tail = chi[..., m]
    tail = tail - tail.mean(axis=-1, keepdims=True)
    if np.iscomplexobj(tail):
        return float(np.sqrt(0.5 * np.mean(np.abs(tail) ** 2)))
    return float(np.sqrt(np.mean(tail ** 2)))


def write_qc(report, dups, path):
    rows = []
    for k, v in dups.items():
        if v.get("kind") == "pressures":
            rows.append(dict(element=k[0], pressure=k[1], pressure_b=k[2], **v))
        else:
            rows.append(dict(pressure=k[0], a=k[1], b=k[2], **v))
    with open(path, "w") as f:
        json.dump(dict(quality=report, duplicate_edges=rows,
                       duplicates=describe_duplicates(dups)), f, indent=2)
