"""Load, quality-screen and group the measured files: |chi(R)| or chi(k).

Layout of the source tree
-------------------------
    <root>/<Element>_<N><unit>/<anything>sample<M>.csv     a series
    <root>/<Element>/<anything>sample<M>.csv               one measurement

One folder per absorption edge and series point. `<N>` is an integer that
indexes the series - a pressure (`Cr_3GPa`), a temperature (`Fe_300K`), a
composition step - and `<unit>` is whatever letters follow it, one unit for
the whole tree. A folder named by the element alone is a single series point
(0, no unit). Any prefix may come before the element (`chik_Cr_3GPa`), and
the separator before `<N>` may be left out (`Cr3GPa`).

`sample<M>` (or, failing that, the last integer in the file name) is a
position on the sample; positions are fitted one by one and as a median. A
folder holding one file is one position. Two file kinds are understood, told
apart by the header (or, failing that, by the numbers):

    R, chi_mag      |chi(R)| on the Athena / Larch R grid (dR = pi/102.4 A).
                    Its phase, window, k weight and E0 are gone, and the fit
                    works on the magnitude.
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

Per-position fits pair the edges by position index. If the edges were
measured on different samples or in different cells, `sample<M>` does not
name the same spot at every edge, and only the position-median fit describes
one material state.

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

PEAK_LO, PEAK_HI = 1.5, 3.0
BASE_LO = 8.0
AMP_TOL = 3.0
SNR_MIN = 5.0
SPAN_TOL = 1.0     # A^-1 a scan may fall short of the folder median k range
KINDS = ("mag_R", "chi_k")
QC_WINDOW = Window(3.0, 13.0, (3,), 1.0)     # for screening chi(k) folders only
_ANY_UNIT = r"[A-Za-z%]*"


class Spectra(dict):
    """{(P, element): entry} as `screen` returns it, carrying the element
    order the fit uses (`elements`), the series points found (`series`) and
    the unit that follows the series index in the folder names (`unit`, ""
    when there is none)."""
    elements = ()
    series = ()
    unit = ""


def point_label(P, unit=""):
    """A series point for people: '3 GPa', '300 K', or 'point 0' without a unit."""
    return f"{P} {unit}" if unit else f"point {P}"


def point_dir(P, unit=""):
    """A series point's folder in a results tree: '3GPa', '300K', 'point0'."""
    return f"{P}{unit}" if unit else f"point{P}"


_QUANTITY = {"gpa": "pressure", "mpa": "pressure", "kbar": "pressure",
             "bar": "pressure", "k": "temperature", "c": "temperature"}


def axis_label(unit=""):
    """An axis label for the series: 'pressure (GPa)', 'temperature (K)'."""
    if not unit:
        return "series point"
    return f"{_QUANTITY.get(unit.lower(), 'series point')} ({unit})"


def is_pressure(unit):
    """True when the series unit is GPa, the unit `equation_of_state` takes."""
    return (unit or "").lower() == "gpa"


def _need_elements(elements):
    """The edges asked for, else the session default alloy's (see
    `chimera.system.set_default`)."""
    if elements:
        return tuple(elements)
    from .system import get_default
    if get_default() is not None:
        return get_default().elements
    raise ValueError("name the absorber edges: elements=(...) in model order")


def _els(spectra, elements=None):
    """The element order to use: the caller's, else the tree's."""
    if elements is not None:
        return tuple(elements)
    return _need_elements(getattr(spectra, "elements", ()))


def _folder_re(el, unit=None):
    u = _ANY_UNIT if unit is None else re.escape(unit)
    return re.compile(rf"(?:.*?[_-])?{el}(?:[_-]?(\d+)\s*({u}))?", re.I)


def _point(m):
    """(series index, unit) of a matched folder name; a bare element is (0, "")."""
    return (0, "") if m.group(1) is None else (int(m.group(1)), m.group(2) or "")


def _folder(data_root, el, P, unit=None):
    """The folder for one (element, series point).

    Accepts `Cr_3GPa` and the export variants seen in practice: an arbitrary
    prefix, with or without the separator (`chik_Cr3GPa`), any case, or the
    bare element for a tree with one series point.
    """
    pat = _folder_re(el, unit)
    for d in sorted(os.listdir(data_root)):
        m = pat.fullmatch(d)
        if m and os.path.isdir(os.path.join(data_root, d)) and _point(m)[0] == P:
            return os.path.join(data_root, d)
    raise FileNotFoundError(f"no folder for {el} {point_label(P, unit or '')} "
                            f"in {data_root}")


def series_of(data_root, elements, unit=None):
    """(points, unit): the series points present for ANY of `elements`, read
    off the folder names and sorted, and the unit they share. `unit` given
    restricts the match to it; folders that disagree on the unit are refused."""
    els = _need_elements(elements)
    found, units = set(), {}
    for d in os.listdir(data_root):
        if not os.path.isdir(os.path.join(data_root, d)):
            continue
        for el in els:
            m = _folder_re(el, unit).fullmatch(d)
            if m:
                P, u = _point(m)
                found.add(P)
                units.setdefault(u.lower(), u)
    if len(units) > 1:
        raise ValueError(f"{data_root}: folders mix series units {sorted(units.values())}; "
                         f"pass unit= to pick one")
    return sorted(found), (next(iter(units.values())) if units else (unit or ""))


def discover(data_root, elements, unit=None):
    """The series points present for ANY of `elements`, sorted."""
    return series_of(data_root, elements, unit)[0]


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


def scan(data_root, points=None, strict=False, elements=None, unit=None):
    """{(P, element): {sample index: path}} for the whole tree.

    `points` defaults to every series point found for any of `elements`. A
    series point with no folder is skipped rather than raised on: a dataset
    need not cover every nominal point. A series point with SOME of its edges
    present is a different matter - that is a missing folder, not a missing
    point - so it is reported either way, and refused under `strict`.
    """
    els = _need_elements(elements)
    found_points, unit = series_of(data_root, els, unit)
    if points is None:
        points = found_points
    out, partial = {}, []
    for P in points:
        found = {}
        for el in els:
            try:
                found[el] = _index(_folder(data_root, el, P, unit))
            except FileNotFoundError:
                pass
        if not found:
            continue                      # this series point is simply absent
        if len(found) < len(els):
            partial.append((P, sorted(set(els) - set(found))))
        out.update({(P, el): v for el, v in found.items()})
    if partial:
        msg = "; ".join(f"{point_label(P, unit)} has no " + ", ".join(m)
                        for P, m in partial)
        if strict:
            raise FileNotFoundError(msg)
        print(f"  !! incomplete series point(s): {msg} - cannot be fitted")
    return out


def quality(R, chi):
    pk = float(chi[(R >= PEAK_LO) & (R <= PEAK_HI)].max())
    base = float(chi[R >= BASE_LO].mean())
    return pk, base, pk / max(base, 1e-12)


def screen(data_root, verbose=False, kind=None, qc_window=QC_WINDOW,
           points=None, elements=None, unit=None):
    """Load everything, apply the QC cuts, return (spectra, report).

    spectra[(P, el)] = dict(R=..., chi={sample: |chi(R)|}, rejected={sample: why},
                            kind=..., and for chi(k) files k=..., chi_k={sample: chi(k)})

    `elements` are the absorber edges in model order (the system's). The
    result is a `Spectra` dict carrying `.elements`, `.series` (the points
    found) and `.unit` (the series unit read off the folder names, or the
    one given). `kind` forces the file kind; by default it is read off each
    file.
    """
    els = _need_elements(elements)
    unit = series_of(data_root, els, unit)[1]
    idx = scan(data_root, points, elements=els, unit=unit)
    spectra, report = Spectra(), []
    spectra.elements = els
    spectra.series = tuple(sorted({P for (P, _el) in idx}))
    spectra.unit = unit
    for (P, el), files in sorted(idx.items()):
        where = f"{el} {point_label(P, unit)}"
        raw, raw_k, qual, kinds = {}, {}, {}, set()
        xs, ys, keys = [], [], []
        for s, f in sorted(files.items()):
            x, y, kd = load_one(f, kind)
            kinds.add(kd)
            xs.append(x); ys.append(y); keys.append(s)
        assert len(kinds) <= 1, f"{where}: mixed file kinds {sorted(kinds)}"
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
        assert ok, f"{where}: no file reaches the folder's median k range"
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
            xunit = "A^-1" if kd == "chi_k" else "A"
            print(f"  {where}: grid {X[0]:.2f}-{X[-1]:.2f} {xunit}"
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
        report.append(dict(point=P, unit=unit, element=el, kind=kd, n_files=len(files),
                           n_kept=len(keep), median_peak=float(med),
                           k_range=[float(X[0]), float(X[-1])] if kd == "chi_k" else None,
                           rejected={str(k): v for k, v in drop.items()}))
        if verbose:
            print(f"  {where}: {len(keep):3d}/{len(files):3d} kept"
                  + ("  [chi(k)]" if kd == "chi_k" else "")
                  + (f"   dropped {sorted(drop)}" if drop else ""))
    return spectra, report


def has_chi_k(spectra, P, elements=None):
    """True when every edge at series point P holds chi(k)."""
    return all(spectra[(P, el)].get("kind") == "chi_k" for el in _els(spectra, elements))


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


def series_points(spectra):
    """The series points actually present in a screened tree."""
    return sorted({P for (P, _el) in spectra})


def duplicate_edges(spectra, tol=1e-8, elements=None):
    """Folders holding the same data - a provenance error in the export.

    Checked both ways round: two edges of one series point holding the same
    spectrum (a folder filled with another edge's files), and two series
    points of ONE edge (a mislabelled point). The second kind is the more
    dangerous: nothing downstream compares series points file by file, so it
    silently turns one measurement into two points of the series. Keys are
    (P, a, b) for an edge pair and (el, P, Q) for a pair of series points.
    """
    out = {}
    els = _els(spectra, elements)
    P_all = series_points(spectra)
    for P in P_all:
        for i, a in enumerate(els):
            for b in els[i + 1:]:
                if (P, a) not in spectra or (P, b) not in spectra:
                    continue
                dup, n, mx = _same_data(spectra[(P, a)], spectra[(P, b)], tol)
                if dup:
                    out[(P, a, b)] = dict(n=n, max_rel=mx, kind="edges")
    for el in els:
        for i, P in enumerate(P_all):
            for Q in P_all[i + 1:]:
                if (P, el) not in spectra or (Q, el) not in spectra:
                    continue
                dup, n, mx = _same_data(spectra[(P, el)], spectra[(Q, el)], tol)
                if dup:
                    out[(el, P, Q)] = dict(n=n, max_rel=mx, kind="points")
    return out


def describe_duplicates(dups, unit=""):
    """One human-readable line per duplicate, for the log and the report."""
    out = []
    for k, v in sorted(dups.items(), key=lambda kv: str(kv[0])):
        if v.get("kind") == "points":
            el, P, Q = k
            out.append(f"{el}: the {point_label(P, unit)} and {point_label(Q, unit)} "
                       f"files are the SAME data (max rel diff {v['max_rel']:.1e} over "
                       f"{v['n']} positions) - one of the two points is mislabelled")
        else:
            P, a, b = k
            out.append(f"{point_label(P, unit)}: the {a} and {b} files are the SAME data "
                       f"(max rel diff {v['max_rel']:.1e} over {v['n']} positions) "
                       f"- one fewer independent edge there")
    return out


def locations(spectra, P, require_all=True, elements=None):
    """Sample indices usable at series point P."""
    sets = [set(spectra[(P, el)]["chi"]) for el in _els(spectra, elements)]
    return sorted(set.intersection(*sets) if require_all else set.union(*sets))


# ------------------------------------------------------------------ |chi(R)|
def stack(spectra, P, sample, elements=None):
    """(nsp, nR) array of |chi(R)| for one location, in element order."""
    return np.array([spectra[(P, el)]["chi"][sample] for el in _els(spectra, elements)])


def average(spectra, P, samples=None, robust=True, elements=None):
    """Location-averaged |chi(R)| per edge, (nsp, nR).

    Uses the median over positions by default: the maps contain occasional
    hot positions that survive the QC cut but still skew a mean.
    """
    out = []
    for el in _els(spectra, elements):
        d = spectra[(P, el)]["chi"]
        keys = sorted(d) if samples is None else [s for s in samples if s in d]
        A = np.array([d[s] for s in keys])
        out.append(np.median(A, axis=0) if robust else A.mean(axis=0))
    return np.array(out)


def spread(spectra, P, samples=None, elements=None):
    """Robust per-point scatter of |chi(R)| over locations (1.4826 * MAD), (nsp, nR)."""
    out = []
    for el in _els(spectra, elements):
        d = spectra[(P, el)]["chi"]
        keys = sorted(d) if samples is None else [s for s in samples if s in d]
        A = np.array([d[s] for s in keys])
        med = np.median(A, axis=0)
        out.append(1.4826 * np.median(np.abs(A - med), axis=0))
    return np.array(out)


# ------------------------------------------------------------------ chi(k)
def stack_k(spectra, P, sample, elements=None):
    """ChiK for one location (needs chi(k) files at every edge)."""
    els = _els(spectra, elements)
    assert has_chi_k(spectra, P, els), \
        f"{point_label(P, getattr(spectra, 'unit', ''))}: not every edge holds chi(k)"
    return ChiK([spectra[(P, el)]["k"] for el in els],
                [spectra[(P, el)]["chi_k"][sample] for el in els],
                elements=els)


def average_k(spectra, P, samples=None, robust=True, elements=None):
    """Location-median chi(k) per edge as a ChiK (point-wise in k; the
    transform is linear, so this is the median spectrum's transform up to
    the median-vs-mean difference)."""
    els = _els(spectra, elements)
    assert has_chi_k(spectra, P, els), \
        f"{point_label(P, getattr(spectra, 'unit', ''))}: not every edge holds chi(k)"
    k, chi = [], []
    for el in els:
        d = spectra[(P, el)]["chi_k"]
        keys = sorted(d) if samples is None else [s for s in samples if s in d]
        A = np.array([d[s] for s in keys])
        chi.append(np.median(A, axis=0) if robust else A.mean(axis=0))
        k.append(spectra[(P, el)]["k"])
    return ChiK(k, chi, elements=els)


def fit_input(spectra, P, sample=None, robust=True, elements=None):
    """What the fit should be given at series point P: a ChiK when chi(k) was
    measured, the |chi(R)| array otherwise. One location with `sample`, the
    location median without."""
    els = _els(spectra, elements)
    if has_chi_k(spectra, P, els):
        return stack_k(spectra, P, sample, els) if sample is not None \
            else average_k(spectra, P, robust=robust, elements=els)
    return stack(spectra, P, sample, els) if sample is not None \
        else average(spectra, P, robust=robust, elements=els)


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


def write_qc(report, dups, path, unit=""):
    rows = []
    for k, v in dups.items():
        if v.get("kind") == "points":
            rows.append(dict(element=k[0], point=k[1], point_b=k[2], **v))
        else:
            rows.append(dict(point=k[0], a=k[1], b=k[2], **v))
    with open(path, "w") as f:
        json.dump(dict(unit=unit, quality=report, duplicate_edges=rows,
                       duplicates=describe_duplicates(dups, unit)), f, indent=2)
