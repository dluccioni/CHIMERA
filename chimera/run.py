"""Fit every series point and sample position of a data tree; write a results tree.

    python -m chimera.run --data <root> --out <results> --elements Fe Ni --a0-ref 3.59
                          [--lattice fcc] [--composition 0.5 0.5]
                          [--tables native | --tables <FEFF set> --ms-tables <set>]
                          [--points 0 5 10] [--unit GPa] [--quick]
                          [--kweights 1,2,3] [--kmin 3 --kmax 13] [--rmin 1.4 --rmax 5.0]

The data layout is described in `chimera/dataio.py`: one folder per edge and
series point, `<Element>_<N><unit>` (or `<Element>` alone for one
measurement), holding chi(k) (`k, chi`) or |chi(R)| (`R, chi_mag`) files, one
per sample position. The series points and their unit are read off the
folder names unless `--points` / `--unit` are given.

The alloy comes from `--elements`, `--composition`, `--lattice`, `--a0-ref`
(and `--c-over-a` for hcp), or from `--tables <set>` when that set's
meta.json records them (feff8l_tables.py and native_tables.py write it).
The scattering paths default to `native`: computed in-process by CHIMERA's
own engine (exafs_gpu/abinitio) for the alloy at `--a0-ref`, with no FEFF and
no files. `native:seed=1,rpath=5.8` changes the recipe; the engine version
and recipe are written into calibration.json and summary.json under
"scattering".

The input kind is read off the files: `R, chi_mag` folders are fitted on
|chi(R)|; `k, chi` folders (measured chi(k), see chimera/chik.py) are fitted
on the complex chi(R) under our own window, by default with k^1, k^2 and k^3
together, and every position folder then also gets a k-space comparison
figure. `--kweights` entries are single weights or comma-joined sets fitted
jointly; the calibration picks the best-scoring entry.

The a0 / E0 treatment: a0 is free at every series point, the shared
per-edge E0 - including its common mode - is fitted in the joint stage
together with every point's a0 and sigma^2, and a mean-zero per-edge E0
shift is fitted per point. `--no-free-a0` anchors a0 at the reference point
to `--a0-anchor` instead, which |chi(R)| data require. Either way the
absolute a0 inherits the scattering calculation's distance error; quote it
against an independent lattice constant (diffraction) where one exists.

Output (one folder per series point, one sub-folder per position plus
`averaged`):

    <out>/<N><unit>/averaged/        fit of the position-median spectrum, plus
                                     the calibration-systematic band
    <out>/<N><unit>/sample<M>/       fit of that sample position
    <out>/<N><unit>/summary_*.csv    the per-position tables
    <out>/calibration.json           the shared processing calibration
    <out>/data_quality.json          QC and the duplicate-folder check
    <out>/summary.json, summary_by_point.csv, series_summary.png
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np

from . import dataio as io, model as M, calibrate as C, fitrun as F
from . import figures as FIG
from . import system as SYS
from .chik import as_kweights


def _j(o):
    """json default. A per-edge k window makes `kmax` an array, and the plain
    `float` default raises on anything longer than one element."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return float(o)


def write_csv(path, header, rows, fmt="%.6g"):
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt % v if isinstance(v, (int, float, np.floating))
                             else str(v) for v in r) + "\n")


def rdf_grid(S, rmax):
    """The r grid the partial RDFs are written on: from inside the first
    shell to just past the fit range."""
    return np.linspace(0.6 * float(S.shells[0]), float(rmax) + 0.5, 400)


def save_location(folder, S, res, R_data, data, mask_d, title, r_grid, spread=None,
                  wc_band=None, extra=None):
    """Write one fit: the alpha table, partial RDFs, shell coordination, the
    configuration, fit.json and the figures."""
    os.makedirs(folder, exist_ok=True)
    sp = np.array(res["species"], np.int32)
    els, nsp = S.elements, S.nsp
    labels = F.wc_labels(els)
    nwc = len(labels)

    crlb = res.get("wc_crlb") or [""] * nwc
    crlb_n = res.get("wc_crlb_noise") or [""] * nwc
    post = res.get("wc_posterior_sd") or [""] * nwc
    det = res.get("wc_determined")
    write_csv(os.path.join(folder, "warren_cowley.csv"),
              ["parameter", "value", "sd_seeds", "posterior_sd", "crlb",
               "crlb_noise_only", "sys_lo", "sys_hi", "determined"],
              [[labels[i], res["wc"][i], res["wc_sd_seeds"][i], post[i],
                crlb[i], crlb_n[i],
                (wc_band["wc_min"][i] if wc_band else ""),
                (wc_band["wc_max"][i] if wc_band else ""),
                ("yes" if det[i] else "no") if det is not None else "no"]
               for i in range(nwc)])

    g, g_tot = F.broadened_partial_rdf(sp, S, r_grid)
    hdr = ["r_Angstrom"] + [f"g_{els[a]}{els[b]}"
                            for a in range(nsp) for b in range(nsp)] + ["g_total"]
    rows = [[r_grid[t]] + [g[a, b, t] for a in range(nsp) for b in range(nsp)]
            + [g_tot[t]] for t in range(r_grid.size)]
    write_csv(os.path.join(folder, "partial_rdf.csv"), hdr, rows)

    shells = F.shell_table(sp, S, nshell=4)
    write_csv(os.path.join(folder, "shell_coordination.csv"),
              ["shell", "radius_Angstrom", "pair", "N_AB", "alpha_AB"],
              [[s["shell"], s["radius"], f"{els[a]}-{els[b]}",
                s["N"][a][b], s["alpha"][a][b]]
               for s in shells for a in range(nsp) for b in range(nsp)])

    np.savetxt(os.path.join(folder, "configuration.csv"), sp, fmt="%d",
               header="species index per site (%s); sites are the ideal %s lattice "
                      "of a0 in fit.json" % (", ".join(f"{i}={e}" for i, e in enumerate(els)),
                                             S.system.lattice), comments="# ")
    meta = {k: v for k, v in res.items()
            if k not in ("species",) + F.MODEL_ARRAY_KEYS}
    meta["shell_table"] = shells
    meta["system"] = S.system.to_dict()
    if wc_band is not None:
        meta["wc_systematic"] = {k: v for k, v in wc_band.items() if k != "variants"}
    if extra:
        meta.update(extra)
    json.dump(meta, open(os.path.join(folder, "fit.json"), "w"), indent=1, default=_j)

    model, ideal = F.model_arrays(res)
    tgt = M.target(data, S)                 # |chi(R)| as given, or complex chi(R)
    FIG.fit_comparison(os.path.join(folder, "fit_comparison.png"), R_data, tgt,
                       model, ideal, mask_d, title, res["r_factor"],
                       res["r_factor_ideal"], shells=res["shells"], spread=spread,
                       labels=[f"{c} edge" if " " not in c else c
                               for c in res.get("channels", [])] or None,
                       elements=els)
    if M.is_chik(data):
        FIG.k_comparison(os.path.join(folder, "fit_comparison_k.png"), data, S, sp,
                         np.array(res["scales"]), title,
                         ideal_species=S.random_config(0))
    band = None if wc_band is None else (wc_band["wc_min"], wc_band["wc_max"])
    FIG.rdf_figure(os.path.join(folder, "partial_rdf.png"), r_grid, g, g_tot,
                   res["shells"], title, res["wc"], wc_band=band, elements=els)
    FIG.sro_by_shell(os.path.join(folder, "sro_by_shell.png"),
                     [t["radius"] for t in shells],
                     [np.array(t["alpha"]) for t in shells], els,
                     title=f"Short-Range Order Parameters by Shell\n{title}")
    return g, g_tot


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m chimera.run",
        description="Fit multi-edge EXAFS of a substitutional alloy: calibration, "
                    "reverse Monte Carlo of the species arrangement, Warren-Cowley "
                    "parameters with their uncertainty budget.")
    B = argparse.BooleanOptionalAction
    # ---- input and output
    ap.add_argument("--data", required=True, help="the data root (see chimera/dataio.py)")
    ap.add_argument("--out", required=True,
                    help="output directory. A partial run merges its calibration "
                         "into an existing tree but rewrites summary.json with only "
                         "the points it ran, so point smoke tests somewhere else.")
    ap.add_argument("--points", type=int, nargs="+", default=None,
                    help="series points to fit (default: every one in the data)")
    ap.add_argument("--unit", default=None,
                    help="the series unit in the folder names (GPa, K, ...); default: "
                         "read off the folders")
    ap.add_argument("--steps", type=int, default=6000, help="sampler steps per seed")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--quick", action="store_true",
                    help="fit the position-median spectrum only, not every position")
    ap.add_argument("--calibrate-only", action="store_true",
                    help="stop after writing calibration.json (a0, E0 and the "
                         "processing parameters at every series point)")
    ap.add_argument("--max-locations", type=int, default=0,
                    help="fit at most this many positions per point (0: all)")
    # ---- the alloy
    ap.add_argument("--elements", nargs="+", default=None,
                    help="absorber edges in model order (needed unless the table "
                         "set's meta.json names them)")
    ap.add_argument("--composition", type=float, nargs="+", default=None,
                    help="atomic fractions in element order (default equiatomic)")
    ap.add_argument("--lattice", default=None, choices=list(SYS.A0_PER_NN),
                    help="sc, bcc, fcc or hcp (default from the tables, else fcc)")
    ap.add_argument("--a0-ref", type=float, default=None,
                    help="reference lattice constant the supercell is built at, and "
                         "the native paths computed at (hcp: the in-plane a). "
                         "Default: from the table set's nearest-neighbour spacing")
    ap.add_argument("--c-over-a", type=float, default=None, help="hcp only")
    ap.add_argument("--system-name", default=None,
                    help="label for titles (default: the element symbols)")
    ap.add_argument("--tables", default=None,
                    help="scattering paths: 'native' (default; computed in-process, "
                         "'native:seed=1,...' sets the recipe) or a table set under "
                         "data/ written by feff8l_tables.py or native_tables.py")
    ap.add_argument("--ms-tables", default=None,
                    help="the matching multiple-scattering set for a FEFF table set "
                         "(native computes its own); without one the model is "
                         "single-scattering and --rmax should stop before the first "
                         "collinear shell")
    ap.add_argument("--ms-library", default=None,
                    help="a path library under data/ (chimera/mslibrary.py): the "
                         "sampler then sums multiple scattering over each "
                         "configuration's own paths instead of using the "
                         "composition-averaged term. Off by default")
    ap.add_argument("--k-axis", default="constant", choices=list(SYS.K_AXES),
                    help="how the data's k maps onto the model's internal momentum. "
                         "'constant': one E0 gap per edge, which then also carries "
                         "the inner potential. 'table': through the scattering "
                         "calculation's own energy-dependent k(p), so dE0 is only "
                         "the E0 offset")
    ap.add_argument("--energy-prior", default=None,
                    help="a JSON file whose 'V' holds effective pair interactions "
                         "(nshell, nsp, nsp) in eV, as chimera.physics."
                         "pair_interactions writes: every sampler move is then "
                         "weighed by exp(-dE/kT) at --anneal-T as well as by the "
                         "data. Off by default")
    ap.add_argument("--anneal-T", type=float, default=None,
                    help="temperature (K) the sample's order was frozen in at, for "
                         "--energy-prior")
    # ---- the calibration
    ap.add_argument("--legacy-calibration", action="store_true",
                    help="the simple Nelder-Mead calibration (no dr, C3, per-shell "
                         "sigma^2; fixed k weight and window)")
    ap.add_argument("--kweights", nargs="+", default=None,
                    help="k weights the calibration chooses between; an entry "
                         "like 1,2,3 is one candidate fitted jointly. Default: "
                         "2 3 4 for |chi(R)| files, 1,2,3 for chi(k) files")
    ap.add_argument("--de-iter", type=int, default=120,
                    help="differential-evolution generations per k weight")
    ap.add_argument("--calibrate-points", type=int, nargs="+", default=None,
                    help="series points the shared calibration is fitted to "
                         "(default: those being fitted). Set it to the FULL "
                         "series when splitting a run across jobs, so each job "
                         "calibrates identically and only the per-point work "
                         "is divided.")
    ap.add_argument("--reuse-calibration", default=None, metavar="CALIBRATION_JSON",
                    help="skip the calibration search and take the shared "
                         "parameters from an existing calibration.json. Points it "
                         "does not cover get the shared parameters with a0 and "
                         "the sigma^2 scale refined per spectrum. Use it to ADD "
                         "points to a tree without refitting the ones already "
                         "there, or to keep a questionable point from steering "
                         "the calibration for the rest.")
    ap.add_argument("--kmin", type=float, default=C.KMIN)
    ap.add_argument("--kmax", type=float, default=C.KMAX,
                    help="upper window limit. With chi(k) this is OUR choice and "
                         "is applied to data and model alike, so it must leave "
                         "room for the taper inside the measured k range; the "
                         "run checks that and says so if it does not.")
    ap.add_argument("--per-edge-kmax", action=B, default=True,
                    help="give every edge the widest window ITS OWN data can "
                         "support, instead of cutting all edges at the shortest "
                         "(chi(k) only). Adds no parameters - it is purely more "
                         "data. --kmax is then the ceiling. Default on; "
                         "--no-per-edge-kmax for one shared window.")
    ap.add_argument("--kmax-margin", type=float, default=0.5,
                    help="how far inside each edge's last measured k the window "
                         "stops, with --per-edge-kmax (default 0.5, half the "
                         "default taper width)")
    ap.add_argument("--rmin", type=float, default=C.FIT_RMIN,
                    help=f"fit range start, A (default {C.FIT_RMIN}, set for fcc "
                         f"3d metals; start just inside the first-shell peak)")
    ap.add_argument("--rmax", type=float, default=C.FIT_RMAX,
                    help=f"fit range end, A (default {C.FIT_RMAX}: four fcc shells "
                         f"near a0 = 3.6 A; stop where the model's paths stop)")
    ap.add_argument("--free-a0", action=B, default=True,
                    help="let a0 float at every point, the reference included, "
                         "with the shared E0 (its common mode included) fitted "
                         "jointly with every point's a0 and sigma^2 in the joint "
                         "stage. Needs chi(k). --no-free-a0 anchors a0 at the "
                         "reference point to --a0-anchor and pins the common E0 "
                         "there.")
    ap.add_argument("--joint-a0", action=B, default=True,
                    help="in the joint calibration stage, fit every point's a0 and "
                         "sigma^2 (and its per-edge E0 shift) alongside the "
                         "shared parameters, instead of holding them at their "
                         "stage-2 values. With them held the common E0 cannot "
                         "move off the a0/E0 ridge.")
    ap.add_argument("--free-dE0-edge", action=B, default=True,
                    help="fit a MEAN-ZERO per-edge E0 shift at every point (nsp-1 "
                         "numbers per point), so that edges whose distance scales "
                         "do not follow one curve along the series are not forced "
                         "to, and the difference does not land on alpha.")
    ap.add_argument("--free-dE0-common", action=B, default=False,
                    help="ALSO fit a common E0 shift PER POINT on top of the "
                         "shared one. Off by default: with --free-a0 and "
                         "--joint-a0 the common mode is already free and shared "
                         "across the series; one per point lets each point walk "
                         "its own stretch of the a0/E0 ridge.")
    ap.add_argument("--joint-nuisance", action=B, default=True,
                    help="fit the sigma^2 scale and the mean-zero per-edge E0 "
                         "shifts JOINTLY with the first-shell order (least "
                         "squares on the pair-count model; a0 stays at its "
                         "random-configuration value), then sample at those "
                         "values. Without it the nuisances stay where the RANDOM "
                         "configuration put them - a profile at alpha = 0. "
                         "Default on.")
    ap.add_argument("--a0-anchor", type=float, default=None,
                    help="with --no-free-a0: lattice constant to hold at the "
                         "reference point while E0 is fitted (default: --a0-ref). "
                         "When the series does not start at ambient this must be "
                         "the independent value AT that point, or E0 absorbs the "
                         "difference. With --free-a0 it only seeds the search.")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    DATA = os.path.abspath(args.data)
    OUT = os.path.abspath(args.out)
    os.makedirs(OUT, exist_ok=True)
    print(f"output -> {OUT}")
    t_start = time.time()

    try:
        system = SYS.resolve(args.tables or "native", elements=args.elements,
                             composition=args.composition, lattice=args.lattice,
                             a0_ref=args.a0_ref, c_over_a=args.c_over_a,
                             ms_name=args.ms_tables, name=args.system_name,
                             k_axis=args.k_axis)
    except (ValueError, FileNotFoundError) as e:
        ap.error(f"cannot describe the alloy: {e}")
    elements, nsp = system.elements, system.nsp
    labels = system.wc_labels()
    nwc = len(labels)
    print(f"system: {system.name}  {'-'.join(elements)}  {system.lattice}  "
          f"a0_ref {system.a0_ref:.4f} A  composition "
          f"{np.round(system.composition, 3).tolist()}  tables {system.tables_name}"
          f"{' + ' + system.ms_name if system.ms_name else ' (no MS set)'}"
          f"  k axis {system.k_axis}")
    a0_anchor = float(args.a0_anchor) if args.a0_anchor is not None else system.a0_ref

    print("Loading and screening ...", flush=True)
    spectra, report = io.screen(DATA, verbose=True, elements=elements,
                                points=args.points, unit=args.unit)
    unit = spectra.unit
    lab = lambda P: io.point_label(P, unit)
    if not spectra.series:
        ap.error(f"no folders for {', '.join(elements)} under {DATA} "
                 f"(see chimera/dataio.py for the layout)")
    if args.points is None:
        args.points = list(spectra.series)
        print(f"  series points found: {', '.join(lab(P) for P in args.points)}")
    dups = io.duplicate_edges(spectra)
    io.write_qc(report, dups, os.path.join(OUT, "data_quality.json"), unit)
    # the calibration's reference point, which is where a0 is anchored - not
    # simply the first point being fitted, since a split run may not include it
    P_ref = min(args.calibrate_points or args.points)
    R_data = spectra[(P_ref, elements[0])]["R"]
    chik = all(io.has_chi_k(spectra, P) for P in args.points)
    kmax = args.kmax
    if chik:
        print("  chi(k) input: fitting the complex chi(R) under our own window")
        # The window is applied to the data too, so it has to fit inside the
        # measured k range of every edge used - taper included. Silently
        # running off the end would window the model and the data differently.
        kend = {(P, el): spectra[(P, el)]["k"][-1]
                for P in sorted(set(args.points)
                                | set(args.calibrate_points or ()))
                for el in elements if (P, el) in spectra}
        if args.per_edge_kmax:
            # Each edge gets its own ceiling, set by the SHORTEST of its own
            # spectra across EVERY point the tree will hold, not just the ones
            # this invocation fits, so one window serves the whole series and
            # the points stay comparable (a split run that sized the window
            # on its own half would hand the other half a window that overruns
            # its data).
            win_P = sorted(set(args.points) | set(args.calibrate_points or ()))
            kmax = [min(args.kmax,
                        np.floor((min(kend[(P, el)] for P in win_P
                                      if (P, el) in kend)
                                  - args.kmax_margin) * 20) / 20)
                    for el in elements]
            print("  per-edge k window: "
                  + ",  ".join(f"{el} {args.kmin}-{m:.2f}"
                               for el, m in zip(elements, kmax)))
            # The honest baseline is the widest SHARED window, which is
            # capped by the shortest edge - not --kmax, which is only a
            # ceiling and which no shared window could reach.
            shared_hi = min(kmax)
            gain = (np.mean([m - args.kmin for m in kmax])
                    / (shared_hi - args.kmin) - 1.0)
            print(f"     {gain * 100:+.0f}% mean window length against the "
                  f"widest shared window ({args.kmin}-{shared_hi:.2f} A^-1), "
                  f"which the shortest edge caps")
        need = np.asarray(kmax, float) + 0.5   # half the default dk_win taper
        lim = dict(zip(elements, np.atleast_1d(need) * np.ones(nsp)))
        short = {k: v for k, v in kend.items() if v < lim[k[1]]}
        if short:
            # Fit it anyway and say so. Where the window runs past the measured
            # range the data is windowed to zero there while the model is not,
            # so the fit is against a spectrum that stops early.
            print(f"  !! the window needs data to {np.round(need, 2)} A^-1, and")
            for (P, el), v in sorted(short.items()):
                hi = lim[el] - 0.5
                print(f"     {el} {lab(P)} stops at {v:.2f} A^-1 "
                      f"({100 * max(v - args.kmin, 0) / (hi - args.kmin):.0f}% "
                      f"of the window covered)")
            print("     those points are fitted, not skipped")
        else:
            print(f"  window fits inside every edge "
                  f"(narrowest measured range {min(kend.values()):.2f} A^-1)")
    elif args.free_a0 and not args.legacy_calibration and not args.reuse_calibration:
        print("  |chi(R)| input: a0 and E0 are degenerate without the phase, so a0 "
              "is anchored at the reference point (--no-free-a0)")
        args.free_a0 = False
    for line in io.describe_duplicates(dups, unit):
        print(f"  !! {line}")
    kweights = None if args.kweights is None else [as_kweights(k) for k in args.kweights]

    print("\nCalibrating the processing parameters ...", flush=True)
    if args.reuse_calibration:
        src = json.load(open(args.reuse_calibration))["calibration"]
        ref = str(min(int(k) for k in src))
        cal, S = {}, None
        for P in args.points:
            base = dict(src.get(str(P), src[ref]))
            base.update(point=P, unit=unit, rmin=base.get("rmin", args.rmin),
                        rmax=base.get("rmax", args.rmax))
            if str(P) not in src:
                # no entry of its own: take the shared parameters and let
                # fit_one's refine_local settle a0 and sigma^2 on the data
                base["a0_from"] = f"refined from the {lab(int(ref))} starting point"
            cal[P] = base
        S = M.Spectrum.from_cal(cal[args.points[0]])
        missing = [P for P in args.points if str(P) not in src]
        note = (f"calibration reused from {args.reuse_calibration}"
                + (f"; {', '.join(lab(P) for P in missing)} given the shared "
                   f"parameters with a0 and sigma^2 refined" if missing else ""))
        print(f"  reused from {args.reuse_calibration}")
        for P in args.points:
            print(f"  {lab(P)}: a0={cal[P]['a0']:.4f} A  "
                  + ("(own entry)" if str(P) in src else "(shared, a0 refined per fit)"))
    elif args.legacy_calibration:
        cal, S = C.calibrate_legacy(spectra, R_data, kmin=args.kmin, kmax=kmax,
                                    rmin=args.rmin, rmax=args.rmax,
                                    a0_ambient=a0_anchor, system=system)
        note = ("legacy: dE0 fitted at the reference point with a0 anchored, "
                "then held fixed")
    else:
        calib_P = args.calibrate_points or args.points
        if set(calib_P) != set(args.points):
            print(f"  calibrating on {', '.join(lab(p) for p in sorted(calib_P))}, "
                  f"fitting {', '.join(lab(p) for p in args.points)}")
        if args.free_dE0_common and not args.free_a0:
            print("  NOTE: --free-dE0-common without --free-a0 frees only one "
                  "arm of the a0/E0 pair, which is degenerate point by point; "
                  "free both or neither.", flush=True)
        cal, S = C.calibrate(spectra, R_data, kweights=kweights,
                             de_maxiter=args.de_iter, kmin=args.kmin, kmax=kmax,
                             rmin=args.rmin, rmax=args.rmax,
                             points=calib_P, anchor_a0=not args.free_a0,
                             a0_ambient=a0_anchor,
                             free_dE0_per_point=args.free_dE0_common,
                             free_dE0_edge_per_point=args.free_dE0_edge,
                             system=system, joint_a0=args.joint_a0)
        note = (f"global search at {lab(P_ref)}"
                + (" with a0 free" if args.free_a0
                   else f" with a0 anchored to {a0_anchor} A")
                + ": dE0 per edge"
                + (" (common mode included, fitted jointly with every point's "
                   "a0 and sigma^2)" if (args.free_a0 and args.joint_a0) else "")
                + (" plus a COMMON E0 shift per point"
                   if args.free_dE0_common else "")
                + (" plus a mean-zero per-edge E0 shift per point"
                   if args.free_dE0_edge else "") + ", "
                + ("sigma^2 scales" if chik else "k window and weight, sigma^2 scales")
                + ", pair-type bond offsets dr, C3, MS; then a0 and sigma^2 per "
                "point with the rest held"
                + (f"; target = complex chi(R) from measured chi(k), window "
                   + (", ".join(f"{el} {args.kmin}-{m:.2f}"
                                for el, m in zip(elements, kmax))
                      if args.per_edge_kmax else f"{args.kmin}-{kmax}")
                   + " A^-1" if chik else ""))
    band0 = C.band_report(S, S.random_config(0), M.match_grid(S.R, R_data),
                          io.fit_input(spectra, P_ref), R_data)
    # MERGE into any calibration already in the tree rather than replace it, so
    # adding a point to an existing run does not erase the ones beside it
    cal_path = os.path.join(OUT, "calibration.json")
    merged = {str(k): v for k, v in cal.items()}
    if os.path.exists(cal_path):
        prev = json.load(open(cal_path))
        keep = {k: v for k, v in prev.get("calibration", {}).items()
                if k not in merged}
        if keep:
            print(f"  keeping the existing calibration for "
                  f"{', '.join(lab(int(k)) for k in sorted(keep, key=int))}")
        merged = {**keep, **merged}
    eos = (C.equation_of_state({int(k): v for k, v in merged.items()})
           if io.is_pressure(unit) else None)
    if args.joint_nuisance:
        note += ("; fits take sigma^2 and mean-zero per-edge E0 from a joint "
                 "least-squares fit with the order (a0 held at its random-"
                 "configuration value), then sample at those values "
                 "(--joint-nuisance)")
    json.dump(dict(system=system.to_dict(), scattering=M.scattering_provenance(system),
                   unit=unit, calibration=merged,
                   equation_of_state=eos, validity_bands=band0,
                   fit_range=[args.rmin, args.rmax], note=note,
                   command=" ".join(sys.argv)),
              open(cal_path, "w"), indent=1, default=_j)
    del S
    if args.calibrate_only:
        print(f"\ncalibration only: {cal_path} written in "
              f"{(time.time() - t_start) / 60:.1f} min")
        return

    JOINT = 1 if args.joint_nuisance else 0
    if JOINT:
        print("\nNuisance treatment: sigma^2 and mean-zero per-edge E0 fitted "
              "jointly with the order (a0 held at its random-configuration "
              "value), then sampled at those values (--joint-nuisance)", flush=True)
    prior = None
    if args.energy_prior:
        if not args.anneal_T:
            ap.error("--energy-prior needs --anneal-T")
        prior = (np.asarray(json.load(open(args.energy_prior))["V"], float), float(args.anneal_T))
        print(f"\nEnergy prior: pair interactions from {args.energy_prior} at "
              f"{args.anneal_T:.0f} K (the systematic band is computed without it)", flush=True)
    if args.ms_library:
        print(f"\nExact multiple scattering in the sampler: path library {args.ms_library} "
              "(the systematic band uses the averaged term)", flush=True)
    per_point, wc_bands = {}, {}
    for P in args.points:
        t0 = time.time()
        c = cal[P]
        print(f"\n=== {lab(P)} ===", flush=True)
        pdir = os.path.join(OUT, io.point_dir(P, unit))
        os.makedirs(pdir, exist_ok=True)
        S = M.Spectrum.from_cal(c)
        if args.ms_library:
            S.attach_ms_library(args.ms_library)
        idx = M.match_grid(S.R, R_data)
        mask_d = C.fit_mask(R_data, c["rmin"], c["rmax"])
        r_grid = rdf_grid(S, c["rmax"])
        locs = io.locations(spectra, P)
        if args.max_locations:
            locs = locs[:args.max_locations]

        # ---------------- averaged
        avg = io.fit_input(spectra, P)
        spr = io.spread(spectra, P) / np.sqrt(max(len(locs), 1))
        print("  averaged: systematic band over calibration variants ...", flush=True)
        wcb = F.systematic_band(R_data, avg, c, seeds=(0,), nsteps=args.steps,
                                verbose=True, joint=bool(JOINT), free_dE0=bool(JOINT))
        wc_bands[P] = wcb
        json.dump(wcb["variants"], open(os.path.join(OUT, f"_variants_{P}.json"), "w"),
                  indent=1, default=_j)
        res = F.fit_one(S, R_data, avg, idx, mask_d, c["a0"], c["sigma_scale"],
                        seeds=tuple(range(args.seeds)), nsteps=args.steps,
                        joint=bool(JOINT), free_dE0=bool(JOINT), energy_prior=prior)
        per_point[P] = {k: res.get(k) for k in
                        ("a0", "nn_distance", "sigma2_nn", "r_factor",
                         "r_factor_ideal", "wc", "chi2", "wc_crlb",
                         "wc_crlb_noise", "wc_determined", "wc_posterior_sd")}
        save_location(os.path.join(pdir, "averaged"), S, res, R_data, avg, mask_d,
                      f"{system.name} {lab(P)} — median over {len(locs)} positions",
                      r_grid, spread=spr, wc_band=wcb,
                      extra=dict(point=P, unit=unit, n_locations=len(locs),
                                 locations=locs, kind="location-median"))
        print(f"    a0={res['a0']:.4f} A  r_NN={res['nn_distance']:.4f} A  "
              f"sigma2={res['sigma2_nn']:.5f}  R={res['r_factor']:.4f}  "
              f"alpha 1-sigma bounds={np.round(res.get('wc_crlb', []), 2)}", flush=True)

        # ---------------- per position
        if not args.quick:
            rows = []
            for n, s in enumerate(locs):
                data = io.fit_input(spectra, P, s)
                r = F.fit_one(S, R_data, data, idx, mask_d, c["a0"],
                              c["sigma_scale"], seeds=tuple(range(args.seeds)),
                              nsteps=args.steps, collect=False,
                              joint=bool(JOINT), free_dE0=bool(JOINT),
                              energy_prior=prior)
                save_location(os.path.join(pdir, f"sample{s}"), S, r, R_data, data,
                              mask_d, f"{system.name} {lab(P)} — position {s}",
                              r_grid, wc_band=wcb,
                              extra=dict(point=P, unit=unit, sample=int(s),
                                         kind="single position"))
                rows.append([s, r["a0"], r["nn_distance"], r["sigma2_nn"],
                             r["r_factor_ideal"], r["r_factor"]] + list(r["wc"])
                            + list(r["scales"]))
                if (n + 1) % 10 == 0 or n == len(locs) - 1:
                    print(f"    {n+1}/{len(locs)} positions "
                          f"[{time.time()-t0:.0f}s]", flush=True)
            write_csv(os.path.join(pdir, "summary_by_location.csv"),
                      ["sample", "a0_Angstrom", "r_NN_Angstrom", "sigma2_NN_A2",
                       "R_factor_ideal", "R_factor"] + labels
                      + [f"scale_{e}" for e in elements], rows)
            if not rows:
                print("    no position holds every edge: nothing to fit per position")
                continue
            A = np.array(rows, float)
            if len(rows) > 1:
                FIG.location_map(os.path.join(pdir, "map_lattice_constant.png"),
                                 P, A[:, 0], A[:, 1], r"$a_0$  (Å)",
                                 f"{system.name} {lab(P)} — fitted lattice constant by position")
                FIG.location_map(os.path.join(pdir, "map_sigma2.png"),
                                 P, A[:, 0], A[:, 3] * 1e3,
                                 r"$\sigma^2_{\rm NN}$  ($10^{-3}$ Å$^2$)",
                                 f"{system.name} {lab(P)} — first-shell disorder by position")
                first = labels[0][2:]
                FIG.location_map(os.path.join(pdir, f"map_wc_{first}.png"),
                                 P, A[:, 0], A[:, 6], rf"$\alpha_{{\rm {first}}}$",
                                 f"{system.name} {lab(P)} — a_{first} by position "
                                 f"(systematic band {wcb['wc_min'][0]:+.2f} to "
                                 f"{wcb['wc_max'][0]:+.2f})")
            stats = dict(n=len(rows),
                         a0_median=float(np.median(A[:, 1])), a0_sd=float(A[:, 1].std()),
                         sigma2_median=float(np.median(A[:, 3])),
                         r_factor_median=float(np.median(A[:, 5])),
                         wc_median=np.median(A[:, 6:6 + nwc], axis=0).tolist(),
                         wc_sd_over_locations=A[:, 6:6 + nwc].std(axis=0).tolist())
            json.dump(stats, open(os.path.join(pdir, "summary_by_location.json"), "w"),
                      indent=1, default=_j)
            per_point[P]["location_stats"] = stats
        del S
        print(f"  [{lab(P)} done in {time.time()-t0:.0f}s]", flush=True)

    json.dump(dict(system=system.to_dict(), scattering=M.scattering_provenance(system),
                   unit=unit,
                   per_point={str(k): v for k, v in per_point.items()},
                   wc_systematic_band={str(k): {kk: vv for kk, vv in v.items()
                                                if kk != "variants"}
                                       for k, v in wc_bands.items()},
                   equation_of_state=eos),
              open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=_j)
    if len(per_point) > 1:
        FIG.series_summary(os.path.join(OUT, "series_summary.png"),
                           per_point, wc_bands, elements, unit)
    write_csv(os.path.join(OUT, "summary_by_point.csv"),
              ["point", "unit", "a0_Angstrom", "r_NN_Angstrom", "sigma2_NN_A2",
               "R_factor_ideal", "R_factor"]
              + labels + [f"{l}_sys_lo" for l in labels]
              + [f"{l}_sys_hi" for l in labels],
              [[P, unit, per_point[P]["a0"], per_point[P]["nn_distance"],
                per_point[P]["sigma2_nn"], per_point[P]["r_factor_ideal"],
                per_point[P]["r_factor"]] + list(per_point[P]["wc"])
               + list(wc_bands[P]["wc_min"]) + list(wc_bands[P]["wc_max"])
               for P in sorted(per_point)])
    print(f"\nAll done in {(time.time()-t_start)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
