# CHIMERA

**CHI-space Multi-Edge Reconstruction of Alloys**

Reconstructs the atomic configuration of a substitutional alloy from
multi-edge EXAFS — measured `chi(k)`, or `|chi(R)|` when that is all there
is — and reports Warren-Cowley short-range-order parameters and partial
radial distribution functions with a systematic uncertainty budget.

## What it does

* Reads `chi(k)` or `|chi(R)|` per absorption edge and per sample position
  (the kind is read off the files), screens it for quality, and forms
  position medians.
* With `chi(k)`, transforms data and model with one window of our choosing
  and fits the **complex** `chi(R)` — Re and Im — with as many k weights at
  once as asked for (`k^1, k^2, k^3` together by default). The phase pins E0
  and the distances, which is what separates `sigma^2` from the
  short-range order; `|chi(R)|` alone cannot (see `chimera/chik.py`).
* With `|chi(R)|`, infers the processing parameters the files no longer
  record — E0 per edge, the k window and k weight — together with the
  lattice constant, the Debye-Waller scales, the pair-type bond-length
  offsets and the first-shell third cumulant, by a global fit of the
  random-alloy model to the data.
* Forward-models `chi(k)` on the GPU from a supercell (500 atoms by default)
  using FEFF8L scattering tables, adds the FEFF8L multiple-scattering paths,
  and transforms with the same window as the data.
* Samples species configurations by parallel-tempering reverse Monte Carlo,
  with an incremental delta-chi so a move costs about 1/125 of a full
  recomputation.
* Reports alpha and `g_AB(r)` per shell, with the Cramer-Rao bound each alpha
  carries at the fit's own residual level, the posterior spread, and the
  spread over calibration variants.

## Requirements

```
numpy  scipy  matplotlib  cupy   (CUDA GPU required)
```

`cupy` must match the installed CUDA toolkit. Nothing in the fitting path
imports `larch` — the scattering tables are plain text read with numpy.

## Scattering tables

CHIMERA computes its own scattering paths, in-process, with no FEFF and no
table files: name the table set `native`.

```bash
python scripts/run_all.py --data <root> --out <tree> --tables native
python scripts/run_all.py ... --tables native --elements Fe Ni --lattice fcc --a0-ref 3.59
```

```python
S = Spectrum(system="native")            # equiatomic fcc CrCoNi, paths computed here
```

The native engine (below) builds the nearest-neighbour single-scattering
path for every absorber-scatterer pair and every 3- and 4-leg path to
5.3 A, at the system's reference spacing, in about 4 s for CrCoNi; the
result is cached in `data/_cache` (or `$CHIMERA_CACHE`), so later runs load
it in under a second. `native:seed=1,rpath=5.8` changes the recipe (the
random cluster, the path cutoff, `nleg`, `radius`, `exchange`, `kmax`). The
engine version and recipe go into `calibration.json` and `summary.json`
under `scattering`. The paths are handed to the fit through the same
conversion the file reader uses, so they equal the files `native_tables.py`
writes to the rounding of the text format (`tests/test_native_provider.py`).

The default CrCoNi system still names the FEFF8L set, so earlier results
reproduce unchanged. Those tables are **not** shipped; generating them is
the only step that needs `xraylarch` (which bundles the FEFF8L binaries),
and it runs in a *separate* interpreter that the main environment never
imports:

```bash
python feff8l_tables.py --larch-python <venv>/Scripts/python.exe \
    --elements Cr Co Ni --d-nn 2.517 --name crconi_fcc2517

python feff8l_tables.py --larch-python <venv>/Scripts/python.exe \
    --elements Cr Co Ni --d-nn 2.517 --name crconi_fcc2517_ms \
    --rpath 5.30 --nleg 4
```

The first call writes the nearest-neighbour single-scattering paths (one FEFF
run per absorber, every absorber-scatterer pair present); the second the
multiple-scattering set. Both land in `data/<name>/` alongside a `meta.json`
recording the cluster they were computed for. Point the model at them with
`Spectrum(tables_name=..., ms_name=...)`.

### The native engine

`native_tables.py` takes the same arguments as `feff8l_tables.py` and
writes the same files from CHIMERA's own ab initio engine
(`exafs_gpu/abinitio/`), for inspection or for other codes; fits do not
need them (`--tables native` above):

```bash
python native_tables.py --elements Cr Co Ni --d-nn 2.517 --name crconi_fcc2517_native
python native_tables.py --elements Cr Co Ni --d-nn 2.517 --name crconi_fcc2517_native_ms \
    --rpath 5.3 --nleg 4
```

The chain is FEFF's, written from the physics: self-consistent LDA atoms
(ground state, and the absorber with a screened 1s hole), overlapped-atom
muffin-tin potentials, the Hedin-Lundqvist plasmon-pole self-energy (energy-
dependent exchange and the mean free path), complex phase shifts, path
enumeration with degeneracies, and an exact curved-wave multiple-scattering
path sum in the angular-momentum basis - no Rehr-Albers separable truncation.
Phase shifts, the loss integral and the path sum are compiled CUDA kernels; a
full multiple-scattering set takes about 3 s per absorber on an RTX 4090.

Checked against independent references (`tests/test_abinitio_*.py` on the
development branch): the atoms reproduce the NIST LDA tables to 1e-6 Ha; the
path list and degeneracies reproduce FEFF8L's exactly; single-scattering
amplitudes agree with FEFF8L to 1-2% (10% at p = 4 A^-1, where FEFF adds a
sub-threshold loss) with a slowly varying phase offset of 0.1-0.3 rad, larger
where Cr scatters. That offset is FEFF8L's self-consistent potential, which
the native engine does not have yet (it uses overlapped atoms, so its Fermi
level and Cr charge differ). Fitted to Kuzmin's Cu and Cr foil temperature
series with one Debye temperature, the native tables give S0^2 = 0.87 and
theta_D = 323 K for Cu (literature 315-330 K) and R-factors of 0.02-0.04
(`scripts/test_native_foils.py`).

## Blind synthetic data

`scripts/make_blind_dataset.py` writes a CrCoNi pressure series in the staged
export layout, generated from an atomistic model that shares none of the
fit's approximations: chemical order from Monte Carlo on random pair
interactions (a no-order control one time in four), Morse-bonded Langevin
dynamics at 300 K, single scattering for every pair of every snapshot and
multiple scattering computed exactly on the distorted geometry, with S0^2,
E0 offsets, a smooth theory error, noise and a residual background drawn at
random. The answer is sealed with a passphrase and committed to by its
SHA-256 in the dataset's `manifest.json`:

```bash
python scripts/make_blind_dataset.py --out ".../Blind/blind_001" --key-dir ".../Blind_keys"
python scripts/run_all.py ...                        # fit it like any measured series
python scripts/unblind.py --data ".../Blind/blind_001" --tree <results tree> \
    --key ".../Blind_keys/blind_001.KEY.txt"
```

## Using it

```python
from chimera import dataio as io, calibrate as C, model as M, fitrun as F

spectra, report = io.screen("path/to/data")     # per-edge chi(k) or |chi(R)|, screened
R_data = spectra[(0, "Cr")]["R"]                # the common R grid

cal, S = C.calibrate(spectra, R_data)           # E0/edge, a0, sigma^2 scales, dr,
                                                # C3, MS (+ window and k weight
                                                # when only |chi(R)| is available)

c    = cal[0]                                   # calibration for series point 0
S    = M.Spectrum.from_cal(c)
idx  = M.match_grid(S.R, R_data)
mask = C.fit_mask(R_data, c["rmin"], c["rmax"])

data = io.fit_input(spectra, 0)                 # a ChiK, or the |chi(R)| median
res  = F.fit_one(S, R_data, data, idx, mask,
                 c["a0"], c["sigma_scale"], seeds=(0, 1))

res["wc"]        # Warren-Cowley alpha, one per unordered pair
res["wc_crlb"]   # Cramer-Rao bound on each, at this fit's residual level
res["r_factor"]
res["target_complex"], res["kweights"]   # what was fitted
```

`chi(k)` arrays from anywhere else go in as `chik.ChiK(k, chi)` (one row per
edge, in element order); `R_data` is then `data.R`, the model's own grid.
`S.set_window(kweight=(1, 2, 3))` fits several k weights at once.

Another alloy is one object away. The elements, composition, lattice,
reference cell and table names live in an `AlloySystem` (`chimera/system.py`),
which every class derives its species count from and writes into every
record:

```python
from chimera.system import AlloySystem
sys = AlloySystem.from_tables("feni_bcc2482", ms_name=None, composition=(0.5, 0.5))
S = M.Spectrum(system=sys)                      # or Spectrum(elements=("Fe", "Ni"),
                                                #    lattice="bcc", a0_ref=2.87, tables="analytic")
cal, S = C.calibrate(spectra, R_data, system=sys)
```

`from_tables` reads the `meta.json` that `feff8l_tables.py` writes (elements,
composition, lattice, spacing); with nothing given the default is equiatomic
fcc CrCoNi, exactly as before. `scripts/run_all.py --tables <set>` or
`--elements .. --lattice .. --a0-ref ..` does the same from the command line,
and the series points are read off the data folders. The CUDA kernels are
compiled per species count, the pair-count model, the bounds, the sampler,
the labels and the figures all follow it; `tests/test_general_system.py`
runs a binary on bcc, a quaternary on fcc and a binary on hcp end to end.

`F.systematic_band(...)` re-fits under each admissible calibration variant and
returns the spread — the part of the uncertainty the data cannot resolve.
`F.broadened_partial_rdf(...)` and `F.shell_table(...)` turn a fitted
configuration into `g_AB(r)` and per-shell coordination numbers.

Athena project files are read by `scripts/export_athena.py`, which needs a
Python with `xraylarch` (passed as `--larch-python`, as for the tables) and
writes `k, chi` files in the layout above plus a `processing.json` with the
E0, background and normalisation Athena used.

## Layout

```
chimera/
  dataio.py        loading (chi(k) or |chi(R)| files), quality screening,
                   per-position grouping
  chik.py          chi(k) data: the shared transform to complex chi(R),
                   k-weight channels, R-space noise
  model.py         configuration -> chi(k) -> chi(R); cheap calibration updates
  shellmodel.py    the same model in pair-count space (CPU, exact for swap-only
                   fits): calibration objective, Cramer-Rao bounds on alpha
                   (white, or with a full residual covariance)
  system.py        the alloy: elements, composition, lattice, reference cell,
                   table names; the default is equiatomic fcc CrCoNi
  multiscat.py     multiple-scattering paths (3- and 4-leg), from a table
                   directory or handed over in memory by the native engine
  multift.py       per-edge Fourier transform (each edge has its own E0)
  calibrate.py     global search for E0, k window and weight, a0, sigma^2
                   scales, bond offsets dr, C3, MS amplitude
  fitrun.py        one spectrum: refine lattice, sample, bounds, systematic band
  figures.py       fit comparison, partial RDFs, alpha by shell
exafs_gpu/
  abinitio/        native scattering engine: atom.py, potential.py,
                   selfenergy.py, phaseshift.py, sphere.py, pathsum.py,
                   paths.py, cluster.py, tables.py; provider.py computes
                   the paths in-process for a fit ("native", cached);
                   synth.py builds the blind synthetic data
  scattering.py    scattering tables; feffNNNN.dat reader and the shared
                   convention conversion
  lattice.py       supercell, neighbour lists, force constants, Warren-Cowley
  forward.py       GPU forward model and analytic adjoint
  fourier.py       k -> R transform, windows, adjoint
  kernels.py       CUDA source (NVRTC via CuPy)
methods/
  method_a_rmc.py  parallel-tempering RMC sampler
scripts/
  run_all.py       fit a whole tree; postprocess.py runs the tables, tests,
                   grading, figures and README on it
  correlated_bound.py   Cramer-Rao bound with the residual's correlation in R
                        and between k-weight channels (model-discrepancy view)
  detectable_sro.py     minimum detectable order by injection-recovery against
                        the production residual: the experiment's power
  design_experiment.py  information budget: what each edge, k band, window
                        extension and auxiliary measurement is worth
  test_*.py        the analysis record behind testing/MODEL_CHANGES.md
                   (development branch only)
tools/
  promote_to_main.py   builds main from development (see Branches)
  make_blind_dataset.py, unblind.py   blind synthetic series and their scoring
feff8l_tables.py   generates data/ for an alloy on sc, bcc, fcc or hcp
                   (needs xraylarch)
native_tables.py   the same tables from exafs_gpu/abinitio, no FEFF needed
```

## Branches

Work is committed to `development`, which holds everything: the library,
the scripts, `tests/` and the `scripts/test_*.py` investigations. `main` is
the same history without the paths listed in `tools/devonly_paths.txt`, so
no commit on `main` contains tests. After committing to `development`, run

```bash
python tools/promote_to_main.py
```

to append the new commits to `main` (it never touches the working tree, and
refuses to drop commits that exist only on `main`).

## Scope and limits

* **Lattice.** `CrystalSupercell` handles `sc`, `bcc`, `fcc` and `hcp`,
  including the non-orthogonal hcp cell, and `Spectrum`, `feff8l_tables.py`
  and the CIF export take the lattice from the `AlloySystem`. For hcp the
  reference `a0_ref` is the in-plane constant `a`.
* **Species count.** Any number of species: the CUDA source takes `NSP` as a
  compile-time define, and the Python side reads it from the system. The
  mean-zero per-edge E0 shift has `nsp - 1` free entries, the bond-offset
  matrix `nsp(nsp+1)/2 - 1`, and there are `nsp(nsp-1)/2` order directions
  per shell.
* **Multiple scattering** needs a FEFF8L path set for the alloy
  (`feff8l_tables.py --nleg 4 --rpath ..`); without one the model is
  single-scattering and the fit range should stop before the first
  collinear shell (`FIT_RMAX_NOMS`).
* **Site equivalence.** Every basis site must have the same coordination: the
  neighbour-list machinery and the MC proposal kernels assume a uniform
  coordination number.
* **Input layout.** `dataio.screen` expects `<Element>_<N><tag>/…sample<M>.csv`
  with columns `k, chi` (measured chi(k); Athena's `.chi` files with a `#`
  header are read too) or `R, chi_mag` (magnitude only), where `<N>` indexes
  a series (pressure, temperature, composition) and `<M>` a position on the
  sample. A folder holds one kind on one grid. Data arranged otherwise can
  be handed to `calibrate` and `fit_one` directly — they take arrays or
  `ChiK` objects, not paths. Raw `mu(E)` is not read: take it through
  Athena / Larch (or `scripts/export_athena.py`) to `chi(k)` first.
* **Composition** is conserved by construction: moves are species swaps, so a
  fit cannot drift away from the composition you set.

## Conventions worth knowing

* Tables are on the **internal momentum**, not k relative to E0.
  `ScatteringTables.e0_offset_ev` records where k = 0 sits relative to the
  Fermi level (10.3 eV for the CrCoNi tables above); expect a fitted E0 shift
  of about that size against data referenced to the edge. That gap is not a
  constant: with the self-energy it falls from ~15 to ~6 eV (native) or
  ~11 to ~1 eV (FEFF8L) over k = 3-13. `--k-axis table` (the system's
  `k_axis`) maps the data's k through the tables' own `k_of_p`, so the
  fitted dE0 is only the offset between the experimenter's E0 and the
  calculation's Fermi level; the default `constant` keeps one gap per edge,
  as every fit before 2026-11-09 did. The mode is written into every
  calibration and fit record and restored on reload.
* Warren-Cowley sign: **positive = like atoms avoid, negative = like atoms
  cluster**. Checked on constructed configurations — a segregated cell gives
  alpha_AA = -1.38, an A-avoiding one +0.43.
* Moves are species swaps. `swap_frac=1.0` always *attempts* a swap, but when
  all 8 candidate partners drawn share the atom's species — probability
  (1/n)^8 for n equiatomic species — the proposal falls through to a small
  displacement. Over a full run that leaves an RMS drift under 3 mA, about 2%
  of sigma.
* Fitting is done in R space over a windowed R range, against the complex
  `chi(R)` when `chi(k)` was measured and against `|chi(R)|` otherwise; the
  R-factor quoted everywhere is `sum |fit - data|^2 / sum |data|^2` on that
  window. A complex target counts Re and Im as separate residuals: `chi^2`
  divides by twice the number of points, the sampler's `nfit` does the same,
  and every noise or model-error sigma is the per-component value. Results
  record `target_complex` and `kweights`.
* Several k weights are fitted as extra channels: `chi_R(...)` returns one
  row per (edge, k weight), edge-major, `S.chan_edge` names the edge of each,
  and the profiled amplitude is one value per edge over all its channels.
* With `chi(k)` the window is a choice applied to data and model alike, so
  `calibrate` does not search it, and `anchor_a0=False` is allowed (the
  phase separates a0 from E0). From `|chi(R)|` alone both remain as before.
* With atoms on ideal sites, chi depends on the configuration only through the
  pair counts per shell, `N[absorber, scatterer, shell]`. `shellmodel` uses
  that: the calibration objective and the alpha bounds run on the CPU in
  milliseconds, and agree with the GPU pair sum to 1e-15.
* **a0 and E0 (settled 2026-10-07).** a0 is free at every series point and
  the shared per-edge E0 - its common mode included - is fitted in the
  joint calibration stage together with every point's a0 and sigma^2, with a
  mean-zero per-edge E0 shift per point on top. The common mode is one
  number for the whole series, not one per point: the fitted per-point
  values agreed within 0.13 eV when tried, and one per point lets each
  point walk its own stretch of the a0/E0 ridge, which shows up as scatter
  in the equation of state. The cost is that the absolute a0 lands below
  the diffraction value (about 0.7% on the CrCoNi data) and needs an
  independent lattice constant before it is quoted; the compression and
  the bulk modulus are the better-posed numbers. `--no-free-a0` restores
  the anchored treatment.
* `dr[a, b]` is the nearest-neighbour bond-length offset of pair type ab
  (Angstrom, symmetric, composition-weighted mean zero so a0 keeps the mean);
  `C3` is the first-shell third cumulant (Angstrom^3), entering the phase as
  `-4/3 k^3 C3`. Both are zero unless calibrated.
* Every fit reports `wc_crlb`: the 1-sigma Cramer-Rao bound on each alpha at
  the fit's own residual level, with all shells and the nuisance parameters
  free. A bound above 0.3 marks the value "not determined", whatever the
  seed-to-seed scatter says.
* The sampler profiles the per-edge amplitude inside chi^2 (the same closed
  form the R-factor uses) and draws posterior samples from the rung at
  beta = 1/2, which is the likelihood temperature for its chi^2 definition.
