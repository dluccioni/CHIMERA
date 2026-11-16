# CHIMERA

**CHI-space Multi-Edge Reconstruction of Alloys**

Reconstructs the atomic configuration of a substitutional alloy from
multi-edge EXAFS — measured `chi(k)`, or `|chi(R)|` when that is all there
is — and reports Warren-Cowley short-range-order parameters and partial
radial distribution functions with a systematic uncertainty budget.

## What it does

* Reads `chi(k)` or `|chi(R)|` per absorption edge and per sample position
  (the kind is read off the files), screens it for quality, and forms
  position medians. A series of measurements (pressures, temperatures,
  compositions) is fitted point by point with shared processing parameters.
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
  with single-scattering paths for every absorber-scatterer pair and the 3-
  and 4-leg multiple-scattering paths, computed by its own ab initio engine
  or read from FEFF output, and transforms with the same window as the data.
* Samples species configurations by parallel-tempering reverse Monte Carlo,
  with an incremental delta-chi so a move costs about 1/125 of a full
  recomputation.
* Reports alpha and `g_AB(r)` per shell, with the Cramer-Rao bound each alpha
  carries at the fit's own residual level, the posterior spread, and the
  spread over calibration variants.

## Requirements

```
python 3   numpy  scipy  matplotlib  cupy   (CUDA GPU required)
```

`cupy` must match the installed CUDA toolkit. Optional, for the opt-in
features below: `numba` (lattice Monte Carlo in `chimera/physics.py`), a
separate Python with `xraylarch` (FEFF8L tables, Athena import), FEFF10
binaries. Nothing in the fitting path imports `larch`.

## Quick start

**1. Arrange the data.** One folder per absorption edge and series point,
one file per sample position:

```
<data root>/
    Fe_300K/   scan_sample1.csv  scan_sample2.csv  ...
    Ni_300K/   scan_sample1.csv  scan_sample2.csv  ...
    Fe_600K/   ...
    Ni_600K/   ...
```

* Folder names are `<Element>_<N><unit>`: `<N>` is an integer indexing the
  series (a pressure, a temperature, a composition step) and `<unit>` any
  letters after it, one unit per tree (`Cr_3GPa`, `Fe_300K`). A prefix is
  allowed (`chik_Cr_3GPa`). For a single measurement name the folders by the
  element alone (`Fe/`, `Ni/`).
* Files are CSV or whitespace tables with columns `k, chi` (chi(k), e.g. an
  Athena or Larch export; Athena's `.chi` files with a `#` header are read
  as they are) or `R, chi_mag` (`|chi(R)|` only). A folder holds one kind on
  one grid.
* `sample<M>` in the file name (or else the last integer in it) is the
  position on the sample. Positions are fitted one by one and as a median;
  the per-position fits pair the edges by that index.
* Raw `mu(E)` is not read. Take it through Athena or Larch to `chi(k)` first;
  `export_athena.py` writes this layout from Athena project files:

  ```bash
  python export_athena.py --larch-python <venv>/Scripts/python.exe \
      --element Fe --point 300 --unit K --out <data root> fe_300K.prj
  ```

**2. Fit.**

```bash
python -m chimera.run --data <data root> --out <results> \
    --elements Fe Ni --lattice fcc --a0-ref 3.59
```

`--elements` lists the absorbing elements in the order the model uses,
`--composition` their atomic fractions (default equiatomic), `--lattice`
one of `sc`, `bcc`, `fcc`, `hcp` (with `--c-over-a`), and `--a0-ref` the
reference lattice constant the supercell and the scattering paths are built
at (for hcp the in-plane `a`). `--quick` fits only the position medians;
`--calibrate-only` stops after the processing calibration. Every option is
listed by `python -m chimera.run --help`.

**3. Read the results.**

```
<results>/calibration.json          the shared processing calibration, per point
<results>/data_quality.json         QC cuts and the duplicate-folder check
<results>/summary.json              per point: a0, r_NN, sigma^2, R-factor, alpha,
                                    bounds; the equation of state for a GPa series
<results>/summary_by_point.csv      the same as a table
<results>/series_summary.png        structure and the alpha band along the series
<results>/<N><unit>/averaged/       fit of the position median, with the
                                    calibration-systematic band
<results>/<N><unit>/sample<M>/      fit of one position
```

Each fit folder holds `warren_cowley.csv` (alpha, seed scatter, posterior
sd, Cramer-Rao bound, systematic band, and whether the value is determined),
`partial_rdf.csv`, `shell_coordination.csv`, `configuration.csv` (the
species on every site), `fit.json`, and figures of the fit in R (and k) and
of the order by shell.

Check before quoting a number:

* **Fit range.** The default R range, 1.4-5.0 A, covers four shells of an
  fcc 3d-metal alloy with a0 near 3.6 A. Set `--rmin`/`--rmax` for anything
  else; `validity_bands` in calibration.json scores the model band by band
  against the data and the noise.
* **Whether alpha is measured.** A Cramer-Rao bound (`wc_crlb`) above 0.3
  marks an alpha as not determined, whatever the seed-to-seed scatter says,
  and the systematic band (`sys_lo`, `sys_hi`) is usually the larger
  uncertainty.
* **The absolute a0** inherits the scattering calculation's distance error;
  compare it with an independent lattice constant (diffraction). Changes of
  a0 along a series are better determined than its absolute value.

## Scattering paths

By default the paths are computed in-process by CHIMERA's own engine for the
alloy as described on the command line (`--tables native`): no FEFF and no
table files. `native:seed=1,rpath=5.8` changes the recipe (the random
cluster, the path cutoff, `nleg`, `radius`, `exchange`, `kmax`). The first
run computes the nearest-neighbour single-scattering path for every
absorber-scatterer pair and every 3- and 4-leg path to 5.3 A, in seconds to
tens of seconds; the result is cached in `data/_cache` (or
`$CHIMERA_CACHE`), so later runs load it in under a second. The engine
version and recipe go into `calibration.json` and `summary.json` under
`scattering`.

```python
from chimera.system import AlloySystem
from chimera.model import Spectrum
alloy = AlloySystem(elements=("Fe", "Ni"), lattice="fcc", a0_ref=3.59,
                    tables_name="native", ms_name="native")
S = Spectrum(system=alloy)
```

### The native engine

The chain is FEFF's, written from the physics (`exafs_gpu/abinitio/`):
self-consistent LDA atoms (ground state, and the absorber with a screened 1s
hole), overlapped-atom muffin-tin potentials, the Hedin-Lundqvist
plasmon-pole self-energy (energy-dependent exchange and the mean free path),
complex phase shifts, path enumeration with degeneracies, and an exact
curved-wave multiple-scattering path sum in the angular-momentum basis - no
Rehr-Albers separable truncation. Phase shifts, the loss integral and the
path sum are compiled CUDA kernels; a full multiple-scattering set takes a
few seconds per absorber on a current GPU.

Checked against independent references: the atoms reproduce the NIST LDA
tables to 1e-6 Ha; the path list and degeneracies reproduce FEFF8L's
exactly; single-scattering amplitudes agree with FEFF8L to 1-2% with a
slowly varying phase offset of 0.1-0.3 rad, most of it from FEFF8L's
self-consistent potential. Fitted to published low-temperature Cu and Cr
foil spectra, the native paths put the first-shell bond length within
0.1% of dilatometry (FEFF8L: 0.26-0.40%) and give theta_D = 323 K for Cu
(literature 315-330 K).

More physics is chosen in the recipe (`--tables native:rel=1,scf=1,sigma=mpse`;
`native_tables.py --rel --scf --sigma mpse`); the defaults are the engine above.

* `scf=1`: self-consistent potentials from the cluster Green's function
  (`scf.py`): charge transfer and the Fermi level of the solid.
* `rel=1`: scalar-relativistic atoms (Koelling-Harmon, relativistic exchange)
  and scattering states, with the relativistic interstitial momentum.
* `sigma=mpse`: the many-pole self-energy of the material's measured loss
  function (`losses.py`: CXRO f1/f2 and Werner et al. 2009 REELS optical
  constants, downloaded on first use, mixed by composition and held to the
  perfect-screening sum rule).
* `quinn=1`: Quinn's electron-hole loss below the plasmon (HL only).

`native_tables.py` writes the same paths as feffNNNN.dat files for
inspection or for other codes; fits do not need them:

```bash
python native_tables.py --elements Fe Ni --d-nn 2.538 --name feni_fcc2538_native
python native_tables.py --elements Fe Ni --d-nn 2.538 --name feni_fcc2538_native_ms \
    --rpath 5.3 --nleg 4
```

### FEFF8L and FEFF10 tables

`feff8l_tables.py` runs FEFF8L (bundled with `xraylarch`) on a cluster of
the alloy, in a *separate* interpreter that the main environment never
imports, and writes `data/<name>/` with a `meta.json` recording the
elements, composition, lattice and spacing:

```bash
python feff8l_tables.py --larch-python <venv>/Scripts/python.exe \
    --elements Fe Ni --d-nn 2.538 --name feni_fcc2538
python feff8l_tables.py --larch-python <venv>/Scripts/python.exe \
    --elements Fe Ni --d-nn 2.538 --name feni_fcc2538_ms --rpath 5.3 --nleg 4
python -m chimera.run ... --tables feni_fcc2538 --ms-tables feni_fcc2538_ms
```

The first call writes the nearest-neighbour single-scattering paths (one
FEFF run per absorber, every absorber-scatterer pair present); the second
the multiple-scattering set. `--d-nn` is the nearest-neighbour distance of
the reference cell. Without a multiple-scattering set the model is
single-scattering, and the fit range should stop before the first
collinear shell (`FIT_RMAX_NOMS`, 4.2 A for fcc near 3.6 A).

`feff10_tables.py --from <FEFF8L set> --tag fsr` rebuilds a set with
FEFF10's modules (binaries from `$FEFF10_BIN` or the PATH), changing only
the physics cards: core hole (`fsr`, `rpa`, `none`) and loss model
(`_mpse`).

## Using it from Python

```python
from chimera import dataio as io, calibrate as C, model as M, fitrun as F
from chimera.system import AlloySystem

alloy = AlloySystem(elements=("Fe", "Ni"), lattice="fcc", a0_ref=3.59,
                    tables_name="native", ms_name="native")
spectra, report = io.screen("path/to/data", elements=alloy.elements)
P = spectra.series[0]                           # the first series point
R_data = spectra[(P, "Fe")]["R"]                # the common R grid

cal, S = C.calibrate(spectra, R_data, system=alloy) # E0/edge, a0, sigma^2, dr,
                                                    # C3, MS (+ window and k weight
                                                    # when only |chi(R)| is available)
c    = cal[P]                                   # calibration for that point
S    = M.Spectrum.from_cal(c)
idx  = M.match_grid(S.R, R_data)
mask = C.fit_mask(R_data, c["rmin"], c["rmax"])

data = io.fit_input(spectra, P)                 # a ChiK, or the |chi(R)| median
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
`chimera.system.set_default(alloy)` names the alloy once for a session, after
which calls that are not given a system use it.

`F.systematic_band(...)` re-fits under each admissible calibration variant and
returns the spread — the part of the uncertainty the data cannot resolve.
`F.broadened_partial_rdf(...)` and `F.shell_table(...)` turn a fitted
configuration into `g_AB(r)` and per-shell coordination numbers.

## Opt-in extensions

None of these changes a default.

- **Exact multiple scattering** (`chimera/mslibrary.py`). Every 3- and 4-leg
  path falls into one of a few shapes (25 for fcc to 5.3 A).
  `mslibrary.build(<FEFF10 MS set>)` has FEFF10 compute each shape with every
  assignment of species to its sites, on the potentials of an existing run.
  A library name passed to `--ms-tables` gives the exact random-alloy average
  for calibration, in place of one random cluster's average.
  `--ms-library <name>` (or `Spectrum.attach_ms_library`) makes the sampler
  sum multiple scattering over each configuration's own paths, updated
  exactly at every swap.
- **Energy-informed sampling.** `--energy-prior <epi.json> --anneal-T <K>`
  (or `fit_one(energy_prior=(V, T))`) weighs every sampler move by the change
  in the alloy's pair-interaction energy at the temperature its order froze
  in, as well as by the data. `chimera.physics.pair_interactions` fits such
  interactions to relaxed energies from an embedded-atom potential (a LAMMPS
  `eam/alloy` file you supply; fcc only). The prior is only as good as the
  potential's chemistry; prefer interactions from a DFT-quality source.
- **Disorder from molecular dynamics.** `ShellModel.set_physics(sigma2, dr, c3)`
  takes the bond-length spread, mean bond length and third cumulant per shell
  and pair type from `chimera.physics.pair_statistics`, in place of the
  fitted values.
- **Synthetic data.** `exafs_gpu/abinitio/synth.py` builds realistic
  synthetic multi-edge spectra from atomistic snapshots (Monte Carlo order,
  Langevin dynamics, exact single and multiple scattering on the distorted
  geometry) that share none of the fit's approximations: a known-answer test
  for a measurement's analysis before it is run on the data.

## Layout

```
chimera/
  run.py           the command-line driver: python -m chimera.run
  dataio.py        loading (chi(k) or |chi(R)| files), quality screening,
                   per-position grouping, the series and its unit
  chik.py          chi(k) data: the shared transform to complex chi(R),
                   k-weight channels, R-space noise
  model.py         configuration -> chi(k) -> chi(R); cheap calibration updates
  shellmodel.py    the same model in pair-count space (CPU, exact for swap-only
                   fits): calibration objective, Cramer-Rao bounds on alpha
                   (white, or with a full residual covariance)
  system.py        the alloy: elements, composition, lattice, reference cell,
                   table names; the optional session default
  multiscat.py     multiple-scattering paths (3- and 4-leg), from a table
                   directory or handed over in memory by the native engine
  mslibrary.py     exact multiple scattering: species-resolved path library
                   (built with FEFF10), random-alloy expectation, and the
                   sampler's per-configuration path counts
  physics.py       molecular-dynamics pair statistics and pair interactions
                   from an interatomic potential (opt-in inputs)
  multift.py       per-edge Fourier transform (each edge has its own E0)
  calibrate.py     global search for E0, k window and weight, a0, sigma^2
                   scales, bond offsets dr, C3, MS amplitude; equation of state
  fitrun.py        one spectrum: refine lattice, sample, bounds, systematic band
  figures.py       fit comparison, partial RDFs, alpha by shell, series summary
exafs_gpu/
  abinitio/        native scattering engine: atom.py, potential.py, scf.py,
                   selfenergy.py, losses.py, phaseshift.py, sphere.py,
                   pathsum.py, paths.py, cluster.py, tables.py; provider.py
                   computes the paths in-process for a fit ("native",
                   cached); synth.py builds synthetic data
  scattering.py    scattering tables; feffNNNN.dat reader and the shared
                   convention conversion
  lattice.py       supercells, neighbour lists, force constants, Warren-Cowley
  forward.py       GPU forward model and analytic adjoint
  fourier.py       k -> R transform, windows, adjoint
  kernels.py       CUDA source (NVRTC via CuPy)
  eam.py           embedded-atom potential on the GPU: energies, forces,
                   FIRE relaxation, Langevin dynamics
  feff10.py        FEFF10 input writer and module runner
methods/
  method_a_rmc.py  parallel-tempering RMC sampler
export_athena.py   Athena project files -> chi(k) in the data layout (needs
                   xraylarch)
feff8l_tables.py   FEFF8L tables for an alloy on sc, bcc, fcc or hcp (needs
                   xraylarch)
native_tables.py   the same tables from exafs_gpu/abinitio, no FEFF needed
feff10_tables.py   a FEFF8L table set rebuilt with FEFF10 (core hole, exchange,
                   loss-model options)
tools/
  promote_to_main.py   builds main from development (see Branches)
```

## Branches

Work is committed to `development`, which also holds the tests and the
dataset-specific analysis scripts. `main` is the same history without the
paths listed in `tools/devonly_paths.txt`; after committing to
`development`, `python tools/promote_to_main.py` appends the new commits to
`main` (it never touches the working tree, and refuses to drop commits that
exist only on `main`).

## Scope and limits

* **Lattice.** `CrystalSupercell` handles `sc`, `bcc`, `fcc` and `hcp`,
  including the non-orthogonal hcp cell, and `Spectrum`, the table
  generators and the native engine take the lattice from the `AlloySystem`.
* **Species count.** Any number of species: the CUDA source takes `NSP` as a
  compile-time define, and the Python side reads it from the system. The
  mean-zero per-edge E0 shift has `nsp - 1` free entries, the bond-offset
  matrix `nsp(nsp+1)/2 - 1`, and there are `nsp(nsp-1)/2` order directions
  per shell.
* **Every element is an edge.** Each species must have been measured as an
  absorber: the model has one edge per element of the system.
* **Site equivalence.** Every basis site must have the same coordination: the
  neighbour-list machinery and the MC proposal kernels assume a uniform
  coordination number.
* **Composition** is conserved by construction: moves are species swaps, so a
  fit cannot drift away from the composition you set.
* **Series index.** Series points are integers in the folder names; a
  non-integer series (2.5 GPa) needs its own integer labels.

## Conventions worth knowing

* Tables are on the **internal momentum**, not k relative to E0.
  `ScatteringTables.e0_offset_ev` records where k = 0 sits relative to the
  Fermi level (typically ~10 eV); expect a fitted E0 shift of about that size
  against data referenced to the edge. That gap is not a constant: with the
  self-energy it falls by several eV over the fit range. `--k-axis table`
  (the system's `k_axis`) maps the data's k through the tables' own `k_of_p`,
  so the fitted dE0 is only the offset between the experimenter's E0 and the
  calculation's Fermi level; the default `constant` keeps one gap per edge.
  The mode is written into every calibration and fit record and restored on
  reload.
* Warren-Cowley sign: **positive = like atoms avoid, negative = like atoms
  cluster**.
* Moves are species swaps. `swap_frac=1.0` always *attempts* a swap, but when
  all 8 candidate partners drawn share the atom's species — probability
  (1/n)^8 for n equiatomic species — the proposal falls through to a small
  displacement.
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
  `calibrate` does not search it, and a0 may float with E0 (the phase
  separates them). From `|chi(R)|` alone a0 is anchored at the reference
  point (`--a0-anchor`, default `--a0-ref`), which must be right for that
  point's conditions.
* a0 is free at every series point; the shared per-edge E0 - its common mode
  included - is fitted in the joint calibration stage together with every
  point's a0 and sigma^2, with a mean-zero per-edge E0 shift per point on
  top. The common mode is one number for the whole series, not one per
  point, so that the points do not each walk their own stretch of the a0/E0
  ridge.
* With atoms on ideal sites, chi depends on the configuration only through the
  pair counts per shell, `N[absorber, scatterer, shell]`. `shellmodel` uses
  that: the calibration objective and the alpha bounds run on the CPU in
  milliseconds, and agree with the GPU pair sum to 1e-15.
* `dr[a, b]` is the nearest-neighbour bond-length offset of pair type ab
  (Angstrom, symmetric, composition-weighted mean zero so a0 keeps the mean);
  `C3` is the first-shell third cumulant (Angstrom^3), entering the phase as
  `-4/3 k^3 C3`. Both are zero unless calibrated.
* For a series in GPa, `equation_of_state` in calibration.json and
  summary.json gives V/V0, secant bulk moduli and, when the series includes
  0 GPa, a second-order Birch-Murnaghan B0.
* The sampler profiles the per-edge amplitude inside chi^2 (the same closed
  form the R-factor uses) and draws posterior samples from the rung at
  beta = 1/2, which is the likelihood temperature for its chi^2 definition.
