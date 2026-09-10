# CHIMERA

**CHI-space Multi-Edge Reconstruction of Alloys**

Reconstructs the atomic configuration of a substitutional alloy from
multi-edge EXAFS `|chi(R)|`, and reports Warren-Cowley short-range-order
parameters and partial radial distribution functions with a systematic
uncertainty budget.

## What it does

* Reads `|chi(R)|` per absorption edge and per sample position, screens it for
  quality, and forms position medians.
* Infers the processing parameters the files no longer record — E0 per edge,
  the k window and k weight — together with the lattice constant, the
  Debye-Waller scales, the pair-type bond-length offsets and the first-shell
  third cumulant, by a global fit of the random-alloy model to the data.
* Forward-models `chi(k)` on the GPU from a supercell (500 atoms by default)
  using FEFF8L scattering tables, adds the FEFF8L multiple-scattering paths,
  and transforms with the same window the experiment used.
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

The tables are **not** shipped; generate them for your alloy first. This is
the only step that needs `xraylarch` (which bundles the FEFF8L binaries), and
it runs in a *separate* interpreter that the main environment never imports:

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

## Using it

```python
from chimera import dataio as io, calibrate as C, model as M, fitrun as F

spectra, report = io.screen("path/to/data")     # per-edge |chi(R)|, screened
R_data = spectra[(0, "Cr")]["R"]                # the common R grid

cal, S = C.calibrate(spectra, R_data)           # E0/edge, k window and weight,
                                                # a0, sigma^2 scales, dr, C3, MS

c    = cal[0]                                   # calibration for series point 0
S    = M.Spectrum.from_cal(c)
idx  = M.match_grid(S.R, R_data)
mask = C.fit_mask(R_data, c["rmin"], c["rmax"])

res = F.fit_one(S, R_data, io.average(spectra, 0), idx, mask,
                c["a0"], c["sigma_scale"], seeds=(0, 1))

res["wc"]        # Warren-Cowley alpha, one per unordered pair
res["wc_crlb"]   # Cramer-Rao bound on each, at this fit's residual level
res["r_factor"]
```

`F.systematic_band(...)` re-fits under each admissible calibration variant and
returns the spread — the part of the uncertainty the data cannot resolve.
`F.broadened_partial_rdf(...)` and `F.shell_table(...)` turn a fitted
configuration into `g_AB(r)` and per-shell coordination numbers.

## Layout

```
chimera/
  dataio.py        loading, quality screening, per-position grouping
  model.py         configuration -> |chi(R)|; cheap calibration updates
  shellmodel.py    the same model in pair-count space (CPU, exact for swap-only
                   fits): calibration objective, Cramer-Rao bounds on alpha
  multiscat.py     FEFF8L multiple-scattering paths (3- and 4-leg)
  multift.py       per-edge Fourier transform (each edge has its own E0)
  calibrate.py     global search for E0, k window and weight, a0, sigma^2
                   scales, bond offsets dr, C3, MS amplitude
  fitrun.py        one spectrum: refine lattice, sample, bounds, systematic band
  figures.py       fit comparison, partial RDFs, alpha by shell
exafs_gpu/
  scattering.py    scattering tables; FEFF8L reader and convention conversion
  lattice.py       supercell, neighbour lists, force constants, Warren-Cowley
  forward.py       GPU forward model and analytic adjoint
  fourier.py       k -> R transform, windows, adjoint
  kernels.py       CUDA source (NVRTC via CuPy)
methods/
  method_a_rmc.py  parallel-tempering RMC sampler
feff8l_tables.py   generates data/ for an alloy (needs xraylarch)
```

## Scope and limits

* **Lattice.** `CrystalSupercell` handles `sc`, `bcc`, `fcc` and `hcp`,
  including the non-orthogonal hcp cell. `Spectrum` currently builds an fcc
  supercell, and `feff8l_tables.py` builds an fcc cluster; other lattices need
  those two call sites opened up.
* **Species count.** The CUDA source takes `NSP` as a compile-time define, so
  the kernels serve any number of species. The Python side does not yet follow:
  retargeting to another alloy means editing four values in place —
  `ELEMENTS` and `Z_OF` in `exafs_gpu/scattering.py`, the reference lattice
  constant `self.a0_ref` in `model.py`, and `D_NN_FEFF` in `multiscat.py`
  (the spacing the FEFF paths were computed for). Everything else is already
  an argument: `Spectrum(a0=, tables_name=, ms_name=)`,
  `calibrate(a0_ambient=)`, `ShellModel(a0_ref=)`.
* **Site equivalence.** Every basis site must have the same coordination: the
  neighbour-list machinery and the MC proposal kernels assume a uniform
  coordination number.
* **Input layout.** `dataio.screen` expects `<Element>_<N><tag>/…sample<M>.csv`
  with columns `R, chi_mag`, where `<N>` indexes a series (pressure,
  temperature, composition) and `<M>` a position on the sample. Data arranged
  otherwise can be handed to `calibrate` and `fit_one` directly — they take
  arrays, not paths.
* **Composition** is conserved by construction: moves are species swaps, so a
  fit cannot drift away from the composition you set.

## Conventions worth knowing

* Tables are on the **internal momentum**, not k relative to E0.
  `ScatteringTables.e0_offset_ev` records where k = 0 sits relative to the
  Fermi level (10.3 eV for the CrCoNi tables above); expect a fitted E0 shift
  of about that size against data referenced to the edge.
* Warren-Cowley sign: **positive = like atoms avoid, negative = like atoms
  cluster**. Checked on constructed configurations — a segregated cell gives
  alpha_AA = -1.38, an A-avoiding one +0.43.
* Moves are species swaps. `swap_frac=1.0` always *attempts* a swap, but when
  all 8 candidate partners drawn share the atom's species — probability
  (1/n)^8 for n equiatomic species — the proposal falls through to a small
  displacement. Over a full run that leaves an RMS drift under 3 mA, about 2%
  of sigma.
* Fitting is done against `|chi(R)|` over a windowed R range; the R-factor
  quoted everywhere is `sum (fit - data)^2 / sum data^2` on that window.
* With atoms on ideal sites, chi depends on the configuration only through the
  pair counts per shell, `N[absorber, scatterer, shell]`. `shellmodel` uses
  that: the calibration objective and the alpha bounds run on the CPU in
  milliseconds, and agree with the GPU pair sum to 1e-15.
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
