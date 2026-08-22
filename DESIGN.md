# torchebsd design

Differentiable, GPU-accelerated dynamical EBSD master-pattern simulation in
PyTorch. This document records the architecture, the conventions, the
dynamical algorithm as ported from EMsoft, the performance and display plans,
the validation strategy, and the roadmap. It is the reference for every
milestone that follows.

## 1. Goals and non-goals

Goals:

- Simulate EBSD master patterns dynamically (Bloch waves, Callahan &
  De Graef 2013) with torch tensors, batched over beam directions and
  energies, on GPU, in FP32 where accuracy allows.
- An object-oriented, composable API in the style of abTEM: small objects,
  one obvious chain, every stage inspectable (`.show()`) and refinable.
- Differentiability along the geometry path (orientation, detector, and
  eventually structure parameters), so the simulation can sit inside
  optimization loops (HR-EBSD, pattern matching, structure refinement).
- Bit-comparable physics with EMsoft wherever a direct port is possible;
  EMsoft output is the validation oracle, never a runtime dependency.

Non-goals (for this package): pattern indexing, Monte Carlo electron
transport (EMsoft's MC output is consumed, not reproduced), TKD/ECP variants
until the EBSD chain is validated.

## 2. Architecture

The simulation chain mirrors abTEM's `Atoms -> Potential -> PlaneWave ->
multislice -> Detector -> Measurement` composition:

| abTEM               | torchebsd                    | Responsibility                                                         |
|---------------------|------------------------------|------------------------------------------------------------------------|
| `Atoms`             | `Crystal`                    | lattice, space group, asymmetric unit, metric tensors, orbits          |
| `Potential`         | `StructureFactors`           | complex U_g with absorption, V_0, wavelength (stage, cached)           |
| `PlaneWave`/`Probe` | `Microscope` + `KVectorGrid` | beam energy, geometry; incident directions on the Lambert grid         |
| `multislice()`      | `MasterPatternBackend`       | intensity per beam direction (Bloch / kinematical / scattering matrix) |
| measurement         | `MasterPattern`              | Lambert-stored sphere; `.show()`, `.sample()`, `.project()`, `.save()` |
| `Detector`          | `EBSDDetector`               | pixel grid, projection center, tilts; direction cosines                |

```
Crystal --+
          +--> MasterPatternSimulator --> MasterPattern --> EBSDDetector.project
Microscope+          |
                     +-- structure_factors()  : StructureFactors   (stage, .show)
                     +-- reflections(k)       : ReflectionList     (stage, .show)
                     +-- dynamical_matrix(k)  : DynamicalMatrix    (stage, .show)
                     +-- compute(device)      : MasterPattern      (result, .show)
                     +-- refine(**changes)    : new simulator, caches carried over
```

Module layout (flat, one concern per module):

| Module                      | Contents                                                                       |
|-----------------------------|--------------------------------------------------------------------------------|
| `crystal.py`                | `AtomSite`, `Crystal` (diffpy space-group operators; origin choice 2 handling) |
| `microscope.py`             | `Microscope` (energy, sample tilt; relativistic quantities)                    |
| `bethe.py`                  | `BetheParameters` (EMsoft defaults 8 / 50 / 100 / 1.0)                         |
| `scattering.py`             | Weickenmeier-Kohl `FSCATT` port (numpy float64)                                |
| `data/scattering_tables.py` | WK coefficient tables, generated from the Fortran source                       |
| `structure_factors.py`      | `compute_structure_factors`, `StructureFactors` (U_g lookup box)               |
| `lambert.py`                | square-Lambert forward/inverse (torch), master-pattern grid                    |
| `kvectors.py`               | incident-direction sampling (`KVectorGrid`)                                    |
| `reflections.py`            | `ReflectionList`, excitation errors, Bethe partition                           |
| `dynamical.py`              | `build_dynamical_matrix`, `DynamicalMatrix`, backend seam + backends           |
| `montecarlo.py`             | `DepthDistribution` (EMsoft MC HDF5 reader, fallback model)                    |
| `simulator.py`              | `MasterPatternSimulator` (composition root, stages, refine)                    |
| `masterpattern.py`          | `MasterPattern` (sample/project/show/save)                                     |
| `detector.py`               | `EBSDDetector` (kikuchipy-parity direction cosines)                            |
| `rotation.py`               | orix-order quaternion rotations (torch)                                        |
| `display.py`                | the anyplotlib display layer (all `.show()` implementations)                   |
| `widgets.py`                | `SetupWidget` (anywidget geometry schematic)                                   |

Design rules:

- Stages are plain data objects; the simulator owns caching and wiring.
- Setup-time math (structure factors, symmetry) is numpy float64; the
  per-direction hot path is torch and batched. The boundary is explicit
  (`StructureFactors.as_torch_lut`).
- `refine()` returns a new simulator rather than mutating, and carries over
  every cache that is still consistent (raising `npx` or changing Bethe
  thresholds never recomputes U_g; changing `dmin` or the energy does).
- One display module; no plotting code anywhere else.

## 3. Units and conventions

- Lengths nm, reciprocal lengths nm^-1, energies eV at the API surface
  (`Microscope(energy=20e3)`), keV internally; U_g in nm^-2; V_g in volts.
- Cartesian crystal frame: EMsoft's `mod_crystallography.f90` convention,
  x parallel to a, z parallel to c*; the direct structure matrix `dsm` and
  reciprocal `rsm = inv(dsm)^T` are element-for-element EMsoft's (verified
  against the source). All beam directions and g-vectors used in the
  dynamical computation live in this frame.
- Square Lambert storage: array `[row, col]` with row = Lambert Y, col =
  Lambert X, both spanning `[-sqrt(pi/2), +sqrt(pi/2)]`; hemisphere index
  0 = north (z >= 0), 1 = south. Interpolation is bilinear
  `grid_sample(align_corners=True)`. These are exactly kikuchipy's and
  `nn_hrebsd.sim`'s conventions, so a torchebsd master pattern feeds the
  validated external projector unchanged.
- Quaternions are orix order (a, b, c, d), scalar first.
- Projection centers use the Bruker convention, detector geometry follows
  kikuchipy (`tilt`, `azimuthal`, `twist`, gnomonic bounds per Britton et
  al. 2016); direction cosines are tested equal to kikuchipy's to 1e-10.
- Space groups: diffpy tables (origin choice 1); for the 24 two-origin
  groups, `Crystal` defaults to origin choice 2 by conjugating the operators
  onto the nearest inversion center (verified for the cubic groups; other
  two-origin groups should be checked against the International Tables
  before use -- roadmap M3 pins the full ITA shift table).

## 4. The dynamical algorithm (ported from EMsoft)

Per energy bin E and incident direction k (EMsoft
`program_mods/mod_EBSDmaster.f90`; state of each step in brackets):

1. **Structure factors** [implemented]. For every hkl in the lookup box
   (out to twice the dmin sphere): Weickenmeier-Kohl complex atomic
   amplitudes with phonon + core absorption (`mod_others.f90 FSCATT`,
   ported in `scattering.py` with EMsoft's exact single-precision
   constants), summed over the orbit with phases `exp(-2 pi i g.r)` and
   scaled by `pref = 0.04787801 / (4 pi V_cell)`, `U = preg V`,
   `preg = 0.664840340614319` (`mod_diffraction.f90 CalcUcg`, WK branch).
   V_0, U'_0, and the refraction-corrected relativistic wavelength
   (`CalcWaveLength`) come from the g = 0 entry.
2. **Incident directions** [implemented, unreduced]. Rosca-Lambert square
   grid, both hemispheres, unit vectors in the Cartesian crystal frame
   (`mod_kvectors.f90 Calckvectors/AddkVector`; `k = k_hat / lambda`,
   `k_n = 1 / lambda`). EMsoft samples only the point-group fundamental
   sector and replicates by symmetry (`SamplingType`,
   `Apply3DPGSymmetry`); torchebsd v0 samples the full sphere, which is
   redundant but cannot mis-symmetrize -- the reduction is a pure
   optimization scheduled for M3.
3. **Reflection list** [implemented]. All centering-allowed hkl with
   d >= dmin and `|s_g| / (lambda |U_g|) <= c3`; forbidden reflections kept
   within `|s_g| <= sg_double` as double-diffraction channels
   (`Initialize_ReflectionList`). Excitation errors from the Ewald-sphere
   relation `s_g = -g.(2k + g) / (2 (k + g).n)` with the foil normal along
   the beam (`CalcsgDouble`).
4. **Bethe partition** [implemented]. For each candidate g,
   `m = min_h |s_g| / (lambda |U_{g-h}|)` over all candidates h (forbidden
   differences excluded); strong when `m <= c1`, weak when `c1 < m <= c2`,
   dropped otherwise (`Apply_BethePotentials`; defaults c1 = 8, c2 = 50).
5. **Dynamical matrix** [implemented]. Bloch form, nm^-2
   (`mod_gvectors.f90 GetDynMat`):
   `A_gh = U_{g-h} - (lambda/2) sum_w U_{g-w} U_{w-h} / s_w` off diagonal,
   `A_gg = 2 s_g / lambda - (lambda/2) sum_w |U_{g-w}|^2 / s_w + i U'_0`.
6. **Bloch solve and depth integration** [scheduled, M2 -- the stub raises
   with a pointer here]. `CalcLgh`: eigendecompose A (LAPACK ZGEEV; torch:
   `torch.linalg.eig`), scale eigenvalues `W -> W / (2 k_n)`, invert the
   eigenvector matrix C, take `alpha = C^-1[:, 0]` (incident-beam column),
   and build
   `I_jk = conj(alpha_j) alpha_k sum_z lambda_E(z) exp(-q_jk z) dz / t`
   with `q_jk = 2 pi [ (Im W_j + Im W_k) + i (Re W_j - Re W_k) ]`
   (real part folded positive), then `L_gh = conj(C) I C^T`.
   `lambda_E(z)` is the Monte Carlo exit-depth fraction times
   `exp(+2 pi z / x_gp)` which cancels the normal absorption already in
   the eigenvalues [MC reader implemented]; the depth limit is the 99 %
   histogram depth (`izzmax`) [implemented].
7. **S_gh weighting** [scheduled, M2]. Per asymmetric-unit site:
   `S_gh = sum_atoms Z^2 exp(-B s^2) exp(+2 pi i (h - g).r)`
   (`preCalcSgh`), the master-pattern intensity is
   `I(k) = Re sum_gh L_gh S_gh / n_atoms`.
8. **Accumulation** [implemented for the unreduced grid]. Intensities land
   at their Lambert pixel on the correct hemisphere; with symmetry
   reduction (M3) each computed pixel is written to all equivalents
   (`Apply3DPGSymmetry`), and trigonal/hexagonal systems use the hexagonal
   Lambert grid with a final square resampling.
9. **Energy loop** [scheduled, M3]. Steps 2-8 per energy bin (wavelength,
   reflection lists, and matrices change with E); the master pattern
   stores one map per bin, shape `(nE, 2, n, n)`.

## 5. Backends and performance

All intensity models sit behind one seam
(`dynamical.MasterPatternBackend`): `prepare(SimulationInputs)` once, then
`intensities(directions (K, 3)) -> (K,)` in chunks. The simulator owns
gridding, chunking, and assembly; backends own physics.

- `KinematicalBackend` [implemented]: each allowed reflection paints a band
  of half-width theta_B weighted by |U_g|^2. One (K, G) matmul; correct
  band geometry, placeholder profiles. It exists to validate conventions
  end to end and to keep the pipeline runnable.
- `BlochWaveBackend` (`bloch`) [M2, implemented]: the exact reference. Per
  direction: classify (Bethe), build the dynamical matrix, eigendecompose and
  depth-integrate (`calc_lgh`, with a Horner depth sum), weight with `S_gh`.
  float64, one direction at a time -- the correctness oracle. The non-Hermitian
  eigendecomposition neither batches nor GPU-accelerates (cuSOLVER has no
  batched general-eig kernel; `torch.linalg.eig` on cuda ~= cpu), so this path
  is CPU-bound and exists to validate the fast backend, not for production.
- `BlochPrismBackend` (`bloch_prism`) [implemented]: the production GPU path.
  Replaces the eigendecomposition with the physically equivalent scattering
  matrix `exp(2 pi i A/(2 kn) z)` (batched `torch.linalg.matrix_exp`, pure
  matmul) followed by a depth recurrence -- no eig. The whole pipeline is
  batched: excitation errors, a candidate-restricted Bethe partition, the
  dynamical-matrix build (weak folding), the propagation, and the `S_gh`
  reduction, bucketed by strong-beam count so each batch is a uniform
  `[B, S, S]`. Reproduces the reference to ~1e-13 (fp64) / ~1e-4 (fp32).
  Measured (Pascal TITAN X, npx-500 Si, ~2e6 directions): **~10 s** in fp32 /
  ~16 s in fp64 at **r = 1.0000** vs EMsoft, with the point-group fold (M3)
  cutting it from ~181 s -- the run that makes a native-resolution parity check
  practical (the eig reference would need ~1e6 eigendecompositions). A further
  PRISM-style coarse-grid interpolation of the scattering matrix (M5) remains an
  optional accuracy/speed knob.
- Symmetry reduction [M3]: sample only the fundamental sector
  (EMsoft `SamplingType` tables) and replicate -- an order-of-magnitude
  saving for high-symmetry crystals, orthogonal to the backend choice.

## 6. Display and widgets

One display contract: every stage object exposes `.show(...)` returning an
`anyplotlib.Figure` (WebGL/HTML; interactive in Jupyter via
`_repr_html_`, exportable with `save_html`/`to_html`). All rendering lives
in `display.py`:

- `StructureFactors.show()` -- bar chart of |U_g| per family (absences
  visible), V_0 in the title.
- `ReflectionList.show()` -- bars colored strong / weak / ignored.
- `DynamicalMatrix.show()` -- log10 |A| heat map.
- `MasterPattern.show(projection=...)` -- `"lambert"` and
  `"stereographic"` hemisphere pairs (`imshow`), `"sphere"` as an
  interactive 3D surface with the pattern as an equirectangular texture
  (`plot_surface` + `set_texture(cull_backfaces=True)`).

anyplotlib is imported lazily so headless compute never needs it.

`widgets.SetupWidget` (anywidget + traitlets) draws the experiment geometry
-- beam, tilted sample, detector plane, projection center -- as a live SVG
schematic with sliders; traitlets sync both ways and `to_microscope()` /
`to_detector()` read the adjusted configuration back into API objects.
Entry points: `sim.show_setup(detector=...)` and `detector.show()`.

## 7. Differentiability

Differentiable today (pure torch): Lambert mappings, master-pattern
interpolation, detector direction application, quaternion rotation -- i.e.
gradients flow from projected patterns to orientation and master-pattern
values. Scattering factors and symmetry are setup-time numpy by design;
lifting the U_g computation to torch (gradients into lattice parameters and
Debye-Waller factors, and through the Bloch solve via `torch.linalg.eig`'s
autograd) is roadmap M4. The backend seam keeps this incremental: the
Bloch backend is written in torch from the start, so M4 is mostly about
replacing the numpy structure-factor input with a torch one.

## 8. Validation

EMsoft is the ground-truth oracle -- for *validation only*; torchebsd computes
the master pattern itself and does not call EMsoft at runtime:

- Ni, 20 keV, npx = 200: `kikuchipy.data.nickel_ebsd_master_pattern_small`
  (bundled with kikuchipy -- Ni validates with no external files).
- Si, 20 keV, npx = 500, 11 energy bins: a local EMsoft `Si_MC.h5` run (path
  via `--mc` / `TORCHEBSD_SI_MC`; the file is not redistributed). It also
  provides the reference-run Monte Carlo depth statistics, crystal data, and
  Bethe parameters. When no MC file is supplied the depth integration falls back
  to torchebsd's own `single_energy` model, so simulation stays EMsoft-free; a
  native Monte Carlo that makes that path physically accurate is the next
  milestone (M-MC below).

`scripts/validate_master_pattern.py` builds the same crystal, simulates,
resamples the reference onto the simulated grid through the shared
(kikuchipy-validated) interpolation, and reports per-pixel Pearson
correlation per hemisphere -- scale invariant, so normalization differences
do not matter; frame or symmetry errors collapse it toward zero.

M2 status (backend done; Ni parity open). The exact `bloch` reference and the
batched `bloch_prism` backend both reproduce EMsoft. The residual that initially capped Si near 0.94
was traced not to the scattering factors (torchebsd `U_g` match EMsoft
`CalcUcg` to <0.003%) but to the Bethe partition promoting systematically
absent reflections to strong beams; excluding them (`status[forbidden] =
IGNORED`) makes the strong-beam map pixel-exact against EMsoft. At matched
resolution torchebsd equals EMsoft's own master to r = 1.00000; a full native
npx-500 Si master via `bloch_prism` scores **r = 1.0000** against `Si_MC.h5`
in ~10 s (fp32) on a Pascal TITAN X (numbers from the maintainer's GPU, not
reproduced in CI). The acceptance gate (r >= 0.99 on both references) is met:
**Si r = 1.0000**, and **Ni r = 1.00000** against a fresh, matched-settings
EMsoft master. The old "Ni 0.93" was a settings artifact (Bethe defaults 8/50/100
vs EMsoft 4/8/50, plus the kikuchipy reference's unknown provenance), not the
physics -- confirmed by generating an EMsoft Ni master with matched Bethe/dmin/
depth and correlating to 1.00000. See [BENCHMARKS.md](BENCHMARKS.md) for the full
cross-code audit (parity, speed, crystal-system sweep). The kinematical
placeholder scores Ni 0.64 / Si 0.47 with
identical north/south values -- band positions align, so the crystal frame,
Lambert layout, and hemisphere conventions were right.

The pytest suite (78 tests) covers: metric tensors and EMsoft structure
matrices, orbit expansion including origin choice 2, WK tables verbatim
against the Fortran source, Ei(x) against Abramowitz & Stegun, structure
factor absences (diamond 200/222, FCC mixed parity), V_0 and wavelength
values, Friedel symmetry, Lambert round trips and grid conventions,
excitation errors on the Ewald sphere, Bethe partition behavior, dynamical
matrix entries against the defining formulas, MC reader against the Si
file, detector direction cosines equal to kikuchipy, end-to-end compute /
refine / save / project, and headless display + widget smoke tests.

## 9. Roadmap

- **M1 (this milestone, done)** -- repository, architecture, and the
  verifiable chain: crystal / WK structure factors with absorption /
  Lambert / reflections + Bethe / dynamical matrix / MC reader / detector
  / display + widget; kinematical end-to-end; validation harness green.
- **M2 -- Bloch-wave backend [done].** `CalcLgh` + `S_gh` ported: the
  `bloch` float64 reference (one direction at a time) and the batched GPU
  `bloch_prism` backend. The GPU path uses the scattering matrix
  (`matrix_exp`), not batched eig -- non-Hermitian eig does not GPU-batch, and
  the depth integration, not the eig, was the cost. Acceptance: Si r = 1.0000
  (meets the >= 0.99 gate); Ni r = 0.93 (below the gate, open item).
- **M-MC -- native Monte Carlo [done].** An EMsoft-independent electron
  transport Monte Carlo for the exit depth/energy statistics: screened
  Rutherford elastic scattering + a CSDA (Joy-Luo/Bethe) stopping power,
  electrons into a tilted bulk sample, histogrammed to `accum_z` and reduced to
  `lambda_E(z)`. Removes the last EMsoft *runtime* dependency; a plain
  `compute()` now builds it automatically (`from_simulation`), retiring the
  coarse `single_energy` stand-in, and EMsoft `accum_z` remains the validation
  oracle only. The hard-mode forward reproduces EMsoft's `accum_z` per energy
  bin to correlation > 0.995 for Si and Ni with matching yields (M-MC.4); a
  differentiable soft mode (`soft_depth_histogram`: reparameterized draws + soft
  gates + bilinear binning) gives gradients to the material parameters, matching
  finite differences to ~2 % (M-MC.5/.6). Full scope, task breakdown, and
  validation numbers in [MC_DESIGN.md](MC_DESIGN.md). Optional later: Mott
  cross-sections, energy-bin master loop (M3).
- **M3 -- energy and symmetry [done].** Symmetry-reduced k sampling:
  `reduce_to_fundamental_sector` folds the grid onto the point-group fundamental
  sector via orix (`in_fundamental_sector`); symmetry-equivalent grid points
  reduce to a bit-identical representative, so the fold is exact (~16x fewer
  evaluations for cubic) and matches the fully-sampled master bit-for-bit.
  Energy loop: `compute(energy_resolved=True)` produces one Lambert map per MC
  energy bin (structure factors and depth profile per energy) -> an
  `(nE, 2, n, n)` master; the top bin is bit-identical to the single-energy
  path, and Si bins correlate ~0.93 with `Si_MC.h5`'s `mLPNH` at coarse npx
  (the top bin reaches r = 1.0 at native resolution, M2). Remaining (optional):
  the hexagonal Lambert grid for trigonal/hexagonal systems.
- **M4 -- differentiability [done].** Autograd runs end to end from a crystal
  parameter to intensity: the `bloch_prism` path is torch-native (`matrix_exp` +
  depth recurrence + `Sgh`), so gradients flow through the non-Hermitian complex
  solve back to `U_g`; `differentiable_ucg_lut` assembles
  `U_g = pre * sum_site occ * f_site(g) * sum_r exp(-2 pi i g.r)` in torch; and
  `scattering_torch.wk_scattering_factor_torch` ports the full WK amplitude
  (elastic `WEKO`, phonon/core absorption via the exponential-integral integrals
  `RI1`/`RI2`) to torch. So `U_g`, and thus the master pattern, is differentiable
  w.r.t. atom **positions**, **occupancies**, and the **Debye-Waller** factor --
  every stage forward bit-exact vs the numpy path (amplitude to ~1e-8) and every
  gradient matching finite differences (`d(intensity)/dB` to ~1e-5). Remaining
  (optional): **lattice** gradients, which additionally need the metric/`g`
  differentiable in torch; and an FP32/FP64 accuracy audit + dtype policy.
- **M5 -- scattering-matrix backend [core done as `bloch_prism`].** The exact
  per-direction scattering matrix (`matrix_exp`) is implemented and is the
  production backend. Remaining optional extension: PRISM-style coarse-grid
  precomputation + interpolation of the scattering matrix across nearby
  directions, with a coarse-grid-spacing accuracy/speed study.
- **M6 -- interoperability.** EMsoft-format HDF5 export, kikuchipy signal
  adapters, documented handoff to the `nn_hrebsd` projector.
