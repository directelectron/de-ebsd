# de-ebsd

Differentiable, GPU-accelerated dynamical EBSD master-pattern simulation in
PyTorch, with a composable object-oriented API inspired by
[abTEM](https://github.com/abTEM/abTEM).

A master pattern is the backscattered-electron intensity as a function of
crystal direction, stored on the square-Lambert grid. de-ebsd builds it from
first principles -- crystal, structure factors with absorption, reflections,
Bloch-wave dynamical matrix, depth-integrated dynamical solve -- with every
intermediate stage inspectable and the whole chain refinable. The physics is
ported from [EMsoft](https://github.com/EMsoft-org/EMsoftOO) (Callahan &
De Graef 2013), which remains the validation oracle (for validation only --
de-ebsd does not call EMsoft at runtime). See [DESIGN.md](DESIGN.md) for the
architecture and the roadmap. The Bloch-wave solve is implemented: an exact
float64 reference (`bloch`) and a batched GPU scattering-matrix backend
(`bloch_prism`) that reproduces it. 

## Installation

Windows / Linux with uv (Python >= 3.11; torch comes from the CUDA 12.4
index, configured in `pyproject.toml`):

```
git clone <this repository> de-ebsd
cd de-ebsd
uv sync
```

The distribution is named `de-ebsd`; the Python package is `de_ebsd`.
When upgrading from `torchebsd`, update imports to `de_ebsd` and run `uv sync`
again. The validation scripts now use `DE_EBSD_SI_MC` and `DE_EBSD_NI_MC`
instead of the `TORCHEBSD_*` environment variables. Restart existing Jupyter
kernels after upgrading. Saved master-pattern HDF5 files are unchanged;
Python pickles referring to the old module path need migration in the old
environment. No legacy import alias is installed.

## Quickstart

```python
import de_ebsd as de

crystal    = de.Crystal.from_parameters(
    a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
)
microscope = de.Microscope(energy=20e3, sample_tilt=70.0)
sim        = de.MasterPatternSimulator(
    crystal, microscope, npx=500, dmin=0.05, bethe=de.BetheParameters(),
    backend="bloch_prism",   # the default: batched dynamical solve; "bloch" is
    #                          the one-direction-at-a-time float64 reference
)

# every stage is inspectable (all figures render through anyplotlib):
sim.structure_factors().show()        # |U_g| per family, absences, V_0
sim.reflections(k=(0, 0, 1)).show()   # Bethe strong / weak split for one beam
sim.dynamical_matrix(k=(0, 0, 1)).show()

master = sim.compute(device="cuda")   # -> MasterPattern
master.show(projection="lambert")     # also "stereographic" and "sphere"
sim.crystal.show()                     # real 3-D expanded crystal structure

# refine and recompute (structure factors are reused when still valid):
better = sim.refine(npx=1000, dmin=0.04)

# project to a detector for a batch of orientations (orix Rotation):
detector = de.EBSDDetector(shape=(480, 480), pc=(0.5, 0.5, 0.6))
patterns = master.project(detector, orientation=rotations)

# interactive geometry widget in Jupyter (anywidget):
sim.show_setup(detector=detector)

# phase composition for Jupyter or a SpyDE popover; figures stay native:
phase = sim.crystal.overview(master_pattern=master)
phase                              # displays metadata + real anyplotlib figures
phase.figures["phase_overview"]   # SpyDE mounts one native figure/bridge
sim.microscope.show(detector=detector, crystal=sim.crystal)

# STEM zone-axis exploration: geometric Kikuchi lines + kinematic spots,
# independently toggleable in one projection; no master pattern required:
explorer = sim.crystal.explore(energy_kev=200, dmin=0.1)
explorer                            # live Jupyter controls and native figures

# inspect the incident-direction sampling:
from de_ebsd.diffraction.kvectors import sample_incident_directions
sample_incident_directions(sim.npx).show()
```

## Examples

The [STEM orientation explorer](examples/simulation/stem_orientation_explorer.py)
links a band-line sphere, crystal structure, and a combined lines/spots view.
Its energy setting is in keV and its lattice-spacing cutoff in nm. It provides
geometric band centerlines and relative elastic kinematic spot intensities;
it does not simulate transmission Kikuchi contrast or convergent-beam discs.
Linked recalculation requires a running Python kernel or application bridge.

The [examples folder](examples/) contains executable Sphinx-Gallery tutorials,
grouped by the layer they exercise.

**[Master-pattern simulation](examples/simulation/)** -- the forward chain:

- [CIF to detector](examples/simulation/cif_to_detector.py): silicon structure,
  native Monte Carlo, diffraction intermediates, full 3D master pattern,
  orix orientations, detector images, phase overview, and HDF5 round trip.
- [Energy-resolved simulation](examples/simulation/energy_resolved_simulation.py):
  nickel transport statistics and dynamical patterns at multiple energies.
- [Resolution and backends](examples/simulation/resolution_and_backends.py):
  grid refinement, detector-space differences, and PRISM/Bloch comparison,
  plus the microscope, probe, detector and depth-slice inspectors.
- [Compare with EMsoft](examples/simulation/compare_emsoft.py): use the bundled
  nickel reference's crystal, Bethe settings and Monte Carlo profile, then
  compare both hemispheres, normalized residuals, and detector patterns.

**[Orientation indexing](examples/indexing/)** -- the inverse problem, each on a
synthetic two-grain scan with known orientations:

- [Dictionary indexing](examples/indexing/dictionary_indexing.py): background
  removal, a fundamental-zone template library, a `CrystalMap` with IPF and
  quality maps, and Levenberg-Marquardt refinement to sub-degree accuracy.
- [Spherical indexing](examples/indexing/spherical_indexing.py): the SO(3)
  cross-correlation surface, the bandwidth accuracy/cost trade-off and how it
  fails when truncated too far, then refinement off the peak.

Run scripts with `uv run python examples/simulation/cif_to_detector.py`. They
use small CPU-friendly teaching settings; see
[examples/README.rst](examples/README.rst) for execution, display, and
convergence notes. The existing CIF notebook is preserved.

## Documentation

The full site -- user guide, the executable example gallery, and an API
reference generated from the docstrings -- builds with:

```shell
uv sync --group docs --group notebook
uv run python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html`. The build executes every example, so a
broken example fails the build.

Start at [`docs/user/conventions.rst`](docs/user/conventions.rst) before
interpreting any number: units, the Cartesian crystal frame, square-Lambert
storage, the orientation convention and the space-group origin choice are all
load-bearing, and they are what make de-ebsd's output comparable to EMsoft's
and kikuchipy's. The deeper reasoning lives in [DESIGN.md](DESIGN.md),
[INDEXING_DESIGN.md](INDEXING_DESIGN.md), [MC_DESIGN.md](MC_DESIGN.md) and
[BENCHMARKS.md](BENCHMARKS.md).

## Status

| Component | State |
| --- | --- |
| Crystal, space-group orbits, metric tensors (EMsoft frame conventions) | implemented, tested |
| Weickenmeier-Kohl scattering factors with absorption (EMsoft `FSCATT` port) | implemented, tested |
| Structure factors U_g, mean inner potential, refraction-corrected wavelength | implemented, tested |
| Square-Lambert mappings and master-pattern grid (kikuchipy conventions) | implemented, tested |
| Reflection lists, excitation errors, Bethe strong/weak partition | implemented, tested |
| Dynamical (Bloch) matrix with Bethe potentials (`GetDynMat` port) | implemented, tested |
| Monte Carlo exit-depth statistics: EMsoft HDF5 reader + `single_energy` fallback | implemented, tested |
| Detector direction cosines (kikuchipy parity), master-pattern projection | implemented, tested |
| Native anyplotlib displays, host-neutral phase overview, microscope editor | implemented, tested |
| Bloch-wave backend: eigendecomposition + `CalcLgh` depth integration (`bloch`) | implemented, tested |
| Scattering-matrix (PRISM) GPU backend, exact vs `bloch` (`bloch_prism`) | implemented, tested |
| Native electron Monte Carlo (EMsoft-independent depth/energy stats) | implemented, validated vs EMsoft `accum_z` |
| Symmetry-reduced sampling (point-group fundamental sector, via orix) | implemented, tested |
| Energy-resolved master patterns (per MC energy bin) | implemented, tested |

## Validation

EMsoft is the oracle (validation only). The script compares a simulated master
pattern with EMsoft references by per-pixel Pearson correlation of the Lambert
maps -- scale invariant, so normalization differences do not matter:

```
uv run python scripts/validate_master_pattern.py --material ni --backend bloch_prism
uv run python scripts/validate_master_pattern.py --material si --backend bloch_prism \
    --mc /path/to/Si_MC.h5 --device cuda
```

Ni uses kikuchipy's bundled master pattern and runs with no external files. Si
needs a local EMsoft `Si_MC.h5` reference run (not redistributed here) -- point
`--mc` or `DE_EBSD_SI_MC` at your copy.

Dynamical backend results: **Si r = 1.0000** at native npx-500 against
`Si_MC.h5`, and **Ni r = 1.00000** against a fresh matched-settings EMsoft master
(the "Ni 0.93" was a Bethe-default / reference-provenance artifact, not physics --
see [BENCHMARKS.md](BENCHMARKS.md)).

That npx-500 Si master (~2 million directions) computes in **~10 s** (fp32) /
~16 s (fp64) on a Pascal TITAN X with `bloch_prism`, or ~181 s with the
point-group fold disabled -- an ~11x symmetry speedup at bit-identical output.

## Development

```
uv run pytest
uv run ruff check
```
