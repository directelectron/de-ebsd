# de-ebsd

Differentiable, GPU-accelerated EBSD in PyTorch: dynamical master-pattern
simulation from a crystal structure, native Monte Carlo electron transport,
orientation indexing (dictionary, spherical-harmonic and live), gradient and
high-resolution refinement, pattern-center calibration and HR-EBSD strain. The
API is composable and object oriented, in the manner of
[abTEM](https://github.com/abTEM/abTEM), and every intermediate stage is
inspectable.

A master pattern is the backscattered-electron intensity as a function of
crystal direction, stored on the square-Lambert grid. de-ebsd builds it from
first principles -- crystal, structure factors with absorption, reflections,
Bloch-wave dynamical solve, depth integration over a Monte Carlo exit-depth
profile -- and then runs the inverse problem against measured patterns. The
physics is ported from [EMsoft](https://github.com/EMsoft-org/EMsoftOO)
(Callahan & De Graef 2013), which remains the validation oracle (de-ebsd does
not call EMsoft at runtime); see [NOTICE](NOTICE). Indexing follows EMSphInx
(Lenthe, Singh & De Graef 2019) and emits [orix](https://github.com/pyxem/orix)
objects, so [kikuchipy](https://github.com/pyxem/kikuchipy) and other tools can
use it as a backend.

## Installation

Windows / Linux with uv (Python >= 3.11; torch comes from the CUDA 12.4
index on Windows and the CPU index on Linux, configured in `pyproject.toml`):

```
git clone https://github.com/CSSFrancis/de-ebsd
cd de-ebsd
uv sync
```

The distribution is named `de-ebsd`; the Python package is `de_ebsd`. The
fused CUDA kernels of live indexing need CuPy: `uv sync --extra live`.

## Quickstart: simulation

```python
import de_ebsd as de

crystal    = de.Crystal.from_parameters(
    a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
)
microscope = de.Microscope(energy=20e3, sample_tilt=70.0)
sim        = de.MasterPatternSimulator(
    crystal, microscope, npx=500, dmin=0.05, bethe=de.BetheParameters(),
    # backend="bloch_prism" is the default: the batched dynamical solve.
    # "bloch" is the one-direction-at-a-time float64 reference it reproduces.
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
detector = de.EBSDDetector(shape=(480, 480), pattern_center=(0.5, 0.5, 0.6))
patterns = master.project(detector, orientation=rotations)

# interactive geometry widget in Jupyter (anywidget):
sim.show_setup(detector=detector)

# phase composition for Jupyter or a host application; figures stay native:
phase = sim.crystal.overview(master_pattern=master)
sim.microscope.show(detector=detector, crystal=sim.crystal)

# STEM zone-axis exploration: geometric Kikuchi lines + kinematic spots,
# independently toggleable in one projection; no master pattern required:
explorer = sim.crystal.explore(energy_kev=200, dmin=0.1)
```

The exit-depth statistics the dynamical solve integrates over come from a native
Monte Carlo (`de_ebsd.sim.montecarlo`) unless you pass a `DepthDistribution`,
for example one read from an EMsoft Monte Carlo file.

## Quickstart: indexing

```python
from de_ebsd.index import (
    DictionaryIndexer, SphericalIndexer, refine_orientation, remove_background,
)

patterns = remove_background(raw, static_bg=static_bg, operation="divide")

# Spherical indexing: one SO(3) cross-correlation per pattern, on the GPU.
indexer = SphericalIndexer(master, detector, bandwidth=68, device="cuda")
rotations, quality = indexer.index(patterns)

# Dictionary indexing over a fundamental-zone sample, as an orix CrystalMap:
xmap = DictionaryIndexer(master, detector, resolution=2.5).index_scan(scan, step=0.5)

# Sharpen a coarse index by gradient or Levenberg-Marquardt refinement:
refined, score = refine_orientation(patterns, master, detector, rotations, method="lm")
```

Also in `de_ebsd.index`: `refine_hr` (band-edge refinement of orientation and,
optionally, pattern center), `determine_pattern_center` (one center shared by
many patterns), `hrebsd` (lattice strain and rotation by cross-correlation
against a reference pattern) and `LiveIndexer`, which indexes, refines and
strain-fits a camera's frames where they already are on the GPU, at thousands of
patterns a second.

## Examples

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
- [STEM orientation explorer](examples/simulation/stem_orientation_explorer.py):
  a band-line sphere, crystal structure, and a combined lines/spots view. It
  provides geometric band centerlines and relative elastic kinematic spot
  intensities; it does not simulate transmission Kikuchi contrast or
  convergent-beam discs.

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
convergence notes.

## Documentation

The full site -- user guide, design notes, the executable example gallery, and
an API reference generated from the docstrings -- builds with:

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
and kikuchipy's. The design notes under [`docs/dev/`](docs/dev/) cover the
architecture, the dynamical algorithm, the Monte Carlo, the indexers and the
validation numbers.

## Validation

EMsoft is the oracle (validation only). The script compares a simulated master
pattern with EMsoft references by per-pixel Pearson correlation of the Lambert
maps -- scale invariant, so normalization differences do not matter:

```
uv run python scripts/validate_master_pattern.py --material ni
uv run python scripts/validate_master_pattern.py --material si \
    --mc /path/to/Si_MC.h5 --device cuda
```

Ni uses kikuchipy's bundled master pattern and runs with no external files. Si
needs a local EMsoft `Si_MC.h5` reference run (not redistributed here) -- point
`--mc` or `DE_EBSD_SI_MC` at your copy.

**Si r = 1.0000** at native npx-500 against `Si_MC.h5`, and **Ni r = 1.00000**
against a matched-settings EMsoft master; cubic and tetragonal phases, single-
and multi-component, agree to that precision. The npx-500 Si master (about two
million directions) computes in about 10 s (fp32) on a Pascal TITAN X, roughly
10 times faster than EMsoft on a 24-thread CPU for a full energy-resolved
master. Monoclinic settings and one hexagonal intensity residual are the open
items; see the validation page of the design notes.

## Development

```
uv run pytest
uv run ruff check
uv run ruff format --check
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for the work this
package is built on.
