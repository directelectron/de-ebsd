# torchebsd

Differentiable, GPU-accelerated dynamical EBSD master-pattern simulation in
PyTorch, with a composable object-oriented API inspired by
[abTEM](https://github.com/abTEM/abTEM).

A master pattern is the backscattered-electron intensity as a function of
crystal direction, stored on the square-Lambert grid. torchebsd builds it from
first principles -- crystal, structure factors with absorption, reflections,
Bloch-wave dynamical matrix, depth-integrated dynamical solve -- with every
intermediate stage inspectable and the whole chain refinable. The physics is
ported from [EMsoft](https://github.com/EMsoft-org/EMsoftOO) (Callahan &
De Graef 2013), which remains the validation oracle (for validation only --
torchebsd does not call EMsoft at runtime). See [DESIGN.md](DESIGN.md) for the
architecture and the roadmap. The Bloch-wave solve is implemented: an exact
float64 reference (`bloch`) and a batched GPU scattering-matrix backend
(`bloch_prism`) that reproduces it. The next milestone is a native electron
Monte Carlo, to drop the last runtime dependency on EMsoft-generated
exit-depth statistics.

## Installation

Windows / Linux with uv (Python >= 3.11; torch comes from the CUDA 12.4
index, configured in `pyproject.toml`):

```
git clone <this repository>
cd torchebsd
uv sync
```

## Quickstart

```python
import torchebsd as te

crystal    = te.Crystal.from_parameters(
    a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
)
microscope = te.Microscope(energy=20e3, sample_tilt=70.0)
sim        = te.MasterPatternSimulator(
    crystal, microscope, npx=500, dmin=0.05, bethe=te.BetheParameters(),
    backend="bloch_prism",   # batched GPU dynamical solve; "bloch" = exact
    #                          float64 reference, "kinematical" = geometry only
)

# every stage is inspectable (all figures render through anyplotlib):
sim.structure_factors().show()        # |U_g| per family, absences, V_0
sim.reflections(k=(0, 0, 1)).show()   # Bethe strong / weak split for one beam
sim.dynamical_matrix(k=(0, 0, 1)).show()

master = sim.compute(device="cuda")   # -> MasterPattern
master.show(projection="lambert")     # also "stereographic" and "sphere"

# refine and recompute (structure factors are reused when still valid):
better = sim.refine(npx=1000, dmin=0.04)

# project to a detector for a batch of orientations (orix Rotation):
detector = te.EBSDDetector(shape=(480, 480), pc=(0.5, 0.5, 0.6))
patterns = master.project(detector, orientation=rotations)

# interactive geometry widget in Jupyter (anywidget):
sim.show_setup(detector=detector)
```

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
| Display layer (anyplotlib) and setup widget (anywidget) | implemented, tested |
| Kinematical backend (band geometry; validation placeholder) | implemented, tested |
| Bloch-wave backend: eigendecomposition + `CalcLgh` depth integration (`bloch`) | implemented, tested |
| Scattering-matrix (PRISM) GPU backend, exact vs `bloch` (`bloch_prism`) | implemented, tested |
| Native electron Monte Carlo (EMsoft-independent depth/energy stats) | roadmap; DESIGN.md M-MC |
| Energy-resolved master patterns, symmetry-reduced sampling | roadmap; DESIGN.md M3 |

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
`--mc` or `TORCHEBSD_SI_MC` at your copy.

Dynamical backend results: **Si r = 1.0000** at native npx-500 against
`Si_MC.h5` (meets the r >= 0.99 acceptance gate); **Ni r = 0.93**, still below
the gate and an open item (see DESIGN.md). The kinematical placeholder
(`--backend kinematical`) scores Ni 0.64 / Si 0.47 -- band positions align
(frame and Lambert conventions correct) but band profiles do not.

## Development

```
uv run pytest
uv run ruff check
```
