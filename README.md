# torchebsd

Differentiable, GPU-accelerated dynamical EBSD master-pattern simulation in
PyTorch, with a composable object-oriented API inspired by
[abTEM](https://github.com/abTEM/abTEM).

A master pattern is the backscattered-electron intensity as a function of
crystal direction, stored on the square-Lambert grid. torchebsd builds it from
first principles -- crystal, structure factors with absorption, reflections,
Bloch-wave dynamical matrix -- with every intermediate stage inspectable and
the whole chain refinable. The physics is ported from
[EMsoft](https://github.com/EMsoft-org/EMsoftOO) (Callahan & De Graef 2013),
which remains the validation oracle. See [DESIGN.md](DESIGN.md) for the
architecture and the roadmap; the Bloch-wave solve itself is the next
milestone and is currently stubbed behind a clean backend interface.

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
    crystal, microscope, npx=500, dmin=0.05, bethe=te.BetheParameters()
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
| Monte Carlo depth statistics reader (EMsoft HDF5) | implemented, tested |
| Detector direction cosines (kikuchipy parity), master-pattern projection | implemented, tested |
| Display layer (anyplotlib) and setup widget (anywidget) | implemented, tested |
| Kinematical backend (band geometry; validation placeholder) | implemented, tested |
| Bloch-wave backend (eigendecomposition + depth integration) | stubbed; DESIGN.md M2 |
| Energy-resolved master patterns, symmetry-reduced sampling | roadmap; DESIGN.md M3 |
| PRISM-style scattering-matrix acceleration | designed extension point; DESIGN.md M5 |

## Validation

EMsoft is the oracle. The script compares a simulated master pattern with
EMsoft references (Ni from kikuchipy's bundled data, Si from a local EMsoft
run) by per-pixel Pearson correlation of the Lambert maps:

```
uv run python scripts/validate_master_pattern.py --material ni --device cuda
uv run python scripts/validate_master_pattern.py --material si --device cuda
```

Current results with the kinematical placeholder backend: Ni r = 0.64,
Si r = 0.47 -- band positions align (frame and Lambert conventions are
correct); full parity (r -> 1.0) is the acceptance gate for the Bloch-wave
backend.

## Development

```
uv run pytest
uv run ruff check
```
