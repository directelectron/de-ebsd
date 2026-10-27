# de-ebsd

Differentiable, GPU-accelerated EBSD: dynamical master-pattern
simulation from a crystal structure, native Monte Carlo electron transport,
orientation indexing, gradient and
high-resolution refinement, pattern-center calibration and HR-EBSD strain. The
API is composable and object oriented, in the manner of
[abTEM](https://github.com/abTEM/abTEM).

A master pattern is the backscattered-electron intensity as a function of
crystal direction, stored on the square-Lambert grid. de-ebsd builds it from
first principles, crystal, structure factors with absorption, reflections,
Bloch-wave dynamical solve, depth integration over a Monte Carlo exit-depth
profile - and then runs the inverse problem against measured patterns. The
results are validated against  [EMsoft](https://github.com/EMsoft-org/EMsoftOO)
(Callahan & De Graef 2013).


## Installation

Windows / Linux with uv (Python >= 3.11; torch comes from the CUDA 12.4
index on Windows and the CPU index on Linux, configured in `pyproject.toml`):

```
git clone https://github.com/directelectron/de-ebsd
cd de-ebsd
uv sync
```

The distribution is named `de-ebsd`; the Python package is `de_ebsd`. The
fused CUDA kernels of live indexing need CuPy: `uv sync --extra live`.

## Quickstart: simulation

```python
import de_ebsd as de

crystal = de.Crystal.from_parameters(
    a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
)
microscope = de.Microscope(energy=20e3, sample_tilt=70.0)
sim = de.MasterPatternSimulator(
    crystal,
    microscope,
    npx=500,
    dmin=0.05,
    bethe=de.BetheParameters(),
    # backend="bloch_prism" is the default: the batched dynamical solve.
    # "bloch" is the one-direction-at-a-time float64 reference it reproduces.
)

# every stage is inspectable (all figures render through anyplotlib):
sim.structure_factors().show()  # |U_g| per family, absences, V_0
sim.reflections(k=(0, 0, 1)).show()  # Bethe strong / weak split for one beam
sim.dynamical_matrix(k=(0, 0, 1)).show()

master = sim.compute(device="cuda")  # -> MasterPattern
master.show(projection="lambert")  # also "stereographic" and "sphere"
sim.crystal.show()  # real 3-D expanded crystal structure

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

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for the work this
package is built on.
