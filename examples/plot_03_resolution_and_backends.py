"""
Compare grid resolution and dynamical backends
===============================================================================

Compute nickel master patterns at two grid resolutions, project onto an
identical detector, and quantify the interpolation difference. Then compare
the scattering-matrix backend with the Bloch eigensolver at identical settings.

An explicitly prescribed exponential depth distribution keeps this numerical
comparison short and deterministic. It is an approximate depth model, not a
replacement for the native Monte Carlo workflow in the preceding examples.
Two small grids demonstrate a convergence procedure; agreement between two
solvers is not evidence that the physical model is converged.
"""

from time import perf_counter

import anyplotlib as apl
import numpy as np
import torch
from orix.quaternion import Rotation

import torchebsd as te

# %%
# Hold the experiment and depth model fixed
# -----------------------------------------------------------------------------
#
# Use double precision in both solvers for this comparison. Changing only
# ``npx`` preserves the crystal, reflection cutoff and structure factors.

device = "cpu"
crystal = te.Crystal.from_parameters(
    a=0.35236, space_group=225, atoms=[("Ni", (0.0, 0.0, 0.0))], debye_waller=0.0035,
)
microscope = te.Microscope(energy=20e3, sample_tilt=70.0)
depth = te.DepthDistribution.single_energy(
    microscope.energy_kev, thickness=40.0, depth_step=2.0, decay_length=15.0,
)
coarse_sim = te.MasterPatternSimulator(
    crystal, microscope, npx=8, dmin=0.12, monte_carlo=depth, backend="bloch_prism",
)
coarse_sim.structure_factors()
fine_sim = coarse_sim.refine(npx=16)
assert fine_sim.structure_factors() is coarse_sim.structure_factors()

# %%
# Compute both grids
# -----------------------------------------------------------------------------
#
# These timings include preparation and any first-use overhead, so they are
# illustrative rather than a controlled solver benchmark.

start = perf_counter()
coarse = coarse_sim.compute(device=device, dtype=torch.float64, chunk_size=512)
print(f"17 x 17 per hemisphere: {perf_counter() - start:.3f} s")
start = perf_counter()
fine = fine_sim.compute(device=device, dtype=torch.float64, chunk_size=512)
print(f"33 x 33 per hemisphere: {perf_counter() - start:.3f} s")
fine.show(projection="sphere", texture_height=128, n_grid=64)

# %%
# Compare on the same detector grid
# -----------------------------------------------------------------------------
#
# Compare raw intensities so independent image normalization cannot hide an
# amplitude change. Report relative RMS error against the finer projection.
# A larger Lambert grid samples the same diffraction model more finely;
# lowering ``dmin`` is a separate convergence axis that adds reflections.

detector = te.EBSDDetector(
    shape=(80, 100), pc=(0.45, 0.55, 0.60), sample_tilt=microscope.sample_tilt,
)
orientation = Rotation.from_euler([[35.0, 45.0, 10.0]], degrees=True)
coarse_image = coarse.project(detector, orientation, rescale=False)[0]
fine_image = fine.project(detector, orientation, rescale=False)[0]
difference = fine_image - coarse_image
relative_rms = difference.square().mean().sqrt() / fine_image.square().mean().sqrt()
assert torch.isfinite(relative_rms)
print(f"Grid-change relative RMS difference: {relative_rms.item():.3%}")
figure, axes = apl.subplots(1, 3, figsize=(900, 300))
for ax, values, title in zip(
    np.ravel(axes), (coarse_image, fine_image, difference),
    ("npx = 8", "npx = 16", "Fine minus coarse (own contrast)"), strict=True,
):
    panel = ax.imshow(values.detach().cpu().numpy(), cmap="gray", origin="upper")
    panel.set_title(title)
figure

# %%
# Compare the two solvers at identical sampling and depth settings
# -----------------------------------------------------------------------------
#
# Use the smaller grid for a quick numerical check. Reuse the same supplied
# depth distribution and float64 directions. This checks implementation
# consistency; neither solver is being compared with measured data here.

reference_sim = coarse_sim.refine(backend="bloch")
reference = reference_sim.compute(device=device, dtype=torch.float64, chunk_size=512)
scale = reference.data.abs().max().clamp_min(1e-30)
relative_max = (coarse.data - reference.data).abs().max() / scale
assert torch.isfinite(relative_max) and relative_max < 1e-7
print(f"PRISM versus Bloch maximum relative difference: {relative_max.item():.3e}")

# %%
# Continue a convergence study
# -----------------------------------------------------------------------------
#
# Increase ``npx`` again and repeat the detector-space comparison. Separately
# lower ``dmin``, increase the depth window, and increase the Monte Carlo
# electron count until the relevant measured quantity is stable. Changing
# ``texture_height`` in ``.show()`` only changes the rendering and cannot
# improve the computed master pattern.

print("Next grid size:", fine_sim.refine(npx=32).npx)
