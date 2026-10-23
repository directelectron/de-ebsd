"""
Compare grid resolution and dynamical backends
===============================================================================

Compute nickel master patterns at two grid resolutions, project onto an
identical detector, and quantify the interpolation difference. Then compare
the scattering-matrix backend with the Bloch eigensolver at identical settings.

An explicitly prescribed exponential depth distribution keeps this numerical
comparison short and deterministic. It is an approximate depth model, not a
replacement for the native Monte Carlo workflow in the preceding examples.
Two grids demonstrate a convergence procedure; agreement between two solvers is
not evidence that the physical model is converged.
"""

from time import perf_counter

import anyplotlib as apl
import numpy as np
import torch
from orix.quaternion import Rotation

import de_ebsd as de

# %%
# Hold the experiment and depth model fixed
# -----------------------------------------------------------------------------
#
# Use double precision in both solvers for this comparison. Changing only
# ``npx`` preserves the crystal, reflection cutoff and structure factors.

device = "cpu"
crystal = de.Crystal.from_parameters(
    a=0.35236, space_group=225, atoms=[("Ni", (0.0, 0.0, 0.0))], debye_waller=0.0035,
)
microscope = de.Microscope(energy=20e3, sample_tilt=70.0, probe_diameter_nm=10.0)
detector = de.EBSDDetector(
    shape=(80, 100), pattern_center=(0.45, 0.65, 0.60), sample_tilt=microscope.sample_tilt,
    tilt=5.0, px_size=400.0,
)
depth = de.DepthDistribution.single_energy(
    microscope.energy_kev, thickness=40.0, depth_step=2.0, decay_length=15.0,
)

# %%
# Inspect the crystal
# -----------------------------------------------------------------------------
#
# This is the actual expanded Ni unit cell in Cartesian nanometres. Drag to
# rotate the native anyplotlib figure; the same figure is usable in SpyDE.

crystal.show()

# %%
# Inspect the microscope, probe and detector
# -----------------------------------------------------------------------------
#
# The beam is vertical in the laboratory side view. Positive sample tilt slopes
# the surface down toward the detector; the arrow n points out of the sample.
# Incidence is measured from n (70 degrees), not from the surface (20 degrees).
# The example's **assumed** 10 nm circular parallel probe projects to a
# 29.24 x 10 nm footprint; it does not model electron scattering or broadening.
#
# Here the **assumed effective screen pitch** is 400 micrometres per binned
# pixel: the 100 x 80 image spans a 40 x 32 mm screen, with a 19.2 mm normal
# distance to its pattern center. This is screen calibration, not camera-sensor
# pitch. PCy is measured down from the top; PCz is normalized by screen height.
# The screen is tilted 5 degrees from vertical, 25 degrees relative to the
# sample plane. Its center is 3.11 mm above the impact point in laboratory
# height, and 19.43 mm above the sample plane along the sample normal.
# Click the detector face or edit the numeric fields to explore the geometry.
#
# Edits change this inspector only. In a notebook, explicitly read
# ``setup.to_microscope()`` and ``setup.to_detector()``, rebuild the simulator,
# and rerun downstream cells to use them. Exported gallery controls cannot
# recompute Python results. ``detector.show(microscope=microscope)`` offers the
# same geometry inspector when starting from a detector object.

setup = microscope.show(detector=detector, crystal=crystal)
setup

# %%
# Inspect the depth slices used by both backends
# -----------------------------------------------------------------------------
#
# The prescribed model has 20 depth bins, each 2 nm thick, spanning 0--40 nm
# below the surface. The viewer reads the real distribution and the same 99%
# cutoff used by backend preparation. Select a slice to inspect its yield and
# cumulative fraction; selection never changes the distribution.
#
# These are normal-depth integration slices, not crystal layers or a simulated
# electron trajectory. The stored exponential is sampled at z = j * dz;
# PRISM evaluates the propagated wave at (j + 1) * dz. Both solvers use the
# absorption-corrected weights internally; this viewer shows uncorrected exit
# statistics, so it does not mistake an increasing correction for deeper yield.

depth.show(energy_kev=microscope.energy_kev)

# %%
# Prepare the same diffraction model at two grid sizes
# -----------------------------------------------------------------------------

coarse_sim = de.MasterPatternSimulator(
    crystal, microscope, npx=100, dmin=0.05, monte_carlo=depth, backend="bloch_prism",
)
coarse_sim.structure_factors()
fine_sim = coarse_sim.refine(npx=200)
assert fine_sim.structure_factors() is coarse_sim.structure_factors()

# %%
# Compute both grids
# -----------------------------------------------------------------------------
#
# These timings include preparation and any first-use overhead, so they are
# illustrative rather than a controlled solver benchmark.

start = perf_counter()
coarse = coarse_sim.compute(device=device, dtype=torch.float64, chunk_size=512)
print(f"201 x 201 per hemisphere: {perf_counter() - start:.3f} s")
start = perf_counter()
fine = fine_sim.compute(device=device, dtype=torch.float64, chunk_size=512)
print(f"401 x 401 per hemisphere: {perf_counter() - start:.3f} s")
fine.show(projection="sphere", texture_height=128, n_grid=64)

# %%
# Compare on the same detector grid
# -----------------------------------------------------------------------------
#
# Compare raw intensities so independent image normalization cannot hide an
# amplitude change. Report relative RMS error against the finer projection.
# A larger Lambert grid samples the same diffraction model more finely;
# lowering ``dmin`` is a separate convergence axis: it adds reflections rather
# than sampling the existing ones more finely. Both grids here share one
# ``dmin=0.05`` reflection list, so this comparison isolates sampling alone.

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
    ("npx = 100", "npx = 200", "Fine minus coarse (own contrast)"), strict=True,
):
    panel = ax.imshow(values.detach().cpu().numpy(), cmap="gray", origin="upper")
    panel.set_title(title)
figure

# %%
# Compare the two solvers at identical sampling and depth settings
# -----------------------------------------------------------------------------
#
# Use the coarser of the two grids: the exact eigensolver costs ~4x the
# scattering-matrix backend here, and ~20x at ``npx=200``. Reuse the same
# supplied depth distribution and float64 directions. This checks
# implementation consistency; neither solver is being compared with measured
# data here.

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

print("Next grid size:", fine_sim.refine(npx=400).npx)
