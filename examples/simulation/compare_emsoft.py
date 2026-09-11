"""
Compare a dynamical simulation with EMsoft
===============================================================================

Compute a nickel master pattern using the crystal, Bethe parameters, reflection
cutoff and Monte Carlo depth distribution stored in an EMsoft reference file.
Compare both Lambert hemispheres, normalized residuals and a detector projection.

The default reference is the small nickel dataset bundled with kikuchipy:
20 keV, 401-by-401 pixels per hemisphere, originally simulated with EMsoft.
It is distributed under CC BY 4.0 and was converted from float32 to uint8.
See the `dataset documentation
<https://kikuchipy.org/en/stable/reference/generated/kikuchipy.data.nickel_ebsd_master_pattern_small.html>`_.
No download or EMsoft executable is required.

Matching the available settings makes the comparison more informative, but
quantization, resampling and any differences in solver conventions remain.
Pearson correlation measures pattern agreement, not equality of absolute
intensity. This example reports the observed agreement without imposing or
claiming a parity threshold.
"""

from pathlib import Path
from time import perf_counter

import anyplotlib as apl
import h5py
import numpy as np
import torch
from orix.quaternion import Rotation

import de_ebsd as de
from de_ebsd.diffraction.kvectors import sample_incident_directions
from de_ebsd.index import master_from_emsoft

# %%
# Select the reference
# -----------------------------------------------------------------------------
#
# Set ``reference_path = Path("/path/to/your_master.h5")`` to compare your own
# EMsoft run. It must contain both master hemispheres, crystal data, Bethe
# settings, the reflection cutoff and the associated Monte Carlo statistics.
# The existing importer sums all asymmetric-unit site contributions.

reference_path = None
if reference_path is None:
    import kikuchipy as kp

    signal = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
    source = signal.tmp_parameters
    reference_path = Path(source.folder) / f"{source.filename}.{source.extension}"

reference = master_from_emsoft(reference_path)
crystal = reference.metadata.get("crystal")
dmin = reference.metadata.get("dmin")
bethe_values = reference.metadata.get("bethe")
if crystal is None or dmin is None or bethe_values is None or len(bethe_values) != 4:
    raise ValueError("The EMsoft file must provide crystal data, dmin and four Bethe parameters.")
bethe = de.BetheParameters(*bethe_values)
depth = de.DepthDistribution.from_emsoft(str(reference_path))
energy_kev = float(reference.energies_kev.max())
if not np.any(np.isclose(depth.energies_kev, energy_kev)):
    raise ValueError("No Monte Carlo depth bin matches the selected master-pattern energy.")

with h5py.File(reference_path, "r") as handle:
    sample_tilt = float(np.asarray(handle["NMLparameters/MCCLNameList/sig"]).ravel()[0])
    azimuthal = float(np.asarray(handle["NMLparameters/MCCLNameList/omega"]).ravel()[0])
    stored_dtype = str(handle["EMData/EBSDmaster/mLPNH"].dtype)

microscope = de.Microscope(energy=energy_kev * 1e3, sample_tilt=sample_tilt, azimuthal=azimuthal)
print("Reference:", reference_path.name)
print("Stored intensity dtype:", stored_dtype)
print(crystal)
print(f"Energy: {energy_kev:g} keV; dmin: {dmin:.5f} nm")
print("Matched Bethe settings:", bethe)
print("Shared EMsoft depth distribution:", depth)

# %%
# Compute with matched inputs and native reference sampling
# -----------------------------------------------------------------------------
#
# Match the reference's grid, reflection cutoff and full depth profile. The
# bundled file has ``npx=200`` (401-by-401 samples per hemisphere). Setting
# ``npx=20`` is a faster preview, but can miss narrow features and produce a
# misleadingly high sampled-map correlation despite poor detector agreement.
# Use ``device="cuda"`` if available. The measured time below includes
# preparation, so it is not a controlled speed comparison with EMsoft.

device = "cpu"
npx = reference.npx
simulator = de.MasterPatternSimulator(
    crystal, microscope, npx=npx, dmin=dmin,
    bethe=bethe, monte_carlo=depth, backend="bloch_prism",
)
started = perf_counter()
simulated = simulator.compute(device=device, dtype=torch.float64, chunk_size=512).to("cpu")
if device == "cuda":
    torch.cuda.synchronize()
assert torch.isfinite(simulated.data).all()
print(f"de-ebsd computation including preparation: {perf_counter() - started:.2f} s")
print(f"Reference npx={reference.npx}; simulated npx={simulated.npx}")

# %%
# Align the hemispheres before computing errors
# -----------------------------------------------------------------------------
#
# Sample the EMsoft reference on the simulation's actual direction grid. The
# southern grid changes the sign of z only, preserving the Lambert pixel x/y
# ordering. Negating all three coordinates would compare antipodal pixels and
# can hide a convention error for centrosymmetric crystals.

directions = sample_incident_directions(npx, dtype=torch.float64).directions
ref_maps = reference.sample(directions, energy_kev=energy_kev).double()
our_maps = simulated.data[0].double()
assert ref_maps.shape == our_maps.shape == (2, 2 * npx + 1, 2 * npx + 1)


def standardized(values):
    """Remove offset and scale, rejecting non-finite or constant inputs."""
    values = values.double()
    if not torch.isfinite(values).all():
        raise ValueError("Cannot compare non-finite intensity values.")
    sigma = values.std(correction=0)
    if sigma <= 0:
        raise ValueError("Cannot compare a constant pattern.")
    return (values - values.mean()) / sigma


def correlation(a, b):
    """Pearson correlation over all elements, with population normalization."""
    return float((standardized(a) * standardized(b)).mean())


normalized_reference = torch.stack([standardized(values) for values in ref_maps])
normalized_simulation = torch.stack([standardized(values) for values in our_maps])
residual = normalized_simulation - normalized_reference
for i, hemisphere in enumerate(("north", "south")):
    r = correlation(our_maps[i], ref_maps[i])
    rms = float(residual[i].square().mean().sqrt())
    print(f"{hemisphere:5s}: Pearson r={r:.6f}; normalized RMS residual={rms:.6f}")
print(f"Both hemispheres, joint normalization: r={correlation(our_maps, ref_maps):.6f}")

# %%
# Inspect both hemispheres and their residuals
# -----------------------------------------------------------------------------
#
# Each input image is normalized to zero mean and unit standard deviation;
# the same display limits apply to all input panels. Residuals use a shared
# symmetric scale across both hemispheres, in normalized intensity units.
# Only the displayed contrast is clipped; the metrics above use every value.

comparison, axes = apl.subplots(2, 3, figsize=(1000, 650))
residual_limit = max(float(residual.abs().max()), 1e-12)
for i, hemisphere in enumerate(("north", "south")):
    for j, (values, title) in enumerate((
        (normalized_reference[i], "EMsoft"),
        (normalized_simulation[i], "de-ebsd"),
        (residual[i], "de-ebsd minus EMsoft"),
    )):
        limit = residual_limit if j == 2 else 3.0
        panel = axes[i, j].imshow(
            values.numpy(), origin="lower", cmap="coolwarm" if j == 2 else "gray",
            vmin=-limit, vmax=limit,
        )
        panel.set_title(f"{hemisphere}: {title}")
comparison

# %%
# Compare detector patterns at one known orientation
# -----------------------------------------------------------------------------
#
# Use identical Bruker PC, detector tilt and orix orientation for both masters.
# ``rescale=False`` retains the original projected intensities; normalization
# is performed explicitly for the scale-independent comparison. With the
# default settings both masters use the same grid resolution; reducing npx
# additionally introduces a sampling/interpolation difference.

detector = de.EBSDDetector(
    shape=(96, 128), pc=(0.45, 0.55, 0.60), tilt=5.0,
    sample_tilt=microscope.sample_tilt, azimuthal=microscope.azimuthal,
)
orientation = Rotation.from_euler([[35.0, 45.0, 10.0]], degrees=True)
ref_detector = reference.project(detector, orientation, energy_kev=energy_kev, rescale=False)[0]
our_detector = simulated.project(detector, orientation, energy_kev=energy_kev, rescale=False)[0]
ref_image, our_image = standardized(ref_detector), standardized(our_detector)
detector_residual = our_image - ref_image
print(f"Detector-pattern correlation: r={correlation(our_detector, ref_detector):.6f}")
print(f"Detector normalized RMS residual: {float(detector_residual.square().mean().sqrt()):.6f}")
detector_comparison, detector_axes = apl.subplots(1, 3, figsize=(1000, 320))
for ax, values, title, is_residual in zip(
    np.ravel(detector_axes), (ref_image, our_image, detector_residual),
    ("EMsoft detector", "de-ebsd detector", "Normalized residual"), (False, False, True),
    strict=True,
):
    limit = max(float(detector_residual.abs().max()), 1e-12) if is_residual else 3.0
    panel = ax.imshow(
        values.numpy(), origin="upper", cmap="coolwarm" if is_residual else "gray",
        vmin=-limit, vmax=limit,
    )
    panel.set_title(title)
detector_comparison

# %%
# Inspect the EMsoft sphere
# -----------------------------------------------------------------------------
#
# These are the same anyplotlib sphere viewers used by the other examples.
# The surface mesh is deliberately modest; the intensity texture is separate
# from the mesh resolution. Smooth GPU rotation requires a supported browser
# and a secure context, such as localhost or trusted HTTPS.

reference.show(projection="sphere", energy_kev=energy_kev, texture_height=256, n_grid=32)

# %%
# Inspect the de-ebsd sphere
# -----------------------------------------------------------------------------
#
# Fine bands are limited by the computed hemisphere grid. Increasing the
# display texture size cannot recover features absent from that grid.

simulated.show(projection="sphere", texture_height=256, n_grid=32)

# %%
# Interpreting the result
# -----------------------------------------------------------------------------
#
# High correlation is encouraging but does not establish bitwise or absolute
# intensity parity. For a stricter check, use an unquantized EMsoft file from
# a documented run, set ``npx=reference.npx``, and check the crystal setting,
# site summation, depth profile, Bethe cutoffs and energy selection. Residual
# structure is worth inspecting even when Pearson r is close to one.
