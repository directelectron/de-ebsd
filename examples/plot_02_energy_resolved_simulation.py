"""
Native Monte Carlo and energy-resolved EBSD simulation
===============================================================================

Simulate nickel electron transport, inspect its energy/depth histogram, and
compute a dynamical master pattern and detector image for each energy bin.
Everything is computed locally, without an EMsoft statistics file.

The coarse energy bins and 10,000 incident electrons illustrate the workflow.
They are not a converged transport calculation. The plotted histogram contains
the events retained within the requested energy and depth windows; its sum is
not the total backscatter yield over all depths and energies.
"""

import anyplotlib as apl
import numpy as np
import torch
from orix.quaternion import Rotation

import torchebsd as te

# %%
# Material, instrument, and transport
# -----------------------------------------------------------------------------
#
# Specify lengths in nm and beam energy in eV. The 15--20 keV interval contains
# three bin centers spaced by 2.5 keV. The depth bins are 2 nm apart.

device = "cpu"  # change to "cuda" on a CUDA-capable workstation
crystal = te.Crystal.from_parameters(
    a=0.35236, space_group=225, atoms=[("Ni", (0.0, 0.0, 0.0))], debye_waller=0.0035,
)
microscope = te.Microscope(energy=20e3, sample_tilt=70.0)
depth = te.DepthDistribution.from_simulation(
    crystal, microscope.energy_kev, n_electrons=10_000,
    tilt_deg=microscope.sample_tilt, e_min_kev=15.0, e_bin_kev=2.5,
    depth_max_nm=60.0, depth_step_nm=2.0, seed=11, device=device,
)
counts = depth.accum_z.sum(axis=(2, 3))
assert np.all(counts.sum(axis=1) > 0), "Increase electron count to populate every energy bin."
print("Energy bin centers (keV):", depth.energies_kev)
print("Retained counts per energy bin:", counts.sum(axis=1).astype(int))
print("Integration bin counts:", depth.integration_depths())

# %%
# Inspect retained transport events
# -----------------------------------------------------------------------------
#
# Plot event counts directly. The simulation itself performs its absorption
# correction and depth weighting; these raw counts are not substituted for
# the backend's weighting function.

histogram, ax = apl.subplots(1, 1, figsize=(620, 320))
panel = ax.imshow(counts, cmap="viridis", origin="lower")
panel.set_title("Retained electrons: rows = energy bins, columns = depth bins (2 nm)")
histogram

# %%
# Compute a separate full master pattern at every energy
# -----------------------------------------------------------------------------
#
# ``energy_resolved=True`` rebuilds the structure factors and selects the
# transport depth weights for each energy. Explicitly use a dynamical backend.

sim = te.MasterPatternSimulator(
    crystal, microscope, npx=12, dmin=0.12, monte_carlo=depth, backend="bloch_prism",
)
master = sim.compute(device=device, dtype=torch.float64, energy_resolved=True, chunk_size=512)
assert master.data.shape == (3, 2, 25, 25)
assert torch.isfinite(master.data).all()
print(master)
master.show(projection="sphere", energy_kev=20.0, texture_height=128, n_grid=64)

# %%
# Project each bin through the same detector and orientation
# -----------------------------------------------------------------------------
#
# Keep ``rescale=False`` so projection does not normalize each image to [-1, 1].
# The figure panels use their own display contrast. Do not interpret their
# brightness as a calibrated inter-bin yield or sum normalized images into a
# quantitative detector signal; that requires a specified detector response
# and a consistent energy-integration convention.

detector = te.EBSDDetector(
    shape=(80, 100), pc=(0.48, 0.58, 0.62), sample_tilt=microscope.sample_tilt, tilt=5.0,
)
orientation = Rotation.from_euler([[35.0, 45.0, 10.0]], degrees=True)
patterns = torch.stack([
    master.project(detector, orientation, energy_kev=float(energy), rescale=False)[0]
    for energy in master.energies_kev
])
assert patterns.shape == (3, 80, 100) and torch.isfinite(patterns).all()
figure, axes = apl.subplots(1, 3, figsize=(900, 300))
for ax, energy, pattern in zip(np.ravel(axes), master.energies_kev, patterns, strict=True):
    panel = ax.imshow(pattern.detach().cpu().numpy(), cmap="gray", origin="upper")
    panel.set_title(f"{energy:g} keV (individual display contrast)")
figure

# %%
# Verify the top bin against a single-energy calculation
# -----------------------------------------------------------------------------
#
# The same transport object and numerical settings isolate the effect of
# requesting multiple energies; the beam-energy map should be unchanged.

single_energy = sim.compute(device=device, dtype=torch.float64, chunk_size=512)
torch.testing.assert_close(master.data[-1], single_energy.data[0], rtol=1e-8, atol=1e-10)
print("The 20 keV map matches the single-energy calculation.")
