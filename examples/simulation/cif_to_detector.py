"""
From a CIF to a dynamical master pattern and detector images
===============================================================================

Load silicon, inspect the crystal and diffraction objects, simulate electron
transport, compute both master-pattern hemispheres, and project them for known
orix orientations. Finish with a compact phase overview and an HDF5 round trip.

The bundled CIF contains an asymmetric unit in angstroms. This example uses
5,000 Monte Carlo electrons, a 33-by-33 grid per hemisphere, and a relatively
short reflection list so it is practical on a CPU. Increase the electron
count, integration-depth window and grid resolution, and lower ``dmin`` in a
convergence study before interpreting fine experimental contrast.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import anyplotlib as apl
import numpy as np
import torch
from orix.quaternion import Rotation

import de_ebsd as de
from de_ebsd.diffraction.kvectors import sample_incident_directions

# %%
# Load the crystal
# -----------------------------------------------------------------------------
#
# Resolve the CIF beside this script, or beside a downloaded notebook. The last
# fallback also supports running the notebook from the repository root.

cif_path = Path(__file__).with_name("Si.cif") if "__file__" in globals() else Path("Si.cif")
if not cif_path.is_file():
    cif_path = Path("examples") / "Si.cif"
crystal = de.Crystal.from_cif(cif_path)
print(crystal)
print(f"Unit-cell volume: {crystal.volume:.5f} nm^3; density: {crystal.density:.3f} g/cm^3")
crystal.show()

# %%
# Configure the experiment and simulate electron transport
# -----------------------------------------------------------------------------
#
# ``energy`` is in eV; the transport API takes keV. Explicitly pass the same
# sample tilt to transport and detector geometry. A fixed seed makes repeat
# runs on the same device reproducible. Set ``device = "cuda"`` for a CUDA GPU.

device = "cpu"
microscope = de.Microscope(energy=20e3, sample_tilt=70.0)
detector = de.EBSDDetector(
    shape=(96, 128), pc=(0.45, 0.55, 0.60), tilt=5.0,
    sample_tilt=microscope.sample_tilt, px_size=50.0,
)
depth = de.DepthDistribution.from_simulation(
    crystal, microscope.energy_kev,
    n_electrons=5_000, tilt_deg=microscope.sample_tilt,
    e_min_kev=15.0, e_bin_kev=2.5, depth_max_nm=60.0, depth_step_nm=2.0,
    seed=7, device=device,
)
print(depth)
print("Bruker PC:", detector.pc, "gnomonic bounds:", detector.gnomonic_bounds)

# %%
# Inspect the simulation before computing
# -----------------------------------------------------------------------------
#
# Select ``bloch_prism`` explicitly: the simulator's default is the
# kinematical model. Reusing the supplied depth distribution avoids repeating
# Monte Carlo when computing or refining this simulator.

sim = de.MasterPatternSimulator(
    crystal, microscope, npx=16, dmin=0.12,
    monte_carlo=depth, backend="bloch_prism",
)
sim.structure_factors().show(n=12)

# %%
# Classify strong and weak reflections for a particular incident direction.

sim.reflections(k=(1, 1, 3)).show(n=16)

# %%
# Inspect the corresponding dynamical matrix, including Bethe corrections.

sim.dynamical_matrix(k=(1, 1, 3)).show()

# %%
# The sampling points cover both hemispheres, including their shared equator.

sample_incident_directions(sim.npx).show()

# %%
# Compute the full master pattern
# -----------------------------------------------------------------------------
#
# Symmetry folding reduces the number of evaluated directions. The result
# still contains both complete hemispheres in square-Lambert storage.
# This single-energy calculation uses the 20 keV depth bin.

master = sim.compute(device=device, dtype=torch.float64, chunk_size=512)
assert master.data.shape == (1, 2, 33, 33)
assert torch.isfinite(master.data).all() and master.data.std() > 0
print(master)
master.show(projection="lambert")

# %%
# Rotate and zoom the real anyplotlib 3-D master sphere. ``texture_height`` and
# ``n_grid`` control display resolution only; they do not refine the simulation.

master.show(projection="sphere", texture_height=128, n_grid=64)

# %%
# Project for known orientations
# -----------------------------------------------------------------------------
#
# Orix Bunge Euler angles are given explicitly in degrees. Detector row zero
# is at the top, unlike the Lambert display. ``rescale=False`` preserves
# simulation intensities for comparison across orientations.

eulers = np.array([[0.0, 0.0, 0.0], [35.0, 45.0, 10.0], [90.0, 25.0, 60.0]])
orientations = Rotation.from_euler(eulers, degrees=True)
patterns = master.project(detector, orientations, rescale=False)
assert patterns.shape == (3, 96, 128) and torch.isfinite(patterns).all()
figure, axes = apl.subplots(1, 3, figsize=(900, 300))
for ax, pattern, angles in zip(np.ravel(axes), patterns, eulers, strict=True):
    panel = ax.imshow(pattern.detach().cpu().numpy(), cmap="gray", origin="upper")
    panel.set_title(f"Euler {tuple(angles.astype(int))} deg")
figure

# %%
# Compose a phase overview for Jupyter or SpyDE
# -----------------------------------------------------------------------------
#
# The overview exposes metadata and a single native figure that an application
# can mount through its existing anyplotlib bridge. In Jupyter, ``phase`` also
# displays the metadata card; use ``phase.figure`` here for Gallery HTML capture.

phase = crystal.overview(master_pattern=master, name="Silicon", n_grid=64)
print(phase.metadata)
phase.figure

# %%
# Save and reload without leaving generated files in the source tree
# -----------------------------------------------------------------------------
#
# Replace the temporary path with a persistent output path in your own work.
# A standalone HTML export retains local camera interaction. A notebook setup
# editor can additionally be opened with ``sim.show_setup(detector=detector)``;
# read edits back using ``to_microscope()`` and ``to_detector()``.

with TemporaryDirectory(prefix="de_ebsd-example-") as output_dir:
    master_path = Path(output_dir) / "silicon_master.h5"
    master.save(str(master_path))
    restored = de.MasterPattern.load(str(master_path), device=device)
    torch.testing.assert_close(restored.data, master.data)
    np.testing.assert_array_equal(restored.energies_kev, master.energies_kev)
print("HDF5 round trip preserved the master-pattern data and energy bins.")
