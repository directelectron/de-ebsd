"""
Spherical indexing: cross-correlation over all of SO(3)
===============================================================================

Indexes the same kind of synthetic scan as ``dictionary_indexing.py``, but with
:class:`~de_ebsd.index.spherical.indexer.SphericalIndexer`: each pattern is
back-projected onto the sphere, expanded in spherical harmonics, and
cross-correlated against the master's spectrum over *every* rotation at once via
the Wigner-D transform (after EMSphInx; Lenthe, Singh & De Graef, 2019). The
orientation is read off the peak of that SO(3) surface.

The practical difference from a dictionary: there is no template library and no
fundamental-zone sample, so the cost does not depend on the crystal symmetry --
only on the band limit ``L``. A low-symmetry phase, whose fundamental zone is 24
or 48 times larger, costs a spherical indexer nothing extra.

Like the dictionary example, this is a closed round trip through the same master
pattern, so it validates conventions and the workflow, not experimental accuracy.
Bandwidths and scan size are kept small to stay CPU-practical.
"""

import time

import numpy as np
import torch
from orix.quaternion import Orientation, Rotation

import de_ebsd as de
from de_ebsd.index import SphericalIndexer, phase_from_crystal, remove_background
from de_ebsd.viz import show_correlation, show_map, show_match, show_refinement

# %%
# The phase, the instrument and the master pattern
# -----------------------------------------------------------------------------
#
# Nickel at 20 keV again, so the numbers here can be read beside the dictionary
# example. Set ``device = "cuda"`` on a CUDA-capable workstation; the SO(3)
# surface is the memory driver, and it is exactly what a GPU handles well.

device = "cpu"
crystal = de.Crystal.from_parameters(
    a=0.35236, space_group=225, atoms=[("Ni", (0.0, 0.0, 0.0))], debye_waller=0.0035,
)
microscope = de.Microscope(energy=20e3, sample_tilt=70.0)
detector = de.EBSDDetector(
    shape=(60, 80), pc=(0.45, 0.55, 0.60), tilt=5.0,
    sample_tilt=microscope.sample_tilt, px_size=100.0,
)
depth = de.DepthDistribution.single_energy(
    microscope.energy_kev, thickness=40.0, depth_step=2.0, decay_length=15.0,
)
sim = de.MasterPatternSimulator(
    crystal, microscope, npx=100, dmin=0.10, monte_carlo=depth, backend="bloch_prism",
)
master = sim.compute(device=device, dtype=torch.float32)
symmetry = phase_from_crystal(crystal).point_group
print(master)

# %%
# A small scan with known orientations
# -----------------------------------------------------------------------------
#
# Two grains split by a slanted boundary, each with a gentle gradient, plus a
# static background and read noise -- the same recipe as the dictionary example,
# at 6-by-8 because a spherical index costs far more per pattern than a
# dictionary lookup (one SO(3) transform each, against one row of a matmul).

rows, cols = 6, 8
row_idx, col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
grain = (col_idx >= cols // 2 + (row_idx - rows / 2) * 0.35).astype(int)
grain_euler = np.array([[15.0, 35.0, 25.0], [205.0, 60.0, 130.0]])
euler = grain_euler[grain.ravel()] + np.column_stack([
    2.0 * col_idx.ravel() / cols,
    1.5 * row_idx.ravel() / rows,
    np.zeros(rows * cols),
])
truth = Rotation.from_euler(euler, degrees=True)
truth_o = Orientation(truth.data, symmetry=symmetry)

rng = np.random.default_rng(0)
py, px = np.mgrid[0:detector.nrows, 0:detector.ncols]
static_bg = 0.6 + 0.4 * np.exp(-(((py - 20) / 45.0) ** 2 + ((px - 40) / 60.0) ** 2))
clean = master.project(detector, truth).numpy()
raw = clean * static_bg + rng.normal(0.0, 0.03, clean.shape)
patterns = remove_background(raw, static_bg=static_bg, operation="divide", device=device)
print(f"{rows * cols} patterns of shape {tuple(patterns.shape[-2:])}")

# %%
# The SO(3) correlation surface for one pattern
# -----------------------------------------------------------------------------
#
# ``return_surface=True`` hands back the whole score volume that was arg-maxed,
# on the SOFT ``(alpha, beta, gamma)`` grid of ``N = 2L + 2`` points per axis.
# :func:`~de_ebsd.viz.maps.show_correlation` collapses it to an
# ``(alpha, gamma)`` heatmap by taking the maximum over ``beta``, so the global
# peak survives, and marks the peak cell.
#
# The repeated structure across the map is the cubic symmetry: every one of the
# 24 rotations that carries nickel onto itself produces an equally good peak.
# The indexer does not know about symmetry and does not reduce to the
# fundamental zone -- ``Orientation`` with a point group does that afterwards.

indexer = SphericalIndexer(master, detector, bandwidth=53, device=device)
rotations, quality, surface = indexer.index(patterns[:1], return_surface=True)
print(f"L = {indexer.bandwidth}, SOFT grid step "
      f"{360 / (2 * indexer.bandwidth + 2):.2f} deg, surface {tuple(surface.shape)}")
show_correlation(surface)

# %%
# Bandwidth is the accuracy/cost knob
# -----------------------------------------------------------------------------
#
# The SOFT grid step is ``360 / (2L + 2)`` degrees, and cost grows as ~``L^4``
# in both time and memory, because
# :func:`~de_ebsd.index.spherical.wigner.wigner_d` materializes an
# ``(N, L+1, 2L+1, 2L+1)`` tensor.
#
# Note *how* the table below breaks down. Above the band limit the pattern
# needs, the error tracks the grid step and refinement cleans up the rest.
# Below it the error does not degrade gracefully: the truncated pattern stops
# resolving the narrow bands, the global peak moves to a different basin, and
# the answer is tens of degrees wrong. A dictionary indexer misbehaves the same
# way when ``resolution`` is too coarse. This is why the quality map, not the
# orientation map, is the one to read first -- a wrong-basin point scores
# visibly low, while its orientation just looks like a differently colored
# pixel.

probe = patterns[:6]
probe_truth = truth_o[:6]
for bandwidth in (32, 40, 53):
    start = time.perf_counter()
    trial = SphericalIndexer(master, detector, bandwidth=bandwidth, device=device)
    rot, _ = trial.index(probe)
    elapsed = time.perf_counter() - start
    error = Orientation(rot.data, symmetry=symmetry).angle_with(probe_truth, degrees=True)
    print(f"L = {bandwidth:3d}  step {360 / (2 * bandwidth + 2):5.2f} deg  "
          f"median error {np.median(error):5.2f} deg  "
          f"max {error.max():5.2f} deg  {elapsed:5.2f} s for {probe.shape[0]} patterns")

# %%
# Index the scan
# -----------------------------------------------------------------------------
#
# :meth:`~de_ebsd.index.spherical.indexer.SphericalIndexer.index_scan` mirrors
# the dictionary indexer's: ``(rows, cols, H, W)`` in, an orix ``CrystalMap``
# out, with the peak score under ``prop["quality"]``. ``batch_size`` bounds the
# ``(batch, N, N, N)`` surface, which is what dominates memory at production
# bandwidth; the results do not depend on it.
#
# ``normalize=True`` (the default) divides out the master's local norm inside
# each field of view. Without it, orientations that merely place a bright part
# of the master on the detector outscore the true one -- a failure mode worth
# knowing about, because it misindexes by ~100 degrees rather than degrading
# gracefully.

scan = patterns.reshape(rows, cols, detector.nrows, detector.ncols)
xmap = SphericalIndexer(master, detector, bandwidth=40, device=device).index_scan(
    scan, step=0.5, batch_size=16,
)
print(xmap)
show_map(xmap, value="both")

coarse_error = Orientation(xmap.rotations.data, symmetry=symmetry).angle_with(
    truth_o, degrees=True
)
print(f"coarse: median {np.median(coarse_error):.3f} deg, max {coarse_error.max():.3f} deg")

# %%
# Refine off the peak
# -----------------------------------------------------------------------------
#
# :meth:`~de_ebsd.index.spherical.indexer.SphericalIndexer.refine` maximizes the
# spherical masked NCC by autograd, seeded from the coarse peak and scored over
# the same in-field-of-view directions the coarse pass used. It is Adam here
# (the dictionary indexer defaults to Levenberg-Marquardt); ``keep_best=True``
# makes it monotone, so a seed is never made worse.

refined, score, history = indexer.refine(
    patterns, rotations=xmap.rotations, n_iters=60, lr=0.02, return_history=True,
)
show_refinement(history)

refined_error = Orientation(refined.data, symmetry=symmetry).angle_with(truth_o, degrees=True)
print(f"refined: median {np.median(refined_error):.3f} deg, "
      f"max {refined_error.max():.3f} deg, mean NCC {float(score.mean()):.3f}")

# %%
# The refined match
# -----------------------------------------------------------------------------
#
# The same sanity check the dictionary example ends on: the measured pattern,
# the master re-projected at the refined orientation, and their difference.

show_match(patterns[0], master, detector, refined[0])
