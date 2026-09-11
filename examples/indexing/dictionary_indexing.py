"""
Dictionary indexing: a synthetic scan, indexed and refined
===============================================================================

Simulates a two-grain scan from a de-ebsd master pattern, dirties it with a
static background and noise, then recovers the orientations with
:class:`~de_ebsd.index.dictionary.DictionaryIndexer` -- the brute-force
template-matching baseline -- and sharpens them with gradient refinement.

This is a **closed round trip**: the same master pattern produces the synthetic
scan and the dictionary templates, so it validates the conventions and the
workflow, not experimental accuracy. A real scan carries detector response,
pattern-center error and a phase list that none of this exercises. See
``simulation/compare_emsoft.py`` for a comparison against an external reference.

Sizes are kept small so the example runs on a CPU in seconds: a 100-pixel
Lambert grid, a 60-by-80 detector, and a 140-point scan. Raise ``npx``, lower
``dmin`` and lower ``resolution`` for production work.
"""

import numpy as np
import torch
from orix.quaternion import Orientation, Rotation

import de_ebsd as de
from de_ebsd.index import (
    DictionaryIndexer,
    phase_from_crystal,
    remove_background,
    to_crystal_map,
)
from de_ebsd.viz import show_map, show_match, show_refinement

# %%
# The phase, the instrument and the master pattern
# -----------------------------------------------------------------------------
#
# Nickel, 20 keV, a 70-degree sample tilt. The master pattern is the shared
# object: :meth:`~de_ebsd.sim.MasterPattern.project` turns it into the scan
# below, and the indexer projects it again at every dictionary orientation.
# Set ``device = "cuda"`` on a CUDA-capable workstation.

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
print(master)

# %%
# Build a synthetic two-grain scan
# -----------------------------------------------------------------------------
#
# A 10-by-14 scan split by a slanted boundary into two grains, each carrying a
# small orientation gradient so the IPF map is not flat. ``truth`` is kept for
# the accuracy check at the end; a real scan has no such reference.

rows, cols = 10, 14
row_idx, col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
grain = (col_idx >= cols // 2 + (row_idx - rows / 2) * 0.35).astype(int)
grain_euler = np.array([[15.0, 35.0, 25.0], [205.0, 60.0, 130.0]])
euler = grain_euler[grain.ravel()] + np.column_stack([
    2.0 * col_idx.ravel() / cols,        # a gentle gradient across the grain
    1.5 * row_idx.ravel() / rows,
    np.zeros(rows * cols),
])
truth = Rotation.from_euler(euler, degrees=True)
clean = master.project(detector, truth)
print(f"{rows * cols} patterns of shape {tuple(clean.shape[-2:])}")

# %%
# Dirty the scan the way a detector does
# -----------------------------------------------------------------------------
#
# A smooth multiplicative static background (the detector's own illumination
# profile) plus Gaussian read noise. This is a stand-in for a real background,
# not a detector-response model: no energy weighting, no gain or blur.

rng = np.random.default_rng(0)
py, px = np.mgrid[0:detector.nrows, 0:detector.ncols]
static_bg = 0.6 + 0.4 * np.exp(-(((py - 20) / 45.0) ** 2 + ((px - 40) / 60.0) ** 2))
raw = clean.numpy() * static_bg + rng.normal(0.0, 0.03, clean.shape)

# %%
# Remove the background before indexing
# -----------------------------------------------------------------------------
#
# :func:`~de_ebsd.index.preprocess.remove_background` runs the usual pipeline on
# ``device``: divide or subtract the static background, then subtract each
# pattern's own Gaussian-blurred copy, and rescale to ``[0, 1]``. Normalized
# cross-correlation is already invariant to offset and gain, so what this buys
# is the removal of the *spatially varying* background the NCC cannot absorb.

patterns = remove_background(raw, static_bg=static_bg, operation="divide", device=device)
print(f"preprocessed {tuple(patterns.shape)}, range "
      f"[{float(patterns.min()):.2f}, {float(patterns.max()):.2f}]")

# %%
# Build the dictionary
# -----------------------------------------------------------------------------
#
# The dictionary is the master projected at every orientation of a
# fundamental-zone sample, normalized once at construction; indexing is then a
# single matmul per chunk. ``resolution`` is the target average disorientation
# of that sample, and it drives the cost: the m-3m fundamental zone holds ~1,700
# orientations at 8 degrees but ~11,900 at 4 degrees. The point group comes from
# ``master.metadata["crystal"]`` when it is not passed explicitly.
#
# Coarse-plus-refine is the intended workflow, so ``resolution`` only has to be
# fine enough to land in the right basin; 4 degrees is already conservative for
# cubic nickel. Pseudo-symmetric phases need finer.

indexer = DictionaryIndexer(
    master, detector, resolution=4.0, device=device, dtype=torch.float32,
)
print(f"point group {indexer.point_group.name}, "
      f"{indexer.n_dict} dictionary orientations at {indexer.resolution} deg")

# %%
# Index the scan
# -----------------------------------------------------------------------------
#
# :meth:`~de_ebsd.index.dictionary.DictionaryIndexer.index_scan` takes the scan
# as ``(rows, cols, H, W)`` and assembles an orix ``CrystalMap`` carrying one
# orientation and one ``prop["quality"]`` (the peak NCC) per point. The IPF-z
# map shows the two grains; the quality map is the map to inspect first on real
# data, where it exposes bad points before the orientations do.

scan = patterns.reshape(rows, cols, detector.nrows, detector.ncols)
xmap = indexer.index_scan(scan, step=0.5)
print(xmap)
show_map(xmap, value="both")

# %%
# Check one point against its template
# -----------------------------------------------------------------------------
#
# :func:`~de_ebsd.viz.maps.show_match` re-projects the master at the indexed
# orientation and puts it beside the measured pattern with their difference.
# The bands should line up; the residual is the interpolation error plus the
# noise plus whatever is left of the grid quantization.

point = 0
show_match(patterns[point], master, detector, xmap.rotations[point])

# %%
# Refine to sub-degree
# -----------------------------------------------------------------------------
#
# The coarse orientations sit on a 4-degree grid, so they cannot be better than
# ~2 degrees. :meth:`~de_ebsd.index.dictionary.DictionaryIndexer.refine`
# maximizes the same NCC by autograd through the projection, seeded from the
# coarse result. The default ``method="lm"`` (Levenberg-Marquardt with an
# analytic Jacobian) converges in ~12 steps where Adam needs ~60.

refined, score, history = indexer.refine(
    patterns, rotations=xmap.rotations, return_history=True,
)
show_refinement(history)

# %%
# How close did it get?
# -----------------------------------------------------------------------------
#
# Disorientation against ``truth``, reduced by the cubic point group so that
# symmetry-equivalent descriptions of the same orientation count as identical.
# The coarse pass lands within the dictionary cell; refinement takes it roughly
# two orders of magnitude closer, which is the whole argument for the two-stage
# workflow -- and the reason the coarse grid does not need to be fine.

symmetry = phase_from_crystal(crystal).point_group
truth_o = Orientation(truth.data, symmetry=symmetry)


def disorientation_deg(rotations):
    return Orientation(rotations.data, symmetry=symmetry).angle_with(truth_o, degrees=True)


coarse_err = disorientation_deg(xmap.rotations)
refined_err = disorientation_deg(refined)
print(f"coarse  : median {np.median(coarse_err):6.3f} deg, max {coarse_err.max():6.3f} deg, "
      f"mean NCC {float(xmap.prop['quality'].mean()):.3f}")
print(f"refined : median {np.median(refined_err):6.3f} deg, max {refined_err.max():6.3f} deg, "
      f"mean NCC {float(score.mean()):.3f}")

# %%
# The refined map
# -----------------------------------------------------------------------------
#
# :func:`~de_ebsd.index.io.to_crystal_map` assembles a flat, row-major
# ``Rotation`` plus a quality array into a ``CrystalMap``, which is what lets
# refined results re-enter the orix/kikuchipy world. Passing ``refine=True`` to
# ``index_scan`` does the same thing in one call. Against the coarse map above,
# the visible change is inside each grain: the gradient is continuous rather
# than stepped onto dictionary cells.

refined_map = to_crystal_map(refined, score, (rows, cols), crystal, step=0.5)
show_map(refined_map, value="both")
