"""Fundamental-zone dictionary indexing: the symmetry-native, fp16 round trip.

Validates :class:`de_ebsd.index.dictionary.DictionaryIndexer`. The money test is the
same closed round trip the SOFT indexer uses -- project a ``de_ebsd.sim`` master at a
*known* orientation ``R`` with :meth:`~de_ebsd.sim.MasterPattern.project`, then index
it back -- but here the master must carry the **crystal point-group symmetry**, because
that is exactly what makes sampling only the fundamental zone valid: two orientations
related by a symmetry operator project to the same pattern, so one fundamental-zone
representative stands in for the whole orbit.

So the synthetic master here is a sum of great-circle bands placed at **symmetrized**
poles: each seed direction is expanded into its full orbit under the cubic point group,
which makes the band field invariant under the point group (``M(g v) = M(v)``). A test
orientation and its fundamental-zone equivalent then produce identical patterns, and the
fundamental-zone dictionary recovers every orientation to within its (coarse) angular
resolution. Refinement (a *perfect* seed from the coarse cell) then sharpens it to
sub-degree. CPU + float32 throughout for determinism; the fp16 matmul path is a
tensor-core operation, so it is exercised only when CUDA is available.
"""

import numpy as np
import pytest
import torch
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import C1
from orix.sampling import get_sample_fundamental

from de_ebsd import EBSDDetector, MasterPattern
from de_ebsd.core.lambert import master_pattern_grid
from de_ebsd.crystal import Crystal
from de_ebsd.index import DictionaryIndexer
from de_ebsd.index.io import phase_from_crystal

# A coarse fundamental-zone resolution keeps the dictionary small (~800 orientations for
# m-3m) so the whole suite is fast; the recovered orientation must land within ~1.5x the
# resolution of truth (the nearest fundamental-zone cell).
RESOLUTION = 10.0
TOL_DEG = 1.5 * RESOLUTION

# General / asymmetric orientations (Bunge Euler, degrees): a genuine test of the frame
# convention and of the symmetry reduction (none of these sit in the fundamental zone).
_KNOWN_EULER_DEG = np.array(
    [
        [10.0, 20.0, 30.0],
        [45.0, 55.0, 15.0],
        [123.0, 47.0, 205.0],
        [200.0, 100.0, 300.0],
        [300.0, 80.0, 40.0],
        [15.0, 35.0, 95.0],
    ]
)


@pytest.fixture(scope="module")
def crystal() -> Crystal:
    """A cubic Ni crystal (Fm-3m, point group m-3m) for phase / symmetry derivation."""
    return Crystal.from_parameters(
        a=0.352, space_group=225, atoms=[("Ni", (0.0, 0.0, 0.0))]
    )


@pytest.fixture(scope="module")
def point_group(crystal):
    """The m-3m point group orix derives from the crystal's space group."""
    return phase_from_crystal(crystal).point_group


def _symmetric_band_master(crystal, point_group) -> MasterPattern:
    """A band master invariant under the crystal point group (``M(g v) = M(v)``).

    Each seed direction is expanded into its full orbit under the point-group rotation
    matrices, so the set of band poles is closed under the symmetry and the resulting
    great-circle-band field is point-group invariant -- the condition that makes
    fundamental-zone sampling valid. Distinctive enough (five pole families) that the
    partial-FOV patterns are unambiguous. The crystal is stashed in ``metadata`` so the
    indexer can auto-derive the point group and the scan phase.
    """
    mats = np.asarray(point_group.to_matrix())  # (n_sym, 3, 3) proper rotations
    seeds = np.array(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [2.0, 1.0, 0.0], [3.0, 1.0, 1.0]]
    )
    seeds = seeds / np.linalg.norm(seeds, axis=1, keepdims=True)
    orbit, amps = [], []
    for i, seed in enumerate(seeds):
        for pole in mats @ seed:  # the seed's orbit under the point group
            orbit.append(pole)
            amps.append(0.6 + 0.5 * i / len(seeds))
    poles = torch.tensor(np.array(orbit), dtype=torch.float64)
    amp = torch.tensor(np.array(amps), dtype=torch.float64)

    def field(v: torch.Tensor) -> torch.Tensor:
        d = v.to(torch.float64) @ poles.T
        return (amp * torch.exp(-((d / 0.08) ** 2))).sum(-1)

    npx = 120
    north = master_pattern_grid(npx, dtype=torch.float64)
    south = north.clone()
    south[..., 2] = -south[..., 2]
    data = torch.stack([field(north), field(south)], dim=0)[None]  # (1, 2, n, n)
    return MasterPattern(
        data=data.to(torch.float32),
        energies_kev=np.array([20.0]),
        metadata={"crystal": crystal},
    )


@pytest.fixture(scope="module")
def master(crystal, point_group) -> MasterPattern:
    return _symmetric_band_master(crystal, point_group)


@pytest.fixture(scope="module")
def detector() -> EBSDDetector:
    """A tilted detector with a generous field of view."""
    return EBSDDetector(shape=(70, 70), pattern_center=(0.5, 0.5, 0.5), sample_tilt=70.0, tilt=0.0)


@pytest.fixture(scope="module")
def indexer(master, detector) -> DictionaryIndexer:
    """One shared fp32/CPU dictionary indexer (dictionary built once, reused)."""
    return DictionaryIndexer(
        master, detector, resolution=RESOLUTION, device="cpu", dtype=torch.float32
    )


def _disorientation_deg(r_a: Rotation, r_b: Rotation, point_group) -> np.ndarray:
    """Symmetry-reduced disorientation (degrees) between two orix Rotations."""
    o_a = Orientation(r_a.data, symmetry=point_group)
    o_b = Orientation(r_b.data, symmetry=point_group)
    return np.asarray(o_a.angle_with(o_b, degrees=True)).ravel()


# --------------------------------------------------------------------------- #
# 1. round trip
# --------------------------------------------------------------------------- #
def test_roundtrip_recovers_known_orientations(indexer, master, detector, point_group):
    """THE acceptance test: project at known R, index back to within the FZ resolution.

    Several general / asymmetric orientations are projected and indexed as one batch;
    each recovered fundamental-zone orientation must sit within ~1.5x the sampling
    resolution of the true orientation (symmetry-reduced disorientation). A wrong frame
    convention would land tens of degrees off and fail loudly.
    """
    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=False)  # (Np, H, W)

    rot_out, quality = indexer.index(patterns)

    assert rot_out.size == r_true.size
    assert tuple(quality.shape) == (r_true.size,)
    assert torch.isfinite(quality).all()
    dis = _disorientation_deg(rot_out, r_true, point_group)
    for i, d in enumerate(dis):
        assert d < TOL_DEG, f"orientation {i}: {d:.2f} deg > {TOL_DEG:.2f} deg tol"


def test_match_binning_recovers_same_and_shrinks(master, detector, crystal, point_group):
    """match_binning coarse-indexes at a binned resolution: same answer, smaller dict.

    The coarse match only needs the (low-frequency) band geometry, so binning the match
    resolves the fundamental-zone cell just as well while the dictionary and match matmul
    shrink by ``match_binning**2``. Input patterns are still full-resolution; binning is
    internal. (Full-resolution accuracy is retained by refining separately.)
    """
    b = 2
    idx_full = DictionaryIndexer(master, detector, resolution=RESOLUTION, device="cpu",
                                 dtype=torch.float32)
    idx_bin = DictionaryIndexer(master, detector, resolution=RESOLUTION, device="cpu",
                                dtype=torch.float32, match_binning=b)
    # dictionary shrinks by b**2 in the pixel axis; same orientation count
    assert idx_bin.dictionary.shape == (
        (detector.nrows // b) * (detector.ncols // b), idx_full.n_dict)
    assert idx_bin.n_dict == idx_full.n_dict

    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=False)  # full-res (Np, H, W)
    rot_full, _ = idx_full.index(patterns)
    rot_bin, _ = idx_bin.index(patterns)  # takes full-res input, bins internally
    # binned coarse match lands within a cell of the full-res match AND of the truth
    assert (_disorientation_deg(rot_bin, rot_full, point_group) < TOL_DEG).all()
    assert (_disorientation_deg(rot_bin, r_true, point_group) < TOL_DEG).all()


def test_match_binning_must_divide_detector(master, detector):
    """match_binning that does not divide the detector shape raises clearly."""
    with pytest.raises(ValueError, match="must divide"):
        DictionaryIndexer(master, detector, resolution=RESOLUTION, device="cpu",
                          dtype=torch.float32, match_binning=3)  # 70 not divisible by 3


def test_batched_equals_single(indexer, master, detector):
    """A batched index equals indexing each pattern on its own (same argmax per row)."""
    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=False)

    rot_b, q_b = indexer.index(patterns)
    for i in range(r_true.size):
        rot_s, q_s = indexer.index(patterns[i])
        assert rot_s.size == 1
        assert np.allclose(rot_b[i].data, rot_s[0].data)
        assert float(q_s[0]) == pytest.approx(float(q_b[i]), rel=1e-5)


# --------------------------------------------------------------------------- #
# 6. numpy input + single (H, W) input
# --------------------------------------------------------------------------- #
def test_single_pattern_and_numpy_input(indexer, master, detector, point_group):
    """A bare ``(H, W)`` numpy array indexes to a length-1 Rotation + length-1 quality."""
    r_true = Rotation.from_euler(np.deg2rad([[70.0, 25.0, 110.0]]))
    pattern = master.project(detector, r_true, rescale=False)[0].cpu().numpy()  # (H, W)
    assert pattern.ndim == 2

    rot_out, quality = indexer.index(pattern)
    assert rot_out.size == 1
    assert tuple(quality.shape) == (1,)
    assert torch.isfinite(quality).all()
    assert _disorientation_deg(rot_out, r_true, point_group)[0] < TOL_DEG


def test_return_scores_matches_argmax(indexer, master, detector):
    """``return_scores=True`` returns the full (Np, n_dict) NCC matrix, argmax = result."""
    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG[:3]))
    patterns = master.project(detector, r_true, rescale=False)

    out = indexer.index(patterns, return_scores=True)
    assert len(out) == 3
    rot, quality, scores = out
    assert tuple(scores.shape) == (r_true.size, indexer.n_dict)
    for i in range(r_true.size):
        assert float(scores[i].max()) == pytest.approx(float(quality[i]), rel=1e-5)
        best = int(torch.argmax(scores[i]))
        assert np.allclose(rot[i].data, indexer.dictionary_rotations[best].data)


def test_wrong_pattern_shape_raises(indexer):
    """A pattern whose (H, W) does not match the detector is rejected with a clear error."""
    with pytest.raises(ValueError, match="does not match the detector"):
        indexer.index(torch.zeros(3, 40, 40))


# --------------------------------------------------------------------------- #
# 2. symmetry: dictionary lives in the fundamental zone and is far smaller than SO(3)
# --------------------------------------------------------------------------- #
def test_dictionary_orientations_in_fundamental_zone(indexer, point_group):
    """Every dictionary orientation lies in the fundamental zone.

    Two independent checks: (a) each orientation has the smallest rotation angle among
    all its symmetric equivalents (the defining property of the fundamental-zone
    representative), and (b) reducing it into the symmetry-reduced zone leaves the
    quaternion unchanged (it is already the representative).
    """
    fz = indexer.dictionary_rotations
    # (a) minimum-angle property: the dictionary orientation is the smallest-angle
    # member of its symmetry orbit (up to numerical noise).
    equivalents = point_group.outer(fz)  # (n_sym, n_dict)
    min_angle = np.asarray(equivalents.angle).min(axis=0)  # radians
    self_angle = np.asarray(fz.angle).ravel()
    assert np.rad2deg(np.max(self_angle - min_angle)) < 1e-6

    # (b) reducing an already-in-FZ orientation does not move it (|quat dot| ~ 1)
    o = Orientation(fz.data, symmetry=point_group)
    reduced = o.reduce()
    dot = np.abs((np.asarray(o.data) * np.asarray(reduced.data)).sum(-1))
    assert float(dot.min()) > 1.0 - 1e-4


def test_dictionary_far_smaller_than_full_so3(indexer, point_group):
    """The cubic dictionary carries ~24x fewer orientations than a full-SO(3) grid.

    Sanity that the crystal symmetry is actually used: a full (triclinic, C1) sampling
    at the same resolution is many times larger than the fundamental-zone dictionary.
    """
    full = get_sample_fundamental(resolution=RESOLUTION, point_group=C1)
    assert indexer.n_dict > 0
    # cubic proper order is 24; boundary effects put the measured ratio near ~22.
    assert full.size / indexer.n_dict > 15
    # stored transposed + contiguous as (n_px, n_dict) so the match B operand is
    # contiguous (a transposed view runs at ~half the GPU's matmul throughput)
    assert tuple(indexer.dictionary.shape) == (
        indexer.detector.nrows * indexer.detector.ncols,
        indexer.n_dict,
    )
    assert indexer.dictionary.is_contiguous()


# --------------------------------------------------------------------------- #
# 3. index_scan -> CrystalMap
# --------------------------------------------------------------------------- #
def test_index_scan_grid_layout(indexer, master, detector, crystal, point_group):
    """``(rows, cols, H, W)`` input -> a CrystalMap of the right shape, phase, quality."""
    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG))  # 6 points -> 2x3
    patterns = master.project(detector, r_true, rescale=False).reshape(2, 3, 70, 70)

    xmap = indexer.index_scan(patterns, step=0.5)  # phase auto-derived from metadata

    assert xmap.shape == (2, 3)
    assert xmap.size == 6
    assert "quality" in xmap.prop
    assert xmap.prop["quality"].shape == (6,)
    # the phase carries the crystal's (m-3m) point group, auto-derived from the master
    assert xmap.phases[0].point_group.name == point_group.name
    # and the map recovers the same orientations as index()
    dis = _disorientation_deg(xmap.rotations, r_true, point_group)
    assert (dis < TOL_DEG).all()


def test_index_scan_flat_layout_needs_shape(indexer, master, detector, crystal, point_group):
    """``(N, H, W)`` input + explicit shape assembles the same map; phase passed in."""
    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=False)  # (6, H, W)

    xmap = indexer.index_scan(patterns, shape=(3, 2), phase=crystal)
    assert xmap.shape == (3, 2)
    assert xmap.phases[0].point_group.name == point_group.name

    # missing shape for a flat stack is a clear error
    with pytest.raises(ValueError, match="shape=.*required"):
        indexer.index_scan(patterns)


def test_index_scan_batch_size_invariant(indexer, master, detector, point_group):
    """The scan result is identical whichever mini-batch size is used."""
    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=False)

    xmap_a = indexer.index_scan(patterns, shape=(2, 3), batch_size=1)
    xmap_b = indexer.index_scan(patterns, shape=(2, 3), batch_size=64)
    assert np.allclose(xmap_a.rotations.data, xmap_b.rotations.data)
    assert np.allclose(xmap_a.prop["quality"], xmap_b.prop["quality"])


# --------------------------------------------------------------------------- #
# 4. refine=True sharpens the coarse dictionary orientations
# --------------------------------------------------------------------------- #
def test_refine_sharpens_coarse_orientations(indexer, master, detector, point_group):
    """Refinement takes the coarse dictionary orientation to sub-degree accuracy."""
    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=False)

    rot_coarse, _ = indexer.index(patterns)
    dis_coarse = _disorientation_deg(rot_coarse, r_true, point_group)

    rot_ref, score_ref = indexer.refine(patterns, rotations=rot_coarse, n_iterations=100)
    dis_ref = _disorientation_deg(rot_ref, r_true, point_group)

    # the coarse grid is a few degrees off; refinement drops it well below a degree
    assert (dis_ref < dis_coarse).all()
    assert float(dis_ref.max()) < 1.0
    assert float(dis_ref.max()) < float(dis_coarse.max())
    assert torch.isfinite(score_ref).all()


def test_refine_seeds_from_index_when_rotations_none(indexer, master, detector, point_group):
    """``refine`` with no seed runs ``index`` first, then sharpens."""
    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG[:3]))
    patterns = master.project(detector, r_true, rescale=False)

    rot_ref, _ = indexer.refine(patterns, n_iterations=100)
    dis_ref = _disorientation_deg(rot_ref, r_true, point_group)
    assert float(dis_ref.max()) < 1.0


def test_index_scan_refine_improves_quality(indexer, master, detector, point_group):
    """``index_scan(refine=True)`` sharpens the map and stores the refined NCC."""
    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=False).reshape(2, 3, 70, 70)

    xmap = indexer.index_scan(patterns, refine=True, refine_iters=100, batch_size=3)
    assert xmap.shape == (2, 3)
    dis = _disorientation_deg(xmap.rotations, r_true, point_group)
    assert float(dis.max()) < 1.0  # sub-degree after refinement
    # refined NCC on a clean synthetic master is near 1
    assert float(np.min(xmap.prop["quality"])) > 0.9


def test_refine_defaults_to_lm(indexer, master, detector, point_group):
    """``DictionaryIndexer.refine`` defaults to the second-order LM path.

    The dictionary indexer is the fast coarse-then-refine workflow, so its ``refine``
    (and ``index_scan(refine=True)``) default to ``method="lm"``. Verified behaviorally:
    the default result is byte-identical to an explicit ``method="lm"`` run and differs
    from ``method="adam"`` at the same step budget; and LM still sharpens to sub-degree.
    """
    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=False)
    coarse, _ = indexer.index(patterns)

    default_ref, _ = indexer.refine(patterns, rotations=coarse, n_iterations=15)
    lm_ref, _ = indexer.refine(patterns, rotations=coarse, n_iterations=15, method="lm")
    adam_ref, _ = indexer.refine(patterns, rotations=coarse, n_iterations=15, method="adam")

    # the default IS lm (identical), and is not the adam path
    assert np.allclose(default_ref.data, lm_ref.data)
    assert not np.allclose(default_ref.data, adam_ref.data)
    # and the default (LM) refinement reaches sub-degree
    dis = _disorientation_deg(default_ref, r_true, point_group)
    assert float(dis.max()) < 1.0


# --------------------------------------------------------------------------- #
# 5. dtype: fp32 (CPU) and fp16 (CUDA-guarded)
# --------------------------------------------------------------------------- #
def test_fp32_dictionary_dtype_and_indexes(indexer, master, detector, point_group):
    """The default-CPU indexer stores an fp32 dictionary and indexes correctly."""
    assert indexer.dtype == torch.float32
    assert indexer.dictionary.dtype == torch.float32
    r_true = Rotation.from_euler(np.deg2rad([[33.0, 44.0, 55.0]]))
    pattern = master.project(detector, r_true, rescale=False)[0]
    rot_out, _ = indexer.index(pattern)
    assert _disorientation_deg(rot_out, r_true, point_group)[0] < TOL_DEG


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 matmul needs CUDA")
def test_fp16_dictionary_on_cuda_indexes(master, detector, point_group):
    """On CUDA the fp16 dictionary stores as half and still recovers the orientation.

    fp16 matmul accumulates in fp32 on tensor cores, so the NCC is accurate; the
    recovered orientation matches the fp32 dictionary's to within the FZ cell.
    """
    master_cuda = master.to("cuda")
    idx16 = DictionaryIndexer(
        master_cuda, detector, resolution=RESOLUTION, device="cuda", dtype=torch.float16
    )
    assert idx16.dtype == torch.float16
    assert idx16.dictionary.dtype == torch.float16
    assert idx16.device.type == "cuda"

    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG))
    patterns = master_cuda.project(detector, r_true, rescale=False)
    rot_out, quality = idx16.index(patterns)
    assert torch.isfinite(quality).all()
    dis = _disorientation_deg(rot_out, r_true, point_group)
    assert (dis < TOL_DEG).all()


# --------------------------------------------------------------------------- #
# point-group resolution + device plumbing
# --------------------------------------------------------------------------- #
def test_point_group_resolution(master, detector, point_group):
    """Explicit point group wins; else it is derived from the master's crystal."""
    # derived from master.metadata["crystal"]
    idx_derived = DictionaryIndexer(
        master, detector, resolution=RESOLUTION, device="cpu", dtype=torch.float32
    )
    assert idx_derived.point_group.name == point_group.name

    # explicit point group is used as given
    idx_explicit = DictionaryIndexer(
        master,
        detector,
        point_group=point_group,
        resolution=RESOLUTION,
        device="cpu",
        dtype=torch.float32,
    )
    assert idx_explicit.point_group is point_group
    assert idx_explicit.n_dict == idx_derived.n_dict


def test_missing_symmetry_raises(detector):
    """A master with no crystal and no explicit point group is a clear error."""
    npx = 40
    north = master_pattern_grid(npx, dtype=torch.float32)
    data = torch.ones(1, 2, north.shape[0], north.shape[1])
    bare = MasterPattern(data=data, energies_kev=np.array([20.0]))  # no metadata crystal
    with pytest.raises(ValueError, match="needs the crystal symmetry"):
        DictionaryIndexer(bare, detector, resolution=RESOLUTION, device="cpu")


def test_wrong_point_group_type_raises(master, detector):
    """A non-Symmetry ``point_group`` is rejected with a clear TypeError."""
    with pytest.raises(TypeError, match="must be an orix Symmetry"):
        DictionaryIndexer(master, detector, point_group="m-3m", device="cpu")


def test_device_defaults_to_master_device(master, detector):
    """With ``device=None`` the indexer follows the master's device."""
    idx = DictionaryIndexer(master, detector, resolution=RESOLUTION, dtype=torch.float32)
    assert idx.device == torch.device(master.device)
