"""Kinematic Kikuchi bands: the second physics model beside the dynamical master.

A band appears where a direction is nearly perpendicular to a diffracting plane's
normal ``n``: :class:`KinematicMaster` draws each plane as a Gaussian in ``v . n``
of half-width ``w`` (a :class:`Reflectors` entry), summed over the planes. It is a
:class:`~de_ebsd.sim.masterpattern.SphereProjection` like
:class:`~de_ebsd.sim.MasterPattern`, so it projects onto an
:class:`~de_ebsd.sim.detector.EBSDDetector`, builds a
:class:`~de_ebsd.index.dictionary.DictionaryIndexer` and is refined by
:func:`~de_ebsd.index.refine.orientation.refine_orientation` through the same
geometry and the same rotation convention as the dynamical master. It needs no
Bloch-wave simulation, so it is the fallback on a CPU or where no master exists,
and its plane traces are what an overlay draws: :func:`band_lines` and
:func:`zone_axis_points` give them in detector pixels.

Geometry is :mod:`de_ebsd.core.geometry`'s (Bruker pattern center, kikuchipy's
sample-to-detector rotation), orientations are orix's (sample to crystal,
``v_crystal = g @ v_sample``), and lengths are in nm and energies in eV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from de_ebsd.core.constants import electron_wavelength_nm
from de_ebsd.core.device import resolve_device
from de_ebsd.core.geometry import pixels_from_rays, sample_to_detector_matrix
from de_ebsd.core.rotation import quaternion_to_matrix
from de_ebsd.diffraction.kinematic import _clip_line
from de_ebsd.sim.masterpattern import SphereProjection, _as_quaternions

__all__ = [
    "KinematicMaster",
    "Reflectors",
    "band_lines",
    "cubic_reflectors",
    "reflectors_from_phase",
    "zone_axis_points",
]

log = logging.getLogger(__name__)

#: nm per Angstrom, for the d-spacings diffsims reports.
_NM_PER_ANGSTROM = 0.1
#: Elements of the ``(directions, bands)`` table :meth:`KinematicMaster.sample`
#: holds at once, so a whole dictionary chunk never materializes it.
_SAMPLE_BLOCK_ELEMENTS = 1 << 24


@dataclass
class Reflectors:
    """The diffracting planes a kinematic pattern is drawn from.

    Attributes
    ----------
    normals
        ``(N, 3)`` unit plane normals in the crystal's Cartesian frame (``a`` along
        x, ``c*`` along z, as orix and :class:`~de_ebsd.crystal.Crystal` both use),
        one per Friedel pair: ``+g`` and ``-g`` are one band.
    weights
        ``(N,)`` relative band intensity.
    widths
        ``(N,)`` band half-width as ``v . n``, the sine of the angle between a
        direction and the plane: the Bragg angle ``lambda / 2d`` for a real phase.
    hkl
        ``(N, 3)`` Miller indices, for labels; optional.
    """

    normals: np.ndarray
    weights: np.ndarray
    widths: np.ndarray
    hkl: np.ndarray | None = None

    def __post_init__(self):
        self.normals = np.asarray(self.normals, float).reshape(-1, 3)
        n = len(self.normals)
        self.weights = np.broadcast_to(np.asarray(self.weights, float), (n,)).copy()
        self.widths = np.broadcast_to(np.asarray(self.widths, float), (n,)).copy()
        if self.hkl is not None:
            self.hkl = np.asarray(self.hkl).reshape(n, 3)

    def __len__(self) -> int:
        return len(self.normals)

    def brightest(self, n: int | None) -> Reflectors:
        """The *n* strongest bands, in their original order; all of them for ``None``.

        Every reflector of a real phase drawn at once is a grey mesh, so a pattern
        or an overlay keeps the strongest bands, the ones visible in the data.
        """
        if n is None or n >= len(self):
            return self
        keep = np.sort(np.argsort(self.weights)[::-1][: max(1, int(n))])
        hkl = None if self.hkl is None else self.hkl[keep]
        return Reflectors(self.normals[keep], self.weights[keep], self.widths[keep], hkl)


def cubic_reflectors() -> Reflectors:
    """The {111}, {200} and {220} bands of a generic cubic crystal, 13 in all.

    The band set used when no structure is loaded. The widths are wider than a real
    20 keV pattern's on purpose: they keep the bands visible on a detector of a few
    tens of pixels, which is what this set is for.
    """
    families = [((1, 1, 1), 1.00), ((2, 0, 0), 0.70), ((2, 2, 0), 0.45)]
    normals, weights, hkls = [], [], []
    for (h, k, l), weight in families:  # noqa: E741
        seen: set[tuple[int, int, int]] = set()
        for perm in {(h, k, l), (h, l, k), (k, h, l), (k, l, h), (l, h, k), (l, k, h)}:
            for signs in np.ndindex(2, 2, 2):
                v = tuple(int(p * (1 - 2 * s)) for p, s in zip(perm, signs, strict=True))
                if tuple(-x for x in v) not in seen:
                    seen.add(v)
        for v in sorted(seen):
            g = np.array(v, float)
            normals.append(g / np.linalg.norm(g))
            weights.append(weight / np.linalg.norm(g))  # wider d, wider and brighter band
            hkls.append(v)
    weights = np.array(weights)
    widths = 0.055 * (weights / weights.max()) + 0.012
    return Reflectors(np.array(normals), weights, widths, np.array(hkls))


def reflectors_from_phase(phase, *, dmin=0.07, energy=20e3, max_bands=60) -> Reflectors:
    """The bands of a real crystal structure: ``|F_hkl|`` weights, Bragg-angle widths.

    Parameters
    ----------
    phase
        An orix ``Phase`` with a structure (lattice in Angstrom, as from
        :func:`~de_ebsd.index.io.phase_from_cif`), or a de_ebsd
        :class:`~de_ebsd.crystal.Crystal` (converted with
        :func:`~de_ebsd.index.io.phase_from_crystal`).
    dmin
        Smallest d-spacing kept, in nm.
    energy
        Beam energy in eV, for the Bragg angle ``lambda / 2d``.
    max_bands
        Keep only the strongest this many (:meth:`Reflectors.brightest`); ``None``
        keeps all.

    Returns
    -------
    Reflectors
        One band per Friedel pair. Higher orders along one normal, such as (200)
        and (400), are separate bands: the higher order is narrower. A phase with no
        structure factors (a ``Phase(space_group=...)`` with no atoms) gives
        :func:`cubic_reflectors`, with a log message saying so.
    """
    from diffsims.crystallography import ReciprocalLatticeVector

    if not hasattr(phase, "point_group"):
        from de_ebsd.index.io import phase_from_crystal

        phase = phase_from_crystal(phase)
    try:
        rlv = ReciprocalLatticeVector.from_min_dspacing(phase, float(dmin) / _NM_PER_ANGSTROM)
        rlv = rlv.unique(use_symmetry=True).symmetrise()
        rlv.sanitise_phase()
        rlv.calculate_structure_factor()
        # the structure factor is complex: its modulus, never its real part
        factor = np.abs(np.asarray(rlv.structure_factor)).astype(float).reshape(-1)
        normals = np.asarray(rlv.unit.data, float).reshape(-1, 3)
        d = np.asarray(rlv.dspacing, float).reshape(-1) * _NM_PER_ANGSTROM
        hkl = np.rint(np.asarray(rlv.hkl, float).reshape(-1, 3)).astype(int)
    except Exception as e:  # diffsims raises a variety on a structure it cannot use
        log.warning(
            "reflectors for %s could not be computed (%s); using the generic cubic set",
            getattr(phase, "name", phase),
            e,
        )
        return cubic_reflectors()

    usable = np.isfinite(factor) & np.isfinite(d) & (d > 0) & np.isfinite(normals).all(1)
    if not usable.any() or float(factor[usable].max()) <= 0:
        log.info(
            "phase %s has no structure factors; using the generic cubic set",
            getattr(phase, "name", phase),
        )
        return cubic_reflectors()
    # extinct reflections go before the Friedel reduction: for fcc the extinct (100)
    # shares a line with the real (200), and must not stand in for it
    usable &= factor > 1e-6 * float(factor[usable].max())
    normals, factor, d, hkl = normals[usable], factor[usable], d[usable], hkl[usable]

    keep = _one_per_friedel_pair(hkl)
    normals, factor, d, hkl = normals[keep], factor[keep], d[keep], hkl[keep]
    wavelength = electron_wavelength_nm(float(energy) / 1e3)
    widths = np.clip(wavelength / (2.0 * d), 1e-4, 0.25)
    weights = factor / float(factor.max())
    return Reflectors(normals, weights, widths, hkl).brightest(max_bands)


def _one_per_friedel_pair(hkl: np.ndarray) -> np.ndarray:
    """Indices of the first of each ``{hkl, -hkl}`` pair (and of each repeat)."""
    seen: set[tuple[int, ...]] = set()
    keep = []
    for i, h in enumerate(hkl):
        key = tuple(int(x) for x in h)
        if key in seen or tuple(-x for x in key) in seen:
            continue
        seen.add(key)
        keep.append(i)
    return np.asarray(keep, dtype=int)


class KinematicMaster(SphereProjection):
    """Kinematic band intensity over the sphere, used wherever a master pattern is.

    The intensity along a crystal-frame unit vector ``v`` is
    ``sum_i weight_i * exp(-(v . n_i / width_i)^2 / 2)``. Everything a
    :class:`~de_ebsd.sim.MasterPattern` does through its intensity on the sphere it
    does too, with the projection inherited from
    :class:`~de_ebsd.sim.masterpattern.SphereProjection`:
    :meth:`project` onto a detector (differentiable in the orientation),
    :meth:`to_equirectangular` for a sphere texture, a
    :class:`~de_ebsd.index.dictionary.DictionaryIndexer` built from it, and
    :func:`~de_ebsd.index.refine.orientation.refine_orientation` against it.

    Parameters
    ----------
    reflectors
        The bands; :func:`cubic_reflectors` when ``None``.
    device
        Torch device; the CPU when ``None``.
    dtype
        Working floating dtype (default ``float32``).
    metadata
        Free-form provenance, as on a :class:`~de_ebsd.sim.MasterPattern`; a de_ebsd
        :class:`~de_ebsd.crystal.Crystal` under ``"crystal"`` gives a
        :class:`~de_ebsd.index.dictionary.DictionaryIndexer` its point group.
    """

    def __init__(self, reflectors=None, *, device=None, dtype=torch.float32, metadata=None):
        self.reflectors = cubic_reflectors() if reflectors is None else reflectors
        self.metadata = dict(metadata or {})
        device = resolve_device(device)
        self._normals = torch.as_tensor(self.reflectors.normals, device=device, dtype=dtype)
        self._weights = torch.as_tensor(self.reflectors.weights, device=device, dtype=dtype)
        self._widths = torch.as_tensor(self.reflectors.widths, device=device, dtype=dtype)

    @property
    def device(self) -> torch.device:
        return self._normals.device

    @property
    def dtype(self) -> torch.dtype:
        return self._normals.dtype

    def to(self, device) -> KinematicMaster:
        """A copy on another device."""
        return KinematicMaster(
            self.reflectors, device=device, dtype=self.dtype, metadata=self.metadata
        )

    def sample(self, directions, energy_kev=None) -> torch.Tensor:
        """The band intensity at crystal-frame unit vectors ``(..., 3)``, shaped ``(...)``.

        *energy_kev* is accepted for the master-pattern interface and ignored: the
        widths already carry the energy. Evaluated in blocks of directions, so the
        ``(directions, bands)`` table stays bounded, and differentiable in
        *directions*.
        """
        directions = torch.as_tensor(directions).to(self.device, self.dtype)
        lead = directions.shape[:-1]
        flat = directions.reshape(-1, 3)
        block = max(1, _SAMPLE_BLOCK_ELEMENTS // max(1, len(self.reflectors)))
        parts = [
            torch.exp(-0.5 * ((flat[i : i + block] @ self._normals.T) / self._widths) ** 2)
            @ self._weights
            for i in range(0, flat.shape[0], block)
        ]
        return torch.cat(parts).reshape(lead)

    def __repr__(self) -> str:
        return f"KinematicMaster({len(self.reflectors)} bands, device={self.device.type})"


def _detector_normals(orientation, reflectors, detector) -> np.ndarray:
    """The plane normals of one orientation in the detector frame, ``(N, 3)``."""
    q = _as_quaternions(orientation).to(torch.float64)
    if q.shape[0] != 1:
        raise ValueError(f"expected one orientation, got {q.shape[0]}")
    g = quaternion_to_matrix(q[0]).numpy()  # sample to crystal
    to_detector = sample_to_detector_matrix(
        detector.sample_tilt, detector.tilt, detector.azimuthal, detector.twist
    )
    # crystal -> sample is g.T, sample -> detector is to_detector; row vectors
    return reflectors.normals @ g @ to_detector.T


def _pixels(detector, gnomonic: np.ndarray) -> np.ndarray:
    """Gnomonic ``(..., 2)`` to ``(..., 2)`` detector pixels ``(col, row)``."""
    ray = np.concatenate([gnomonic, np.ones(gnomonic.shape[:-1] + (1,))], axis=-1)
    row, col = pixels_from_rays(detector, np.asarray(detector.pattern_center, float), ray)
    return np.stack([col, row], axis=-1)


def band_lines(orientation, reflectors=None, detector=None, *, max_bands=None):
    """The band center lines one orientation casts on the detector.

    The center of a band is where the ray is perpendicular to the plane normal,
    ``r . n = 0``, which is a straight line on the flat detector; the bands'
    diffuse edges are not drawn.

    Parameters
    ----------
    orientation
        One orientation: an orix ``Rotation`` of size 1 or a quaternion ``(4,)``.
    reflectors
        The bands; :func:`cubic_reflectors` when ``None``.
    detector
        An :class:`~de_ebsd.sim.detector.EBSDDetector`; the default detector when
        ``None``.
    max_bands
        Keep the brightest this many of the bands that cross the detector.

    Returns
    -------
    segments : numpy.ndarray
        ``(M, 2, 2)`` float32 endpoints ``[[x0, y0], [x1, y1]]`` in detector pixels,
        ``x`` the column and ``y`` the row, with pixel centers on integers (the
        detector spans ``-0.5 ... size - 0.5``). Only bands that cross the detector.
    weights : numpy.ndarray
        ``(M,)`` float32, each segment's band weight.
    """
    from de_ebsd.sim.detector import EBSDDetector

    refl = cubic_reflectors() if reflectors is None else reflectors
    detector = EBSDDetector() if detector is None else detector
    normals = _detector_normals(orientation, refl, detector)
    bounds = detector.gnomonic_bounds
    segments, weights = [], []
    for normal, weight in zip(normals, refl.weights, strict=True):
        segment = _clip_line(normal, bounds)  # n . (x_g, y_g, 1) = 0
        if segment is not None:
            segments.append(_pixels(detector, segment))
            weights.append(weight)
    segments = np.asarray(segments, np.float32).reshape(-1, 2, 2)
    weights = np.asarray(weights, np.float32)
    if max_bands is not None and len(weights) > max_bands:
        brightest = np.argsort(weights)[::-1][: int(max_bands)]
        segments, weights = segments[brightest], weights[brightest]
    return segments, weights


def zone_axis_points(orientation, reflectors=None, detector=None, *, max_axes=12, min_bands=3):
    """Where band lines meet: the zone axes on the detector, ``(M, 2)`` pixels.

    Any two plane normals share the zone axis ``n_i x n_j``; an axis that at least
    *min_bands* planes share (``min_bands * (min_bands - 1) / 2`` pairs agree on it)
    is a real junction, one that only a single pair gives is two lines crossing.
    The *max_axes* most shared ones that point at the detector are returned, as
    ``(x, y)`` = (column, row) in the pixel convention of :func:`band_lines`.
    """
    from de_ebsd.sim.detector import EBSDDetector

    refl = cubic_reflectors() if reflectors is None else reflectors
    detector = EBSDDetector() if detector is None else detector
    normals = _detector_normals(orientation, refl, detector)
    empty = np.zeros((0, 2), np.float32)
    if len(normals) < 2:
        return empty
    i, j = np.triu_indices(len(normals), k=1)
    axes = np.cross(normals[i], normals[j])
    norm = np.linalg.norm(axes, axis=1)
    axes = axes[norm > 1e-6] / norm[norm > 1e-6, None]
    if not len(axes):
        return empty
    axes *= np.sign(axes[:, 2:3] + 1e-30)  # one hemisphere: the detector's

    axes = axes[np.lexsort((axes[:, 2], axes[:, 1], axes[:, 0]))]
    unique: list[np.ndarray] = []
    counts: list[int] = []
    for v in axes:
        for k, u in enumerate(unique):
            if float(v @ u) > 0.9995:
                counts[k] += 1
                break
        else:
            unique.append(v)
            counts.append(1)
    unique, counts = np.asarray(unique), np.asarray(counts)
    shared = counts >= max(1, min_bands * (min_bands - 1) // 2)
    if not shared.any():
        return empty
    unique, counts = unique[shared], counts[shared]
    unique = unique[np.argsort(counts)[::-1][: int(max_axes)]]
    unique = unique[unique[:, 2] > 1e-6]  # behind the detector is not on it
    points = _pixels(detector, unique[:, :2] / unique[:, 2:3])
    rows, cols = detector.nrows, detector.ncols
    x, y = points[:, 0], points[:, 1]
    inside = (x >= -0.5) & (x <= cols - 0.5) & (y >= -0.5) & (y <= rows - 0.5)
    return points[inside].astype(np.float32)
