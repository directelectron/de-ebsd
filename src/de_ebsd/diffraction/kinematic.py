"""Lightweight kinematic STEM orientation patterns.

This module contains the geometry used by the orientation explorer.  It is
deliberately independent of the master-pattern and Bloch-wave code: a
reflection is represented by a reciprocal-lattice vector, a Kikuchi line is a
gnomonic trace of its plane, and a diffraction spot is an Ewald-sphere
intersection.  Diffsims is used only for the elastic structure-factor
intensity calculation through :class:`_DiffsimsElasticScatteringAdapter`.

All public geometry in this module uses nm and nm :sup:`-1`.  The adapter is
the only place where conversion to diffsims' Angstrom convention occurs.
Quaternions are scalar-first ``(w, x, y, z)`` and rotate sample/beam-frame
vectors into the crystal frame, matching :meth:`MasterPattern.project`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import Any

import numpy as np

from de_ebsd.core.constants import electron_wavelength_nm

__all__ = [
    "KinematicPattern",
    "KinematicPatternModel",
    "orientation_from_zone_axis",
]

_ABSENCE_TOL = 1.0e-8
_GEOMETRY_TOL = 1.0e-10


@dataclass(frozen=True)
class KinematicPattern:
    """One projected kinematic pattern.

    ``lines`` contains clipped line segments in gnomonic slope coordinates,
    with shape ``(L, 2, 2)`` and the corresponding Miller indices in
    ``line_hkl``.  ``spots`` and ``spot_hkl`` contain only reflections that
    intersect the requested Ewald-sphere excitation window and the displayed
    square.  ``intensities`` are relative values normalized to the strongest
    *non-direct* spot.  The direct beam is represented separately at
    ``direct_beam`` so it cannot accidentally affect that normalization.
    """

    lines: np.ndarray
    line_hkl: np.ndarray
    spots: np.ndarray
    spot_hkl: np.ndarray
    intensities: np.ndarray
    direct_beam: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    excitation_errors: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False
    )


class _DiffsimsElasticScatteringAdapter:
    """Use diffsims' Lobato factors with per-site occupancies and thermal motion.

    Geometry stays in the native crystal frame. Only reciprocal lengths and
    isotropic B factors cross the Angstrom unit boundary. Fractional atom
    positions and Miller indices are basis duals, independent of Cartesian
    conventions. The full symmetry-expanded cell supplies basis extinctions.
    """

    def __init__(self, crystal: Any) -> None:
        from diffsims.utils.sim_utils import (
            get_atomic_scattering_factors,
            get_scattering_params_dict,
        )

        self._scattering = get_atomic_scattering_factors
        parameters = get_scattering_params_dict("lobato")
        atoms = [
            (site, position)
            for site, positions in zip(crystal.sites, crystal.positions, strict=True)
            for position in positions
        ]
        if not atoms:
            raise ValueError("kinematic patterns require at least one atom")
        missing = sorted({site.element for site, _ in atoms} - parameters.keys())
        if missing:
            raise ValueError(f"Lobato scattering factors unavailable for {', '.join(missing)}")
        self._positions = np.asarray([position for _, position in atoms], dtype=float)
        self._occupancies = np.asarray([site.occupancy for site, _ in atoms], dtype=float)
        self._b_angstrom2 = np.asarray([site.debye_waller * 100 for site, _ in atoms])
        self._coeffs = np.asarray([parameters[site.element] for site, _ in atoms])
        if (not np.isfinite(self._positions).all()
                or not np.isfinite(self._occupancies).all()
                or not np.isfinite(self._b_angstrom2).all()
                or np.any((self._occupancies < 0) | (self._occupancies > 1))
                or np.any(self._b_angstrom2 < 0)):
            raise ValueError("atoms require finite positions, occupancy in [0, 1], and B >= 0")

    def elastic_intensities(self, hkl: np.ndarray, g_nm: np.ndarray) -> np.ndarray:
        """Return coherent elastic |F_g| squared, in bounded reflection batches."""
        hkl = np.asarray(hkl, dtype=np.int64)
        g_angstrom = np.asarray(g_nm, dtype=float) * 0.1
        result = np.empty(len(hkl), dtype=float)
        for start in range(0, len(hkl), 256):
            take = slice(start, start + 256)
            g2 = g_angstrom[take] ** 2
            atomic = self._scattering(g2, self._coeffs, "lobato")
            phase = np.exp(2j * np.pi * (hkl[take] @ self._positions.T))
            dw = np.exp(-0.25 * g2[:, None] * self._b_angstrom2)
            amplitudes = np.sum(atomic * self._occupancies * phase * dw, axis=1)
            result[take] = np.abs(amplitudes) ** 2
        return result


def _validate_energy(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"energy_kev must be finite and > 0, got {value!r}")
    return value


def _validate_dmin(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"dmin must be finite and > 0 nm, got {value!r}")
    return value


def _as_rotation_matrix(orientation: Any) -> np.ndarray:
    """Convert one scalar-first quaternion (or proper 3x3 matrix) to ``R``."""

    data = getattr(orientation, "data", orientation)
    q_or_r = np.asarray(data, dtype=np.float64)
    if q_or_r.shape == (3, 3):
        if not np.all(np.isfinite(q_or_r)):
            raise ValueError("orientation matrix must contain finite values")
        if not np.allclose(q_or_r.T @ q_or_r, np.eye(3), atol=1e-7, rtol=1e-7):
            raise ValueError("orientation matrix must be orthonormal")
        if not np.isclose(np.linalg.det(q_or_r), 1.0, atol=1e-7, rtol=1e-7):
            raise ValueError("orientation matrix must have determinant +1")
        return q_or_r
    if q_or_r.shape == (1, 4):
        q_or_r = q_or_r[0]
    if q_or_r.shape != (4,):
        raise ValueError(
            "orientation must be one scalar-first quaternion with shape (4,) "
            "or a proper rotation matrix with shape (3, 3)"
        )
    if not np.all(np.isfinite(q_or_r)):
        raise ValueError("orientation quaternion must contain finite values")
    norm = float(np.linalg.norm(q_or_r))
    if norm <= _GEOMETRY_TOL:
        raise ValueError("orientation quaternion must have nonzero norm")
    w, x, y, z = q_or_r / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper matrix to a scalar-first quaternion."""

    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0))
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / max(s, _GEOMETRY_TOL)
            z = (m[0, 2] + m[2, 0]) / max(s, _GEOMETRY_TOL)
            w = (m[2, 1] - m[1, 2]) / max(s, _GEOMETRY_TOL)
        elif i == 1:
            s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0))
            x = (m[0, 1] + m[1, 0]) / max(s, _GEOMETRY_TOL)
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / max(s, _GEOMETRY_TOL)
            w = (m[0, 2] - m[2, 0]) / max(s, _GEOMETRY_TOL)
        else:
            s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0))
            x = (m[0, 2] + m[2, 0]) / max(s, _GEOMETRY_TOL)
            y = (m[1, 2] + m[2, 1]) / max(s, _GEOMETRY_TOL)
            z = 0.25 * s
            w = (m[1, 0] - m[0, 1]) / max(s, _GEOMETRY_TOL)
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def orientation_from_zone_axis(crystal: Any, uvw: Any, roll_deg: float = 0.0) -> np.ndarray:
    """Return the orientation that places direct-lattice direction ``uvw`` on +z.

    ``uvw`` is interpreted in the direct lattice basis, then converted with
    ``Crystal.direct_structure_matrix``.  This distinction is essential for
    non-cubic cells, where treating ``uvw`` as Cartesian gives the wrong zone.
    ``roll_deg`` rotates the screen x/y axes around the resulting beam axis.
    """

    uvw = np.asarray(uvw, dtype=np.float64)
    if uvw.shape != (3,) or not np.all(np.isfinite(uvw)):
        raise ValueError("uvw must be a finite three-component direct-lattice direction")
    direct = uvw @ np.asarray(crystal.direct_structure_matrix, dtype=np.float64).T
    norm = float(np.linalg.norm(direct))
    if norm <= _GEOMETRY_TOL:
        raise ValueError("uvw must not be the zero direction")
    beam = direct / norm
    roll = float(roll_deg)
    if not np.isfinite(roll):
        raise ValueError("roll_deg must be finite")

    # Choose a deterministic in-plane basis.  The e_y reference makes [001]
    # return the identity orientation, which is convenient for the explorer.
    ref = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(ref, beam))) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(ref, beam)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(beam, x_axis)
    base = np.column_stack((x_axis, y_axis, beam))
    theta = np.deg2rad(roll)
    around_beam = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return _matrix_to_quaternion(base @ around_beam)


def _primitive_hkl(hkl: np.ndarray) -> tuple[int, int, int]:
    """Canonical sign and harmonic representative for a plane family."""

    values = [int(v) for v in hkl]
    divisor = gcd(gcd(abs(values[0]), abs(values[1])), abs(values[2]))
    values = [v // divisor for v in values]
    for v in values:
        if v:
            if v < 0:
                values = [-x for x in values]
            break
    return tuple(values)


def _clip_line(normal: np.ndarray, bound: float) -> np.ndarray | None:
    """Clip ``normal[:2] @ (x,y) + normal[2] == 0`` to a square."""

    nx, ny, nz = (float(v) for v in normal)
    if abs(nx) <= _GEOMETRY_TOL and abs(ny) <= _GEOMETRY_TOL:
        return None
    points: list[tuple[float, float]] = []
    if abs(ny) > _GEOMETRY_TOL:
        for x in (-bound, bound):
            y = -(nx * x + nz) / ny
            if -bound - _GEOMETRY_TOL <= y <= bound + _GEOMETRY_TOL:
                points.append((x, float(np.clip(y, -bound, bound))))
    if abs(nx) > _GEOMETRY_TOL:
        for y in (-bound, bound):
            x = -(ny * y + nz) / nx
            if -bound - _GEOMETRY_TOL <= x <= bound + _GEOMETRY_TOL:
                points.append((float(np.clip(x, -bound, bound)), y))
    unique: list[tuple[float, float]] = []
    for point in points:
        if not any(np.linalg.norm(np.subtract(point, old)) < 1.0e-9 for old in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    # Deterministic endpoint ordering helps artist updates and tests.
    unique.sort(key=lambda p: (p[0], p[1]))
    return np.asarray(unique[:2], dtype=np.float64)


class KinematicPatternModel:
    """Cached reflection geometry and lightweight kinematic STEM projection."""

    def __init__(self, crystal: Any, energy_kev: float = 200, dmin: float = 0.1) -> None:
        if not hasattr(crystal, "g_cartesian") or not hasattr(crystal, "direct_structure_matrix"):
            raise TypeError("crystal must be a de_ebsd Crystal")
        self.crystal = crystal
        self.dmin = _validate_dmin(dmin)
        self._energy_kev = _validate_energy(energy_kev)
        # Enumerate the reciprocal sphere once.  Intensities and absences are
        # also orientation independent, so they are cached on the model.
        self._adapter = _DiffsimsElasticScatteringAdapter(crystal)
        # The reciprocal-space sphere can project farther along a Miller
        # coordinate than its axial reflection in a skew cell (metric terms
        # can cancel).  The direct-metric ellipsoid bound
        # ``|h_i| <= |a_i| / dmin`` is safe for every cell.
        lengths = np.asarray([crystal.a, crystal.b, crystal.c], dtype=float)
        if not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError("lattice lengths must be finite and positive")
        limits = np.ceil(lengths / self.dmin)
        if np.prod(2 * limits + 1) > 2_000_000:
            raise ValueError("reflection search too large for the explorer; increase dmin")
        axes = [np.arange(-int(limit), int(limit) + 1) for limit in limits]
        hh, kk, ll = np.meshgrid(*axes, indexing="ij")
        all_hkl = np.stack((hh, kk, ll), axis=-1).reshape(-1, 3)
        all_g = np.asarray(crystal.g_length(all_hkl), dtype=np.float64)
        keep = all_g > 0.0
        keep &= np.divide(1.0, all_g, out=np.zeros_like(all_g), where=all_g > 0.0) >= self.dmin
        keep &= crystal.allowed_by_centering(all_hkl)
        order = np.argsort(all_g[keep], kind="stable")
        self._all_hkl = all_hkl[keep][order]
        self._all_g = all_g[keep][order]
        self._all_g_cart = np.asarray(crystal.g_cartesian(self._all_hkl), dtype=np.float64)
        all_elastic = self._adapter.elastic_intensities(self._all_hkl, self._all_g)
        allowed = all_elastic > _ABSENCE_TOL
        self._elastic = all_elastic[allowed].copy()
        self.forbidden_hkl = self._all_hkl[~allowed].copy()
        self._hkl = self._all_hkl[allowed].copy()
        self._g = self._all_g[allowed].copy()
        self._g_cart = self._all_g_cart[allowed].copy()

        # Unique primitive planes, retaining the shortest allowed harmonic.
        selected: dict[tuple[int, int, int], int] = {}
        for i, hkl in enumerate(self._hkl):
            selected.setdefault(_primitive_hkl(hkl), i)
        plane_indices = np.asarray(list(selected.values()), dtype=np.int64)
        self._band_hkl = self._hkl[plane_indices].copy()
        band_g = self._g_cart[plane_indices]
        self._band_normals = band_g / np.linalg.norm(band_g, axis=1, keepdims=True)

    @property
    def energy_kev(self) -> float:
        return self._energy_kev

    @energy_kev.setter
    def energy_kev(self, value: float) -> None:
        self._energy_kev = _validate_energy(value)

    @property
    def band_normals(self) -> np.ndarray:
        return self._band_normals.copy()

    @property
    def band_hkl(self) -> np.ndarray:
        return self._band_hkl.copy()

    def project(
        self,
        orientation: Any,
        semiangle_deg: float = 5,
        excitation_width: float = 0.1,
    ) -> KinematicPattern:
        """Project centerlines and Ewald-aware spots into a square view."""

        semiangle = float(semiangle_deg)
        width = float(excitation_width)
        if not np.isfinite(semiangle) or not 0.0 < semiangle < 89.9:
            raise ValueError("semiangle_deg must be finite and in (0, 89.9)")
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("excitation_width must be finite and > 0 nm^-1")
        rotation = _as_rotation_matrix(orientation)
        bound = float(np.tan(np.deg2rad(semiangle)))

        # Crystal -> beam/sample frame is R.T; with row vectors this is g @ R.
        normals_beam = self._band_normals @ rotation
        line_segments: list[np.ndarray] = []
        line_indices: list[np.ndarray] = []
        for normal, hkl in zip(normals_beam, self._band_hkl, strict=True):
            segment = _clip_line(normal, bound)
            if segment is not None:
                line_segments.append(segment)
                line_indices.append(hkl)

        wavelength = float(electron_wavelength_nm(self._energy_kev))
        g_beam = self._g_cart @ rotation
        k_radius = 1.0 / wavelength
        transverse_sq = np.sum(g_beam[:, :2] ** 2, axis=1)
        real_ewald = transverse_sq <= k_radius * k_radius
        expected_z = np.full(len(g_beam), np.nan, dtype=np.float64)
        # With the documented incident beam +z and scattering vector
        # ``g = k_out - k_in``, the forward Ewald intersection has a small
        # negative z component: g_z = sqrt(k²-|g_perp|²) - k.
        # Rationalized form avoids subtracting nearly equal large values.
        expected_z[real_ewald] = -transverse_sq[real_ewald] / (
            k_radius + np.sqrt(k_radius * k_radius - transverse_sq[real_ewald])
        )
        excitation = g_beam[:, 2] - expected_z
        selected = real_ewald & (np.abs(excitation) <= width)
        selected_idx = np.flatnonzero(selected)
        if len(selected_idx):
            transverse = wavelength * g_beam[selected_idx, :2]
            transverse_norm = np.linalg.norm(transverse, axis=1)
            front = transverse_norm < 1.0 - _GEOMETRY_TOL
            selected_idx = selected_idx[front]
            transverse = transverse[front]
            transverse_norm = transverse_norm[front]
            # The spot position is set by the transverse Ewald intersection,
            # not by the off-sphere g_z of a finite-width rel-rod.
            slopes = transverse / np.sqrt(
                np.maximum(1.0 - transverse_norm**2, _GEOMETRY_TOL)
            )[:, None]
            in_view = np.all(np.abs(slopes) <= bound + _GEOMETRY_TOL, axis=1)
            selected_idx = selected_idx[in_view]
            slopes = slopes[in_view]
        else:
            slopes = np.empty((0, 2), dtype=np.float64)

        spot_hkl = self._hkl[selected_idx].copy()
        spot_excitation = excitation[selected_idx]
        shape = 1.0 / (1.0 + (spot_excitation / width) ** 2)
        intensities = self._elastic[selected_idx] * shape
        if len(intensities):
            maximum = float(np.max(intensities))
            if maximum > 0.0:
                intensities = intensities / maximum

        return KinematicPattern(
            lines=np.asarray(line_segments, dtype=np.float64).reshape(-1, 2, 2),
            line_hkl=np.asarray(line_indices, dtype=np.int64).reshape(-1, 3),
            spots=slopes,
            spot_hkl=spot_hkl,
            intensities=np.asarray(intensities, dtype=np.float64),
            excitation_errors=np.asarray(spot_excitation, dtype=np.float64),
        )
