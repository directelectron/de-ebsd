"""Interactive, geometric STEM orientation explorer.

The explorer deliberately keeps the display layer small.  The kinematic model
owns reflection generation and this module only projects its result into a
gnomonic canvas, draws a reciprocal-normal sphere, and wires anyplotlib events
to one scalar-first sample-to-crystal quaternion.
"""

# The anyplotlib calls below are intentionally kept visually grouped; several
# pass descriptive labels alongside their geometry.
# ruff: noqa: E501,E701

from __future__ import annotations

import math
from typing import Any

import numpy as np

from de_ebsd.viz.figures import apl

__all__ = ["CrystalExplorer", "orientation_from_zone_axis"]

_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _as_quaternion(value: Any) -> np.ndarray:
    """Return one normalized scalar-first quaternion, rejecting bad rotations."""
    data = getattr(value, "data", value)
    q = np.asarray(data, dtype=np.float64)
    if q.size != 4:
        raise ValueError("orientation must contain exactly one quaternion of shape (4,)")
    q = q.reshape(4)
    if not np.isfinite(q).all():
        raise ValueError("orientation must contain finite values")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("orientation quaternion must have non-zero norm")
    q = q / norm
    # A single sign makes programmatic comparisons and widget state stable.
    return q if q[0] >= 0 else -q


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    a, b, c, d = q
    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to scalar-first quaternion."""
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("rotation matrix must be finite and have shape (3, 3)")
    if not np.allclose(r.T @ r, np.eye(3), atol=2e-8) or np.linalg.det(r) <= 0:
        raise ValueError("rotation matrix must be a proper rotation")
    trace = float(np.trace(r))
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = np.array(
            [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        )
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = 2.0 * math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 0.0))
            q = np.array(
                [
                    (r[2, 1] - r[1, 2]) / s,
                    0.25 * s,
                    (r[0, 1] + r[1, 0]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                ]
            )
        elif i == 1:
            s = 2.0 * math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 0.0))
            q = np.array(
                [
                    (r[0, 2] - r[2, 0]) / s,
                    (r[0, 1] + r[1, 0]) / s,
                    0.25 * s,
                    (r[1, 2] + r[2, 1]) / s,
                ]
            )
        else:
            s = 2.0 * math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 0.0))
            q = np.array(
                [
                    (r[1, 0] - r[0, 1]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                    (r[1, 2] + r[2, 1]) / s,
                    0.25 * s,
                ]
            )
    return _as_quaternion(q)


def _axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    t = math.radians(float(angle_deg))
    c, s = math.cos(t), math.sin(t)
    x, y, z = axis
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def _direction_frame(direction: np.ndarray, roll_deg: float = 0.0) -> np.ndarray:
    """Build R with R[:, 2] equal to *direction* and a deterministic roll."""
    n = np.asarray(direction, dtype=np.float64).reshape(3)
    if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
        raise ValueError("zone axis must be a finite, non-zero 3-vector")
    n = n / np.linalg.norm(n)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ex = ref - np.dot(ref, n) * n
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    base = np.column_stack([ex, ey, n])
    # Positive roll rotates the sample x direction toward sample y on screen.
    return _axis_angle(n, float(roll_deg)) @ base


def _fallback_orientation_from_zone_axis(crystal, uvw, roll_deg: float = 0.0) -> np.ndarray:
    """Return a scalar-first sample-to-crystal quaternion for a direct zone axis.

    ``uvw`` is interpreted in the direct lattice basis, so this also works for
    non-cubic cells.  The resulting orientation maps sample/beam ``+z`` onto
    the Cartesian crystal direction ``u a + v b + w c``.
    """
    uvw = np.asarray(uvw, dtype=np.float64)
    if uvw.size != 3:
        raise ValueError("zone axis must have shape (3,)")
    uvw = uvw.reshape(3)
    if not np.isfinite(uvw).all() or np.linalg.norm(uvw) < 1e-12:
        raise ValueError("zone axis must be a finite, non-zero 3-vector")
    basis = np.asarray(crystal.direct_structure_matrix, dtype=np.float64)
    if basis.shape != (3, 3) or not np.isfinite(basis).all():
        raise ValueError("crystal direct_structure_matrix must be finite with shape (3, 3)")
    direction = basis @ uvw
    return _matrix_to_quaternion(_direction_frame(direction, roll_deg))


def orientation_from_zone_axis(crystal, uvw, roll_deg: float = 0.0) -> np.ndarray:
    """Return the shared physics-layer zone-axis quaternion convention."""
    try:
        from de_ebsd.diffraction.kinematic import orientation_from_zone_axis as helper
    except (ImportError, AttributeError):
        helper = None
    if helper is not None:
        return _as_quaternion(helper(crystal, uvw, roll_deg=roll_deg))
    return _fallback_orientation_from_zone_axis(crystal, uvw, roll_deg)


def _camera_direction(azimuth: float, elevation: float) -> np.ndarray:
    """Near-facing sphere direction for anyplotlib's Plot3D camera convention."""
    a, e = np.deg2rad([float(azimuth), float(elevation)])
    return np.array([math.cos(e) * math.sin(a), -math.cos(e) * math.cos(a), math.sin(e)])


def _camera_angles(direction: np.ndarray) -> tuple[float, float]:
    n = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    return float(np.rad2deg(np.arctan2(n[0], -n[1]))), float(
        np.rad2deg(np.arcsin(np.clip(n[2], -1, 1)))
    )


def _camera_frame(azimuth: float, elevation: float) -> np.ndarray:
    """The exact right/up/facing basis used by anyplotlib's Plot3D renderer."""
    a, e = np.deg2rad([float(azimuth), float(elevation)])
    ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    facing = np.array([ce * sa, -ce * ca, se])
    right = np.array([ca, sa, 0.0])
    up = np.array([-se * sa, se * ca, ce])
    return np.column_stack([right, up, facing])


def _band_path(normals: np.ndarray) -> np.ndarray:
    """Pack reciprocal-plane great circles into a dense native point path.

    Plot3D has one connected line primitive and no pen-lift/NaN contract.  A
    fine point cloud preserves each great circle without drawing false chords
    between unrelated planes.
    """
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    rings = []
    t = np.linspace(0.0, 2.0 * np.pi, 256)
    for normal in normals:
        n = normal / max(np.linalg.norm(normal), 1e-15)
        if not np.isfinite(n).all():
            continue
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = ref - np.dot(ref, n) * n
        u /= max(np.linalg.norm(u), 1e-15)
        v = np.cross(n, u)
        rings.append(np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
    if not rings:
        return np.zeros((2, 3), dtype=float)
    return np.concatenate(rings, axis=0)


class CrystalExplorer:
    """Linked STEM orientation explorer with geometric bands and kinematic spots."""

    def __init__(
        self,
        crystal,
        energy_kev: float = 200.0,
        dmin: float = 0.1,
        *,
        semiangle_deg: float = 5.0,
        excitation_width: float = 0.1,
        orientation=None,
        model=None,
        show_lines: bool = True,
        show_spots: bool = True,
        show_labels: bool = True,
        name: str | None = None,
    ):
        self.crystal = crystal
        self.name = name or "STEM crystal explorer"
        self._energy_kev = self._positive(energy_kev, "energy_kev")
        self._dmin = self._positive(dmin, "dmin")
        self._semiangle_deg = self._fov(semiangle_deg)
        self._excitation_width = self._positive(excitation_width, "excitation_width")
        self.model = model
        if self.model is None:
            from de_ebsd.diffraction.kinematic import KinematicPatternModel

            self.model = KinematicPatternModel(
                crystal, energy_kev=self._energy_kev, dmin=self._dmin
            )
        self._model_dmin = self._dmin
        self._orientation = _as_quaternion(_IDENTITY if orientation is None else orientation)
        self._base_orientation = self._orientation.copy()
        self._zone_axis: np.ndarray | None = None
        self._roll_deg = 0.0
        self._roll_deg_before = 0.0
        self._tilt_x_deg = 0.0
        self._tilt_y_deg = 0.0
        self._show_lines = bool(show_lines)
        self._show_spots = bool(show_spots)
        self._show_labels = bool(show_labels)
        self._figure = None
        self._projection_ax = self._sphere_ax = self._crystal_ax = None
        self._line_artist = self._spot_artist = self._beam_artist = self._label_artist = None
        self._sphere_artist = self._crystal_artist = None
        self._last_result = None
        self._updating = False
        self._camera = (0.0, 90.0)
        self._initial_energy_kev = self._energy_kev
        self._initial_dmin = self._dmin
        self._initial_semiangle_deg = self._semiangle_deg
        self._initial_show_lines = self._show_lines
        self._initial_show_spots = self._show_spots
        self._initial_show_labels = self._show_labels
        self._initial_orientation = self._orientation.copy()
        self._initial_roll_deg = self._roll_deg
        self._initial_tilt = (self._tilt_x_deg, self._tilt_y_deg)

    @staticmethod
    def _positive(value, name):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0")
        return value

    @staticmethod
    def _fov(value):
        value = float(value)
        if not np.isfinite(value) or value <= 0 or value >= 89:
            raise ValueError("semiangle_deg must be finite and in (0, 89)")
        return value

    @property
    def orientation(self) -> np.ndarray:
        return self._orientation.copy()

    @property
    def energy_kev(self) -> float:
        return self._energy_kev

    @property
    def dmin(self) -> float:
        return self._dmin

    @property
    def semiangle_deg(self) -> float:
        return self._semiangle_deg

    @property
    def excitation_width(self) -> float:
        return self._excitation_width

    @property
    def roll_deg(self) -> float:
        return self._roll_deg

    @property
    def fine_tilt(self) -> tuple[float, float]:
        return self._tilt_x_deg, self._tilt_y_deg

    @property
    def zone_axis(self):
        return None if self._zone_axis is None else self._zone_axis.copy()

    @property
    def show_lines(self) -> bool:
        return self._show_lines

    @property
    def show_spots(self) -> bool:
        return self._show_spots

    @property
    def show_labels(self) -> bool:
        return self._show_labels

    @property
    def figure(self):
        if self._figure is None:
            self._make_figure()
        return self._figure

    @property
    def figures(self) -> dict[str, object]:
        return {"explorer": self.figure}

    @property
    def result(self):
        """Most recent model projection result, useful to host integrations."""
        _ = self.figure
        return self._last_result

    def _make_figure(self):
        fig, axes = apl().subplots(1, 3, figsize=(900, 410), width_ratios=[1.35, 1.0, 1.0])
        self._figure = fig
        self._projection_ax, self._sphere_ax, self._crystal_ax = np.ravel(axes)[:3]
        bound = math.tan(math.radians(self._semiangle_deg))
        self._projection_ax = self._projection_ax.axes2d(
            xlim=(-bound, bound),
            ylim=(-bound, bound),
            aspect="equal",
            units="gnomonic slope",
            y_units="gnomonic slope",
        )
        self._projection_ax.set_xlabel("x/z (right)")
        self._projection_ax.set_ylabel("y/z (up)")
        self._projection_ax.set_title("Kikuchi lines + kinematic spots")

        normals = np.asarray(
            getattr(self.model, "band_normals", np.empty((0, 3))), dtype=float
        ).reshape(-1, 3)
        band_path = _band_path(normals)
        self._sphere_artist = self._sphere_ax.scatter3d(
            band_path[:, 0],
            band_path[:, 1],
            band_path[:, 2],
            color="#48d1cc",
            point_size=1.4,
            bounds=((-1, 1),) * 3,
            x_label="crystal x",
            y_label="crystal y",
            z_label="crystal z",
            azimuth=0,
            elevation=90,
            gpu="auto",
        )
        self._sphere_artist.set_sphere(radius=1, color="#9aa4b2", alpha=0.12, wireframe=True)
        self._sphere_artist.set_title("Band-line sphere (drag to tilt)")
        # CallbackRegistry.connect accepts bound methods directly.  The
        # decorator-style anyplotlib API annotates callbacks and therefore
        # cannot annotate a bound method on Python 3.12.
        self._sphere_artist.callbacks.connect("pointer_move", self._on_sphere_event)
        self._sphere_artist.callbacks.connect("pointer_up", self._on_sphere_event)
        # Draw each reciprocal-plane great circle as a separate native 3-D
        # line.  Keeping these static means camera drags never regenerate the
        # reflection set and the sphere remains useful when lines are hidden
        # in the central projection.

        cart = self._crystal_cartesian()
        self._crystal_artist = self._crystal_ax.scatter3d(
            cart[:, 0] if len(cart) else [],
            cart[:, 1] if len(cart) else [],
            cart[:, 2] if len(cart) else [],
            color="#56b4e9",
            point_size=6,
            bounds=self._crystal_bounds(cart),
            x_label="crystal x (nm)",
            y_label="crystal y (nm)",
            z_label="crystal z (nm)",
            azimuth=-55,
            elevation=24,
            gpu="auto",
        )
        self._crystal_artist.set_title("Crystal (linked orientation)")
        self._recompute(push=False)

    def _crystal_bounds(self, cart):
        if len(cart) == 0:
            return ((-1, 1),) * 3
        radius = max(float(np.max(np.linalg.norm(cart, axis=1))) * 1.12, 1e-6)
        return ((-radius, radius),) * 3

    def _crystal_cartesian(self):
        parts = [np.asarray(p, dtype=float) for p in getattr(self.crystal, "positions", ())]
        if not parts:
            return np.empty((0, 3))
        frac = np.concatenate(parts, axis=0)
        return frac @ np.asarray(self.crystal.direct_structure_matrix, dtype=float).T

    def _apply_tilts(self):
        rx = _axis_angle(np.array([1.0, 0.0, 0.0]), self._tilt_x_deg)
        ry = _axis_angle(np.array([0.0, 1.0, 0.0]), self._tilt_y_deg)
        return self._base_orientation_matrix @ rx @ ry

    @property
    def _base_orientation_matrix(self):
        return _quaternion_to_matrix(self._base_orientation)

    def _recompute(self, *, push=True):
        if self._updating:
            return
        self._updating = True
        try:
            self._orientation = _matrix_to_quaternion(self._apply_tilts())
            self._last_result = self.model.project(
                self._orientation,
                semiangle_deg=self._semiangle_deg,
                excitation_width=self._excitation_width,
            )
            if self._projection_ax is not None:
                lines = np.asarray(
                    getattr(self._last_result, "lines", np.empty((0, 2, 2))), dtype=float
                ).reshape(-1, 2, 2)
                spots = np.asarray(
                    getattr(self._last_result, "spots", np.empty((0, 2))), dtype=float
                ).reshape(-1, 2)
                intensities = np.asarray(
                    getattr(self._last_result, "intensities", np.ones(len(spots))), dtype=float
                ).reshape(-1)
                if self._line_artist is None:
                    self._line_artist = self._projection_ax.add_lines(
                        lines if self._show_lines else np.empty((0, 2, 2)),
                        name="kikuchi_lines",
                        edgecolors="#48d1cc",
                        linewidths=1.2,
                        label="geometric Kikuchi lines",
                    )
                else:
                    self._line_artist.set(
                        segments=lines if self._show_lines else np.empty((0, 2, 2))
                    )
                sizes = 2.5 + 4.0 * np.sqrt(
                    np.clip(intensities, 0, None)
                    / max(float(np.max(intensities)) if len(intensities) else 1, 1e-12)
                )
                if self._spot_artist is None:
                    self._spot_artist = self._projection_ax.add_circles(
                        spots if self._show_spots else np.empty((0, 2)),
                        name="kinematic_spots",
                        radius=sizes if self._show_spots else np.empty((0,)),
                        facecolors="#f48fb1",
                        edgecolors="#f48fb1",
                        alpha=0.85,
                        labels=[str(x) for x in getattr(self._last_result, "spot_hkl", [])]
                        if self._show_labels
                        else [],
                        label="kinematic spots",
                    )
                else:
                    self._spot_artist.set(
                        offsets=spots if self._show_spots else np.empty((0, 2)),
                        sizes=sizes if self._show_spots else np.empty((0,)),
                        labels=[str(x) for x in getattr(self._last_result, "spot_hkl", [])]
                        if self._show_labels
                        else [],
                    )
                if self._beam_artist is None:
                    self._beam_artist = self._projection_ax.add_circles(
                        [[0.0, 0.0]] if self._show_spots else np.empty((0, 2)),
                        name="direct_beam",
                        radius=3,
                        edgecolors="#ffffff",
                        linewidths=1.0,
                        label="direct beam",
                    )
                self._beam_artist.set(
                    offsets=[[0.0, 0.0]] if self._show_spots else np.empty((0, 2)), sizes=3
                )
                spot_hkl = getattr(self._last_result, "spot_hkl", [])
                label_offsets = (
                    spots if self._show_spots and self._show_labels else np.empty((0, 2))
                )
                label_texts = (
                    ["(" + ",".join(str(int(v)) for v in row) + ")" for row in spot_hkl]
                    if self._show_spots and self._show_labels
                    else []
                )
                if self._label_artist is None:
                    self._label_artist = self._projection_ax.add_texts(
                        label_offsets, label_texts, name="spot_labels", color="#f48fb1", fontsize=10
                    )
                else:
                    self._label_artist.set(offsets=label_offsets, texts=label_texts)
                self._update_linked_views()
        finally:
            self._updating = False

    def _update_linked_views(self):
        if self._crystal_artist is not None:
            cart = self._crystal_cartesian() @ self._quaternion_matrix_inverse().T
            self._crystal_artist.set_data(cart[:, 0], cart[:, 1], cart[:, 2])
        if self._sphere_artist is not None:
            beam = _quaternion_to_matrix(self._orientation)[:, 2]
            self._sphere_artist.set_highlight(
                float(beam[0]), float(beam[1]), float(beam[2]), color="#ffffff", size=8
            )

    def _quaternion_matrix_inverse(self):
        return _quaternion_to_matrix(self._orientation).T

    def set_orientation(self, orientation):
        self._orientation = _as_quaternion(orientation)
        self._base_orientation = self._orientation.copy()
        self._zone_axis = None
        self._tilt_x_deg = self._tilt_y_deg = 0.0
        self._roll_deg = 0.0
        self._roll_deg_before = 0.0
        self._set_camera_for_beam(_quaternion_to_matrix(self._orientation)[:, 2])
        self._recompute()
        return self

    def set_zone_axis(self, uvw, roll_deg: float | None = None):
        uvw = np.asarray(uvw, dtype=float).reshape(-1)
        if uvw.size != 3:
            raise ValueError("zone axis must have shape (3,)")
        self._zone_axis = uvw.copy()
        if roll_deg is not None:
            roll_deg = float(roll_deg)
            if not np.isfinite(roll_deg):
                raise ValueError("roll_deg must be finite")
            self._roll_deg = roll_deg
        self._roll_deg_before = self._roll_deg
        self._base_orientation = orientation_from_zone_axis(self.crystal, uvw, self._roll_deg)
        self._tilt_x_deg = self._tilt_y_deg = 0.0
        self._set_camera_for_beam(_quaternion_to_matrix(self._base_orientation)[:, 2])
        self._recompute()
        return self

    def set_roll(self, roll_deg: float):
        roll_deg = float(roll_deg)
        if not np.isfinite(roll_deg):
            raise ValueError("roll_deg must be finite")
        matrix = _quaternion_to_matrix(self._orientation)
        beam = matrix[:, 2]
        self._roll_deg = roll_deg
        # Plot3D has no camera-roll state.  Apply roll about the current beam
        # directly and fold it into the authoritative orientation, clearing
        # the two convenience tilt offsets so changing roll cannot move +z.
        delta = roll_deg - getattr(self, "_roll_deg_before", 0.0)
        self._orientation = _matrix_to_quaternion(_axis_angle(beam, delta) @ matrix)
        self._base_orientation = self._orientation.copy()
        self._tilt_x_deg = self._tilt_y_deg = 0.0
        self._roll_deg_before = roll_deg
        self._recompute()
        return self

    def set_fine_tilt(self, tilt_x_deg: float = 0.0, tilt_y_deg: float = 0.0):
        tx, ty = float(tilt_x_deg), float(tilt_y_deg)
        if not np.isfinite([tx, ty]).all():
            raise ValueError("fine tilt values must be finite")
        self._tilt_x_deg, self._tilt_y_deg = tx, ty
        self._recompute()
        return self

    set_tilt = set_fine_tilt

    def set_energy(self, energy_kev: float):
        self._energy_kev = self._positive(energy_kev, "energy_kev")
        try:
            self.model.energy_kev = self._energy_kev
        except Exception:
            from de_ebsd.diffraction.kinematic import KinematicPatternModel

            self.model = KinematicPatternModel(
                self.crystal, energy_kev=self._energy_kev, dmin=self._dmin
            )
        self._recompute()
        return self

    def set_dmin(self, dmin: float):
        self._dmin = self._positive(dmin, "dmin")
        from de_ebsd.diffraction.kinematic import KinematicPatternModel

        self.model = KinematicPatternModel(
            self.crystal, energy_kev=self._energy_kev, dmin=self._dmin
        )
        self._model_dmin = self._dmin
        if self._sphere_artist is not None:
            normals = np.asarray(
                getattr(self.model, "band_normals", np.empty((0, 3))), dtype=float
            ).reshape(-1, 3)
            path = _band_path(normals)
            self._sphere_artist.set_data(path[:, 0], path[:, 1], path[:, 2])
        self._recompute()
        return self

    def set_semiangle(self, semiangle_deg: float):
        self._semiangle_deg = self._fov(semiangle_deg)
        if self._projection_ax is not None:
            bound = math.tan(math.radians(self._semiangle_deg))
            self._projection_ax.set_xlim(-bound, bound)
            self._projection_ax.set_ylim(-bound, bound)
        self._recompute()
        return self

    def set_excitation_width(self, excitation_width: float):
        self._excitation_width = self._positive(excitation_width, "excitation_width")
        self._recompute()
        return self

    def set_show_lines(self, visible: bool):
        self._show_lines = bool(visible)
        self._recompute()
        return self

    def set_show_spots(self, visible: bool):
        self._show_spots = bool(visible)
        self._recompute()
        return self

    def set_show_labels(self, visible: bool):
        self._show_labels = bool(visible)
        self._recompute()
        return self

    def reset(self):
        self._energy_kev, self._dmin, self._semiangle_deg = (
            self._initial_energy_kev,
            self._initial_dmin,
            self._initial_semiangle_deg,
        )
        self._show_lines, self._show_spots, self._show_labels = (
            self._initial_show_lines,
            self._initial_show_spots,
            self._initial_show_labels,
        )
        self._zone_axis = None
        self._roll_deg = self._tilt_x_deg = self._tilt_y_deg = 0.0
        self._roll_deg_before = self._initial_roll_deg
        self._base_orientation = self._initial_orientation.copy()
        self._orientation = self._initial_orientation.copy()
        if self._dmin != self._model_dmin:
            from de_ebsd.diffraction.kinematic import KinematicPatternModel

            self.model = KinematicPatternModel(
                self.crystal, energy_kev=self._energy_kev, dmin=self._dmin
            )
            self._model_dmin = self._dmin
        else:
            try:
                self.model.energy_kev = self._energy_kev
            except Exception:
                pass
        if self._projection_ax is not None:
            bound = math.tan(math.radians(self._semiangle_deg))
            self._projection_ax.set_xlim(-bound, bound)
            self._projection_ax.set_ylim(-bound, bound)
        self._set_camera_for_beam(np.array([0.0, 0.0, 1.0]))
        self._recompute()
        return self

    def _set_camera_for_beam(self, beam):
        self._camera = _camera_angles(beam)
        if self._sphere_artist is not None:
            self._sphere_artist.set_view(*self._camera)

    def _on_sphere_event(self, event):
        if self._updating:
            return
        az = getattr(event, "azimuth", None)
        el = getattr(event, "elevation", None)
        if az is None or el is None or not np.isfinite([az, el]).all():
            return
        self._camera = (float(az), float(el))
        beam = _camera_direction(float(az), float(el))
        self._base_orientation = _matrix_to_quaternion(
            _axis_angle(beam, self._roll_deg) @ _camera_frame(float(az), float(el))
        )
        self._zone_axis = None
        self._tilt_x_deg = self._tilt_y_deg = 0.0
        # anyplotlib does not merge camera fields into Plot3D state.  Push the
        # received angles back so the next Python update cannot snap the view.
        self._sphere_artist.set_view(float(az), float(el))
        self._recompute()

    @property
    def controls(self):
        """Notebook controls; values call the public setters immediately."""
        if hasattr(self, "_controls"):
            return self._controls
        try:
            import ipywidgets as widgets
        except ImportError:
            self._controls = None
            return None

        def compact(control):
            control.layout = widgets.Layout(width="145px")
            return control

        energy = compact(widgets.FloatText(value=self.energy_kev, description="keV"))
        fov = compact(widgets.FloatText(value=self.semiangle_deg, description="± angle °"))
        dmin = compact(widgets.FloatText(value=self.dmin, description="dmin nm"))
        zone = compact(widgets.Text(value="", description="zone [uvw]"))
        roll = compact(widgets.FloatText(value=self.roll_deg, description="roll °"))
        tx = compact(widgets.FloatText(value=0.0, description="tilt x °"))
        ty = compact(widgets.FloatText(value=0.0, description="tilt y °"))
        lines = widgets.Checkbox(value=self.show_lines, description="Kikuchi lines")
        spots = widgets.Checkbox(value=self.show_spots, description="Kinematic spots")
        labels = widgets.Checkbox(value=self.show_labels, description="hkl labels")
        error = widgets.Label(value="", layout=widgets.Layout(color="#d55e00", width="100%"))
        reset = widgets.Button(description="Reset", button_style="")

        def observe(change):
            if change["name"] != "value" or self._updating:
                return
            owner = change["owner"]
            try:
                if owner is energy:
                    self.set_energy(change["new"])
                elif owner is fov:
                    self.set_semiangle(change["new"])
                elif owner is dmin:
                    self.set_dmin(change["new"])
                elif owner is roll:
                    self.set_roll(change["new"])
                elif owner is tx or owner is ty:
                    self.set_fine_tilt(tx.value, ty.value)
                elif owner is lines:
                    self.set_show_lines(change["new"])
                elif owner is spots:
                    self.set_show_spots(change["new"])
                elif owner is labels:
                    self.set_show_labels(change["new"])
                elif owner is zone and str(change["new"]).strip():
                    self.set_zone_axis(
                        [float(x) for x in str(change["new"]).replace(",", " ").split()]
                    )
                error.value = ""
            except (TypeError, ValueError) as exc:
                error.value = str(exc)

        def do_reset(_):
            self.reset()
            energy.value, dmin.value, fov.value = self.energy_kev, self.dmin, self.semiangle_deg
            roll.value, tx.value, ty.value = self.roll_deg, self._tilt_x_deg, self._tilt_y_deg
            lines.value, spots.value, labels.value = (
                self.show_lines,
                self.show_spots,
                self.show_labels,
            )
            zone.value, error.value = "", ""

        reset.on_click(do_reset)
        for control in (energy, fov, dmin, zone, roll, tx, ty, lines, spots):
            control.observe(observe)
        self._controls = widgets.VBox(
            [
                widgets.HBox([energy, dmin, fov], layout=widgets.Layout(flex_flow="row wrap")),
                widgets.HBox([zone, roll, tx, ty], layout=widgets.Layout(flex_flow="row wrap")),
                widgets.HBox([lines, spots, labels, reset]),
                error,
            ]
        )
        return self._controls

    def _ipython_display_(self):
        from IPython.display import display

        if self.controls is not None:
            display(self.controls)
        display(self.figure)

    def __repr__(self):
        return f"CrystalExplorer(energy_kev={self.energy_kev:g}, dmin={self.dmin:g}, orientation={self.orientation.tolist()!r})"

    def to_html(self):
        """Export the native figure with a clear live-kernel limitation note."""
        note = (
            "<p style='font:12px system-ui;color:#666'>"
            "This exported view is a static snapshot; Python pattern recalculation "
            "requires a live Jupyter or SpyDE kernel.</p>"
        )
        return "<!doctype html><meta charset='utf-8'>" + note + self.figure.to_html()
