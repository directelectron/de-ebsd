"""The in-plane-factored dictionary search of :class:`~de_ebsd.index.live.LiveIndexer`.

Turning the crystal about the axis through the pattern center normal to the screen
turns its pattern about the pattern center by the same angle. Every orientation is

.. math::

    g = g_0(u)\\, R_n(\\theta),

with :math:`n` the screen normal in the sample frame, :math:`R_n(\\theta)` the turn
by :math:`\\theta` about it, :math:`u = g\\,n` the crystal direction along the
normal (two angles, on the sphere modulo the point group's proper rotations) and
:math:`g_0(u)` the smallest rotation taking :math:`n` to :math:`u`. A pattern drawn
on a polar grid about the pattern center, radius :math:`\\rho_k` by angle
:math:`\\phi_j`, then turns by a shift along :math:`j`: the pattern of
:math:`g_0 R_n(\\theta)` at :math:`\\phi` is :math:`g_0`'s at :math:`\\phi + \\theta`.
So one polar template per sphere direction :math:`u` holds every in-plane angle,
and scoring a pattern against all of them is a correlation along :math:`\\phi`.

The score is the normalized cross-correlation over the screen's visible samples,
weighted by the polar area element :math:`\\rho`. With :math:`w_{kj}` that weight on
the screen and zero off it, :math:`q` the pattern made mean-free and unit over the
visible samples and :math:`t` the template,

.. math::

    \\mathrm{NCC}(u, \\theta) = \\frac{\\sum_{kj} w_{kj}\\, q_{kj}\\, t_{k, j+s}}
    {\\sqrt{D(u, s)}}, \\qquad
    D(u, s) = \\sum_{kj} w_{kj} t_{k,j+s}^2
    - \\frac{\\left(\\sum_{kj} w_{kj} t_{k,j+s}\\right)^2}{\\sum_{kj} w_{kj}},

:math:`s` the shift of :math:`\\theta`. The template's visible variance
:math:`D` is two correlations along :math:`\\phi` -- of the mask with :math:`t` and
with :math:`t^2` -- worked out once per template whenever the mask changes, by FFT.
The numerator, per angular frequency :math:`f`, is a product of the pattern's and
the template's spectra summed over the radii, one real matrix product per kept
frequency over (templates x radii) by (radii x patterns), then one more product
back from the frequencies to the in-plane angles. The best angle's neighbours give
a parabolic sub-step.

The two-angle grid is an icosahedral sampling of the sphere kept where it lies in
the Dirichlet region of the proper rotations about a generic direction (plus half a
step, so a direction on the region's edge still has its neighbours), which tiles
the sphere for every point group alike.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from de_ebsd.core.constants import SINGLE_PRECISION_FLOOR
from de_ebsd.core.geometry import sample_to_detector_matrix
from de_ebsd.core.rotation import quaternion_to_matrix
from de_ebsd.index import _kernels
from de_ebsd.index.live.common import FUSED_BATCH, average_to, pool_into
from de_ebsd.index.live.workspace import InplaneWorkspace

__all__ = [
    "InplaneSearch",
    "PolarGrid",
    "in_plane_rotations",
    "proper_rotations",
    "rotation_taking",
    "screen_frame",
    "search_memory",
    "sphere_grid",
]

#: The sphere grid's step and the in-plane step, as multiples of the search's
#: equivalent misorientation step: the two-angle grid whose 99th-percentile covering
#: radius matches a fundamental-zone grid's at that step.
SPHERE_STEP_FACTOR = 1.25

#: The coarsest in-plane step the search takes, in degrees. On simulated nickel
#: at 12 counts per pixel and a 1-degree step, a 1.875-degree in-plane step missed
#: no more patterns than 1.25 degrees did, with 2/3 of the work.
MIN_THETA_STEP_DEG = 1.875

#: Angular frequencies kept: ``max(MIN_FREQUENCIES, round(FREQUENCY_SCALE / step))``
#: for a step in degrees, 48 at 2 degrees and 64 at 1. More (96, 128 at 1 degree)
#: found the same orientations.
FREQUENCY_SCALE = 64.0
MIN_FREQUENCIES = 48

#: The polar grid's radial step, in search pixels.
RADIAL_STEP = 0.5

#: The fewest polar angles; the count is the in-plane count's smallest multiple at
#: or above it.
MIN_POLAR_ANGLES = 200

#: How far past the screen's farthest corner the polar grid reaches, as a share of
#: that distance, so that a nearby pattern center needs no new grid.
RADIUS_MARGIN = 0.04

#: The most device memory the search's working buffers take by default, and how
#: much of the device's free memory it leaves for everything else (a camera's frame
#: buffers among it). On a Titan X at a 1-degree step and 1,024 patterns a batch,
#: 192 MB searched as fast as 384 MB; 96 MB, chunks of 16 templates, at half speed.
MEMORY_LIMIT = 256 * 2**20
MEMORY_RESERVE = 2**30
MEMORY_FLOOR = 64 * 2**20

#: The fewest templates a chunk of the search scores at once.
MIN_CHUNK = 32

#: A generic direction: its Dirichlet region under any point group's rotations is a
#: fundamental region of the sphere.
_GENERIC_DIRECTION = (0.913, 0.317, 0.2563)


# --- geometry -------------------------------------------------------------------


def screen_frame(detector) -> np.ndarray:
    """``(3, 3)``: the screen's axes as rows in the sample frame, so a detector-frame
    row ``x`` is the sample-frame direction ``x @ frame``; the last row is the screen
    normal."""
    return sample_to_detector_matrix(
        detector.sample_tilt, detector.tilt, detector.azimuthal, detector.twist
    )


def in_plane_rotations(frame: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` turns by *theta* (radians) about the screen normal, in the
    sample frame: ``frame.T @ Rz(theta) @ frame``. The pattern of ``g0 @`` this at
    polar angle ``phi`` about the pattern center is ``g0``'s at ``phi + theta``."""
    frame = frame.to(theta.dtype)
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(theta), torch.ones_like(theta)
    rz = torch.stack([c, -s, z, s, c, z, z, z, o], -1).view(*theta.shape, 3, 3)
    return frame.T @ rz @ frame


def rotation_taking(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(M, 3, 3)``: the smallest rotation taking the unit vector *a* ``(3,)`` to
    each unit vector *b* ``(M, 3)`` (a half turn where they are opposite)."""
    v = torch.linalg.cross(a.expand_as(b), b, dim=1)
    c = b @ a
    k = torch.zeros(len(b), 3, 3, dtype=b.dtype, device=b.device)
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    k = k - k.transpose(1, 2)
    eye = torch.eye(3, dtype=b.dtype, device=b.device)
    out = eye + k + k @ k / (1 + c).clamp_min(1e-12)[:, None, None]
    opposite = c < -1 + 1e-9
    if opposite.any():
        helper = torch.tensor([1.0, 0.0, 0.0], dtype=b.dtype, device=b.device)
        if abs(float(a[0])) > 0.9:
            helper = torch.tensor([0.0, 1.0, 0.0], dtype=b.dtype, device=b.device)
        axis = F.normalize(torch.linalg.cross(a, helper, dim=0), dim=0)
        out[opposite] = 2 * torch.outer(axis, axis) - eye
    return out


def proper_rotations(point_group, device=None) -> torch.Tensor:
    """``(S, 3, 3)`` float64: the rotations that leave a crystal's patterns unchanged,
    the proper part of its Laue group, as crystal-frame matrices (``g ~ S @ g``)."""
    group = point_group.laue.proper_subgroup
    q = torch.as_tensor(np.asarray(group.data), dtype=torch.float64, device=device)
    return quaternion_to_matrix(q)


def sphere_grid(point_group, step_deg: float, margin_deg: float, device=None) -> torch.Tensor:
    """``(U, 3)`` float64: crystal directions of an icosahedral sphere grid at
    *step_deg*, kept where they lie in, or within *margin_deg* of, the Dirichlet
    region of a generic direction ``k`` under the proper rotations ``S``: ``u . k >=
    (S u) . k`` for every ``S``. Any finite group's Dirichlet region of a point it
    moves nowhere is a fundamental region, which suits every point group alike."""
    from orix.sampling import sample_S2

    v = torch.as_tensor(sample_S2(step_deg, method="icosahedral").data, dtype=torch.float64)
    v = F.normalize(v, dim=1).to(device)
    operators = proper_rotations(point_group, device)
    key = F.normalize(torch.tensor(_GENERIC_DIRECTION, dtype=torch.float64, device=device), dim=0)
    eye = torch.eye(3, dtype=torch.float64, device=device)
    identity = int(torch.argmin(((operators - eye) ** 2).sum((1, 2))))
    seen = torch.einsum("sij,mj->msi", operators, v) @ key
    keep = seen[:, identity] >= seen.max(1).values - math.radians(margin_deg)
    return v[keep]


def gnomonic_bounds(pattern_centers, aspect: float) -> np.ndarray:
    """``(N, 4)`` ``[x_min, x_max, y_min, y_max]`` of the screen at each pattern
    center ``(N, 3)``, as :attr:`~de_ebsd.sim.detector.EBSDDetector.gnomonic_bounds`."""
    pcx, pcy, pcz = np.atleast_2d(np.asarray(pattern_centers, float)).T
    return np.stack([-aspect * pcx / pcz, aspect * (1 - pcx) / pcz, -(1 - pcy) / pcz, pcy / pcz], 1)


def farthest_corner(pattern_centers, aspect: float) -> float:
    """The largest gnomonic distance from a pattern center to a corner of its screen,
    over *pattern_centers* ``(N, 3)``."""
    b = gnomonic_bounds(pattern_centers, aspect)
    return float(max(np.hypot(b[:, i], b[:, j]).max() for i in (0, 1) for j in (2, 3)))


def search_memory(device) -> int:
    """The device memory the in-plane search's working buffers may take by default:
    :data:`MEMORY_LIMIT`, or less where the device's free memory less
    :data:`MEMORY_RESERVE` is smaller, and never below :data:`MEMORY_FLOOR`."""
    device = torch.device(device)
    if device.type != "cuda":
        return MEMORY_LIMIT
    free, _ = torch.cuda.mem_get_info(device)
    return int(max(MEMORY_FLOOR, min(MEMORY_LIMIT, free - MEMORY_RESERVE)))


# --- the polar grid ---------------------------------------------------------------


class PolarGrid:
    """A polar grid about the pattern center in gnomonic units: ``K`` radii stepping
    by *step* out to *radius*, by ``J`` = *angles* angles. Its sample directions are
    in the sample frame of a screen with axes *frame*; they do not depend on the
    pattern center, so neither do the templates drawn at them."""

    def __init__(self, frame: torch.Tensor, step: float, radius: float, angles: int, device):
        count = math.ceil(radius / step)
        self.step = step
        self.radius = count * step
        self.radii = (torch.arange(count, dtype=torch.float64, device=device) + 0.5) * step
        phi = 2 * math.pi * torch.arange(angles, dtype=torch.float64, device=device) / angles
        self.cosines, self.sines = torch.cos(phi), torch.sin(phi)
        self.x = self.radii[:, None] * self.cosines[None]
        self.y = self.radii[:, None] * self.sines[None]
        rows = torch.stack([self.x, self.y, torch.ones_like(self.x)], -1).view(-1, 3)
        #: ``(K * J, 3)`` float32 sample-frame directions, radius-major.
        self.directions = F.normalize(rows @ frame.to(device, torch.float64), dim=1).float()

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.radii), len(self.cosines)

    def visible(self, pattern_centers, aspect: float) -> torch.Tensor:
        """``(K, J)`` bool: the samples on the screen at every one of
        *pattern_centers* ``(M, 3)``.

        Each screen edge is a linear inequality in the center -- ``x >= -aspect *
        pcx / pcz`` is ``x pcz + aspect pcx >= 0`` -- so a sample is on the screen
        at every center of a set exactly when it is at every vertex of its convex
        hull: for a scan whose centers are a plane over the beam position, the
        plane's four corners."""
        centers = torch.as_tensor(
            np.atleast_2d(np.asarray(pattern_centers, float)), device=self.x.device
        )
        x, y = self.x[None], self.y[None]
        least = torch.full((4, *self.shape), math.inf, dtype=torch.float64, device=x.device)
        for lo in range(0, len(centers), 64):
            pcx, pcy, pcz = (c[:, None, None] for c in centers[lo : lo + 64].unbind(1))
            margins = torch.stack(
                [
                    (x * pcz + aspect * pcx).amin(0),
                    (aspect * (1 - pcx) - x * pcz).amin(0),
                    (y * pcz + 1 - pcy).amin(0),
                    (pcy - y * pcz).amin(0),
                ]
            )
            least = torch.minimum(least, margins)
        return (least >= 0).all(0)

    def sampling(self, pattern_centers, aspect: float) -> torch.Tensor:
        """``(P, K, J, 2)`` float32: where each sample falls on the screen at each
        pattern center ``(P, 3)``, as ``grid_sample`` coordinates
        (``align_corners=False``)."""
        b = torch.as_tensor(gnomonic_bounds(pattern_centers, aspect), device=self.x.device)
        b = b[:, :, None, None]
        gx = (self.x - b[:, 0]) / (b[:, 1] - b[:, 0]) * 2 - 1
        gy = (b[:, 3] - self.y) / (b[:, 3] - b[:, 2]) * 2 - 1
        return torch.stack([gx, gy], -1).float()


# --- the search -------------------------------------------------------------------


class InplaneSearch:
    """The in-plane-factored search of a :class:`~de_ebsd.index.live.LiveIndexer`
    (``search="inplane"``), which builds it and calls it; see the module.

    Parameters
    ----------
    indexer
        The live indexer: its device, search dtype, detector, search and refine
        shapes, search texture and projection.
    point_groups
        Each phase's orix point group. Every phase's templates are stacked, each
        sampled over its own symmetry, and the best score gives the phase.
    step
        The equivalent misorientation step, in degrees: the fundamental-zone
        dictionary's resolution this search stands in for. The sphere and
        in-plane steps, the kept frequencies and the polar angles follow from it.
    sphere_step, theta_step, frequencies, angles, margin
        Overrides of the sphere grid's step, the in-plane step (degrees), the
        angular frequencies kept, the polar angles (a multiple of the in-plane
        count) and the sphere grid's margin (degrees, half its step by default).
    radial
        The polar grid's radial step, in search pixels.
    substep
        Whether the best in-plane angle is moved to its parabola's vertex.
    memory
        The device memory, in bytes, the working buffers may take; the chunk of
        templates scored at once is the most that fits. :func:`search_memory` by
        default.
    capacity
        The most patterns one call holds; more are searched that many at a time.
    """

    def __init__(
        self,
        indexer,
        point_groups,
        *,
        step: float,
        sphere_step: float | None = None,
        theta_step: float | None = None,
        frequencies: int | None = None,
        angles: int | None = None,
        margin: float | None = None,
        radial: float = RADIAL_STEP,
        substep: bool = True,
        memory: int | None = None,
        capacity: int = FUSED_BATCH,
    ) -> None:
        self.indexer = indexer
        device = indexer.device
        self.device = device
        self.dtype = indexer.dtype
        self.step = float(step)
        self.sphere_step = float(sphere_step or SPHERE_STEP_FACTOR * step)
        self.theta_step = float(theta_step or max(MIN_THETA_STEP_DEG, SPHERE_STEP_FACTOR * step))
        #: In-plane angles searched: ``T``.
        self.thetas = round(360.0 / self.theta_step)
        #: Angular frequencies kept: ``F``.
        self.frequencies = int(frequencies or max(MIN_FREQUENCIES, round(FREQUENCY_SCALE / step)))
        if angles is None:
            angles = self.thetas * math.ceil(MIN_POLAR_ANGLES / self.thetas)
        if angles % self.thetas:
            raise ValueError("the polar angles must be a multiple of the in-plane angles")
        if self.frequencies > angles // 2 + 1:
            raise ValueError("more frequencies than the polar angles hold")
        #: Polar angles: ``J``.
        self.angles = int(angles)
        self.radial = float(radial)
        self.substep = substep
        self.memory = int(search_memory(device) if memory is None else memory)
        self.capacity = int(capacity)
        margin = 0.5 * self.sphere_step if margin is None else margin

        detector = indexer.detector
        self.aspect = float(detector.aspect_ratio)
        self.frame = torch.as_tensor(screen_frame(detector), dtype=torch.float64, device=device)
        self.frame32 = self.frame.float().contiguous()
        normal = self.frame[2]
        axes, phases, self.templates_per_phase = [], [], []
        for number, point_group in enumerate(point_groups):
            directions = sphere_grid(point_group, self.sphere_step, margin, device)
            axes.append(rotation_taking(normal, directions))
            phases.append(torch.full((len(directions),), number, dtype=torch.int64, device=device))
            self.templates_per_phase.append(len(directions))
        #: ``(U, 3, 3)`` float32: each template's orientation at in-plane angle zero.
        self.axes = torch.cat(axes).float().contiguous()
        #: ``(U,)``: each template's phase.
        self.template_phase = torch.cat(phases)
        self._template_phase32 = self.template_phase.to(torch.int32)
        #: Entries per phase: its templates times the in-plane angles.
        self.sizes = [count * self.thetas for count in self.templates_per_phase]
        search_rows, search_cols = indexer.search_shape
        #: The shape a pattern is resampled from: the refine rows averaged to twice the
        #: search shape.
        self.source_shape = (2 * search_rows, 2 * search_cols)
        bounds = detector.gnomonic_bounds
        #: A search pixel, in gnomonic units, at the pattern center built at.
        self.pixel = float(bounds[1] - bounds[0]) / search_cols
        self.scan_centers = None
        #: Called with each stage's name as it is queued, for a benchmark's timing:
        #: ``"start"``, ``"patterns"``, ``"spectra"``, then per chunk ``"products"``,
        #: ``"angles"`` and ``"best"``, and ``"finish"``. None to skip.
        self.marker = None
        self._mask_centers = None
        self._center = torch.empty(3, dtype=torch.float32, device=device)
        self._build(farthest_corner(detector.pattern_center, self.aspect))

    # --- building -----------------------------------------------------------------

    @property
    def template_count(self) -> int:
        """``U``: templates over every phase."""
        return len(self.axes)

    def _build(self, corner: float) -> None:
        """The polar grid out past *corner*, the templates' spectra, the angle and
        frequency matrices and the working buffers."""
        device, dtype = self.device, self.dtype
        self.grid = PolarGrid(
            self.frame, self.radial * self.pixel, corner * (1 + RADIUS_MARGIN), self.angles, device
        )
        k, j = self.grid.shape
        f, t = self.frequencies, self.thetas
        #: Each sample's area weight, its radius, as float32 ``(K, 1)``.
        self._radius_weights = self.grid.radii.float()[:, None]
        self._radii32 = self.grid.radii.float().contiguous()
        self._cosines32 = self.grid.cosines.float().contiguous()
        self._sines32 = self.grid.sines.float().contiguous()
        self.chunk = self._chunk_for(self.memory)
        self._template_chunk = max(1, min(self.template_count, self.memory // (k * j * 4 * 8)))

        #: ``(F, U, 2K)``: each template's kept spectrum, ``[real | imaginary]`` over
        #: the radii, the left operand of the per-frequency products.
        self.spectra = torch.empty(f, self.template_count, 2 * k, dtype=dtype, device=device)
        for lo in range(0, self.template_count, self._template_chunk):
            hi = min(self.template_count, lo + self._template_chunk)
            spectrum = torch.fft.rfft(self._templates(lo, hi), dim=-1)[..., :f]
            self.spectra[:, lo:hi, :k] = spectrum.real.permute(2, 0, 1)
            self.spectra[:, lo:hi, k:] = spectrum.imag.permute(2, 0, 1)

        # (T, 2F): from the kept frequencies back to the in-plane angles, the
        # one-sided spectrum counted twice but for its zero (and Nyquist) term,
        # columns [real | imaginary], with the correlation's 1/J
        shift = torch.arange(t, dtype=torch.float64, device=device)[:, None]
        frequency = torch.arange(f, dtype=torch.float64, device=device)[None]
        twice = torch.where(frequency == 0, 1.0, 2.0)
        if j % 2 == 0:
            twice = torch.where(frequency == j // 2, 1.0, twice)
        angle = 2 * math.pi * frequency * shift / t
        weights = torch.cat([twice * torch.cos(angle), -twice * torch.sin(angle)], 1) / j
        #: ``(T, 2F)`` float32, and in the search dtype for the product.
        self.angle_weights = weights.float().contiguous()
        self._angle_weights_typed = self.angle_weights.to(dtype)

        # (4F, J), rows (r, f, c): r = 0 gives [Re; Im] of each pattern's spectrum,
        # r = 1 [-Im; Re], so that [Tr | Ti] times them is Re and Im of conj(Q) T
        phase = 2 * math.pi * frequency.T * torch.arange(j, dtype=torch.float64, device=device) / j
        cos, sin = torch.cos(phase), torch.sin(phase)
        self.dft = torch.stack([cos, -sin, sin, cos]).view(2, 2, f, j).transpose(1, 2)
        self.dft = self.dft.reshape(4 * f, j).to(dtype).contiguous()

        self.inverse_energy = torch.empty(t, self.template_count, device=device)
        self.buffers = None
        if self.indexer.fused:
            self.buffers = InplaneWorkspace(
                self.capacity,
                source_pixels=self.source_shape[0] * self.source_shape[1],
                polar_shape=(k, j),
                frequencies=f,
                thetas=t,
                chunk=self.chunk,
                dtype=dtype,
                device=device,
            )
        self._mask_centers = None

    def _chunk_for(self, memory: int) -> int:
        """The most templates a chunk scores at once with the working buffers within
        *memory* bytes, for :attr:`capacity` patterns."""
        k, j = self.grid.shape
        f, t = self.frequencies, self.thetas
        sizes = InplaneWorkspace.sizes(
            self.capacity,
            source_pixels=self.source_shape[0] * self.source_shape[1],
            polar_shape=(k, j),
            frequencies=f,
            thetas=t,
            chunk=1,
            dtype=self.dtype,
        )
        room = memory - sizes["fixed"]
        chunk = room // sizes["per_template"] if room > sizes["polar"] else 0
        chunk = int(min(self.template_count, max(MIN_CHUNK, chunk)))
        if chunk == MIN_CHUNK and room < max(sizes["polar"], MIN_CHUNK * sizes["per_template"]):
            need = sizes["fixed"] + max(sizes["polar"], MIN_CHUNK * sizes["per_template"])
            warnings.warn(
                f"the in-plane search needs {need / 2**20:.0f} MB for {self.capacity} patterns, "
                f"more than the {memory / 2**20:.0f} MB it was given",
                RuntimeWarning,
                stacklevel=4,
            )
        return chunk

    def _templates(self, lo: int, hi: int) -> torch.Tensor:
        """Templates *lo*:*hi* on the polar grid, ``(n, K, J)`` float32, each made
        mean-free and unit over the whole disk with the area weights."""
        k, j = self.grid.shape
        indexer = self.indexer
        t = indexer.project(
            indexer.search_texture,
            self.axes[lo:hi],
            self.template_phase[lo:hi],
            self.grid.directions,
        ).view(-1, k, j)
        w = self._radius_weights
        t = t - (t * w).sum((1, 2), keepdim=True) / (w.sum() * j)
        norm = ((t * t) * w).sum((1, 2), keepdim=True).sqrt().clamp_min(SINGLE_PRECISION_FLOOR)
        return t / norm

    # --- the mask -----------------------------------------------------------------

    def set_scan_centers(self, pattern_centers) -> None:
        """Search a scan whose patterns lie at *pattern_centers* ``(M, 3)``: the
        visibility mask becomes the samples on the screen at every one of them, for
        the whole scan, until the next call or the indexer's next
        :meth:`~de_ebsd.index.live.LiveIndexer.set_pattern_center`. Pass the four
        corners of the scan's center plane
        (:meth:`~de_ebsd.index.live.calibration.CenterPlane.at`), which give the
        mask every center of the plane does, or ``None`` for the indexer's center."""
        self.scan_centers = (
            None
            if pattern_centers is None
            else np.atleast_2d(np.asarray(pattern_centers, dtype=np.float64)).copy()
        )

    def _wanted_centers(self) -> np.ndarray:
        if self.scan_centers is not None:
            return self.scan_centers
        return np.asarray(self.indexer.detector.pattern_center, dtype=np.float64)[None]

    def prepare_mask(self) -> None:
        """Bring the mask and the templates' visible variances up to date with the
        centers searched at: a pass over the templates whenever they change, and a
        new polar grid where a center's screen reaches past it."""
        centers = self._wanted_centers()
        if self._mask_centers is not None and np.array_equal(centers, self._mask_centers):
            return
        corner = farthest_corner(centers, self.aspect)
        if corner > self.grid.radius:
            self._build(corner)
        k, j = self.grid.shape
        visible = self.grid.visible(centers, self.aspect)
        #: ``(K, J)`` float32: each sample's area weight on the screen, zero off it.
        self.weights = (self.grid.radii[:, None] * visible).float().contiguous()
        #: The weights' sum.
        self.visible = float(self.weights.sum())
        mask = torch.conj(torch.fft.rfft(self.weights, dim=-1))
        stride = j // self.thetas
        for lo in range(0, self.template_count, self._template_chunk):
            hi = min(self.template_count, lo + self._template_chunk)
            t = self._templates(lo, hi)
            first = torch.fft.irfft(mask * torch.fft.rfft(t, dim=-1), n=j, dim=-1).sum(1)
            second = torch.fft.irfft(mask * torch.fft.rfft(t * t, dim=-1), n=j, dim=-1).sum(1)
            variance = (second - first * first / self.visible)[:, ::stride]
            self.inverse_energy[:, lo:hi] = variance.clamp_min(1e-12).rsqrt().T
        self._center.copy_(
            torch.as_tensor(self.indexer.detector.pattern_center, dtype=torch.float32)
        )
        self._mask_centers = centers.copy()

    # --- the patterns -------------------------------------------------------------

    def _source(self, refine_rows: torch.Tensor, out=None) -> torch.Tensor:
        """The refine rows ``(N, pixels)`` averaged to :attr:`source_shape`, ``(N,
        rows, cols)`` float32, into *out* where given."""
        n = len(refine_rows)
        images = refine_rows.float().view(n, *self.indexer.refine_shape)
        if tuple(images.shape[-2:]) == self.source_shape:
            return images
        if out is None:
            return average_to(images, self.source_shape)
        return pool_into(images, out)

    def operands(self, images: torch.Tensor, pattern_centers=None) -> torch.Tensor:
        """``(2, F, 2K, N)``: each pattern of *images* ``(N, rows, cols)`` resampled
        on the polar grid about its center (*pattern_centers* ``(N, 3)``, else the
        indexer's), weighted, made mean-free and unit over the visible samples, and
        its kept spectrum as ``[Re; Im]`` (first) and ``[-Im; Re]`` (second), the
        right operands of the per-frequency products. In plain torch."""
        k, j = self.grid.shape
        n = len(images)
        centers = pattern_centers
        if centers is None:
            centers = self.indexer.detector.pattern_center
        grid = self.grid.sampling(centers, self.aspect)
        if len(grid) == 1:
            q = F.grid_sample(
                images[None], grid, mode="bilinear", padding_mode="border", align_corners=False
            )[0]
        else:
            q = F.grid_sample(
                images[:, None], grid, mode="bilinear", padding_mode="border", align_corners=False
            )[:, 0]
        w = self.weights
        total = (q * w).sum((1, 2))
        mean = total / self.visible
        squares = (q * q * w).sum((1, 2))
        scale = 1 / (squares - mean * total).clamp_min(0).sqrt().clamp_min(SINGLE_PRECISION_FLOOR)
        polar = w * (q - mean[:, None, None]) * scale[:, None, None]
        polar = polar.transpose(0, 1).reshape(k * n, j).to(self.dtype)
        f = self.frequencies
        return (self.dft @ polar.T).view(2, f, 2 * k, n)

    # --- the search ---------------------------------------------------------------

    def search(self, refine_rows: torch.Tensor, pattern_centers=None):
        """The best template and in-plane angle for each pattern: ``(phase, g, score,
        entry)``, *entry* ``u * thetas + m`` (see :meth:`entry_rotations`), *score*
        the masked normalized cross-correlation. *refine_rows* ``(N, pixels)`` are
        the indexer's refine rows; *pattern_centers* ``(N, 3)``, one per pattern,
        default to the indexer's. Returns new tensors."""
        self.prepare_mask()
        n = len(refine_rows)
        if pattern_centers is not None:
            pattern_centers = np.asarray(pattern_centers, dtype=np.float64)
            if pattern_centers.shape != (n, 3):
                raise ValueError("one pattern center (x, y, z) per pattern")
        columns = []
        for lo in range(0, n, self.capacity):
            rows = refine_rows[lo : lo + self.capacity]
            centers = None if pattern_centers is None else pattern_centers[lo : lo + self.capacity]
            if self.buffers is not None:
                out = (
                    torch.empty(len(rows), dtype=torch.int64, device=self.device),
                    torch.empty(len(rows), 3, 3, device=self.device),
                    torch.empty(len(rows), device=self.device),
                    torch.empty(len(rows), dtype=torch.int64, device=self.device),
                )
                columns.append(self.search_into(rows, out, centers))
            else:
                columns.append(self._search_torch(rows, centers))
        return tuple(torch.cat(column) for column in zip(*columns, strict=True))

    def search_into(self, refine_rows: torch.Tensor, out, pattern_centers=None):
        """:meth:`search` of at most :attr:`capacity` patterns on the fused kernels,
        written into *out* ``(phase, g, score, entry)``; nothing is allocated on the
        device. The mask must be current (:meth:`prepare_mask`)."""
        n = len(refine_rows)
        buffers = self.buffers
        k, j = self.grid.shape
        f, t = self.frequencies, self.thetas
        mark = self.marker or _skip
        mark("start")
        source = self._source(refine_rows, buffers.source(n, *self.source_shape))
        if pattern_centers is None:
            centers = self._center
        else:
            centers = buffers.centers[:n]
            centers.copy_(torch.as_tensor(pattern_centers, dtype=torch.float32))
        polar = _kernels.polar_patterns(
            source.contiguous(),
            centers,
            self.aspect,
            self._radii32,
            self._cosines32,
            self._sines32,
            self.weights,
            self.visible,
            SINGLE_PRECISION_FLOOR,
            buffers.polar(n),
        )
        mark("patterns")
        operands = buffers.operands(n)
        torch.mm(self.dft, polar.view(k * n, j).T, out=operands.view(4 * f, k * n))
        mark("spectra")
        best = buffers.best[:n].zero_()
        for lo in range(0, self.template_count, self.chunk):
            hi = min(self.template_count, lo + self.chunk)
            products = buffers.products(hi - lo, n)
            spectra = self.spectra[:, lo:hi]
            torch.bmm(spectra, operands[0], out=products[0])
            torch.bmm(spectra, operands[1], out=products[1])
            mark("products")
            scores = buffers.scores(hi - lo, n)
            torch.mm(self._angle_weights_typed, products.view(2 * f, -1), out=scores)
            mark("angles")
            _kernels.inplane_best(scores, self.inverse_energy, lo, t, hi - lo, n, best)
            mark("best")
        found = _kernels.inplane_finish(
            best,
            operands,
            self.spectra,
            self.angle_weights,
            self.inverse_energy,
            self.axes,
            self._template_phase32,
            self.frame32,
            t,
            self.substep,
            out,
        )
        mark("finish")
        return found

    def _search_torch(self, refine_rows: torch.Tensor, pattern_centers=None):
        """:meth:`search` of at most :attr:`capacity` patterns in plain torch."""
        operands = self.operands(self._source(refine_rows), pattern_centers)
        n = operands.shape[-1]
        f, t = self.frequencies, self.thetas
        best = torch.full((n,), -math.inf, device=self.device)
        entry = torch.zeros(n, dtype=torch.int64, device=self.device)
        for lo in range(0, self.template_count, self.chunk):
            hi = min(self.template_count, lo + self.chunk)
            spectra = self.spectra[:, lo:hi]
            products = torch.stack([spectra @ operands[0], spectra @ operands[1]])
            scores = (self._angle_weights_typed @ products.view(2 * f, -1)).float()
            scores = scores.view(t, hi - lo, n) * self.inverse_energy[:, lo:hi, None]
            value, angle = scores.max(0)
            value, template = value.max(0)
            angle = angle.gather(0, template[None])[0]
            better = value > best
            best = torch.where(better, value, best)
            entry = torch.where(better, (template + lo) * t + angle, entry)
        return self._finish_torch(operands, best, entry)

    def _finish_torch(self, operands, best, entry):
        """``(phase, g, score, entry)`` of each pattern's winner, its angle moved to
        the parabola through its scores either side (with :attr:`substep`)."""
        f, t = self.frequencies, self.thetas
        template, angle = entry // t, entry % t
        offset = torch.zeros(len(entry), dtype=torch.float64, device=self.device)
        if self.substep:
            spectra = self.spectra[:, template].float()
            real = torch.einsum("fnc,fcn->nf", spectra, operands[0].float())
            imaginary = torch.einsum("fnc,fcn->nf", spectra, operands[1].float())
            steps = torch.tensor([-1, 0, 1], device=self.device)
            around = (angle[None] + steps[:, None]) % t
            weights = self.angle_weights[around]
            s = (weights[..., :f] * real + weights[..., f:] * imaginary).sum(-1)
            s = s * self.inverse_energy[around, template[None]]
            curvature = s[0] - 2 * s[1] + s[2]
            vertex = 0.5 * (s[0] - s[2]) / curvature.clamp_max(-1e-30)
            offset = torch.where(curvature < 0, vertex.clamp(-0.5, 0.5), 0.0).double()
        theta = 2 * math.pi * (angle.double() + offset) / t
        g = self.axes[template] @ in_plane_rotations(self.frame, theta).float()
        return self.template_phase[template], g, best, entry

    # --- the entries --------------------------------------------------------------

    def entry_rotations(self, entry) -> torch.Tensor:
        """``(N, 3, 3)``: the orientation of each *entry* ``u * thetas + m``, template
        ``u`` at in-plane angle ``m`` steps, as :meth:`search` numbers them."""
        entry = torch.as_tensor(entry, device=self.device)
        theta = 2 * math.pi * (entry % self.thetas).double() / self.thetas
        return self.axes[entry // self.thetas] @ in_plane_rotations(self.frame, theta).float()

    def template_bytes(self) -> int:
        """The device memory the templates take: their spectra and visible norms."""
        return (
            self.spectra.numel() * self.spectra.element_size()
            + self.inverse_energy.numel() * self.inverse_energy.element_size()
        )


def _skip(stage: str) -> None:
    """No stage marker."""
