"""Live strain against a real reference pattern, fitted over the whole pattern.

A :class:`StrainReference` is built once, before a scan, from the frames taken
at a strain-free point (summed: its noise is in every fit, so it wants many
times a scan point's electrons). It projects the reference onto an equal-area
texture over the sample frame's directions; each test pattern is then fitted
against that texture by the same fused kernel that fits the master
(:func:`~de_ebsd.index._kernels.fit_passes`): eight trace-free numbers, three
rotations and five deviatoric strains, every pixel inside the screen's margin in
one least-squares fit, one block per pattern. The reference stands in for the
simulation, so nothing about how well a master matches the camera limits the
strain, and the strain is relative to the reference's.

Each pattern, the reference's included, is binned to the fit's shape and divided
by its own smooth background (:data:`FLATTEN_SIGMA`), less one. Every pattern
may be fitted at its own pattern center (the beam moves the center across a
scan): the difference from the reference's is exact in the model
(:func:`~de_ebsd.index.live.deformation.center_maps`), so it is not taken for
strain.

What was measured, on a 15 kV Si nano-indent at 2048 x 2048 (2026-11-24):

* binned to 1024 the strain's error floor on synthetic patterns with the scan's
  own bands and counts was 0.2-0.3e-4, at 512 0.5e-4 (the pixel's size): hence
  :data:`STRAIN_SIDE`;
* a reference of 9 frames' electrons gave the odd point far off, 25 or more
  none, and a Wiener filter or a noise-whitened fit gained nothing;
* the camera's gain must be right at the pixel scale: a gain error is the same
  in every frame, does not move with the lattice, and pulls every fit toward
  zero strain (by a third at 1024 on that scan). It is the gain reference's to
  remove, not this fit's.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from de_ebsd.core.constants import SINGLE_PRECISION_FLOOR
from de_ebsd.core.geometry import sample_to_detector_matrix
from de_ebsd.core.lambert import SQRT_PI_HALF, square_to_sphere
from de_ebsd.index.live.common import as_frames
from de_ebsd.index.preprocess import normalize_rows

#: The most rows the fit bins a frame to (by a whole number). On synthetic Si
#: patterns 1024 halved 512's error; the fit of 1024 x 1024 patterns keeps up
#: with a scan at well over five a second.
STRAIN_SIDE = 1024

#: Each pattern is divided by its own Gaussian-blurred self, of this share of its
#: width, less one: the backscatter's envelope goes and the bands stay. The blur
#: keeps some of the bands' slow variation, which moves with the lattice, so it
#: costs a few % of the strain once the lattice turns by degrees: on synthetic
#: 256-pixel band patterns a strain of 2e-3 came back within 3e-5 at no turn and
#: 2.6e-4 at a 2 degree turn. Hence the reference is for a grain's small turns.
FLATTEN_SIGMA = 1.0 / 16.0

#: The blur's width, in pixels, that the patterns are averaged down to before it
#: is taken (:func:`flatten`): a 64-pixel blur at 1024 is taken at 128 x 128.
FLATTEN_COARSEN = 8

#: The share of each side left out at the screen's edges, where the deformed
#: reference would be sampled outside the screen it was taken on.
MARGIN = 0.05

#: Texels of the reference's texture per pattern pixel along a side: the fit
#: samples it bilinearly, and coarser textures blur the bands it matches.
OVERSAMPLE = 6

#: Full-resolution Gauss-Newton passes of one fit. From the identity on the Si
#: indent's far field 30 passes converged to well under 1e-5.
PASSES = 30

#: Patterns fitted in one launch; a 1024 x 1024 pattern's rows are 4 MB.
FIT_BATCH = 64

#: Frames binned and flattened at once: a 2048 x 2048 frame is 16 MB as float.
FLATTEN_BATCH = 16


@dataclass(frozen=True)
class ReferenceStrain:
    """What :meth:`StrainReference.fit` measured, per test pattern."""

    strain: torch.Tensor
    """``(N, 3, 3)`` deviatoric strain relative to the reference, in the sample
    frame: the left stretch of the deformation less the identity, less its
    trace. NaN where the fit was refused."""

    rotation_vector: torch.Tensor
    """``(N, 3)`` lattice rotation from the reference, axis times angle in
    radians, in the sample frame. NaN where refused."""

    score: torch.Tensor
    """``(N,)`` the normalized correlation of the pattern with the fitted
    reference: low at a boundary, a scratch or another grain."""


def fit_shape(frame_shape, most: int = STRAIN_SIDE) -> tuple[int, int]:
    """The shape a frame of *frame_shape* is fitted at: binned by the smallest
    whole number that divides both sides and leaves at most *most* rows."""
    rows, cols = int(frame_shape[0]), int(frame_shape[1])
    for factor in range(1, rows + 1):
        if rows % factor == 0 and cols % factor == 0 and rows // factor <= most:
            return rows // factor, cols // factor
    raise ValueError(f"{(rows, cols)} has no whole-number binning to {most} rows")


def binned(frames, shape, device) -> torch.Tensor:
    """*frames* ``(N, H, W)`` (any dtype, anywhere) summed in whole-number bins
    to *shape* on *device*: ``(N, *shape)`` float32."""
    t = as_frames(frames)
    t = t[None] if t.ndim == 2 else t
    rows, cols = t.shape[-2] // shape[0], t.shape[-1] // shape[1]
    if rows * shape[0] != t.shape[-2] or cols * shape[1] != t.shape[-1]:
        raise ValueError(f"{tuple(t.shape[-2:])} does not bin evenly to {tuple(shape)}")
    out = torch.empty((len(t), *shape), device=device, dtype=torch.float32)
    for lo in range(0, len(t), FLATTEN_BATCH):
        x = t[lo : lo + FLATTEN_BATCH].to(device=device, dtype=torch.float32)
        if rows > 1 or cols > 1:
            x = x.reshape(len(x), shape[0], rows, shape[1], cols).sum((2, 4))
        out[lo : lo + len(x)] = x
    return out


def flatten(patterns: torch.Tensor) -> torch.Tensor:
    """*patterns* ``(N, H, W)`` over their smooth background, their own Gaussian
    blur (:data:`FLATTEN_SIGMA`), less one.

    The blur is taken on the patterns averaged down by :data:`FLATTEN_COARSEN`
    and drawn back up bilinearly: a blur 64 pixels wide is smooth on that scale,
    and at full resolution its 385-tap filter cost 8.5 ms a 1024 x 1024 pattern,
    as long as the fit itself."""
    from de_ebsd.index.reference_fit import _smooth

    rows, cols = patterns.shape[-2:]
    sigma = FLATTEN_SIGMA * cols
    factor = max(1, int(sigma // FLATTEN_COARSEN))
    if rows % factor or cols % factor:
        factor = 1
    coarse = F.avg_pool2d(patterns[:, None], factor)[:, 0] if factor > 1 else patterns
    envelope = _smooth(coarse, sigma / factor)
    if factor > 1:
        envelope = F.interpolate(
            envelope[:, None], size=(rows, cols), mode="bilinear", align_corners=False
        )[:, 0]
    return patterns / envelope.clamp_min(SINGLE_PRECISION_FLOOR) - 1.0


def project_to_texture(image: torch.Tensor, detector, texels: int, chunk: int = 512):
    """*image* ``(rows, cols)`` on the *detector* as a ``(1, 2, texels, texels)``
    equal-area texture over the sample frame's two hemispheres, as a master's;
    where no pixel sees a direction, the image's mean.

    The image is interpolated by its cubic B-spline, which passes through every
    pixel. Torch's bicubic (Keys) interpolation does not quite, and its error
    depends on where between pixels a band falls, so it moves with the band: on
    256-pixel synthetic patterns it held strains of 1e-4 to 1e-3 some 15% short
    (1e-4 at most beyond), where the spline left 1e-6."""
    from de_ebsd.index.reference_fit import _spline_coefficients, _spline_pad, _spline_weights

    device = image.device
    rows, cols = detector.nrows, detector.ncols
    s = torch.linspace(-1, 1, texels, device=device, dtype=torch.float64) * SQRT_PI_HALF
    om = torch.as_tensor(
        sample_to_detector_matrix(
            detector.sample_tilt, detector.tilt, detector.azimuthal, detector.twist
        ),
        device=device,
        dtype=torch.float64,
    )
    x_min, x_max, y_min, y_max = detector.gnomonic_bounds
    image = image.reshape(rows, cols).double()
    coefficients = _spline_coefficients(image)
    pad, width = _spline_pad(rows, cols), coefficients.shape[1]
    flat = coefficients.reshape(-1)
    mean = float(image.mean())
    out = torch.empty(1, 2, texels, texels, device=device)
    for hemisphere, sign in enumerate((1.0, -1.0)):
        flip = torch.tensor([1.0, 1.0, sign], device=device, dtype=torch.float64)
        for r0 in range(0, texels, chunk):
            ly, lx = torch.meshgrid(s[r0 : r0 + chunk], s, indexing="ij")
            on_screen = (square_to_sphere(torch.stack([lx, ly], -1)) * flip) @ om.T
            front = on_screen[..., 2] > 1e-6
            z = torch.where(front, on_screen[..., 2], torch.ones_like(on_screen[..., 2]))
            # pixel indices, centers at whole numbers (direction_cosines' layout)
            col = (on_screen[..., 0] / z - x_min) / (x_max - x_min) * cols - 0.5
            row = (y_max - on_screen[..., 1] / z) / (y_max - y_min) * rows - 0.5
            inside = (
                front & (row >= -0.5) & (row <= rows - 0.5) & (col >= -0.5) & (col <= cols - 0.5)
            )
            row, col = row.clamp(0, rows - 1), col.clamp(0, cols - 1)
            i, j = torch.floor(row), torch.floor(col)
            wr, _ = _spline_weights(row - i)
            wc, _ = _spline_weights(col - j)
            base = (i.long() - 1 + pad) * width + (j.long() - 1 + pad)
            value = torch.zeros_like(row)
            for a in range(4):
                for b in range(4):
                    value += wr[a] * wc[b] * flat[base + (a * width + b)]
            out[0, hemisphere, r0 : r0 + chunk] = torch.where(
                inside, value, torch.full_like(value, mean)
            ).float()
    return out


class StrainReference:
    """A strain-free reference pattern, ready to fit scan patterns against.

    Parameters
    ----------
    frames
        ``(H, W)`` or ``(N, H, W)``: the reference point's frames, summed. A few
        neighboring points' frames (a 2 x 2 raster) spread the dose; together
        they want some 25 times a scan point's electrons.
    detector
        The detector, its pattern center the reference's (its shape is taken
        from the frames).
    rotation, symmetry, phase
        The reference's indexed orientation ``(3, 3)`` (sample to crystal), its
        phase's symmetry and the phase: with the first two :meth:`fit` seeds
        each pattern with the lattice turn its own orientation gives.
    most
        The most rows the frames are binned to (:data:`STRAIN_SIDE`).
    device
        A CUDA device: the fit is the fused kernel's.
    """

    def __init__(
        self,
        frames,
        detector,
        *,
        rotation=None,
        symmetry=None,
        phase: int | None = None,
        most: int = STRAIN_SIDE,
        device="cuda",
    ):
        from de_ebsd.index import _kernels
        from de_ebsd.index.live.deformation import _texture_gradients

        if not _kernels.available(device):
            raise RuntimeError(f"the reference fit runs on a CUDA device with CuPy, not {device}")
        t = as_frames(frames)
        t = t[None] if t.ndim == 2 else t
        self.device = torch.device(device)
        self.frame_shape = tuple(int(v) for v in t.shape[-2:])
        self.shape = fit_shape(self.frame_shape, most)
        self.detector = replace(detector, shape=self.shape)
        self.rotation = None if rotation is None else np.asarray(rotation, np.float64)
        self.symmetry = symmetry
        self.phase = phase
        total = torch.zeros(self.frame_shape, device=self.device, dtype=torch.float32)
        for frame in t:
            total += frame.to(device=self.device, dtype=torch.float32)
        image = flatten(binned(total, self.shape, self.device))[0]
        rows, cols = self.shape
        inset_rows, inset_cols = round(MARGIN * rows), round(MARGIN * cols)
        y, x = torch.meshgrid(
            torch.arange(rows, device=self.device),
            torch.arange(cols, device=self.device),
            indexing="ij",
        )
        self.keep = (
            (y >= inset_rows)
            & (y < rows - inset_rows)
            & (x >= inset_cols)
            & (x < cols - inset_cols)
        ).reshape(-1)
        self.every_direction = self.detector.direction_cosines(device=self.device)
        self.directions = self.every_direction[self.keep].contiguous()
        self.texture = project_to_texture(image, self.detector, OVERSAMPLE * max(rows, cols) + 1)
        self.gradients = _texture_gradients(self.texture)
        self._offset = float(self.texture.mean())

    def fit(self, frames, *, pattern_centers=None, rotations=None):
        """The strain of each of *frames* ``(N, H, W)``, at the reference's
        frame shape, relative to the reference (:class:`ReferenceStrain`).

        *pattern_centers* ``(N, 3)`` are each frame's (the reference's when
        None). With each frame's indexed orientation *rotations* ``(N, 3, 3)``
        each fit starts from the lattice turn the orientations give (it needs
        the reference's *rotation* and *symmetry*); without them from none,
        which holds for a fraction of a degree.

        Each pattern is flattened (:func:`flatten`) and fitted in
        :data:`PASSES` Gauss-Newton passes of the fused kernel."""
        from de_ebsd.index.live.deformation import center_maps

        t = as_frames(frames)
        t = t[None] if t.ndim == 2 else t
        if tuple(t.shape[-2:]) != self.frame_shape:
            raise ValueError(f"frames are {tuple(t.shape[-2:])}, the reference {self.frame_shape}")
        n = len(t)
        seed = self._seed(n, rotations)
        maps = (
            None
            if pattern_centers is None
            else center_maps(self.detector, np.asarray(pattern_centers, float), self.device)
        )
        p = torch.empty(n, 8, device=self.device)
        score = torch.empty(n, device=self.device)
        refused = torch.empty(n, dtype=torch.bool, device=self.device)
        for lo in range(0, n, FIT_BATCH):
            hi = min(n, lo + FIT_BATCH)
            flat = flatten(binned(t[lo:hi], self.shape, self.device))
            batch_maps = None if maps is None else maps[lo:hi].contiguous()
            p[lo:hi], score[lo:hi], refused[lo:hi] = self._passes(
                flat, seed[lo:hi], batch_maps, PASSES
            )
        strain, rotation_vector = _deformation_parts(p, seed)
        strain = torch.where(refused[:, None, None], torch.nan, strain)
        rotation_vector = torch.where(refused[:, None], torch.nan, rotation_vector)
        return ReferenceStrain(strain=strain, rotation_vector=rotation_vector, score=score)

    def _passes(self, flat, g, maps, passes):
        """The fused kernel's fit of the flattened patterns *flat* ``(N, H, W)``
        from *g* ``(N, 3, 3)``: ``(p, score, refused)``."""
        from de_ebsd.index import _kernels
        from de_ebsd.index.live.deformation import DEFORMATION_SCORE_MARGIN, STRAIN_MAX_STEP

        n = len(flat)
        rows = normalize_rows(flat.reshape(n, -1)[:, self.keep])
        full = (self.directions, rows, None)
        return _kernels.fit_passes(
            self.texture,
            g.contiguous(),
            torch.zeros(n, dtype=torch.int32, device=self.device),
            maps,
            full,
            full,
            ((False, 1.0),) * passes,
            max_step=STRAIN_MAX_STEP,
            damping=1e-3,
            margin=DEFORMATION_SCORE_MARGIN,
            offset=self._offset,
            floor=SINGLE_PRECISION_FLOOR,
            gradients=self.gradients,
            strain=True,
            threads=_kernels.FIT_STRAIN_THREADS,
        )

    def _seed(self, n, rotations) -> torch.Tensor:
        """``(N, 3, 3)`` the kernel's starting ``g``: a test direction's
        reference direction, ``F^-1``, as the orientations give it."""
        if rotations is None:
            return torch.eye(3, device=self.device).expand(n, 3, 3).contiguous()
        if self.rotation is None or self.symmetry is None:
            raise ValueError("a seed from orientations needs the reference's rotation and symmetry")
        from de_ebsd.index.reference_fit import lattice_turns

        turns = lattice_turns(rotations, self.rotation, self.symmetry)
        return torch.as_tensor(
            np.ascontiguousarray(turns.transpose(0, 2, 1)), device=self.device, dtype=torch.float32
        )


def _deformation_parts(p: torch.Tensor, seed: torch.Tensor):
    """The kernel's numbers *p* ``(N, 8)`` about *seed*: the candidate is
    ``g (I - beta)``, a test direction's reference direction, so the lattice's
    deformation from the reference is ``F = (g (I - beta))^-1``. Its polar parts
    ``F = V R``: the deviatoric ``V - I`` and the rotation vector of ``R``, in
    the sample frame."""
    from de_ebsd.index.live.deformation import trace_free_basis
    from de_ebsd.index.reference_fit import _polar_strain

    beta = torch.einsum("nc,cij->nij", p.double(), trace_free_basis(p.device).double())
    eye = torch.eye(3, device=p.device, dtype=torch.float64)
    f = torch.linalg.inv(seed.double() @ (eye - beta))
    strain, rotation_vector = _polar_strain(f, eye)
    return strain.float(), rotation_vector.float()
