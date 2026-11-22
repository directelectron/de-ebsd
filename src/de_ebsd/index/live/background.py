"""The backgrounds removed from a scan's patterns: log-polynomial or static, then
dynamic; and the ghost.

By default (:data:`LOG_POLYNOMIAL`) each pattern's background is its own: the
log of the pattern (plus :data:`LOG_OFFSET` of its mean), less the least-squares
polynomial surface of degree
:data:`BACKGROUND_DEGREE` that fits it. The screen's backscattered intensity is a
smooth envelope that multiplies the bands, so in the log it is a smooth surface
added to them, and a low-degree surface removes it without taking the bands. It
needs no other frame: no grain's bands come into it from its neighbors, and the
first pattern of a scan is corrected as well as the last.

The static background (:data:`STATIC`) is instead the running mean of the scan's
frames; a pattern is divided by it. Early in a scan that mean is of a few grains
and carries their bands, and the patterns divided by it are indexed wrong.
Against OIM, counting the points more than 5 degrees from every OIM point within
3 steps, the static background left 599 of a 200 x 200 Ni scan's 40,000 and 3 of
a counted 500 x 500 one's 250,000; the log-polynomial none of either.

Either way each pattern's own slowly varying (dynamic) background is then
removed. The ghost is what the static background keeps of the scan's grains. A
residual background, fitted (:func:`~de_ebsd.index.appearance.fit_appearance`),
is what the corrected patterns still carry that no simulation has.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from de_ebsd.core.constants import SINGLE_PRECISION_FLOOR
from de_ebsd.index.live.common import average_to, normalize_rows_into
from de_ebsd.index.preprocess import gaussian_blur, normalize_rows

#: Frames the static background is averaged over before it is used: fewer still
#: carry the bands of the few grains they came from.
MIN_FRAMES_FOR_STATIC = 32

#: Each pattern's log less its own polynomial surface (:class:`ScanBackground`).
LOG_POLYNOMIAL = "log-polynomial"
#: The pattern divided by the scan's running mean.
STATIC = "static"
BACKGROUNDS = (LOG_POLYNOMIAL, STATIC)
#: The log-polynomial surface's degree: 2 (six terms) was measured on real Ni
#: scans; degrees 4 and 6 indexed no better.
BACKGROUND_DEGREE = 2
#: The log is taken of the pattern plus this share of its own mean: a counted
#: pattern's dark pixels are zero or a few counts, whose log is all noise, and an
#: integrating camera's pattern sits on an offset. Measured on real scans (0,
#: 0.1, 0.25, 0.5 and 1 of the mean): at 0.5 a counted 500 x 500 Ni scan had no
#: point more than 5 degrees from every OIM point within 3 steps (28 with none),
#: an integrating scan scored 0.52 (0.16 with none), and the others as well as
#: at any.
LOG_OFFSET = 0.5
#: And floored here, for a pattern with negative pixels.
LOG_FLOOR = 1e-3


def legendre_terms(shape, order: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """The smooth detector backgrounds ``(terms, rows * cols)``: products of Legendre
    polynomials ``P_i(x) P_j(y)`` with ``1 <= i + j <= order`` over the detector,
    ``x`` and ``y`` running from -1 to 1 across its pixel centers, row-major. The
    constant is left out: a normalized cross-correlation cannot see it."""
    rows, cols = int(shape[0]), int(shape[1])
    y = (torch.arange(rows, device=device, dtype=dtype) + 0.5) / rows * 2.0 - 1.0
    x = (torch.arange(cols, device=device, dtype=dtype) + 0.5) / cols * 2.0 - 1.0

    def legendre(t, n):
        previous, current = torch.ones_like(t), t  # Bonnet's recursion from P0, P1
        if n == 0:
            return previous
        for k in range(1, n):
            previous, current = current, ((2 * k + 1) * t * current - k * previous) / (k + 1)
        return current

    terms = [
        (legendre(y, j)[:, None] * legendre(x, total - j)[None, :]).reshape(-1)
        for total in range(1, int(order) + 1)
        for j in range(total + 1)
    ]
    if not terms:
        return torch.zeros(0, rows * cols, device=device, dtype=dtype)
    return torch.stack(terms)


def polynomial_basis(shape, degree: int, device=None) -> torch.Tensor:
    """Orthonormal rows ``(terms, rows * cols)`` spanning the polynomials of total
    degree up to *degree* over the detector, the constant among them: the
    log-polynomial background is a pattern's projection on them. Any basis of the
    same polynomials projects the same."""
    rows, cols = int(shape[0]), int(shape[1])
    constant = torch.ones(1, rows * cols, dtype=torch.float64, device=device)
    terms = torch.cat([constant, legendre_terms(shape, degree, device, torch.float64)])
    q, _ = torch.linalg.qr(terms.T)
    return q.T.to(torch.float32).contiguous()


@dataclass(frozen=True)
class ResidualBackground:
    """A smooth background a scan's corrected patterns still carry, fixed on the
    detector, as an appearance fit finds it
    (:func:`~de_ebsd.index.appearance.fit_appearance`).

    The fit's model of a corrected pattern is ``t + B``: the simulation ``t`` plus
    ``B``, the :func:`legendre_terms` of *order* weighted by *coefficients*. A
    normalized pattern is ``(t + B) / |t + B|`` (each less its mean), so removing
    ``share * B / |B|`` from it leaves the simulation alone, normalized again.
    *share*, ``|B| / |t + B|``, is the mean over the fit's patterns, and about the
    same at any detector resolution; a pattern whose own ``|t|`` is off the mean
    keeps a small part of the background. With *high_pass*, the patterns had their
    dynamic background removed (:meth:`ScanBackground.remove`), and ``B`` is taken
    through the same.
    """

    coefficients: tuple[float, ...]
    order: int
    share: float
    high_pass: bool = False

    def direction(self, shape, device=None) -> torch.Tensor:
        """``B`` at detector *shape*, less its mean and of unit length, ``(pixels,)``."""
        terms = legendre_terms(shape, self.order, device, torch.float32)
        background = torch.as_tensor(self.coefficients, device=device, dtype=torch.float32)
        background = background @ terms
        if self.high_pass:
            grid = background.view(1, *(int(v) for v in shape))
            # the dynamic background alone: the terms are not an intensity to take the log of
            background = ScanBackground(tuple(grid.shape[-2:]), grid.device, kind=STATIC).remove(
                grid
            )
        return normalize_rows(background.reshape(1, -1))[0]


class ScanBackground:
    """The backgrounds and ghost of one scan, at the refine shape.

    *kind* is :data:`LOG_POLYNOMIAL` (each pattern's own, of *degree*) or
    :data:`STATIC` (the scan's running mean). The static background is kept
    either way, for what looks at it, but only the static kind divides by it.

    *min_frames* is how many frames the static background waits for. A live scan's
    frames come from a few grains at a time, so it waits for
    :data:`MIN_FRAMES_FOR_STATIC`; frames chosen from all over a scan, as a
    pattern-center search's are, average their bands away in fewer.
    """

    def __init__(
        self,
        refine_shape: tuple[int, int],
        device: torch.device,
        min_frames: int = MIN_FRAMES_FOR_STATIC,
        kind: str = LOG_POLYNOMIAL,
        degree: int = BACKGROUND_DEGREE,
    ) -> None:
        if kind not in BACKGROUNDS:
            raise ValueError(f"the background is one of {BACKGROUNDS}, not {kind!r}")
        self.refine_shape = refine_shape
        self.device = device
        self.min_frames = int(min_frames)
        self.kind = kind
        self.degree = int(degree)
        self._bases: dict[tuple[int, int], torch.Tensor] = {}
        self.static = None
        self.blur = None
        self._static_sum = torch.zeros(refine_shape, device=device)
        self._static_count = 0
        # what one observe adds, and the mean the static background is, written in
        # place: a scan observes every buffer of frames it indexes
        self._observed = torch.empty(refine_shape, device=device)
        self._static_mean = torch.empty(refine_shape, device=device)
        self._fit_sum = torch.zeros(refine_shape[0] * refine_shape[1], device=device)
        self._fit_count = 0

    def restart(self, blur: float | None = None) -> None:
        """Forget the scan. With *blur* (a Gaussian sigma, as a share of the pattern's
        width) the static background will be the frames' mean blurred, for a single
        crystal: its frames all carry its bands, so their mean would divide them
        out. It then has no ghost."""
        self.static = None
        self.blur = blur
        self._static_sum.zero_()
        self._static_count = 0
        self._fit_sum.zero_()
        self._fit_count = 0

    def observe(self, pooled: torch.Tensor) -> None:
        """Add *pooled* frames ``(N, H, W)``, at the refine shape, to the static background."""
        self._static_sum += torch.sum(pooled, 0, out=self._observed)
        self._static_count += len(pooled)
        if self._static_count >= self.min_frames:
            static = torch.div(self._static_sum, self._static_count, out=self._static_mean)
            if self.blur:
                static = gaussian_blur(static[None], self.blur * static.shape[-1], 4.0)[0]
            self.static = static

    def basis(self, shape=None) -> torch.Tensor | None:
        """The log-polynomial background's :func:`polynomial_basis` at *shape*
        (the refine shape by default); None for the static kind."""
        if self.kind != LOG_POLYNOMIAL:
            return None
        shape = self.refine_shape if shape is None else tuple(shape)
        if shape not in self._bases:
            self._bases[shape] = polynomial_basis(shape, self.degree, self.device)
        return self._bases[shape]

    def remove(self, t: torch.Tensor) -> torch.Tensor:
        """*t* ``(N, H, W)`` less its log-polynomial background, or divided by the
        static background, then less each pattern's dynamic one, at whatever
        shape it is.

        Divided rather than less: the bands' contrast scales with the
        backscattered intensity, so a pattern less its static background still
        has its bands weighted by the bright middle of the screen, and that
        weighting moves a band's center differently in every grain. In the log
        that weighting is a sum, which the polynomial takes away."""
        shape = tuple(t.shape[-2:])
        basis = self.basis(shape)
        if basis is not None:
            offset = LOG_OFFSET * t.mean(dim=(-2, -1), keepdim=True)
            flat = torch.log((t + offset).clamp_min(LOG_FLOOR)).flatten(1)
            t = (flat - (flat @ basis.T) @ basis).view(t.shape)
        elif self.static is not None:
            static = self.static
            if shape != self.refine_shape:
                static = F.interpolate(
                    static[None, None], size=shape, mode="bilinear", align_corners=False
                )[0, 0]
            t = t / static.clamp_min(SINGLE_PRECISION_FLOOR) - 1.0
        # The dynamic background is an eighth of the width wide, so it is found at a
        # quarter of the refine resolution and drawn back up: a full-size blur that
        # wide was most of the preparation's time.
        coarse = average_to(t, (self.refine_shape[0] // 4, self.refine_shape[1] // 4))
        coarse = gaussian_blur(coarse, coarse.shape[-1] / 8.0, 4.0)
        return (
            t
            - F.interpolate(coarse[:, None], size=shape, mode="bilinear", align_corners=False)[:, 0]
        )

    @property
    def has_ghost(self) -> bool:
        """Whether the patterns carry a ghost: only a static background that is not
        blurred puts one in."""
        return self.kind == STATIC and not self.blur

    def add_fits(self, simulations: torch.Tensor) -> None:
        """Add the simulations ``(N, pixels)`` of fitted orientations to the :attr:`ghost`."""
        if not self.has_ghost:
            return
        self._fit_sum += normalize_rows(simulations).sum(0)
        self._fit_count += len(simulations)

    def add_fits_in_place(self, simulations: torch.Tensor, scratch: torch.Tensor) -> None:
        """:meth:`add_fits`, normalizing *simulations* where they are and summing
        them into *scratch*, at least ``max(2 N, pixels)`` floats: the same
        operations, so the same ghost to the bit."""
        if not self.has_ghost:
            return
        normalize_rows_into(simulations, scratch)
        self._fit_sum += torch.sum(simulations, 0, out=scratch[: simulations.shape[1]])
        self._fit_count += len(simulations)

    def ghost_into(self, out: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor | None:
        """:attr:`ghost` written into *out* ``(pixels,)``, *scratch* at least two
        floats, or None where there is none."""
        if not self.has_ghost or self.static is None or not self._fit_count:
            return None
        return normalize_rows_into(out.copy_(self._fit_sum)[None], scratch)[0]

    @property
    def ghost(self) -> torch.Tensor | None:
        """The scan's grains as the static background shows them: the normalized
        mean simulation of the orientations fitted so far (:meth:`add_fits`).

        The static background is the mean of the scan's frames, so it carries a
        faint copy of every grain's bands, and a frame divided by it carries
        their negative. A simulation has no such ghost; fitted without it, a
        strain bends to match the ghost of the other grains, differently in each
        grain. A blurred static background has no bands, so then there is no ghost,
        and nor is there with the log-polynomial background, which divides by none.
        """
        if not self.has_ghost or self.static is None or not self._fit_count:
            return None
        return normalize_rows(self._fit_sum[None])[0]
