"""The backgrounds a scan's patterns are divided by: static, dynamic and ghost.

The static background is the running mean of the scan's frames; a pattern is
divided by it, then its own slowly varying (dynamic) background is removed. The
ghost is what the static background keeps of the scan's grains. A residual
background, fitted (:func:`~de_ebsd.index.appearance.fit_appearance`), is what the
corrected patterns still carry that no simulation has.
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
            background = ScanBackground(tuple(grid.shape[-2:]), grid.device).remove(grid)
        return normalize_rows(background.reshape(1, -1))[0]


class ScanBackground:
    """The static background and ghost of one scan, at the refine shape.

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
    ) -> None:
        self.refine_shape = refine_shape
        self.device = device
        self.min_frames = int(min_frames)
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

    def remove(self, t: torch.Tensor) -> torch.Tensor:
        """*t* ``(N, H, W)`` divided by the static background, less each pattern's
        dynamic one, at whatever shape it is.

        Divided rather than less: the bands' contrast scales with the
        backscattered intensity, so a pattern less its static background still
        has its bands weighted by the bright middle of the screen, and that
        weighting moves a band's center differently in every grain."""
        shape = tuple(t.shape[-2:])
        if self.static is not None:
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

    def add_fits(self, simulations: torch.Tensor) -> None:
        """Add the simulations ``(N, pixels)`` of fitted orientations to the :attr:`ghost`."""
        if self.blur:
            return
        self._fit_sum += normalize_rows(simulations).sum(0)
        self._fit_count += len(simulations)

    def add_fits_in_place(self, simulations: torch.Tensor, scratch: torch.Tensor) -> None:
        """:meth:`add_fits`, normalizing *simulations* where they are and summing
        them into *scratch*, at least ``max(2 N, pixels)`` floats: the same
        operations, so the same ghost to the bit."""
        if self.blur:
            return
        normalize_rows_into(simulations, scratch)
        self._fit_sum += torch.sum(simulations, 0, out=scratch[: simulations.shape[1]])
        self._fit_count += len(simulations)

    def ghost_into(self, out: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor | None:
        """:attr:`ghost` written into *out* ``(pixels,)``, *scratch* at least two
        floats, or None where there is none."""
        if self.static is None or not self._fit_count:
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
        grain. A blurred static background has no bands, so then there is no ghost.
        """
        if self.static is None or not self._fit_count:
            return None
        return normalize_rows(self._fit_sum[None])[0]
