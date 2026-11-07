"""The backgrounds a scan's patterns are divided by: static, dynamic and ghost.

The static background is the running mean of the scan's frames; a pattern is
divided by it, then its own slowly varying (dynamic) background is removed. The
ghost is what the static background keeps of the scan's grains.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from de_ebsd.core.constants import SINGLE_PRECISION_FLOOR
from de_ebsd.index.live.common import average_to, normalize_rows_into
from de_ebsd.index.preprocess import gaussian_blur, normalize_rows

#: Frames the static background is averaged over before it is used: fewer still
#: carry the bands of the few grains they came from.
MIN_FRAMES_FOR_STATIC = 32


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
