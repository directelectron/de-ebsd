"""Background removal for experimental patterns.

Static (mean over the scan) and dynamic (Gaussian-blurred per pattern)
background correction, batched on the GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "remove_static_background",
    "remove_dynamic_background",
    "remove_background",
]

_EPS = 1e-12


def _as_stack(patterns, device, dtype):
    x = torch.as_tensor(patterns, device=device, dtype=dtype)
    if x.ndim == 2:
        lead = ()
    elif x.ndim == 3:
        lead = (x.shape[0],)
    elif x.ndim == 4:
        lead = (x.shape[0], x.shape[1])
    else:
        raise ValueError(
            f"patterns must be (H,W), (N,H,W) or (rows,cols,H,W); got shape {tuple(x.shape)}"
        )
    h, w = x.shape[-2], x.shape[-1]
    stack = x.reshape(-1, h, w)

    def restore(processed):
        return processed.reshape(*lead, h, w)

    return stack, restore


def _rescale01(p):
    lo = p.amin(dim=(-2, -1), keepdim=True)
    hi = p.amax(dim=(-2, -1), keepdim=True)
    return (p - lo) / (hi - lo).clamp_min(_EPS)


def _apply(pattern, background, operation):
    if operation == "subtract":
        return pattern - background
    if operation == "divide":
        return pattern / background.clamp_min(_EPS)
    raise ValueError(f"operation must be 'subtract' or 'divide', got {operation!r}")


def _gaussian_kernel_1d(std, truncate, device, dtype):
    radius = int(truncate * float(std) + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / float(std)) ** 2)
    return k / k.sum(), radius


def _gaussian_blur(stack, std, truncate):
    k, r = _gaussian_kernel_1d(std, truncate, stack.device, stack.dtype)
    x = stack[:, None]  # (N, 1, H, W)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x[:, 0]


def remove_static_background(
    patterns, static_bg, operation="subtract", device=None, dtype=torch.float32
):
    """Remove the fixed detector/beam background from a stack of patterns.

    Parameters
    ----------
    patterns
        Detector images, ``(H, W)`` / ``(N, H, W)`` / ``(rows, cols, H, W)``, numpy or
        torch. ``H``, ``W`` are the detector shape.
    static_bg
        The static background image ``(H, W)`` (e.g. a kikuchipy signal's
        ``static_background``, or the scan mean), numpy or torch.
    operation
        ``"subtract"`` (default) or ``"divide"``.
    device
        Torch device to run on. ``None`` (default) keeps the patterns' device
        (``"cuda"`` recommended for a whole scan).
    dtype
        Working floating dtype (default ``float32``).

    Returns
    -------
    torch.Tensor
        The cleaned patterns, same leading shape as the input, each rescaled to
        ``[0, 1]`` (indexing is invariant to the per-pattern affine scale).
    """
    if device is None:
        device = torch.as_tensor(patterns).device if torch.is_tensor(patterns) else "cpu"
    stack, restore = _as_stack(patterns, device, dtype)
    bg = torch.as_tensor(static_bg, device=stack.device, dtype=dtype)
    if bg.shape != stack.shape[-2:]:
        raise ValueError(
            f"static_bg shape {tuple(bg.shape)} != detector {tuple(stack.shape[-2:])}"
        )
    out = _rescale01(_apply(stack, bg[None], operation))
    return restore(out)


def remove_dynamic_background(
    patterns, std=None, operation="subtract", truncate=4.0, device=None, dtype=torch.float32
):
    """Remove the per-pattern dynamic background (a Gaussian blur of each pattern).

    Parameters
    ----------
    patterns
        Detector images, ``(H, W)`` / ``(N, H, W)`` / ``(rows, cols, H, W)``, numpy or
        torch.
    std
        Standard deviation of the Gaussian window. ``None`` (default) uses
        ``width / 8`` (kikuchipy's default), the width being the number of columns.
    operation
        ``"subtract"`` (default) or ``"divide"``.
    truncate
        Truncate the Gaussian window at this many standard deviations (default ``4.0``,
        matching scipy/kikuchipy).
    device, dtype
        As in :func:`remove_static_background`.

    Returns
    -------
    torch.Tensor
        The cleaned patterns, same leading shape as the input, each rescaled to
        ``[0, 1]``.
    """
    if device is None:
        device = torch.as_tensor(patterns).device if torch.is_tensor(patterns) else "cpu"
    stack, restore = _as_stack(patterns, device, dtype)
    if std is None:
        std = stack.shape[-1] / 8.0
    bg = _gaussian_blur(stack, std, truncate)
    out = _rescale01(_apply(stack, bg, operation))
    return restore(out)


def remove_background(
    patterns,
    static_bg=None,
    dynamic=True,
    operation="subtract",
    std=None,
    truncate=4.0,
    device=None,
    dtype=torch.float32,
):
    """Full EBSD preprocessing: static then dynamic background removal, on the GPU.

    Convenience wrapper that runs :func:`remove_static_background` (when ``static_bg``
    is given) followed by :func:`remove_dynamic_background` (when ``dynamic``), the
    usual pre-indexing pipeline, all on ``device`` in one pass over the scan.

    Parameters
    ----------
    patterns
        The raw scan, ``(H, W)`` / ``(N, H, W)`` / ``(rows, cols, H, W)``, numpy or torch.
    static_bg
        The static background image ``(H, W)``; if ``None``, the static step is skipped.
    dynamic
        Whether to also remove the dynamic (per-pattern Gaussian) background
        (default ``True``).
    operation, std, truncate, device, dtype
        Passed through to the two steps (see those functions).

    Returns
    -------
    torch.Tensor
        The cleaned patterns, same leading shape as the input, rescaled to ``[0, 1]``,
        ready to hand to :meth:`~de_ebsd.index.dictionary.DictionaryIndexer.index`.
    """
    out = patterns
    if static_bg is not None:
        out = remove_static_background(
            out, static_bg, operation=operation, device=device, dtype=dtype
        )
    if dynamic:
        out = remove_dynamic_background(
            out, std=std, operation=operation, truncate=truncate, device=device, dtype=dtype
        )
    return out
