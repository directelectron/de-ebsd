"""Shared anyplotlib primitives.

Every ``.show()`` in torchebsd renders through :mod:`torchebsd.viz`, so the
package has one way to draw. Helpers return an ``anyplotlib.Figure``:
interactive in Jupyter, exportable via ``figure.save_html(path)``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["apl", "bar_figure", "image_figure", "label", "to_uint8_rgb"]


def apl():
    """Import anyplotlib lazily so headless jobs never pay for the display stack."""
    try:
        import anyplotlib
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "anyplotlib is required for .show(); install it with `pip install anyplotlib`"
        ) from error
    return anyplotlib


def label(panel, title: str) -> None:
    """Attach a title to a plot panel if the panel supports one."""
    if not title:
        return
    for method in ("set_title", "set_label"):
        setter = getattr(panel, method, None)
        if callable(setter):
            try:
                setter(title)
            except Exception:  # noqa: BLE001 - a missing title must never break .show()
                pass
            return


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize a 2D array to a grayscale (H, W, 3) uint8 texture."""
    image = np.asarray(image, dtype=np.float64)
    lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
    scale = 255.0 / (hi - lo) if hi > lo else 0.0
    gray = ((image - lo) * scale).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def bar_figure(values, labels, title: str = "", colors=None):
    """Bar chart of one value per labeled item (structure factors, reflections)."""
    values = np.asarray(values, dtype=np.float64)
    fig, ax = apl().subplots(1, 1, figsize=(max(560, 18 * len(values)), 340))
    kwargs = {"x_labels": list(labels), "show_values": False}
    if colors is not None:
        kwargs["colors"] = list(colors)
    else:
        kwargs["color"] = "#4fc3f7"
    label(ax.bar(values, **kwargs), title)
    return fig


def image_figure(image: np.ndarray, title: str = "", cmap: str = "gray"):
    """Single image panel. Row 0 is drawn at the bottom (Lambert +Y up)."""
    image = np.asarray(image, dtype=np.float64)
    fig, ax = apl().subplots(1, 1, figsize=(520, 520))
    label(ax.imshow(image, cmap=cmap, origin="lower"), title)
    return fig
