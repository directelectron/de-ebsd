"""The anyplotlib display layer.

Every ``.show()`` in torchebsd renders through these helpers, so the whole
package has exactly one way to draw. Each helper returns an
``anyplotlib.Figure`` -- interactive in Jupyter (``_repr_html_``), exportable
anywhere via ``figure.save_html(path)`` / ``figure.to_html()``.

anyplotlib is imported lazily so that headless compute jobs never pay for
(or require) the display stack.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bar_figure",
    "image_figure",
    "hemispheres_figure",
    "sphere_figure",
    "to_uint8_rgb",
]


def _apl():
    try:
        import anyplotlib as apl
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "anyplotlib is required for .show(); install it with `pip install anyplotlib`"
        ) from error
    return apl


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize a 2D array to a grayscale (H, W, 3) uint8 texture."""
    image = np.asarray(image, dtype=np.float64)
    lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
    scale = 255.0 / (hi - lo) if hi > lo else 0.0
    gray = ((image - lo) * scale).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def bar_figure(values, labels, title: str = "", colors=None):
    """Bar chart of one value per labeled item (structure factors, reflections)."""
    apl = _apl()
    values = np.asarray(values, dtype=np.float64)
    fig, ax = apl.subplots(1, 1, figsize=(max(560, 18 * len(values)), 340))
    kwargs = {"x_labels": list(labels), "show_values": False}
    if colors is not None:
        kwargs["colors"] = list(colors)
    else:
        kwargs["color"] = "#4fc3f7"
    bar = ax.bar(values, **kwargs)
    _label(bar, title)
    return fig


def image_figure(image: np.ndarray, title: str = "", cmap: str = "gray"):
    """Single image panel. Row 0 is drawn at the bottom (Lambert +Y up)."""
    apl = _apl()
    image = np.asarray(image, dtype=np.float64)
    fig, ax = apl.subplots(1, 1, figsize=(520, 520))
    panel = ax.imshow(image, cmap=cmap, origin="lower")
    _label(panel, title)
    return fig


def hemispheres_figure(north: np.ndarray, south: np.ndarray, title: str = "", cmap: str = "gray"):
    """North and south hemisphere maps side by side."""
    apl = _apl()
    fig, axes = apl.subplots(1, 2, figsize=(940, 470))
    ax_n, ax_s = np.ravel(axes)[:2]
    panel_n = ax_n.imshow(np.asarray(north, dtype=np.float64), cmap=cmap, origin="lower")
    panel_s = ax_s.imshow(np.asarray(south, dtype=np.float64), cmap=cmap, origin="lower")
    _label(panel_n, f"{title} - north" if title else "north")
    _label(panel_s, f"{title} - south" if title else "south")
    return fig


def sphere_figure(
    equirectangular: np.ndarray,
    title: str = "",
    azimuth: float = -60.0,
    elevation: float = 30.0,
    n_grid: int = 96,
):
    """The master pattern textured onto an interactive 3D sphere.

    Parameters
    ----------
    equirectangular
        Intensity image sampled on a latitude/longitude grid; row 0 is the
        north pole, columns span longitude -pi..pi (see
        ``MasterPattern.to_equirectangular``).
    """
    apl = _apl()
    theta = np.linspace(0.0, np.pi, n_grid)
    phi = np.linspace(-np.pi, np.pi, 2 * n_grid)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    x = np.sin(t) * np.cos(p)
    y = np.sin(t) * np.sin(p)
    z = np.cos(t)

    fig, ax = apl.subplots(1, 1, figsize=(560, 560))
    surface = ax.plot_surface(
        x, y, z, bounds=((-1.0, 1.0),) * 3, azimuth=azimuth, elevation=elevation, gpu="auto"
    )
    surface.set_texture(to_uint8_rgb(equirectangular), cull_backfaces=True)
    _label(surface, title)
    return fig


def _label(panel, title: str) -> None:
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
