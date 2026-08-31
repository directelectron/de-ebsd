"""Master-pattern, crystal-structure and direction-sampling figures."""

from __future__ import annotations

import numpy as np

from torchebsd.viz.figures import apl, label, to_uint8_rgb

__all__ = [
    "crystal_structure_figure",
    "hemispheres_figure",
    "phase_overview_figure",
    "sampling_sphere_figure",
    "sphere_figure",
]

_ELEMENT_COLORS = {
    "H": (1.00, 1.00, 1.00),
    "C": (0.36, 0.40, 0.45),
    "N": (0.25, 0.40, 0.95),
    "O": (0.93, 0.20, 0.22),
    "Al": (0.75, 0.65, 0.65),
    "Si": (0.94, 0.66, 0.48),
    "Ti": (0.75, 0.76, 0.78),
    "Cr": (0.54, 0.60, 0.78),
    "Mn": (0.61, 0.48, 0.78),
    "Fe": (0.88, 0.40, 0.20),
    "Co": (0.94, 0.56, 0.63),
    "Ni": (0.31, 0.82, 0.31),
    "Cu": (0.78, 0.50, 0.20),
    "Zn": (0.49, 0.50, 0.69),
}
_DEFAULT_COLOR = (0.40, 0.75, 0.90)


def hemispheres_figure(north: np.ndarray, south: np.ndarray, title: str = "", cmap: str = "gray"):
    """North and south hemisphere maps side by side."""
    fig, axes = apl().subplots(1, 2, figsize=(940, 470))
    ax_n, ax_s = np.ravel(axes)[:2]
    panel_n = ax_n.imshow(np.asarray(north, dtype=np.float64), cmap=cmap, origin="lower")
    panel_s = ax_s.imshow(np.asarray(south, dtype=np.float64), cmap=cmap, origin="lower")
    label(panel_n, f"{title} - north" if title else "north")
    label(panel_s, f"{title} - south" if title else "south")
    return fig


def sphere_figure(
    equirectangular: np.ndarray,
    title: str = "",
    azimuth: float = -60.0,
    elevation: float = 30.0,
    n_grid: int = 96,
):
    """The master pattern textured onto an interactive 3D sphere.

    ``equirectangular`` is sampled on a latitude/longitude grid: row 0 is the
    north pole, columns span longitude -pi..pi (``MasterPattern.to_equirectangular``).
    """
    fig, ax = apl().subplots(1, 1, figsize=(560, 560))
    _sphere_panel(ax, equirectangular, title, azimuth, elevation, n_grid)
    return fig


def _sphere_panel(ax, equirectangular, title, azimuth, elevation, n_grid):
    theta = np.linspace(0.0, np.pi, n_grid)
    phi = np.linspace(-np.pi, np.pi, 2 * n_grid)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    x = np.sin(t) * np.cos(p)
    y = np.sin(t) * np.sin(p)
    z = np.cos(t)

    surface = ax.plot_surface(
        x, y, z, bounds=((-1.0, 1.0),) * 3, azimuth=azimuth, elevation=elevation, gpu="auto"
    )
    surface.set_texture(to_uint8_rgb(equirectangular), cull_backfaces=True)
    label(surface, title)
    return surface


def phase_overview_figure(crystal, equirectangular: np.ndarray, *, name="Phase", n_grid=96):
    """Dominant master pattern and compact structure in one figure.

    One figure means one anywidget in Jupyter and one registration through
    SpyDE's Electron bridge.
    """
    fig, axes = apl().subplots(1, 2, figsize=(820, 400), width_ratios=[1.45, 1.0])
    master_ax, crystal_ax = np.ravel(axes)[:2]
    _sphere_panel(master_ax, equirectangular, f"{name} master pattern", -60.0, 30.0, n_grid)
    _crystal_structure_panel(crystal_ax, crystal, f"{name} structure")
    return fig


def crystal_structure_figure(crystal, title: str = ""):
    """Expanded atoms of one unit cell as a 3-D plot, in Cartesian nanometres."""
    fig, ax = apl().subplots(1, 1, figsize=(480, 420))
    _crystal_structure_panel(ax, crystal, title or "Crystal structure")
    return fig


def _crystal_structure_panel(ax, crystal, title: str):
    fractional = np.concatenate(crystal.positions, axis=0)
    elements = [
        site.element
        for site, positions in zip(crystal.sites, crystal.positions, strict=True)
        for _ in positions
    ]
    cartesian = fractional @ crystal.direct_structure_matrix.T
    colors = np.asarray([_ELEMENT_COLORS.get(e, _DEFAULT_COLOR) for e in elements])

    # include the cell corners so non-orthogonal cells are not clipped
    corners = np.asarray([[i, j, k] for i in (0.0, 1.0) for j in (0.0, 1.0) for k in (0.0, 1.0)])
    extent = np.vstack((cartesian, corners @ crystal.direct_structure_matrix.T))
    lo, hi = extent.min(axis=0), extent.max(axis=0)
    padding = 0.08 * np.maximum(hi - lo, 1e-6)
    bounds = tuple((float(a), float(b)) for a, b in zip(lo - padding, hi + padding, strict=True))

    points = ax.scatter3d(
        cartesian[:, 0],
        cartesian[:, 1],
        cartesian[:, 2],
        colors=colors,
        point_size=max(5.0, min(12.0, 48.0 / np.sqrt(len(cartesian)))),
        bounds=bounds,
        x_label="x (nm)",
        y_label="y (nm)",
        z_label="z (nm)",
        azimuth=-55.0,
        elevation=24.0,
        gpu="auto",
    )
    label(points, title)
    return points


def sampling_sphere_figure(directions: np.ndarray, title: str = "Incident-direction sampling"):
    """Sampled unit directions on an interactive reference sphere.

    The first half of ``directions`` is colored as the northern hemisphere and
    the second half as the southern, matching ``KVectorGrid.flat``.
    """
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if directions.size == 0 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (N, 3) and contain at least one vector")

    n_north = directions.shape[0] // 2
    colors = np.empty((directions.shape[0], 3), dtype=np.float64)
    colors[:n_north] = (0.35, 0.71, 0.98)
    colors[n_north:] = (0.98, 0.63, 0.36)

    fig, ax = apl().subplots(1, 1, figsize=(560, 560))
    points = ax.scatter3d(
        directions[:, 0],
        directions[:, 1],
        directions[:, 2],
        colors=colors,
        point_size=3.0,
        bounds=((-1.0, 1.0),) * 3,
        azimuth=-60.0,
        elevation=30.0,
        gpu="auto",
    )
    points.set_sphere(radius=1.0, color="#9aa4b2", alpha=0.10, wireframe=True)
    label(points, title)
    return fig
