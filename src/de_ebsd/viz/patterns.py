"""Master-pattern, crystal-structure and direction-sampling figures."""

from __future__ import annotations

import itertools

import numpy as np

from de_ebsd.crystal.lattice import element_symbol
from de_ebsd.viz.figures import apl, label, to_uint8_rgb

__all__ = [
    "ELEMENT_COLORS",
    "crystal_structure_figure",
    "element_color",
    "hemispheres_figure",
    "phase_overview_figure",
    "sampling_sphere_figure",
    "sphere_figure",
    "unit_cell",
]

#: Display colors of common elements, Jmol's scheme, as hex; others get
#: :data:`DEFAULT_COLOR`. One table for every figure that draws atoms.
ELEMENT_COLORS = {
    "H": "#ffffff", "C": "#909090", "N": "#3050f8", "O": "#ff0d0d",
    "F": "#90e050", "Na": "#ab5cf2", "Mg": "#8aff00", "Al": "#bfa6a6",
    "Si": "#f0c8a0", "P": "#ff8000", "S": "#ffff30", "Cl": "#1ff01f",
    "K": "#8f40d4", "Ca": "#3dff00", "Ti": "#bfc2c7", "V": "#a6a6ab",
    "Cr": "#8a99c7", "Mn": "#9c7ac7", "Fe": "#e06633", "Co": "#f090a0",
    "Ni": "#50d050", "Cu": "#c88033", "Zn": "#7d80b0", "Ga": "#c28f8f",
    "Ge": "#668f8f", "As": "#bd80e3", "Zr": "#94e0e0", "Nb": "#73c2c9",
    "Mo": "#54b5b5", "Ag": "#c0c0c0", "Sn": "#668080", "Hf": "#4dc2ff",
    "Ta": "#4da6ff", "W": "#2194d6", "Pt": "#d0d0e0", "Au": "#ffd123",
    "Pb": "#575961",
}  # fmt: skip
DEFAULT_COLOR = "#66bfe6"


def element_color(symbol: str) -> tuple[float, float, float]:
    """The display color of *symbol* (or of a CIF label such as ``"Fe2+"``) as RGB
    in ``[0, 1]``."""
    try:
        symbol = element_symbol(symbol)
    except ValueError:
        pass
    code = ELEMENT_COLORS.get(symbol, DEFAULT_COLOR)
    return tuple(int(code[i : i + 2], 16) / 255.0 for i in (1, 3, 5))


def unit_cell(crystal) -> tuple[np.ndarray, list[str], np.ndarray]:
    """``(positions, elements, edges)`` of a :class:`~de_ebsd.crystal.Crystal`'s unit cell.

    *positions* ``(N, 3)`` are Cartesian, in nm, in the crystal frame
    (``Crystal.direct_structure_matrix``): every atom of the cell plus its periodic
    images on the faces, edges and corners, so a cubic cell shows its 8 corner atoms
    rather than one. *elements* are their symbols; *edges* ``(12, 2, 3)`` the cell's
    edges, in nm.
    """
    base = np.asarray(crystal.direct_structure_matrix, dtype=np.float64)  # columns a, b, c
    fractional, elements = [], []
    for site, orbit in zip(crystal.sites, crystal.positions, strict=True):
        for xyz in np.asarray(orbit, dtype=np.float64).reshape(-1, 3):
            xyz = np.mod(xyz, 1.0)
            xyz[np.isclose(xyz, 1.0, atol=1e-6)] = 0.0
            # an atom on a face, edge or corner has images on the opposite side
            shifts = [(0.0, 1.0) if np.isclose(v, 0.0, atol=1e-6) else (0.0,) for v in xyz]
            for shift in itertools.product(*shifts):
                image = xyz + shift
                if not any(np.allclose(image, seen, atol=1e-6) for seen in fractional):
                    fractional.append(image)
                    elements.append(site.element)
    positions = np.asarray(fractional).reshape(-1, 3) @ base.T
    corners = np.array(list(itertools.product((0.0, 1.0), repeat=3)))
    edges = np.array(
        [
            [corners[i] @ base.T, corners[j] @ base.T]
            for i, j in itertools.combinations(range(8), 2)
            if np.abs(corners[i] - corners[j]).sum() == 1
        ]
    )
    return positions, elements, edges


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
    """One unit cell's atoms, with their images on its faces, edges and corners
    (:func:`unit_cell`), as a 3-D plot in Cartesian nanometres."""
    fig, ax = apl().subplots(1, 1, figsize=(480, 420))
    _crystal_structure_panel(ax, crystal, title or "Crystal structure")
    return fig


def _crystal_structure_panel(ax, crystal, title: str):
    cartesian, elements, _ = unit_cell(crystal)
    colors = np.asarray([element_color(e) for e in elements])

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
