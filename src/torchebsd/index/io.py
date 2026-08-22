"""orix-native output + array ingest -- the ecosystem handoff.

torchebsd is the *foundation*: it emits orix objects and takes plain pattern
arrays, so it never imports kikuchipy. The intended direction is that kikuchipy
(and other tools) depend on torchebsd -- they pass their pattern data + detector
in and consume the orix ``Rotation`` / ``CrystalMap`` out. See
``INDEXING_DESIGN.md`` (milestones I5b + I8).

This module holds the two orix bridges scan indexing needs (I5b):

* :func:`phase_from_crystal` -- turn a torchebsd :class:`~torchebsd.sim.Crystal`
  into an :class:`orix.crystal_map.Phase`, which is what gives a ``CrystalMap``
  its crystal symmetry (orix derives the point group from the space group);
* :func:`to_crystal_map` -- assemble a flat orix ``Rotation`` + a per-point
  quality metric into an :class:`orix.crystal_map.CrystalMap` laid out on the
  scan grid.

``master_from_emsoft`` (EMsoft import) stays a stub -- that is milestone I8.
"""

from __future__ import annotations

import numpy as np


def phase_from_crystal(crystal):
    """Bridge a torchebsd :class:`~torchebsd.sim.Crystal` to an orix ``Phase``.

    The returned :class:`orix.crystal_map.Phase` is what gives a
    :class:`~orix.crystal_map.CrystalMap` its **crystal symmetry**: orix derives
    the point group (hence the symmetry operations used for disorientation,
    IPF coloring, fundamental-zone reduction, ...) from the space-group number.
    The lattice + space group is therefore the load-bearing part; the atom sites
    are added for completeness when present.

    Parameters
    ----------
    crystal
        A :class:`torchebsd.sim.Crystal`. Its ``space_group`` (int), lattice
        (``a, b, c`` in nm and ``alpha, beta, gamma`` in degrees) and asymmetric
        unit (``sites``) are read off.

    Returns
    -------
    orix.crystal_map.Phase
        A phase carrying the space group (and the point group orix derives from
        it) plus a diffpy ``Structure`` with the lattice and the asymmetric-unit
        atoms.

    Notes
    -----
    The ``Crystal.origin`` setting (origin choice 1 vs 2 for the two-origin
    space groups) shifts the atom positions but not the point group, so it does
    not change the symmetry the ``CrystalMap`` carries; it is recorded on the
    diffpy lattice only through the fractional coordinates already stored on the
    crystal. Lattice lengths are passed through in nm (the crystal's own unit).
    """
    from diffpy.structure import Atom, Lattice, Structure
    from orix.crystal_map import Phase

    name = getattr(crystal, "name", None) or _crystal_name(crystal)

    lattice = Lattice(
        crystal.a, crystal.b, crystal.c, crystal.alpha, crystal.beta, crystal.gamma
    )
    atoms = []
    for site in getattr(crystal, "sites", ()):  # asymmetric unit
        atoms.append(
            Atom(
                atype=site.element,
                xyz=np.asarray(site.position, dtype=np.float64),
                occupancy=float(site.occupancy),
            )
        )
    structure = Structure(atoms=atoms, lattice=lattice)

    # orix derives the point group from the space-group number -- this is the
    # symmetry the CrystalMap inherits.
    return Phase(name=name, space_group=int(crystal.space_group), structure=structure)


def _crystal_name(crystal) -> str:
    """A readable phase name from the crystal's asymmetric-unit elements."""
    sites = getattr(crystal, "sites", ())
    elements = []
    for site in sites:
        if site.element not in elements:
            elements.append(site.element)
    return "".join(elements) if elements else f"sg{int(crystal.space_group)}"


def to_crystal_map(rotations, quality, shape, phase, step=1.0):
    """Assemble indexed orientations + a quality metric into an orix ``CrystalMap``.

    Parameters
    ----------
    rotations : orix.quaternion.Rotation
        A **flat** orix ``Rotation`` of length ``rows * cols`` in **row-major**
        scan order (point ``(r, c)`` is at flat index ``r * cols + c``), e.g. the
        concatenated output of :meth:`SphericalIndexer.index` over the scan.
    quality
        A length-``rows * cols`` real array or torch tensor of the per-point peak
        score, stored under ``prop["quality"]`` as a plain numpy array.
    shape : tuple of int
        ``(rows, cols)`` of the scan.
    phase
        An :class:`orix.crystal_map.Phase`, **or** a torchebsd
        :class:`~torchebsd.sim.Crystal` (converted via :func:`phase_from_crystal`).
        This carries the crystal symmetry of the returned map.
    step : float, optional
        Scan step size (same value for both axes). Default ``1.0``.

    Returns
    -------
    orix.crystal_map.CrystalMap
        A single-phase map: ``x = column index * step``, ``y = row index * step``
        following orix conventions, so ``xmap.plot()``/``get_map_data`` reshape to
        the correct ``(rows, cols)`` 2-D layout and ``xmap[r, c]`` selects the
        row-major flat index ``r * cols + c``. Every point has ``phase_id == 0``.

    Notes
    -----
    The scan is **single-phase** here (``phase_id`` all zero). Multi-phase scans
    (a phase list plus a per-point phase assignment) are a future extension.
    """
    from orix.crystal_map import CrystalMap, PhaseList, create_coordinate_arrays

    rows, cols = int(shape[0]), int(shape[1])
    n = rows * cols
    if rotations.size != n:
        raise ValueError(
            f"rotations has length {rotations.size}, expected rows*cols = {n} for "
            f"shape {(rows, cols)}"
        )

    quality_np = _to_numpy(quality).reshape(-1)
    if quality_np.size != n:
        raise ValueError(
            f"quality has length {quality_np.size}, expected rows*cols = {n} for "
            f"shape {(rows, cols)}"
        )

    if not _is_orix_phase(phase):
        phase = phase_from_crystal(phase)  # accept a torchebsd Crystal

    # idiomatic orix x/y: x varies fastest (columns), y is constant per row, so
    # the flat arrays ravel row-major and match the flat rotation order above.
    coords, n_coords = create_coordinate_arrays((rows, cols), (float(step), float(step)))
    assert n_coords == n  # create_coordinate_arrays agrees with rows*cols

    return CrystalMap(
        rotations=rotations,
        phase_id=np.zeros(n, dtype=int),
        x=coords["x"],
        y=coords["y"],
        phase_list=PhaseList(phase),
        prop={"quality": quality_np},
        scan_unit="um",
    )


def _is_orix_phase(obj) -> bool:
    """True if ``obj`` is an orix ``Phase`` (without importing orix eagerly here)."""
    from orix.crystal_map import Phase

    return isinstance(obj, Phase)


def _to_numpy(x) -> np.ndarray:
    """Detach + move a torch tensor to a plain numpy array (pass numpy through)."""
    detach = getattr(x, "detach", None)
    if detach is not None:
        x = detach().cpu().numpy()
    return np.asarray(x)


def master_from_emsoft(path):
    """Load an EMsoft master pattern (via ``torchebsd.sim``) for indexing."""
    raise NotImplementedError("I8: EMsoft master import")
