"""Bridges to orix and kikuchipy: phases, crystal maps and EMsoft master patterns, and
the maps read off an indexed scan.

Every orix ``Phase`` made here has its lattice in Angstrom, orix's and diffsims'
unit, whatever the source's: a :class:`~de_ebsd.crystal.Crystal` is in nm.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

#: nm to Angstrom, for a Crystal's lattice handed to orix.
ANGSTROM_PER_NM = 10.0


def phase_from_crystal(crystal):
    """Bridge a de_ebsd :class:`~de_ebsd.sim.Crystal` to an orix ``Phase``.

    The returned :class:`orix.crystal_map.Phase` is what gives a
    :class:`~orix.crystal_map.CrystalMap` its **crystal symmetry**: orix derives
    the point group (hence the symmetry operations used for disorientation,
    IPF coloring, fundamental-zone reduction, ...) from the space-group number.
    The lattice + space group is therefore the load-bearing part; the atom sites
    are added for completeness when present.

    Parameters
    ----------
    crystal
        A :class:`de_ebsd.sim.Crystal`. Its ``space_group`` (int), lattice
        (``a, b, c`` in nm and ``alpha, beta, gamma`` in degrees) and asymmetric
        unit (``sites``) are read off.

    Returns
    -------
    orix.crystal_map.Phase
        A phase carrying the space group (and the point group orix derives from
        it) plus a diffpy ``Structure`` with the lattice, **in Angstrom**, and the
        asymmetric-unit atoms at the crystal's fractional coordinates.

    Notes
    -----
    The ``Crystal.origin`` setting (origin choice 1 vs 2 for the two-origin
    space groups) shifts the atom positions but not the point group, so it does
    not change the symmetry the ``CrystalMap`` carries; it is recorded on the
    diffpy lattice only through the fractional coordinates already stored on the
    crystal. Lattice lengths are converted from the crystal's nm to Angstrom, the
    unit orix and diffsims compute d-spacings and structure factors in.
    """
    from diffpy.structure import Atom, Lattice, Structure

    if not hasattr(crystal, "space_group"):
        # MasterPattern.save() stringifies metadata, so a master round-tripped
        # through HDF5 carries a repr here rather than the Crystal itself
        raise TypeError(
            f"expected a de_ebsd Crystal, got {type(crystal).__name__}; a master "
            "loaded from HDF5 keeps only a string copy of metadata['crystal'], so "
            "pass the phase (or point_group) explicitly for such a master"
        )

    name = getattr(crystal, "name", None) or _crystal_name(crystal)

    lengths = (ANGSTROM_PER_NM * length for length in (crystal.a, crystal.b, crystal.c))
    lattice = Lattice(*lengths, crystal.alpha, crystal.beta, crystal.gamma)
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
    return _orix_phase(name, int(crystal.space_group), structure)


def phase_from_cif(path):
    """An orix ``Phase`` read from the CIF at *path*, named after the file.

    The lattice is the file's, in Angstrom, and the atoms are the whole cell (the
    file's asymmetric unit expanded by its symmetry) at exactly the fractional
    coordinates the file gives. ``orix.crystal_map.Phase.from_cif`` would move
    them; see :func:`_orix_phase`.

    For a :class:`~de_ebsd.crystal.Crystal` to simulate with, read the file with
    :meth:`Crystal.from_cif <de_ebsd.crystal.Crystal.from_cif>` instead; this is
    the phase that gives a crystal map, an IPF key or a misorientation its
    symmetry, for a CIF that is not simulated.

    Parameters
    ----------
    path
        Path to the CIF file.

    Returns
    -------
    orix.crystal_map.Phase
        Its ``space_group`` is diffpy's number, which for a non-standard setting
        is 1000 times the setting plus the International Tables number (a
        rhombohedral cell on rhombohedral axes, ``R 3 m :R``, is 1160); the point
        group is the same either way.
    """
    from diffpy.structure.parsers import p_cif

    path = Path(path)
    parser = p_cif.P_cif()
    # newer diffpy names it parse_file and deprecates parseFile, older has only parseFile
    read = getattr(parser, "parse_file", None) or parser.parseFile
    structure = read(str(path))
    space_group = getattr(getattr(parser, "spacegroup", None), "number", None)
    if space_group is None:
        warnings.warn(f"could not read the space group from {str(path)!r}", stacklevel=2)
    return _orix_phase(path.stem, space_group, structure)


def _orix_phase(name, space_group, structure):
    """An orix ``Phase`` holding *structure* with its atoms where *structure* has them.

    orix re-expresses a structure in its own Cartesian frame (``a`` along x, ``c*``
    along z) by turning the lattice while keeping every atom's Cartesian position.
    For a cell whose ``a`` is not already along x in diffpy's frame -- every
    hexagonal, trigonal, monoclinic and triclinic cell -- the lattice turns and the
    atoms do not, so their fractional coordinates come out wrong: hexagonal BaTiO3
    put Ba at (0.577, 0.577, z) instead of (1/3, 2/3, z), and the structure lost
    the symmetry of its own space group.

    Fractional coordinates do not depend on the Cartesian frame, so the given ones
    are right in orix's frame too. Restoring them is a no-op for the cells orix
    does not turn, and stays one if orix moves the atoms itself one day.
    """
    from orix.crystal_map import Phase

    phase = Phase(name=name, space_group=space_group, structure=structure)
    for atom, given in zip(phase.structure, structure, strict=True):
        atom.xyz = np.array(given.xyz, dtype=np.float64)
    return phase


def _crystal_name(crystal) -> str:
    sites = getattr(crystal, "sites", ())
    elements = []
    for site in sites:
        if site.element not in elements:
            elements.append(site.element)
    return "".join(elements) if elements else f"sg{int(crystal.space_group)}"


def to_crystal_map(rotations, quality=None, shape=None, phase=None, step=1.0):
    """Assemble indexed orientations + a quality metric into an orix ``CrystalMap``.

    Parameters
    ----------
    rotations : orix.quaternion.Rotation or array
        A **flat** orix ``Rotation`` of length ``rows * cols`` in **row-major**
        scan order (point ``(r, c)`` is at flat index ``r * cols + c``), e.g. the
        concatenated output of :meth:`SphericalIndexer.index` over the scan; or
        Bunge Euler angles in radians, ``(rows, cols, 3)`` or row-major
        ``(rows * cols, 3)``.
    quality
        A length-``rows * cols`` real array or torch tensor of the per-point peak
        score, stored under ``prop["quality"]`` as a plain numpy array; ``None``
        stores none.
    shape : tuple of int
        ``(rows, cols)`` of the scan; taken from ``(rows, cols, 3)`` Euler angles
        when ``None``.
    phase
        An :class:`orix.crystal_map.Phase`, **or** a de_ebsd
        :class:`~de_ebsd.sim.Crystal` (converted via :func:`phase_from_crystal`).
        This carries the crystal symmetry of the returned map. Required.
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
    from orix.quaternion import Rotation

    if not isinstance(rotations, Rotation):
        euler = np.asarray(rotations, dtype=np.float64)
        if shape is None and euler.ndim == 3:
            shape = euler.shape[:2]
        rotations = Rotation.from_euler(euler.reshape(-1, 3))
    if shape is None:
        raise ValueError("shape=(rows, cols) is required unless Euler angles are (rows, cols, 3)")
    if phase is None:
        raise ValueError("phase is required: it gives the map its crystal symmetry")
    rows, cols = int(shape[0]), int(shape[1])
    n = rows * cols
    if rotations.size != n:
        raise ValueError(
            f"rotations has length {rotations.size}, expected rows*cols = {n} for "
            f"shape {(rows, cols)}"
        )

    prop = {}
    if quality is not None:
        quality_np = _to_numpy(quality).reshape(-1)
        if quality_np.size != n:
            raise ValueError(
                f"quality has length {quality_np.size}, expected rows*cols = {n} for "
                f"shape {(rows, cols)}"
            )
        prop["quality"] = quality_np

    if not _is_orix_phase(phase):
        phase = phase_from_crystal(phase)  # accept a de_ebsd Crystal

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
        prop=prop,
        scan_unit="um",
    )


def orientation_similarity_map(indices, shape) -> np.ndarray:
    """How much each point's ranked match list agrees with its four neighbors'.

    A score map shows how well a pattern matched *something*, and a confidently
    wrong match scores as high as a right one; agreement with the neighborhood is
    what exposes it. Each point's value is the mean, over its neighbors, of the
    fraction of its *k* best dictionary entries that are among the neighbor's.

    Parameters
    ----------
    indices
        ``(rows * cols, k)`` dictionary indices, best first, in row-major scan order:
        :meth:`DictionaryIndexer.match <de_ebsd.index.dictionary.DictionaryIndexer.match>`
        with ``keep=k``. With ``k = 1`` it says only whether the best matches agree.
    shape
        ``(rows, cols)`` of the scan.

    Returns
    -------
    numpy.ndarray
        ``(rows, cols)`` in ``[0, 1]``.
    """
    idx = _to_numpy(indices)
    if idx.ndim != 2:
        raise ValueError(f"indices must be (rows * cols, k), got shape {idx.shape}")
    if shape is None:
        raise ValueError("shape=(rows, cols) is required to find the neighbors")
    rows, cols = int(shape[0]), int(shape[1])
    n, k = idx.shape
    if rows * cols != n:
        raise ValueError(f"shape {(rows, cols)} does not match {n} points")
    grid = idx.reshape(rows, cols, k)
    total = np.zeros((rows, cols))
    count = np.zeros((rows, cols))
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        here = (slice(max(0, dr), rows + min(0, dr)), slice(max(0, dc), cols + min(0, dc)))
        there = (slice(max(0, -dr), rows + min(0, -dr)), slice(max(0, -dc), cols + min(0, -dc)))
        shared = (grid[here][..., :, None] == grid[there][..., None, :]).any(-1)
        total[here] += shared.sum(-1) / float(k)
        count[here] += 1.0
    return total / np.maximum(count, 1.0)


def merge_phases(orientations, scores, shape=None):
    """Choose each point's phase from per-phase indexing runs: the best score wins.

    Parameters
    ----------
    orientations
        One per phase, in the order of *scores*: orix ``Rotation`` objects, or arrays
        whose leading axes are the scan (Euler angles ``(..., 3)``, quaternions
        ``(..., 4)``), all covering the same points.
    scores
        One per phase: the best match score at each point.
    shape
        ``(rows, cols)`` to shape the results to; ``None`` leaves them flat.

    Returns
    -------
    orientations
        The winning phase's orientation at each point: a ``Rotation`` when given
        ``Rotation`` objects, else an array.
    phase_id : numpy.ndarray
        The index of the winning phase at each point.
    score : numpy.ndarray
        The winning score at each point.
    """
    from orix.quaternion import Rotation

    as_rotations = all(isinstance(o, Rotation) for o in orientations)
    values = [np.asarray(o.data if as_rotations else o, dtype=np.float64) for o in orientations]
    width = 4 if as_rotations else values[0].shape[-1]
    values = [v.reshape(-1, width) for v in values]
    raw_scores = [_to_numpy(s).astype(np.float64).reshape(-1) for s in scores]
    if len(values) != len(raw_scores):
        raise ValueError(f"{len(values)} phase maps but {len(raw_scores)} score arrays")
    # checked before stacking, whose own error does not say which phase disagreed
    sizes = {len(v) for v in values} | {s.size for s in raw_scores}
    if len(sizes) != 1:
        raise ValueError(f"every phase map must cover the same positions, got {sorted(sizes)}")
    score = np.stack(raw_scores)  # (n_phase, P)
    phase_id = score.argmax(0)
    points = np.arange(score.shape[1])
    best = np.stack(values)[phase_id, points]
    score = score[phase_id, points]
    if shape is not None:
        shape = (int(shape[0]), int(shape[1]))
        best = best.reshape(shape + (width,))
        phase_id = phase_id.reshape(shape)
        score = score.reshape(shape)
    return (Rotation(best) if as_rotations else best), phase_id, score


def _is_orix_phase(obj) -> bool:
    from orix.crystal_map import Phase

    return isinstance(obj, Phase)


def _to_numpy(x) -> np.ndarray:
    detach = getattr(x, "detach", None)
    if detach is not None:
        x = detach().cpu().numpy()
    return np.asarray(x)


def master_from_emsoft(path, device=None):
    """Load an EMsoft EBSD master-pattern HDF5 file as a de_ebsd ``MasterPattern``.

    Reads the square-Lambert masters ``EMData/EBSDmaster/mLPNH``/``mLPSH``, summing
    their leading asymmetric-unit **site** axis (indexing only the first site is a
    known bug); an already-collapsed ``(n_energy, n, n)`` array is used as-is. All
    ``EkeVs`` energy bins are kept -- pick one with
    ``SphericalIndexer(energy_kev=...)``.

    ``CrystalData`` is rebuilt into a :class:`de_ebsd.crystal.Crystal` and stashed
    in ``metadata["crystal"]`` so the indexer can auto-derive the phase. Atomic
    numbers map to symbols via ``ELEMENT_SYMBOLS[Z - 1]``, and the space-group
    setting (1 or 2) becomes ``Crystal(origin=...)``, defaulting to 2::

        master = master_from_emsoft("Si_MC.h5")
        xmap = SphericalIndexer(master, detector).index_scan(patterns, shape=shape)

    Parameters
    ----------
    path
        Path to the EMsoft master-pattern HDF5 file.
    device
        Optional torch device the returned master is moved to.

    Returns
    -------
    de_ebsd.sim.MasterPattern
        ``data`` of shape ``(n_energy, 2, n, n)`` (axis 1 = hemisphere, north
        first), with ``metadata`` carrying ``"crystal"``, ``"dmin"``, ``"bethe"``
        and ``"source"``. A missing or malformed ``CrystalData`` group leaves
        ``metadata["crystal_error"]`` instead, rather than sinking the whole load.
    """
    import h5py
    import torch

    from de_ebsd.sim import MasterPattern

    path = str(path)
    with h5py.File(path, "r") as f:
        mlpnh = np.asarray(f["EMData/EBSDmaster/mLPNH"], dtype=np.float32)
        mlpsh = np.asarray(f["EMData/EBSDmaster/mLPSH"], dtype=np.float32)
        energies = np.asarray(f["EMData/EBSDmaster/EkeVs"], dtype=np.float64).ravel()
        dmin = _read_emsoft_scalar(f, "NMLparameters/EBSDMasterNameList/dmin", float)
        bethe = _read_emsoft_array(f, "EMData/EBSDmaster/BetheParameters")
        crystal_arrays = _read_crystal_arrays(f)

    # site-summed hemispheres -> (n_energy, 2, n, n), north = 0 / south = 1.
    north = _sum_over_sites(mlpnh)
    south = _sum_over_sites(mlpsh)
    data = torch.as_tensor(np.stack([north, south], axis=1).astype(np.float32))

    metadata = {
        "backend": "EMsoft (imported)",
        "source": path,
        "dmin": dmin,
        "bethe": tuple(float(x) for x in bethe) if bethe is not None else None,
    }

    # Rebuild the crystal for phase auto-derivation, but never let a malformed or
    # partial CrystalData sink the load -- record the reason and return the master.
    crystal, crystal_error = _crystal_from_arrays(crystal_arrays)
    if crystal is not None:
        metadata["crystal"] = crystal
    if crystal_error is not None:
        metadata["crystal_error"] = crystal_error

    master = MasterPattern(data=data, energies_kev=energies, metadata=metadata)
    if device is not None:
        master = master.to(device)
    return master


def _sum_over_sites(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 4:
        return array.sum(axis=0)
    if array.ndim == 3:
        return array
    raise ValueError(
        f"mLP master must be 3-D (n_energy, n, n) or 4-D (n_sites, n_energy, n, n), "
        f"got ndim {array.ndim}"
    )


def _read_emsoft_scalar(f, key: str, cast):
    try:
        return cast(np.asarray(f[key]).ravel()[0])
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def _read_emsoft_array(f, key: str):
    try:
        return np.asarray(f[key], dtype=np.float64).ravel()
    except (KeyError, ValueError, TypeError):
        return None


def _read_crystal_arrays(f):
    try:
        cd = f["CrystalData"]
        arrays = {
            "lattice": np.asarray(cd["LatticeParameters"], dtype=np.float64).ravel(),
            "atom_data": np.asarray(cd["AtomData"], dtype=np.float64),
            "atom_types": np.asarray(cd["Atomtypes"]).ravel(),
            "space_group": int(np.asarray(cd["SpaceGroupNumber"]).ravel()[0]),
        }
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    # setting is optional; origin choice 2 is the modern EMsoft default.
    setting = _read_emsoft_scalar(cd, "SpaceGroupSetting", int)
    arrays["setting"] = setting if setting in (1, 2) else 2
    return arrays


def _crystal_from_arrays(arrays):
    if arrays is None:
        return None, "CrystalData group missing or unreadable"
    try:
        from de_ebsd.crystal import AtomSite, Crystal
        from de_ebsd.crystal.data.scattering_tables import ELEMENT_SYMBOLS

        lattice = arrays["lattice"]
        if lattice.size < 6:
            return None, f"LatticeParameters has {lattice.size} values, expected 6"
        a, b, c, alpha, beta, gamma = (float(x) for x in lattice[:6])

        atom_data = arrays["atom_data"]
        atom_types = arrays["atom_types"]
        n_atoms = min(atom_data.shape[1], atom_types.shape[0])
        if n_atoms < 1:
            return None, "CrystalData has no atoms"

        sites = []
        for j in range(n_atoms):
            z = int(atom_types[j])
            if not 1 <= z <= len(ELEMENT_SYMBOLS):
                return None, f"atomic number {z} (site {j}) is out of the known range"
            x, y, z_frac, occupancy, dw = (float(v) for v in atom_data[:5, j])
            sites.append(
                AtomSite(
                    element=ELEMENT_SYMBOLS[z - 1],
                    position=(x, y, z_frac),
                    occupancy=occupancy,
                    debye_waller=dw,
                )
            )

        crystal = Crystal(
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            space_group=int(arrays["space_group"]),
            sites=sites,
            origin=int(arrays["setting"]),
        )
        return crystal, None
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        return None, f"could not build crystal from CrystalData: {exc}"
