"""Bridges to orix and kikuchipy: phases, crystal maps, EMsoft master patterns."""

from __future__ import annotations

import numpy as np


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

    if not hasattr(crystal, "space_group"):
        # MasterPattern.save() stringifies metadata, so a master round-tripped
        # through HDF5 carries a repr here rather than the Crystal itself
        raise TypeError(
            f"expected a de_ebsd Crystal, got {type(crystal).__name__}; a master "
            "loaded from HDF5 keeps only a string copy of metadata['crystal'], so "
            "pass the phase (or point_group) explicitly for such a master"
        )

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
        An :class:`orix.crystal_map.Phase`, **or** a de_ebsd
        :class:`~de_ebsd.sim.Crystal` (converted via :func:`phase_from_crystal`).
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
        prop={"quality": quality_np},
        scan_unit="um",
    )


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
