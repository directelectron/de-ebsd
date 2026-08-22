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

* :func:`master_from_emsoft` -- load an EMsoft EBSD master-pattern HDF5 file into
  a torchebsd :class:`~torchebsd.sim.MasterPattern`, rebuilding the crystal into
  ``metadata["crystal"]`` so a loaded EMsoft master flows straight into
  :meth:`SphericalIndexer.index_scan` with the phase auto-derived (milestone I8).
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


def master_from_emsoft(path, energy_kev=None, device=None):
    """Load an EMsoft EBSD master-pattern HDF5 file as a torchebsd ``MasterPattern``.

    This closes the real-data loop (milestone I8): an EMsoft master on disk flows
    straight into :meth:`SphericalIndexer.index_scan` with the phase auto-derived,
    so an EMsoft ``*_MC.h5`` (or master-pattern) run indexes into an orix
    ``CrystalMap`` with no manual crystal bookkeeping.

    What it reads (via ``h5py``), all group paths verbatim from an EMsoft file:

    * ``EMData/EBSDmaster/mLPNH`` / ``mLPSH`` -- the north/south square-Lambert
      masters, shape ``(n_sites, n_energy, n, n)`` with a leading **site** axis
      (one component per asymmetric-unit atom). The components are **summed over
      the site axis** (indexing only the first site is a known bug); an
      already-collapsed ``(n_energy, n, n)`` array is used as-is.
    * ``EMData/EBSDmaster/EkeVs`` -- the energy list (keV); **all** bins are kept.
    * ``NMLparameters/EBSDMasterNameList/dmin`` -- ``dmin`` (metadata, provenance).
    * ``EMData/EBSDmaster/BetheParameters`` -- ``(c1, c2, c3, sgdbdiff)`` (metadata).
    * ``CrystalData/{LatticeParameters, AtomData, Atomtypes, SpaceGroupNumber,
      SpaceGroupSetting}`` -- the crystal, rebuilt into a
      :class:`torchebsd.sim.Crystal` and stashed in ``metadata["crystal"]`` so the
      indexer can auto-derive the phase (via :func:`phase_from_crystal`). Lattice
      lengths are in nm and angles in degrees (torchebsd's own convention, passed
      through as-is); atomic **numbers** ``Z`` map to element symbols via
      ``ELEMENT_SYMBOLS[Z - 1]`` (so ``Z = 14 -> "Si"``); the space-group setting
      (1 or 2) becomes ``Crystal(origin=...)`` (default 2 if absent).

    The returned :class:`~torchebsd.sim.MasterPattern` has
    ``data`` of shape ``(n_energy, 2, n, n)`` (axis 1 = hemisphere, north = 0,
    south = 1) as float32, ``energies_kev`` the full ``EkeVs`` list, and
    ``metadata`` carrying ``"crystal"`` (the rebuilt crystal), ``"dmin"``,
    ``"bethe"``, and ``"source"`` (the file path). It feeds ``index_scan``
    directly::

        master = master_from_emsoft("Si_MC.h5")
        indexer = SphericalIndexer(master, detector)
        xmap = indexer.index_scan(patterns, shape=(rows, cols))  # phase auto-derived

    Parameters
    ----------
    path
        Path to the EMsoft master-pattern HDF5 file.
    energy_kev
        Currently informational only: **all** energy bins are loaded (the indexer
        selects the energy at sample time, defaulting to the highest bin). When
        given it is recorded in ``metadata["requested_energy_kev"]`` for provenance.
    device
        Optional torch device the returned ``MasterPattern`` is moved to.

    Returns
    -------
    torchebsd.sim.MasterPattern
        The imported master, with ``metadata["crystal"]`` set (unless the
        ``CrystalData`` group is missing/malformed, in which case the master is
        still returned and ``metadata["crystal_error"]`` explains why -- a partial
        file never sinks the whole load).

    Notes
    -----
    ``kikuchipy`` is never imported; ``h5py`` (an existing repo dependency used by
    :meth:`MasterPattern.save`/:meth:`~MasterPattern.load`) is imported inside the
    function.
    """
    import h5py
    import torch

    from torchebsd.sim import MasterPattern

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
    if energy_kev is not None:
        metadata["requested_energy_kev"] = float(energy_kev)

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
    """Collapse an EMsoft square-Lambert master over its leading **site** axis.

    EMsoft stores ``(n_sites, n_energy, n, n)`` with one component per
    asymmetric-unit atom; summing axis 0 combines them (indexing only the first
    site is a known bug). An already-collapsed ``(n_energy, n, n)`` array is
    returned unchanged.
    """
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
    """Read a possibly length-1 / bytes EMsoft dataset as a scalar (``None`` if absent)."""
    try:
        return cast(np.asarray(f[key]).ravel()[0])
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def _read_emsoft_array(f, key: str):
    """Read an EMsoft dataset as a flat float array (``None`` if absent/unreadable)."""
    try:
        return np.asarray(f[key], dtype=np.float64).ravel()
    except (KeyError, ValueError, TypeError):
        return None


def _read_crystal_arrays(f):
    """Pull the raw ``CrystalData`` arrays out of an open EMsoft HDF5 file.

    Returns a dict of raw values, or ``None`` if the group (or a required field)
    is missing -- the caller then imports the master without a crystal. Values are
    read defensively (some EMsoft datasets are length-1 arrays or bytes).
    """
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
    """Build a torchebsd ``Crystal`` from raw EMsoft ``CrystalData`` arrays.

    Returns ``(crystal, None)`` on success, or ``(None, message)`` if the data is
    missing/malformed so the master still imports with a metadata note. Atomic
    numbers ``Z`` map to symbols via ``ELEMENT_SYMBOLS[Z - 1]`` (index 0 is ``"H"``,
    so ``Z = 14 -> "Si"``); site ``j`` is ``AtomData[:, j] = (x, y, z, occupancy, B)``.
    """
    if arrays is None:
        return None, "CrystalData group missing or unreadable"
    try:
        from torchebsd.sim import AtomSite, Crystal
        from torchebsd.sim.data.scattering_tables import ELEMENT_SYMBOLS

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
