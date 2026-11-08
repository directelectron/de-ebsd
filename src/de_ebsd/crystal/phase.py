"""A phase as an application lists it: a name, the CIF it was read from, and its structure.

The structure is an orix ``Phase`` read through
:func:`~de_ebsd.index.io.phase_from_cif`, which keeps a hexagonal or trigonal cell's
atoms where the file put them. It is what gives a crystal map, an IPF key or a
misorientation its symmetry, and what
:func:`~de_ebsd.sim.kinematic_bands.reflectors_from_phase` draws kinematic bands from;
for a :class:`~de_ebsd.crystal.Crystal` to simulate dynamically, read the same file
with :meth:`Crystal.from_cif <de_ebsd.crystal.Crystal.from_cif>`.

How a phase is shown (its color, its place in a list) is the application's:
:data:`de_ebsd.viz.PHASE_COLORS` is a default palette.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Phase"]


@dataclass
class Phase:
    """One phase: a name, its CIF and the orix phase read from it.

    Parameters
    ----------
    name
        What the phase is called.
    cif_path
        The CIF it was read from.
    structure
        The orix ``Phase``, lattice in Angstrom.
    cod_id
        The Crystallography Open Database entry it came from, when it did
        (:func:`de_ebsd.crystal.cod.fetch`).
    """

    name: str
    cif_path: str
    structure: object = field(default=None, repr=False)
    cod_id: str | None = None

    @classmethod
    def from_cif(cls, path, *, name: str | None = None, cod_id=None) -> Phase:
        """The phase in the CIF at *path*, named *name*, else after the file
        (:func:`~de_ebsd.index.io.phase_from_cif`), else by its elements."""
        from de_ebsd.index.io import phase_from_cif

        structure = phase_from_cif(path)
        label = name or structure.name or _elements_formula(structure) or "Phase"
        structure.name = label
        return cls(
            name=label,
            cif_path=str(path),
            structure=structure,
            cod_id=None if cod_id is None else str(cod_id),
        )

    @property
    def space_group(self) -> int | None:
        """The International Tables number. diffpy numbers a non-standard setting of
        group n as a multiple of 1000 plus n (``R 3 m :R`` is 1160)."""
        group = getattr(self.structure, "space_group", None)
        return int(group.number) % 1000 if group is not None else None

    @property
    def point_group(self):
        """The orix point group, or ``None`` without a structure."""
        return getattr(self.structure, "point_group", None)

    def elements(self) -> list[str]:
        """The element symbols of the structure, each once, in the file's order."""
        return _elements(self.structure)

    def to_dict(self) -> dict:
        """The phase as plain values, for JSON."""
        return {
            "name": self.name,
            "cif_path": self.cif_path,
            "cod_id": self.cod_id,
            "space_group": self.space_group,
            "point_group": getattr(self.point_group, "name", None),
            "elements": self.elements(),
        }


def _elements(structure) -> list[str]:
    symbols: list[str] = []
    for atom in getattr(structure, "structure", None) or ():
        symbol = str(getattr(atom, "element", "") or "").strip()
        # diffpy writes isotopes and oxidation states ("Fe2+", "O2-")
        symbol = "".join(ch for ch in symbol if ch.isalpha()).capitalize()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _elements_formula(structure) -> str:
    return "".join(_elements(structure))
