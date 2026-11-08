"""
phases.py — a phase: its structure from a CIF, and what indexing needs of it.

A phase here is what SpyDE's Composition dock holds (``name``, the elements,
``cif_path``): the structure is read once, as an orix phase, through
:func:`..crystal_phase.phase_from_cif` — the door both SpyDE's EBSD wizard and
its 4D-STEM orientation mapping use, and the one that keeps a hexagonal or
trigonal cell's atoms where the file put them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

#: Phase colors, as SpyDE cycles them, so a phase looks the same in both apps.
COLORS = ("#f38ba8", "#a6e3a1", "#89b4fa", "#f9e2af", "#94e2d5", "#cba6f7")


@dataclass
class Phase:
    """One phase: a name, its CIF, and the orix phase read from it."""

    id: int
    name: str
    cif_path: str
    color: str = COLORS[0]
    #: The COD entry it came from, when it did.
    cod_id: str | None = None
    structure: object = field(default=None, repr=False)

    @classmethod
    def from_cif(cls, ident: int, path: str, *, name: str | None = None, cod_id=None) -> "Phase":
        from ..crystal_phase import phase_from_cif

        structure = phase_from_cif(path)
        label = name or structure.name or _formula(structure) or f"Phase {ident + 1}"
        structure.name = label
        return cls(
            id=int(ident),
            name=label,
            cif_path=str(path),
            color=COLORS[ident % len(COLORS)],
            cod_id=None if cod_id is None else str(cod_id),
            structure=structure,
        )

    @property
    def space_group(self) -> int | None:
        """The International Tables number. diffpy numbers a non-standard
        setting of group n as a multiple of 1000 plus n (``R 3 m :R`` is 1160)."""
        group = getattr(self.structure, "space_group", None)
        return int(group.number) % 1000 if group is not None else None

    @property
    def point_group(self):
        return getattr(self.structure, "point_group", None)

    def elements(self) -> list[str]:
        return _elements(self.structure)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "cif_path": self.cif_path,
            "color": self.color,
            "cod_id": self.cod_id,
            "space_group": self.space_group,
            "point_group": getattr(self.point_group, "name", None),
            "elements": self.elements(),
        }


def _elements(structure) -> list[str]:
    symbols: list[str] = []
    for atom in getattr(structure, "structure", None) or ():
        symbol = str(getattr(atom, "element", "") or "").strip()
        # diffpy writes isotopes and oxidation states ("Fe2+", "O2-").
        symbol = "".join(ch for ch in symbol if ch.isalpha()).capitalize()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _formula(structure) -> str:
    return "".join(_elements(structure))
