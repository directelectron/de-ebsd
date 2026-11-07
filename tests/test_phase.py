"""A phase read from a CIF keeps the file's symmetry and draws its own bands.

The structure is :func:`~de_ebsd.index.io.phase_from_cif`'s, whose own checks
(every polymorph's atoms where the file has them) are in
``tests/index/test_phases.py``; here is what :class:`~de_ebsd.crystal.Phase` adds.
Ported from Autopilot's ``tests/test_cif_phases.py`` and ``tests/test_ebsd.py``.
"""

import pathlib

import numpy as np
import pytest

from de_ebsd.crystal import Phase
from de_ebsd.sim.kinematic_bands import cubic_reflectors, reflectors_from_phase
from de_ebsd.viz import PHASE_COLORS, phase_color

CIF_DIRECTORY = pathlib.Path(__file__).parent / "data" / "cif"

#: polymorph -> (file, International Tables space group number)
POLYMORPHS = {
    "tetragonal": ("BaTiO3_tetragonal_COD1507756.cif", 99),
    "cubic": ("BaTiO3_cubic_COD1507757.cif", 221),
    "rhombohedral": ("BaTiO3_rhombohedral_COD9014074.cif", 160),
    "orthorhombic": ("BaTiO3_orthorhombic_COD9014492.cif", 38),
    "hexagonal": ("BaTiO3_hexagonal_COD2009488.cif", 194),
}


def _phase(polymorph, **kwargs) -> Phase:
    return Phase.from_cif(CIF_DIRECTORY / POLYMORPHS[polymorph][0], **kwargs)


@pytest.mark.parametrize("polymorph", sorted(POLYMORPHS))
def test_the_space_group_is_the_files_international_number(polymorph):
    """Even on rhombohedral axes, which diffpy numbers 1160."""
    assert _phase(polymorph).space_group == POLYMORPHS[polymorph][1]


def test_a_phase_is_named_as_asked_else_after_its_file():
    assert _phase("cubic", name="BaTiO3").name == "BaTiO3"
    assert _phase("cubic", name="BaTiO3").structure.name == "BaTiO3"
    assert _phase("cubic").name == "BaTiO3_cubic_COD1507757"


def test_a_phase_lists_its_elements_and_describes_itself():
    phase = _phase("hexagonal", name="BaTiO3", cod_id=2009488)
    assert phase.elements() == ["Ba", "Ti", "O"]
    assert phase.point_group.name == "6/mmm"
    assert phase.to_dict() == {
        "name": "BaTiO3",
        "cif_path": str(CIF_DIRECTORY / POLYMORPHS["hexagonal"][0]),
        "cod_id": "2009488",
        "space_group": 194,
        "point_group": "6/mmm",
        "elements": ["Ba", "Ti", "O"],
    }


def test_silver_is_face_centred_cubic():
    phase = Phase.from_cif(CIF_DIRECTORY / "Silver_AMCSD0011135.cif", name="Ag")
    assert (phase.space_group, phase.elements()) == (225, ["Ag"])


@pytest.mark.parametrize("polymorph", ["cubic", "hexagonal"])
def test_a_phase_draws_its_own_bands_not_the_cubic_stand_in(polymorph):
    reflectors = reflectors_from_phase(_phase(polymorph).structure, max_bands=20)
    assert len(reflectors) > 0
    assert not np.array_equal(reflectors.hkl, cubic_reflectors().hkl)


def test_phases_take_the_palettes_colors_in_turn():
    assert [phase_color(n) for n in range(len(PHASE_COLORS))] == list(PHASE_COLORS)
    assert phase_color(len(PHASE_COLORS)) == PHASE_COLORS[0]
    assert len(set(PHASE_COLORS)) == len(PHASE_COLORS)
