"""A crystal's unit cell as atoms and edges (:func:`de_ebsd.viz.patterns.unit_cell`),
and the one element-color table. Ported from SpyDE's
``test_fcc_cell_has_its_face_and_corner_atoms``, in nm where SpyDE's was in Angstrom."""

from pathlib import Path

import numpy as np
import pytest

from de_ebsd import Crystal
from de_ebsd.viz.patterns import DEFAULT_COLOR, ELEMENT_COLORS, element_color, unit_cell

CIF = Path(__file__).parent / "data" / "cif"


def test_fcc_cell_has_its_face_and_corner_atoms():
    positions, elements, edges = unit_cell(Crystal.from_cif(CIF / "Silver_AMCSD0011135.cif"))
    assert len(positions) == 14 and set(elements) == {"Ag"}  # 8 corners + 6 faces
    assert edges.shape == (12, 2, 3)
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    np.testing.assert_allclose(lengths, 0.40862, atol=1e-4)
    assert positions.min() == pytest.approx(0.0, abs=1e-9)
    assert positions.max() == pytest.approx(0.40862, abs=1e-4)


def test_a_body_centered_atom_has_no_images():
    crystal = Crystal.from_parameters(a=0.2866, space_group=229, atoms=[("Fe", (0, 0, 0))])
    positions, _, _ = unit_cell(crystal)
    assert len(positions) == 9  # 8 corners + the body center, once


def test_a_cells_atoms_carry_their_sites_elements():
    crystal = Crystal.from_cif(CIF / "BaTiO3_cubic_COD1507757.cif")
    _, elements, _ = unit_cell(crystal)
    # this setting has Ti on the corners (8), Ba in the body (1), O mid-edge (4 images of 3)
    assert {e: elements.count(e) for e in set(elements)} == {"Ti": 8, "Ba": 1, "O": 12}


def test_one_color_table_in_jmols_colors():
    assert len(ELEMENT_COLORS) == 37
    assert element_color("Ni") == pytest.approx((0x50 / 255, 0xD0 / 255, 0x50 / 255))
    assert element_color("Fe2+") == element_color("Fe")  # a CIF label reads as its element
    assert element_color("Xe") == element_color("Xx9")
    expected = tuple(int(DEFAULT_COLOR[i : i + 2], 16) / 255 for i in (1, 3, 5))
    assert element_color("Xe") == pytest.approx(expected)
