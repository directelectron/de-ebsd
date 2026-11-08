"""orix phases from CIFs and from Crystals keep their atoms where the structure has them,
and their lattices in Angstrom, orix's unit.

``orix.crystal_map.Phase`` re-expresses a structure in orix's own Cartesian frame by
turning the lattice without moving the atoms, so for every hexagonal, trigonal,
monoclinic and triclinic cell the atoms' fractions come out wrong: hexagonal
BaTiO3 put Ba at (0.577, 0.577, z) where the file has (1/3, 2/3, z).

The CIFs are trimmed Crystallography Open Database entries (cell, symmetry, atom
sites): 1507756 (tetragonal P4mm), 1507757 (cubic Pm-3m), 9014074 (rhombohedral
R3m, on rhombohedral axes), 9014492 (orthorhombic Amm2) and 2009488 (hexagonal
P6_3/mmc).
"""

import pathlib

import numpy as np
import pytest
from diffpy.structure.parsers import p_cif

from de_ebsd.crystal import Crystal
from de_ebsd.index.io import phase_from_cif, phase_from_crystal

CIF_DIRECTORY = pathlib.Path(__file__).parents[1] / "data" / "cif"

#: polymorph -> (file, International Tables space group number)
POLYMORPHS = {
    "tetragonal": ("BaTiO3_tetragonal_COD1507756.cif", 99),
    "cubic": ("BaTiO3_cubic_COD1507757.cif", 221),
    "rhombohedral": ("BaTiO3_rhombohedral_COD9014074.cif", 160),
    "orthorhombic": ("BaTiO3_orthorhombic_COD9014492.cif", 38),
    "hexagonal": ("BaTiO3_hexagonal_COD2009488.cif", 194),
}


def _path(polymorph: str) -> pathlib.Path:
    return CIF_DIRECTORY / POLYMORPHS[polymorph][0]


def _parsed(polymorph: str):
    """The parser and the asymmetric unit exactly as the file writes it."""
    parser = p_cif.P_cif()
    (getattr(parser, "parse_file", None) or parser.parseFile)(str(_path(polymorph)))
    return parser, [(atom.element, np.asarray(atom.xyz, float)) for atom in parser.asymmetric_unit]


def _on_an_atom(structure, element, position) -> bool:
    for atom in structure:
        offset = (np.asarray(atom.xyz) - position + 0.5) % 1.0 - 0.5
        if atom.element == element and np.all(np.abs(offset) < 1e-4):
            return True
    return False


@pytest.mark.parametrize("polymorph", sorted(POLYMORPHS))
class TestEveryPolymorphKeepsTheFilesStructure:
    def test_the_space_group_is_the_files(self, polymorph):
        # diffpy numbers a non-standard setting of group n as 1000 k + n (R3m:R is 1160)
        assert (
            phase_from_cif(_path(polymorph)).space_group.number % 1000 == (POLYMORPHS[polymorph][1])
        )

    def test_every_site_of_the_file_is_an_atom(self, polymorph):
        structure = phase_from_cif(_path(polymorph)).structure
        _, sites = _parsed(polymorph)
        for element, position in sites:
            assert _on_an_atom(structure, element, position), (element, position)

    def test_the_structure_has_the_symmetry_of_its_space_group(self, polymorph):
        """Every symmetry image of every atom is an atom: what a turned lattice
        under unturned atoms breaks."""
        structure = phase_from_cif(_path(polymorph)).structure
        parser, _ = _parsed(polymorph)
        for operation in parser.spacegroup.iter_symops():
            for atom in structure:
                image = operation(np.asarray(atom.xyz))
                assert _on_an_atom(structure, atom.element, image), (atom.element, image)


def test_hexagonal_barium_sits_on_one_third_two_thirds():
    structure = phase_from_cif(_path("hexagonal")).structure
    barium = np.array([atom.xyz for atom in structure if str(atom.element).startswith("Ba")])
    on_axis = np.all(np.isclose(barium[:, :2], [1 / 3, 2 / 3], atol=1e-6), axis=1)
    assert on_axis.any(), barium


def test_an_orthogonal_cell_is_orix_s_own():
    from orix.crystal_map import Phase

    path = _path("tetragonal")
    ours = np.array([atom.xyz for atom in phase_from_cif(path).structure])
    theirs = np.array([atom.xyz for atom in Phase.from_cif(str(path)).structure])
    assert np.allclose(ours, theirs)


def test_a_cif_phase_is_named_after_its_file_and_measured_in_angstrom():
    phase = phase_from_cif(_path("hexagonal"))
    assert phase.name == "BaTiO3_hexagonal_COD2009488"
    lattice = phase.structure.lattice
    assert (lattice.a, lattice.c, lattice.gamma) == pytest.approx((5.7238, 13.9649, 120.0))


def test_a_crystal_phase_keeps_its_hexagonal_atoms():
    """phase_from_crystal goes through orix's Phase as phase_from_cif does, so it
    had the same turned atoms."""
    crystal = Crystal.from_cif(str(_path("hexagonal")))
    structure = phase_from_crystal(crystal).structure
    barium = np.array([atom.xyz for atom in structure if atom.element == "Ba"])
    assert np.allclose(barium[1, :2], [1 / 3, 2 / 3], atol=1e-9), barium


def test_a_crystal_phase_is_in_angstrom_as_a_cif_phase_is():
    """A Crystal's lattice is in nm; orix and diffsims read a Phase's in Angstrom.
    The d-spacing diffsims computes is where a wrong unit would show."""
    from diffsims.crystallography import ReciprocalLatticeVector

    silicon = Crystal.from_parameters(a=0.5431, space_group=227, atoms=[("Si", (0.125,) * 3)])
    phase = phase_from_crystal(silicon)
    assert phase.structure.lattice.a == pytest.approx(5.431)
    spacing = ReciprocalLatticeVector(phase, hkl=[[1, 1, 1]]).dspacing
    assert spacing[0] == pytest.approx(5.431 / np.sqrt(3))

    hexagonal = phase_from_crystal(Crystal.from_cif(str(_path("hexagonal")))).structure.lattice
    from_file = phase_from_cif(_path("hexagonal")).structure.lattice
    # The six parameters by name: diffpy renamed cell_parms() to cell_params()
    # between the versions Python 3.11 and 3.13 resolve to.
    parameters = ("a", "b", "c", "alpha", "beta", "gamma")
    assert np.allclose(
        [getattr(hexagonal, name) for name in parameters],
        [getattr(from_file, name) for name in parameters],
    )
