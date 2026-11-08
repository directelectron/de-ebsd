"""Crystal structures from the Crystallography Open Database keep their symmetry
on the way in, and build an EBSD dictionary.

orix's ``Phase.from_cif`` turns the lattice of any cell whose ``a`` axis is not
already along x (hexagonal, trigonal, monoclinic) without moving the atoms, so
the structure loses its own symmetry: hexagonal BaTiO3 put Ba at (0.577, 0.577,
z) where the file has (1/3, 2/3, z). Every CIF here is read through
``de_ebsd.pipeline.crystal_phase.phase_from_cif``, which keeps the file's fractions.
SpyDE's ``tests/migrated/test_cif_phases.py`` is the same check on its side.

Fixtures are trimmed COD entries (cell, symmetry, atom sites): 1507756
(tetragonal P4mm), 1507757 (cubic Pm-3m), 9014074 (rhombohedral R3m, on
rhombohedral axes), 9014492 (orthorhombic Amm2) and 2009488 (hexagonal
P6_3/mmc).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

CIF_DIR = pathlib.Path(__file__).parent / "data" / "cif"

#: polymorph → (fixture, International Tables space group number)
POLYMORPHS = {
    "tetragonal": ("tetragonal_COD1507756", 99),
    "cubic": ("cubic_COD1507757", 221),
    "rhombohedral": ("rhombohedral_COD9014074", 160),
    "orthorhombic": ("orthorhombic_COD9014492", 38),
    "hexagonal": ("hexagonal_COD2009488", 194),
}


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("SPYDE_EBSD_DEVICE", "cpu")


def _cif(polymorph: str) -> pathlib.Path:
    return CIF_DIR / f"BaTiO3_{POLYMORPHS[polymorph][0]}.cif"


def _phase(polymorph: str):
    from de_ebsd.pipeline.orientation.phases import Phase

    return Phase.from_cif(0, str(_cif(polymorph)))


def _file_sites(path):
    """The asymmetric unit exactly as the file writes it: (element, fractions)."""
    from diffpy.structure.parsers import p_cif

    parser = p_cif.P_cif()
    parser.parseFile(str(path))
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
        assert _phase(polymorph).space_group == POLYMORPHS[polymorph][1]

    def test_every_site_of_the_file_is_an_atom(self, polymorph):
        structure = _phase(polymorph).structure.structure
        _, sites = _file_sites(_cif(polymorph))
        for element, position in sites:
            assert _on_an_atom(structure, element, position), (element, position)

    def test_the_structure_has_the_symmetry_of_its_space_group(self, polymorph):
        """Every symmetry image of every atom is an atom: what a turned lattice
        under unturned atoms breaks."""
        structure = _phase(polymorph).structure.structure
        parser, _ = _file_sites(_cif(polymorph))
        for operation in parser.spacegroup.iter_symops():
            for atom in structure:
                image = operation(np.asarray(atom.xyz))
                assert _on_an_atom(structure, atom.element, image), (atom.element, image)


def test_hexagonal_barium_sits_on_one_third_two_thirds():
    structure = _phase("hexagonal").structure.structure
    barium = np.array([atom.xyz for atom in structure if str(atom.element).startswith("Ba")])
    on_axis = np.all(np.isclose(barium[:, :2], [1 / 3, 2 / 3], atol=1e-6), axis=1)
    assert on_axis.any(), barium


def test_an_orthogonal_cell_is_unchanged():
    from orix.crystal_map import Phase as OrixPhase

    path = _cif("tetragonal")
    ours = np.array([atom.xyz for atom in _phase("tetragonal").structure.structure])
    theirs = np.array([atom.xyz for atom in OrixPhase.from_cif(str(path)).structure])
    assert np.allclose(ours, theirs)


@pytest.mark.slow
@pytest.mark.parametrize("polymorph", sorted(POLYMORPHS))
def test_each_builds_a_dictionary_from_its_own_structure(polymorph):
    """The EBSD workflow's library: its bands come from the structure, not the
    cubic stand-in a failed structure-factor calculation falls back to. At 15°
    the simulation adds little to the ~10 s a low-symmetry phase's structure
    factors take."""
    from de_ebsd.pipeline.ebsd.bands import cubic_reflectors
    from de_ebsd.pipeline.orientation.indexing import Geometry, PhaseDictionary

    dictionary = PhaseDictionary(_phase(polymorph), Geometry((40, 40)), step_deg=15.0)
    assert len(dictionary) > 0
    assert not np.array_equal(dictionary.reflectors.hkl, cubic_reflectors().hkl)
