"""Crystal.from_cif: asymmetric-unit reading, unit conversion, origin choice."""

import numpy as np
import pytest

from de_ebsd.crystal import Crystal
from de_ebsd.crystal.lattice import _element_symbol

SI_CIF = """data_Si
_cell_length_a                         5.4310
_cell_length_b                         5.4310
_cell_length_c                         5.4310
_cell_angle_alpha                      90.0
_cell_angle_beta                       90.0
_cell_angle_gamma                      90.0
_space_group_IT_number                 227
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Si1 Si 0.125 0.125 0.125 1.0 0.0063
"""

RUTILE_CIF = """data_TiO2
_cell_length_a     4.5941
_cell_length_b     4.5941
_cell_length_c     2.9589
_cell_angle_alpha  90.0
_cell_angle_beta   90.0
_cell_angle_gamma  90.0
_space_group_IT_number  136
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Ti1 Ti4+ 0.0 0.0 0.0 1.0
O1  O2-  0.3053 0.3053 0.0 1.0
"""


def _write(tmp_path, text, name="x.cif"):
    path = tmp_path / name
    path.write_text(text, encoding="utf8")
    return str(path)


def test_reads_asymmetric_unit_not_expanded_cell(tmp_path):
    # diffpy's loadStructure returns all 16 expanded atoms; Crystal does its own
    # expansion, so from_cif must keep only the 1-atom asymmetric unit
    crystal = Crystal.from_cif(_write(tmp_path, SI_CIF))
    assert len(crystal.sites) == 1
    assert sum(len(p) for p in crystal.positions) == 8  # diamond cubic


def test_matches_hand_built_crystal(tmp_path):
    crystal = Crystal.from_cif(_write(tmp_path, SI_CIF))
    reference = Crystal.from_parameters(
        a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
    )
    assert crystal.a == pytest.approx(reference.a)
    assert crystal.space_group == reference.space_group
    assert np.allclose(
        np.sort(crystal.positions[0], axis=0), np.sort(reference.positions[0], axis=0)
    )


def test_angstrom_to_nm_and_density(tmp_path):
    crystal = Crystal.from_cif(_write(tmp_path, SI_CIF))
    assert crystal.a == pytest.approx(0.5431)  # CIF is 5.4310 A
    assert crystal.density == pytest.approx(2.329, abs=0.002)  # g/cm^3


def test_uiso_becomes_debye_waller(tmp_path):
    crystal = Crystal.from_cif(_write(tmp_path, SI_CIF))
    # B = 8 pi^2 U [A^2], then A^2 -> nm^2
    assert crystal.sites[0].debye_waller == pytest.approx(8.0 * np.pi**2 * 0.0063 / 100.0)


def test_missing_uiso_falls_back(tmp_path):
    crystal = Crystal.from_cif(_write(tmp_path, RUTILE_CIF), debye_waller=0.007)
    assert all(s.debye_waller == pytest.approx(0.007) for s in crystal.sites)


def test_multi_site_ionic_symbols(tmp_path):
    crystal = Crystal.from_cif(_write(tmp_path, RUTILE_CIF))
    assert [s.element for s in crystal.sites] == ["Ti", "O"]  # Ti4+, O2- reduced
    assert sum(len(p) for p in crystal.positions) == 6  # 2 Ti + 4 O
    assert crystal.c < crystal.a  # tetragonal, not forced cubic


def test_origin_defaults_to_two_and_is_overridable(tmp_path):
    path = _write(tmp_path, SI_CIF)  # records no coordinate-system code
    assert Crystal.from_cif(path).origin == 2
    assert Crystal.from_cif(path, origin=1).origin == 1


def test_origin_read_from_cif(tmp_path):
    tagged = SI_CIF.replace(
        "_space_group_IT_number                 227",
        "_space_group_IT_number                 227\n_space_group_IT_coordinate_system_code 1",
    )
    assert Crystal.from_cif(_write(tmp_path, tagged)).origin == 1


def test_has_two_origins(tmp_path):
    assert Crystal.from_cif(_write(tmp_path, SI_CIF)).has_two_origins  # Fd-3m
    assert not Crystal.from_cif(_write(tmp_path, RUTILE_CIF)).has_two_origins  # P4_2/mnm


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Si", "Si"), ("Si4+", "Si"), ("O2-", "O"), ("Ti1", "Ti"), ("FE", "Fe"), ("fe", "Fe")],
)
def test_element_symbol(raw, expected):
    assert _element_symbol(raw) == expected


def test_element_symbol_rejects_nonsense():
    with pytest.raises(ValueError):
        _element_symbol("Xx9")


# Barium titanate, rhombohedral R3m on rhombohedral axes (COD 9014074).
RHOMBOHEDRAL_CIF = """data_9014074
_space_group_IT_number           160
_symmetry_space_group_name_H-M   'R 3 m :R'
_cell_angle_alpha                89.855
_cell_angle_beta                 89.855
_cell_angle_gamma                89.855
_cell_length_a                   4.0043
_cell_length_b                   4.0043
_cell_length_c                   4.0043
loop_
_space_group_symop_operation_xyz
x,y,z
x,z,y
z,x,y
y,x,z
y,z,x
z,y,x
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ba 0.00000 0.00000 0.00000
Ti 0.48700 0.48700 0.48700
O 0.50920 0.50920 0.01890
"""

# Barium titanate, hexagonal P6_3/mmc (COD 2009488): a cell whose a axis is not
# along x, the case orix's Phase.from_cif gets wrong.
HEXAGONAL_CIF = """data_2009488
_space_group_IT_number           194
_symmetry_space_group_name_H-M   'P 63/m m c'
_cell_angle_alpha                90
_cell_angle_beta                 90
_cell_angle_gamma                120
_cell_length_a                   5.7238
_cell_length_b                   5.7238
_cell_length_c                   13.9649
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ba1 0 0 0.25
Ba2 0.333333333333333 0.666666666666667 0.09671
Ti1 0 0 0
Ti2 0.333333333333333 0.666666666666667 0.84633
O1 0.5185 0.0370 0.25
O2 0.8349 0.6698 0.0802
"""


def test_hexagonal_cell_keeps_the_file_fractions(tmp_path):
    crystal = Crystal.from_cif(_write(tmp_path, HEXAGONAL_CIF))
    assert crystal.space_group == 194
    assert crystal.gamma == pytest.approx(120.0)
    assert crystal.sites[1].position == pytest.approx((1 / 3, 2 / 3, 0.09671))
    assert sum(len(p) for p in crystal.positions) == 30  # Z = 6 BaTiO3


def test_rhombohedral_axes_become_hexagonal_axes(tmp_path):
    from diffpy.structure import loadStructure

    from de_ebsd.crystal.lattice import _RHOMBOHEDRAL_TO_HEXAGONAL

    path = _write(tmp_path, RHOMBOHEDRAL_CIF)
    crystal = Crystal.from_cif(path)
    assert crystal.space_group == 160
    assert crystal.gamma == pytest.approx(120.0)
    assert crystal.alpha == pytest.approx(90.0) and crystal.beta == pytest.approx(90.0)
    rhombohedral = loadStructure(path)
    assert crystal.volume == pytest.approx(3 * rhombohedral.lattice.volume / 1000.0)
    assert [s.element for s in crystal.sites] == ["Ba", "Ti", "O"]
    assert sum(len(p) for p in crystal.positions) == 15  # three cells of 5 atoms

    # Every atom of the hexagonal cell, taken back to rhombohedral axes, is an
    # atom of the file's own expansion, and every one of those is reached.
    expected = {(a.element, *np.round(np.asarray(a.xyz) % 1.0, 4) % 1.0) for a in rhombohedral}
    found = {
        (site.element, *np.round((_RHOMBOHEDRAL_TO_HEXAGONAL @ x) % 1.0, 4) % 1.0)
        for site, positions in zip(crystal.sites, crystal.positions, strict=True)
        for x in positions
    }
    assert found == expected


def test_other_non_standard_settings_are_refused(tmp_path):
    pbnm = RUTILE_CIF.replace(
        "_space_group_IT_number  136", "_symmetry_space_group_name_H-M 'P b n m'"
    )
    with pytest.raises(ValueError, match="non-standard setting"):
        Crystal.from_cif(_write(tmp_path, pbnm))


@pytest.mark.parametrize("third", ["0.3333", "0.333333", "0.333333333333333"])
def test_hcp_titanium_has_two_atoms_however_one_third_is_written(tmp_path, third):
    two_thirds = f"{1 - float(third):.{len(third) - 2}f}"
    titanium = f"""data_Ti
_space_group_IT_number           194
_cell_length_a                   2.9506
_cell_length_b                   2.9506
_cell_length_c                   4.6835
_cell_angle_alpha                90
_cell_angle_beta                 90
_cell_angle_gamma                120
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ti1 {third} {two_thirds} 0.25
"""
    crystal = Crystal.from_cif(_write(tmp_path, titanium))
    assert crystal.multiplicities == (2,)
    assert crystal.density == pytest.approx(4.50, abs=0.01)  # g/cm^3
