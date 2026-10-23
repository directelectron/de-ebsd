"""Unit cells: lattice parameters, symmetry expansion, and metric tensors.

Space-group symmetry operations come from :mod:`diffpy.structure`. The Cartesian
crystal frame and the direct and reciprocal structure matrices follow EMsoft's
``mod_crystallography.f90``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from diffpy.structure.spacegroups import GetSpaceGroup

from de_ebsd.core.constants import C_AVOGADRO
from de_ebsd.crystal.scattering import atomic_number, atomic_weight

__all__ = ["AtomSite", "Crystal"]


@dataclass(frozen=True)
class AtomSite:
    """One site of the asymmetric unit.

    Parameters
    ----------
    element
        Element symbol, e.g. ``"Si"``.
    position
        Fractional coordinates (x, y, z) in the unit cell.
    occupancy
        Site occupancy in [0, 1].
    debye_waller
        Isotropic Debye-Waller factor B in nm^2 (EMsoft convention;
        0.005 nm^2 = 0.5 A^2 is a reasonable room-temperature default).
    """

    element: str
    position: tuple[float, float, float]
    occupancy: float = 1.0
    debye_waller: float = 0.005


class Crystal:
    """A crystal defined by its lattice, space group, and asymmetric unit.

    Prefer the :meth:`from_parameters` constructor::

        crystal = Crystal.from_parameters(
            a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
        )

    Parameters
    ----------
    a, b, c
        Lattice parameters in nm.
    alpha, beta, gamma
        Lattice angles in degrees.
    space_group
        International space-group number (1-230).
    sites
        The asymmetric unit as a sequence of :class:`AtomSite`.
    origin
        Origin choice for the 24 centrosymmetric space groups that the
        International Tables list with two origins (e.g. Fd-3m). The default,
        2, places the origin on an inversion center (the modern standard used
        by EMsoft and kikuchipy reference data); pass 1 for the tabulated
        diffpy setting. The choice-2 operators are derived by conjugating the
        tabulated operators with the smallest inversion-center shift; this is
        verified for the cubic two-origin groups -- for other two-origin
        groups, confirm the convention against the International Tables.
    """

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        alpha: float,
        beta: float,
        gamma: float,
        space_group: int,
        sites: Sequence[AtomSite],
        origin: int = 2,
    ) -> None:
        if not 1 <= int(space_group) <= 230:
            raise ValueError(f"space_group must be in 1..230, got {space_group}")
        if origin not in (1, 2):
            raise ValueError("origin must be 1 or 2")
        if len(sites) == 0:
            raise ValueError("at least one atom site is required")
        self.a, self.b, self.c = float(a), float(b), float(c)
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        self.space_group = int(space_group)
        self.origin = int(origin)
        self.sites = tuple(sites)
        self._spacegroup = GetSpaceGroup(self.space_group)
        self._symops = _symmetry_operations(self._spacegroup, self.origin)
        self._positions = tuple(
            _expand_orbit(self._symops, np.asarray(s.position, dtype=np.float64))
            for s in self.sites
        )

    @classmethod
    def from_parameters(
        cls,
        a: float,
        b: float | None = None,
        c: float | None = None,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        space_group: int = 1,
        atoms: Sequence[AtomSite | tuple] = (),
        debye_waller: float = 0.005,
    ) -> Crystal:
        """Create a crystal from lattice parameters and a simple atom list.

        ``atoms`` entries may be :class:`AtomSite` objects or plain tuples
        ``(element, (x, y, z))`` / ``(element, (x, y, z), occupancy)``;
        tuples get the shared ``debye_waller`` default.
        """
        sites: list[AtomSite] = []
        for atom in atoms:
            if isinstance(atom, AtomSite):
                sites.append(atom)
            else:
                element, position, *rest = atom
                occupancy = rest[0] if rest else 1.0
                sites.append(
                    AtomSite(
                        element=element,
                        position=tuple(float(p) for p in position),
                        occupancy=float(occupancy),
                        debye_waller=float(debye_waller),
                    )
                )
        return cls(
            a=a,
            b=a if b is None else b,
            c=a if c is None else c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            space_group=space_group,
            sites=sites,
        )

    @classmethod
    def from_cif(
        cls,
        path: str,
        origin: int | None = None,
        debye_waller: float = 0.005,
    ) -> Crystal:
        """Create a crystal from a CIF file.

        Reads only the **asymmetric unit**; the symmetry expansion is this class's
        own job, so passing the expanded cell that ``diffpy.structure.loadStructure``
        returns would double-count every atom.

        CIF lengths are Angstrom and are converted to nm. An ``U_iso`` column becomes
        the Debye-Waller factor as ``B = 8 pi^2 U_iso``, converted to nm^2; sites
        without one (or with ``U_iso <= 0``) fall back to ``debye_waller``. Ionic
        ``_atom_site_type_symbol`` entries are reduced to the element (``"Si4+"`` ->
        ``"Si"``), since the scattering tables are per element.

        Parameters
        ----------
        path
            Path to the CIF file. Only its first data block is read.
        origin
            Origin choice for the two-origin space groups. ``None`` (default) reads
            ``_space_group_IT_coordinate_system_code`` from the file and falls back to
            2 -- the modern standard used by EMsoft and kikuchipy. **Getting this
            wrong silently produces a plausible but incorrect master pattern**, so
            pass it explicitly when the file does not record it and the group has two
            origins (:attr:`has_two_origins`).
        debye_waller
            Fallback isotropic B in nm^2 for sites with no usable ``U_iso``.

        Returns
        -------
        Crystal
        """
        from diffpy.structure.parsers import getParser

        parser = getParser("cif")
        parser.parseFile(str(path))
        lattice = parser.asymmetric_unit[0].lattice if parser.asymmetric_unit else None
        if lattice is None:
            raise ValueError(f"{path}: no atom sites found in the CIF")

        if origin is None:
            origin = _cif_origin_choice(parser)

        sites: list[AtomSite] = []
        for atom in parser.asymmetric_unit:
            b_nm2 = debye_waller
            u_iso = float(getattr(atom, "Uisoequiv", 0.0) or 0.0)
            if u_iso > 0.0:
                b_nm2 = 8.0 * np.pi**2 * u_iso / 100.0  # 8 pi^2 U [A^2] -> B [nm^2]
            sites.append(
                AtomSite(
                    element=_element_symbol(atom.element),
                    position=tuple(float(x) for x in atom.xyz),
                    occupancy=float(atom.occupancy),
                    debye_waller=b_nm2,
                )
            )

        return cls(
            a=lattice.a / 10.0,
            b=lattice.b / 10.0,
            c=lattice.c / 10.0,
            alpha=lattice.alpha,
            beta=lattice.beta,
            gamma=lattice.gamma,
            space_group=parser.spacegroup.number,
            sites=sites,
            origin=origin,
        )

    @property
    def has_two_origins(self) -> bool:
        """Whether this space group is one the Tables list with two origin choices.

        When ``True``, :attr:`origin` changes the atom positions and therefore the
        master pattern, so a CIF that does not record its choice is ambiguous.
        """
        return self.space_group in _TWO_ORIGIN_GROUPS

    @property
    def direct_metric(self) -> np.ndarray:
        """Direct metric tensor g_ij in nm^2."""
        a, b, c = self.a, self.b, self.c
        ca, cb, cg = (np.cos(np.deg2rad(x)) for x in (self.alpha, self.beta, self.gamma))
        return np.array(
            [
                [a * a, a * b * cg, a * c * cb],
                [a * b * cg, b * b, b * c * ca],
                [a * c * cb, b * c * ca, c * c],
            ]
        )

    @property
    def reciprocal_metric(self) -> np.ndarray:
        """Reciprocal metric tensor g*_ij in nm^-2."""
        return np.linalg.inv(self.direct_metric)

    @property
    def volume(self) -> float:
        """Unit-cell volume in nm^3."""
        return float(np.sqrt(np.linalg.det(self.direct_metric)))

    @property
    def direct_structure_matrix(self) -> np.ndarray:
        """Columns are the basis vectors a, b, c in the Cartesian crystal frame (nm)."""
        a, b, c = self.a, self.b, self.c
        ca, cb, cg = (np.cos(np.deg2rad(x)) for x in (self.alpha, self.beta, self.gamma))
        sg = np.sin(np.deg2rad(self.gamma))
        vol = self.volume
        return np.array(
            [
                [a, b * cg, c * cb],
                [0.0, b * sg, -c * (cb * cg - ca) / sg],
                [0.0, 0.0, vol / (a * b * sg)],
            ]
        )

    @property
    def reciprocal_structure_matrix(self) -> np.ndarray:
        """Columns are the reciprocal basis vectors a*, b*, c* in the Cartesian frame (nm^-1)."""
        return np.linalg.inv(self.direct_structure_matrix).T

    def g_length(self, hkl: np.ndarray) -> np.ndarray:
        """Length ``|g|`` in nm^-1 for Miller indices ``hkl`` of shape (..., 3)."""
        hkl = np.asarray(hkl, dtype=np.float64)
        gstar = self.reciprocal_metric
        return np.sqrt(np.einsum("...i,ij,...j->...", hkl, gstar, hkl))

    def d_spacing(self, hkl: np.ndarray) -> np.ndarray:
        """Interplanar spacing d in nm for Miller indices ``hkl``."""
        return 1.0 / self.g_length(hkl)

    def g_cartesian(self, hkl: np.ndarray) -> np.ndarray:
        """Reciprocal-lattice vectors in the Cartesian crystal frame (nm^-1)."""
        hkl = np.asarray(hkl, dtype=np.float64)
        return hkl @ self.reciprocal_structure_matrix.T

    @property
    def positions(self) -> tuple[np.ndarray, ...]:
        """Expanded orbit (n_m, 3) of fractional positions, one array per site."""
        return self._positions

    @property
    def multiplicities(self) -> tuple[int, ...]:
        """Number of equivalent atoms per asymmetric-unit site."""
        return tuple(len(p) for p in self._positions)

    @property
    def _site_counts(self) -> np.ndarray:
        return np.array(
            [s.occupancy * m for s, m in zip(self.sites, self.multiplicities, strict=True)]
        )

    @property
    def mean_atomic_number(self) -> float:
        """Composition- (number-) averaged atomic number, sum(n_i Z_i) / sum(n_i)."""
        counts = self._site_counts
        z = np.array([atomic_number(s.element) for s in self.sites], dtype=np.float64)
        return float((counts * z).sum() / counts.sum())

    @property
    def mean_atomic_weight(self) -> float:
        """Composition- (number-) averaged atomic weight in g/mol."""
        counts = self._site_counts
        a = np.array([atomic_weight(s.element) for s in self.sites])
        return float((counts * a).sum() / counts.sum())

    @property
    def density(self) -> float:
        """Mass density in g/cm^3 from the unit-cell contents and volume."""
        counts = self._site_counts
        a = np.array([atomic_weight(s.element) for s in self.sites])
        cell_mass_g = (counts * a).sum() / C_AVOGADRO
        volume_cm3 = self.volume * 1e-21  # nm^3 -> cm^3
        return float(cell_mass_g / volume_cm3)

    @property
    def centering(self) -> str:
        """Lattice centering symbol (first letter of the space-group symbol)."""
        return self._spacegroup.short_name[0]

    def allowed_by_centering(self, hkl: np.ndarray) -> np.ndarray:
        """Boolean mask of reflections allowed by the lattice centering.

        Mirrors EMsoft ``IsGAllowed``: only centering absences are tested;
        glide/screw absences follow automatically from ``|U_g| = 0``.
        """
        hkl = np.asarray(hkl, dtype=np.int64)
        h, k, l = hkl[..., 0], hkl[..., 1], hkl[..., 2]  # noqa: E741
        symbol = self.centering
        if symbol == "P":
            return np.ones(h.shape, dtype=bool)
        if symbol == "F":
            return (h % 2 == k % 2) & (k % 2 == l % 2)
        if symbol == "I":
            return (h + k + l) % 2 == 0
        if symbol == "A":
            return (k + l) % 2 == 0
        if symbol == "B":
            return (h + l) % 2 == 0
        if symbol == "C":
            return (h + k) % 2 == 0
        if symbol == "R":  # obverse hexagonal setting
            return (-h + k + l) % 3 == 0
        raise ValueError(f"Unknown centering symbol {symbol!r}")

    def show(self, title: str | None = None):
        """Display the expanded crystal structure as a native 3-D figure."""
        from de_ebsd.viz.patterns import crystal_structure_figure

        return crystal_structure_figure(self, title=title or "Crystal structure")

    def explore(self, *, energy_kev: float = 200.0, dmin: float = 0.1, **kwargs):
        """Explore STEM orientations using geometric lines and kinematic spots.

        Returns a :class:`~de_ebsd.viz.explorer.CrystalExplorer` with a shared
        projection, band sphere, and crystal view. Beam energy is in keV and
        the minimum lattice spacing ``dmin`` in nm. Further keyword arguments
        are passed to the explorer. No master-pattern simulation is run.

        Display the result in Jupyter for interactive controls, or mount its
        native ``figure`` in an application with a live Python bridge.
        """
        from de_ebsd.viz.explorer import CrystalExplorer

        return CrystalExplorer(self, energy_kev=energy_kev, dmin=dmin, **kwargs)

    def overview(
        self,
        master_pattern=None,
        name: str | None = None,
        *,
        texture_height: int = 128,
        n_grid: int = 96,
    ):
        """Compose phase metadata with native crystal and master figures.

        The returned :class:`~de_ebsd.viz.views.PhaseOverview` displays in a
        notebook and gives SpyDE named figures to mount with its existing
        anyplotlib bridge. It never serializes a figure into another widget.
        """
        from de_ebsd.viz.views import PhaseOverview

        return PhaseOverview(
            crystal=self,
            master_pattern=master_pattern,
            name=name,
            texture_height=texture_height,
            n_grid=n_grid,
        )

    def __repr__(self) -> str:
        atoms = ", ".join(
            f"{s.element}({len(p)}x)" for s, p in zip(self.sites, self._positions, strict=True)
        )
        return (
            f"Crystal(a={self.a:.4f} nm, b={self.b:.4f}, c={self.c:.4f}, "
            f"alpha={self.alpha:g}, beta={self.beta:g}, gamma={self.gamma:g}, "
            f"space_group={self.space_group}, atoms=[{atoms}])"
        )


def _symmetry_operations(spacegroup, origin: int) -> list[tuple[np.ndarray, np.ndarray]]:
    ops = [
        (np.asarray(op.R, dtype=np.float64), np.asarray(op.t, dtype=np.float64))
        for op in spacegroup.iter_symops()
    ]
    if origin == 1:
        return ops

    inversions = [t for r, t in ops if np.allclose(r, -np.eye(3))]
    if not inversions:  # non-centrosymmetric: no origin ambiguity
        return ops
    # Inversion centers sit at (t + m) / 2 for integer vectors m; pick the one
    # closest to the tabulated origin. If it is the origin itself, nothing to do.
    candidates = []
    for t in inversions:
        for m in np.ndindex(2, 2, 2):
            s = np.mod((t + np.asarray(m, dtype=np.float64)) / 2.0, 1.0)
            dist = np.linalg.norm(np.minimum(s, 1.0 - s))
            candidates.append((dist, s))
    dist, shift = min(candidates, key=lambda c: c[0])
    if dist < 1e-9:
        return ops
    return [(r, np.mod(t + r @ shift - shift, 1.0)) for r, t in ops]


def _expand_orbit(
    symops: list[tuple[np.ndarray, np.ndarray]], position: np.ndarray, decimals: int = 6
) -> np.ndarray:
    seen: dict[tuple, np.ndarray] = {}
    for r, t in symops:
        new = r @ position + t
        new = np.mod(np.round(new, decimals), 1.0)
        new = np.mod(new, 1.0)  # fold 0.999999... rounded to 1.0 back to 0
        seen.setdefault(tuple(new), new)
    return np.array(list(seen.values()))


# The space groups the International Tables list with two origin choices. For these
# `Crystal.origin` moves the atoms, so a CIF that does not record its choice is
# ambiguous -- see `Crystal.from_cif`.
_TWO_ORIGIN_GROUPS = frozenset(
    {
        48,
        50,
        59,
        68,
        70,
        85,
        86,
        88,
        125,
        126,
        129,
        130,
        133,
        134,
        137,
        138,
        141,
        142,
        201,
        203,
        222,
        224,
        227,
        228,
    }
)


def _cif_origin_choice(parser, default: int = 2) -> int:
    """Origin choice recorded in a CIF, or ``default`` when it says nothing."""
    try:
        block = parser.ciffile[next(iter(parser.ciffile.keys()))]
    except Exception:  # noqa: BLE001 - an unreadable block just means "unspecified"
        return default
    for key in (
        "_space_group_it_coordinate_system_code",
        "_symmetry_space_group_setting",
    ):
        value = str(block.get(key, "")).strip()
        if value in ("1", "2"):
            return int(value)
    return default


def _element_symbol(raw: str) -> str:
    """Element symbol from a CIF ``type_symbol``/label (``"Si4+"``, ``"Si1"`` -> ``"Si"``)."""
    letters = re.match(r"[A-Za-z]+", str(raw).strip())
    if letters is None:
        raise ValueError(f"cannot read an element symbol from {raw!r}")
    text = letters.group(0)
    for n in (len(text), 2, 1):
        candidate = text[:n].capitalize()
        try:
            atomic_number(candidate)
        except ValueError:
            continue
        return candidate
    raise ValueError(f"{raw!r} is not a known element symbol")
