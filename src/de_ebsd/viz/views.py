"""Host-neutral compositions of de_ebsd display objects.

Scientific panels remain real :class:`anyplotlib.Figure` instances.  This
module only groups figures with small JSON-compatible metadata so Jupyter and
SpyDE can choose their own layout and lifecycle without nested HTML/iframes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from de_ebsd.crystal.lattice import Crystal
    from de_ebsd.sim.masterpattern import MasterPattern

__all__ = ["PhaseOverview"]


@dataclass
class PhaseOverview:
    """A phase's metadata and native anyplotlib panels.

    ``figures`` is the integration boundary for application hosts. SpyDE can
    register each value with its existing anyplotlib Electron bridge; Jupyter
    displays this object directly through :meth:`_ipython_display_`.
    """

    crystal: Crystal
    master_pattern: MasterPattern | None = None
    name: str | None = None
    texture_height: int = 128
    n_grid: int = 96

    def __post_init__(self) -> None:
        if self.texture_height < 8:
            raise ValueError("texture_height must be >= 8")
        if self.n_grid < 8:
            raise ValueError("n_grid must be >= 8")

    @property
    def metadata(self) -> dict[str, str | int | float | None]:
        """Compact, JSON-compatible phase metadata for a sidebar/popover."""
        return {
            "name": self.name or _phase_name(self.crystal),
            "space_group": self.crystal.space_group,
            "space_group_symbol": str(self.crystal._spacegroup.short_name),
            "a_nm": self.crystal.a,
            "b_nm": self.crystal.b,
            "c_nm": self.crystal.c,
            "alpha_deg": self.crystal.alpha,
            "beta_deg": self.crystal.beta,
            "gamma_deg": self.crystal.gamma,
            "volume_nm3": self.crystal.volume,
            "density_g_cm3": self.crystal.density,
            "atom_count": sum(self.crystal.multiplicities),
            "simulation_backend": (
                str(self.master_pattern.metadata.get("backend", "computed"))
                if self.master_pattern is not None
                else None
            ),
        }

    @cached_property
    def figure(self):
        """The single native figure mounted by Jupyter or SpyDE."""
        if self.master_pattern is None:
            return self.crystal.show(title=f"{self.metadata['name']} structure")

        from de_ebsd.viz.patterns import phase_overview_figure

        texture = self.master_pattern.to_equirectangular(height=self.texture_height)
        return phase_overview_figure(
            self.crystal,
            texture.detach().cpu().numpy(),
            name=str(self.metadata["name"]),
            n_grid=self.n_grid,
        )

    @property
    def figures(self) -> dict[str, object]:
        """Named native figures for notebook or application host mounting."""
        return {"phase_overview": self.figure}

    def _ipython_display_(self) -> None:
        from IPython.display import HTML, display

        display(HTML(_metadata_card(self.metadata)))
        for figure in self.figures.values():
            display(figure)

    def __repr__(self) -> str:
        metadata = self.metadata
        return (
            f"PhaseOverview(name={metadata['name']!r}, "
            f"space_group={metadata['space_group']}, "
            f"master_pattern={self.master_pattern is not None})"
        )


def _phase_name(crystal: Crystal) -> str:
    elements = "+".join(dict.fromkeys(site.element for site in crystal.sites))
    return elements or "phase"


def _metadata_card(metadata: dict[str, str | int | float | None]) -> str:
    name = escape(str(metadata["name"]))
    symbol = escape(str(metadata["space_group_symbol"]))
    backend = metadata["simulation_backend"] or "not computed"
    cell = " × ".join(f"{float(metadata[key]):.4f}" for key in ("a_nm", "b_nm", "c_nm"))
    return f"""
    <div style="font:12px var(--jp-ui-font-family,system-ui,sans-serif);
      color:var(--jp-ui-font-color1,#d8dee9);background:var(--jp-layout-color1,#20242d);
      border:1px solid var(--jp-border-color2,#3b4252);border-radius:6px;
      padding:8px 10px;max-width:940px;display:flex;gap:18px;align-items:baseline;flex-wrap:wrap">
      <strong style="font-size:13px">{name}</strong>
      <span>{symbol} ({int(metadata["space_group"])})</span>
      <span>cell {cell} nm</span>
      <span>{int(metadata["atom_count"])} atoms</span>
      <span>ρ {float(metadata["density_g_cm3"]):.2f} g/cm³</span>
      <span>{escape(str(backend))}</span>
    </div>
    """
