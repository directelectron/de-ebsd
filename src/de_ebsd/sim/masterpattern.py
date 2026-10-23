"""The simulated master pattern and projection onto a detector."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch

from de_ebsd.core.constants import DIVISION_FLOOR
from de_ebsd.core.lambert import sample_texture
from de_ebsd.core.rotation import quaternion_to_matrix
from de_ebsd.sim.detector import EBSDDetector

if TYPE_CHECKING:
    from anyplotlib import Figure

__all__ = ["MasterPattern"]


@dataclass
class MasterPattern:
    """Backscatter intensity on the sphere in square-Lambert storage.

    Attributes
    ----------
    data
        Intensities, shape ``(n_energies, 2, n, n)``; hemisphere 0 is north.
    energies_kev
        Energy bin centers in keV, shape ``(n_energies,)``.
    metadata
        Free-form provenance (crystal, dmin, backend name, ...).
    """

    data: torch.Tensor
    energies_kev: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape_ok = (
            self.data.ndim == 4
            and self.data.shape[1] == 2
            and self.data.shape[2] == self.data.shape[3]
        )
        if not shape_ok:
            raise ValueError("data must have shape (n_energies, 2, n, n)")
        self.energies_kev = np.atleast_1d(np.asarray(self.energies_kev, dtype=np.float64))
        if len(self.energies_kev) != self.data.shape[0]:
            raise ValueError("energies_kev length must match data.shape[0]")

    @property
    def npx(self) -> int:
        """Half-width of the Lambert grid (n = 2 npx + 1)."""
        return (self.data.shape[-1] - 1) // 2

    @property
    def device(self) -> torch.device:
        return self.data.device

    def to(self, device: torch.device | str) -> MasterPattern:
        """Copy to another device."""
        return MasterPattern(self.data.to(device), self.energies_kev, dict(self.metadata))

    def energy_index(self, energy_kev: float | None) -> int:
        if energy_kev is None:
            return len(self.energies_kev) - 1  # highest energy bin
        return int(np.argmin(np.abs(self.energies_kev - energy_kev)))

    def sample(self, directions: torch.Tensor, energy_kev: float | None = None) -> torch.Tensor:
        """Interpolate the master pattern at unit vectors (..., 3).

        Bilinear interpolation on the square-Lambert grid via
        ``grid_sample(align_corners=True)``, identical to kikuchipy's
        interpolation. The hemisphere is selected by ``sign(z)``.
        """
        directions = directions.to(self.data.device, self.data.dtype)
        return sample_texture(self.data[self.energy_index(energy_kev)][None], directions)

    def project(
        self,
        detector: EBSDDetector,
        orientation,
        energy_kev: float | None = None,
        rescale: bool = True,
    ) -> torch.Tensor:
        """Project onto a detector for one or more crystal orientations.

        Parameters
        ----------
        detector
            The detector geometry (fixed pattern center).
        orientation
            Crystal orientation(s): an ``orix.quaternion.Rotation`` or a
            tensor/array of unit quaternions (N, 4) in (a, b, c, d) order.
        energy_kev
            Energy bin to project (default: highest).
        rescale
            Linearly rescale each pattern to [-1, 1].

        Returns
        -------
        torch.Tensor
            Patterns of shape (N, n_rows, n_cols).
        """
        q = _as_quaternions(orientation).to(self.data.device, self.data.dtype)
        dc = detector.direction_cosines(device=self.data.device, dtype=self.data.dtype)

        rot = quaternion_to_matrix(q)  # (N, 3, 3)
        v = torch.einsum("nij,pj->npi", rot, dc)  # sample frame -> crystal frame
        pattern = self.sample(v, energy_kev=energy_kev)  # (N, P)

        if rescale:
            pmin = pattern.amin(dim=1, keepdim=True)
            pmax = pattern.amax(dim=1, keepdim=True)
            pattern = 2.0 * (pattern - pmin) / (pmax - pmin).clamp_min(DIVISION_FLOOR) - 1.0
        return pattern.reshape(-1, detector.nrows, detector.ncols)

    def to_stereographic(self, n: int = 501, hemisphere: str = "north",
                         energy_kev: float | None = None) -> torch.Tensor:
        """Resample one hemisphere into a stereographic (equal-angle) image."""
        sign = 1.0 if hemisphere == "north" else -1.0
        coords = torch.linspace(-1.0, 1.0, n, device=self.device)
        ys, xs = torch.meshgrid(coords, coords, indexing="ij")
        r2 = xs * xs + ys * ys
        # inverse stereographic projection of the unit disk
        vz = sign * (1.0 - r2) / (1.0 + r2)
        vx = 2.0 * xs / (1.0 + r2)
        vy = 2.0 * ys / (1.0 + r2)
        v = torch.stack([vx, vy, vz], dim=-1)
        image = self.sample(v, energy_kev=energy_kev)
        return torch.where(r2 <= 1.0, image, torch.zeros_like(image))

    def to_equirectangular(
        self, height: int = 256, energy_kev: float | None = None
    ) -> torch.Tensor:
        """Resample the full sphere into an equirectangular (lat-long) image.

        Row 0 is the north pole; used as the texture for the sphere view.
        """
        width = 2 * height
        theta = torch.linspace(0.0, math.pi, height, device=self.device)  # polar
        phi = torch.linspace(-math.pi, math.pi, width, device=self.device)
        t, p = torch.meshgrid(theta, phi, indexing="ij")
        v = torch.stack(
            [torch.sin(t) * torch.cos(p), torch.sin(t) * torch.sin(p), torch.cos(t)], dim=-1
        )
        return self.sample(v, energy_kev=energy_kev)

    def show(
        self,
        projection: str = "lambert",
        energy_kev: float | None = None,
        texture_height: int = 256,
        **kwargs,
    ) -> Figure:
        """Display the master pattern.

        Parameters
        ----------
        projection
            ``"lambert"`` (both hemispheres side by side), ``"stereographic"``
            (both hemispheres, equal-angle), or ``"sphere"`` (interactive 3D
            sphere with the pattern as texture).
        energy_kev
            Energy bin to display (default: highest).
        texture_height
            Height of the equirectangular texture used by the interactive
            sphere. Lower values reduce notebook/SpyDE payload size while
            leaving the master-pattern data unchanged.
        """
        from de_ebsd.viz.patterns import hemispheres_figure, sphere_figure

        i_e = self.energy_index(energy_kev)
        label = f"{self.energies_kev[i_e]:g} keV"
        if projection == "lambert":
            north = self.data[i_e, 0].detach().cpu().numpy()
            south = self.data[i_e, 1].detach().cpu().numpy()
            return hemispheres_figure(north, south, title=f"Master pattern ({label})")
        if projection == "stereographic":
            north = self.to_stereographic(hemisphere="north", energy_kev=energy_kev)
            south = self.to_stereographic(hemisphere="south", energy_kev=energy_kev)
            return hemispheres_figure(
                north.detach().cpu().numpy(),
                south.detach().cpu().numpy(),
                title=f"Master pattern, stereographic ({label})",
            )
        if projection == "sphere":
            if texture_height < 8:
                raise ValueError("texture_height must be >= 8")
            texture = self.to_equirectangular(
                height=int(texture_height), energy_kev=energy_kev
            )
            return sphere_figure(
                texture.detach().cpu().numpy(), title=f"Master pattern ({label})", **kwargs
            )
        raise ValueError("projection must be 'lambert', 'stereographic', or 'sphere'")

    def save(self, path: str) -> None:
        """Save to HDF5 (data, energies, and string metadata)."""
        with h5py.File(path, "w") as f:
            f.create_dataset("master_pattern", data=self.data.detach().cpu().numpy())
            f.create_dataset("energies_kev", data=self.energies_kev)
            meta = f.create_group("metadata")
            for key, value in self.metadata.items():
                meta.attrs[key] = str(value)

    @classmethod
    def load(cls, path: str, device: torch.device | str | None = None) -> MasterPattern:
        """Load a master pattern saved with :meth:`save`."""
        with h5py.File(path, "r") as f:
            data = torch.as_tensor(np.asarray(f["master_pattern"]), device=device)
            energies = np.asarray(f["energies_kev"])
            metadata = dict(f["metadata"].attrs) if "metadata" in f else {}
        return cls(data=data, energies_kev=energies, metadata=metadata)

    def __repr__(self) -> str:
        e = self.energies_kev
        return (
            f"MasterPattern(npx={self.npx}, {len(e)} energy bin(s) "
            f"[{e.min():g}..{e.max():g} keV], device={self.data.device.type}, "
            f"backend={self.metadata.get('backend', '?')})"
        )


def _as_quaternions(orientation) -> torch.Tensor:
    data = getattr(orientation, "data", orientation)  # orix Rotation -> .data
    q = torch.as_tensor(np.asarray(data) if not isinstance(data, torch.Tensor) else data)
    q = q.reshape(-1, 4)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(DIVISION_FLOOR)
