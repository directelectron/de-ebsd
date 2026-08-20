"""Dynamical matrix and master-pattern backends.

The dynamical (Bloch-wave) matrix follows EMsoft ``GetDynMat`` in its Bloch
form: for strong reflections g, h

    A_gh = U_{g-h} - (lambda/2) sum_w U_{g-w} U_{w-h} / s_w      (g != h)
    A_gg = 2 s_g / lambda - (lambda/2) sum_w |U_{g-w}|^2 / s_w + i U'_0

with the weak beams w folded in perturbatively (Bethe potentials). Units are
nm^-2; eigenvalues are later scaled by 1 / (2 k_n).

Backends implement the intensity calculation for a batch of incident beam
directions behind one small interface, which is the extension seam for

- :class:`BlochWaveBackend` -- the physical model (eigendecomposition +
  depth integration; scheduled, see DESIGN.md roadmap M2),
- :class:`KinematicalBackend` -- runs today; band geometry only,
- a PRISM-style scattering-matrix backend (roadmap M5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from torchebsd.bethe import BetheParameters
from torchebsd.reflections import (
    STRONG,
    WEAK,
    ReflectionList,
    candidate_reflections,
    excitation_errors,
    master_reflection_list,
    partition_strong_weak,
)
from torchebsd.structure_factors import StructureFactors

if TYPE_CHECKING:
    from anyplotlib import Figure

    from torchebsd.montecarlo import DepthDistribution

__all__ = [
    "DynamicalMatrix",
    "MasterPatternBackend",
    "KinematicalBackend",
    "BlochWaveBackend",
    "build_dynamical_matrix",
    "classify_reflections",
]


# --------------------------------------------------------------------------- #
# per-direction reflection classification and matrix construction
# --------------------------------------------------------------------------- #
def classify_reflections(
    sf: StructureFactors,
    k_hat: np.ndarray,
    bethe: BetheParameters,
) -> ReflectionList:
    """Candidate + strong/weak classification for one beam direction.

    ``k_hat`` is a unit vector in the Cartesian crystal frame.
    """
    refl = master_reflection_list(sf)
    k = torch.as_tensor(k_hat, dtype=torch.float64)
    g_cart = torch.as_tensor(sf.crystal.g_cartesian(refl.hkl), dtype=torch.float64)
    sg = excitation_errors(g_cart, k, sf.wavelength).numpy()

    keep = candidate_reflections(refl, sg, sf.wavelength, bethe)
    # prepend the incident beam g = 0
    hkl = np.concatenate([np.zeros((1, 3), dtype=np.int64), refl.hkl[keep]])
    sg_c = np.concatenate([[0.0], sg[keep]])
    u_mod = np.concatenate([[np.abs(sf.lookup(np.zeros(3, dtype=np.int64)))], refl.u_mod[keep]])
    forbidden = np.concatenate([[False], refl.forbidden[keep]])

    status = partition_strong_weak(hkl, sg_c, u_mod, forbidden, sf, bethe)
    g_len = sf.crystal.g_length(hkl)
    return ReflectionList(
        hkl=hkl,
        g=g_len,
        d=1.0 / np.maximum(g_len, 1e-12),
        u_mod=u_mod,
        forbidden=forbidden,
        k=np.asarray(k_hat, dtype=np.float64),
        sg=sg_c,
        status=status,
    )


def build_dynamical_matrix(sf: StructureFactors, refl: ReflectionList) -> np.ndarray:
    """The Bloch dynamical matrix A (complex, nm^-2) for one classified list.

    Port of EMsoft ``GetDynMat`` (Bloch branch, with Bethe corrections).
    """
    if refl.status is None or refl.sg is None:
        raise ValueError("reflection list must be classified for one beam direction")
    strong = refl.status == STRONG
    weak = refl.status == WEAK
    hkl_s = refl.hkl[strong]
    hkl_w = refl.hkl[weak]
    sg_s = refl.sg[strong]
    sg_w = refl.sg[weak]
    lam = sf.wavelength

    diff = hkl_s[:, None, :] - hkl_s[None, :, :]
    a = sf.lookup(diff).astype(np.complex128)  # off-diagonal U_{g-h}

    if len(hkl_w) > 0:
        u_gw = sf.lookup(hkl_s[:, None, :] - hkl_w[None, :, :])  # (Ns, Nw)
        u_wh = sf.lookup(hkl_w[:, None, :] - hkl_s[None, :, :])  # (Nw, Ns)
        a -= 0.5 * lam * (u_gw / sg_w[None, :]) @ u_wh
        weak_diag = 0.5 * lam * (np.abs(u_gw) ** 2 / sg_w[None, :]).sum(axis=1)
    else:
        weak_diag = np.zeros(len(hkl_s))

    np.fill_diagonal(a, 2.0 * sg_s / lam - weak_diag + 1j * sf.upz0)
    return a


@dataclass
class DynamicalMatrix:
    """The dynamical matrix for one beam direction (inspectable stage)."""

    matrix: np.ndarray
    hkl: np.ndarray
    k: np.ndarray
    reflections: ReflectionList

    @property
    def n_beams(self) -> int:
        return self.matrix.shape[0]

    def show(self) -> Figure:
        """Heat map of log10 |A_gh|."""
        from torchebsd import display

        with np.errstate(divide="ignore"):
            image = np.log10(np.maximum(np.abs(self.matrix), 1e-12))
        return display.image_figure(
            image,
            title=f"log10 |A| for k = {np.round(self.k, 4)} ({self.n_beams} strong beams)",
            cmap="viridis",
        )

    def __repr__(self) -> str:
        return (
            f"DynamicalMatrix({self.n_beams} strong beams, "
            f"{self.reflections.n_weak} weak folded in, k={np.round(self.k, 4)})"
        )


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
@dataclass
class SimulationInputs:
    """Everything a backend needs, prepared once by the simulator."""

    sf: StructureFactors
    bethe: BetheParameters
    monte_carlo: DepthDistribution | None
    device: torch.device
    dtype: torch.dtype


class MasterPatternBackend(ABC):
    """Interface every master-pattern intensity backend implements.

    A backend receives a batch of unit incident-beam directions (Cartesian
    crystal frame) and returns one backscatter intensity per direction. The
    simulator handles gridding, chunking, and assembly into the Lambert maps.
    """

    name: str = "abstract"

    @abstractmethod
    def prepare(self, inputs: SimulationInputs) -> None:
        """Build and cache per-simulation tensors on the target device."""

    @abstractmethod
    def intensities(self, directions: torch.Tensor) -> torch.Tensor:
        """Intensities (K,) for unit directions (K, 3). Called after prepare."""


class KinematicalBackend(MasterPatternBackend):
    """Kinematical band model: correct band geometry, approximate profiles.

    Each allowed reflection paints a band of angular half-width equal to its
    Bragg angle, weighted by |U_g|^2, centered on its Kikuchi great circle.
    This is a geometry-validation placeholder for the Bloch-wave backend: it
    puts every band in the right place with a plausible relative weight but
    does not reproduce dynamical band profiles or excess/deficiency effects.
    """

    name = "kinematical"

    def __init__(self, max_reflections: int | None = None) -> None:
        self.max_reflections = max_reflections
        self._g_hat: torch.Tensor | None = None
        self._sin_bragg: torch.Tensor | None = None
        self._weight: torch.Tensor | None = None

    def prepare(self, inputs: SimulationInputs) -> None:
        sf = inputs.sf
        refl = master_reflection_list(sf)
        keep = ~refl.forbidden
        hkl = refl.hkl[keep]
        u_mod = refl.u_mod[keep]
        g_len = refl.g[keep]
        if self.max_reflections is not None:
            hkl = hkl[: self.max_reflections]
            u_mod = u_mod[: self.max_reflections]
            g_len = g_len[: self.max_reflections]

        g_cart = sf.crystal.g_cartesian(hkl)
        g_hat = g_cart / np.linalg.norm(g_cart, axis=1, keepdims=True)
        sin_bragg = 0.5 * sf.wavelength * g_len

        dev, dt = inputs.device, inputs.dtype
        self._g_hat = torch.as_tensor(g_hat, device=dev, dtype=dt)
        self._sin_bragg = torch.as_tensor(sin_bragg, device=dev, dtype=dt)
        self._weight = torch.as_tensor(u_mod**2, device=dev, dtype=dt)
        self._weight = self._weight / self._weight.sum()

    def intensities(self, directions: torch.Tensor) -> torch.Tensor:
        if self._g_hat is None:
            raise RuntimeError("call prepare() first")
        # distance from the Kikuchi great circle, in units of the band half-width
        t = (directions @ self._g_hat.T) / self._sin_bragg  # (K, G)
        profile = torch.clamp(1.0 - t * t, min=0.0)
        return profile @ self._weight


class BlochWaveBackend(MasterPatternBackend):
    """The physical Bloch-wave backend (EMsoft algorithm). Scheduled.

    The full algorithm is specified in DESIGN.md (section "The dynamical
    algorithm", roadmap M2): per energy bin and beam direction, classify
    reflections, build the dynamical matrix (:func:`build_dynamical_matrix`,
    implemented and testable today), eigendecompose with
    ``torch.linalg.eig``, depth-integrate against the Monte Carlo profile
    (EMsoft ``CalcLgh``), weight with the S_gh matrix, and accumulate.
    """

    name = "bloch"

    def prepare(self, inputs: SimulationInputs) -> None:
        self._inputs = inputs

    def intensities(self, directions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "The Bloch-wave solve (eigendecomposition + CalcLgh depth integration) "
            "is scheduled work: see DESIGN.md, roadmap milestone M2. The dynamical "
            "matrix itself is available via MasterPatternSimulator.dynamical_matrix(k). "
            "Use backend='kinematical' for a geometry-only master pattern today."
        )


def resolve_backend(backend: str | MasterPatternBackend) -> MasterPatternBackend:
    """Map a backend name to an instance; pass instances through."""
    if isinstance(backend, MasterPatternBackend):
        return backend
    named = {"kinematical": KinematicalBackend, "bloch": BlochWaveBackend}
    try:
        return named[backend]()
    except KeyError:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {sorted(named)} "
            "or a MasterPatternBackend instance"
        ) from None
