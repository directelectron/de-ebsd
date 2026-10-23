"""Master-pattern backends: how a set of incident directions becomes intensity.

:class:`BlochWaveBackend` is the dynamical solver (Bethe-perturbed dynamical
matrix, eigen-decomposition, depth integration) in float64, one direction at a
time; :class:`~de_ebsd.diffraction.prism.BlochPrismBackend` is its batched GPU
equivalent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from de_ebsd.crystal.scattering import atomic_number
from de_ebsd.crystal.structure_factors import StructureFactors
from de_ebsd.diffraction.bethe import BetheParameters
from de_ebsd.diffraction.reflections import (
    STRONG,
    WEAK,
    ReflectionList,
    candidate_reflections,
    excitation_errors,
    master_reflection_list,
    partition_strong_weak,
)

if TYPE_CHECKING:
    from anyplotlib import Figure

    from de_ebsd.sim.montecarlo.depth import DepthDistribution

__all__ = [
    "DynamicalMatrix",
    "MasterPatternBackend",
    "BlochWaveBackend",
    "build_dynamical_matrix",
    "classify_reflections",
]


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

    status = partition_strong_weak(hkl, sg_c, forbidden, sf, bethe)
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
        """Heat map of ``log10 |A_gh|``."""
        from de_ebsd.viz.figures import image_figure

        with np.errstate(divide="ignore"):
            image = np.log10(np.maximum(np.abs(self.matrix), 1e-12))
        return image_figure(
            image,
            title=f"log10 |A| for k = {np.round(self.k, 4)} ({self.n_beams} strong beams)",
            cmap="viridis",
        )

    def __repr__(self) -> str:
        return (
            f"DynamicalMatrix({self.n_beams} strong beams, "
            f"{self.reflections.n_weak} weak folded in, k={np.round(self.k, 4)})"
        )


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
    #: whether this backend consumes a Monte Carlo exit-depth profile; the
    #: simulator builds a native one (EMsoft-free) when the user supplies none.
    uses_depth_profile: bool = False

    @abstractmethod
    def prepare(self, inputs: SimulationInputs) -> None:
        """Build and cache per-simulation tensors on the target device."""

    @abstractmethod
    def intensities(self, directions: torch.Tensor) -> torch.Tensor:
        """Intensities (K,) for unit directions (K, 3). Called after prepare."""


def calc_lgh(
    a: np.ndarray,
    kn: float,
    thick: float,
    depthstep: float,
    lambda_e: np.ndarray,
    gzero: int,
) -> np.ndarray:
    """The depth-integrated intensity matrix L_gh (EMsoft ``CalcLgh``).

    Eigendecompose the dynamical matrix ``a`` (complex, nm^-2), scale
    eigenvalues by ``1/(2 kn)``, and integrate the Bloch-wave interference
    against the Monte Carlo exit-depth profile ``lambda_e`` (already corrected
    for normal absorption), returning ``L_gh = conj(C) I C^T``.
    """
    w, c = np.linalg.eig(a)  # right eigenvectors are the columns of c (LAPACK zgeev)
    c_inv = np.linalg.inv(c)
    w = w / (2.0 * kn)
    alpha = c_inv[:, gzero]  # incident-beam excitation amplitudes CGinv(:, gzero)

    tpi = 2.0 * np.pi * depthstep
    re, im = w.real, w.imag
    q_re = tpi * (im[:, None] + im[None, :])  # real part of q_jk
    q_im = tpi * (re[:, None] - re[None, :])  # imaginary part
    qold = q_re + 1j * q_im
    qold = np.where(q_re < 0.0, -qold, qold)  # fold the real part positive

    # sum_z lambda_E[z] exp(-qold z) is a Horner polynomial in r = exp(-qold):
    # sum_{z=1..Z} lambda_z r^z, evaluated with Z complex multiply-adds instead
    # of materializing and exponentiating the full [S, S, Z] tensor -- identical
    # to machine precision, ~20x fewer exponentials (the dominant cost here).
    r = np.exp(-qold)
    q = np.zeros_like(r)
    for z in range(len(lambda_e), 0, -1):
        q = q * r + lambda_e[z - 1]
    q = q * r
    ijk = np.conj(alpha)[:, None] * q * alpha[None, :] * (depthstep / thick)
    return np.conj(c) @ ijk @ c.T


class BlochWaveBackend(MasterPatternBackend):
    """The physical Bloch-wave backend (EMsoft ``CalcLgh`` + ``Sgh``).

    Per beam direction: classify reflections (Bethe), build the dynamical
    matrix, eigendecompose and depth-integrate (:func:`calc_lgh`), weight with
    the structure-factor-like ``Sgh`` matrix, and reduce to one backscatter
    intensity ``I(k) = Re sum_gh L_gh S_gh / n_atoms``.

    This is the float64 reference path (one direction at a time), the M2
    correctness milestone; the batched GPU path (bucket by strong-beam count +
    batched eig) is the follow-up optimization behind the same interface.
    """

    name = "bloch"
    uses_depth_profile = True

    def prepare(self, inputs: SimulationInputs) -> None:
        from de_ebsd.sim.montecarlo.depth import DepthDistribution

        self.sf = inputs.sf
        self.bethe = inputs.bethe
        self.device = inputs.device
        self.out_dtype = inputs.dtype
        crystal = inputs.sf.crystal

        # depth integration (EMsoft lambdaE / thick / izzmax, one energy bin)
        energy = inputs.sf.energy_kev
        mc = inputs.monte_carlo or DepthDistribution.single_energy(energy)
        izzmax = max(int(mc.integration_depths()[mc.energy_index(energy)]), 1)
        self.depthstep = float(mc.depth_step)
        self.thick = izzmax * self.depthstep
        weights = mc.depth_weights(energy, inputs.sf.normal_absorption_length)
        self.lambda_e = np.ascontiguousarray(weights[:izzmax], dtype=np.float64)
        self.kn = 1.0 / inputs.sf.wavelength
        self.gzero = 0
        self.n_atoms = int(sum(p.shape[0] for p in crystal.positions))

        self._build_sgh_lut(inputs.sf, crystal)

    def _build_sgh_lut(self, sf: StructureFactors, crystal) -> None:
        self.hmax = sf.hmax
        m = 2 * self.hmax
        rng = np.arange(-m, m + 1)
        hh, kk, ll = np.meshgrid(rng, rng, rng, indexing="ij")
        box = np.stack([hh, kk, ll], axis=-1).reshape(-1, 3)
        g2 = crystal.g_length(box) ** 2  # |h-g|^2 in the reciprocal frame

        pos, z2occ, bfac = [], [], []
        for site, positions in zip(crystal.sites, crystal.positions, strict=True):
            z = atomic_number(site.element)
            pos.append(positions)
            z2occ.append(np.full(positions.shape[0], z**2 * site.occupancy))
            bfac.append(np.full(positions.shape[0], site.debye_waller))
        pos = np.concatenate(pos)
        z2occ = np.concatenate(z2occ)
        bfac = np.concatenate(bfac)

        phase = np.exp(2j * np.pi * (box @ pos.T))
        dwf = np.exp(-0.25 * g2[:, None] * bfac[None, :])
        sgh = (phase * dwf * z2occ[None, :]).sum(axis=1)
        n = 4 * self.hmax + 1
        self.sgh_lut = sgh.reshape(n, n, n).astype(np.complex128)

    def _sgh_matrix(self, hkl_strong: np.ndarray) -> np.ndarray:
        diff = hkl_strong[None, :, :] - hkl_strong[:, None, :]  # [g, h] = h - g
        idx = diff + 2 * self.hmax
        return self.sgh_lut[idx[..., 0], idx[..., 1], idx[..., 2]]

    def _intensity(self, k_hat: np.ndarray) -> float:
        refl = classify_reflections(self.sf, k_hat, self.bethe)
        hkl_s = refl.hkl[refl.status == STRONG]
        a = build_dynamical_matrix(self.sf, refl)
        lgh = calc_lgh(a, self.kn, self.thick, self.depthstep, self.lambda_e, self.gzero)
        sgh = self._sgh_matrix(hkl_s)
        return float(np.real(np.sum(lgh * sgh)) / self.n_atoms)

    def intensities(self, directions: torch.Tensor) -> torch.Tensor:
        dirs = directions.detach().cpu().numpy().astype(np.float64)
        out = np.empty(dirs.shape[0], dtype=np.float64)
        for i in range(dirs.shape[0]):
            out[i] = self._intensity(dirs[i])
        return torch.as_tensor(out, device=self.device, dtype=self.out_dtype)


def resolve_backend(backend: str | MasterPatternBackend) -> MasterPatternBackend:
    """Map a backend name to an instance; pass instances through."""
    if isinstance(backend, MasterPatternBackend):
        return backend
    from de_ebsd.diffraction.prism import BlochPrismBackend  # deferred: prism imports this module

    named = {
        "bloch": BlochWaveBackend,
        "bloch_prism": BlochPrismBackend,
    }
    try:
        return named[backend]()
    except KeyError:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {sorted(named)} "
            "or a MasterPatternBackend instance"
        ) from None
