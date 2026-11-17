"""An energy-resolved band master and patterns rendered through a known appearance,
for the appearance fit's tests: fast on a CPU, no dynamical simulation."""

import numpy as np
import torch
from orix.quaternion import Rotation

from de_ebsd import EBSDDetector, KinematicMaster, MasterPattern
from de_ebsd.crystal import Crystal
from de_ebsd.index.appearance import DetectorAppearance
from de_ebsd.index.appearance import _Model as Model
from de_ebsd.sim.kinematic_bands import reflectors_from_phase

ENERGIES_KEV = np.arange(10.0, 21.0, 2.0)


def nickel():
    return Crystal.from_parameters(a=0.352, space_group=225, atoms=[("Ni", (0.0, 0.0, 0.0))])


def resolved_band_master(npx=48, energies=ENERGIES_KEV) -> MasterPattern:
    """One band master per energy: widths from each energy's Bragg angle (widened so a
    small detector resolves them) and each band's strength changing with energy, so
    the energies differ in more than a blur. Weighted by a rising Monte Carlo-like
    share."""
    crystal = nickel()
    shares = np.linspace(0.3, 1.0, len(energies)) ** 2
    shares = shares / shares.sum() * 0.4
    layers = []
    for energy, share in zip(energies, shares, strict=True):
        # every band down to d = 0.07 nm: whole families, so the master keeps the
        # crystal's symmetry, which the indexer's fundamental zone relies on
        reflectors = reflectors_from_phase(crystal, energy=energy * 1e3, dmin=0.07, max_bands=None)
        # each family brightens or fades its own way towards low energy, as dynamical
        # bands do, so the energies differ in more than a blur
        families = [tuple(sorted(np.abs(h).astype(int))) for h in reflectors.hkl]
        rng = np.random.default_rng(1)
        own = {family: rng.normal(0.0, 1.5) for family in sorted(set(families))}
        change = np.array([own[family] for family in families])
        reflectors.widths = reflectors.widths * 3.0
        reflectors.weights = reflectors.weights * np.exp(change * (20.0 / energy - 1.0))
        layer = KinematicMaster(reflectors, dtype=torch.float32).to_master_pattern(npx).data[0]
        layers.append(layer * share)
    return MasterPattern(torch.stack(layers), energies, {"crystal": crystal}, energy_weights=shares)


def detector(shape=(40, 40)) -> EBSDDetector:
    return EBSDDetector(shape=shape, pattern_center=(0.5, 0.45, 0.6), sample_tilt=70.0)


def render(resolved, appearance: DetectorAppearance, rotations, det, counts=None, seed=0):
    """Patterns ``(N, rows, cols)`` of *rotations* (``(N, 3, 3)``) seen through
    *appearance*, with Poisson noise at *counts* per pixel where given."""
    model = Model(
        resolved,
        det,
        resolved.device,
        energy_order=len(appearance.energy_curve),
        halo=True,
        background_order=appearance.background_order,
    )
    texture = appearance.master(resolved).data[0][None]
    directions = model.directions(
        {
            "center": torch.as_tensor(appearance.pattern_center_shift, dtype=torch.float32),
            "distortion": torch.tensor(float(appearance.distortion)),
        }
    )
    from de_ebsd.core.lambert import sample_texture

    g = torch.as_tensor(rotations, dtype=torch.float32)
    values = sample_texture(texture, torch.einsum("nij,pj->npi", g, directions))
    if appearance.background.size:
        values = values + torch.as_tensor(appearance.background, dtype=torch.float32) @ (
            model.background_terms
        )
    values = values.clamp_min(0)
    if counts is not None:
        generator = torch.Generator().manual_seed(seed)
        values = values / values.mean(-1, keepdim=True) * counts
        values = torch.poisson(values, generator=generator)
    return values.reshape(len(g), *det.shape)


def random_orientations(count, seed=0) -> np.ndarray:
    q = np.random.default_rng(seed).normal(size=(count, 4))
    return Rotation(q / np.linalg.norm(q, axis=1, keepdims=True)).to_matrix()
