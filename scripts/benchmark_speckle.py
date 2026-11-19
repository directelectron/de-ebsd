"""Speckle on a simulated scan: indexed with one candidate, with several, and with the
post-scan neighbor retry, against the orientations the patterns were rendered at.

A scan of Voronoi grains is rendered from a dynamical master pattern (one saved
with :meth:`~de_ebsd.sim.MasterPattern.save`) under a backscatter falloff with
counting noise, then indexed with :class:`~de_ebsd.index.live.LiveIndexer` as a
live scan is: the static background from the scan's frames, ``fit=True``. For
each setting it prints the share of points more than 5 degrees from the truth,
the speckle fraction (:func:`~de_ebsd.index.neighbors.speckle_fraction`), the
median error, how many patterns the candidates were rechecked for, and patterns
a second (indexing only, the GPU warm)::

    uv run python scripts/benchmark_speckle.py ni_master.h5 --space-group 225 \\
        --counts 12 --candidates 1 4 8 16

The master is simulated once, for example::

    import de_ebsd
    crystal = de_ebsd.Crystal.from_parameters(
        a=0.35236, space_group=225, atoms=[("Ni", (0, 0, 0))]
    )
    sim = de_ebsd.MasterPatternSimulator(
        crystal, de_ebsd.Microscope(energy=20e3, sample_tilt=70.0), npx=300, dmin=0.05
    )
    sim.compute(device="cuda").save("ni_master.h5")
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from de_ebsd import EBSDDetector, MasterPattern
from de_ebsd.core.rotation import quaternion_to_matrix
from de_ebsd.index.io import phase_from_crystal
from de_ebsd.index.live import LiveIndexer, fitted_refine_steps_deg, retry_speckle
from de_ebsd.index.live.indexer import RECHECK_MARGIN, RECHECK_SCORE
from de_ebsd.index.neighbors import _operators, _pair_angles, speckle_fraction


def point_group_of(space_group: int):
    from de_ebsd.crystal import Crystal

    # any crystal of the space group gives its point group
    crystal = Crystal.from_parameters(
        a=0.4, c=0.5, space_group=space_group, atoms=[("Fe", (0, 0, 0))]
    )
    return phase_from_crystal(crystal).point_group


def voronoi_scan(shape, grains, spread_deg, seed):
    """``(rows * cols, 4)`` quaternions of a scan of *grains* Voronoi grains, each
    point within *spread_deg* of its grain's orientation."""
    rng = np.random.default_rng(seed)
    rows, cols = shape
    centers = rng.uniform(0, 1, (grains, 2)) * (rows, cols)
    y, x = np.mgrid[:rows, :cols]
    grain = np.argmin(
        (y[..., None] - centers[:, 0]) ** 2 + (x[..., None] - centers[:, 1]) ** 2, -1
    ).ravel()
    q = rng.normal(size=(grains, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = q[grain]
    # a small turn about a random axis for each point
    axes = rng.normal(size=(len(q), 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    half = np.radians(rng.uniform(0, spread_deg, len(q))) / 2
    turn = np.concatenate([np.cos(half)[:, None], np.sin(half)[:, None] * axes], 1)
    w1, v1 = turn[:, :1], turn[:, 1:]
    w2, v2 = q[:, :1], q[:, 1:]
    product = np.concatenate(
        [w1 * w2 - (v1 * v2).sum(1, keepdims=True), w1 * v2 + w2 * v1 + np.cross(v1, v2)], 1
    )
    return product


def render(master, detector, g, counts, seed, device):
    """Camera frames of orientations *g* ``(N, 3, 3)``: the patterns under a
    backscatter falloff, *counts* electrons a pixel at its peak, Poisson noise."""
    rows, cols = detector.shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, rows, device=device),
        torch.linspace(-1, 1, cols, device=device),
        indexing="ij",
    )
    backscatter = torch.exp(-(x**2 + (y - 0.3) ** 2) / 0.8)
    generator = torch.Generator(device=device).manual_seed(seed)
    directions = detector.direction_cosines(device=device)
    texture = master.data[-1].to(device, torch.float32)
    from de_ebsd.core.lambert import sample_texture

    frames = torch.empty(len(g), rows, cols, device=device, dtype=torch.float32)
    for lo in range(0, len(g), 512):
        rotations = g[lo : lo + 512]
        v = torch.einsum("mij,pj->mpi", rotations, directions)
        pattern = sample_texture(texture[None], v).view(-1, rows, cols)
        low = pattern.flatten(1).amin(1).view(-1, 1, 1)
        high = pattern.flatten(1).amax(1).view(-1, 1, 1)
        pattern = (pattern - low) / (high - low)
        mean = counts * backscatter * (0.6 + 0.4 * pattern)
        frames[lo : lo + 512] = torch.poisson(mean, generator=generator)
    return frames


def errors_deg(found, truth, operators):
    return _pair_angles(found.astype(np.float64), truth.astype(np.float64), operators)


def run(indexer, frames, fit):
    torch.cuda.synchronize()
    start = time.perf_counter()
    phase, g, score, entry = indexer.index(frames, fit=fit)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return phase, g, score, entry, len(frames) / seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("master", help="a MasterPattern .h5 (MasterPattern.save)")
    parser.add_argument("--space-group", type=int, default=225)
    parser.add_argument("--scan", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--grains", type=int, default=60)
    parser.add_argument("--frame", type=int, default=128, help="frame side, pixels")
    parser.add_argument("--counts", type=float, default=12.0, help="electrons a pixel")
    parser.add_argument("--resolution", type=float, default=3.0, help="dictionary step, deg")
    parser.add_argument("--candidates", type=int, nargs="+", default=(1, 4, 8, 16))
    parser.add_argument("--no-fit", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search-blur", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    master = MasterPattern.load(args.master).to(device)
    point_group = point_group_of(args.space_group)
    operators = _operators(point_group)
    detector = EBSDDetector(
        shape=(args.frame, args.frame), pattern_center=(0.5, 0.45, 0.6), sample_tilt=70.0, tilt=0.0
    )
    shape = tuple(args.scan)
    q = voronoi_scan(shape, args.grains, 0.5, args.seed)
    truth = quaternion_to_matrix(torch.as_tensor(q, dtype=torch.float32, device=device))
    frames = render(master, detector, truth, args.counts, args.seed, device)
    truth = truth.cpu().numpy()
    fit = not args.no_fit
    extra = {} if args.search_blur is None else {"search_blur_deg": args.search_blur}
    indexer = LiveIndexer(
        [master],
        [point_group],
        detector,
        resolution=args.resolution,
        refine_steps_deg=fitted_refine_steps_deg(device) if fit else (2.0, 1.0, 0.5, 0.25, 0.12),
        device=device,
        **extra,
    )
    print(
        f"{shape[0]} x {shape[1]} scan, {args.grains} grains, {args.frame} px frames, "
        f"{args.counts:g} counts/px, {args.resolution:g} deg dictionary "
        f"({len(indexer.rotations)} entries), fit={fit}"
    )
    header = f"{'setting':<34}{'>5 deg':>9}{'speckle':>9}{'median':>9}{'rechecked':>11}{'pat/s':>9}"
    print(header)

    def report(name, g, rechecked, rate):
        error = errors_deg(g.reshape(-1, 3, 3), truth, operators)
        speckle = speckle_fraction(g.reshape(*shape, 3, 3), point_group)
        print(
            f"{name:<34}{100 * np.mean(error > 5):>8.2f}%{100 * speckle:>8.2f}%"
            f"{np.median(error):>8.3f} {rechecked:>11}{rate:>9.0f}"
        )

    settings = []
    for count in args.candidates:
        settings.append((count, None))
        if count > 1:
            settings.append((count, math.inf))
    for count, gate in settings:
        indexer.set_candidates(count)
        if gate is None:
            indexer.recheck_margin, indexer.recheck_score = RECHECK_MARGIN, RECHECK_SCORE
        else:
            indexer.recheck_margin = indexer.recheck_score = gate
        indexer.restart_static(frames[:: max(1, len(frames) // 2048)])
        run(indexer, frames[:2048], fit)  # warm
        indexer.restart_static(frames[:: max(1, len(frames) // 2048)])
        indexer.rechecked = 0
        phase, g, score, entry, rate = run(indexer, frames, fit)
        g = g.cpu().numpy()
        name = f"K={count}" + ("" if count == 1 else (" gated" if gate is None else " all"))
        report(name, g, indexer.rechecked, rate)
        if gate is None:
            start = time.perf_counter()
            retried = retry_speckle(indexer, frames, phase, g, score, entry, shape=shape, fit=fit)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            report(
                f"  + post-scan retry ({retried.changed.sum()} changed)",
                retried.g,
                retried.tried,
                len(frames) / seconds,
            )


if __name__ == "__main__":
    main()
